import AVFoundation
import Foundation
import MLX
import Speech

/// OuteTTS engine - TTS with custom speaker profiles
///
/// Supports custom speaker profiles with reference audio.
@Observable
@MainActor
public final class OuteTTSEngine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .outetts
  public let streamingGranularity: StreamingGranularity = .sentence
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - OuteTTS-Specific Properties

  /// Temperature for sampling (higher = more variation)
  public var temperature: Float = 0.4

  /// Top-p (nucleus) sampling threshold
  public var topP: Float = 0.9

  // MARK: - Private Properties

  @ObservationIgnored private var outeTTS: OuteTTS?
  @ObservationIgnored private let audioPlayer = AudioSamplePlayer(sampleRate: TTSProvider.outetts.sampleRate)
  @ObservationIgnored private var generationTask: Task<Void, Never>?
  @ObservationIgnored private var streamingCancelled: Bool = false

  // MARK: - Initialization

  public init() {
    Log.tts.debug("OuteTTSEngine initialized")
  }

  deinit {
    generationTask?.cancel()
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("OuteTTSEngine already loaded")
      return
    }

    Log.model.info("Loading OuteTTS model...")

    do {
      outeTTS = try await OuteTTS.load(
        progressHandler: progressHandler ?? { _ in },
      )

      isLoaded = true
      Log.model.info("OuteTTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load OuteTTS model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    streamingCancelled = true
    generationTask?.cancel()
    generationTask = nil
    isGenerating = false

    await audioPlayer.stop()
    isPlaying = false

    Log.tts.debug("OuteTTSEngine stopped")
  }

  public func unload() async {
    await stop()

    outeTTS = nil
    isLoaded = false

    Log.tts.debug("OuteTTSEngine unloaded")
  }

  public func cleanup() async throws {
    await unload()
  }

  // MARK: - Playback

  public func play(_ audio: AudioResult) async {
    guard case let .samples(samples, _, _) = audio else {
      Log.audio.warning("Cannot play AudioResult.file - use AudioFilePlayer instead")
      return
    }

    isPlaying = true
    await audioPlayer.play(samples: samples)
    isPlaying = false
  }

  // MARK: - Generation

  /// Generate audio from text
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    guard let outeTTS else {
      throw TTSError.modelNotLoaded
    }

    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedText.isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    generationTask?.cancel()
    isGenerating = true
    generationTime = 0

    do {
      let result = try await outeTTS.generate(
        text: trimmedText,
        speaker: speaker,
        temperature: temperature,
        topP: topP,
      )

      generationTime = result.processingTime
      isGenerating = false

      Log.tts.timing("OuteTTS generation", duration: result.processingTime)
      Log.tts.rtf("OuteTTS", rtf: result.realTimeFactor)

      do {
        let fileURL = try AudioFileWriter.save(
          samples: result.audio,
          sampleRate: result.sampleRate,
          filename: TTSConstants.outputFilename,
        )
        lastGeneratedAudioURL = fileURL
      } catch {
        Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
      }

      return .samples(
        data: result.audio,
        sampleRate: result.sampleRate,
        processingTime: result.processingTime,
      )

    } catch {
      isGenerating = false
      Log.tts.error("OuteTTS generation failed: \(error.localizedDescription)")
      throw TTSError.generationFailed(underlying: error)
    }
  }

  /// Generate and immediately play audio
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  public func say(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) async throws {
    let audio = try await generate(text, speaker: speaker)
    await play(audio)
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks
  ///
  /// Text splitting is handled by OuteTTS.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    AsyncThrowingStream { continuation in
      let task = Task { @MainActor [weak self] in
        guard let self else {
          continuation.finish()
          return
        }

        // Auto-load if needed
        if !isLoaded {
          do {
            try await load()
          } catch {
            continuation.finish(throwing: error)
            return
          }
        }

        guard let outeTTS else {
          continuation.finish(throwing: TTSError.modelNotLoaded)
          return
        }

        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
          continuation.finish(throwing: TTSError.invalidArgument("Text cannot be empty"))
          return
        }

        let startTime = Date()
        let sampleRate = provider.sampleRate

        do {
          for try await samples in await outeTTS.generateStreaming(
            text: trimmedText,
            speaker: speaker,
            temperature: temperature,
            topP: topP,
          ) {
            guard !Task.isCancelled else { break }

            let chunk = AudioChunk(
              samples: samples,
              sampleRate: sampleRate,
              processingTime: Date().timeIntervalSince(startTime),
            )
            continuation.yield(chunk)
          }
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }

      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }

  /// Play audio with streaming (plays as chunks arrive)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  @discardableResult
  public func sayStreaming(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    // Stop any previous playback
    await audioPlayer.stop()
    streamingCancelled = false

    isPlaying = true
    isGenerating = true
    var allSamples: [Float] = []
    var totalProcessingTime: TimeInterval = 0

    do {
      for try await chunk in generateStreaming(text, speaker: speaker) {
        if streamingCancelled { break }
        allSamples.append(contentsOf: chunk.samples)
        totalProcessingTime = chunk.processingTime
        audioPlayer.enqueue(samples: chunk.samples, prebufferSeconds: 0)
      }

      // Streaming complete - audio continues playing from queue
      isPlaying = false
      isGenerating = false

      if !allSamples.isEmpty {
        do {
          let fileURL = try AudioFileWriter.save(
            samples: allSamples,
            sampleRate: provider.sampleRate,
            filename: TTSConstants.outputFilename,
          )
          lastGeneratedAudioURL = fileURL
        } catch {
          Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
        }
      }

      return .samples(
        data: allSamples,
        sampleRate: provider.sampleRate,
        processingTime: totalProcessingTime,
      )
    } catch {
      isPlaying = false
      isGenerating = false
      throw error
    }
  }

  // MARK: - Speaker Profile Creation

  /// Create a speaker profile from an audio file with automatic transcription
  /// - Parameter url: URL to the audio file (local or remote)
  /// - Returns: A speaker profile that can be used for voice cloning
  public func createSpeakerProfile(from url: URL) async throws -> OuteTTSSpeakerProfile {
    if !isLoaded {
      try await load()
    }

    guard let outeTTS else {
      throw TTSError.modelNotLoaded
    }

    // Load audio samples
    let (samples, sampleRate) = try await loadAudioSamples(from: url)
    let audioArray = MLXArray(samples)

    // Transcribe audio with word-level timestamps
    let (text, words) = try await transcribeAudio(url: url)

    guard !words.isEmpty else {
      throw TTSError.invalidArgument("Could not transcribe audio - no words detected")
    }

    // Resample to 24kHz if needed (OuteTTS native sample rate)
    let targetSampleRate = provider.sampleRate
    let resampledAudio: MLXArray = if sampleRate != targetSampleRate {
      AudioResampler.resample(audioArray, from: sampleRate, to: targetSampleRate)
    } else {
      audioArray
    }

    // Create speaker profile using the audio processor
    let profile = try await outeTTS.getSpeaker(
      referenceAudio: resampledAudio,
      referenceText: text,
      referenceWords: words,
    )

    guard let profile else {
      throw TTSError.invalidArgument("Failed to create speaker profile from audio")
    }

    return profile
  }

  // MARK: - Audio Loading

  private func loadAudioSamples(from url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    if url.isFileURL {
      try await loadAudioFromFile(url)
    } else {
      try await loadAudioFromRemoteURL(url)
    }
  }

  private func loadAudioFromRemoteURL(_ url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    Log.tts.debug("Downloading reference audio from URL: \(url)")

    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse,
          (200 ... 299).contains(httpResponse.statusCode)
    else {
      throw TTSError.invalidArgument("Failed to download reference audio from URL")
    }

    // Save to temporary file and load
    let tempURL = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
      .appendingPathExtension(url.pathExtension.isEmpty ? "mp3" : url.pathExtension)

    try data.write(to: tempURL)
    defer { try? FileManager.default.removeItem(at: tempURL) }

    return try await loadAudioFromFile(tempURL)
  }

  private func loadAudioFromFile(_ url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    Log.tts.debug("Loading reference audio from file: \(url.path)")

    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw TTSError.invalidArgument("Failed to create audio buffer")
    }

    try audioFile.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw TTSError.invalidArgument("Failed to read audio data")
    }

    // Convert to mono if stereo
    let samples: [Float]
    if format.channelCount == 1 {
      samples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength)))
    } else {
      // Mix stereo to mono
      let left = UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength))
      let right = UnsafeBufferPointer(start: floatData[1], count: Int(buffer.frameLength))
      samples = zip(left, right).map { ($0 + $1) / 2.0 }
    }

    return (samples, Int(format.sampleRate))
  }

  // MARK: - Speech Transcription

  // TODO: Replace Apple's SFSpeechRecognizer with Whisper or another on-device STT engine
  // when available in MLXAudio. Apple's API sends audio data to Apple servers and requires
  // user permission (NSSpeechRecognitionUsageDescription). An on-device solution would be
  // more private and wouldn't require additional permissions.

  private nonisolated func transcribeAudio(url: URL) async throws -> (text: String, words: [(word: String, start: Double, end: Double)]) {
    // Request speech recognition authorization
    let authStatus = await withCheckedContinuation { continuation in
      SFSpeechRecognizer.requestAuthorization { status in
        continuation.resume(returning: status)
      }
    }

    guard authStatus == .authorized else {
      throw TTSError.invalidArgument("Speech recognition not authorized. Please enable it in Settings.")
    }

    guard let recognizer = SFSpeechRecognizer() else {
      throw TTSError.invalidArgument("Speech recognition not available for this locale")
    }

    guard recognizer.isAvailable else {
      throw TTSError.invalidArgument("Speech recognition is not currently available")
    }

    let request = SFSpeechURLRecognitionRequest(url: url)
    request.shouldReportPartialResults = false

    return try await withCheckedThrowingContinuation { continuation in
      recognizer.recognitionTask(with: request) { result, error in
        if let error {
          continuation.resume(throwing: TTSError.invalidArgument("Transcription failed: \(error.localizedDescription)"))
          return
        }

        guard let result, result.isFinal else {
          return
        }

        let transcription = result.bestTranscription
        let text = transcription.formattedString

        // Extract word-level timestamps
        let words: [(word: String, start: Double, end: Double)] = transcription.segments.map { segment in
          (
            word: segment.substring,
            start: segment.timestamp,
            end: segment.timestamp + segment.duration,
          )
        }

        continuation.resume(returning: (text, words))
      }
    }
  }
}
