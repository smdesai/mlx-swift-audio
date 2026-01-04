// Copyright © Sachin Desai

import Foundation

// MARK: - Word Timing Result

extension ChatterboxTurboEngine {
  /// Result type for word-level timing generation
  public struct WordTimingResult: Sendable {
    /// All word timings across all chunks
    public let wordTimings: [HighlightedWord]

    /// All audio samples collected
    public let allSamples: [Float]

    /// Total audio duration in seconds
    public let duration: TimeInterval

    /// Processing time in seconds (total generation time)
    public let processingTime: TimeInterval

    /// Time to first audio chunk in seconds (latency before audio starts)
    public let timeToFirstAudio: TimeInterval

    /// The display text that matches the character ranges in wordTimings
    /// This may differ from the input text if the aligner skips certain characters
    public let displayText: String

    public init(
      wordTimings: [HighlightedWord],
      allSamples: [Float],
      duration: TimeInterval,
      processingTime: TimeInterval,
      timeToFirstAudio: TimeInterval,
      displayText: String
    ) {
      self.wordTimings = wordTimings
      self.allSamples = allSamples
      self.duration = duration
      self.processingTime = processingTime
      self.timeToFirstAudio = timeToFirstAudio
      self.displayText = displayText
    }
  }
}

// MARK: - Word Highlighting Methods

extension ChatterboxTurboEngine {
  /// Generate audio with word-level timings for highlighting.
  ///
  /// This method generates audio and returns word-level timing information
  /// that can be used to highlight words as they're spoken during playback.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  ///   - aligner: The forced aligner to use for word timing
  /// - Returns: WordTimingResult containing all word timings and metadata
  public func generateWithTimings(
    _ text: String,
    referenceAudio: ChatterboxTurboReferenceAudio? = nil,
    aligner: ForcedAligner? = nil
  ) async throws -> WordTimingResult {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    let startTime = Date()
    var timeToFirstAudio: TimeInterval = 0

    // Stream with timings
    let stream = generateStreamingWithTimings(
      trimmedText,
      referenceAudio: referenceAudio,
      aligner: aligner
    )

    var allTimings: [HighlightedWord] = []
    var allSamples: [Float] = []
    var timeOffset: TimeInterval = 0
    var isFirstChunk = true

    for try await chunk in stream {
      // Capture time to first audio on first chunk
      if isFirstChunk {
        timeToFirstAudio = Date().timeIntervalSince(startTime)
        isFirstChunk = false
      }

      // Adjust time offsets for consecutive chunks
      let adjustedTimings = chunk.wordTimings.map { timing in
        HighlightedWord(
          word: timing.word,
          start: timing.start + timeOffset,
          end: timing.end + timeOffset,
          charRange: timing.charRange
        )
      }

      allTimings.append(contentsOf: adjustedTimings)
      allSamples.append(contentsOf: chunk.samples)

      // Update time offset for next chunk
      timeOffset += Double(chunk.samples.count) / Double(chunk.sampleRate)
    }

    let processingTime = Date().timeIntervalSince(startTime)
    let duration = Double(allSamples.count) / Double(provider.sampleRate)

    return WordTimingResult(
      wordTimings: allTimings,
      allSamples: allSamples,
      duration: duration,
      processingTime: processingTime,
      timeToFirstAudio: timeToFirstAudio,
      displayText: trimmedText
    )
  }

  /// Generate audio as a stream with word-level timings.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  ///   - aligner: The forced aligner to use for word timing
  /// - Returns: An async stream of audio chunks with word timings
  public func generateStreamingWithTimings(
    _ text: String,
    referenceAudio: ChatterboxTurboReferenceAudio? = nil,
    aligner: ForcedAligner? = nil
  ) -> AsyncThrowingStream<AudioChunkWithTimings, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    // Capture current parameter values
    let currentTemperature = temperature
    let currentTopK = topK
    let currentTopP = topP
    let currentRepetitionPenalty = repetitionPenalty
    let currentMaxNewTokens = maxNewTokens
    let effectiveAligner = aligner ?? TokenBasedAligner()

    return AsyncThrowingStream { continuation in
      Task { @MainActor [weak self] in
        guard let self else {
          continuation.finish()
          return
        }

        do {
          // Auto-load if needed
          if !isLoaded {
            try await load()
          }

          // Prepare reference audio if needed
          let ref: ChatterboxTurboReferenceAudio
          if let referenceAudio {
            ref = referenceAudio
          } else {
            ref = try await prepareDefaultReferenceAudio()
          }

          // Get stream from model with timings
          let modelStream = await getChatterboxTurboTTS()?.generateStreamingWithTimings(
            text: trimmedText,
            conditionals: ref.conditionals,
            aligner: effectiveAligner,
            temperature: currentTemperature,
            topK: currentTopK,
            topP: currentTopP,
            repetitionPenalty: currentRepetitionPenalty,
            maxNewTokens: currentMaxNewTokens
          )

          guard let modelStream else {
            throw TTSError.modelNotLoaded
          }

          var timeOffset: TimeInterval = 0

          for try await chunk in modelStream {
            guard !Task.isCancelled else { break }

            // Adjust time offsets for streaming
            let adjustedTimings = chunk.wordTimings.map { timing in
              HighlightedWord(
                word: timing.word,
                start: timing.start + timeOffset,
                end: timing.end + timeOffset,
                charRange: timing.charRange
              )
            }

            let adjustedChunk = AudioChunkWithTimings(
              samples: chunk.samples,
              sampleRate: chunk.sampleRate,
              processingTime: chunk.processingTime,
              wordTimings: adjustedTimings
            )

            continuation.yield(adjustedChunk)
            timeOffset += Double(chunk.samples.count) / Double(chunk.sampleRate)
          }

          continuation.finish()
        } catch is CancellationError {
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
    }
  }

  /// Play audio with streaming and return word timings for highlighting.
  ///
  /// This method plays audio as it's generated and returns the word timings
  /// that can be used to highlight words during playback.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  ///   - aligner: The forced aligner to use for word timing
  /// - Returns: WordTimingResult containing all word timings and metadata
  @discardableResult
  public func sayWithTimings(
    _ text: String,
    referenceAudio: ChatterboxTurboReferenceAudio? = nil,
    aligner: ForcedAligner? = nil
  ) async throws -> WordTimingResult {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    let startTime = Date()
    var timeToFirstAudio: TimeInterval = 0

    // First, collect all the chunks with timings
    let stream = generateStreamingWithTimings(
      trimmedText,
      referenceAudio: referenceAudio,
      aligner: aligner
    )

    var allTimings: [HighlightedWord] = []
    var allSamples: [Float] = []
    var isFirstChunk = true

    for try await chunk in stream {
      // Capture time to first audio on first chunk
      if isFirstChunk {
        timeToFirstAudio = Date().timeIntervalSince(startTime)
        isFirstChunk = false
      }

      allTimings.append(contentsOf: chunk.wordTimings)
      allSamples.append(contentsOf: chunk.samples)
    }

    let processingTime = Date().timeIntervalSince(startTime)

    // Play the collected audio
    let audioResult = AudioResult.samples(
      data: allSamples,
      sampleRate: provider.sampleRate,
      processingTime: processingTime
    )

    await play(audioResult)

    let duration = Double(allSamples.count) / Double(provider.sampleRate)

    return WordTimingResult(
      wordTimings: allTimings,
      allSamples: allSamples,
      duration: duration,
      processingTime: processingTime,
      timeToFirstAudio: timeToFirstAudio,
      displayText: trimmedText
    )
  }

  /// Generate audio stream with attention-based alignment for accurate word timings.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  /// - Returns: An async stream of audio chunks with attention-aligned word timings
  public func generateStreamingWithAttentionAlignment(
    _ text: String,
    referenceAudio: ChatterboxTurboReferenceAudio? = nil
  ) async throws -> AsyncThrowingStream<AudioChunkWithTimings, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    // Capture current parameter values
    let currentTemperature = temperature
    let currentTopK = topK
    let currentTopP = topP
    let currentRepetitionPenalty = repetitionPenalty
    let currentMaxNewTokens = maxNewTokens

    // Auto-load if needed
    if !isLoaded {
      try await load()
    }

    // Prepare reference audio if needed
    let ref: ChatterboxTurboReferenceAudio
    if let referenceAudio {
      ref = referenceAudio
    } else {
      ref = try await prepareDefaultReferenceAudio()
    }

    // Get stream from model with attention alignment
    guard let tts = await getChatterboxTurboTTS() else {
      throw TTSError.modelNotLoaded
    }

    return await tts.generateStreamingWithAttentionAlignment(
      text: trimmedText,
      conditionals: ref.conditionals,
      temperature: currentTemperature,
      topK: currentTopK,
      topP: currentTopP,
      repetitionPenalty: currentRepetitionPenalty,
      maxNewTokens: currentMaxNewTokens
    )
  }

  /// Play audio with true streaming - audio plays as chunks are generated.
  ///
  /// Unlike `sayWithTimings()` which waits for all audio before playing,
  /// this method starts playback immediately when the first chunk is ready,
  /// reducing perceived latency.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  ///   - useAttentionAlignment: If true, uses attention-based alignment (~95% accuracy) instead of heuristic (~80%)
  ///   - onTimingsUpdate: Called each time new word timings are available (for incremental UI updates)
  ///   - onFirstAudio: Called when the first audio chunk is ready (for TTFA tracking)
  /// - Returns: WordTimingResult containing all word timings and metadata
  @discardableResult
  public func sayStreamingWithTimings(
    _ text: String,
    referenceAudio: ChatterboxTurboReferenceAudio? = nil,
    useAttentionAlignment: Bool = false,
    onTimingsUpdate: (([HighlightedWord]) -> Void)? = nil,
    onFirstAudio: ((TimeInterval) -> Void)? = nil
  ) async throws -> WordTimingResult {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    let startTime = Date()
    var timeToFirstAudio: TimeInterval = 0
    let playback = getPlaybackController()

    // Stop any previous playback before starting new generation
    await playback.stopAudio()

    // Get the streaming chunks with timings
    let stream: AsyncThrowingStream<AudioChunkWithTimings, Error>
    if useAttentionAlignment {
      stream = try await generateStreamingWithAttentionAlignment(trimmedText, referenceAudio: referenceAudio)
    } else {
      stream = generateStreamingWithTimings(trimmedText, referenceAudio: referenceAudio, aligner: nil)
    }

    var allTimings: [HighlightedWord] = []
    var allSamples: [Float] = []
    var isFirstChunk = true

    // Start playback state
    setIsPlaying(true)
    setIsGenerating(true)

    do {
      for try await chunk in stream {
        // Capture time to first audio on first chunk
        if isFirstChunk {
          timeToFirstAudio = Date().timeIntervalSince(startTime)
          onFirstAudio?(timeToFirstAudio)
          isFirstChunk = false
        }

        // Timings from generateStreamingWithTimings() are already adjusted with absolute times
        // Just accumulate them directly
        allTimings.append(contentsOf: chunk.wordTimings)
        onTimingsUpdate?(allTimings)

        // Collect samples and enqueue immediately for playback
        allSamples.append(contentsOf: chunk.samples)
        playback.enqueue(samples: chunk.samples, prebufferSeconds: 0)
      }

      setIsGenerating(false)

      // Wait for all audio to finish playing
      await playback.awaitAudioDrain()
      setIsPlaying(false)

    } catch {
      setIsGenerating(false)
      setIsPlaying(false)
      await playback.stopAudio()
      throw error
    }

    let processingTime = Date().timeIntervalSince(startTime)
    let duration = Double(allSamples.count) / Double(provider.sampleRate)

    return WordTimingResult(
      wordTimings: allTimings,
      allSamples: allSamples,
      duration: duration,
      processingTime: processingTime,
      timeToFirstAudio: timeToFirstAudio,
      displayText: trimmedText
    )
  }
}
