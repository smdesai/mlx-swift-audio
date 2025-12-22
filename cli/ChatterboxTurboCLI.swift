//
// Chatterbox Turbo CLI - Text-to-Speech Command Line Tool
//
// IMPORTANT: This CLI must be built and run from Xcode to work correctly.
// The swift build command does not compile Metal shaders, which are required
// for MLX operations.
//
// To build and run:
// 1. Open Package.swift in Xcode
// 2. Select the "chatterbox-turbo" scheme
// 3. Build with Cmd+B (Release mode recommended: Product > Build For > Running)
// 4. Run with Cmd+R (set arguments in Product > Scheme > Edit Scheme > Arguments)
//
// Or from command line after building in Xcode:
// ./mlx-run chatterbox-turbo --text "Hello" --reference voice.wav

import ArgumentParser
import AVFoundation
import Foundation
import MLX
import MLXAudio

@main
struct ChatterboxTurboCLI: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "chatterbox-turbo",
    abstract: "Generate speech from text using Chatterbox Turbo TTS",
    discussion: """
      Chatterbox Turbo is a fast text-to-speech model that clones voices from reference audio.
      It uses a GPT-2 backbone and 2-step CFM for efficient generation.

      Example:
        chatterbox-turbo --text "Hello world" --reference voice.wav --output speech.wav
      """
  )

  @Option(name: [.short, .long], help: "Text to synthesize into speech")
  var text: String

  @Option(name: [.short, .long], help: "Path to reference audio file for voice cloning")
  var reference: String

  @Option(name: [.short, .long], help: "Output audio file path (default: audio.wav)")
  var output: String = "audio.wav"

  @Option(name: [.short, .long], help: "Model quantization: full, fp16, 8bit, 4bit (default: 4bit)")
  var model: String = "4bit"

  @Option(name: .long, help: "Sampling temperature (default: 0.8, use 0 for deterministic)")
  var temperature: Float = 0.8

  @Option(name: .long, help: "Top-k sampling (default: 1000)")
  var topK: Int = 1000

  @Option(name: .long, help: "Top-p sampling (default: 0.95)")
  var topP: Float = 0.95

  func run() async throws {
    // Parse quantization
    guard let quantization = ChatterboxTurboQuantization(rawValue: model) else {
      throw ValidationError("Invalid model quantization '\(model)'. Must be one of: full, fp16, 8bit, 4bit")
    }

    // Validate reference audio exists
    let referenceURL = URL(fileURLWithPath: reference)
    guard FileManager.default.fileExists(atPath: referenceURL.path) else {
      throw ValidationError("Reference audio file not found: \(reference)")
    }

    // Ensure output has .wav extension
    var outputPath = output
    if !outputPath.lowercased().hasSuffix(".wav") {
      outputPath += ".wav"
    }
    let outputURL = URL(fileURLWithPath: outputPath)

    // Create output directory if needed
    let outputDir = outputURL.deletingLastPathComponent()
    if !FileManager.default.fileExists(atPath: outputDir.path) {
      try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
    }

    print("Loading Chatterbox Turbo model (\(quantization.rawValue))...")

    // Load the model
    let tts = try await ChatterboxTurboTTS.load(
      quantization: quantization,
      progressHandler: { progress in
        let percent = Int(progress.fractionCompleted * 100)
        print("\rDownloading model: \(percent)%", terminator: "")
        fflush(stdout)
      }
    )
    print("\nModel loaded.")

    // Load reference audio
    print("Loading reference audio: \(reference)")
    let (samples, sampleRate) = try loadAudio(from: referenceURL)
    let duration = Double(samples.count) / Double(sampleRate)
    print(String(format: "Reference audio: %.1f seconds at %d Hz", duration, sampleRate))

    // Trim silence from reference audio
    let trimmedSamples = AudioTrimmer.trimSilence(
      samples,
      sampleRate: sampleRate,
      config: .chatterbox
    )
    let trimmedDuration = Double(trimmedSamples.count) / Double(sampleRate)
    print(String(format: "After silence trimming: %.1f seconds", trimmedDuration))

    // Prepare conditionals from reference audio
    print("Preparing voice conditionals...")
    let refWav = MLXArray(trimmedSamples)
    let conditionals = await tts.prepareConditionals(refWav: refWav, refSr: sampleRate)

    // Generate speech
    print("Generating speech for: \"\(text)\"")
    let startTime = CFAbsoluteTimeGetCurrent()

    let result = await tts.generate(
      text: text,
      conditionals: conditionals,
      temperature: temperature,
      topK: topK,
      topP: topP
    )

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    let audioDuration = Double(result.audio.count) / Double(result.sampleRate)
    let rtf = elapsed / audioDuration

    print(String(format: "Generated %.2f seconds of audio in %.2f seconds (RTF: %.2fx)", audioDuration, elapsed, rtf))

    // Save output
    print("Saving to: \(outputURL.path)")
    try saveAudio(samples: result.audio, sampleRate: result.sampleRate, to: outputURL)

    print("Done!")
  }

  // MARK: - Audio I/O

  private func loadAudio(from url: URL) throws -> (samples: [Float], sampleRate: Int) {
    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw ValidationError("Failed to create audio buffer")
    }

    try audioFile.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw ValidationError("Failed to read audio data")
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

  private func saveAudio(samples: [Float], sampleRate: Int, to url: URL) throws {
    guard let audioFormat = AVAudioFormat(
      standardFormatWithSampleRate: Double(sampleRate),
      channels: 1
    ) else {
      throw ValidationError("Failed to create audio format")
    }

    guard let buffer = AVAudioPCMBuffer(
      pcmFormat: audioFormat,
      frameCapacity: AVAudioFrameCount(samples.count)
    ) else {
      throw ValidationError("Failed to create audio buffer")
    }

    buffer.frameLength = AVAudioFrameCount(samples.count)

    guard let channelData = buffer.floatChannelData else {
      throw ValidationError("Failed to get channel data")
    }

    for i in 0 ..< samples.count {
      channelData[0][i] = samples[i]
    }

    let audioFile = try AVAudioFile(
      forWriting: url,
      settings: audioFormat.settings,
      commonFormat: .pcmFormatFloat32,
      interleaved: false
    )
    try audioFile.write(from: buffer)
  }
}
