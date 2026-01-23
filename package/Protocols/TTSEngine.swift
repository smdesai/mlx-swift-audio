// Copyright © Anthony DePasquale

import Foundation

/// Core protocol that all TTS engines must conform to.
///
/// Voice selection and generation are engine-specific since each engine
/// has different voice types (enums, speaker profiles with reference audio, etc.)
///
/// TODO: Evaluate configuration approach when adding more engines.
/// Currently: Kokoro passes `speed` to methods, other engines use mutable properties
/// (temperature, topP, qualityLevel, etc.). Consider unifying if a clear pattern emerges.
@MainActor
public protocol TTSEngine: Observable {
  /// The provider type for this engine
  var provider: TTSProvider { get }

  /// The streaming granularities this engine supports.
  ///
  /// Most engines support at least `.sentence`. Some engines support multiple
  /// granularities - use `defaultStreamingGranularity` to get the recommended mode,
  /// or choose based on your latency requirements.
  ///
  /// For engines that support multiple granularities (like CosyVoice3), you can
  /// specify the desired granularity via a parameter:
  ///
  /// ```swift
  /// // Use the engine's default granularity
  /// for try await chunk in engine.generateStreaming(text) { ... }
  ///
  /// // Request a specific granularity (if supported)
  /// for try await chunk in engine.generateStreaming(text, granularity: .token) { ... }
  /// ```
  var supportedStreamingGranularities: Set<StreamingGranularity> { get }

  /// The recommended default streaming granularity for this engine.
  ///
  /// This is typically the most reliable or highest-quality streaming mode.
  /// For lower latency, check if `.token` or `.frame` is in `supportedStreamingGranularities`.
  var defaultStreamingGranularity: StreamingGranularity { get }

  // MARK: - State Properties

  /// Whether the model is loaded and ready for generation
  var isLoaded: Bool { get }

  /// Whether audio generation is currently in progress
  var isGenerating: Bool { get }

  /// Whether audio playback is currently in progress
  var isPlaying: Bool { get }

  /// Current playback position in seconds (for word highlighting)
  var playbackPosition: TimeInterval { get }

  /// URL of the last generated audio file (for sharing/export)
  var lastGeneratedAudioURL: URL? { get }

  /// Time taken for the last generation (seconds)
  var generationTime: TimeInterval { get }

  // MARK: - Lifecycle Methods

  /// Load the model with optional progress reporting
  /// - Parameter progressHandler: Optional callback for download/load progress
  func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws

  /// Stop any ongoing generation or playback
  func stop() async

  /// Unload model weights to free GPU memory.
  ///
  /// Preserves cached data (speaker profiles, reference audio, text processing)
  /// for faster reload. Use this when switching between engines to free memory
  /// while keeping expensive pre-computed data.
  func unload() async

  /// Full cleanup - releases everything including cached data.
  ///
  /// Use before deallocating the engine or when you need to free all resources.
  func cleanup() async throws

  // MARK: - Playback

  /// Play an audio result using the engine's internal player.
  ///
  /// Each engine maintains its own audio player, avoiding the overhead of
  /// creating a new player for each playback.
  func play(_ audio: AudioResult) async
}

// MARK: - Default Implementations

public extension TTSEngine {
  /// Load the model without progress reporting
  func load() async throws {
    try await load(progressHandler: nil)
  }

  /// Default playback position (engines without word highlighting support)
  var playbackPosition: TimeInterval {
    0
  }
}

// MARK: - Factory

/// Namespace for discovering and creating TTS engines with full type safety.
///
/// Each method returns a concrete engine type, enabling autocomplete for
/// engine-specific features like typed voices.
///
/// ```swift
/// let engine = TTS.orpheus()
/// try await engine.load()
/// try await engine.say("Hello", voice: .dan)  // typed voice enum
/// ```
///
/// For Kokoro, import MLXAudioKokoro and use `KokoroEngine()` directly.
@MainActor
public enum TTS {
  /// Orpheus: 8 voices, emotional expressions
  public static func orpheus() -> OrpheusEngine { OrpheusEngine() }

  /// Marvis: streaming, quality levels
  public static func marvis() -> MarvisEngine { MarvisEngine() }

  /// OuteTTS: custom speaker profiles
  public static func outetts() -> OuteTTSEngine { OuteTTSEngine() }

  /// Chatterbox: reference audio, emotion control
  public static func chatterbox() -> ChatterboxEngine { ChatterboxEngine() }

  /// CosyVoice2: voice matching with zero-shot and cross-lingual modes
  public static func cosyVoice2() -> CosyVoice2Engine { CosyVoice2Engine() }

  /// CosyVoice3: DiT-based voice matching with instruct mode and voice conversion
  public static func cosyVoice3() -> CosyVoice3Engine { CosyVoice3Engine() }
}

/// Describes how an engine streams audio output
///
/// Engines may support multiple granularities. Check `supportedStreamingGranularities`
/// to see what an engine supports, and `defaultStreamingGranularity` for the recommended mode.
public enum StreamingGranularity: Sendable, Hashable, CaseIterable {
  /// Audio is streamed sentence-by-sentence. Each chunk contains a complete sentence.
  /// Higher latency to first audio (~1-3s), but natural break points.
  /// Most engines support this mode.
  case sentence

  /// Audio is streamed frame-by-frame at regular time intervals.
  /// Lower latency, continuous output based on elapsed time.
  case frame

  /// Audio is streamed token-by-token as speech tokens are generated.
  /// Low latency (~0.5-1s to first audio), chunks based on token count.
  /// Useful for real-time applications where responsiveness matters.
  case token

  /// Human-readable description for UI display
  public var description: String {
    switch self {
      case .sentence:
        "Sentence-by-sentence"
      case .frame:
        "Frame-by-frame"
      case .token:
        "Token-by-token"
    }
  }

  /// Short description for compact UI
  public var shortDescription: String {
    switch self {
      case .sentence:
        "Per sentence"
      case .frame:
        "Continuous"
      case .token:
        "Low latency"
    }
  }

  /// Relative latency indicator for time-to-first-audio.
  /// Values are ordinal (1=fastest, 3=slowest), not milliseconds.
  public var relativeLatency: Int {
    switch self {
      case .token: 1
      case .frame: 2
      case .sentence: 3
    }
  }
}

/// A chunk of audio data for streaming playback
public struct AudioChunk: Sendable {
  /// Raw audio samples
  public let samples: [Float]

  /// Sample rate in Hz (e.g., 24000)
  public let sampleRate: Int

  /// Processing time for this chunk
  public let processingTime: TimeInterval

  public init(samples: [Float], sampleRate: Int, processingTime: TimeInterval) {
    self.samples = samples
    self.sampleRate = sampleRate
    self.processingTime = processingTime
  }
}

/// Result from TTS audio generation
public struct TTSGenerationResult: Sendable {
  /// Raw audio samples
  public let audio: [Float]

  /// Sample rate in Hz (e.g., 24000)
  public let sampleRate: Int

  /// Duration of the generated audio in seconds
  public let duration: TimeInterval

  /// Time taken to generate the audio in seconds
  public let processingTime: TimeInterval

  /// Real-time factor (processingTime / duration)
  /// Values < 1.0 mean faster than real-time
  public var realTimeFactor: Double {
    duration > 0 ? processingTime / duration : 0
  }

  public init(audio: [Float], sampleRate: Int, duration: TimeInterval, processingTime: TimeInterval) {
    self.audio = audio
    self.sampleRate = sampleRate
    self.duration = duration
    self.processingTime = processingTime
  }

  /// Convenience initializer that computes duration from sample count
  public init(audio: [Float], sampleRate: Int, processingTime: TimeInterval) {
    self.audio = audio
    self.sampleRate = sampleRate
    duration = Double(audio.count) / Double(sampleRate)
    self.processingTime = processingTime
  }
}
