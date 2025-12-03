//
//  ChatterboxSession.swift
//  MLXAudio
//
//  Actor wrapper for ChatterboxTTS to enable background generation.
//  Similar to OrpheusTTS, OuteTTSSession, and MarvisTTSSession patterns.
//

import Foundation
import MLX

/// Actor wrapper for ChatterboxTTS that runs generation on a background thread
public actor ChatterboxSession {
  // MARK: - Properties

  // Model is nonisolated(unsafe) because it contains non-Sendable types (MLXArray)
  // but is only accessed within the actor's methods
  private nonisolated(unsafe) let model: ChatterboxTTS

  // MARK: - Initialization

  private init(model: ChatterboxTTS) {
    self.model = model
  }

  /// Load ChatterboxSession from Hugging Face Hub
  public static func load(
    repoId: String = ChatterboxTTS.defaultRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in },
  ) async throws -> ChatterboxSession {
    let model = try await ChatterboxTTS.load(
      repoId: repoId,
      progressHandler: progressHandler,
    )
    return ChatterboxSession(model: model)
  }

  // MARK: - Conditionals

  /// Prepare conditioning from reference audio
  ///
  /// Returns the pre-computed conditionals that can be reused across multiple generation calls.
  /// This is the expensive operation that extracts voice characteristics from reference audio.
  ///
  /// - Parameters:
  ///   - refWav: Reference audio waveform
  ///   - refSr: Sample rate of the reference audio
  ///   - exaggeration: Emotion exaggeration factor (0-1)
  /// - Returns: Pre-computed conditionals for generation
  public func prepareConditionals(
    refWav: MLXArray,
    refSr: Int,
    exaggeration: Float = 0.5,
  ) -> ChatterboxConditionals {
    model.prepareConditionals(
      refWav: refWav,
      refSr: refSr,
      exaggeration: exaggeration,
    )
  }

  // MARK: - Generation

  /// Generate audio from text using pre-computed conditionals
  ///
  /// This runs on the actor's background executor, not blocking the main thread.
  ///
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - conditionals: Pre-computed reference audio conditionals
  ///   - exaggeration: Emotion exaggeration factor
  ///   - cfgWeight: Classifier-free guidance weight
  ///   - temperature: Sampling temperature
  ///   - repetitionPenalty: Penalty for repeated tokens
  ///   - minP: Minimum probability threshold
  ///   - topP: Top-p sampling threshold
  ///   - maxNewTokens: Maximum tokens to generate
  /// - Returns: Generated audio as MLXArray
  public func generate(
    text: String,
    conditionals: ChatterboxConditionals,
    exaggeration: Float = 0.1,
    cfgWeight: Float = 0.5,
    temperature: Float = 0.8,
    repetitionPenalty: Float = 1.2,
    minP: Float = 0.05,
    topP: Float = 1.0,
    maxNewTokens: Int = 1000,
  ) -> MLXArray {
    model.generate(
      text: text,
      conds: conditionals,
      exaggeration: exaggeration,
      cfgWeight: cfgWeight,
      temperature: temperature,
      repetitionPenalty: repetitionPenalty,
      minP: minP,
      topP: topP,
      maxNewTokens: maxNewTokens,
    )
  }

  /// Generate audio and return as Float array
  ///
  /// This runs on the actor's background executor, not blocking the main thread.
  ///
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - conditionals: Pre-computed reference audio conditionals
  ///   - exaggeration: Emotion exaggeration factor
  ///   - cfgWeight: Classifier-free guidance weight
  ///   - temperature: Sampling temperature
  ///   - repetitionPenalty: Penalty for repeated tokens
  ///   - minP: Minimum probability threshold
  ///   - topP: Top-p sampling threshold
  ///   - maxNewTokens: Maximum tokens to generate
  /// - Returns: Generated audio samples as Float array
  public func generateAudio(
    text: String,
    conditionals: ChatterboxConditionals,
    exaggeration: Float = 0.1,
    cfgWeight: Float = 0.5,
    temperature: Float = 0.8,
    repetitionPenalty: Float = 1.2,
    minP: Float = 0.05,
    topP: Float = 1.0,
    maxNewTokens: Int = 1000,
  ) -> [Float] {
    let audioArray = generate(
      text: text,
      conditionals: conditionals,
      exaggeration: exaggeration,
      cfgWeight: cfgWeight,
      temperature: temperature,
      repetitionPenalty: repetitionPenalty,
      minP: minP,
      topP: topP,
      maxNewTokens: maxNewTokens,
    )

    // Ensure computation is complete
    audioArray.eval()

    // Convert to Float array
    return audioArray.asArray(Float.self)
  }

  /// Output sample rate
  public var sampleRate: Int {
    ChatterboxS3GenSr
  }
}
