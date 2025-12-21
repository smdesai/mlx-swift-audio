// Copyright © 2025 Resemble AI (original model implementation)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

import Foundation
import MLX
import Synchronization

/// Actor wrapper for ChatterboxTurboModel that provides thread-safe generation
///
/// ChatterboxTurbo is a faster variant of Chatterbox that uses:
/// - GPT-2 Medium backbone instead of LLaMA (faster token generation)
/// - Meanflow mode with 2 CFM steps instead of 10 (faster audio generation)
/// - GPT-2 tokenizer instead of custom EnTokenizer
///
/// Key differences from regular ChatterboxTTS:
/// - No CFG (classifier-free guidance) support
/// - No emotion exaggeration support
/// - Uses topK instead of minP for sampling
/// - Faster overall generation (roughly 2-3x speedup)
actor ChatterboxTurboTTS {
  // MARK: - Properties

  // Model is nonisolated(unsafe) because it contains non-Sendable types (MLXArray)
  // but is only accessed within the actor's methods
  private nonisolated(unsafe) let model: ChatterboxTurboModel

  // MARK: - Constants

  /// Maximum character count for a single chunk before splitting.
  ///
  /// Turbo's T3 model handles longer sequences better than regular Chatterbox,
  /// but we still split at 400 characters to maintain quality and avoid truncation.
  private static let maxChunkCharacters = 400

  // MARK: - Initialization

  private init(model: ChatterboxTurboModel) {
    self.model = model
  }

  /// Load ChatterboxTurboTTS from Hugging Face Hub
  ///
  /// - Parameters:
  ///   - quantization: Quantization level (fp16, 8bit, 4bit). Default is 4bit.
  ///   - progressHandler: Optional callback for download progress
  /// - Returns: Initialized ChatterboxTurboTTS instance
  static func load(
    quantization: ChatterboxTurboQuantization = .q4,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> ChatterboxTurboTTS {
    let model = try await ChatterboxTurboModel.load(
      quantization: quantization,
      progressHandler: progressHandler
    )
    return ChatterboxTurboTTS(model: model)
  }

  // MARK: - Conditionals

  /// Prepare conditioning from reference audio
  ///
  /// Returns the pre-computed conditionals that can be reused across multiple generation calls.
  /// This is the expensive operation that extracts voice characteristics from reference audio.
  /// Note: Turbo requires at least 5 seconds of reference audio for best results.
  ///
  /// - Parameters:
  ///   - refWav: Reference audio waveform (should be > 5 seconds)
  ///   - refSr: Sample rate of the reference audio
  /// - Returns: Pre-computed conditionals for generation
  func prepareConditionals(
    refWav: MLXArray,
    refSr: Int
  ) -> ChatterboxTurboConditionals {
    model.prepareConditionals(
      refWav: refWav,
      refSr: refSr
    )
  }

  // MARK: - Generation

  /// Generate audio from text using pre-computed conditionals
  ///
  /// This runs on the actor's background executor, not blocking the main thread.
  /// Long text is automatically split into chunks and processed separately to avoid truncation.
  ///
  /// Note: ChatterboxTurbo does not support CFG (classifier-free guidance), emotion
  /// exaggeration, or minP sampling. Use the standard Chatterbox for these features.
  ///
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - conditionals: Pre-computed reference audio conditionals
  ///   - temperature: Sampling temperature (default 0.8)
  ///   - topK: Top-k sampling parameter (default 1000)
  ///   - topP: Top-p sampling threshold (default 0.95)
  ///   - repetitionPenalty: Penalty for repeated tokens (default 1.2)
  ///   - maxNewTokens: Maximum tokens to generate per chunk (default 800)
  /// - Returns: Generated audio result
  func generate(
    text: String,
    conditionals: ChatterboxTurboConditionals,
    temperature: Float = 0.8,
    topK: Int = 1000,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.2,
    maxNewTokens: Int = 800
  ) -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    // Split text into manageable chunks
    let chunks = splitTextForGeneration(text)

    var allAudio: [Float] = []

    for chunk in chunks {
      // Check for cancellation between chunks
      if Task.isCancelled {
        break
      }

      let audioArray = model.generate(
        text: chunk,
        conds: conditionals,
        temperature: temperature,
        topK: topK,
        topP: topP,
        repetitionPenalty: repetitionPenalty,
        maxNewTokens: maxNewTokens
      )

      // Ensure computation is complete
      audioArray.eval()

      allAudio.append(contentsOf: audioArray.asArray(Float.self))

      // Clear GPU memory between chunks
      MLXMemory.clearCache()
    }

    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

    return TTSGenerationResult(
      audio: allAudio,
      sampleRate: sampleRate,
      processingTime: processingTime
    )
  }

  /// Generate audio using built-in voice conditionals
  ///
  /// This uses the pre-computed conditionals bundled with the model, if available.
  /// Useful for quick testing without providing reference audio.
  ///
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - temperature: Sampling temperature (default 0.8)
  ///   - topK: Top-k sampling parameter (default 1000)
  ///   - topP: Top-p sampling threshold (default 0.95)
  ///   - repetitionPenalty: Penalty for repeated tokens (default 1.2)
  ///   - maxNewTokens: Maximum tokens to generate per chunk (default 800)
  /// - Returns: Generated audio result
  func generate(
    text: String,
    temperature: Float = 0.8,
    topK: Int = 1000,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.2,
    maxNewTokens: Int = 800
  ) -> TTSGenerationResult {
    guard let conditionals = model.conds else {
      fatalError("No built-in conditionals available. Call prepareConditionals first or use generate(text:conditionals:)")
    }

    return generate(
      text: text,
      conditionals: conditionals,
      temperature: temperature,
      topK: topK,
      topP: topP,
      repetitionPenalty: repetitionPenalty,
      maxNewTokens: maxNewTokens
    )
  }

  /// Output sample rate
  var sampleRate: Int {
    ChatterboxTurboConstants.s3genSr
  }

  // MARK: - Streaming Generation

  /// Generate audio from text as a stream of chunks.
  ///
  /// Text is split into sentences, then oversized sentences are further split.
  /// Each chunk is yielded as it's generated for streaming playback.
  ///
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - conditionals: Pre-computed reference audio conditionals
  ///   - temperature: Sampling temperature (default 0.8)
  ///   - topK: Top-k sampling parameter (default 1000)
  ///   - topP: Top-p sampling threshold (default 0.95)
  ///   - repetitionPenalty: Penalty for repeated tokens (default 1.2)
  ///   - maxNewTokens: Maximum tokens to generate per chunk (default 800)
  /// - Returns: An async stream of audio sample chunks
  func generateStreaming(
    text: String,
    conditionals: ChatterboxTurboConditionals,
    temperature: Float = 0.8,
    topK: Int = 1000,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.2,
    maxNewTokens: Int = 800
  ) -> AsyncThrowingStream<[Float], Error> {
    let chunks = splitTextForGeneration(text)
    let chunkIndex = Atomic<Int>(0)

    return AsyncThrowingStream {
      let i = chunkIndex.wrappingAdd(1, ordering: .relaxed).oldValue
      guard i < chunks.count else { return nil }

      try Task.checkCancellation()

      let audioArray = self.model.generate(
        text: chunks[i],
        conds: conditionals,
        temperature: temperature,
        topK: topK,
        topP: topP,
        repetitionPenalty: repetitionPenalty,
        maxNewTokens: maxNewTokens
      )

      audioArray.eval()
      let samples = audioArray.asArray(Float.self)
      MLXMemory.clearCache()
      return samples
    }
  }

  /// Generate audio using built-in voice conditionals as a stream of chunks.
  func generateStreaming(
    text: String,
    temperature: Float = 0.8,
    topK: Int = 1000,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.2,
    maxNewTokens: Int = 800
  ) -> AsyncThrowingStream<[Float], Error> {
    guard let conditionals = model.conds else {
      return AsyncThrowingStream { throw ChatterboxTurboError.noConditionals }
    }

    return generateStreaming(
      text: text,
      conditionals: conditionals,
      temperature: temperature,
      topK: topK,
      topP: topP,
      repetitionPenalty: repetitionPenalty,
      maxNewTokens: maxNewTokens
    )
  }

  // MARK: - Private Methods

  /// Split text into chunks suitable for generation.
  ///
  /// First splits into sentences using SentenceTokenizer, then splits any
  /// oversized sentences using TextSplitter.
  private func splitTextForGeneration(_ text: String) -> [String] {
    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    var result: [String] = []

    for sentence in sentences {
      let chunks = TextSplitter.splitToMaxLength(sentence, maxCharacters: Self.maxChunkCharacters)
      result.append(contentsOf: chunks)
    }

    return result.isEmpty ? [text] : result
  }
}

// MARK: - Errors

enum ChatterboxTurboError: Error, LocalizedError {
  case noConditionals
  case tokenizerLoadFailed(String)

  var errorDescription: String? {
    switch self {
    case .noConditionals:
      return "No conditionals available. Call prepareConditionals first or provide reference audio."
    case .tokenizerLoadFailed(let message):
      return "Failed to load tokenizer: \(message)"
    }
  }
}
