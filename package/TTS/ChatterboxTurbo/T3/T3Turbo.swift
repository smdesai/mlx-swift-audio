// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Sachin Desai (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom

// MARK: - T3 Turbo Model

/// Token-To-Token (T3) Turbo TTS model using GPT-2 as backbone.
/// This is the faster variant that uses GPT-2 Medium instead of LLaMA.
class T3Turbo: Module {
  let hp: T3TurboConfig
  let cfg: GPT2Config
  let dim: Int

  @ModuleInfo(key: "tfmr") var tfmr: GPT2Model
  @ModuleInfo(key: "cond_enc") var condEnc: T3TurboCondEnc
  @ModuleInfo(key: "text_emb") var textEmb: Embedding
  @ModuleInfo(key: "speech_emb") var speechEmb: Embedding
  @ModuleInfo(key: "text_head") var textHead: Linear
  @ModuleInfo(key: "speech_head") var speechHead: Linear

  init(hp: T3TurboConfig? = nil) {
    let config = hp ?? T3TurboConfig.turbo()
    self.hp = config

    // Create GPT-2 config
    cfg = GPT2Config.gpt2Medium
    dim = cfg.hiddenSize

    // GPT-2 backbone
    _tfmr.wrappedValue = GPT2Model(config: cfg)

    // Conditioning encoder
    _condEnc.wrappedValue = T3TurboCondEnc(hp: config)

    // Text and speech embeddings
    _textEmb.wrappedValue = Embedding(embeddingCount: config.textTokensDictSize, dimensions: dim)
    _speechEmb.wrappedValue = Embedding(embeddingCount: config.speechTokensDictSize, dimensions: dim)

    // Output heads
    _textHead.wrappedValue = Linear(dim, config.textTokensDictSize, bias: false)
    _speechHead.wrappedValue = Linear(dim, config.speechTokensDictSize, bias: true)
  }

  /// Prepare conditioning embeddings.
  /// Token conditioning data needs to be embedded here.
  func prepareConditioning(_ t3Cond: inout T3TurboCond) -> MLXArray {
    // Embed conditioning tokens if not already done
    if t3Cond.condPromptSpeechTokens != nil && t3Cond.condPromptSpeechEmb == nil {
      t3Cond.condPromptSpeechEmb = speechEmb(t3Cond.condPromptSpeechTokens!)
    }

    return condEnc(t3Cond)
  }

  /// Prepare input embeddings for the transformer.
  func prepareInputEmbeds(
    t3Cond: inout T3TurboCond,
    textTokens: MLXArray,
    speechTokens: MLXArray
  ) -> (MLXArray, Int) {
    // Get conditioning embeddings
    let condEmb = prepareConditioning(&t3Cond) // (B, len_cond, dim)

    // Get text and speech embeddings
    let textEmbeddings = textEmb(textTokens) // (B, len_text, dim)
    let speechEmbeddings = speechEmb(speechTokens) // (B, len_speech, dim)

    let lenCond = condEmb.shape[1]

    // Expand condEmb if batch size differs
    var condEmbExpanded = condEmb
    if condEmb.shape[0] != textEmbeddings.shape[0] {
      condEmbExpanded = MLX.broadcast(
        condEmb,
        to: [textEmbeddings.shape[0], condEmb.shape[1], condEmb.shape[2]]
      )
    }

    // Concatenate: [cond, text, speech]
    let embeds = MLX.concatenated([condEmbExpanded, textEmbeddings, speechEmbeddings], axis: 1)

    return (embeds, lenCond)
  }

  /// Turbo inference: generate speech tokens from text tokens.
  ///
  /// - Parameters:
  ///   - t3Cond: Conditioning data
  ///   - textTokens: Input text tokens (B, T)
  ///   - temperature: Sampling temperature (default 0.8)
  ///   - topK: Top-k sampling parameter (default 1000)
  ///   - topP: Top-p (nucleus) sampling parameter (default 0.95)
  ///   - repetitionPenalty: Penalty for repeating tokens (default 1.2)
  ///   - maxGenLen: Maximum generation length (default 1000)
  /// - Returns: Generated speech tokens
  func inferenceTurbo(
    t3Cond: T3TurboCond,
    textTokens: MLXArray,
    temperature: Float = 0.8,
    topK: Int = 1000,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.2,
    maxGenLen: Int = 1000
  ) -> MLXArray {
    var cond = t3Cond

    // Ensure batch dimension
    var tokens = textTokens
    if tokens.ndim == 1 {
      tokens = tokens.expandedDimensions(axis: 0)
    }

    let B = tokens.shape[0]

    // Initial speech token (start token)
    let speechStartToken = MLXArray.ones([B, 1], type: Int32.self) * Int32(hp.startSpeechToken)

    // Prepare initial embeddings
    let (embeds, _) = prepareInputEmbeds(t3Cond: &cond, textTokens: tokens, speechTokens: speechStartToken)

    // Initial forward pass
    var (hiddenStates, cache) = tfmr(inputsEmbeds: embeds, cache: nil)

    // Get first speech prediction - take last timestep, keep dims: (B, 1, dim)
    let T = hiddenStates.shape[1]
    let lastTimestep = hiddenStates[0..., (T - 1)..<T]
    var speechLogits = speechHead(lastTimestep)

    // Sample first token - speechLogits is (B, 1, vocab), squeeze to (B, vocab)
    var generatedSpeechTokens: [MLXArray] = []
    var nextSpeechToken = sampleToken(
      logits: speechLogits.squeezed(axis: 1),
      temperature: temperature,
      topK: topK,
      topP: topP,
      generatedTokens: nil,
      repetitionPenalty: repetitionPenalty
    )
    generatedSpeechTokens.append(nextSpeechToken)
    var currentSpeechToken = nextSpeechToken

    // Generation loop
    for _ in 0 ..< maxGenLen {
      // Get embedding for current token
      let currentSpeechEmbed = speechEmb(currentSpeechToken)

      // Forward pass with cache
      (hiddenStates, cache) = tfmr(inputsEmbeds: currentSpeechEmbed, cache: cache)

      // Get logits - hiddenStates is (B, 1, C) in generation loop
      speechLogits = speechHead(hiddenStates)

      // Gather generated tokens for repetition penalty
      let allGenerated = MLX.concatenated(generatedSpeechTokens, axis: 1)

      // Sample next token - speechLogits is (B, 1, vocab), squeeze to (B, vocab)
      nextSpeechToken = sampleToken(
        logits: speechLogits.squeezed(axis: 1),
        temperature: temperature,
        topK: topK,
        topP: topP,
        generatedTokens: allGenerated,
        repetitionPenalty: repetitionPenalty
      )

      generatedSpeechTokens.append(nextSpeechToken)
      currentSpeechToken = nextSpeechToken

      // Check for EOS
      eval(nextSpeechToken)
      if nextSpeechToken[0, 0].item(Int32.self) == Int32(hp.stopSpeechToken) {
        break
      }
    }

    // Concatenate all tokens
    var allTokens = MLX.concatenated(generatedSpeechTokens, axis: 1)

    // Remove EOS token if present
    if allTokens.shape[1] > 0 {
      eval(allTokens)
      let lastToken = allTokens[0, -1].item(Int32.self)
      if lastToken == Int32(hp.stopSpeechToken) {
        allTokens = allTokens[0..., 0 ..< (allTokens.shape[1] - 1)]
      }
    }

    return allTokens
  }

  /// Streaming turbo inference: generate speech tokens from text tokens,
  /// yielding chunks of tokens as they're generated.
  ///
  /// - Parameters:
  ///   - t3Cond: Conditioning data
  ///   - textTokens: Input text tokens (B, T)
  ///   - temperature: Sampling temperature (default 0.8)
  ///   - topK: Top-k sampling parameter (default 1000)
  ///   - topP: Top-p (nucleus) sampling parameter (default 0.95)
  ///   - repetitionPenalty: Penalty for repeating tokens (default 1.2)
  ///   - chunkSize: Number of tokens to accumulate before yielding (default 40)
  ///   - maxGenLen: Maximum generation length (default 1000)
  /// - Returns: Sequence of (chunk, isFinal) tuples
  func inferenceTurboStream(
    t3Cond: T3TurboCond,
    textTokens: MLXArray,
    temperature: Float = 0.8,
    topK: Int = 1000,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.2,
    chunkSize: Int = 40,
    maxGenLen: Int = 1000
  ) -> [(MLXArray, Bool)] {
    var cond = t3Cond
    var results: [(MLXArray, Bool)] = []

    // Ensure batch dimension
    var tokens = textTokens
    if tokens.ndim == 1 {
      tokens = tokens.expandedDimensions(axis: 0)
    }

    let B = tokens.shape[0]

    // Initial speech token (start token)
    let speechStartToken = MLXArray.ones([B, 1], type: Int32.self) * Int32(hp.startSpeechToken)

    // Prepare initial embeddings
    let (embeds, _) = prepareInputEmbeds(t3Cond: &cond, textTokens: tokens, speechTokens: speechStartToken)

    // Initial forward pass
    var (hiddenStates, cache) = tfmr(inputsEmbeds: embeds, cache: nil)

    // Get first speech prediction
    let speechHidden = hiddenStates[0..., (-1)..., 0...]
    var speechLogits = speechHead(speechHidden)

    // Pre-allocate buffer for generated tokens
    let allGenerated = MLXArray.zeros([B, maxGenLen + 1], type: Int32.self)
    var numGenerated = 0

    // Sample first token
    var nextSpeechToken = sampleToken(
      logits: speechLogits[0..., -1, 0...],
      temperature: temperature,
      topK: topK,
      topP: topP,
      generatedTokens: nil,
      repetitionPenalty: repetitionPenalty
    )

    allGenerated[0..., numGenerated ..< (numGenerated + 1)] = nextSpeechToken
    numGenerated += 1

    var chunkTokens: [MLXArray] = [nextSpeechToken]
    var currentSpeechToken = nextSpeechToken

    // Generation loop
    for _ in 0 ..< maxGenLen {
      // Get embedding for current token
      let currentSpeechEmbed = speechEmb(currentSpeechToken)

      // Forward pass with cache
      (hiddenStates, cache) = tfmr(inputsEmbeds: currentSpeechEmbed, cache: cache)

      // Get logits
      speechLogits = speechHead(hiddenStates)

      // Sample next token
      nextSpeechToken = sampleToken(
        logits: speechLogits[0..., -1, 0...],
        temperature: temperature,
        topK: topK,
        topP: topP,
        generatedTokens: allGenerated[0..., 0 ..< numGenerated],
        repetitionPenalty: repetitionPenalty
      )

      // Update buffer
      allGenerated[0..., numGenerated ..< (numGenerated + 1)] = nextSpeechToken
      numGenerated += 1

      chunkTokens.append(nextSpeechToken)
      currentSpeechToken = nextSpeechToken

      // Check for EOS
      eval(nextSpeechToken)
      if nextSpeechToken[0, 0].item(Int32.self) == Int32(hp.stopSpeechToken) {
        // Yield remaining tokens (excluding EOS)
        if chunkTokens.count > 1 {
          let chunk = MLX.concatenated(Array(chunkTokens.dropLast()), axis: 1)
          eval(chunk)
          results.append((chunk, true))
        }
        return results
      }

      // Yield chunk if we've accumulated enough tokens
      if chunkTokens.count >= chunkSize {
        let chunk = MLX.concatenated(chunkTokens, axis: 1)
        eval(chunk)
        results.append((chunk, false))
        chunkTokens = []
      }
    }

    // Yield any remaining tokens
    if !chunkTokens.isEmpty {
      let chunk = MLX.concatenated(chunkTokens, axis: 1)
      eval(chunk)
      results.append((chunk, true))
    }

    return results
  }

  // MARK: - Sampling Helpers

  /// Sample a token from logits with various sampling strategies
  private func sampleToken(
    logits: MLXArray,
    temperature: Float,
    topK: Int,
    topP: Float,
    generatedTokens: MLXArray?,
    repetitionPenalty: Float
  ) -> MLXArray {
    var processedLogits = logits

    // Apply repetition penalty
    if let generated = generatedTokens, repetitionPenalty != 1.0 {
      processedLogits = applyRepetitionPenalty(
        logits: processedLogits,
        generatedTokens: generated,
        penalty: repetitionPenalty
      )
    }

    // Apply temperature
    if temperature > 0 && temperature != 1.0 {
      processedLogits = processedLogits / temperature
    }

    // Apply top-k and top-p (only if not greedy sampling)
    if temperature > 0 {
      if topK > 0 {
        processedLogits = topKFiltering(logits: processedLogits, topK: topK)
      }
      if topP < 1.0 {
        processedLogits = topPFiltering(logits: processedLogits, topP: topP)
      }
    }

    // Sample: greedy (argmax) for temperature=0, otherwise categorical
    let nextToken: MLXArray
    if temperature == 0 {
      // Greedy decoding - pick the highest probability token
      nextToken = MLX.argMax(processedLogits, axis: -1)
    } else {
      // Stochastic sampling
      nextToken = MLXRandom.categorical(processedLogits)
    }

    return nextToken.expandedDimensions(axis: -1)
  }

  /// Apply repetition penalty to logits
  private func applyRepetitionPenalty(
    logits: MLXArray,
    generatedTokens: MLXArray,
    penalty: Float
  ) -> MLXArray {
    if penalty == 1.0 {
      return logits
    }

    let vocabSize = logits.shape[logits.ndim - 1]

    // Get flat tokens
    let flatTokens = generatedTokens.reshaped(-1)

    // Get unique tokens
    // Note: MLX doesn't have a direct unique function, so we use a workaround
    // Create a mask for tokens within vocab range
    var tokenMask = MLXArray.zeros([vocabSize])

    // For each token, mark it in the mask
    // This is a simplified approach - in production, you'd want vectorized operations
    eval(flatTokens)
    let tokensArray = flatTokens.asArray(Int32.self)
    var mask = [Float](repeating: 0, count: vocabSize)
    for token in tokensArray {
      let idx = Int(token)
      if idx >= 0 && idx < vocabSize {
        mask[idx] = 1.0
      }
    }
    tokenMask = MLXArray(mask)

    // Apply penalty: if score < 0, multiply by penalty; if > 0, divide by penalty
    let penalized = MLX.where(logits .< 0, logits * penalty, logits / penalty)
    let result = MLX.where(tokenMask .> 0, penalized, logits)

    return result
  }

  /// Filter logits to only keep top-k values
  private func topKFiltering(logits: MLXArray, topK: Int) -> MLXArray {
    if topK <= 0 {
      return logits
    }

    // Ensure logits is 2D (batch, vocab)
    let origShape = logits.shape
    let vocabSize = origShape[origShape.count - 1]
    let batchSize = origShape.dropLast().reduce(1, *)
    let flatLogits = logits.reshaped([batchSize, vocabSize])

    let k = min(topK, vocabSize)

    // argpartition puts the k largest at the end
    let partitioned = MLX.argPartition(flatLogits, kth: -k, axis: -1)
    let kthIndices = partitioned[0..., (-k) ..< (-k + 1)] // (batch, 1)

    // Get k-th largest value using takeAlong
    let kthValues = takeAlong(flatLogits, kthIndices, axis: -1) // (batch, 1)

    // Create mask: keep values >= kth value
    let mask = flatLogits .>= kthValues

    // Apply mask
    let result = MLX.where(mask, flatLogits, MLXArray(-Float.infinity))

    // Reshape back to original shape
    return result.reshaped(origShape)
  }

  /// Filter logits using nucleus (top-p) sampling
  private func topPFiltering(logits: MLXArray, topP: Float) -> MLXArray {
    if topP >= 1.0 {
      return logits
    }

    // Ensure logits is 2D (batch, vocab)
    let origShape = logits.shape
    let vocabSize = origShape[origShape.count - 1]
    let batchSize = origShape.dropLast().reduce(1, *)
    let flatLogits = logits.reshaped([batchSize, vocabSize])

    // Sort logits in descending order
    let sortedIndices = MLX.argSort(-flatLogits, axis: -1)
    let sortedLogits = takeAlong(flatLogits, sortedIndices, axis: -1)

    // Compute cumulative probabilities
    let sortedProbs = MLX.softmax(sortedLogits, axis: -1)
    let cumulativeProbs = MLX.cumsum(sortedProbs, axis: -1)

    // Remove tokens with cumulative probability above threshold
    var sortedIndicesToRemove = cumulativeProbs .> topP

    // Shift right to keep first token above threshold
    let zeros = MLXArray.zeros([batchSize, 1])
    let shiftedPart = sortedIndicesToRemove[0..., 0 ..< (vocabSize - 1)]
    sortedIndicesToRemove = MLX.concatenated([zeros.asType(.bool), shiftedPart], axis: -1)

    // Set removed tokens to -inf
    let maskedSortedLogits = MLX.where(sortedIndicesToRemove, MLXArray(-Float.infinity), sortedLogits)

    // Scatter back using inverse permutation
    let inverseIndices = MLX.argSort(sortedIndices, axis: -1)
    let result = takeAlong(maskedSortedLogits, inverseIndices, axis: -1)

    // Reshape back to original shape
    return result.reshaped(origShape)
  }
}
