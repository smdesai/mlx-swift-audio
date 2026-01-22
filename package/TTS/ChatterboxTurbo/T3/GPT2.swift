// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Sachin Desai (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - GELU New Activation

/// Implementation of the GELU activation function used in GPT-2
/// Also known as "GELU approximate" or "tanh GELU"
/// See: https://arxiv.org/abs/1606.08415
func geluNew(_ x: MLXArray) -> MLXArray {
  0.5 * x * (1.0 + MLX.tanh(sqrt(2.0 / .pi) * (x + 0.044715 * MLX.pow(x, 3.0))))
}

// MARK: - GPT2 Attention

/// GPT-2 multi-head causal self-attention
class GPT2Attention: Module {
  let embedDim: Int
  let numHeads: Int
  let headDim: Int
  let scale: Float

  @ModuleInfo(key: "c_attn") var cAttn: Linear
  @ModuleInfo(key: "c_proj") var cProj: Linear

  init(config: GPT2Config) {
    embedDim = config.nEmbd
    numHeads = config.nHead
    headDim = embedDim / numHeads
    scale = pow(Float(headDim), -0.5)

    // Combined QKV projection
    _cAttn.wrappedValue = Linear(embedDim, 3 * embedDim)
    _cProj.wrappedValue = Linear(embedDim, embedDim)
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil,
    cache: KVCacheSimple? = nil
  ) -> (MLXArray, KVCacheSimple?) {
    let B = hiddenStates.shape[0]
    let T = hiddenStates.shape[1]
    // Use known embedDim instead of reading from input shape (which may be corrupted)

    // QKV projection
    let qkv = cAttn(hiddenStates)
    let qkvSplit = MLX.split(qkv, parts: 3, axis: -1)
    var q = qkvSplit[0]
    var k = qkvSplit[1]
    var v = qkvSplit[2]

    // Reshape to (B, num_heads, T, head_dim)
    q = q.reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
    k = k.reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
    v = v.reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)

    // Update KV cache if provided
    if let cache = cache {
      (k, v) = cache.update(keys: k, values: v)
    }

    // Attention with causal mask
    let output = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: .causal
    )

    // Reshape back to (B, T, embedDim)
    let transposed = output.transposed(0, 2, 1, 3)
    let attnOutput = transposed.reshaped([B, T, embedDim])

    // Output projection
    return (cProj(attnOutput), cache)
  }
}

// MARK: - GPT2 MLP

/// GPT-2 feed-forward network (MLP)
class GPT2MLP: Module {
  @ModuleInfo(key: "c_fc") var cFc: Linear
  @ModuleInfo(key: "c_proj") var cProj: Linear

  init(config: GPT2Config) {
    let innerDim = config.innerDim
    _cFc.wrappedValue = Linear(config.nEmbd, innerDim)
    _cProj.wrappedValue = Linear(innerDim, config.nEmbd)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    cProj(geluNew(cFc(x)))
  }
}

// MARK: - GPT2 Block

/// GPT-2 transformer block
class GPT2Block: Module {
  @ModuleInfo(key: "ln_1") var ln1: LayerNorm
  @ModuleInfo(key: "attn") var attn: GPT2Attention
  @ModuleInfo(key: "ln_2") var ln2: LayerNorm
  @ModuleInfo(key: "mlp") var mlp: GPT2MLP

  init(config: GPT2Config) {
    _ln1.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
    _attn.wrappedValue = GPT2Attention(config: config)
    _ln2.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
    _mlp.wrappedValue = GPT2MLP(config: config)
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil,
    cache: KVCacheSimple? = nil
  ) -> (MLXArray, KVCacheSimple?) {
    // Self-attention with pre-norm
    var residual = hiddenStates
    var x = ln1(hiddenStates)
    let (attnOutput, updatedCache) = attn(hiddenStates: x, attentionMask: attentionMask, cache: cache)
    x = residual + attnOutput

    // MLP with pre-norm
    residual = x
    x = residual + mlp(ln2(x))

    return (x, updatedCache)
  }
}

// MARK: - GPT2 Model

/// GPT-2 base model (without LM head)
class GPT2Model: Module {
  let config: GPT2Config

  @ModuleInfo(key: "wte") var wte: Embedding
  @ModuleInfo(key: "wpe") var wpe: Embedding
  @ModuleInfo(key: "h") var h: [GPT2Block]
  @ModuleInfo(key: "ln_f") var lnF: LayerNorm

  init(config: GPT2Config) {
    self.config = config

    // Token embeddings (not used when inputs_embeds is provided)
    _wte.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.nEmbd)
    _wpe.wrappedValue = Embedding(embeddingCount: config.nPositions, dimensions: config.nEmbd)

    // Transformer blocks
    _h.wrappedValue = (0 ..< config.nLayer).map { _ in GPT2Block(config: config) }

    // Final layer norm
    _lnF.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
  }

  /// Forward pass of GPT-2
  ///
  /// - Parameters:
  ///   - inputIds: Token IDs (B, T) - optional if inputsEmbeds provided
  ///   - inputsEmbeds: Pre-computed embeddings (B, T, D) - takes precedence over inputIds
  ///   - attentionMask: Optional attention mask
  ///   - cache: Optional array of KV caches for each layer
  /// - Returns: Tuple of (hidden_states, updated_cache)
  func callAsFunction(
    inputIds: MLXArray? = nil,
    inputsEmbeds: MLXArray? = nil,
    attentionMask: MLXArray? = nil,
    cache: [KVCacheSimple]? = nil
  ) -> (MLXArray, [KVCacheSimple]) {
    var hiddenStates: MLXArray

    if let embeds = inputsEmbeds {
      hiddenStates = embeds
    } else if let ids = inputIds {
      hiddenStates = wte(ids)
    } else {
      fatalError("Either inputIds or inputsEmbeds must be provided")
    }

    let T = hiddenStates.shape[1]

    // Determine past length from cache
    let pastLength: Int
    if let cache = cache, !cache.isEmpty {
      pastLength = cache[0].offset
    } else {
      pastLength = 0
    }

    // Add positional embeddings
    let positionIds = MLXArray(Int32(pastLength) ..< Int32(pastLength + T))
    let positionEmbeds = wpe(positionIds)
    hiddenStates = hiddenStates + positionEmbeds

    // Initialize cache if not provided
    let kvCache = cache ?? (0 ..< config.nLayer).map { _ in KVCacheSimple() }

    // Forward through transformer blocks
    for i in 0 ..< h.count {
      let (output, _) = h[i](
        hiddenStates: hiddenStates,
        attentionMask: attentionMask,
        cache: kvCache[i]
      )
      hiddenStates = output
    }

    // Final layer norm
    hiddenStates = lnF(hiddenStates)

    return (hiddenStates, kvCache)
  }
}
