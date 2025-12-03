//
//  T3LlamaBackbone.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/t3/t3.py
//  LLaMA backbone for T3 model
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Llama3 Scaled RoPE

/// Llama3-style RoPE with scaling - uses MLXFast.RoPE for efficiency
class T3Llama3ScaledRoPE: Module {
  let dims: Int
  let scale: Float

  /// Pre-computed frequencies for llama3 scaling (set base to nil when using this)
  private var freqs: MLXArray?

  init(
    dims: Int,
    maxSeqLen _: Int = 2048,
    base: Float = 500_000.0,
    scaleFactor: Float = 8.0,
    lowFreqFactor: Float = 1.0,
    highFreqFactor: Float = 4.0,
    oldContextLen: Float = 8192.0,
  ) {
    precondition(dims % 2 == 0, "RoPE dims must be even")
    self.dims = dims
    scale = 1.0
    super.init()

    // Compute scaled frequencies for llama3 (matches mlx-swift-lm's DynamicNTKScalingRoPE)
    computeFreqs(
      base: base,
      scaleFactor: scaleFactor,
      lowFreqFactor: lowFreqFactor,
      highFreqFactor: highFreqFactor,
      oldContextLen: oldContextLen,
    )
  }

  convenience init(dims: Int, config: LlamaConfig) {
    self.init(
      dims: dims,
      maxSeqLen: config.maxPositionEmbeddings,
      base: config.ropeTheta,
      scaleFactor: config.ropeScaling.factor,
      lowFreqFactor: config.ropeScaling.lowFreqFactor,
      highFreqFactor: config.ropeScaling.highFreqFactor,
      oldContextLen: Float(config.ropeScaling.originalMaxPositionEmbeddings),
    )
  }

  private func computeFreqs(
    base: Float,
    scaleFactor: Float,
    lowFreqFactor: Float,
    highFreqFactor: Float,
    oldContextLen: Float,
  ) {
    let lowFreqWavelen = oldContextLen / lowFreqFactor
    let highFreqWavelen = oldContextLen / highFreqFactor

    // Compute base frequencies
    let indices = MLXArray(stride(from: 0, to: dims, by: 2))
    var frequencies = MLX.pow(MLXArray(base), indices / Float(dims))
    let wavelens = 2 * Float.pi * frequencies

    // Apply llama3 scaling
    frequencies = MLX.where(
      wavelens .> MLXArray(lowFreqWavelen),
      frequencies * scaleFactor,
      frequencies,
    )

    let isMediumFreq = MLX.logicalAnd(
      wavelens .> MLXArray(highFreqWavelen),
      wavelens .< MLXArray(lowFreqWavelen),
    )

    let smoothFactors = (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
    let smoothFreqs = frequencies / ((1 - smoothFactors) / scaleFactor + smoothFactors)

    freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
  }

  func callAsFunction(_ x: MLXArray, offset: Int? = nil) -> MLXArray {
    // Use MLXFast.RoPE with pre-computed frequencies (base is nil when using custom freqs)
    MLXFast.RoPE(
      x,
      dimensions: dims,
      traditional: false,
      base: nil,
      scale: scale,
      offset: offset ?? 0,
      freqs: freqs,
    )
  }
}

// MARK: - LLaMA Attention

class T3LlamaAttention: Module {
  let config: LlamaConfig
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear

  let rope: T3Llama3ScaledRoPE

  init(_ config: LlamaConfig) {
    self.config = config

    let dim = config.hiddenSize
    let heads = config.numAttentionHeads
    let kvHeads = config.numKeyValueHeads
    let headDim = config.headDim

    scale = pow(Float(headDim), -0.5)

    _qProj.wrappedValue = Linear(dim, heads * headDim, bias: config.attentionBias)
    _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
    _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
    _oProj.wrappedValue = Linear(heads * headDim, dim, bias: config.attentionBias)

    rope = T3Llama3ScaledRoPE(dims: headDim, config: config)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: T3KVCache?,
  ) -> MLXArray {
    let B = x.shape[0]
    let L = x.shape[1]

    var queries = qProj(x)
    var keys = kProj(x)
    var values = vProj(x)

    queries = queries.reshaped([B, L, config.numAttentionHeads, -1]).transposed(0, 2, 1, 3)
    keys = keys.reshaped([B, L, config.numKeyValueHeads, -1]).transposed(0, 2, 1, 3)
    values = values.reshaped([B, L, config.numKeyValueHeads, -1]).transposed(0, 2, 1, 3)

    if let cache {
      queries = rope(queries, offset: cache.offset)
      keys = rope(keys, offset: cache.offset)
    } else {
      queries = rope(queries)
      keys = rope(keys)
    }

    if let cache {
      let (updatedKeys, updatedValues) = cache.updateAndFetch(keys, values)
      let attnResult = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: updatedKeys,
        values: updatedValues,
        scale: scale,
        mask: mask,
      )
      let transposed = attnResult.transposed(0, 2, 1, 3)
      let output = transposed.reshaped([B, L, config.numAttentionHeads * config.headDim])
      return oProj(output)
    } else {
      let attnResult = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: mask,
      )
      let transposed = attnResult.transposed(0, 2, 1, 3)
      let output = transposed.reshaped([B, L, config.numAttentionHeads * config.headDim])
      return oProj(output)
    }
  }
}

// MARK: - MLP

class T3MLP: Module, UnaryLayer {
  @ModuleInfo(key: "gate_proj") var gate: Linear
  @ModuleInfo(key: "down_proj") var down: Linear
  @ModuleInfo(key: "up_proj") var up: Linear

  init(_ config: LlamaConfig) {
    _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
    _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let activation = silu(gate(x))
    return down(activation * up(x))
  }
}

// MARK: - Transformer Block

class T3TransformerBlock: Module {
  @ModuleInfo(key: "self_attn") var attention: T3LlamaAttention
  @ModuleInfo(key: "mlp") var mlp: T3MLP

  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(_ config: LlamaConfig) {
    _attention.wrappedValue = T3LlamaAttention(config)
    _mlp.wrappedValue = T3MLP(config)
    _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: T3KVCache?,
  ) -> MLXArray {
    var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
    let h = x + r
    r = mlp(postAttentionLayerNorm(h))
    return h + r
  }
}

// MARK: - LLaMA Model Inner

/// Inner model class for weight path matching (tfmr.model.layers, tfmr.model.norm)
class T3LlamaModel: Module {
  @ModuleInfo(key: "layers") var layers: [T3TransformerBlock]
  @ModuleInfo(key: "norm") var norm: RMSNorm

  init(_ config: LlamaConfig) {
    _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in T3TransformerBlock(config) }
    _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }
}

// MARK: - LLaMA Backbone

/// LLaMA backbone for T3 model
public class T3LlamaBackbone: Module {
  public let config: LlamaConfig
  public let kvHeads: [Int]

  @ModuleInfo(key: "model") var model: T3LlamaModel

  public init(_ config: LlamaConfig) {
    self.config = config
    kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
    _model.wrappedValue = T3LlamaModel(config)
  }

  public func callAsFunction(_ inputs: MLXArray, cache: [T3KVCache]?) -> MLXArray {
    var h = inputs

    let mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal

    for (i, layer) in model.layers.enumerated() {
      h = layer(h, mask: mask, cache: cache?[i])
    }

    return model.norm(h)
  }

  /// Create KV caches for all layers
  public func createCache(batchSize _: Int = 1) -> [T3KVCache] {
    (0 ..< config.numHiddenLayers).map { _ in
      T3KVCache(headDim: config.headDim, nKVHeads: config.numKeyValueHeads)
    }
  }
}

// MARK: - T3KVCache for T3

/// KV Cache implementation for T3 model
public final class T3KVCache {
  let nKVHeads: Int
  let headDim: Int

  private(set) var keys: MLXArray?
  private(set) var values: MLXArray?
  private(set) var offset: Int = 0
  var step: Int = 256

  public init(headDim: Int, nKVHeads: Int, step: Int = 256) {
    self.nKVHeads = nKVHeads
    self.headDim = headDim
    self.step = step
  }

  public func reset() {
    offset = 0
    keys = nil
    values = nil
  }

  public func updateAndFetch(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
    let B = k.shape[0]
    precondition(k.shape[1] == nKVHeads && v.shape[1] == nKVHeads, "nKVHeads mismatch")
    let t = k.shape[2]
    precondition(k.shape[3] == headDim, "k head dim mismatch")
    precondition(v.shape[3] == headDim, "v head dim mismatch")
    if let kk = keys { precondition(kk.shape[0] == B, "batch size changed in KV cache") }

    ensureCapacity(timeToAppend: t, batch: B, dtype: k.dtype)

    let prev = offset
    offset += t

    // Use slice assignment instead of split+concat (much more efficient)
    keys![0..., 0..., prev ..< offset, 0...] = k
    values![0..., 0..., prev ..< offset, 0...] = v

    // Use slice instead of split for extracting used portion
    let kUsed = keys![0..., 0..., 0 ..< offset, 0...]
    let vUsed = values![0..., 0..., 0 ..< offset, 0...]
    return (kUsed, vUsed)
  }

  private func ensureCapacity(timeToAppend t: Int, batch B: Int, dtype: DType) {
    let prev = offset
    if keys == nil || (prev + t) > keys!.shape[2] {
      let nSteps = (t + step - 1) / step
      let allocT = nSteps * step

      let newK = MLXArray.zeros([B, nKVHeads, allocT, headDim]).asType(dtype)
      let newV = MLXArray.zeros([B, nKVHeads, allocT, headDim]).asType(dtype)

      if var kExisting = keys, var vExisting = values {
        if prev % step != 0 {
          kExisting = MLX.split(kExisting, indices: [prev], axis: 2)[0]
          vExisting = MLX.split(vExisting, indices: [prev], axis: 2)[0]
        }
        keys = MLX.concatenated([kExisting, newK], axis: 2)
        values = MLX.concatenated([vExisting, newV], axis: 2)
      } else {
        keys = newK
        values = newV
      }
    }
  }
}
