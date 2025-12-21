// Copyright © 2025 Resemble AI (original model implementation)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

import Foundation
import MLX
import MLXNN

// MARK: - T3 Turbo Conditioning Data

/// Container for T3 Turbo conditioning information
///
/// Marked `@unchecked Sendable` because it contains non-Sendable MLXArray fields,
/// but all access is controlled within the `ChatterboxTurboTTS` actor's methods.
public struct T3TurboCond: @unchecked Sendable {
  /// Speaker embedding from voice encoder (B, 256)
  public var speakerEmb: MLXArray

  /// Optional CLAP embedding (not used in Turbo)
  public var clapEmb: MLXArray?

  /// Optional conditioning speech tokens (B, speechCondPromptLen)
  public var condPromptSpeechTokens: MLXArray?

  /// Optional conditioning speech embeddings (B, speechCondPromptLen, dim)
  /// Pre-computed from condPromptSpeechTokens
  public var condPromptSpeechEmb: MLXArray?

  /// Optional emotion adversarial value (not used in Turbo)
  public var emotionAdv: MLXArray?

  public init(
    speakerEmb: MLXArray,
    clapEmb: MLXArray? = nil,
    condPromptSpeechTokens: MLXArray? = nil,
    condPromptSpeechEmb: MLXArray? = nil,
    emotionAdv: MLXArray? = nil
  ) {
    self.speakerEmb = speakerEmb
    self.clapEmb = clapEmb
    self.condPromptSpeechTokens = condPromptSpeechTokens
    self.condPromptSpeechEmb = condPromptSpeechEmb
    self.emotionAdv = emotionAdv
  }
}

// MARK: - T3 Turbo Conditioning Encoder

/// Handles all non-text conditioning for T3 Turbo:
/// speaker embeddings, prompt speech tokens, CLAP, emotion, etc.
class T3TurboCondEnc: Module {
  let hp: T3TurboConfig

  @ModuleInfo(key: "spkr_enc") var spkrEnc: Linear
  @ModuleInfo(key: "emotion_adv_fc") var emotionAdvFc: Linear?

  init(hp: T3TurboConfig) {
    self.hp = hp

    // Speaker encoder projection
    if hp.encoderType == "voice_encoder" {
      _spkrEnc.wrappedValue = Linear(hp.speakerEmbedSize, hp.nChannels)
    } else {
      fatalError("Unsupported encoder type: \(hp.encoderType)")
    }

    // Emotion adv (not used in Turbo, but kept for compatibility)
    if hp.emotionAdv {
      _emotionAdvFc.wrappedValue = Linear(1, hp.nChannels, bias: false)
    } else {
      _emotionAdvFc.wrappedValue = nil
    }
  }

  func callAsFunction(_ cond: T3TurboCond) -> MLXArray {
    // Validate: tokens and embeddings must match
    assert(
      (cond.condPromptSpeechTokens == nil) == (cond.condPromptSpeechEmb == nil),
      "condPromptSpeechTokens and condPromptSpeechEmb must both be present or both be nil"
    )

    // Speaker embedding projection
    let speakerEmb = cond.speakerEmb.reshaped(-1, hp.speakerEmbedSize)
    let condSpkr = spkrEnc(speakerEmb).expandedDimensions(axis: 1) // (B, 1, dim)

    let B = condSpkr.shape[0]
    let dim = condSpkr.shape[2]

    // Empty tensor for unused conditions
    let empty = MLXArray.zeros([B, 0, dim])

    // CLAP (not implemented/used in Turbo)
    assert(cond.clapEmb == nil, "CLAP embedding not implemented")
    let condClap = empty

    // Conditioning prompt speech embeddings
    var condPromptSpeechEmb = cond.condPromptSpeechEmb
    if condPromptSpeechEmb == nil {
      condPromptSpeechEmb = empty
    }
    // Perceiver resampler not used in Turbo

    // Emotion Adv (not used in Turbo)
    var condEmotionAdv = empty
    if hp.emotionAdv, let emotionAdv = cond.emotionAdv, let fc = emotionAdvFc {
      let emotionVal = emotionAdv.reshaped(-1, 1, 1)
      condEmotionAdv = fc(emotionVal)
    }

    // Concatenate all conditions
    let condEmbeds = MLX.concatenated(
      [
        condSpkr,
        condClap,
        condPromptSpeechEmb ?? empty,
        condEmotionAdv,
      ],
      axis: 1
    )

    return condEmbeds
  }
}
