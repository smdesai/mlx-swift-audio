// Copyright © 2025 Resemble AI (original model implementation)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

import Foundation

// MARK: - GPT-2 Configuration

/// GPT-2 Medium configuration for T3 Turbo backbone
struct GPT2Config: Codable, Sendable {
  var vocabSize: Int = 50276
  var nPositions: Int = 8196
  var nEmbd: Int = 1024
  var nLayer: Int = 24
  var nHead: Int = 16
  var nInner: Int? = nil // Defaults to 4 * nEmbd
  var activationFunction: String = "gelu_new"
  var residPdrop: Float = 0.1
  var embdPdrop: Float = 0.1
  var attnPdrop: Float = 0.1
  var layerNormEpsilon: Float = 1e-5

  /// Hidden size (alias for nEmbd)
  var hiddenSize: Int { nEmbd }

  /// Number of attention heads (alias for nHead)
  var numAttentionHeads: Int { nHead }

  /// Number of hidden layers (alias for nLayer)
  var numHiddenLayers: Int { nLayer }

  /// Inner dimension for MLP (4 * nEmbd if not specified)
  var innerDim: Int { nInner ?? (4 * nEmbd) }

  enum CodingKeys: String, CodingKey {
    case vocabSize = "vocab_size"
    case nPositions = "n_positions"
    case nEmbd = "n_embd"
    case nLayer = "n_layer"
    case nHead = "n_head"
    case nInner = "n_inner"
    case activationFunction = "activation_function"
    case residPdrop = "resid_pdrop"
    case embdPdrop = "embd_pdrop"
    case attnPdrop = "attn_pdrop"
    case layerNormEpsilon = "layer_norm_epsilon"
  }

  init() {}

  /// Default GPT-2 Medium configuration for T3 Turbo
  static let gpt2Medium = GPT2Config()
}

// MARK: - T3 Turbo Configuration

/// Configuration for T3 Turbo (Token-to-Token) model
struct T3TurboConfig: Codable, Sendable {
  // Text token configuration
  var textTokensDictSize: Int = 50276 // GPT-2 vocabulary size
  var startTextToken: Int = 255
  var stopTextToken: Int = 0
  var maxTextTokens: Int = 2048

  // Speech token configuration
  var speechTokensDictSize: Int = 6563
  var startSpeechToken: Int = 6561
  var stopSpeechToken: Int = 6562
  var maxSpeechTokens: Int = 4096

  // Model architecture
  var gpt2ConfigName: String = "GPT2_medium"
  var inputPosEmb: String? = nil // Turbo uses GPT-2's built-in position embeddings
  var speechCondPromptLen: Int = 375

  // Conditioning
  var encoderType: String = "voice_encoder"
  var speakerEmbedSize: Int = 256
  var usePerceiverResampler: Bool = false // Not used in Turbo
  var emotionAdv: Bool = false // Not used in Turbo

  /// Get hidden size from GPT-2 config
  var nChannels: Int {
    GPT2Config.gpt2Medium.hiddenSize
  }

  /// Check if model is multilingual
  var isMultilingual: Bool {
    textTokensDictSize == 2454
  }

  enum CodingKeys: String, CodingKey {
    case textTokensDictSize = "text_tokens_dict_size"
    case startTextToken = "start_text_token"
    case stopTextToken = "stop_text_token"
    case maxTextTokens = "max_text_tokens"
    case speechTokensDictSize = "speech_tokens_dict_size"
    case startSpeechToken = "start_speech_token"
    case stopSpeechToken = "stop_speech_token"
    case maxSpeechTokens = "max_speech_tokens"
    case gpt2ConfigName = "gpt2_config_name"
    case inputPosEmb = "input_pos_emb"
    case speechCondPromptLen = "speech_cond_prompt_len"
    case encoderType = "encoder_type"
    case speakerEmbedSize = "speaker_embed_size"
    case usePerceiverResampler = "use_perceiver_resampler"
    case emotionAdv = "emotion_adv"
  }

  init() {}

  /// Create default Turbo configuration
  static func turbo() -> T3TurboConfig {
    T3TurboConfig()
  }
}

// MARK: - Voice Encoder Configuration (reuse from Chatterbox)

// VoiceEncConfig is shared with regular Chatterbox

// MARK: - Main Configuration

/// Main configuration for Chatterbox Turbo TTS model
struct ChatterboxTurboModelConfig: Codable, Sendable {
  // Model type for auto-detection
  var modelType: String = "chatterbox_turbo"

  // Model components
  var t3Config: T3TurboConfig

  // Sample rates
  var s3Sr: Int = 16000 // S3 tokenizer sample rate
  var s3genSr: Int = 24000 // S3Gen output sample rate
  var sampleRate: Int = 24000 // Output sample rate (alias for s3genSr)

  // Conditioning lengths
  var encCondLen: Int = 15 * 16000 // 15 seconds at 16kHz (longer than regular Chatterbox)
  var decCondLen: Int = 10 * 24000 // 10 seconds at 24kHz

  // Model path (set by load_model for tokenizer initialization)
  var modelPath: String?

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case t3Config = "t3_config"
    case s3Sr = "s3_sr"
    case s3genSr = "s3gen_sr"
    case sampleRate = "sample_rate"
    case encCondLen = "enc_cond_len"
    case decCondLen = "dec_cond_len"
    case modelPath = "model_path"
  }

  init() {
    t3Config = T3TurboConfig.turbo()
  }

  init(
    modelType: String = "chatterbox_turbo",
    t3Config: T3TurboConfig? = nil,
    s3Sr: Int = 16000,
    s3genSr: Int = 24000,
    sampleRate: Int? = nil,
    encCondLen: Int = 15 * 16000,
    decCondLen: Int = 10 * 24000,
    modelPath: String? = nil
  ) {
    self.modelType = modelType
    self.t3Config = t3Config ?? T3TurboConfig.turbo()
    self.s3Sr = s3Sr
    self.s3genSr = s3genSr
    self.sampleRate = sampleRate ?? s3genSr
    self.encCondLen = encCondLen
    self.decCondLen = decCondLen
    self.modelPath = modelPath
  }
}

// MARK: - Quantization Options

/// Quantization options for Chatterbox Turbo model weights
public enum ChatterboxTurboQuantization: String, Sendable, CaseIterable {
  case fp16
  case q8 = "8bit"
  case q4 = "4bit"
}

// MARK: - Constants

/// Constants used throughout the Chatterbox Turbo model
enum ChatterboxTurboConstants {
  static let s3Sr: Int = 16000 // S3 tokenizer sample rate
  static let s3genSr: Int = 24000 // S3Gen output sample rate
  static let speechVocabSize: Int = 6561 // 3^8 vocabulary size
  static let encCondLen: Int = 15 * 16000 // 15 seconds at 16kHz (240000 samples)
  static let decCondLen: Int = 10 * 24000 // 10 seconds at 24kHz (240000 samples)
  static let cfmSteps: Int = 2 // Turbo uses 2 CFM steps (vs 10 for regular)
  static let silenceToken: Int = 1516 // Silence token for S3Gen
}
