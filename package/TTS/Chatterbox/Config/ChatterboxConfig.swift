//
//  ChatterboxConfig.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/config.py
//

import Foundation

/// LLaMA 520M configuration for T3 backbone
public struct LlamaConfig: Codable, Sendable {
    public var modelType: String = "llama"
    public var vocabSize: Int = 8  // Unused due to custom input layers
    public var hiddenSize: Int = 1024
    public var numHiddenLayers: Int = 30
    public var intermediateSize: Int = 4096
    public var numAttentionHeads: Int = 16
    public var numKeyValueHeads: Int = 16
    public var headDim: Int = 64
    public var maxPositionEmbeddings: Int = 131072
    public var rmsNormEps: Float = 1e-05
    public var ropeTheta: Float = 500000.0
    public var ropeScaling: RopeScaling = RopeScaling()
    public var attentionBias: Bool = false
    public var mlpBias: Bool = false
    public var tieWordEmbeddings: Bool = false

    public struct RopeScaling: Codable, Sendable {
        public var factor: Float = 8.0
        public var highFreqFactor: Float = 4.0
        public var lowFreqFactor: Float = 1.0
        public var originalMaxPositionEmbeddings: Int = 8192
        public var ropeType: String = "llama3"

        enum CodingKeys: String, CodingKey {
            case factor
            case highFreqFactor = "high_freq_factor"
            case lowFreqFactor = "low_freq_factor"
            case originalMaxPositionEmbeddings = "original_max_position_embeddings"
            case ropeType = "rope_type"
        }

        public init() {}
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init() {}

    /// Default LLaMA 520M configuration
    public static let llama520M = LlamaConfig()
}

/// Configuration for T3 (Token-to-Token) model
public struct T3Config: Codable, Sendable {
    // Text token configuration
    public var textTokensDictSize: Int = 704  // English: 704, Multilingual: 2454
    public var startTextToken: Int = 255
    public var stopTextToken: Int = 0
    public var maxTextTokens: Int = 2048

    // Speech token configuration
    public var speechTokensDictSize: Int = 8194
    public var startSpeechToken: Int = 6561
    public var stopSpeechToken: Int = 6562
    public var maxSpeechTokens: Int = 4096

    // Model architecture
    public var llamaConfigName: String = "Llama_520M"
    public var inputPosEmb: String = "learned"  // "learned" or "rope"
    public var speechCondPromptLen: Int = 150

    // Conditioning
    public var encoderType: String = "voice_encoder"
    public var speakerEmbedSize: Int = 256
    public var usePerceiverResampler: Bool = true
    public var emotionAdv: Bool = true

    /// Get hidden size from LLaMA config
    public var nChannels: Int {
        return LlamaConfig.llama520M.hiddenSize
    }

    /// Check if model is multilingual
    public var isMultilingual: Bool {
        return textTokensDictSize == 2454
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
        case llamaConfigName = "llama_config_name"
        case inputPosEmb = "input_pos_emb"
        case speechCondPromptLen = "speech_cond_prompt_len"
        case encoderType = "encoder_type"
        case speakerEmbedSize = "speaker_embed_size"
        case usePerceiverResampler = "use_perceiver_resampler"
        case emotionAdv = "emotion_adv"
    }

    public init() {}

    /// Create configuration for English-only TTS model
    public static func englishOnly() -> T3Config {
        var config = T3Config()
        config.textTokensDictSize = 704
        return config
    }

    /// Create configuration for multilingual TTS model
    public static func multilingual() -> T3Config {
        var config = T3Config()
        config.textTokensDictSize = 2454
        return config
    }
}

/// Voice encoder configuration
public struct VoiceEncConfig: Codable, Sendable {
    public var numMels: Int = 40
    public var sampleRate: Int = 16000
    public var speakerEmbedSize: Int = 256
    public var veHiddenSize: Int = 256
    public var nFft: Int = 400
    public var hopSize: Int = 160
    public var winSize: Int = 400
    public var fmax: Int = 8000
    public var fmin: Int = 0
    public var preemphasis: Float = 0.0
    public var melPower: Float = 2.0
    public var melType: String = "amp"
    public var normalizedMels: Bool = false
    public var vePartialFrames: Int = 160
    public var veFinalRelu: Bool = true
    public var stftMagnitudeMin: Float = 1e-4

    enum CodingKeys: String, CodingKey {
        case numMels = "num_mels"
        case sampleRate = "sample_rate"
        case speakerEmbedSize = "speaker_embed_size"
        case veHiddenSize = "ve_hidden_size"
        case nFft = "n_fft"
        case hopSize = "hop_size"
        case winSize = "win_size"
        case fmax
        case fmin
        case preemphasis
        case melPower = "mel_power"
        case melType = "mel_type"
        case normalizedMels = "normalized_mels"
        case vePartialFrames = "ve_partial_frames"
        case veFinalRelu = "ve_final_relu"
        case stftMagnitudeMin = "stft_magnitude_min"
    }

    public init() {}
}

/// Main configuration for Chatterbox TTS model
public struct ChatterboxModelConfig: Codable, Sendable {
    // Model type for auto-detection
    public var modelType: String = "chatterbox"

    // Model components
    public var t3Config: T3Config

    // Sample rates
    public var s3Sr: Int = 16000  // S3 tokenizer sample rate
    public var s3genSr: Int = 24000  // S3Gen output sample rate
    public var sampleRate: Int = 24000  // Output sample rate (alias for s3genSr)

    // Conditioning lengths
    public var encCondLen: Int = 6 * 16000  // 6 seconds at 16kHz
    public var decCondLen: Int = 10 * 24000  // 10 seconds at 24kHz

    // Model path (set by load_model for tokenizer initialization)
    public var modelPath: String?

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

    public init() {
        self.t3Config = T3Config.englishOnly()
    }

    public init(
        modelType: String = "chatterbox",
        t3Config: T3Config? = nil,
        s3Sr: Int = 16000,
        s3genSr: Int = 24000,
        sampleRate: Int? = nil,
        encCondLen: Int = 6 * 16000,
        decCondLen: Int = 10 * 24000,
        modelPath: String? = nil
    ) {
        self.modelType = modelType
        self.t3Config = t3Config ?? T3Config.englishOnly()
        self.s3Sr = s3Sr
        self.s3genSr = s3genSr
        self.sampleRate = sampleRate ?? s3genSr
        self.encCondLen = encCondLen
        self.decCondLen = decCondLen
        self.modelPath = modelPath
    }
}

/// Constants used throughout the Chatterbox model
public enum ChatterboxConstants {
    public static let s3Sr: Int = 16000  // S3 tokenizer sample rate
    public static let s3genSr: Int = 24000  // S3Gen output sample rate
    public static let speechVocabSize: Int = 6561  // 3^8 vocabulary size
    public static let encCondLen: Int = 6 * 16000  // 6 seconds at 16kHz (96000 samples)
    public static let decCondLen: Int = 10 * 24000  // 10 seconds at 24kHz (240000 samples)
}
