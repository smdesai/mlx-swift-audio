//
//  S3TokenizerConfig.swift
//  MLXAudio
//
//  Ported from mlx_audio/codec/models/s3/
//

import Foundation

/// S3Tokenizer constants
public enum S3TokenizerConstants {
    public static let s3Sr: Int = 16_000  // Sample rate for S3Tokenizer
    public static let s3Hop: Int = 160  // 100 frames/sec
    public static let s3TokenHop: Int = 640  // 25 tokens/sec
    public static let s3TokenRate: Int = 25
    public static let speechVocabSize: Int = 6561  // 3^8
}

/// Configuration for S3Tokenizer V2
public struct S3TokenizerModelConfig: Codable, Sendable {
    public var nMels: Int = 128
    public var nAudioCtx: Int = 1500
    public var nAudioState: Int = 1280
    public var nAudioHead: Int = 20
    public var nAudioLayer: Int = 6
    public var nCodebookSize: Int = 6561  // 3^8

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case nAudioCtx = "n_audio_ctx"
        case nAudioState = "n_audio_state"
        case nAudioHead = "n_audio_head"
        case nAudioLayer = "n_audio_layer"
        case nCodebookSize = "n_codebook_size"
    }

    public init() {}

    public init(
        nMels: Int = 128,
        nAudioCtx: Int = 1500,
        nAudioState: Int = 1280,
        nAudioHead: Int = 20,
        nAudioLayer: Int = 6,
        nCodebookSize: Int = 6561
    ) {
        self.nMels = nMels
        self.nAudioCtx = nAudioCtx
        self.nAudioState = nAudioState
        self.nAudioHead = nAudioHead
        self.nAudioLayer = nAudioLayer
        self.nCodebookSize = nCodebookSize
    }
}
