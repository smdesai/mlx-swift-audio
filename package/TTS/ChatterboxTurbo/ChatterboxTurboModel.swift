// Copyright © 2025 Resemble AI (original model implementation)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Turbo Conditionals

/// Container for T3 Turbo and S3Gen conditioning.
///
/// Marked `@unchecked Sendable` because it contains non-Sendable MLXArray fields
/// (via `T3TurboCond` and `S3GenRefDict`), but all access is controlled within the
/// `ChatterboxTurboTTS` actor's methods.
struct ChatterboxTurboConditionals: @unchecked Sendable {
  /// T3 Turbo conditioning (speaker embedding, prompt tokens)
  var t3: T3TurboCond

  /// S3Gen reference dictionary
  var gen: S3GenRefDict

  init(t3: T3TurboCond, gen: S3GenRefDict) {
    self.t3 = t3
    self.gen = gen
  }
}

// MARK: - ChatterboxTurboModel

/// Chatterbox Turbo neural network model (Module-based)
///
/// This integrates:
/// - T3Turbo: GPT-2-based text-to-speech-token generator (faster than regular Chatterbox)
/// - S3Gen: Flow matching decoder with HiFi-GAN vocoder (with meanflow for 2-step CFM)
/// - VoiceEncoder: Speaker embedding extractor
/// - S3Tokenizer: Speech tokenizer for reference audio (loaded from separate repo)
///
/// Key differences from regular Chatterbox:
/// - Uses GPT-2 Medium backbone instead of LLaMA
/// - Uses GPT-2 tokenizer instead of EnTokenizer
/// - Uses meanflow mode (2 CFM steps vs 10) for faster generation
/// - Longer encoder conditioning (15s vs 6s)
class ChatterboxTurboModel: Module {
  /// Hugging Face repository for S3TokenizerV2 (shared across TTS models)
  static let s3TokenizerRepoId = "mlx-community/S3TokenizerV2"

  /// Hugging Face repository for GPT-2 tokenizer base (has tokenizer.json + tokenizer_class)
  static let gpt2TokenizerRepoId = "EleutherAI/gpt-neo-125m"

  /// Additional tokenizer files from model directory (emotion tokens)
  static let additionalTokenizerFiles = ["added_tokens.json"]

  /// Get Hugging Face repository ID for specified quantization level
  static func repoId(quantization: ChatterboxTurboQuantization = .q4) -> String {
    switch quantization {
    case .fp16:
      return "mlx-community/chatterbox-turbo-fp16"
    case .q8:
      return "mlx-community/chatterbox-turbo-8bit"
    case .q4:
      return "mlx-community/chatterbox-turbo-4bit"
    }
  }

  /// Default Hugging Face repository ID (4-bit quantized)
  static var defaultRepoId: String {
    repoId(quantization: .q4)
  }

  /// Output sample rate
  let sr: Int = ChatterboxTurboConstants.s3genSr

  /// Configuration
  let config: ChatterboxTurboModelConfig

  /// T3 Turbo model (text to speech tokens using GPT-2)
  @ModuleInfo(key: "t3") var t3: T3Turbo

  /// S3Gen model (speech tokens to waveform)
  @ModuleInfo(key: "s3gen") var s3gen: S3Token2Wav

  /// Voice encoder (speaker embedding)
  @ModuleInfo(key: "ve") var ve: VoiceEncoder

  /// S3 tokenizer (speech tokenization)
  @ModuleInfo(key: "s3_tokenizer") var s3Tokenizer: S3TokenizerV2

  /// GPT-2 text tokenizer
  var textTokenizer: Tokenizer?

  /// Pre-computed conditionals (optional)
  var conds: ChatterboxTurboConditionals?

  init(config: ChatterboxTurboModelConfig? = nil) {
    self.config = config ?? ChatterboxTurboModelConfig()

    _t3.wrappedValue = T3Turbo(hp: self.config.t3Config)
    _s3gen.wrappedValue = S3Token2Wav(meanflow: true)
    _ve.wrappedValue = VoiceEncoder()
    _s3Tokenizer.wrappedValue = S3TokenizerV2()
  }

  /// Load text tokenizer with emotion control tokens
  ///
  /// Downloads GPT-2's tokenizer.json and merges in the emotion tokens from
  /// the model's added_tokens.json to support emotion control tags like
  /// [laugh], [sigh], [cough], [gasp], [chuckle], etc.
  func loadTextTokenizer(gpt2Directory: URL, modelDirectory: URL) async throws {
    // Load GPT-2's tokenizer.json
    let gpt2TokenizerURL = gpt2Directory.appending(path: "tokenizer.json")
    let gpt2TokenizerData = try Data(contentsOf: gpt2TokenizerURL)
    guard var tokenizerDict = try JSONSerialization.jsonObject(with: gpt2TokenizerData) as? [String: Any] else {
      throw ChatterboxTurboError.tokenizerLoadFailed("Failed to parse GPT-2 tokenizer.json")
    }

    // Load added_tokens.json from model directory (contains emotion tokens)
    let addedTokensURL = modelDirectory.appending(path: "added_tokens.json")
    if FileManager.default.fileExists(atPath: addedTokensURL.path) {
      let addedTokensData = try Data(contentsOf: addedTokensURL)
      if let addedTokensDict = try JSONSerialization.jsonObject(with: addedTokensData) as? [String: Int] {
        // Get existing added_tokens array from tokenizer
        var addedTokensArray = tokenizerDict["added_tokens"] as? [[String: Any]] ?? []

        // Add emotion tokens to the array
        for (content, id) in addedTokensDict {
          let newToken: [String: Any] = [
            "id": id,
            "content": content,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": false,
          ]
          addedTokensArray.append(newToken)
        }

        tokenizerDict["added_tokens"] = addedTokensArray
        Log.model.info("Added \(addedTokensDict.count) emotion tokens to tokenizer")
      }
    }

    // Write merged tokenizer to temporary directory with required filename
    let tempDir = FileManager.default.temporaryDirectory.appending(path: "chatterbox_turbo_tokenizer")
    try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    let mergedTokenizerURL = tempDir.appending(path: "tokenizer.json")
    let mergedData = try JSONSerialization.data(withJSONObject: tokenizerDict, options: .prettyPrinted)
    try mergedData.write(to: mergedTokenizerURL)

    // Copy tokenizer_config.json from GPT-2 directory (has tokenizer_class)
    let gpt2ConfigURL = gpt2Directory.appending(path: "tokenizer_config.json")
    let tempConfigURL = tempDir.appending(path: "tokenizer_config.json")
    if FileManager.default.fileExists(atPath: gpt2ConfigURL.path) {
      try? FileManager.default.removeItem(at: tempConfigURL)
      try FileManager.default.copyItem(at: gpt2ConfigURL, to: tempConfigURL)
    }

    // Load tokenizer from temporary directory
    textTokenizer = try await AutoTokenizer.from(modelFolder: tempDir)
  }

  /// Output sample rate
  var sampleRate: Int {
    ChatterboxTurboConstants.s3genSr
  }

  // MARK: - Model Loading

  /// Sanitize weights by removing keys that shouldn't be loaded
  private static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]

    for (key, value) in weights {
      // Skip S3Tokenizer weights - loaded from separate repo
      if key.hasPrefix("s3_tokenizer.") { continue }

      // Skip computed buffers
      if key.contains("freqsCis") || key.contains("freqs_cis") { continue }
      if key.contains("trimFade") || key.contains("trim_fade") { continue }
      if key.contains("randNoise") || key.contains("rand_noise") { continue }
      if key.contains("stftWindow") || key.contains("stft_window") { continue }
      if key.contains("pos_enc.pe") || key.contains("posEnc.pe") { continue }
      if key.contains("num_batches_tracked") { continue }

      var finalKey = key

      // HiFiGAN Conv1d wrapper: Only in mel2wav, Python has .conv. inside Conv1d, Swift doesn't
      // e.g., s3gen.mel2wav.source_resblocks.0.convs1.0.conv.weight -> s3gen.mel2wav.source_resblocks.0.convs1.0.weight
      // But NOT for up_layer which needs .conv. prefix
      if finalKey.contains("mel2wav") && !finalKey.contains("up_layer") {
        finalKey = finalKey.replacingOccurrences(of: ".conv.weight", with: ".weight")
        finalKey = finalKey.replacingOccurrences(of: ".conv.bias", with: ".bias")
      }

      // Upsample1D: Python has up_layer.weight directly, Swift needs up_layer.conv.weight
      finalKey = finalKey.replacingOccurrences(of: ".up_layer.weight", with: ".up_layer.conv.weight")
      finalKey = finalKey.replacingOccurrences(of: ".up_layer.bias", with: ".up_layer.conv.bias")

      // VoiceEncoder LSTM layer naming: ve.lstm1 -> ve.lstm.layers.0, etc.
      finalKey = finalKey.replacingOccurrences(of: "ve.lstm1.", with: "ve.lstm.layers.0.")
      finalKey = finalKey.replacingOccurrences(of: "ve.lstm2.", with: "ve.lstm.layers.1.")
      finalKey = finalKey.replacingOccurrences(of: "ve.lstm3.", with: "ve.lstm.layers.2.")

      // CausalBlock1D structure mapping:
      // Python: block2.block.conv.conv.weight → Swift: block2.conv.conv.weight
      // Python: block2.block.norm.weight     → Swift: block2.norm.weight
      // Python: block2.block.weight          → Swift: block2.norm.weight
      // The key insight: Python has .block.conv.conv. (4 levels), Swift has .conv.conv. (3 levels)

      // Conv patterns: .blockN.block.conv.conv. → .blockN.conv.conv. (remove one level)
      finalKey = finalKey.replacingOccurrences(of: ".block1.block.conv.conv.", with: ".block1.conv.conv.")
      finalKey = finalKey.replacingOccurrences(of: ".block2.block.conv.conv.", with: ".block2.conv.conv.")
      finalKey = finalKey.replacingOccurrences(of: ".final_block.block.conv.conv.", with: ".final_block.conv.conv.")

      // Norm patterns: .blockN.block.norm. → .blockN.norm.
      finalKey = finalKey.replacingOccurrences(of: ".block1.block.norm.", with: ".block1.norm.")
      finalKey = finalKey.replacingOccurrences(of: ".block2.block.norm.", with: ".block2.norm.")
      finalKey = finalKey.replacingOccurrences(of: ".final_block.block.norm.", with: ".final_block.norm.")

      // LayerNorm weight/bias directly under .block.: .blockN.block.weight → .blockN.norm.weight
      finalKey = finalKey.replacingOccurrences(of: ".block1.block.weight", with: ".block1.norm.weight")
      finalKey = finalKey.replacingOccurrences(of: ".block1.block.bias", with: ".block1.norm.bias")
      finalKey = finalKey.replacingOccurrences(of: ".block2.block.weight", with: ".block2.norm.weight")
      finalKey = finalKey.replacingOccurrences(of: ".block2.block.bias", with: ".block2.norm.bias")
      finalKey = finalKey.replacingOccurrences(of: ".final_block.block.weight", with: ".final_block.norm.weight")
      finalKey = finalKey.replacingOccurrences(of: ".final_block.block.bias", with: ".final_block.norm.bias")

      // CausalBlock1D in Python uses a LIST for self.block:
      //   block.0 = CausalConv1d (has Conv1dPT.conv which is Conv1d) → Swift: conv (has CausalConv1d.conv which is Conv1d)
      //   block.1 = LayerNorm → Swift: norm
      // Python: block.0.conv.conv.weight (CausalConv1d → Conv1dPT → Conv1d)
      // Swift:  conv.conv.weight (CausalConv1d → Conv1d)
      // So we need to collapse: .block.0.conv.conv. → .conv.conv.
      finalKey = finalKey.replacingOccurrences(of: ".block1.block.0.conv.conv.", with: ".block1.conv.conv.")
      finalKey = finalKey.replacingOccurrences(of: ".block2.block.0.conv.conv.", with: ".block2.conv.conv.")
      finalKey = finalKey.replacingOccurrences(of: ".final_block.block.0.conv.conv.", with: ".final_block.conv.conv.")
      // LayerNorm: block.1 → norm
      finalKey = finalKey.replacingOccurrences(of: ".block1.block.1.", with: ".block1.norm.")
      finalKey = finalKey.replacingOccurrences(of: ".block2.block.1.", with: ".block2.norm.")
      finalKey = finalKey.replacingOccurrences(of: ".final_block.block.1.", with: ".final_block.norm.")

      // Catch-all for any remaining .block. patterns
      finalKey = finalKey.replacingOccurrences(of: ".block1.block.", with: ".block1.")
      finalKey = finalKey.replacingOccurrences(of: ".block2.block.", with: ".block2.")
      finalKey = finalKey.replacingOccurrences(of: ".final_block.block.", with: ".final_block.")

      // Removed intermediate cleanup - using FINAL cleanup at end of function instead

      // CausalResnetBlock1D mlp: Python has .mlp.0. (LIST with Linear), Swift has .mlp_linear.
      finalKey = finalKey.replacingOccurrences(of: ".mlp.0.", with: ".mlp_linear.")

      // MidBlock/DownBlock/UpBlock: Python has .transformer_blocks., Swift has .transformers.
      finalKey = finalKey.replacingOccurrences(of: ".transformer_blocks.", with: ".transformers.")

      // FeedForward: Python uses self.net = [GELU(...), Linear(...)]
      // Python weight keys: ff.net.0.proj.weight (GELU's proj), ff.net.1.weight (output Linear)
      // Swift uses: ff.layers.0.weight, ff.layers.1.weight
      finalKey = finalKey.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.layers.0.")
      finalKey = finalKey.replacingOccurrences(of: ".ff.net.1.", with: ".ff.layers.1.")

      // BasicTransformerBlock: Python uses attn1, Swift uses attn
      finalKey = finalKey.replacingOccurrences(of: ".attn1.", with: ".attn.")

      // DiffusersAttention/SelfAttention1D: Python uses to_q/to_k/to_v/to_out.0
      // Swift uses query_proj/key_proj/value_proj/out_proj
      finalKey = finalKey.replacingOccurrences(of: ".to_q.", with: ".query_proj.")
      finalKey = finalKey.replacingOccurrences(of: ".to_k.", with: ".key_proj.")
      finalKey = finalKey.replacingOccurrences(of: ".to_v.", with: ".value_proj.")
      finalKey = finalKey.replacingOccurrences(of: ".to_out.0.", with: ".out_proj.")

      // CausalResnetBlock1D res_conv: Python has .res_conv.conv., Swift has .res_conv. directly
      finalKey = finalKey.replacingOccurrences(of: ".res_conv.conv.", with: ".res_conv.")

      // ConditionalDecoder final_proj: Python has .final_proj.conv., Swift has .final_proj. directly
      finalKey = finalKey.replacingOccurrences(of: ".final_proj.conv.", with: ".final_proj.")

      // CausalConv1d in upsample/downsample: Python has .conv.conv., Swift has .conv.
      // e.g., upsample.conv.conv.weight -> upsample.conv.weight
      finalKey = finalKey.replacingOccurrences(of: ".upsample.conv.conv.", with: ".upsample.conv.")
      finalKey = finalKey.replacingOccurrences(of: ".downsample.conv.conv.", with: ".downsample.conv.")

      // S3Gen flow components: s3gen.decoder -> s3gen.flow.decoder, etc.
      // The Swift model has these under flow submodule
      if finalKey.hasPrefix("s3gen.") && !finalKey.hasPrefix("s3gen.mel2wav.") && !finalKey.hasPrefix("s3gen.speaker_encoder.") && !finalKey.hasPrefix("s3gen.flow.") {
        let s3genComponents = ["decoder.", "encoder.", "encoder_proj.", "input_embedding.", "spk_embed_affine_layer."]
        for component in s3genComponents {
          if finalKey.hasPrefix("s3gen.\(component)") {
            finalKey = finalKey.replacingOccurrences(of: "s3gen.\(component)", with: "s3gen.flow.\(component)")
            break
          }
        }
      }

      // Convert block naming patterns
      let blocksPattern = try! NSRegularExpression(pattern: #"(down_blocks|mid_blocks|up_blocks)_(\d+)"#)
      finalKey = blocksPattern.stringByReplacingMatches(
        in: finalKey, range: NSRange(finalKey.startIndex..., in: finalKey),
        withTemplate: "$1.$2"
      )

      let transformerPattern = try! NSRegularExpression(pattern: #"\.transformer_(\d+)\."#)
      finalKey = transformerPattern.stringByReplacingMatches(
        in: finalKey, range: NSRange(finalKey.startIndex..., in: finalKey),
        withTemplate: ".transformers.$1."
      )

      // CAMPPlus (speaker_encoder) weight name mappings
      func replaceWithZeroIndexed(_ input: String, pattern: String, prefix: String, suffix: String) -> String {
        let regex = try! NSRegularExpression(pattern: pattern)
        guard let match = regex.firstMatch(in: input, range: NSRange(input.startIndex..., in: input)) else {
          return input
        }
        let fullRange = Range(match.range, in: input)!
        let numberRange = Range(match.range(at: 1), in: input)!
        let number = Int(input[numberRange])! - 1
        return input.replacingCharacters(in: fullRange, with: "\(prefix)\(number)\(suffix)")
      }

      finalKey = replaceWithZeroIndexed(finalKey, pattern: #"xvector\.block(\d+)\."#, prefix: "blocks.", suffix: ".")
      finalKey = replaceWithZeroIndexed(finalKey, pattern: #"xvector\.transit(\d+)\."#, prefix: "transits.", suffix: ".")
      finalKey = finalKey.replacingOccurrences(of: "xvector.tdnn.", with: "tdnn.")
      finalKey = finalKey.replacingOccurrences(of: "xvector.dense.", with: "dense.")
      finalKey = finalKey.replacingOccurrences(of: "xvector.out_nonlinear.", with: "out_nonlinear.")
      finalKey = replaceWithZeroIndexed(finalKey, pattern: #"\.tdnnd(\d+)\."#, prefix: ".layers.", suffix: ".")

      let nonlinear1BnPattern = try! NSRegularExpression(pattern: #"\.nonlinear(\d+)\.batchnorm\."#)
      finalKey = nonlinear1BnPattern.stringByReplacingMatches(
        in: finalKey, range: NSRange(finalKey.startIndex..., in: finalKey),
        withTemplate: ".nonlinear$1.0."
      )
      finalKey = finalKey.replacingOccurrences(of: ".nonlinear.batchnorm.", with: ".nonlinear.0.")
      finalKey = finalKey.replacingOccurrences(of: ".out_nonlinear.batchnorm.", with: ".out_nonlinear.0.")
      if finalKey.hasPrefix("out_nonlinear.batchnorm.") {
        finalKey = finalKey.replacingOccurrences(of: "out_nonlinear.batchnorm.", with: "out_nonlinear.0.")
      }

      // Conv1d weight transposition for CAMPPlus
      var finalValue = value
      if finalKey.contains("speaker_encoder"), finalKey.hasSuffix(".weight"), value.ndim == 3 {
        if value.shape[1] > value.shape[2] {
          finalValue = value.swappedAxes(1, 2)
        }
      }

      // FINAL cleanup: CausalBlock1D LayerNorm - must be absolute last step
      // Handle any bare weight/bias that should go to norm submodule
      // Do unconditionally - replacingOccurrences is a no-op if pattern not found
      finalKey = finalKey.replacingOccurrences(of: ".block1.weight", with: ".block1.norm.weight")
      finalKey = finalKey.replacingOccurrences(of: ".block2.weight", with: ".block2.norm.weight")
      finalKey = finalKey.replacingOccurrences(of: ".final_block.weight", with: ".final_block.norm.weight")
      finalKey = finalKey.replacingOccurrences(of: ".block1.bias", with: ".block1.norm.bias")
      finalKey = finalKey.replacingOccurrences(of: ".block2.bias", with: ".block2.norm.bias")
      finalKey = finalKey.replacingOccurrences(of: ".final_block.bias", with: ".final_block.norm.bias")

      sanitized[finalKey] = finalValue
    }
    return sanitized
  }

  /// Load Chatterbox Turbo TTS model from Hugging Face Hub
  static func load(
    quantization: ChatterboxTurboQuantization = .q4,
    s3TokenizerRepoId: String = s3TokenizerRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> ChatterboxTurboModel {
    let repoId = repoId(quantization: quantization)

    Log.model.info("Loading ChatterboxTurbo (\(quantization.rawValue)) from \(repoId)...")

    // Model files include added_tokens.json for emotion control
    let modelFiles = ["model.safetensors", "config.json", "conds.safetensors"] + additionalTokenizerFiles
    async let modelDirectoryTask = HubConfiguration.shared.snapshot(
      from: repoId,
      matching: modelFiles,
      progressHandler: progressHandler
    )

    async let s3TokenizerDirectoryTask = HubConfiguration.shared.snapshot(
      from: s3TokenizerRepoId,
      matching: ["model.safetensors", "config.json"],
      progressHandler: progressHandler
    )

    // Download GPT-2 tokenizer files (tokenizer.json has proper format)
    async let gpt2TokenizerDirectoryTask = HubConfiguration.shared.snapshot(
      from: gpt2TokenizerRepoId,
      matching: ["tokenizer.json", "tokenizer_config.json"],
      progressHandler: progressHandler
    )

    let (modelDirectory, s3TokenizerDirectory, gpt2TokenizerDirectory) = try await (
      modelDirectoryTask, s3TokenizerDirectoryTask, gpt2TokenizerDirectoryTask
    )

    // Load Chatterbox Turbo weights and sanitize
    let weightFileURL = modelDirectory.appending(path: "model.safetensors")
    let rawWeights = try MLX.loadArrays(url: weightFileURL)
    let weights = sanitizeWeights(rawWeights)

    // Initialize model
    let model = ChatterboxTurboModel()

    // Check if model is quantized
    let isQuantized = weights.keys.contains { $0.contains(".scales") }
    if isQuantized {
      Log.model.info("Detected quantized ChatterboxTurbo model weights")
      quantize(model: model) { path, _ in
        if weights["\(path).scales"] != nil {
          return (64, 4, .affine)
        }
        return nil
      }
    }

    // Load weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // Load S3Tokenizer weights separately
    let s3TokenizerWeightURL = s3TokenizerDirectory.appending(path: "model.safetensors")
    let s3TokenizerRawWeights = try MLX.loadArrays(url: s3TokenizerWeightURL)

    var s3TokenizerWeights: [String: MLXArray] = [:]
    for (key, value) in s3TokenizerRawWeights {
      s3TokenizerWeights["s3_tokenizer.\(key)"] = value
    }

    let s3TokenizerParameters = ModuleParameters.unflattened(s3TokenizerWeights)
    try model.update(parameters: s3TokenizerParameters, verify: [.noUnusedKeys])

    // Set to eval mode
    model.train(false)
    eval(model)

    // Load tokenizer (merges GPT-2 base with emotion control tokens from model)
    try await model.loadTextTokenizer(gpt2Directory: gpt2TokenizerDirectory, modelDirectory: modelDirectory)

    // Load pre-computed conditionals if available
    let condsURL = modelDirectory.appending(path: "conds.safetensors")
    if FileManager.default.fileExists(atPath: condsURL.path) {
      let condsData = try MLX.loadArrays(url: condsURL)

      let speakerEmb = condsData["t3.speaker_emb"] ?? MLXArray.zeros([1, 256])
      let condTokens = condsData["t3.cond_prompt_speech_tokens"]

      let t3Cond = T3TurboCond(
        speakerEmb: speakerEmb,
        condPromptSpeechTokens: condTokens
      )

      var genDict: [String: MLXArray] = [:]
      for (key, value) in condsData {
        if key.hasPrefix("gen.") {
          genDict[String(key.dropFirst(4))] = value
        }
      }

      // Convert gen dict to S3GenRefDict
      let s3genRefDict = S3GenRefDict(
        promptToken: genDict["prompt_token"] ?? MLXArray.zeros([1, 0]),
        promptTokenLen: genDict["prompt_token_len"] ?? MLXArray([Int32(0)]),
        promptFeat: genDict["prompt_feat"] ?? MLXArray.zeros([1, 0, 80]),
        promptFeatLen: genDict["prompt_feat_len"] ?? MLXArray([Int32(0)]),
        embedding: genDict["embedding"] ?? MLXArray.zeros([1, 192])
      )

      model.conds = ChatterboxTurboConditionals(t3: t3Cond, gen: s3genRefDict)
      Log.model.info("Loaded pre-computed conditionals")
    }

    Log.model.info("ChatterboxTurbo TTS model (\(quantization.rawValue)) loaded successfully")

    return model
  }

  /// Prepare conditioning from a reference audio clip
  func prepareConditionals(
    refWav: MLXArray,
    refSr: Int
  ) -> ChatterboxTurboConditionals {
    var wav = refWav
    if wav.ndim == 2 {
      wav = wav.squeezed(axis: 0)
    }

    // Resample to 24kHz for S3Gen
    var refWav24k = wav
    if refSr != ChatterboxTurboConstants.s3genSr {
      refWav24k = AudioResampler.resample(wav, from: refSr, to: ChatterboxTurboConstants.s3genSr)
    }

    // Truncate to decoder conditioning length (10s)
    if refWav24k.shape[0] > ChatterboxTurboConstants.decCondLen {
      refWav24k = refWav24k[0 ..< ChatterboxTurboConstants.decCondLen]
    }

    // Resample 24kHz to 16kHz for S3Gen tokenization
    let refWav16kFrom24k = AudioResampler.resample(refWav24k, from: ChatterboxTurboConstants.s3genSr, to: ChatterboxTurboConstants.s3Sr)

    // Resample original to 16kHz for T3 encoder conditioning
    var refWav16kFull = wav
    if refSr != ChatterboxTurboConstants.s3Sr {
      refWav16kFull = AudioResampler.resample(wav, from: refSr, to: ChatterboxTurboConstants.s3Sr)
    }

    // Truncate to encoder conditioning length (15s for Turbo)
    var refWav16k = refWav16kFull
    if refWav16k.shape[0] > ChatterboxTurboConstants.encCondLen {
      refWav16k = refWav16k[0 ..< ChatterboxTurboConstants.encCondLen]
    }

    // Get S3Gen reference embeddings
    var s3genRefDict = S3GenRefDict(
      promptToken: MLXArray.zeros([1, 0]),
      promptTokenLen: MLXArray([Int32(0)]),
      promptFeat: MLXArray.zeros([1, 0, 80]),
      promptFeatLen: MLXArray([Int32(0)]),
      embedding: MLXArray.zeros([1, 192])
    )

    // S3Gen tokens (from 10s audio, resampled 24k->16k)
    let s3genMel = logMelSpectrogramChatterbox(audio: refWav16kFrom24k)
    let s3genMelBatch = s3genMel.expandedDimensions(axis: 0)
    let s3genMelLen = MLXArray([Int32(s3genMel.shape[1])])

    let (s3genTokens, _) = s3Tokenizer.quantize(mel: s3genMelBatch, melLen: s3genMelLen)

    // Get S3Gen embeddings
    s3genRefDict = s3gen.embedRef(
      refWav: refWav24k.expandedDimensions(axis: 0),
      refSr: ChatterboxTurboConstants.s3genSr,
      refSpeechTokens: s3genTokens,
      refSpeechTokenLens: MLXArray([Int32(s3genTokens.shape[1])])
    )

    // T3 conditioning tokens (from 15s audio)
    let t3Mel = logMelSpectrogramChatterbox(audio: refWav16k)
    let t3MelBatch = t3Mel.expandedDimensions(axis: 0)
    let t3MelLen = MLXArray([Int32(t3Mel.shape[1])])

    let (t3Tokens, _) = s3Tokenizer.quantize(mel: t3MelBatch, melLen: t3MelLen)

    // Limit T3 tokens to prompt length
    let plen = config.t3Config.speechCondPromptLen
    let t3CondPromptTokens = t3Tokens[0..., 0 ..< min(plen, t3Tokens.shape[1])]

    // Voice encoder speaker embedding
    let veEmbed = ve.embedsFromWavs(wavs: [refWav16kFull])
    let veEmbedMean = veEmbed.mean(axis: 0, keepDims: true)

    let t3Cond = T3TurboCond(
      speakerEmb: veEmbedMean,
      condPromptSpeechTokens: t3CondPromptTokens
    )

    return ChatterboxTurboConditionals(t3: t3Cond, gen: s3genRefDict)
  }

  /// Generate speech from text
  ///
  /// - Parameters:
  ///   - text: Input text to synthesize
  ///   - audioPrompt: Reference audio for voice matching
  ///   - audioPromptSr: Sample rate of audio prompt
  ///   - conds: Pre-computed conditionals (optional)
  ///   - temperature: Sampling temperature (default 0.8)
  ///   - topK: Top-k sampling parameter (default 1000)
  ///   - topP: Top-p (nucleus) sampling threshold (default 0.95)
  ///   - repetitionPenalty: Penalty for repeated tokens (default 1.2)
  ///   - maxNewTokens: Maximum number of tokens to generate (default 800)
  /// - Returns: Generated audio waveform
  func generate(
    text: String,
    audioPrompt: MLXArray? = nil,
    audioPromptSr: Int? = nil,
    conds: ChatterboxTurboConditionals? = nil,
    temperature: Float = 0.8,
    topK: Int = 1000,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.2,
    maxNewTokens: Int = 800
  ) -> MLXArray {
    // Prepare conditionals if needed
    var conditionals = conds
    if conditionals == nil {
      if let prompt = audioPrompt, let sr = audioPromptSr {
        conditionals = prepareConditionals(refWav: prompt, refSr: sr)
      } else if let cached = self.conds {
        conditionals = cached
      } else {
        fatalError("Reference audio is required for ChatterboxTurbo TTS")
      }
    }

    guard let cond = conditionals else {
      fatalError("Failed to prepare conditionals")
    }

    // Normalize text
    let normalizedText = puncNorm(text)

    // Tokenize text using GPT-2 tokenizer
    guard let tokenizer = textTokenizer else {
      fatalError("Text tokenizer not loaded")
    }

    let tokenIds = tokenizer.encode(text: normalizedText)
    let textTokens = MLXArray(tokenIds.map { Int32($0) }).expandedDimensions(axis: 0)

    // Generate speech tokens with T3 Turbo (no CFG needed)
    let speechTokens = t3.inferenceTurbo(
      t3Cond: cond.t3,
      textTokens: textTokens,
      temperature: temperature,
      topK: topK,
      topP: topP,
      repetitionPenalty: repetitionPenalty,
      maxGenLen: maxNewTokens
    )

    // Check for truncation
    let generatedCount = speechTokens.shape[1]
    if generatedCount >= maxNewTokens {
      let lastToken: Int32 = speechTokens[0, generatedCount - 1].item()
      if lastToken != Int32(config.t3Config.stopSpeechToken) {
        Log.tts.warning("ChatterboxTurbo: Generation hit token limit (\(maxNewTokens)), audio may be truncated.")
      }
    }

    // Extract tokens (batch index 0)
    var filteredTokens = speechTokens[0 ..< 1]

    // Drop invalid tokens
    filteredTokens = dropInvalidTokens(filteredTokens)

    // Filter out tokens >= SPEECH_VOCAB_SIZE
    let mask = filteredTokens .< ChatterboxSpeechVocabSize
    let maskValues = mask.asArray(Bool.self)
    let validIndices = maskValues.enumerated().compactMap { $0.element ? Int32($0.offset) : nil }
    if !validIndices.isEmpty {
      filteredTokens = filteredTokens[MLXArray(validIndices)]
    } else {
      filteredTokens = MLXArray([Int32]())
    }

    // Add silence tokens
    let silenceTokens = MLXArray([Int32(ChatterboxTurboConstants.silenceToken), Int32(ChatterboxTurboConstants.silenceToken), Int32(ChatterboxTurboConstants.silenceToken)])
    filteredTokens = MLX.concatenated([filteredTokens, silenceTokens])

    // Reshape for S3Gen
    filteredTokens = filteredTokens.expandedDimensions(axis: 0)

    // Generate waveform with S3Gen using meanflow (2 CFM steps)
    var wav = s3gen(
      speechTokens: filteredTokens,
      refDict: cond.gen,
      finalize: true,
      nTimesteps: ChatterboxTurboConstants.cfmSteps, // 2 steps for Turbo
      meanflow: true // Use meanflow mode for Turbo
    )

    // Flatten to 1D if needed
    if wav.ndim == 2 {
      wav = wav.squeezed(axis: 0)
    }

    return wav
  }
}
