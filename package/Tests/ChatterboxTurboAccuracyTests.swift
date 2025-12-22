// Accuracy tests comparing Swift Chatterbox Turbo against Python checkpoints

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import XCTest

@testable import MLXAudio

/// Accuracy tests for Chatterbox Turbo port validation
///
/// These tests compare Swift implementation outputs against Python-generated
/// checkpoints to verify the port is numerically accurate.
///
/// To generate checkpoints:
///   python scripts/generate_checkpoints.py --ref_audio ~/Downloads/kush.wav --text "Hello world"
///
/// Then run these tests with:
///   xcodebuild test-without-building -scheme mlx-audio-Package -destination "id=<MAC_ID>,arch=arm64" \
///     -only-testing:MLXAudioTests/ChatterboxTurboAccuracyTests
@MainActor
final class ChatterboxTurboAccuracyTests: XCTestCase {
  var manifest: CheckpointReader.Manifest!
  var checkpointPath: URL!

  override func setUp() async throws {
    try await super.setUp()

    // Try to find checkpoints directory
    let fm = FileManager.default

    // Try a few possible locations
    let possiblePaths = [
      URL(fileURLWithPath: "/Users/sachin/Tools/MLX/mlx-swift-audio-cbt/checkpoints"),
      URL(fileURLWithPath: "./checkpoints"),
    ]

    for path in possiblePaths {
      let manifestPath = path.appendingPathComponent("manifest.json")
      if fm.fileExists(atPath: manifestPath.path) {
        checkpointPath = path
        break
      }
    }

    guard let checkpointPath else {
      throw XCTSkip(
        "Checkpoints not found. Generate with: python scripts/generate_checkpoints.py --ref_audio <audio> --text <text>"
      )
    }

    print("[Test] Loading checkpoints from \(checkpointPath.path)")
    manifest = try CheckpointReader.loadManifest(from: checkpointPath)
    print("[Test] Reference audio: \(manifest.refAudio)")
    print("[Test] Text: \(manifest.text)")
    print("[Test] Model: \(manifest.modelId)")
    print("[Test] Stages: \(manifest.stages.count)")
  }

  /// Load audio file and return samples with sample rate
  func loadAudioFile(at url: URL) throws -> (samples: [Float], sampleRate: Int) {
    let file = try AVAudioFile(forReading: url)

    guard let buffer = AVAudioPCMBuffer(
      pcmFormat: file.processingFormat,
      frameCapacity: AVAudioFrameCount(file.length)
    ) else {
      throw NSError(domain: "Test", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer"])
    }

    try file.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw NSError(domain: "Test", code: 2, userInfo: [NSLocalizedDescriptionKey: "No float data"])
    }

    let frameCount = Int(buffer.frameLength)
    let channelCount = Int(buffer.format.channelCount)

    // Convert to mono if stereo
    var samples = [Float](repeating: 0, count: frameCount)
    if channelCount == 1 {
      for i in 0 ..< frameCount {
        samples[i] = floatData[0][i]
      }
    } else {
      for i in 0 ..< frameCount {
        var sum: Float = 0
        for ch in 0 ..< channelCount {
          sum += floatData[ch][i]
        }
        samples[i] = sum / Float(channelCount)
      }
    }

    return (samples, Int(file.fileFormat.sampleRate))
  }

  // MARK: - Stage 0-3: Audio Preprocessing

  func testAudioResampling() async throws {
    print("\n=== Testing Audio Resampling ===")

    // Load original audio
    let audioURL = URL(fileURLWithPath: manifest.refAudio)
    let (audio, sampleRate) = try loadAudioFile(at: audioURL)
    print("[Test] Original audio: \(audio.count) samples @ \(sampleRate) Hz")

    // Load Python checkpoints
    let pyAudio24k = try CheckpointReader.load(from: checkpointPath, name: "s1_audio_24k", manifest: manifest)
    let pyAudio16k = try CheckpointReader.load(from: checkpointPath, name: "s2_audio_16k", manifest: manifest)

    // Resample in Swift
    let audioMlx = MLXArray(audio)
    let swiftAudio24k = AudioResampler.resample(audioMlx, from: sampleRate, to: 24000)
    let swiftAudio16k = AudioResampler.resample(audioMlx, from: sampleRate, to: 16000)

    // Compare 24k
    print("\n--- Comparing 24kHz resampled audio ---")
    print(TensorComparison.summarize(pyAudio24k, name: "Python 24k"))
    print(TensorComparison.summarize(swiftAudio24k, name: "Swift 24k"))

    let result24k = TensorComparison.compare(
      swiftAudio24k, pyAudio24k,
      absoluteTolerance: 1e-3,
      relativeTolerance: 1e-2,
      name: "Audio 24kHz"
    )
    print(result24k.message)

    // Compare 16k
    print("\n--- Comparing 16kHz resampled audio ---")
    print(TensorComparison.summarize(pyAudio16k, name: "Python 16k"))
    print(TensorComparison.summarize(swiftAudio16k, name: "Swift 16k"))

    let result16k = TensorComparison.compare(
      swiftAudio16k, pyAudio16k,
      absoluteTolerance: 1e-3,
      relativeTolerance: 1e-2,
      name: "Audio 16kHz"
    )
    print(result16k.message)

    // These should be close but not identical due to different resampling algorithms
    XCTAssert(result24k.correlation > 0.99, "24kHz correlation too low: \(result24k.correlation)")
    XCTAssert(result16k.correlation > 0.99, "16kHz correlation too low: \(result16k.correlation)")
  }

  // MARK: - Stage 4: Mel Spectrogram

  func testMelSpectrogram() async throws {
    print("\n=== Testing Mel Spectrogram ===")

    // Load Python checkpoint for 16k audio (used for mel)
    let pyAudio = try CheckpointReader.load(from: checkpointPath, name: "s4_audio_16k_from_24k", manifest: manifest)
    let pyMel = try CheckpointReader.load(from: checkpointPath, name: "s4_mel_s3tokenizer", manifest: manifest)

    // Compute Swift mel spectrogram
    // Note: We use Python's audio to isolate mel computation from resampling differences
    let swiftMel = logMelSpectrogramChatterbox(audio: pyAudio, nMels: 128)

    print("\n--- Comparing Mel Spectrograms ---")
    print(TensorComparison.summarize(pyMel, name: "Python mel"))
    print(TensorComparison.summarize(swiftMel, name: "Swift mel"))

    let result = TensorComparison.compare(
      swiftMel, pyMel,
      absoluteTolerance: 1e-3,
      relativeTolerance: 1e-2,
      name: "Mel Spectrogram"
    )
    print(result.message)

    XCTAssert(result.passed || result.correlation > 0.99,
      "Mel spectrogram diverges: \(result.message)")
  }

  // MARK: - Quick Diagnostic Test

  func testQuickDiagnostic() async throws {
    print("\n=== Quick Diagnostic Test ===")
    print("This test identifies the first stage where significant divergence occurs.\n")

    var divergences: [(stage: String, correlation: Float, maxDiff: Float)] = []

    // Stage 1: Audio 24k
    do {
      let pyAudio = try CheckpointReader.load(from: checkpointPath, name: "s1_audio_24k", manifest: manifest)
      let audioURL = URL(fileURLWithPath: manifest.refAudio)
      let (audio, sr) = try loadAudioFile(at: audioURL)
      let swiftAudio = AudioResampler.resample(MLXArray(audio), from: sr, to: 24000)

      let result = TensorComparison.compare(swiftAudio, pyAudio, absoluteTolerance: 1e-3, name: "Stage 1: Audio 24k")
      print("Stage 1 (Audio 24k): correlation=\(String(format: "%.4f", result.correlation)), maxDiff=\(String(format: "%.2e", result.maxAbsDiff))")
      divergences.append(("Stage 1: Audio 24k", result.correlation, result.maxAbsDiff))
    } catch {
      print("Stage 1: Skipped - \(error)")
    }

    // Stage 2: Audio 16k
    do {
      let pyAudio = try CheckpointReader.load(from: checkpointPath, name: "s2_audio_16k", manifest: manifest)
      let audioURL = URL(fileURLWithPath: manifest.refAudio)
      let (audio, sr) = try loadAudioFile(at: audioURL)
      let swiftAudio = AudioResampler.resample(MLXArray(audio), from: sr, to: 16000)

      let result = TensorComparison.compare(swiftAudio, pyAudio, absoluteTolerance: 1e-3, name: "Stage 2: Audio 16k")
      print("Stage 2 (Audio 16k): correlation=\(String(format: "%.4f", result.correlation)), maxDiff=\(String(format: "%.2e", result.maxAbsDiff))")
      divergences.append(("Stage 2: Audio 16k", result.correlation, result.maxAbsDiff))
    } catch {
      print("Stage 2: Skipped - \(error)")
    }

    // Stage 4: Mel spectrogram (using Python's audio input to isolate mel computation)
    do {
      let pyMel = try CheckpointReader.load(from: checkpointPath, name: "s4_mel_s3tokenizer", manifest: manifest)
      let pyAudio = try CheckpointReader.load(from: checkpointPath, name: "s4_audio_16k_from_24k", manifest: manifest)
      let swiftMel = logMelSpectrogramChatterbox(audio: pyAudio, nMels: 128)

      let result = TensorComparison.compare(swiftMel, pyMel, absoluteTolerance: 1e-3, name: "Stage 4: Mel")
      print("Stage 4 (Mel Spectrogram): correlation=\(String(format: "%.4f", result.correlation)), maxDiff=\(String(format: "%.2e", result.maxAbsDiff))")
      divergences.append(("Stage 4: Mel Spectrogram", result.correlation, result.maxAbsDiff))
    } catch {
      print("Stage 4: Skipped - \(error)")
    }

    // Stage 5: S3Tokenizer tokens (info only - requires model)
    do {
      let pyTokens = try CheckpointReader.load(from: checkpointPath, name: "s5_s3tok_tokens", manifest: manifest)
      let first10 = pyTokens.flattened()[0 ..< min(10, pyTokens.shape.reduce(1, *))]
      print("Stage 5 (S3Tokenizer): Python tokens shape=\(pyTokens.shape), first 10=\(first10.asArray(Int32.self))")
      print("  → Requires model loading for full comparison")
    } catch {
      print("Stage 5: Skipped - \(error)")
    }

    // Stage 6: VoiceEncoder embedding (info only - requires model)
    do {
      let pyEmbed = try CheckpointReader.load(from: checkpointPath, name: "s6_speaker_embedding", manifest: manifest)
      let norm = sqrt(MLX.sum(pyEmbed * pyEmbed).item(Float.self))
      print("Stage 6 (VoiceEncoder): Python embedding shape=\(pyEmbed.shape), L2 norm=\(String(format: "%.4f", norm))")
      print("  → Requires model loading for full comparison")
    } catch {
      print("Stage 6: Skipped - \(error)")
    }

    // Stage 10: Generated tokens (info only)
    do {
      let pyTokens = try CheckpointReader.load(from: checkpointPath, name: "s10_generated_tokens", manifest: manifest)
      let first10 = pyTokens.flattened()[0 ..< min(10, pyTokens.shape.reduce(1, *))]
      print("Stage 10 (T3 Tokens): Python tokens shape=\(pyTokens.shape), first 10=\(first10.asArray(Int32.self))")
    } catch {
      print("Stage 10: Skipped - \(error)")
    }

    // Stage 11: Raw waveform (info only)
    do {
      let pyWav = try CheckpointReader.load(from: checkpointPath, name: "s11_waveform_raw", manifest: manifest)
      print("Stage 11 (Waveform): Python shape=\(pyWav.shape)")
      print(TensorComparison.summarize(pyWav, name: "Python waveform"))
    } catch {
      print("Stage 11: Skipped - \(error)")
    }

    // Summary
    print("\n" + String(repeating: "=", count: 60))
    print("DIVERGENCE SUMMARY")
    print(String(repeating: "=", count: 60))

    let significantDivergences = divergences.filter { $0.correlation < 0.99 }
    if significantDivergences.isEmpty {
      print("No significant divergence detected in tested stages.")
      print("Audio resampling and mel spectrogram computation match Python closely.")
    } else {
      print("Significant divergences found:")
      for div in significantDivergences {
        print("  - \(div.stage): correlation=\(String(format: "%.4f", div.correlation))")
      }
    }

    print("\nStages requiring model loading for comparison:")
    print("  - Stage 5: S3Tokenizer (speech tokenization)")
    print("  - Stage 6: VoiceEncoder (speaker embedding)")
    print("  - Stage 9: T3 Conditioning")
    print("  - Stage 10-11: Token generation and waveform synthesis")
    print(String(repeating: "=", count: 60))
  }

  // MARK: - Model-Based Comparison Tests

  /// Load shared model for testing
  func loadModel() async throws -> ChatterboxTurboModel {
    return try await ChatterboxTurboTestHelper.getOrLoadModel()
  }

  // MARK: - Stage 5: S3Tokenizer

  func testS3Tokenizer() async throws {
    print("\n=== Testing S3Tokenizer ===")

    // Load Python checkpoints
    let pyMelBatch = try CheckpointReader.load(from: checkpointPath, name: "s5_mel_batch", manifest: manifest)
    let pyTokens = try CheckpointReader.load(from: checkpointPath, name: "s5_s3tok_tokens", manifest: manifest)

    // Load model
    let model = try await loadModel()

    // Diagnostic: Print encoder weight values to verify they match Python
    let conv1Weight = model.s3Tokenizer.encoder.conv1.weight
    print("Swift encoder.conv1.weight shape: \(conv1Weight.shape)")
    let flatConv1 = conv1Weight.flattened().asArray(Float.self)
    print("Swift first 10 conv1 values: \(flatConv1.prefix(10).map { String(format: "%.6f", $0) })")

    // Expected from Python:
    // [0.002215, 0.010825, 0.010185, 0.001643, 0.010478, -0.012730, 0.001968, -0.007826, 0.000427, 0.000577]
    let expectedConv1 = [0.002215, 0.010825, 0.010185, 0.001643, 0.010478]
    let actualConv1 = Array(flatConv1.prefix(5))
    let conv1Match = zip(expectedConv1, actualConv1).allSatisfy { abs($0 - Double($1)) < 0.0001 }
    print("Conv1 weights match Python: \(conv1Match)")

    print("Python mel shape: \(pyMelBatch.shape)")
    print("Python tokens shape: \(pyTokens.shape)")

    // Both Python and Swift expect (batch, n_mels, T) format
    // Python mel is (1, 128, T), Swift expects same format
    let melLen = MLXArray([Int32(pyMelBatch.shape[2])])  // T dimension

    print("Mel for Swift shape: \(pyMelBatch.shape)")

    // DEBUG: Trace the encoder forward pass
    // Step 1: Transpose mel from (B, n_mels, T) to (B, T, n_mels)
    let melTransposed = pyMelBatch.transposed(0, 2, 1)
    print("After transpose: \(melTransposed.shape)")  // Should be (1, 1000, 128)

    // Step 2: Apply conv1
    let conv1Out = model.s3Tokenizer.encoder.conv1(melTransposed)
    let conv1OutGelu = gelu(conv1Out)
    print("After conv1+gelu shape: \(conv1OutGelu.shape)")
    let conv1OutFlat = conv1OutGelu.flattened().asArray(Float.self)
    print("Conv1 output first 10: \(conv1OutFlat.prefix(10).map { String(format: "%.6f", $0) })")

    // Run S3Tokenizer
    let (swiftTokens, _) = model.s3Tokenizer.quantize(mel: pyMelBatch, melLen: melLen)

    print("\n--- Comparing S3Tokenizer Outputs ---")
    print("Python tokens shape: \(pyTokens.shape), first 10: \(pyTokens.flattened()[0..<min(10, pyTokens.shape[1])].asArray(Int32.self))")
    print("Swift tokens shape: \(swiftTokens.shape), first 10: \(swiftTokens.flattened()[0..<min(10, swiftTokens.shape[1])].asArray(Int32.self))")

    // Compare tokens
    let result = TensorComparison.compareExact(swiftTokens, pyTokens, name: "S3Tokenizer tokens")
    print(result.message)

    // Calculate match rate
    let pyFlat = pyTokens.flattened().asArray(Int32.self)
    let swiftFlat = swiftTokens.flattened().asArray(Int32.self)
    let minLen = min(pyFlat.count, swiftFlat.count)
    let matches = zip(pyFlat.prefix(minLen), swiftFlat.prefix(minLen)).filter { $0 == $1 }.count
    let matchRate = Float(matches) / Float(minLen)
    print("Token match rate: \(String(format: "%.1f", matchRate * 100))% (\(matches)/\(minLen))")

    // For discrete tokens, we want high match rate
    XCTAssert(matchRate > 0.90, "Token match rate too low: \(matchRate)")
  }

  // MARK: - Stage 6: VoiceEncoder

  func testVoiceEncoder() async throws {
    print("\n=== Testing VoiceEncoder ===")

    // Load Python checkpoints - use trimmed audio
    let pyTrimmedAudio = try CheckpointReader.load(from: checkpointPath, name: "s6_audio_trimmed", manifest: manifest)
    let pyMel = try CheckpointReader.load(from: checkpointPath, name: "s6_mel_spectrogram", manifest: manifest)
    let pyEmbed = try CheckpointReader.load(from: checkpointPath, name: "s6_speaker_embedding", manifest: manifest)

    print("Python trimmed audio shape: \(pyTrimmedAudio.shape)")
    print("Python mel shape: \(pyMel.shape)")
    print("Python embedding shape: \(pyEmbed.shape)")
    print("Python embedding first 10: \(pyEmbed.flattened()[0..<10].asArray(Float.self).map { String(format: "%.6f", $0) })")

    // Load model
    let model = try await loadModel()

    // Step 1: Compare mel spectrograms (using trimmed audio)
    print("\n--- Step 1: Comparing Mel Spectrograms ---")
    let swiftMel = voiceEncoderMelspectrogram(wav: pyTrimmedAudio, config: model.ve.config)
    print("Swift mel shape: \(swiftMel.shape)")
    print(TensorComparison.summarize(pyMel, name: "Python mel"))
    print(TensorComparison.summarize(swiftMel, name: "Swift mel"))

    let melResult = TensorComparison.compare(
      swiftMel, pyMel,
      absoluteTolerance: 1e-2,
      relativeTolerance: 1e-1,
      name: "VE mel spectrogram"
    )
    print(melResult.message)

    // Step 2: Compare LSTM weights
    print("\n--- Step 2: Comparing LSTM Weights ---")
    let pyLstm1Wx = try CheckpointReader.load(from: checkpointPath, name: "s6_lstm1_Wx", manifest: manifest)
    let pyLstm1Wh = try CheckpointReader.load(from: checkpointPath, name: "s6_lstm1_Wh", manifest: manifest)
    let pyLstm1Bias = try CheckpointReader.load(from: checkpointPath, name: "s6_lstm1_bias", manifest: manifest)

    // Access Swift LSTM weights (lowercase properties)
    let swiftLstm1 = model.ve.lstm.layers[0]
    let swiftLstm1Wx = swiftLstm1.wx
    let swiftLstm1Wh = swiftLstm1.wh
    let swiftLstm1Bias = swiftLstm1.bias ?? MLXArray.zeros([1024])

    print("Python LSTM1 Wx shape: \(pyLstm1Wx.shape)")
    print("Swift LSTM1 Wx shape: \(swiftLstm1Wx.shape)")

    let wxResult = TensorComparison.compare(swiftLstm1Wx, pyLstm1Wx, absoluteTolerance: 1e-5, relativeTolerance: 1e-4, name: "LSTM1 Wx")
    print(wxResult.message)

    let whResult = TensorComparison.compare(swiftLstm1Wh, pyLstm1Wh, absoluteTolerance: 1e-5, relativeTolerance: 1e-4, name: "LSTM1 Wh")
    print(whResult.message)

    let biasResult = TensorComparison.compare(swiftLstm1Bias, pyLstm1Bias, absoluteTolerance: 1e-5, relativeTolerance: 1e-4, name: "LSTM1 bias")
    print(biasResult.message)

    // Step 3: Compare LSTM outputs layer by layer
    print("\n--- Step 3: Comparing LSTM Outputs ---")
    // Load Python intermediate outputs
    let pyLstm1Out = try CheckpointReader.load(from: checkpointPath, name: "s6_lstm1_output", manifest: manifest)
    let pyLstm2Out = try CheckpointReader.load(from: checkpointPath, name: "s6_lstm2_output", manifest: manifest)
    let pyLstm3Out = try CheckpointReader.load(from: checkpointPath, name: "s6_lstm3_output", manifest: manifest)
    let pyFinalHidden = try CheckpointReader.load(from: checkpointPath, name: "s6_final_hidden", manifest: manifest)
    let pyRawEmbeds = try CheckpointReader.load(from: checkpointPath, name: "s6_raw_embeds", manifest: manifest)
    let pyRawEmbedsRelu = try CheckpointReader.load(from: checkpointPath, name: "s6_raw_embeds_relu", manifest: manifest)

    // Run Swift LSTM step by step (using Python mel for fair comparison)
    let melBatch = pyMel.transposed().expandedDimensions(axis: 0)  // (1, T, M)

    // LSTM1
    let (swiftLstm1Out, _) = model.ve.lstm.layers[0](melBatch)
    let lstm1Result = TensorComparison.compare(swiftLstm1Out, pyLstm1Out, absoluteTolerance: 1e-2, relativeTolerance: 1e-1, name: "LSTM1 output")
    print(lstm1Result.message)

    // LSTM2
    let (swiftLstm2Out, _) = model.ve.lstm.layers[1](swiftLstm1Out)
    let lstm2Result = TensorComparison.compare(swiftLstm2Out, pyLstm2Out, absoluteTolerance: 1e-2, relativeTolerance: 1e-1, name: "LSTM2 output")
    print(lstm2Result.message)

    // LSTM3
    let (swiftLstm3Out, _) = model.ve.lstm.layers[2](swiftLstm2Out)
    let lstm3Result = TensorComparison.compare(swiftLstm3Out, pyLstm3Out, absoluteTolerance: 1e-2, relativeTolerance: 1e-1, name: "LSTM3 output")
    print(lstm3Result.message)

    // Final hidden (last timestep)
    let swiftFinalHidden = swiftLstm3Out[0..., swiftLstm3Out.shape[1] - 1, 0...]
    let finalHiddenResult = TensorComparison.compare(swiftFinalHidden, pyFinalHidden, absoluteTolerance: 1e-2, relativeTolerance: 1e-1, name: "Final hidden")
    print(finalHiddenResult.message)

    // Projection
    let swiftRawEmbeds = model.ve.proj(swiftFinalHidden)
    let rawEmbedsResult = TensorComparison.compare(swiftRawEmbeds, pyRawEmbeds, absoluteTolerance: 1e-2, relativeTolerance: 1e-1, name: "Raw embeds")
    print(rawEmbedsResult.message)

    // ReLU
    let swiftRawEmbedsRelu = relu(swiftRawEmbeds)
    let rawEmbedsReluResult = TensorComparison.compare(swiftRawEmbedsRelu, pyRawEmbedsRelu, absoluteTolerance: 1e-2, relativeTolerance: 1e-1, name: "Raw embeds (ReLU)")
    print(rawEmbedsReluResult.message)

    // Step 4: Compare embeddings using Python mel (bypass mel computation)
    print("\n--- Step 4: Comparing Embeddings (using Python mel) ---")
    // Transpose mel from (M, T) to (T, M) for VoiceEncoder
    let pyMelTransposed = pyMel.transposed()
    let swiftEmbedFromPyMel = model.ve.embedsFromMels(
      mels: [pyMelTransposed],
      asSpk: false,
      batchSize: 32,
      rate: 1.3
    )
    print("Swift embedding (from Python mel) first 10: \(swiftEmbedFromPyMel.flattened()[0..<10].asArray(Float.self).map { String(format: "%.6f", $0) })")

    let embedResult = TensorComparison.compare(
      swiftEmbedFromPyMel, pyEmbed,
      absoluteTolerance: 1e-2,
      relativeTolerance: 1e-1,
      name: "VE embedding (from Python mel)"
    )
    print(embedResult.message)

    // Step 5: Full pipeline comparison (Swift mel + Swift VE)
    print("\n--- Step 5: Full Pipeline Comparison ---")
    let swiftEmbed = model.ve.embedsFromWavs(wavs: [pyTrimmedAudio])
    print("Swift embedding (full pipeline) first 10: \(swiftEmbed.flattened()[0..<10].asArray(Float.self).map { String(format: "%.6f", $0) })")

    let fullResult = TensorComparison.compare(
      swiftEmbed, pyEmbed,
      absoluteTolerance: 1e-2,
      relativeTolerance: 1e-1,
      name: "VE embedding (full pipeline)"
    )
    print(fullResult.message)

    // Use LSTM1 output correlation to determine where the issue lies
    XCTAssert(lstm1Result.correlation > 0.95, "LSTM1 output correlation too low: \(lstm1Result.correlation)")
  }

  // MARK: - Stage 7: S3Gen Reference Embedding

  func testS3GenRefEmbedding() async throws {
    print("\n=== Testing S3Gen Reference Embedding ===")

    // Load Python checkpoints
    let pyPromptFeat = try CheckpointReader.load(from: checkpointPath, name: "s7_s3gen_prompt_feat", manifest: manifest)
    let pyEmbedding = try CheckpointReader.load(from: checkpointPath, name: "s7_s3gen_embedding", manifest: manifest)
    let pyPromptToken = try CheckpointReader.load(from: checkpointPath, name: "s7_s3gen_prompt_token", manifest: manifest)

    print("Python prompt_feat shape: \(pyPromptFeat.shape)")
    print("Python embedding shape: \(pyEmbedding.shape)")
    print("Python prompt_token shape: \(pyPromptToken.shape)")

    // Load model and compute Swift version
    let model = try await loadModel()

    // Get 24k truncated audio
    let pyAudio24k = try CheckpointReader.load(from: checkpointPath, name: "s3_audio_24k_trunc", manifest: manifest)

    // Get S3 tokens from Python (to isolate S3Gen from S3Tokenizer differences)
    let s3Tokens = try CheckpointReader.load(from: checkpointPath, name: "s5_s3tok_tokens", manifest: manifest)
    let s3TokenLens = try CheckpointReader.load(from: checkpointPath, name: "s5_s3tok_lens", manifest: manifest)

    // Compute S3Gen ref embedding
    let swiftRefDict = model.s3gen.embedRef(
      refWav: pyAudio24k,
      refSr: 24000,
      refSpeechTokens: s3Tokens,
      refSpeechTokenLens: s3TokenLens
    )

    print("\n--- Comparing S3Gen Reference Embedding ---")
    print("Swift prompt_feat shape: \(swiftRefDict.promptFeat.shape)")
    print("Swift embedding shape: \(swiftRefDict.embedding.shape)")

    // Compare prompt features
    let featResult = TensorComparison.compare(
      swiftRefDict.promptFeat, pyPromptFeat,
      absoluteTolerance: 1e-2,
      relativeTolerance: 1e-1,
      name: "S3Gen prompt_feat"
    )
    print(featResult.message)

    // Compare embeddings
    let embedResult = TensorComparison.compare(
      swiftRefDict.embedding, pyEmbedding,
      absoluteTolerance: 1e-2,
      relativeTolerance: 1e-1,
      name: "S3Gen embedding"
    )
    print(embedResult.message)

    // Note: Python embedding shows all zeros in checkpoint, which might be a bug or expected
    print("\nNote: Python embedding range was [0, 0] in checkpoint - may need investigation")
  }

  // MARK: - Stage 9: T3 Conditioning

  func testT3Conditioning() async throws {
    print("\n=== Testing T3 Conditioning ===")

    // Load Python checkpoints
    let pyCondPrepared = try CheckpointReader.load(from: checkpointPath, name: "s9_t3_cond_prepared", manifest: manifest)
    let pyCondTokens = try CheckpointReader.load(from: checkpointPath, name: "s9_t3_cond_tokens", manifest: manifest)
    let pyVeEmbed = try CheckpointReader.load(from: checkpointPath, name: "s6_speaker_embedding", manifest: manifest)

    print("Python cond_prepared shape: \(pyCondPrepared.shape)")
    print("Python cond_tokens shape: \(pyCondTokens.shape)")
    print("Python speaker_embedding shape: \(pyVeEmbed.shape)")
    print(TensorComparison.summarize(pyVeEmbed, name: "Python speaker_embedding"))

    // Load model
    let model = try await loadModel()

    // Debug: Check speech_emb weights
    let speechEmbWeight = model.t3.speechEmb.weight
    print("\nSwift speech_emb weight shape: \(speechEmbWeight.shape)")
    print("Swift speech_emb weight first 10: \((0..<10).map { String(format: "%.4f", speechEmbWeight[0, $0].item(Float.self)) })")

    // Debug: Check spkr_enc weights
    let spkrEncWeight = model.t3.condEnc.spkrEnc.weight
    print("Swift spkr_enc weight shape: \(spkrEncWeight.shape)")
    print("Swift spkr_enc weight first 10: \((0..<10).map { String(format: "%.4f", spkrEncWeight[0, $0].item(Float.self)) })")

    // Test speech embedding directly
    let testTokens = pyCondTokens[0..., 0..<5]
    let swiftSpeechEmb = model.t3.speechEmb(testTokens)
    print("\nTest speech_emb on first 5 tokens:")
    print("  Tokens: \((0..<5).map { Int(testTokens[0, $0].item(Int32.self)) })")
    print("  Output shape: \(swiftSpeechEmb.shape)")
    print("  Output first 10 values: \((0..<10).map { String(format: "%.4f", swiftSpeechEmb[0, 0, $0].item(Float.self)) })")

    // Test speaker encoding directly
    let swiftSpkrEncOut = model.t3.condEnc.spkrEnc(pyVeEmbed.reshaped(-1, 256)).expandedDimensions(axis: 1)
    print("\nTest spkr_enc output:")
    print("  Output shape: \(swiftSpkrEncOut.shape)")
    print("  Output first 10: \((0..<10).map { String(format: "%.4f", swiftSpkrEncOut[0, 0, $0].item(Float.self)) })")

    // Compare with Python's first token (which should be speaker embedding)
    let pySpkrEmb = pyCondPrepared[0..., 0..<1, 0...]
    print("\nPython first cond token (speaker emb):")
    print("  Shape: \(pySpkrEmb.shape)")
    print("  First 10: \((0..<10).map { String(format: "%.4f", pyCondPrepared[0, 0, $0].item(Float.self)) })")

    // Create T3 conditioning from Python inputs
    var t3Cond = T3TurboCond(
      speakerEmb: pyVeEmbed,
      condPromptSpeechTokens: pyCondTokens
    )

    // Prepare conditioning
    let swiftCondPrepared = model.t3.prepareConditioning(&t3Cond)

    print("\n--- Comparing T3 Conditioning ---")
    print(TensorComparison.summarize(pyCondPrepared, name: "Python cond_prepared"))
    print(TensorComparison.summarize(swiftCondPrepared, name: "Swift cond_prepared"))

    let result = TensorComparison.compare(
      swiftCondPrepared, pyCondPrepared,
      absoluteTolerance: 1e-2,
      relativeTolerance: 1e-1,
      name: "T3 cond_prepared"
    )
    print(result.message)

    XCTAssert(result.correlation > 0.95, "T3 conditioning correlation too low: \(result.correlation)")
  }

  // MARK: - Full Model Comparison

  func testFullModelComparison() async throws {
    print("\n=== Full Model Comparison ===")
    print("Running all model stages and comparing against Python checkpoints\n")

    var results: [(stage: String, correlation: Float, passed: Bool)] = []

    // Load model once
    let model = try await loadModel()

    // Stage 5: S3Tokenizer
    do {
      let pyMelBatch = try CheckpointReader.load(from: checkpointPath, name: "s5_mel_batch", manifest: manifest)
      let pyTokens = try CheckpointReader.load(from: checkpointPath, name: "s5_s3tok_tokens", manifest: manifest)

      // Both Python and Swift expect (batch, n_mels, T) format
      let melLen = MLXArray([Int32(pyMelBatch.shape[2])])
      let (swiftTokens, _) = model.s3Tokenizer.quantize(mel: pyMelBatch, melLen: melLen)

      let pyFlat = pyTokens.flattened().asArray(Int32.self)
      let swiftFlat = swiftTokens.flattened().asArray(Int32.self)
      let minLen = min(pyFlat.count, swiftFlat.count)
      let matches = zip(pyFlat.prefix(minLen), swiftFlat.prefix(minLen)).filter { $0 == $1 }.count
      let matchRate = Float(matches) / Float(minLen)

      print("Stage 5 (S3Tokenizer): match rate=\(String(format: "%.1f%%", matchRate * 100))")
      results.append(("S3Tokenizer", matchRate, matchRate > 0.90))
    } catch {
      print("Stage 5: Error - \(error)")
    }

    // Stage 6: VoiceEncoder
    do {
      let pyAudio = try CheckpointReader.load(from: checkpointPath, name: "s3_audio_16k_trunc", manifest: manifest)
      let pyEmbed = try CheckpointReader.load(from: checkpointPath, name: "s6_speaker_embedding", manifest: manifest)

      let swiftEmbed = model.ve.embedsFromWavs(wavs: [pyAudio])

      let result = TensorComparison.compare(swiftEmbed, pyEmbed, absoluteTolerance: 1e-2, name: "VoiceEncoder")
      print("Stage 6 (VoiceEncoder): correlation=\(String(format: "%.4f", result.correlation))")
      results.append(("VoiceEncoder", result.correlation, result.correlation > 0.95))
    } catch {
      print("Stage 6: Error - \(error)")
    }

    // Stage 9: T3 Conditioning
    do {
      let pyCondPrepared = try CheckpointReader.load(from: checkpointPath, name: "s9_t3_cond_prepared", manifest: manifest)
      let pyCondTokens = try CheckpointReader.load(from: checkpointPath, name: "s9_t3_cond_tokens", manifest: manifest)
      let pyVeEmbed = try CheckpointReader.load(from: checkpointPath, name: "s6_speaker_embedding", manifest: manifest)

      var t3Cond = T3TurboCond(speakerEmb: pyVeEmbed, condPromptSpeechTokens: pyCondTokens)
      let swiftCondPrepared = model.t3.prepareConditioning(&t3Cond)

      let result = TensorComparison.compare(swiftCondPrepared, pyCondPrepared, absoluteTolerance: 1e-2, name: "T3 Cond")
      print("Stage 9 (T3 Conditioning): correlation=\(String(format: "%.4f", result.correlation))")
      results.append(("T3 Conditioning", result.correlation, result.correlation > 0.95))
    } catch {
      print("Stage 9: Error - \(error)")
    }

    // Summary
    print("\n" + String(repeating: "=", count: 60))
    print("FULL MODEL COMPARISON SUMMARY")
    print(String(repeating: "=", count: 60))

    for r in results {
      let status = r.passed ? "✓" : "✗"
      print("\(status) \(r.stage): \(String(format: "%.4f", r.correlation))")
    }

    let allPassed = results.allSatisfy { $0.passed }
    print("\nOverall: \(allPassed ? "PASS" : "FAIL")")
    print(String(repeating: "=", count: 60))
  }
}
