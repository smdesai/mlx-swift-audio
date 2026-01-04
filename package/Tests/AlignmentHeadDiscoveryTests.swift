// Copyright © Anthony DePasquale

// Test to discover which GPT-2 attention heads contain text-to-speech alignment information
// Based on the approach from original Chatterbox AlignmentStreamAnalyzer

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import Testing

@testable import MLXAudio

// MARK: - Alignment Analysis

/// Analyzes attention patterns to find alignment heads
struct AlignmentAnalyzer {
  /// Score for a single attention head indicating how well it tracks text-speech alignment
  struct HeadScore: CustomStringConvertible {
    let layer: Int
    let head: Int
    let diagonalScore: Float      // How diagonal is the attention pattern
    let monotonicScore: Float     // How monotonic is the text position tracking
    let combinedScore: Float      // Overall alignment quality

    var description: String {
      String(format: "Layer %2d Head %2d: diagonal=%.3f monotonic=%.3f combined=%.3f",
             layer, head, diagonalScore, monotonicScore, combinedScore)
    }
  }

  /// Analyze attention weights to compute alignment scores
  /// - Parameters:
  ///   - attention: Attention weights (B, numHeads, T_speech, T_text+cond)
  ///   - numTextTokens: Number of text tokens (excluding conditioning)
  ///   - condLength: Length of conditioning prefix
  /// - Returns: Array of scores for each head
  static func analyzeAttention(
    _ attention: MLXArray,
    numTextTokens: Int,
    condLength: Int
  ) -> [HeadScore] {
    let numHeads = attention.shape[1]
    let layer = 0  // Will be set by caller

    var scores: [HeadScore] = []

    for head in 0..<numHeads {
      // Extract attention for this head: (T_speech, T_total)
      let headAttn = attention[0, head]
      eval(headAttn)

      // Focus on text region only (skip conditioning)
      let textStart = condLength
      let textEnd = condLength + numTextTokens

      // Get attention to text tokens: (T_speech, T_text)
      let textAttn = headAttn[0..., textStart..<textEnd]
      eval(textAttn)

      // Compute diagonal score: how much does each speech position attend to corresponding text position?
      let diagonalScore = computeDiagonalScore(textAttn, numTextTokens: numTextTokens)

      // Compute monotonic score: does argmax of attention move monotonically through text?
      let monotonicScore = computeMonotonicScore(textAttn)

      // Combined score emphasizes both properties
      let combinedScore = (diagonalScore * 0.4 + monotonicScore * 0.6)

      scores.append(HeadScore(
        layer: layer,
        head: head,
        diagonalScore: diagonalScore,
        monotonicScore: monotonicScore,
        combinedScore: combinedScore
      ))
    }

    return scores
  }

  /// Compute how diagonal the attention pattern is
  /// Perfect diagonal would have attention concentrated on positions proportional to speech position
  private static func computeDiagonalScore(_ attn: MLXArray, numTextTokens: Int) -> Float {
    let numSpeechTokens = attn.shape[0]
    guard numSpeechTokens > 0, numTextTokens > 0 else { return 0 }

    // For each speech position, check if it attends to the "expected" text position
    // Expected position = (speech_pos / num_speech) * num_text
    var totalScore: Float = 0
    let attnArray = attn.asArray(Float.self)

    for speechPos in 0..<numSpeechTokens {
      let expectedTextPos = Int((Float(speechPos) / Float(numSpeechTokens)) * Float(numTextTokens))
      let clampedPos = min(max(expectedTextPos, 0), numTextTokens - 1)

      // Get attention value at expected position
      let attnValue = attnArray[speechPos * numTextTokens + clampedPos]
      totalScore += attnValue
    }

    return totalScore / Float(numSpeechTokens)
  }

  /// Compute how monotonically the attention moves through text
  /// Good alignment heads should move forward through text as speech progresses
  private static func computeMonotonicScore(_ attn: MLXArray) -> Float {
    let numSpeechTokens = attn.shape[0]
    let numTextTokens = attn.shape[1]
    guard numSpeechTokens > 1, numTextTokens > 0 else { return 0 }

    // Get argmax for each speech position
    let argmaxPositions = MLX.argMax(attn, axis: 1)
    eval(argmaxPositions)
    let positions = argmaxPositions.asArray(Int32.self)

    // Count how many transitions are forward (monotonic)
    var forwardTransitions = 0
    var totalTransitions = 0

    for i in 1..<positions.count {
      let prev = positions[i - 1]
      let curr = positions[i]
      if curr >= prev {
        forwardTransitions += 1
      }
      totalTransitions += 1
    }

    guard totalTransitions > 0 else { return 0 }
    return Float(forwardTransitions) / Float(totalTransitions)
  }
}

// MARK: - Test Suite

@Suite(.serialized)
struct AlignmentHeadDiscoveryTests {
  static let referenceAudioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
  static let outputDir = URL(fileURLWithPath: "/tmp/alignment-discovery")

  /// Download audio helper
  static func downloadAudio(from url: URL) async throws -> (audio: MLXArray, sampleRate: Int) {
    let cacheURL = try await TestAudioCache.downloadToFile(from: url)
    let file = try AVAudioFile(forReading: cacheURL)

    guard let buffer = AVAudioPCMBuffer(
      pcmFormat: file.processingFormat,
      frameCapacity: AVAudioFrameCount(file.length)
    ) else {
      throw TestError(message: "Failed to create buffer")
    }

    try file.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw TestError(message: "No float data")
    }

    let frameCount = Int(buffer.frameLength)
    var samples = [Float](repeating: 0, count: frameCount)
    for i in 0..<frameCount {
      samples[i] = floatData[0][i]
    }

    return (MLXArray(samples), Int(file.fileFormat.sampleRate))
  }

  /// Discover alignment heads by running inference with attention output
  @Test @MainActor func discoverAlignmentHeads() async throws {
    print("=== GPT-2 Alignment Head Discovery ===\n")

    // Load model
    print("Step 1: Loading ChatterboxTurbo model...")
    let model = try await ChatterboxTurboTestHelper.getOrLoadModel()
    print("  Model loaded")

    // Download reference audio
    print("\nStep 2: Preparing reference audio...")
    let (refAudio, refSampleRate) = try await Self.downloadAudio(from: Self.referenceAudioURL)
    let conditionals = model.prepareConditionals(refWav: refAudio, refSr: refSampleRate)
    eval(conditionals.t3.speakerEmb)
    print("  Reference audio prepared")

    // Test text - short for faster analysis
    let testText = "Hello world, this is a test."
    print("\nStep 3: Tokenizing text...")
    guard let tokenizer = model.textTokenizer else {
      throw TestError(message: "Text tokenizer not loaded")
    }
    let textTokens = tokenizer.encode(text: testText, addSpecialTokens: false)
    let numTextTokens = textTokens.count
    print("  Text tokens: \(numTextTokens)")

    // We need to access the T3 model internals to extract attention
    // This requires running inference with attention output enabled
    print("\nStep 4: Running inference with attention extraction...")
    print("  (This will be slower than normal inference)")

    // Prepare for inference - we'll run just the T3 model portion with attention
    let allScores = try await runInferenceWithAttention(
      model: model,
      conditionals: conditionals,
      textTokens: textTokens
    )

    // Sort by combined score and report
    print("\n=== ALIGNMENT HEAD DISCOVERY RESULTS ===\n")

    let sortedScores = allScores.sorted { $0.combinedScore > $1.combinedScore }
    let topHeads = sortedScores.prefix(10)

    print("Top 10 candidate alignment heads:")
    print("-" * 60)
    for (rank, score) in topHeads.enumerated() {
      print("\(rank + 1). \(score)")
    }

    // Print as Swift code for easy copy-paste
    print("\n// Candidate alignment heads for Chatterbox Turbo GPT-2:")
    print("let alignmentHeads: [(layer: Int, head: Int)] = [")
    for score in topHeads.prefix(5) {
      print("  (\(score.layer), \(score.head)),")
    }
    print("]")

    // Save full results to file
    try FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)
    let resultsURL = Self.outputDir.appendingPathComponent("alignment_heads_report.txt")
    var report = "Alignment Head Discovery Report\n"
    report += "=" * 50 + "\n\n"
    report += "Test text: \"\(testText)\"\n"
    report += "Text tokens: \(numTextTokens)\n\n"
    report += "All heads sorted by combined score:\n"
    report += "-" * 50 + "\n"
    for score in sortedScores {
      report += "\(score)\n"
    }
    try report.write(to: resultsURL, atomically: true, encoding: .utf8)
    print("\nFull report saved to: \(resultsURL.path)")

    // Assertions
    #expect(!allScores.isEmpty, "Should have analyzed attention heads")
    #expect(sortedScores[0].combinedScore > 0.3, "Top head should have reasonable alignment score")

    print("\n=== Discovery Complete ===")
  }

  /// Run T3 inference with attention extraction
  @MainActor
  private func runInferenceWithAttention(
    model: ChatterboxTurboModel,
    conditionals: ChatterboxTurboConditionals,
    textTokens: [Int]
  ) async throws -> [AlignmentAnalyzer.HeadScore] {
    // Access T3 model
    let t3 = model.t3
    let config = t3.config

    // Prepare text tokens with start/stop
    var tokens = [config.startTextToken] + textTokens + [config.stopTextToken]
    let textTokensArray = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)

    // Prepare conditioning
    var t3Cond = conditionals.t3

    // Initial speech token
    let B = 1
    let speechStartToken = MLXArray.full([B, 1], values: MLXArray(Int32(config.startSpeechToken)))

    // Prepare embeddings
    let (embeds, condLength) = t3.prepareInputEmbeds(
      t3Cond: &t3Cond,
      textTokens: textTokensArray,
      speechTokens: speechStartToken
    )
    eval(embeds)

    print("  Conditioning length: \(condLength)")
    print("  Initial sequence length: \(embeds.shape[1])")

    // Create cache
    let cache = t3.tfmr.newCache()

    // All layers to analyze (GPT-2 Medium has 24 layers)
    let allLayers = Set(0..<24)

    // Initial forward pass with attention from all layers
    print("  Running initial prefill with attention extraction...")
    let output = t3.tfmr.forward(
      inputsEmbeds: embeds,
      cache: cache,
      outputAttentionsForLayers: allLayers
    )
    eval(output.hiddenStates)

    // Collect scores from all layers
    var allScores: [AlignmentAnalyzer.HeadScore] = []

    // Generate a few more tokens to build up attention patterns
    print("  Generating tokens to build attention patterns...")
    let maxTokens = 50  // Generate enough for pattern analysis
    var currentToken = MLXArray.full([B, 1], values: MLXArray(Int32(config.startSpeechToken)))
    var generatedCount = 0

    // Accumulate attention patterns
    var accumulatedAttention: [Int: [MLXArray]] = [:]
    for layer in allLayers {
      accumulatedAttention[layer] = []
    }

    // Store initial attention
    for (layer, attn) in output.attentions {
      accumulatedAttention[layer]?.append(attn)
    }

    // Get speech head for sampling
    let speechHidden = output.hiddenStates[0..., (output.hiddenStates.shape[1] - 1)..<output.hiddenStates.shape[1], 0...]
    var speechLogits = t3.speechHead(speechHidden)
    currentToken = MLX.argMax(speechLogits[0..., -1, 0...], axis: -1).expandedDimensions(axis: 0)

    for step in 0..<maxTokens {
      // Get embedding for current token
      let currentEmbed = t3.speechEmb(currentToken)

      // Forward with attention
      let stepOutput = t3.tfmr.forward(
        inputsEmbeds: currentEmbed,
        cache: cache,
        outputAttentionsForLayers: allLayers
      )
      eval(stepOutput.hiddenStates)

      // Store attention from this step
      for (layer, attn) in stepOutput.attentions {
        accumulatedAttention[layer]?.append(attn)
      }

      // Sample next token
      speechLogits = t3.speechHead(stepOutput.hiddenStates)
      let nextToken = MLX.argMax(speechLogits[0..., -1, 0...], axis: -1)
      eval(nextToken)

      let tokenId = nextToken.item(Int32.self)
      if tokenId == Int32(config.stopSpeechToken) {
        print("  EOS reached at step \(step)")
        break
      }

      currentToken = nextToken.expandedDimensions(axis: 0)
      generatedCount += 1

      if step % 10 == 0 {
        print("  Generated \(step) speech tokens...")
      }
    }

    print("  Total speech tokens generated: \(generatedCount)")

    // Analyze accumulated attention for each layer
    print("\nStep 5: Analyzing attention patterns...")

    for layer in 0..<24 {
      guard let layerAttentions = accumulatedAttention[layer], !layerAttentions.isEmpty else {
        continue
      }

      // Use the last attention matrix (has full context)
      // Shape: (B, numHeads, 1, T_total) for single-token generation steps
      // We need to build the full attention matrix or use the prefill attention
      let prefillAttn = layerAttentions[0]  // (B, numHeads, T_prefill, T_prefill)

      // Analyze this layer's attention
      let layerScores = AlignmentAnalyzer.analyzeAttention(
        prefillAttn,
        numTextTokens: textTokens.count,
        condLength: condLength
      )

      // Update layer index in scores
      for var score in layerScores {
        allScores.append(AlignmentAnalyzer.HeadScore(
          layer: layer,
          head: score.head,
          diagonalScore: score.diagonalScore,
          monotonicScore: score.monotonicScore,
          combinedScore: score.combinedScore
        ))
      }

      if layer % 6 == 0 {
        print("  Analyzed layer \(layer)/23")
      }
    }

    return allScores
  }
}

// MARK: - Helpers

private extension String {
  static func * (string: String, count: Int) -> String {
    String(repeating: string, count: count)
  }
}
