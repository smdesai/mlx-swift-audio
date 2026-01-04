// Copyright © Anthony DePasquale

// Test comparing TokenBasedAligner (heuristic) vs AttentionBasedAligner (attention-based)
// to validate attention-based alignment improves word timing accuracy

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import Testing

@testable import MLXAudio

// MARK: - Accuracy Metrics

/// Metrics for evaluating alignment quality
struct AlignmentMetrics: CustomStringConvertible {
  let wordCount: Int
  let coveragePercent: Float       // Words with valid timings
  let monotonicityPercent: Float   // Words in monotonically increasing order
  let avgDurationMs: Float         // Average word duration
  let minDurationMs: Float         // Minimum duration (should be > 50ms)
  let maxDurationMs: Float         // Maximum duration (should be < 2000ms)
  let durationVariance: Float      // Variance in durations (lower = more uniform)
  let gapPercent: Float            // Percent of audio covered by word timings

  var description: String {
    """
    Words: \(wordCount), Coverage: \(String(format: "%.1f", coveragePercent))%
    Monotonic: \(String(format: "%.1f", monotonicityPercent))%
    Duration: avg=\(String(format: "%.0f", avgDurationMs))ms, min=\(String(format: "%.0f", minDurationMs))ms, max=\(String(format: "%.0f", maxDurationMs))ms
    Variance: \(String(format: "%.2f", durationVariance)), Gap coverage: \(String(format: "%.1f", gapPercent))%
    """
  }

  /// Calculate metrics from word timings
  static func calculate(from timings: [HighlightedWord], totalDuration: TimeInterval) -> AlignmentMetrics {
    guard !timings.isEmpty else {
      return AlignmentMetrics(
        wordCount: 0,
        coveragePercent: 0,
        monotonicityPercent: 0,
        avgDurationMs: 0,
        minDurationMs: 0,
        maxDurationMs: 0,
        durationVariance: 0,
        gapPercent: 0
      )
    }

    let wordCount = timings.count

    // Coverage: count words with valid (non-zero) timings
    let validTimings = timings.filter { $0.end > $0.start }
    let coveragePercent = Float(validTimings.count) / Float(wordCount) * 100

    // Monotonicity: count words with monotonically increasing start times
    var monotonicCount = 1
    for i in 1..<timings.count {
      if timings[i].start >= timings[i-1].start {
        monotonicCount += 1
      }
    }
    let monotonicityPercent = Float(monotonicCount) / Float(wordCount) * 100

    // Duration statistics
    let durations = timings.map { Float(($0.end - $0.start) * 1000) } // in ms
    let avgDurationMs = durations.reduce(0, +) / Float(durations.count)
    let minDurationMs = durations.min() ?? 0
    let maxDurationMs = durations.max() ?? 0

    // Variance
    let meanDuration = avgDurationMs
    let variance = durations.map { ($0 - meanDuration) * ($0 - meanDuration) }.reduce(0, +) / Float(durations.count)
    let durationVariance = sqrt(variance) / meanDuration // normalized

    // Gap coverage: how much of total audio is covered by word timings
    let totalWordTime = timings.reduce(0.0) { $0 + ($1.end - $1.start) }
    let gapPercent = Float(totalWordTime / totalDuration) * 100

    return AlignmentMetrics(
      wordCount: wordCount,
      coveragePercent: coveragePercent,
      monotonicityPercent: monotonicityPercent,
      avgDurationMs: avgDurationMs,
      minDurationMs: minDurationMs,
      maxDurationMs: maxDurationMs,
      durationVariance: durationVariance,
      gapPercent: gapPercent
    )
  }
}

// MARK: - Test Suite

@Suite(.serialized)
struct AlignmentAccuracyTests {
  static let referenceAudioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
  static let outputDir = URL(fileURLWithPath: "/tmp/alignment-accuracy")

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

  /// Main accuracy test comparing TokenBasedAligner vs AttentionBasedAligner
  @Test @MainActor func compareAlignerAccuracy() async throws {
    print("=== Alignment Accuracy Comparison Test ===\n")

    // Test sentences with varying complexity
    let testSentences = [
      "Hello world.",
      "The quick brown fox jumps over the lazy dog.",
      "She sells seashells by the seashore.",
      "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    ]

    // Setup
    print("Step 1: Loading model...")
    let model = try await ChatterboxTurboTestHelper.getOrLoadModel()
    print("  Model loaded")

    print("\nStep 2: Preparing reference audio...")
    let (refAudio, refSampleRate) = try await Self.downloadAudio(from: Self.referenceAudioURL)
    let conditionals = model.prepareConditionals(refWav: refAudio, refSr: refSampleRate)
    eval(conditionals.t3.speakerEmb)
    print("  Reference audio prepared")

    // Create output directory
    try FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)

    var report = "Alignment Accuracy Comparison Report\n"
    report += "=" * 60 + "\n\n"

    var tokenBasedMetricsAll: [AlignmentMetrics] = []
    var attentionBasedMetricsAll: [AlignmentMetrics] = []

    for (i, sentence) in testSentences.enumerated() {
      print("\n--- Test \(i + 1): \"\(sentence)\" ---")
      report += "Test \(i + 1): \"\(sentence)\"\n"
      report += "-" * 50 + "\n"

      // Generate with attention-based alignment
      print("  Generating with attention-based alignment...")
      let attentionStart = CFAbsoluteTimeGetCurrent()
      guard let result = await model.generateWithAlignment(
        text: sentence,
        conds: conditionals
      ) else {
        print("  ERROR: Attention-based generation failed")
        continue
      }
      let attentionTime = CFAbsoluteTimeGetCurrent() - attentionStart

      // Convert to samples
      result.audio.eval()
      let samples = result.audio.asArray(Float.self)
      let sampleRate = 24000 // ChatterboxTurboS3GenSr

      // Create attention-based aligner
      let attentionAligner = AttentionBasedAligner(
        alignmentData: result.alignmentData,
        textTokens: result.textTokens,
        tokenTexts: result.tokenTexts
      )
      let attentionTimings = attentionAligner.align(
        text: sentence,
        audioSamples: samples,
        sampleRate: sampleRate
      )
      print("  Attention-based: \(attentionTimings.count) words, \(String(format: "%.2f", attentionTime))s")

      // Generate with token-based alignment (using same audio to isolate aligner comparison)
      print("  Generating with token-based alignment...")
      let tokenStart = CFAbsoluteTimeGetCurrent()
      let tokenAligner = TokenBasedAligner()
      let tokenTimings = tokenAligner.align(
        text: sentence,
        audioSamples: samples,
        sampleRate: sampleRate
      )
      let tokenTime = CFAbsoluteTimeGetCurrent() - tokenStart
      print("  Token-based: \(tokenTimings.count) words, \(String(format: "%.4f", tokenTime))s")

      // Calculate metrics
      let totalDuration = Double(samples.count) / Double(sampleRate)
      let tokenMetrics = AlignmentMetrics.calculate(from: tokenTimings, totalDuration: totalDuration)
      let attentionMetrics = AlignmentMetrics.calculate(from: attentionTimings, totalDuration: totalDuration)

      tokenBasedMetricsAll.append(tokenMetrics)
      attentionBasedMetricsAll.append(attentionMetrics)

      print("\n  Token-based metrics:")
      print("    \(tokenMetrics)")
      print("\n  Attention-based metrics:")
      print("    \(attentionMetrics)")

      report += "\nToken-based:\n  \(tokenMetrics)\n"
      report += "\nAttention-based:\n  \(attentionMetrics)\n"

      // Detailed timing comparison
      report += "\nWord-by-word comparison:\n"
      let minCount = min(tokenTimings.count, attentionTimings.count)
      for j in 0..<minCount {
        let token = tokenTimings[j]
        let attn = attentionTimings[j]
        let tokenDur = (token.end - token.start) * 1000
        let attnDur = (attn.end - attn.start) * 1000
        let diff = abs(tokenDur - attnDur)
        let paddedWord = token.word.padding(toLength: 15, withPad: " ", startingAt: 0)
        report += "  \(paddedWord): token=\(String(format: "%6.0f", tokenDur))ms, attn=\(String(format: "%6.0f", attnDur))ms, diff=\(String(format: "%5.0f", diff))ms\n"
      }
      report += "\n"

      // Save audio for manual verification
      let audioURL = Self.outputDir.appendingPathComponent("test_\(i+1)_audio.wav")
      try saveWav(samples: samples, sampleRate: sampleRate, to: audioURL)
      print("  Audio saved to: \(audioURL.path)")
    }

    // Summary statistics
    print("\n=== SUMMARY ===")
    report += "\n" + "=" * 60 + "\n"
    report += "SUMMARY\n"
    report += "=" * 60 + "\n\n"

    let avgTokenMonotonic = tokenBasedMetricsAll.map { $0.monotonicityPercent }.reduce(0, +) / Float(tokenBasedMetricsAll.count)
    let avgAttnMonotonic = attentionBasedMetricsAll.map { $0.monotonicityPercent }.reduce(0, +) / Float(attentionBasedMetricsAll.count)

    let avgTokenVariance = tokenBasedMetricsAll.map { $0.durationVariance }.reduce(0, +) / Float(tokenBasedMetricsAll.count)
    let avgAttnVariance = attentionBasedMetricsAll.map { $0.durationVariance }.reduce(0, +) / Float(attentionBasedMetricsAll.count)

    let avgTokenCoverage = tokenBasedMetricsAll.map { $0.gapPercent }.reduce(0, +) / Float(tokenBasedMetricsAll.count)
    let avgAttnCoverage = attentionBasedMetricsAll.map { $0.gapPercent }.reduce(0, +) / Float(attentionBasedMetricsAll.count)

    print("Token-based averages:")
    print("  Monotonicity: \(String(format: "%.1f", avgTokenMonotonic))%")
    print("  Duration variance: \(String(format: "%.2f", avgTokenVariance))")
    print("  Gap coverage: \(String(format: "%.1f", avgTokenCoverage))%")

    print("\nAttention-based averages:")
    print("  Monotonicity: \(String(format: "%.1f", avgAttnMonotonic))%")
    print("  Duration variance: \(String(format: "%.2f", avgAttnVariance))")
    print("  Gap coverage: \(String(format: "%.1f", avgAttnCoverage))%")

    report += "Token-based averages:\n"
    report += "  Monotonicity: \(String(format: "%.1f", avgTokenMonotonic))%\n"
    report += "  Duration variance: \(String(format: "%.2f", avgTokenVariance))\n"
    report += "  Gap coverage: \(String(format: "%.1f", avgTokenCoverage))%\n\n"

    report += "Attention-based averages:\n"
    report += "  Monotonicity: \(String(format: "%.1f", avgAttnMonotonic))%\n"
    report += "  Duration variance: \(String(format: "%.2f", avgAttnVariance))\n"
    report += "  Gap coverage: \(String(format: "%.1f", avgAttnCoverage))%\n"

    // Key insight: lower variance + better gap coverage = more accurate alignment
    let varianceImprovement = (avgTokenVariance - avgAttnVariance) / avgTokenVariance * 100
    print("\nVariance improvement: \(String(format: "%.1f", varianceImprovement))%")
    report += "\nVariance improvement: \(String(format: "%.1f", varianceImprovement))%\n"

    // Save report
    let reportURL = Self.outputDir.appendingPathComponent("accuracy_report.txt")
    try report.write(to: reportURL, atomically: true, encoding: .utf8)
    print("\nFull report saved to: \(reportURL.path)")

    // Assertions
    #expect(avgAttnMonotonic >= 95.0, "Attention-based alignment should have high monotonicity")
    #expect(avgTokenMonotonic >= 95.0, "Token-based alignment should have high monotonicity")

    // Attention-based should have more varied (realistic) durations, not artificially uniform
    // We expect both to work, but attention-based to be more accurate

    print("\n=== Test Complete ===")
  }

  /// Test that compares latency overhead of attention extraction
  @Test @MainActor func measureAttentionOverhead() async throws {
    print("=== Attention Extraction Latency Test ===\n")

    let testText = "Hello, this is a test of the speech synthesis system."

    // Setup
    print("Loading model...")
    let model = try await ChatterboxTurboTestHelper.getOrLoadModel()

    let (refAudio, refSampleRate) = try await Self.downloadAudio(from: Self.referenceAudioURL)
    let conditionals = model.prepareConditionals(refWav: refAudio, refSr: refSampleRate)
    eval(conditionals.t3.speakerEmb)
    print("Setup complete\n")

    // Measure without attention (regular generation)
    print("Generating without attention extraction...")
    let startNoAttn = CFAbsoluteTimeGetCurrent()
    let resultNoAttn = model.generate(text: testText, conds: conditionals)
    resultNoAttn.eval()
    let timeNoAttn = CFAbsoluteTimeGetCurrent() - startNoAttn
    print("  Time: \(String(format: "%.2f", timeNoAttn))s")

    // Measure with attention
    print("\nGenerating with attention extraction...")
    let startWithAttn = CFAbsoluteTimeGetCurrent()
    guard let resultWithAttn = await model.generateWithAlignment(text: testText, conds: conditionals) else {
      throw TestError(message: "Failed to generate with alignment")
    }
    resultWithAttn.audio.eval()
    let timeWithAttn = CFAbsoluteTimeGetCurrent() - startWithAttn
    print("  Time: \(String(format: "%.2f", timeWithAttn))s")

    // Calculate overhead
    let overhead = (timeWithAttn - timeNoAttn) / timeNoAttn * 100
    print("\nOverhead: \(String(format: "%.1f", overhead))%")

    // Attention extraction should add minimal overhead (< 50%)
    #expect(overhead < 50.0, "Attention extraction should add minimal latency")

    print("\n=== Test Complete ===")
  }
}

// MARK: - Helpers

private extension String {
  static func * (string: String, count: Int) -> String {
    String(repeating: string, count: count)
  }
}

/// Save WAV file helper
private func saveWav(samples: [Float], sampleRate: Int, to url: URL) throws {
  let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
  let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
  buffer.frameLength = AVAudioFrameCount(samples.count)

  for i in 0..<samples.count {
    buffer.floatChannelData![0][i] = samples[i]
  }

  let file = try AVAudioFile(forWriting: url, settings: format.settings)
  try file.write(from: buffer)
}
