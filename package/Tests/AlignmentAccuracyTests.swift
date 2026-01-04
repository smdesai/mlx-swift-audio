// Test attention-based alignment for accurate word timing

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

  /// Main accuracy test for attention-based alignment
  @Test @MainActor func testAttentionBasedAlignment() async throws {
    print("=== Attention-Based Alignment Test ===\n")

    // Test sentences with varying complexity
    let testSentences = [
      "Hello world.",
      "The quick brown fox jumps over the lazy dog.",
      "She sells seashells by the seashore.",
      "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
      // Stress test: names, brands, medical terms, special characters
      "Saoirse Ronan and Timothée Chalamet starred in that new Netflix series. Dr. Wojciechowski prescribed Xarelto for my atrial fibrillation condition. Can you order the Oculus Quest from Amazon or should we get the PlayStation VR? I took two Advil and one Zyrtec, then ate at Chipotle with Siobhan. I bought a MacBook Pro and AirPods Max at the Apple Store in Schaumburg. Xavier ordered Postmates delivery of Häagen-Dazs and watched Disney Plus. Dr. Patel consulted with Dr. Okonkwo regarding the patient's acute cholecystitis and recommended initiating IV ceftriaxone 2 grams daily, along with ketorolac for pain management, while scheduling a laparoscopic cholecystectomy with Dr. Ramirez-Santos in the morning.",
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

    var report = "Attention-Based Alignment Report\n"
    report += "=" * 60 + "\n\n"

    var metricsAll: [AlignmentMetrics] = []

    for (i, sentence) in testSentences.enumerated() {
      print("\n--- Test \(i + 1): \"\(sentence.prefix(60))...\" ---")
      report += "Test \(i + 1): \"\(sentence)\"\n"
      report += "-" * 50 + "\n"

      // Generate with attention-based alignment
      print("  Generating with attention-based alignment...")
      let startTime = CFAbsoluteTimeGetCurrent()
      guard let result = await model.generateWithAlignment(
        text: sentence,
        conds: conditionals
      ) else {
        print("  ERROR: Attention-based generation failed")
        continue
      }
      let generationTime = CFAbsoluteTimeGetCurrent() - startTime

      // Convert to samples
      result.audio.eval()
      let samples = result.audio.asArray(Float.self)
      let sampleRate = 24000 // ChatterboxTurboS3GenSr

      // Create attention-based aligner
      let aligner = AttentionBasedAligner(
        alignmentData: result.alignmentData,
        textTokens: result.textTokens,
        tokenTexts: result.tokenTexts
      )
      let timings = aligner.align(
        text: sentence,
        audioSamples: samples,
        sampleRate: sampleRate
      )
      print("  Words: \(timings.count), Generation time: \(String(format: "%.2f", generationTime))s")

      // Calculate metrics
      let totalDuration = Double(samples.count) / Double(sampleRate)
      let metrics = AlignmentMetrics.calculate(from: timings, totalDuration: totalDuration)
      metricsAll.append(metrics)

      print("  Metrics: \(metrics)")
      report += "\nMetrics:\n  \(metrics)\n"

      // Word timings detail
      report += "\nWord timings:\n"
      for timing in timings {
        let dur = (timing.end - timing.start) * 1000
        let paddedWord = timing.word.padding(toLength: 20, withPad: " ", startingAt: 0)
        report += "  \(paddedWord): \(String(format: "%.0f", timing.start * 1000))ms - \(String(format: "%.0f", timing.end * 1000))ms (\(String(format: "%.0f", dur))ms)\n"
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

    let avgMonotonic = metricsAll.map { $0.monotonicityPercent }.reduce(0, +) / Float(metricsAll.count)
    let avgVariance = metricsAll.map { $0.durationVariance }.reduce(0, +) / Float(metricsAll.count)
    let avgCoverage = metricsAll.map { $0.gapPercent }.reduce(0, +) / Float(metricsAll.count)

    print("Attention-based averages:")
    print("  Monotonicity: \(String(format: "%.1f", avgMonotonic))%")
    print("  Duration variance: \(String(format: "%.2f", avgVariance))")
    print("  Gap coverage: \(String(format: "%.1f", avgCoverage))%")

    report += "Attention-based averages:\n"
    report += "  Monotonicity: \(String(format: "%.1f", avgMonotonic))%\n"
    report += "  Duration variance: \(String(format: "%.2f", avgVariance))\n"
    report += "  Gap coverage: \(String(format: "%.1f", avgCoverage))%\n"

    // Save report
    let reportURL = Self.outputDir.appendingPathComponent("accuracy_report.txt")
    try report.write(to: reportURL, atomically: true, encoding: .utf8)
    print("\nFull report saved to: \(reportURL.path)")

    // Assertions
    #expect(avgMonotonic >= 95.0, "Attention-based alignment should have high monotonicity")
    #expect(avgCoverage >= 95.0, "Attention-based alignment should have high gap coverage")

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

// MARK: - Sparse Attention Sampling Tests

@Suite(.serialized)
struct SparseAttentionSamplingTests {

  /// Test linear interpolation of attention vectors
  @Test func interpolateAttentionVectors() {
    // Arrange: Create sparse samples at positions 0, 4, 8
    let textTokenCount = 3
    let sparseSamples: [(index: Int, attention: [Float])] = [
      (0, [1.0, 0.0, 0.0]),   // Position 0: attention on first text token
      (4, [0.0, 1.0, 0.0]),   // Position 4: attention on second text token
      (8, [0.0, 0.0, 1.0]),   // Position 8: attention on third text token
    ]
    let totalSpeechTokens = 9

    // Act: Interpolate to get full attention matrix
    let fullAttention = SparseAttentionInterpolator.interpolate(
      sparseSamples: sparseSamples,
      totalSpeechTokens: totalSpeechTokens,
      textTokenCount: textTokenCount
    )

    // Assert: Check interpolated values
    #expect(fullAttention.count == totalSpeechTokens)

    // Position 0: exact sample [1, 0, 0]
    #expect(fullAttention[0] == [1.0, 0.0, 0.0])

    // Position 2: midpoint between 0 and 4 → [0.5, 0.5, 0]
    #expect(abs(fullAttention[2][0] - 0.5) < 0.01)
    #expect(abs(fullAttention[2][1] - 0.5) < 0.01)
    #expect(abs(fullAttention[2][2] - 0.0) < 0.01)

    // Position 4: exact sample [0, 1, 0]
    #expect(fullAttention[4] == [0.0, 1.0, 0.0])

    // Position 6: midpoint between 4 and 8 → [0, 0.5, 0.5]
    #expect(abs(fullAttention[6][0] - 0.0) < 0.01)
    #expect(abs(fullAttention[6][1] - 0.5) < 0.01)
    #expect(abs(fullAttention[6][2] - 0.5) < 0.01)

    // Position 8: exact sample [0, 0, 1]
    #expect(fullAttention[8] == [0.0, 0.0, 1.0])
  }

  /// Test edge case: single sample (no interpolation needed)
  @Test func singleSampleNoInterpolation() {
    let sparseSamples: [(index: Int, attention: [Float])] = [
      (0, [0.5, 0.3, 0.2]),
    ]

    let fullAttention = SparseAttentionInterpolator.interpolate(
      sparseSamples: sparseSamples,
      totalSpeechTokens: 3,
      textTokenCount: 3
    )

    // All positions should extrapolate from the single sample
    #expect(fullAttention.count == 3)
    #expect(fullAttention[0] == [0.5, 0.3, 0.2])
    #expect(fullAttention[1] == [0.5, 0.3, 0.2])
    #expect(fullAttention[2] == [0.5, 0.3, 0.2])
  }

  /// Test extrapolation before first sample
  @Test func extrapolateBeforeFirstSample() {
    let sparseSamples: [(index: Int, attention: [Float])] = [
      (2, [1.0, 0.0]),
      (4, [0.0, 1.0]),
    ]

    let fullAttention = SparseAttentionInterpolator.interpolate(
      sparseSamples: sparseSamples,
      totalSpeechTokens: 5,
      textTokenCount: 2
    )

    // Positions 0, 1 should use first sample (no extrapolation beyond bounds)
    #expect(fullAttention[0] == [1.0, 0.0])
    #expect(fullAttention[1] == [1.0, 0.0])
    #expect(fullAttention[2] == [1.0, 0.0])
  }

  /// Test that sparse sampling produces similar results to full sampling
  @Test func sparseSamplingQuality() {
    // Simulate a realistic attention pattern (smooth transition across text tokens)
    let textTokenCount = 5
    let totalSpeechTokens = 20

    // Create "ground truth" full attention (gradual focus shift across text)
    var fullSamples: [[Float]] = []
    for speechIdx in 0..<totalSpeechTokens {
      let progress = Float(speechIdx) / Float(totalSpeechTokens - 1)
      var attention = [Float](repeating: 0, count: textTokenCount)
      // Create a soft focus that shifts from first to last text token
      for textIdx in 0..<textTokenCount {
        let textProgress = Float(textIdx) / Float(textTokenCount - 1)
        let distance = abs(textProgress - progress)
        attention[textIdx] = exp(-distance * 3) // Gaussian-like falloff
      }
      // Normalize
      let sum = attention.reduce(0, +)
      if sum > 0 { attention = attention.map { $0 / sum } }
      fullSamples.append(attention)
    }

    // Create sparse samples (every 4th token)
    let sampleInterval = 4
    var sparseSamples: [(index: Int, attention: [Float])] = []
    for i in stride(from: 0, to: totalSpeechTokens, by: sampleInterval) {
      sparseSamples.append((index: i, attention: fullSamples[i]))
    }

    // Interpolate
    let interpolated = SparseAttentionInterpolator.interpolate(
      sparseSamples: sparseSamples,
      totalSpeechTokens: totalSpeechTokens,
      textTokenCount: textTokenCount
    )

    // Measure error between full and interpolated
    var totalError: Float = 0
    for i in 0..<totalSpeechTokens {
      for j in 0..<textTokenCount {
        let error = abs(fullSamples[i][j] - interpolated[i][j])
        totalError += error
      }
    }
    let avgError = totalError / Float(totalSpeechTokens * textTokenCount)

    // Sparse sampling should produce low interpolation error for smooth patterns
    print("[SparseQuality] Average interpolation error: \(String(format: "%.4f", avgError))")
    #expect(avgError < 0.1, "Interpolation error should be small for smooth attention patterns")
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
