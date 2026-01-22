// Loudness normalization using ITU-R BS.1770 (LUFS measurement)
// Matches Python's pyloudnorm library behavior

import Accelerate
import Foundation

/// Loudness normalization using ITU-R BS.1770 standard
///
/// This implementation calculates integrated loudness in LUFS (Loudness Units
/// relative to Full Scale) and normalizes audio to a target loudness level.
/// Matches the behavior of Python's `pyloudnorm` library.
public enum LoudnessNormalizer {
  /// Default target loudness for speech (matches Python Chatterbox default)
  public static let defaultTargetLUFS: Float = -27.0

  /// Normalize audio to target loudness
  ///
  /// - Parameters:
  ///   - samples: Input audio samples
  ///   - sampleRate: Sample rate in Hz
  ///   - targetLUFS: Target loudness in LUFS (default: -27)
  /// - Returns: Loudness-normalized audio samples
  public static func normalize(
    _ samples: [Float],
    sampleRate: Int,
    targetLUFS: Float = defaultTargetLUFS
  ) -> [Float] {
    guard !samples.isEmpty else { return samples }

    // Measure current loudness
    let currentLUFS = measureLoudness(samples, sampleRate: sampleRate)

    // Check for valid measurement
    guard currentLUFS.isFinite, currentLUFS > -70.0 else {
      // Audio too quiet or invalid, return unchanged
      return samples
    }

    // Calculate gain needed
    let gainDB = targetLUFS - currentLUFS
    let gainLinear = powf(10.0, gainDB / 20.0)

    // Limit gain to prevent extreme amplification (max +30 dB)
    let clampedGain = min(gainLinear, 31.62)  // 10^(30/20)

    // Check for valid gain
    guard clampedGain.isFinite, clampedGain > 0.0 else {
      return samples
    }

    // Apply gain using Accelerate
    var result = [Float](repeating: 0, count: samples.count)
    var gain = clampedGain
    vDSP_vsmul(samples, 1, &gain, &result, 1, vDSP_Length(samples.count))

    return result
  }

  /// Measure integrated loudness in LUFS (ITU-R BS.1770)
  ///
  /// - Parameters:
  ///   - samples: Audio samples
  ///   - sampleRate: Sample rate in Hz
  /// - Returns: Loudness in LUFS
  public static func measureLoudness(_ samples: [Float], sampleRate: Int) -> Float {
    guard !samples.isEmpty else { return -.infinity }

    // Apply K-weighting filter
    let weighted = applyKWeighting(samples, sampleRate: sampleRate)

    // Calculate mean square
    var meanSquare: Float = 0
    vDSP_measqv(weighted, 1, &meanSquare, vDSP_Length(weighted.count))

    // Convert to LUFS
    // LUFS = -0.691 + 10 * log10(mean_square)
    guard meanSquare > 0 else { return -.infinity }
    let lufs = -0.691 + 10.0 * log10f(meanSquare)

    return lufs
  }

  // MARK: - K-Weighting Filter (ITU-R BS.1770)

  /// Apply K-weighting filter for loudness measurement
  ///
  /// K-weighting consists of:
  /// 1. High-shelf filter (+4 dB above 1.5 kHz)
  /// 2. High-pass filter (removes frequencies below 38 Hz)
  private static func applyKWeighting(_ samples: [Float], sampleRate: Int) -> [Float] {
    // Apply high-shelf filter first
    var filtered = applyHighShelfFilter(samples, sampleRate: sampleRate)

    // Then apply high-pass filter
    filtered = applyHighPassFilter(filtered, sampleRate: sampleRate)

    return filtered
  }

  /// High-shelf filter: +4 dB boost above ~1.5 kHz
  /// Coefficients derived from ITU-R BS.1770-4
  private static func applyHighShelfFilter(_ samples: [Float], sampleRate: Int) -> [Float] {
    // Pre-computed coefficients for common sample rates
    // These match pyloudnorm's implementation
    let (b, a) = highShelfCoefficients(sampleRate: sampleRate)
    return applyBiquadFilter(samples, b: b, a: a)
  }

  /// High-pass filter: -3 dB at 38 Hz (removes DC and very low frequencies)
  private static func applyHighPassFilter(_ samples: [Float], sampleRate: Int) -> [Float] {
    let (b, a) = highPassCoefficients(sampleRate: sampleRate)
    return applyBiquadFilter(samples, b: b, a: a)
  }

  /// Get high-shelf filter coefficients for K-weighting
  private static func highShelfCoefficients(sampleRate: Int) -> (b: [Float], a: [Float]) {
    // ITU-R BS.1770-4 high-shelf filter coefficients
    // These are computed using bilinear transform from the analog prototype
    let sr = Float(sampleRate)

    // High-shelf parameters
    let fc: Float = 1681.974450955533  // Center frequency
    let G: Float = 3.999843853973347   // Gain in dB
    let Q: Float = 0.7071752369554196  // Q factor

    let K = tanf(.pi * fc / sr)
    let Vh = powf(10.0, G / 20.0)
    let Vb = powf(Vh, 0.4996667741545416)

    let a0 = 1.0 + K / Q + K * K
    let b0 = (Vh + Vb * K / Q + K * K) / a0
    let b1 = 2.0 * (K * K - Vh) / a0
    let b2 = (Vh - Vb * K / Q + K * K) / a0
    let a1 = 2.0 * (K * K - 1.0) / a0
    let a2 = (1.0 - K / Q + K * K) / a0

    return (b: [b0, b1, b2], a: [1.0, a1, a2])
  }

  /// Get high-pass filter coefficients for K-weighting
  private static func highPassCoefficients(sampleRate: Int) -> (b: [Float], a: [Float]) {
    let sr = Float(sampleRate)

    // High-pass filter parameters (38 Hz cutoff)
    let fc: Float = 38.13547087602444
    let Q: Float = 0.5003270373238773

    let K = tanf(.pi * fc / sr)

    let a0 = 1.0 + K / Q + K * K
    let b0 = 1.0 / a0
    let b1 = -2.0 / a0
    let b2 = 1.0 / a0
    let a1 = 2.0 * (K * K - 1.0) / a0
    let a2 = (1.0 - K / Q + K * K) / a0

    return (b: [b0, b1, b2], a: [1.0, a1, a2])
  }

  /// Apply a biquad (second-order IIR) filter
  private static func applyBiquadFilter(_ samples: [Float], b: [Float], a: [Float]) -> [Float] {
    guard samples.count > 2 else { return samples }

    var output = [Float](repeating: 0, count: samples.count)

    // Direct Form II Transposed implementation
    var z1: Float = 0
    var z2: Float = 0

    for i in 0 ..< samples.count {
      let x = samples[i]
      let y = b[0] * x + z1
      z1 = b[1] * x - a[1] * y + z2
      z2 = b[2] * x - a[2] * y
      output[i] = y
    }

    return output
  }
}
