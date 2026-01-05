// Copyright © Sachin Desai

import Accelerate
import Foundation

// MARK: - ForcedAligner Protocol

/// Protocol for forced alignment implementations that provide word-level timings.
///
/// Implementations align text to audio, returning word-level timing information
/// that can be used for highlighting words as they're spoken.
public protocol ForcedAligner: Sendable {
  /// Align text to audio, returning word-level timings.
  ///
  /// - Parameters:
  ///   - text: The text that was synthesized
  ///   - audioSamples: The generated audio samples
  ///   - sampleRate: Sample rate of the audio in Hz
  /// - Returns: Array of word timings with start/end times
  func align(
    text: String,
    audioSamples: [Float],
    sampleRate: Int
  ) -> [HighlightedWord]

  /// Align text to audio with full text context for multi-chunk processing.
  ///
  /// Default implementation falls back to the basic align method.
  ///
  /// - Parameters:
  ///   - text: The chunk text that was synthesized
  ///   - fullText: The complete original text
  ///   - baseOffset: Character offset in fullText where this chunk starts
  ///   - audioSamples: The generated audio samples
  ///   - sampleRate: Sample rate of the audio in Hz
  /// - Returns: Array of word timings with start/end times and full-text charRanges
  func align(
    text: String,
    fullText: String,
    baseOffset: String.Index,
    audioSamples: [Float],
    sampleRate: Int
  ) -> [HighlightedWord]
}

// MARK: - Default Implementation

public extension ForcedAligner {
  func align(
    text: String,
    fullText _: String,
    baseOffset _: String.Index,
    audioSamples: [Float],
    sampleRate: Int
  ) -> [HighlightedWord] {
    // Default: use basic align (charRanges will be chunk-relative)
    align(text: text, audioSamples: audioSamples, sampleRate: sampleRate)
  }
}

// MARK: - Sparse Attention Interpolation

/// Interpolates sparse attention samples to produce a full attention matrix.
///
/// When using sparse attention sampling (extracting every Nth token), this utility
/// linearly interpolates between sampled positions to reconstruct the full matrix.
/// This reduces GPU memory and CPU transfer overhead while maintaining alignment quality.
public enum SparseAttentionInterpolator {
  /// Interpolate sparse attention samples to produce full attention for all speech positions.
  ///
  /// - Parameters:
  ///   - sparseSamples: Array of (index, attention) tuples at sampled positions
  ///   - totalSpeechTokens: Total number of speech tokens to interpolate
  ///   - textTokenCount: Number of text tokens (dimension of attention vectors)
  /// - Returns: Full attention matrix as `[[Float]]` with one entry per speech token
  public static func interpolate(
    sparseSamples: [(index: Int, attention: [Float])],
    totalSpeechTokens: Int,
    textTokenCount: Int
  ) -> [[Float]] {
    guard !sparseSamples.isEmpty else {
      return Array(repeating: [Float](repeating: 0, count: textTokenCount), count: totalSpeechTokens)
    }

    // Sort samples by index
    let sorted = sparseSamples.sorted { $0.index < $1.index }

    var result: [[Float]] = []
    result.reserveCapacity(totalSpeechTokens)

    for i in 0 ..< totalSpeechTokens {
      // Find bracketing samples
      let (prevSample, nextSample) = findBracketingSamples(for: i, in: sorted)

      if prevSample.index == i {
        // Exact match - use sample directly
        result.append(prevSample.attention)
      } else if prevSample.index == nextSample.index {
        // Single sample or extrapolation - use nearest
        result.append(prevSample.attention)
      } else {
        // Linear interpolation between prev and next
        let t = Float(i - prevSample.index) / Float(nextSample.index - prevSample.index)
        var interpolated = [Float](repeating: 0, count: textTokenCount)
        for j in 0 ..< min(textTokenCount, prevSample.attention.count, nextSample.attention.count) {
          interpolated[j] = (1 - t) * prevSample.attention[j] + t * nextSample.attention[j]
        }
        result.append(interpolated)
      }
    }

    return result
  }

  /// Find the samples that bracket the given index for interpolation.
  private static func findBracketingSamples(
    for index: Int,
    in sorted: [(index: Int, attention: [Float])]
  ) -> (prev: (index: Int, attention: [Float]), next: (index: Int, attention: [Float])) {
    // Handle edge cases
    guard let first = sorted.first, let last = sorted.last else {
      fatalError("findBracketingSamples called with empty sorted array")
    }

    // Before first sample - extrapolate using first
    if index <= first.index {
      return (first, first)
    }

    // After last sample - extrapolate using last
    if index >= last.index {
      return (last, last)
    }

    // Find bracketing samples
    var prev = first
    for sample in sorted {
      if sample.index == index {
        return (sample, sample)
      }
      if sample.index > index {
        return (prev, sample)
      }
      prev = sample
    }

    return (last, last)
  }
}

// MARK: - AttentionBasedAligner

/// Aligner that uses attention weights from T3 model for accurate word timing.
///
/// This aligner uses the cross-attention weights captured during speech generation
/// to determine which text tokens are being attended to at each speech token position.
/// DTW (Dynamic Time Warping) is then used to find the optimal alignment path.
///
/// Accuracy is significantly higher than heuristic-based alignment (~95% vs ~80%).
public final class AttentionBasedAligner: ForcedAligner, @unchecked Sendable {
  // MARK: - Properties

  /// Captured alignment data from T3 generation
  private let alignmentData: AlignmentData

  /// Text tokens from the tokenizer (for mapping to words)
  private let textTokens: [Int]

  /// Decoded text for each token (for word boundary detection)
  private let tokenTexts: [String]

  /// Time per speech token (40ms at 25Hz)
  private static let secondsPerToken: TimeInterval = 0.04

  /// Sample rate for Chatterbox audio
  private static let sampleRate: Int = 24000

  /// Samples per speech token
  private static let samplesPerToken: Int = 960

  // MARK: - Initialization

  /// Initialize with alignment data from T3 generation.
  ///
  /// - Parameters:
  ///   - alignmentData: Captured attention data from `inferenceTurboStreamWithAttention`
  ///   - textTokens: Token IDs from the GPT-2 tokenizer
  ///   - tokenTexts: Decoded text for each token
  public init(
    alignmentData: AlignmentData,
    textTokens: [Int],
    tokenTexts: [String]
  ) {
    self.alignmentData = alignmentData
    self.textTokens = textTokens
    self.tokenTexts = tokenTexts
  }

  // MARK: - ForcedAligner

  public func align(
    text: String,
    audioSamples: [Float],
    sampleRate: Int
  ) -> [HighlightedWord] {
    align(
      text: text,
      fullText: text,
      baseOffset: text.startIndex,
      audioSamples: audioSamples,
      sampleRate: sampleRate
    )
  }

  public func align(
    text: String,
    fullText: String,
    baseOffset: String.Index,
    audioSamples: [Float],
    sampleRate: Int
  ) -> [HighlightedWord] {
    // Build alignment matrix from attention data
    guard let alignmentMatrix = buildAlignmentMatrix() else {
      // Return empty if attention extraction failed
      return []
    }

    // Run DTW to get optimal alignment path
    let numSpeechTokens = alignmentData.speechTokenCount
    let numTextTokens = alignmentData.textTokenCount

    guard numSpeechTokens > 0, numTextTokens > 0 else {
      return []
    }

    // Convert alignment matrix to cost matrix using Accelerate for vectorization
    // cost = -attention + diagonal_penalty
    // This encourages proportional alignment (speech_pos / total_speech ≈ text_pos / total_text)
    var costMatrix = [Float](repeating: 0, count: numTextTokens * numSpeechTokens)
    let diagonalWeight: Float = 0.3

    // Pre-compute speech ratios vector once (0/(M-1), 1/(M-1), ..., (M-1)/(M-1))
    var speechRatios = [Float](repeating: 0, count: numSpeechTokens)
    let speechScale = 1.0 / Float(max(1, numSpeechTokens - 1))
    vDSP_vramp([Float(0)], [speechScale], &speechRatios, 1, vDSP_Length(numSpeechTokens))

    // Process each text row with vectorized operations
    for textIdx in 0 ..< numTextTokens {
      let expectedRatio = Float(textIdx) / Float(max(1, numTextTokens - 1))
      let rowOffset = textIdx * numSpeechTokens

      // Compute |expectedRatio - speechRatios[j]| for all j using vDSP
      var diagonalPenalties = [Float](repeating: 0, count: numSpeechTokens)

      // diagonalPenalties = speechRatios - expectedRatio
      var negExpected = -expectedRatio
      vDSP_vsadd(speechRatios, 1, &negExpected, &diagonalPenalties, 1, vDSP_Length(numSpeechTokens))

      // diagonalPenalties = |diagonalPenalties|
      vDSP_vabs(diagonalPenalties, 1, &diagonalPenalties, 1, vDSP_Length(numSpeechTokens))

      // diagonalPenalties *= diagonalWeight
      var weight = diagonalWeight
      vDSP_vsmul(diagonalPenalties, 1, &weight, &diagonalPenalties, 1, vDSP_Length(numSpeechTokens))

      // costMatrix[row] = -alignmentMatrix[row] + diagonalPenalties
      alignmentMatrix.withUnsafeBufferPointer { alignBuf in
        // First negate the alignment row
        var negOne: Float = -1.0
        vDSP_vsmul(alignBuf.baseAddress! + rowOffset, 1, &negOne, &costMatrix[rowOffset], 1, vDSP_Length(numSpeechTokens))
      }

      // Then add diagonal penalties
      costMatrix.withUnsafeMutableBufferPointer { costBuf in
        vDSP_vadd(costBuf.baseAddress! + rowOffset, 1,
                  diagonalPenalties, 1,
                  costBuf.baseAddress! + rowOffset, 1, vDSP_Length(numSpeechTokens))
      }
    }

    // Run DTW
    let (textIndices, speechIndices) = costMatrix.withUnsafeBufferPointer { buffer in
      alignmentDtw(buffer, rows: numTextTokens, cols: numSpeechTokens)
    }

    // Map text tokens to words and compute timings
    return mapTokensToWords(
      textIndices: textIndices,
      speechIndices: speechIndices,
      text: text,
      fullText: fullText,
      baseOffset: baseOffset,
      totalSamples: audioSamples.count,
      sampleRate: sampleRate
    )
  }

  // MARK: - Private Methods

  /// Build alignment matrix (T_text, T_speech) from pre-extracted attention.
  /// Attention is already averaged across heads during generation.
  private func buildAlignmentMatrix() -> [Float]? {
    guard alignmentData.config != nil else { return nil }
    guard !alignmentData.textAttention.isEmpty else { return nil }

    let numTextTokens = alignmentData.textTokenCount
    let numSpeechTokens = alignmentData.speechTokenCount

    guard numTextTokens > 0, numSpeechTokens > 0 else { return nil }

    // Initialize matrix (text tokens x speech tokens) in row-major order
    var matrix = [Float](repeating: 0, count: numTextTokens * numSpeechTokens)

    // Copy pre-extracted attention directly (already averaged across heads)
    for (speechIdx, textAttn) in alignmentData.textAttention.enumerated() {
      guard speechIdx < numSpeechTokens else { continue }
      for (textIdx, attn) in textAttn.enumerated() where textIdx < numTextTokens {
        matrix[textIdx * numSpeechTokens + speechIdx] = attn
      }
    }

    return matrix
  }

  /// Map DTW alignment path to word-level timings.
  private func mapTokensToWords(
    textIndices: [Int],
    speechIndices: [Int],
    text: String,
    fullText: String,
    baseOffset: String.Index,
    totalSamples: Int,
    sampleRate: Int
  ) -> [HighlightedWord] {
    // Group consecutive text tokens by their word
    // A word boundary is detected when token text starts with a space or is punctuation

    var wordGroups: [(wordText: String, charRange: Range<String.Index>, speechStart: Int, speechEnd: Int)] = []

    var currentWord = ""
    var currentWordStart: String.Index?
    var currentSpeechStart: Int?
    var currentSpeechEnd = 0
    var charPosition = text.startIndex

    // Build mapping from alignment path
    var tokenToSpeechRange: [Int: (start: Int, end: Int)] = [:]
    for (textIdx, speechIdx) in zip(textIndices, speechIndices) {
      if let existing = tokenToSpeechRange[textIdx] {
        tokenToSpeechRange[textIdx] = (start: existing.start, end: max(existing.end, speechIdx))
      } else {
        tokenToSpeechRange[textIdx] = (start: speechIdx, end: speechIdx)
      }
    }

    // Process each text token
    for (tokenIdx, tokenText) in tokenTexts.enumerated() {
      guard tokenIdx < alignmentData.textTokenCount else { break }

      let trimmedToken = tokenText.trimmingCharacters(in: .whitespaces)
      let startsWithSpace = tokenText.hasPrefix(" ") || tokenText.hasPrefix("Ġ") // GPT-2 uses Ġ for space

      // Get speech range for this token
      let speechRange = tokenToSpeechRange[tokenIdx] ?? (start: currentSpeechEnd, end: currentSpeechEnd)

      if startsWithSpace, !currentWord.isEmpty {
        // Complete previous word
        if let wordStart = currentWordStart, let speechStart = currentSpeechStart {
          let wordEnd = charPosition
          wordGroups.append((
            wordText: currentWord,
            charRange: wordStart ..< wordEnd,
            speechStart: speechStart,
            speechEnd: currentSpeechEnd
          ))
        }

        // Start new word
        currentWord = trimmedToken
        // Advance past whitespace
        while charPosition < text.endIndex, text[charPosition].isWhitespace {
          charPosition = text.index(after: charPosition)
        }
        currentWordStart = charPosition
        currentSpeechStart = speechRange.start
      } else {
        // Continue current word
        if currentWordStart == nil {
          currentWordStart = charPosition
          currentSpeechStart = speechRange.start
        }
        currentWord += trimmedToken
      }

      currentSpeechEnd = speechRange.end

      // Advance character position
      for _ in trimmedToken {
        if charPosition < text.endIndex {
          charPosition = text.index(after: charPosition)
        }
      }
    }

    // Complete final word
    if !currentWord.isEmpty, let wordStart = currentWordStart, let speechStart = currentSpeechStart {
      wordGroups.append((
        wordText: currentWord,
        charRange: wordStart ..< charPosition,
        speechStart: speechStart,
        speechEnd: currentSpeechEnd
      ))
    }

    // Convert to HighlightedWord with proper timing
    let totalDuration = Double(totalSamples) / Double(sampleRate)
    let totalSpeechTokens = alignmentData.speechTokenCount

    return wordGroups.map { group in
      let startTime = (Double(group.speechStart) / Double(max(1, totalSpeechTokens))) * totalDuration
      let endTime = (Double(group.speechEnd + 1) / Double(max(1, totalSpeechTokens))) * totalDuration

      // Offset charRange if needed
      let adjustedRange: Range<String.Index>
      if baseOffset != fullText.startIndex {
        let charOffset = fullText.distance(from: fullText.startIndex, to: baseOffset)
        let startOffset = text.distance(from: text.startIndex, to: group.charRange.lowerBound)
        let endOffset = text.distance(from: text.startIndex, to: group.charRange.upperBound)
        let adjustedStart = fullText.index(fullText.startIndex, offsetBy: charOffset + startOffset)
        let adjustedEnd = fullText.index(fullText.startIndex, offsetBy: charOffset + endOffset)
        adjustedRange = adjustedStart ..< adjustedEnd
      } else {
        adjustedRange = group.charRange
      }

      return HighlightedWord(
        word: Self.cleanupByteLevelText(group.wordText),
        start: startTime,
        end: endTime,
        charRange: adjustedRange
      )
    }
  }

  /// Cleans up byte-level BPE artifacts from decoded text.
  ///
  /// GPT-2 uses byte-level BPE which maps bytes to unicode characters.
  /// When decoding individual tokens, some characters may not be properly
  /// converted back. This function normalizes common problematic characters.
  private static func cleanupByteLevelText(_ text: String) -> String {
    var result = text

    // GPT-2 byte-level BPE mappings that may not decode correctly:
    // - Byte 32 (space) -> U+0120 (Ġ)
    // - Various punctuation and special characters

    // Remove the GPT-2 space marker if present
    result = result.replacingOccurrences(of: "\u{0120}", with: "")

    // Normalize curly quotes to straight quotes
    result = result.replacingOccurrences(of: "\u{2018}", with: "'") // Left single quote
    result = result.replacingOccurrences(of: "\u{2019}", with: "'") // Right single quote
    result = result.replacingOccurrences(of: "\u{201C}", with: "\"") // Left double quote
    result = result.replacingOccurrences(of: "\u{201D}", with: "\"") // Right double quote

    // Handle potential byte-level encoding issues for apostrophe
    // The apostrophe (byte 39 / 0x27) should decode correctly,
    // but if there are issues, try common replacements
    result = result.replacingOccurrences(of: "\u{00A7}", with: "'") // Section sign (sometimes misused)
    result = result.replacingOccurrences(of: "\u{00B4}", with: "'") // Acute accent
    result = result.replacingOccurrences(of: "\u{0060}", with: "'") // Grave accent

    // Handle Unicode replacement character (appears when UTF-8 decoding fails)
    // Remove it since the actual character is likely already present
    result = result.replacingOccurrences(of: "\u{FFFD}", with: "")

    // GPT-2 byte-level BPE maps non-printable bytes to unicode characters:
    // - Bytes 0-31 -> U+0100-U+011F (control characters)
    // - Byte 32 (space) -> U+0120 (Ġ) - but only at word start, usually trimmed
    // - Bytes 127-160 -> U+0121-U+0142
    // - Byte 173 -> U+0143
    //
    // If these appear in the final text, the decoder didn't convert them properly.
    // Map them back to spaces (most likely intended character) or remove if control.
    result = result.replacingOccurrences(of: "\u{0120}", with: " ") // Ġ -> space

    // Remove control character mappings (bytes 0-31, 127)
    for codePoint in 0x0100 ... 0x011F {
      if let scalar = Unicode.Scalar(codePoint) {
        result = result.replacingOccurrences(of: String(scalar), with: "")
      }
    }
    // Byte 127 (DEL) -> U+0121
    result = result.replacingOccurrences(of: "\u{0121}", with: "")

    // Trim any leading/trailing whitespace that might have been introduced
    result = result.trimmingCharacters(in: .whitespaces)

    return result
  }
}

// MARK: - DTW Implementation

/// Dynamic Time Warping for text-to-speech alignment.
/// Cost matrix should be (T_text, T_speech) where lower values = better alignment.
private func alignmentDtw(
  _ costMatrix: UnsafeBufferPointer<Float>,
  rows N: Int,
  cols M: Int
) -> (textIndices: [Int], speechIndices: [Int]) {
  let costRows = N + 1
  let costCols = M + 1
  var cost = [Float](repeating: .infinity, count: costRows * costCols)
  var trace = [Int8](repeating: -1, count: costRows * costCols)

  @inline(__always)
  func idx(_ i: Int, _ j: Int) -> Int { i * costCols + j }

  cost[idx(0, 0)] = 0

  // Main DTW loop
  for j in 1 ... M {
    for i in 1 ... N {
      let c0 = cost[idx(i - 1, j - 1)] // diagonal
      let c1 = cost[idx(i - 1, j)] // vertical
      let c2 = cost[idx(i, j - 1)] // horizontal

      // Prefer diagonal (0) for alignment, then horizontal (2) to extend current text token
      let (c, t): (Float, Int8)
      if c0 <= c1, c0 <= c2 {
        // Diagonal: both text and speech advance - preferred for alignment
        (c, t) = (c0, 0)
      } else if c2 <= c1 {
        // Horizontal: speech advances, text stays - extends current word
        (c, t) = (c2, 2)
      } else {
        // Vertical: text advances, speech stays - skip text token
        (c, t) = (c1, 1)
      }

      cost[idx(i, j)] = costMatrix[(i - 1) * M + (j - 1)] + c
      trace[idx(i, j)] = t
    }
  }

  // Boundary conditions
  for j in 0 ..< costCols {
    trace[idx(0, j)] = 2
  }
  for i in 0 ..< costRows {
    trace[idx(i, 0)] = 1
  }

  // Backtrace
  var i = costRows - 1
  var j = costCols - 1
  var result: [(Int, Int)] = []
  result.reserveCapacity(i + j)

  while i > 0 || j > 0 {
    result.append((i - 1, j - 1))
    switch trace[idx(i, j)] {
      case 0: i -= 1; j -= 1
      case 1: i -= 1
      case 2: j -= 1
      default:
        if i > 0 { i -= 1 }
        if j > 0 { j -= 1 }
    }
  }

  result.reverse()
  return (result.map { $0.0 }, result.map { $0.1 })
}
