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

extension ForcedAligner {
  public func align(
    text: String,
    fullText: String,
    baseOffset: String.Index,
    audioSamples: [Float],
    sampleRate: Int
  ) -> [HighlightedWord] {
    // Default: use basic align (charRanges will be chunk-relative)
    return align(text: text, audioSamples: audioSamples, sampleRate: sampleRate)
  }
}

// MARK: - TokenBasedAligner

/// Lightweight forced aligner that estimates word timings using audio duration and text analysis.
///
/// This aligner uses a heuristic approach based on:
/// 1. Word length (longer words typically take longer to speak)
/// 2. Word complexity (more complex words are spoken slower)
/// 3. Common speech patterns (pause after punctuation, etc.)
///
/// This is a fast but approximate approach suitable for basic highlighting.
/// For production accuracy, consider a Whisper-based or Kaldi-based aligner.
public final class TokenBasedAligner: ForcedAligner, Sendable {

  // MARK: - Configuration

  /// Base speaking rate in characters per second (approximate for normal speech)
  private let baseCharactersPerSecond: Double = 15.0

  /// Minimum duration for any word (seconds)
  private let minWordDuration: Double = 0.08

  /// Maximum duration for any word (seconds)
  private let maxWordDuration: Double = 1.5

  /// Pause duration after punctuation (seconds)
  private let punctuationPauseDuration: Double = 0.15

  /// Words that are typically spoken quickly (short, common words)
  private let quickWords: Set<String> = [
    "a", "an", "the", "is", "it", "at", "in", "on", "to", "of",
    "and", "but", "or", "for", "so", "yet",
    "i", "you", "he", "she", "we", "they",
    "my", "your", "his", "her", "our", "their",
    "this", "that", "these", "those",
    "be", "am", "are", "was", "were", "been",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must"
  ]

  /// Words that are typically spoken slowly (complex, emphasized words)
  private let slowWords: Set<String> = [
    "however", "nevertheless", "furthermore", "moreover",
    "consequently", "accordingly", "therefore", "thus",
    "extraordinary", "magnificent", "phenomenal", "remarkable",
    "unfortunately", "surprisingly", "interestingly",
    "specifically", "particularly", "especially",
    "immediately", "instantly", "momentarily"
  ]

  // MARK: - Initialization

  public init() {}

  // MARK: - ForcedAligner

  public func align(
    text: String,
    audioSamples: [Float],
    sampleRate: Int
  ) -> [HighlightedWord] {
    return align(text: text, fullText: text, baseOffset: text.startIndex, audioSamples: audioSamples, sampleRate: sampleRate)
  }

  /// Align with full text context - offsets charRanges to point to full text positions.
  public func align(
    text: String,
    fullText: String,
    baseOffset: String.Index,
    audioSamples: [Float],
    sampleRate: Int
  ) -> [HighlightedWord] {
    // Calculate total audio duration
    let totalDuration = Double(audioSamples.count) / Double(sampleRate)

    // Extract words with their positions
    let words = extractWordsWithRanges(from: text)

    guard !words.isEmpty else {
      return []
    }

    // Calculate duration weights for each word
    let weights = words.map { calculateWeight(for: $0.word) }

    // Calculate total weight
    let totalWeight = weights.reduce(0, +)

    // Distribute time based on weights
    var timings: [HighlightedWord] = []
    var currentTime: TimeInterval = 0

    for (index, wordInfo) in words.enumerated() {
      let weight = weights[index]
      let rawDuration = (weight / totalWeight) * totalDuration

      // Apply min/max constraints
      let duration = max(minWordDuration, min(maxWordDuration, rawDuration))

      let end = currentTime + duration

      timings.append(HighlightedWord(
        word: wordInfo.word,
        start: currentTime,
        end: end,
        charRange: wordInfo.charRange
      ))

      // Move to next word
      currentTime = end

      // Add pause if this word ends with punctuation
      if wordInfo.hasTrailingPunctuation {
        currentTime += punctuationPauseDuration
      }
    }

    // Normalize to fit total duration (stretch or compress proportionally)
    timings = normalizeTimings(timings, toDuration: totalDuration)

    // Offset charRanges to full text if needed
    let offsetTimings: [HighlightedWord]
    if baseOffset != fullText.startIndex {
      // Calculate character offset from fullText.startIndex to baseOffset
      let charOffset = fullText.distance(from: fullText.startIndex, to: baseOffset)

      offsetTimings = timings.map { timing in
        // Offset the charRange by baseOffset
        let startOffset = text.distance(from: text.startIndex, to: timing.charRange.lowerBound)
        let endOffset = text.distance(from: text.startIndex, to: timing.charRange.upperBound)

        let adjustedStart = fullText.index(fullText.startIndex, offsetBy: charOffset + startOffset)
        let adjustedEnd = fullText.index(fullText.startIndex, offsetBy: charOffset + endOffset)

        return HighlightedWord(
          word: timing.word,
          start: timing.start,
          end: timing.end,
          charRange: adjustedStart..<adjustedEnd
        )
      }
    } else {
      offsetTimings = timings
    }

    return offsetTimings
  }

  // MARK: - Private Methods

  /// Extract words from text with their character ranges.
  private func extractWordsWithRanges(from text: String) -> [(word: String, charRange: Range<String.Index>, hasTrailingPunctuation: Bool)] {
    var result: [(word: String, charRange: Range<String.Index>, hasTrailingPunctuation: Bool)] = []

    let wordRange = text.rangeOfCharacter(from: .letters)
    guard let start = wordRange?.lowerBound else {
      return result
    }

    var currentWordStart = start
    var currentWordEnd = start
    var inWord = true

    var index = text.index(after: start)
    while index < text.endIndex {
      let char = text[index]

      if inWord {
        if char.isLetter || char == "'" {
          currentWordEnd = text.index(after: index)
        } else {
          // Word ended
          let wordRange = currentWordStart..<currentWordEnd
          let word = String(text[wordRange])

          // Check for trailing punctuation
          var lookahead = index
          var hasPunct = false
          while lookahead < text.endIndex {
            let c = text[lookahead]
            if c.isPunctuation {
              hasPunct = true
            } else if !c.isWhitespace {
              break
            }
            lookahead = text.index(after: lookahead)
          }

          result.append((word: word, charRange: wordRange, hasTrailingPunctuation: hasPunct))

          // Find next word start
          let searchRange = text.index(after: index)..<text.endIndex
          if let nextStart = text.rangeOfCharacter(from: .letters, range: searchRange)?.lowerBound {
            currentWordStart = nextStart
            currentWordEnd = nextStart
            index = nextStart
          } else {
            inWord = false
          }
        }
      }
      index = text.index(after: index)
    }

    // Handle last word
    if inWord && currentWordStart < text.endIndex {
      let wordRange = currentWordStart..<currentWordEnd
      let word = String(text[wordRange])
      result.append((word: word, charRange: wordRange, hasTrailingPunctuation: false))
    }

    return result
  }

  /// Calculate duration weight for a word.
  private func calculateWeight(for word: String) -> Double {
    let lowercase = word.lowercased()

    // Quick words get lower weight
    if quickWords.contains(lowercase) {
      return Double(word.count) * 0.6
    }

    // Slow words get higher weight
    if slowWords.contains(lowercase) {
      return Double(word.count) * 1.5
    }

    // Complex words (more syllables roughly indicated by length/vowel count)
    let vowelCount = word.filter { "aeiouy".contains($0.lowercased()) }.count
    let syllableEstimate = max(1, vowelCount)

    return Double(word.count) * (1.0 + Double(syllableEstimate - 1) * 0.2)
  }

  /// Normalize timings to fit exactly within total duration.
  private func normalizeTimings(_ timings: [HighlightedWord], toDuration totalDuration: TimeInterval) -> [HighlightedWord] {
    guard !timings.isEmpty else { return timings }

    let currentTotal = timings.last!.end

    guard currentTotal > 0 else { return timings }

    let scale = totalDuration / currentTotal

    var result: [HighlightedWord] = []
    var accumulatedTime: TimeInterval = 0

    for timing in timings {
      let originalDuration = timing.end - timing.start
      let scaledDuration = originalDuration * scale

      result.append(HighlightedWord(
        word: timing.word,
        start: accumulatedTime,
        end: accumulatedTime + scaledDuration,
        charRange: timing.charRange
      ))

      accumulatedTime += scaledDuration
    }

    return result
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
    return align(
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
      // Fall back to token-based alignment if attention extraction failed
      return TokenBasedAligner().align(
        text: text,
        fullText: fullText,
        baseOffset: baseOffset,
        audioSamples: audioSamples,
        sampleRate: sampleRate
      )
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
    for textIdx in 0..<numTextTokens {
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
    var currentWordStart: String.Index? = nil
    var currentSpeechStart: Int? = nil
    var currentSpeechEnd: Int = 0
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

      if startsWithSpace && !currentWord.isEmpty {
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
        while charPosition < text.endIndex && text[charPosition].isWhitespace {
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
        word: group.wordText,
        start: startTime,
        end: endTime,
        charRange: adjustedRange
      )
    }
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
      if c0 <= c1 && c0 <= c2 {
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
  for j in 0 ..< costCols { trace[idx(0, j)] = 2 }
  for i in 0 ..< costRows { trace[idx(i, 0)] = 1 }

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
