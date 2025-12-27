// Copyright © Sachin Desai

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
