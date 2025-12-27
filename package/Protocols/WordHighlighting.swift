// Copyright © Sachin Desai

import Foundation

/// Word-level timing information for TTS text highlighting during playback.
///
/// Note: This is distinct from the Whisper STT `WordTiming` which includes
/// token IDs and probability scores for speech recognition.
public struct HighlightedWord: Sendable, Equatable {
  /// The word text
  public let word: String

  /// Start time in seconds (relative to the chunk)
  public let start: TimeInterval

  /// End time in seconds (relative to the chunk)
  public let end: TimeInterval

  /// Character range in original text (for mapping back)
  public let charRange: Range<String.Index>

  public init(word: String, start: TimeInterval, end: TimeInterval, charRange: Range<String.Index>) {
    self.word = word
    self.start = start
    self.end = end
    self.charRange = charRange
  }

  public static func == (lhs: HighlightedWord, rhs: HighlightedWord) -> Bool {
    lhs.word == rhs.word &&
    lhs.start == rhs.start &&
    lhs.end == rhs.end
  }
}

/// A chunk of audio data with word-level timing information
public struct AudioChunkWithTimings: Sendable {
  /// Raw audio samples
  public let samples: [Float]

  /// Sample rate in Hz (e.g., 24000)
  public let sampleRate: Int

  /// Processing time for this chunk
  public let processingTime: TimeInterval

  /// Word-level timing information
  public let wordTimings: [HighlightedWord]

  public init(samples: [Float], sampleRate: Int, processingTime: TimeInterval, wordTimings: [HighlightedWord]) {
    self.samples = samples
    self.sampleRate = sampleRate
    self.processingTime = processingTime
    self.wordTimings = wordTimings
  }
}
