import Foundation

/// Available TTS providers
public enum TTSProvider: String, CaseIterable, Identifiable, Sendable {
  case kokoro
  case orpheus
  case marvis
  case outetts
  case chatterbox

  public var id: String { rawValue }

  /// Canonical display name with proper casing/branding
  public var displayName: String {
    switch self {
      case .outetts: "OuteTTS"
      case .chatterbox: "Chatterbox"
      default: rawValue.capitalized
    }
  }

  // MARK: - Audio Properties

  /// Sample rate for this provider's audio output (Hz)
  public var sampleRate: Int {
    switch self {
      default: 24000
    }
  }

  // MARK: - Feature Flags

  /// Whether this provider supports speed adjustment
  public var supportsSpeed: Bool {
    switch self {
      case .kokoro: true
      default: false
    }
  }

  /// Whether this provider supports emotional expressions
  public var supportsExpressions: Bool {
    switch self {
      case .orpheus: true
      default: false
    }
  }

  /// Whether this provider supports quality level selection
  public var supportsQualityLevels: Bool {
    switch self {
      case .marvis: true
      default: false
    }
  }

  /// Whether this provider supports reference audio
  public var supportsReferenceAudio: Bool {
    switch self {
      case .outetts, .chatterbox: true
      default: false
    }
  }
}
