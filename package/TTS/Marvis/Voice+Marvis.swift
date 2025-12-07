import Foundation

public extension Voice {
  /// Create a Voice for Marvis conversational voices
  static func fromMarvisID(_ id: String) -> Voice {
    let displayName: String
    if id.hasPrefix("conversational_") {
      let voiceType = id.dropFirst("conversational_".count)
      displayName = "Conversational \(voiceType.uppercased())"
    } else {
      displayName = id.capitalized
    }
    return Voice(id: id, displayName: displayName, languageCode: "en-US")
  }
}
