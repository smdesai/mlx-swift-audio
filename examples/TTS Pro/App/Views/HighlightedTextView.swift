// Copyright © Sachin Desai

import MLXAudio
import SwiftUI

// MARK: - Highlight Theme

/// Preset theme options for word highlighting
public enum HighlightThemePreset: String, CaseIterable, Sendable {
  case `default` = "Default"
  case highContrast = "High Contrast"
  case subtle = "Subtle"
  case ocean = "Ocean"
  case custom = "Custom"

  /// Get the theme for this preset
  public var theme: HighlightTheme {
    switch self {
      case .default:
        HighlightTheme(spokenColor: .green, currentWordColor: .yellow, upcomingColor: .primary)
      case .highContrast:
        HighlightTheme(spokenColor: .gray, currentWordColor: .white, upcomingColor: .primary)
      case .subtle:
        HighlightTheme(spokenColor: Color.primary.opacity(0.5), currentWordColor: .accentColor, upcomingColor: .primary)
      case .ocean:
        HighlightTheme(spokenColor: .cyan, currentWordColor: .orange, upcomingColor: Color.primary.opacity(0.8))
      case .custom:
        // Custom returns default; actual custom colors are stored separately
        HighlightTheme(spokenColor: .green, currentWordColor: .yellow, upcomingColor: .primary)
    }
  }
}

/// Customizable color theme for word highlighting during TTS playback
public struct HighlightTheme: Sendable, Equatable {
  /// Color for words that have already been spoken
  public let spokenColor: Color

  /// Color for the currently spoken word
  public let currentWordColor: Color

  /// Color for words that haven't been spoken yet
  public let upcomingColor: Color

  public init(
    spokenColor: Color = .green,
    currentWordColor: Color = .yellow,
    upcomingColor: Color = .primary
  ) {
    self.spokenColor = spokenColor
    self.currentWordColor = currentWordColor
    self.upcomingColor = upcomingColor
  }

  // MARK: - Preset Themes

  /// Default theme with green spoken, yellow current
  public static let `default` = HighlightTheme()
}

// MARK: - Highlighted Text View

/// A text view that highlights words as they are spoken during TTS playback
///
/// Has two modes:
/// - Editing mode: Shows a TextEditor for user input
/// - Highlighting mode: Shows a read-only view with word highlighting during playback
struct HighlightedTextView: View {
  @Environment(AppState.self) private var appState
  @State private var isEditing = false

  /// Whether to show the header with character count and clear button
  var showHeader: Bool = true

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      // Header (optional)
      if showHeader {
        @Bindable var appState = appState
        HStack {
          Text("Text Input")
            .font(.headline)

          // Highlighting toggle button
          Button {
            appState.highlightingEnabled.toggle()
          } label: {
            Image(systemName: appState.highlightingEnabled ? "highlighter" : "text.alignleft")
              .font(.caption)
          }
          .buttonStyle(.bordered)
          .controlSize(.small)
          .tint(appState.highlightingEnabled ? .accentColor : .secondary)
          .help(appState.highlightingEnabled ? "Highlighting enabled" : "Highlighting disabled")

          Spacer()

          Text("\(appState.inputText.count) / 5000")
            .font(.caption)
            .foregroundStyle(.secondary)

          if !appState.inputText.isEmpty, !appState.isHighlighting {
            Button {
              appState.inputText = ""
              isEditing = false
            } label: {
              Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
          }
        }
      }

      // Content: highlighting view, editor, or placeholder
      if appState.isHighlighting {
        highlightingModeView
      } else if isEditing {
        editorView
      } else if appState.inputText.isEmpty {
        placeholderView
      } else {
        // Non-empty text but not editing - show read-only view that's tappable to edit
        readOnlyTextView
      }
    }
  }

  // MARK: - Highlighting Mode View

  /// Read-only view showing text with word highlighting
  private var highlightingModeView: some View {
    ScrollView {
      HighlightableAttributedString(
        text: appState.highlightingDisplayText.isEmpty ? appState.inputText : appState.highlightingDisplayText,
        wordTimings: appState.wordTimings,
        currentWordIndex: appState.currentHighlightedWordIndex,
        theme: appState.highlightTheme
      )
      .padding(12)
      .frame(maxWidth: .infinity, alignment: .leading)
    }
    .frame(minHeight: 120, maxHeight: 300)
    .padding(8)
    .background(.background.secondary)
    .clipShape(RoundedRectangle(cornerRadius: 8))
    .overlay(
      RoundedRectangle(cornerRadius: 8)
        .stroke(Color.accentColor.opacity(0.5), lineWidth: 2)
    )
    .onTapGesture {
      // Tap to exit highlighting mode and return to editing
      appState.exitHighlightingMode()
      isEditing = true
    }
  }

  // MARK: - Editor View

  private var editorView: some View {
    @Bindable var appState = appState
    return TextEditor(text: $appState.inputText)
      .font(.body)
      .scrollContentBackground(.hidden)
      .foregroundStyle(.primary)
      .frame(minHeight: 120, maxHeight: 300)
      .padding(8)
      .background(.background.secondary)
      .clipShape(RoundedRectangle(cornerRadius: 8))
      .overlay(
        RoundedRectangle(cornerRadius: 8)
          .stroke(.separator, lineWidth: 1)
      )
      .onSubmit {
        isEditing = false
      }
      .onChange(of: appState.inputText) { _, _ in
        // Keep editing active while user types
      }
  }

  // MARK: - Read-Only Text View

  /// Shows the current text in a read-only view that's tappable to edit
  private var readOnlyTextView: some View {
    ScrollView {
      Text(appState.inputText)
        .font(.body)
        .foregroundStyle(.primary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
    }
    .frame(minHeight: 120, maxHeight: 300)
    .padding(8)
    .background(.background.secondary)
    .clipShape(RoundedRectangle(cornerRadius: 8))
    .overlay(
      RoundedRectangle(cornerRadius: 8)
        .stroke(.separator, lineWidth: 1)
    )
    .contentShape(Rectangle())
    .onTapGesture {
      isEditing = true
    }
  }

  // MARK: - Placeholder View

  private var placeholderView: some View {
    ZStack {
      // Transparent rectangle to maintain consistent height
      Color.clear
        .frame(height: 120)

      Text("Enter text to synthesize...")
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    .frame(maxHeight: 300)
    .padding(8)
    .background(.background.secondary)
    .clipShape(RoundedRectangle(cornerRadius: 8))
    .overlay(
      RoundedRectangle(cornerRadius: 8)
        .stroke(.separator, lineWidth: 1)
    )
    .contentShape(Rectangle())
    .onTapGesture {
      isEditing = true
    }
  }
}

// MARK: - Highlightable Attributed String View

/// Renders text with word-by-word highlighting using AttributedString
/// Uses index-based lookup for efficient highlighting
struct HighlightableAttributedString: View {
  let text: String
  let wordTimings: [HighlightedWord]
  let currentWordIndex: Int?
  let theme: HighlightTheme

  var body: some View {
    Text(buildHighlightedString())
      .font(.body)
      .textSelection(.enabled)
  }

  private func buildHighlightedString() -> AttributedString {
    guard !wordTimings.isEmpty, let currentIdx = currentWordIndex, currentIdx < wordTimings.count else {
      return AttributedString(text)
    }

    let currentTiming = wordTimings[currentIdx]
    let currentWordRange = currentTiming.charRange

    // Validate range bounds
    guard currentWordRange.lowerBound >= text.startIndex,
          currentWordRange.upperBound <= text.endIndex
    else {
      return AttributedString(text)
    }

    // Build the attributed string by position
    var attributedString = AttributedString()

    // Text before current word (already spoken)
    if currentWordRange.lowerBound > text.startIndex {
      var beforePart = AttributedString(String(text[..<currentWordRange.lowerBound]))
      beforePart.foregroundColor = theme.spokenColor
      attributedString.append(beforePart)
    }

    // Current word
    var wordPart = AttributedString(String(text[currentWordRange]))
    wordPart.foregroundColor = theme.currentWordColor
    attributedString.append(wordPart)

    // Text after current word (upcoming)
    if currentWordRange.upperBound < text.endIndex {
      var afterPart = AttributedString(String(text[currentWordRange.upperBound...]))
      afterPart.foregroundColor = theme.upcomingColor
      attributedString.append(afterPart)
    }

    return attributedString
  }
}

#Preview {
  @Previewable @State var appState = AppState()

  HighlightedTextView()
    .environment(appState)
}
