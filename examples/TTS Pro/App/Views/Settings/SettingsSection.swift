// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Error for file access failures
private struct FileAccessError: LocalizedError {
  let message: String
  var errorDescription: String? { message }
}

struct SettingsSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    VStack(alignment: .leading, spacing: 24) {
      // Speed (Kokoro only)
      if appState.selectedProvider.supportsSpeed {
        SliderRow(
          label: "Speed",
          value: $appState.speed,
          range: 0.5 ... 2.0,
          format: "%.1fx"
        )
      }

      // Quality Level (Marvis only)
      if appState.selectedProvider.supportsQualityLevels {
        QualitySection()
      }

      // Reference Audio (OuteTTS)
      if appState.selectedProvider == .outetts {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Speaker Profile",
            infoText: "5-15 seconds of clear speech"
          ),
          statusDescription: appState.outeTTSSpeakerDescription,
          isLoaded: appState.isOuteTTSSpeakerLoaded,
          onLoadDefault: { try await appState.loadDefaultOuteTTSSpeaker() },
          onLoadFromFile: { try await appState.createOuteTTSSpeaker(from: $0) },
          onLoadFromURL: { try await appState.createOuteTTSSpeaker(from: $0) }
        )
      }

      // Reference Audio (Chatterbox)
      if appState.selectedProvider == .chatterbox {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Reference Audio",
            infoText: "10+ seconds of clear speech"
          ),
          statusDescription: appState.chatterboxReferenceAudioDescription,
          isLoaded: appState.isChatterboxReferenceAudioLoaded,
          onLoadDefault: { try await appState.prepareDefaultChatterboxReferenceAudio() },
          onLoadFromFile: { try await appState.prepareChatterboxReferenceAudio(from: $0) },
          onLoadFromURL: { try await appState.prepareChatterboxReferenceAudio(from: $0) }
        )

        SliderRow(
          label: "Emotion",
          value: $appState.chatterboxExaggeration,
          range: 0 ... 2,
          format: "%.1f"
        )
      }

      // Reference Audio (Chatterbox Turbo)
      if appState.selectedProvider == .chatterboxTurbo {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Reference Audio",
            infoText: "10+ seconds of clear speech"
          ),
          statusDescription: appState.chatterboxTurboReferenceAudioDescription,
          isLoaded: appState.isChatterboxTurboReferenceAudioLoaded,
          onLoadDefault: { try await appState.prepareDefaultChatterboxTurboReferenceAudio() },
          onLoadFromFile: { try await appState.prepareChatterboxTurboReferenceAudio(from: $0) },
          onLoadFromURL: { try await appState.prepareChatterboxTurboReferenceAudio(from: $0) }
        )

        HighlightThemeSection()
      }

      // Speaker (CosyVoice2)
      if appState.selectedProvider == .cosyVoice2 {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Speaker",
            infoText: "5-30 seconds of clear speech"
          ),
          statusDescription: appState.cosyVoice2SpeakerDescription,
          isLoaded: appState.isCosyVoice2SpeakerLoaded,
          onLoadDefault: { try await appState.prepareDefaultCosyVoice2Speaker() },
          onLoadFromFile: { try await appState.prepareCosyVoice2Speaker(from: $0) },
          onLoadFromURL: { try await appState.prepareCosyVoice2Speaker(from: $0) }
        )

        ModeSection(
          title: "Mode",
          selection: $appState.cosyVoice2GenerationMode,
          description: appState.cosyVoice2GenerationMode.description
        )

        if appState.cosyVoice2GenerationMode == .instruct {
          TextInputRow(label: "Instructions", text: $appState.cosyVoice2InstructText)
        }

        if appState.cosyVoice2GenerationMode == .voiceConversion {
          SourceAudioRow(
            description: appState.cosyVoice2SourceAudioDescription,
            isLoaded: appState.isCosyVoice2SourceAudioLoaded,
            onLoad: { try await appState.prepareCosyVoice2SourceAudio(from: $0) }
          )
        }

        if appState.cosyVoice2GenerationMode == .zeroShot, let speaker = appState.cosyVoice2Speaker {
          TranscriptionRow(transcription: speaker.transcription)
        }
      }

      // Speaker (CosyVoice3)
      if appState.selectedProvider == .cosyVoice3 {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Speaker",
            infoText: "5-30 seconds of clear speech"
          ),
          statusDescription: appState.cosyVoice3SpeakerDescription,
          isLoaded: appState.isCosyVoice3SpeakerLoaded,
          onLoadDefault: { try await appState.prepareDefaultCosyVoice3Speaker() },
          onLoadFromFile: { try await appState.prepareCosyVoice3Speaker(from: $0) },
          onLoadFromURL: { try await appState.prepareCosyVoice3Speaker(from: $0) }
        )

        ModeSection(
          title: "Mode",
          selection: $appState.cosyVoice3GenerationMode,
          description: appState.cosyVoice3GenerationMode.description
        )

        if appState.cosyVoice3GenerationMode == .instruct {
          TextInputRow(label: "Instructions", text: $appState.cosyVoice3InstructText)
        }

        if appState.cosyVoice3GenerationMode == .voiceConversion {
          SourceAudioRow(
            description: appState.cosyVoice3SourceAudioDescription,
            isLoaded: appState.isCosyVoice3SourceAudioLoaded,
            onLoad: { try await appState.prepareCosyVoice3SourceAudio(from: $0) }
          )
        }

        if appState.cosyVoice3GenerationMode == .zeroShot, let speaker = appState.cosyVoice3Speaker {
          TranscriptionRow(transcription: speaker.transcription)
        }
      }

      // Status message
      if !appState.selectedProvider.statusMessage.isEmpty {
        Text(appState.selectedProvider.statusMessage)
          .font(.system(size: 13))
          .foregroundStyle(.white.opacity(0.5))
      }
    }
  }
}

// MARK: - Minimal Components

private struct SliderRow: View {
  let label: String
  @Binding var value: Float
  let range: ClosedRange<Float>
  let format: String

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text(label)
          .font(.system(size: 14, weight: .medium))
          .foregroundStyle(.white)
        Spacer()
        Text(String(format: format, value))
          .font(.system(size: 13, weight: .medium, design: .monospaced))
          .foregroundStyle(.white.opacity(0.6))
      }
      Slider(value: $value, in: range, step: 0.1)
        .tint(.white)
    }
  }
}

private struct QualitySection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    VStack(alignment: .leading, spacing: 8) {
      Text("Quality")
        .font(.system(size: 14, weight: .medium))
        .foregroundStyle(.white)

      Picker("Quality", selection: $appState.marvisQualityLevel) {
        ForEach(MarvisEngine.QualityLevel.allCases, id: \.self) { level in
          Text("\(level.rawValue.capitalized)")
            .tag(level)
        }
      }
      .pickerStyle(.segmented)
    }
  }
}

private struct ModeSection<T: Hashable & CaseIterable & RawRepresentable>: View where T.RawValue == String {
  let title: String
  @Binding var selection: T
  let description: String

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(title)
        .font(.system(size: 14, weight: .medium))
        .foregroundStyle(.white)

      Picker(title, selection: $selection) {
        ForEach(Array(T.allCases), id: \.self) { mode in
          Text(mode.rawValue).tag(mode)
        }
      }
      .pickerStyle(.segmented)

      Text(description)
        .font(.system(size: 12))
        .foregroundStyle(.white.opacity(0.5))
    }
  }
}

private struct TextInputRow: View {
  let label: String
  @Binding var text: String

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(label)
        .font(.system(size: 14, weight: .medium))
        .foregroundStyle(.white)

      TextField("e.g., Speak slowly", text: $text)
        .textFieldStyle(.plain)
        .font(.system(size: 14))
        .padding(12)
        .background(Color.white.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
  }
}

private struct SourceAudioRow: View {
  let description: String
  let isLoaded: Bool
  let onLoad: (URL) async throws -> Void

  @State private var showFilePicker = false
  @State private var isLoading = false

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Source Audio")
        .font(.system(size: 14, weight: .medium))
        .foregroundStyle(.white)

      HStack {
        Text(description)
          .font(.system(size: 13))
          .foregroundStyle(isLoaded ? .white : .white.opacity(0.5))
        Spacer()
        Button("Choose") {
          showFilePicker = true
        }
        .font(.system(size: 13, weight: .medium))
        .foregroundStyle(.white)
        .disabled(isLoading)
      }
    }
    .fileImporter(isPresented: $showFilePicker, allowedContentTypes: [.audio], allowsMultipleSelection: false) { result in
      Task {
        if case let .success(urls) = result, let url = urls.first {
          guard url.startAccessingSecurityScopedResource() else { return }
          defer { url.stopAccessingSecurityScopedResource() }
          isLoading = true
          try? await onLoad(url)
          isLoading = false
        }
      }
    }
  }
}

private struct TranscriptionRow: View {
  let transcription: String?

  var body: some View {
    if let text = transcription {
      VStack(alignment: .leading, spacing: 8) {
        Text("Transcription")
          .font(.system(size: 14, weight: .medium))
          .foregroundStyle(.white)

        Text(text)
          .font(.system(size: 13))
          .foregroundStyle(.white.opacity(0.7))
          .lineLimit(3)
      }
    }
  }
}

// MARK: - Highlight Theme Section

private struct HighlightThemeSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    VStack(alignment: .leading, spacing: 12) {
      Text("Word Highlighting")
        .font(.system(size: 14, weight: .medium))
        .foregroundStyle(.white)

      // Theme preset picker with preview
      HStack {
        Text("Theme")
          .font(.system(size: 13))
          .foregroundStyle(.white.opacity(0.8))

        Spacer()

        Picker("Theme", selection: $appState.highlightThemePreset) {
          ForEach(HighlightThemePreset.allCases, id: \.self) { preset in
            Text(preset.rawValue).tag(preset)
          }
        }
        .pickerStyle(.menu)
        .tint(.white)
      }

      // Preview of current theme
      ThemePreviewRow(theme: appState.highlightTheme)

      // Custom color pickers (only shown when custom is selected)
      if appState.highlightThemePreset == .custom {
        VStack(alignment: .leading, spacing: 12) {
          ColorPickerRow(label: "Spoken", color: $appState.customSpokenColor)
          ColorPickerRow(label: "Current", color: $appState.customCurrentWordColor)
          ColorPickerRow(label: "Upcoming", color: $appState.customUpcomingColor)
        }
        .padding(.top, 4)
      }
    }
  }
}

private struct ThemePreviewRow: View {
  let theme: HighlightTheme

  var body: some View {
    HStack(spacing: 4) {
      Text("Already")
        .foregroundStyle(theme.spokenColor)
      Text("spoken")
        .foregroundStyle(theme.currentWordColor)
      Text("words")
        .foregroundStyle(theme.upcomingColor)
    }
    .font(.system(size: 13))
    .padding(.vertical, 8)
    .padding(.horizontal, 12)
    .frame(maxWidth: .infinity)
    .background(Color.black.opacity(0.3))
    .clipShape(RoundedRectangle(cornerRadius: 6))
  }
}

private struct ColorPickerRow: View {
  let label: String
  @Binding var color: Color

  var body: some View {
    HStack {
      Text(label)
        .font(.system(size: 13))
        .foregroundStyle(.white.opacity(0.8))

      Spacer()

      ColorPicker("", selection: $color, supportsOpacity: false)
        .labelsHidden()
    }
  }
}
