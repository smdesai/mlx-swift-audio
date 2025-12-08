import MLXAudio
import SwiftUI

struct SettingsSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    VStack(alignment: .leading, spacing: 20) {
      // Speed (Kokoro only)
      if appState.selectedProvider.supportsSpeed {
        SpeedSliderView(
          speed: $appState.speed,
          isDisabled: appState.isGenerating,
        )
      }

      // Quality Level (Marvis only)
      if appState.selectedProvider.supportsQualityLevels {
        QualityLevelSection()
      }

      // Reference Audio (OuteTTS)
      if appState.selectedProvider == .outetts {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Speaker Profile",
            infoText: "Create a speaker profile from reference audio with automatic transcription. Use 5-15 seconds of clear speech.",
            loadingText: "Processing audio...",
          ),
          statusDescription: appState.outeTTSSpeakerDescription,
          isLoaded: appState.isOuteTTSSpeakerLoaded,
          onLoadDefault: {
            try await appState.loadDefaultOuteTTSSpeaker()
          },
          onLoadFromFile: { url in
            try await appState.createOuteTTSSpeaker(from: url)
          },
          onLoadFromURL: { url in
            try await appState.createOuteTTSSpeaker(from: url)
          },
        )
      }

      // Reference Audio (Chatterbox)
      if appState.selectedProvider == .chatterbox {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Reference Audio",
            infoText: "Chatterbox uses reference audio to match voice characteristics. For best results, use 10+ seconds of clear speech.",
          ),
          statusDescription: appState.chatterboxReferenceAudioDescription,
          isLoaded: appState.isChatterboxReferenceAudioLoaded,
          onLoadDefault: {
            try await appState.prepareDefaultChatterboxReferenceAudio()
          },
          onLoadFromFile: { url in
            try await appState.prepareChatterboxReferenceAudio(from: url)
          },
          onLoadFromURL: { url in
            try await appState.prepareChatterboxReferenceAudio(from: url)
          },
        )

        // Emotion exaggeration slider
        VStack(alignment: .leading, spacing: 4) {
          HStack {
            Text("Emotion")
            Spacer()
            Text(String(format: "%.1f", appState.chatterboxExaggeration))
              .foregroundStyle(.secondary)
          }
          Slider(value: $appState.chatterboxExaggeration, in: 0 ... 2, step: 0.1)
        }
      }

      // Provider Status Message
      if !appState.selectedProvider.statusMessage.isEmpty {
        Text(appState.selectedProvider.statusMessage)
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
  }
}

/// Quality level section for Marvis
private struct QualityLevelSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    HStack {
      Picker("Quality", selection: $appState.marvisQualityLevel) {
        ForEach(MarvisEngine.QualityLevel.allCases, id: \.self) { level in
          Text("\(level.rawValue.capitalized) (\(level.codebookCount) codebooks)")
            .tag(level)
        }
      }
      .pickerStyle(.menu)
    }
  }
}
