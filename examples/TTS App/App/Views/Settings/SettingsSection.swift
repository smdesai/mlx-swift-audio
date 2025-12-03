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
          isDisabled: appState.isGenerating
        )
      }

      // Quality Level (Marvis only)
      if appState.selectedProvider.supportsQualityLevels {
        QualityLevelSection()
      }

      // Reference Audio (Chatterbox only)
      if appState.selectedProvider == .chatterbox {
        ReferenceAudioView()

        // Emotion exaggeration slider
        VStack(alignment: .leading, spacing: 4) {
          HStack {
            Text("Emotion")
            Spacer()
            Text(String(format: "%.1f", appState.chatterboxExaggeration))
              .foregroundStyle(.secondary)
          }
          Slider(value: $appState.chatterboxExaggeration, in: 0...2, step: 0.1)
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
        ForEach(MarvisTTS.QualityLevel.allCases, id: \.self) { level in
          Text("\(level.rawValue.capitalized) (\(level.codebookCount) codebooks)")
            .tag(level)
        }
      }
      .pickerStyle(.menu)
    }
  }
}
