// Copyright © Anthony DePasquale

import SwiftUI

struct InputSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    VStack(spacing: 16) {
      // Text Input - use HighlightedTextView for Chatterbox Turbo
      if appState.selectedProvider == .chatterboxTurbo {
        HighlightedTextView()
      } else {
        TextInputView(
          text: $appState.inputText,
          isDisabled: appState.isGenerating,
        )
      }

      // Auto-play toggle
      Toggle("Auto-play", isOn: $appState.autoPlay)
        .toggleStyle(.switch)

      // Generate Button
      GenerateButton(
        isLoading: appState.isModelLoading,
        isGenerating: appState.isGenerating,
        canGenerate: !appState.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        onGenerate: {
          dismissKeyboard()
          Task {
            if !appState.isLoaded {
              try? await appState.loadEngine()
            }
            await appState.generate()
          }
        },
        onStop: {
          Task { await appState.stop() }
        },
      )

      // Streaming option
      VStack {
        Button {
          dismissKeyboard()
          Task {
            if !appState.isLoaded {
              try? await appState.loadEngine()
            }

            // Use word highlighting for Chatterbox Turbo if enabled
            if appState.selectedProvider == .chatterboxTurbo && appState.highlightingEnabled {
              await appState.generateStreamingWithHighlighting()
            } else {
              await appState.generateStreaming()
            }
          }
        } label: {
          HStack(spacing: 8) {
            Image(systemName: appState.selectedProvider == .chatterboxTurbo && appState.highlightingEnabled
              ? "waveform.and.person.filled" : "waveform.path")
            Text("Stream")
          }
          .padding(.vertical, 4)
        }
        .buttonStyle(.glass)
        .disabled(appState.isModelLoading || appState.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        Text(appState.streamingGranularity.shortDescription)
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
  }

  private func dismissKeyboard() {
    #if os(iOS)
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    #endif
  }
}
