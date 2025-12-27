// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

// MARK: - Main Layout

struct CompactLayoutView: View {
  @Environment(AppState.self) private var appState
  @State private var showSettings = false

  var body: some View {
    ZStack {
      // Simple dark background
      Color(red: 0.06, green: 0.06, blue: 0.08)
        .ignoresSafeArea()

      ScrollView {
        VStack(spacing: 32) {
          // Header
          HeaderView(showSettings: $showSettings)

          // Model & Voice
          ModelSection()

          // Text input
          TextInputSection()

          // Actions
          ActionsSection()

          // Output
//          OutputDisplay()
          OutputSection()
        }
        .padding(.horizontal, 24)
        .padding(.top, 20)
        .padding(.bottom, 60)
      }
    }
    .sheet(isPresented: $showSettings) {
      SettingsSheet()
    }
  }
}

// MARK: - Header

private struct HeaderView: View {
  @Binding var showSettings: Bool

  var body: some View {
    HStack {
      Text("TTS Pro")
        .font(.system(size: 28, weight: .bold, design: .rounded))
        .foregroundStyle(.white)

      Spacer()

      Button {
        showSettings = true
      } label: {
        Image(systemName: "gearshape")
          .font(.system(size: 20, weight: .medium))
          .foregroundStyle(.white.opacity(0.6))
      }
    }
  }
}

// MARK: - Model Section

private struct ModelSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Model")
        .font(.system(size: 13, weight: .medium))
        .foregroundStyle(.white.opacity(0.5))
        .textCase(.uppercase)
        .tracking(0.5)

      HStack(spacing: 16) {
        ProviderPickerView(
          selectedProvider: appState.selectedProvider,
          onSelect: { provider in
            Task { await appState.selectProvider(provider) }
          }
        )
        .tint(.white)

        VoicePickerView()
          .tint(.white)

        Spacer()
      }
    }
  }
}

// MARK: - Text Input

private struct TextInputSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState

    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text("Text")
          .font(.system(size: 13, weight: .medium))
          .foregroundStyle(.white.opacity(0.5))
          .textCase(.uppercase)
          .tracking(0.5)

        // Highlighting toggle (only for Chatterbox Turbo)
        if appState.selectedProvider == .chatterboxTurbo {
          Button {
            appState.highlightingEnabled.toggle()
          } label: {
            Image(systemName: appState.highlightingEnabled ? "highlighter" : "text.alignleft")
              .font(.system(size: 14))
              .foregroundStyle(appState.highlightingEnabled ? .white : .white.opacity(0.4))
          }
          .padding(.horizontal, 8)
          .padding(.vertical, 4)
          .background(appState.highlightingEnabled ? Color.accentColor.opacity(0.3) : Color.white.opacity(0.1))
          .clipShape(RoundedRectangle(cornerRadius: 6))
        }

        Spacer()

        Text("\(appState.inputText.count) / 5000")
          .font(.system(size: 12, weight: .medium, design: .monospaced))
          .foregroundStyle(.white.opacity(0.3))

        if !appState.inputText.isEmpty {
          Button {
            appState.inputText = ""
          } label: {
            Image(systemName: "xmark.circle.fill")
              .font(.system(size: 16))
              .foregroundStyle(.white.opacity(0.4))
          }
        }
      }

      // Use HighlightedTextView for Chatterbox Turbo, otherwise regular TextEditor
      if appState.selectedProvider == .chatterboxTurbo {
        HighlightedTextView(showHeader: false)
          .frame(minHeight: 100)
      } else {
        TextEditor(text: $appState.inputText)
          .font(.system(size: 16))
          .scrollContentBackground(.hidden)
          .foregroundStyle(.white)
          .frame(minHeight: 100)
          .padding(16)
          .background(
            RoundedRectangle(cornerRadius: 12)
              .fill(Color.white.opacity(0.05))
          )
      }
    }
  }
}

// MARK: - Actions

private struct ActionsSection: View {
  @Environment(AppState.self) private var appState

  private var canAct: Bool {
    !appState.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }

  var body: some View {
    VStack(spacing: 16) {
      // Loading indicator
      if appState.isModelLoading {
        LoadingBar(progress: appState.loadingProgress)
      }

      // Buttons
      HStack(spacing: 12) {
        // Stream (with highlighting for Chatterbox Turbo if enabled)
        MinimalButton(
          title: appState.selectedProvider == .chatterboxTurbo && appState.highlightingEnabled ? "Stream + Highlight" : "Stream",
          icon: appState.selectedProvider == .chatterboxTurbo && appState.highlightingEnabled ? "waveform.and.person.filled" : "waveform.path",
          style: .secondary,
          isDisabled: appState.isModelLoading || appState.isGenerating || !canAct
        ) {
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
        }

        // Generate / Stop
        MinimalButton(
          title: appState.isGenerating ? "Stop" : "Generate",
          icon: appState.isGenerating ? "stop.fill" : "play.fill",
          style: appState.isGenerating ? .destructive : .primary,
          isDisabled: appState.isModelLoading || (!appState.isGenerating && !canAct)
        ) {
          Task {
            if appState.isGenerating {
              await appState.stop()
            } else {
              if !appState.isLoaded {
                try? await appState.loadEngine()
              }
              await appState.generate()
            }
          }
        }
      }
    }
  }
}

private struct LoadingBar: View {
  let progress: Double

  var body: some View {
    VStack(spacing: 8) {
      Text("Loading...")
        .font(.system(size: 13, weight: .medium))
        .foregroundStyle(.white.opacity(0.5))

      GeometryReader { geo in
        ZStack(alignment: .leading) {
          Rectangle()
            .fill(Color.white.opacity(0.1))

          Rectangle()
            .fill(Color.white.opacity(0.8))
            .frame(width: geo.size.width * progress)
        }
      }
      .frame(height: 3)
      .clipShape(Capsule())
    }
  }
}

private struct MinimalButton: View {
  enum Style { case primary, secondary, destructive }

  let title: String
  let icon: String
  let style: Style
  let isDisabled: Bool
  let action: () -> Void

  private var backgroundColor: Color {
    if isDisabled { return Color.white.opacity(0.05) }
    switch style {
    case .primary: return .white
    case .secondary: return Color.white.opacity(0.1)
    case .destructive: return Color(red: 1.0, green: 0.3, blue: 0.3)
    }
  }

  private var foregroundColor: Color {
    if isDisabled { return .white.opacity(0.3) }
    switch style {
    case .primary: return .black
    case .secondary: return .white
    case .destructive: return .white
    }
  }

  var body: some View {
    Button(action: action) {
      HStack(spacing: 8) {
        Image(systemName: icon)
          .font(.system(size: 14, weight: .semibold))
        Text(title)
          .font(.system(size: 15, weight: .semibold))
      }
      .foregroundStyle(foregroundColor)
      .frame(maxWidth: .infinity)
      .frame(height: 50)
      .background(backgroundColor)
      .clipShape(RoundedRectangle(cornerRadius: 12))
    }
    .disabled(isDisabled)
  }
}

// MARK: - Output

private struct OutputDisplay: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      // Status
      if !appState.statusMessage.isEmpty {
        Text(appState.statusMessage)
          .font(.system(size: 14))
          .foregroundStyle(.white.opacity(0.6))
      }

      // Metrics
      if let result = appState.lastResult, !appState.isGenerating {
        MetricsRow(result: result)
      }

      // Playback
      if appState.lastResult != nil {
        PlaybackRow()
      }

      // Error
      if let error = appState.error {
        Text(error.localizedDescription)
          .font(.system(size: 13))
          .foregroundStyle(Color(red: 1.0, green: 0.4, blue: 0.4))
      }
    }
  }
}

private struct MetricsRow: View {
  let result: AudioResult

  var body: some View {
    HStack(spacing: 24) {
      if let duration = result.duration {
        Metric(label: "Duration", value: format(duration))
      }
      Metric(label: "Time", value: format(result.processingTime))
      if let rtf = result.realTimeFactor {
        Metric(label: "RTF", value: String(format: "%.2fx", rtf))
      }
    }
  }

  private func format(_ seconds: TimeInterval) -> String {
    seconds < 1 ? String(format: "%.0fms", seconds * 1000) : String(format: "%.2fs", seconds)
  }
}

private struct Metric: View {
  let label: String
  let value: String

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(value)
        .font(.system(size: 15, weight: .semibold, design: .monospaced))
        .foregroundStyle(.white)
      Text(label)
        .font(.system(size: 11, weight: .medium))
        .foregroundStyle(.white.opacity(0.4))
    }
  }
}

private struct PlaybackRow: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    HStack(spacing: 12) {
      // Play/Stop
      Button {
        Task {
          if appState.isPlaying {
            await appState.stop()
          } else {
            await appState.play()
          }
        }
      } label: {
        HStack(spacing: 6) {
          Image(systemName: appState.isPlaying ? "stop.fill" : "play.fill")
            .font(.system(size: 12, weight: .semibold))
          Text(appState.isPlaying ? "Stop" : "Play")
            .font(.system(size: 14, weight: .medium))
        }
        .foregroundStyle(.white)
        .frame(height: 40)
        .padding(.horizontal, 20)
        .background(Color.white.opacity(0.1))
        .clipShape(Capsule())
      }
      .disabled(appState.isGenerating)

      // Share
      if let url = appState.lastGeneratedAudioURL {
        ShareLink(item: url) {
          Image(systemName: "square.and.arrow.up")
            .font(.system(size: 14, weight: .medium))
            .foregroundStyle(.white.opacity(0.6))
            .frame(width: 40, height: 40)
            .background(Color.white.opacity(0.05))
            .clipShape(Circle())
        }
      }
    }
  }
}

// MARK: - Settings Sheet

private struct SettingsSheet: View {
  @Environment(AppState.self) private var appState
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    NavigationStack {
      ZStack {
        Color(red: 0.06, green: 0.06, blue: 0.08)
          .ignoresSafeArea()

        ScrollView {
          VStack(spacing: 24) {
            SettingsSection()
          }
          .padding(24)
        }
      }
      .navigationTitle("Settings")
      #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
      #endif
        .toolbar {
          ToolbarItem(placement: .confirmationAction) {
            Button("Done") {
              dismiss()
            }
            .foregroundStyle(.white)
          }
        }
        .toolbarBackground(Color(red: 0.06, green: 0.06, blue: 0.08), for: .navigationBar)
    }
    .presentationBackground(Color(red: 0.06, green: 0.06, blue: 0.08))
  }
}
