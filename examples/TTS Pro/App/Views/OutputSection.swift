// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

struct OutputSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      // Section header
      Text("Output")
        .font(.system(size: 13, weight: .medium))
        .foregroundStyle(.white.opacity(0.5))
        .textCase(.uppercase)
        .tracking(0.5)

      // Status and metrics
      StatusView()

      // Playback controls
      if appState.lastResult != nil {
        PlaybackControls()
      }

      // Error display
      if let error = appState.error {
        ErrorView(error: error)
      }
    }
  }
}

// MARK: - Status View

private struct StatusView: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      // Status message
      if !appState.statusMessage.isEmpty {
        HStack(spacing: 8) {
          if appState.isGenerating {
            ProgressView()
              .controlSize(.small)
          }
          Text(appState.statusMessage)
            .font(.subheadline)
            .foregroundStyle(appState.isGenerating ? .primary : .secondary)
        }
      }

      // Audio metrics when available
      if let result = appState.lastResult, !appState.isGenerating {
        AudioMetricsView(result: result)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .padding()
    .background(.background.secondary)
    .clipShape(RoundedRectangle(cornerRadius: 8))
  }
}

// MARK: - Audio Metrics

private struct AudioMetricsView: View {
  @Environment(AppState.self) private var appState
  let result: AudioResult

  var body: some View {
    HStack(spacing: 16) {
      if let duration = result.duration {
        MetricItem(
          icon: "waveform",
          label: "Duration",
          value: formatTime(duration)
        )
      }

      // Show TTFA if available (only for streaming with highlighting)
      if appState.timeToFirstAudio > 0 {
        MetricItem(
          icon: "bolt.fill",
          label: "TTFA",
          value: formatTime(appState.timeToFirstAudio)
        )
      }

      MetricItem(
        icon: "clock",
        label: "Total time",
        value: formatTime(result.processingTime)
      )

      if let rtf = result.realTimeFactor {
        MetricItem(
          icon: "speedometer",
          label: "RTF",
          value: String(format: "%.2fx", rtf)
        )
      }
    }
    .font(.caption)
  }

  private func formatTime(_ seconds: TimeInterval) -> String {
    if seconds < 1 {
      String(format: "%.0fms", seconds * 1000)
    } else {
      String(format: "%.2fs", seconds)
    }
  }
}

private struct MetricItem: View {
  let icon: String
  let label: String
  let value: String

  var body: some View {
    HStack(spacing: 4) {
      Image(systemName: icon)
        .foregroundStyle(.secondary)
      VStack(alignment: .leading, spacing: 2) {
        Text(label)
          .foregroundStyle(.secondary)
        Text(value)
          .fontWeight(.medium)
      }
    }
  }
}

// MARK: - Playback Controls

private struct PlaybackControls: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    HStack(spacing: 12) {
      // Play/Stop button
      Button {
        Task {
          if appState.isPlaying {
            await appState.stop()
          } else {
            await appState.play()
          }
        }
      } label: {
        HStack(spacing: 8) {
          Image(systemName: appState.isPlaying ? "stop.fill" : "play.fill")
          Text(appState.isPlaying ? "Stop" : "Play")
        }
        .padding(.vertical, 4)
      }
      .buttonStyle(.bordered)
      .disabled(appState.isGenerating)

      // Save button
      if let url = appState.lastGeneratedAudioURL {
        ShareLink(item: url) {
          HStack(spacing: 8) {
            Image(systemName: "square.and.arrow.up")
            Text("Share")
          }
          .padding(.vertical, 4)
        }
        .buttonStyle(.bordered)
      }
    }
  }
}

// MARK: - Error View

private struct ErrorView: View {
  let error: TTSError

  var body: some View {
    HStack(spacing: 8) {
      Image(systemName: "exclamationmark.triangle.fill")
        .foregroundStyle(.red)
      Text(error.localizedDescription)
        .font(.caption)
        .foregroundStyle(.red)
    }
    .padding()
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(.red.opacity(0.1))
    .clipShape(RoundedRectangle(cornerRadius: 8))
  }
}
