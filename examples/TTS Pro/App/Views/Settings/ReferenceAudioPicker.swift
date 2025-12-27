// Copyright © Anthony DePasquale

import AVFoundation
import MLXAudio
import SwiftUI
import UniformTypeIdentifiers

struct ReferenceAudioPickerConfig {
  var title: String = "Reference Audio"
  var infoText: String = ""
  var loadingText: String = "Processing..."
}

struct ReferenceAudioPicker: View {
  let config: ReferenceAudioPickerConfig
  let statusDescription: String
  let isLoaded: Bool
  let onLoadDefault: () async throws -> Void
  let onLoadFromFile: (URL) async throws -> Void
  let onLoadFromURL: (URL) async throws -> Void

  @State private var isShowingFilePicker = false
  @State private var isShowingURLInput = false
  @State private var isShowingRecorder = false
  @State private var customURL = ""
  @State private var isLoading = false
  @State private var errorMessage: String?

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      // Header
      HStack {
        Text(config.title)
          .font(.system(size: 14, weight: .medium))
          .foregroundStyle(.white)
        Spacer()
        if isLoading {
          ProgressView()
            .scaleEffect(0.7)
            .tint(.white)
        }
      }

      // Status
      HStack(spacing: 8) {
        Circle()
          .fill(isLoaded ? Color.green : Color.white.opacity(0.3))
          .frame(width: 8, height: 8)
        Text(statusDescription)
          .font(.system(size: 13))
          .foregroundStyle(isLoaded ? .white : .white.opacity(0.6))
      }

      // Actions
      HStack(spacing: 12) {
        SourceButton(title: "Default", isDisabled: isLoading) {
          Task { await loadDefault() }
        }
        SourceButton(title: "File", isDisabled: isLoading) {
          isShowingFilePicker = true
        }
        SourceButton(title: "URL", isDisabled: isLoading) {
          isShowingURLInput = true
        }
        SourceButton(title: "Record", isDisabled: isLoading) {
          isShowingRecorder = true
        }
      }

      // Error
      if let error = errorMessage {
        Text(error)
          .font(.system(size: 12))
          .foregroundStyle(Color(red: 1.0, green: 0.4, blue: 0.4))
      }

      // Info
      if !config.infoText.isEmpty {
        Text(config.infoText)
          .font(.system(size: 12))
          .foregroundStyle(.white.opacity(0.4))
      }
    }
    .fileImporter(
      isPresented: $isShowingFilePicker,
      allowedContentTypes: [.audio, .wav, .mp3, .aiff],
      allowsMultipleSelection: false
    ) { result in
      handleFileSelection(result)
    }
    .alert("Enter Audio URL", isPresented: $isShowingURLInput) {
      TextField("https://...", text: $customURL)
        .textContentType(.URL)
      #if os(iOS)
        .autocapitalization(.none)
      #endif
      Button("Cancel", role: .cancel) { customURL = "" }
      Button("Load") { Task { await loadFromURL() } }
        .disabled(customURL.isEmpty)
    }
    .sheet(isPresented: $isShowingRecorder) {
      RecorderSheet(
        onComplete: { url in
          Task {
            isLoading = true
            errorMessage = nil
            do {
              try await onLoadFromFile(url)
            } catch {
              errorMessage = error.localizedDescription
            }
            isLoading = false
          }
        }
      )
    }
  }

  private func loadDefault() async {
    isLoading = true
    errorMessage = nil
    do {
      try await onLoadDefault()
    } catch {
      errorMessage = error.localizedDescription
    }
    isLoading = false
  }

  private func handleFileSelection(_ result: Result<[URL], Error>) {
    switch result {
    case let .success(urls):
      guard let url = urls.first else { return }
      guard url.startAccessingSecurityScopedResource() else {
        errorMessage = "Permission denied"
        return
      }
      Task {
        defer { url.stopAccessingSecurityScopedResource() }
        isLoading = true
        errorMessage = nil
        do {
          try await onLoadFromFile(url)
        } catch {
          errorMessage = error.localizedDescription
        }
        isLoading = false
      }
    case let .failure(error):
      errorMessage = error.localizedDescription
    }
  }

  private func loadFromURL() async {
    guard let url = URL(string: customURL) else {
      errorMessage = "Invalid URL"
      return
    }
    isLoading = true
    errorMessage = nil
    do {
      try await onLoadFromURL(url)
    } catch {
      errorMessage = error.localizedDescription
    }
    isLoading = false
    customURL = ""
  }
}

// MARK: - Source Button

private struct SourceButton: View {
  let title: String
  let isDisabled: Bool
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      Text(title)
        .font(.system(size: 13, weight: .medium))
        .foregroundStyle(isDisabled ? .white.opacity(0.3) : .white)
        .frame(maxWidth: .infinity)
        .frame(height: 36)
        .background(Color.white.opacity(isDisabled ? 0.03 : 0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
    .disabled(isDisabled)
  }
}

// MARK: - Recorder Sheet

private struct RecorderSheet: View {
  @Environment(\.dismiss) private var dismiss
  let onComplete: (URL) -> Void

  @State private var isRecording = false
  @State private var elapsedTime: TimeInterval = 0
  @State private var timer: Timer?
  @State private var audioRecorder: AVAudioRecorder?
  @State private var recordingURL: URL?
  @State private var hasRecording = false
  @State private var errorMessage: String?

  private let maxDuration: TimeInterval = 10.0
  private let samplePhrase = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore."

  var body: some View {
    NavigationStack {
      ZStack {
        Color(red: 0.06, green: 0.06, blue: 0.08)
          .ignoresSafeArea()

        VStack(spacing: 40) {
          Spacer()

          // Timer
          Text(formatTime(elapsedTime))
            .font(.system(size: 60, weight: .thin, design: .monospaced))
            .foregroundStyle(.white)

          // Progress
          GeometryReader { geo in
            ZStack(alignment: .leading) {
              Rectangle()
                .fill(Color.white.opacity(0.1))
              Rectangle()
                .fill(Color.white.opacity(0.8))
                .frame(width: geo.size.width * (elapsedTime / maxDuration))
            }
          }
          .frame(height: 3)
          .clipShape(Capsule())
          .padding(.horizontal, 60)

          // Phrase
          Text(samplePhrase)
            .font(.system(size: 15))
            .foregroundStyle(.white.opacity(0.6))
            .multilineTextAlignment(.center)
            .lineSpacing(4)
            .padding(.horizontal, 40)

          Spacer()

          // Error
          if let error = errorMessage {
            Text(error)
              .font(.system(size: 13))
              .foregroundStyle(Color(red: 1.0, green: 0.4, blue: 0.4))
          }

          // Record button
          Button {
            if isRecording {
              stopRecording()
            } else {
              startRecording()
            }
          } label: {
            ZStack {
              Circle()
                .stroke(Color.white.opacity(0.3), lineWidth: 3)
                .frame(width: 80, height: 80)

              if isRecording {
                RoundedRectangle(cornerRadius: 6)
                  .fill(Color.red)
                  .frame(width: 28, height: 28)
              } else {
                Circle()
                  .fill(Color.red)
                  .frame(width: 60, height: 60)
              }
            }
          }

          // Use button
          if hasRecording {
            Button {
              if let url = recordingURL {
                onComplete(url)
                dismiss()
              }
            } label: {
              Text("Use Recording")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.black)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(.white)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .padding(.horizontal, 40)
          }

          Spacer().frame(height: 40)
        }
      }
      .navigationTitle("Record")
      #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
      #endif
        .toolbar {
          ToolbarItem(placement: .cancellationAction) {
            Button("Cancel") {
              stopRecording()
              dismiss()
            }
            .foregroundStyle(.white.opacity(0.6))
          }
        }
    }
    .presentationBackground(Color(red: 0.06, green: 0.06, blue: 0.08))
    .onDisappear { timer?.invalidate() }
  }

  private func startRecording() {
    errorMessage = nil

    #if os(iOS)
      AVAudioApplication.requestRecordPermission { granted in
        DispatchQueue.main.async {
          if granted {
            beginRecording()
          } else {
            errorMessage = "Microphone access denied"
          }
        }
      }
    #else
      beginRecording()
    #endif
  }

  private func beginRecording() {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    recordingURL = docs.appendingPathComponent("voice_\(Date().timeIntervalSince1970).wav")

    guard let url = recordingURL else { return }

    #if os(iOS)
      do {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default)
        try session.setActive(true)
      } catch {
        errorMessage = "Audio session error"
        return
      }
    #endif

    let settings: [String: Any] = [
      AVFormatIDKey: Int(kAudioFormatLinearPCM),
      AVSampleRateKey: 16000.0,
      AVNumberOfChannelsKey: 1,
      AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey: false,
      AVLinearPCMIsBigEndianKey: false,
    ]

    do {
      audioRecorder = try AVAudioRecorder(url: url, settings: settings)
      audioRecorder?.record()
      isRecording = true
      elapsedTime = 0
      hasRecording = false

      timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
        elapsedTime += 0.1
        if elapsedTime >= maxDuration {
          stopRecording()
        }
      }
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  private func stopRecording() {
    timer?.invalidate()
    timer = nil
    audioRecorder?.stop()
    audioRecorder = nil
    isRecording = false
    if elapsedTime >= 1.0 {
      hasRecording = true
    }
  }

  private func formatTime(_ time: TimeInterval) -> String {
    let seconds = Int(time)
    let tenths = Int((time * 10).truncatingRemainder(dividingBy: 10))
    return String(format: "%02d.%d", seconds, tenths)
  }
}

// MARK: - UTType Extensions

extension UTType {
  static var wav: UTType { UTType(filenameExtension: "wav")! }
  static var mp3: UTType { UTType(filenameExtension: "mp3")! }
  static var aiff: UTType { UTType(filenameExtension: "aiff")! }
}
