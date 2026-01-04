// Copyright © Anthony DePasquale

import Accelerate
@preconcurrency import AVFoundation
import Foundation

/// Plays raw audio samples for TTS engines
@Observable
@MainActor
public final class AudioSamplePlayer {
  // MARK: - Public State

  /// Whether audio is currently playing
  public private(set) var isPlaying: Bool = false

  /// Number of samples currently queued for playback
  public private(set) var queuedSampleCount: Int = 0

  /// Current playback position in seconds
  public private(set) var playbackPosition: TimeInterval = 0

  // MARK: - Private Properties

  @ObservationIgnored private var engine: AVAudioEngine!
  @ObservationIgnored private var playerNode: AVAudioPlayerNode!
  @ObservationIgnored private var audioFormat: AVAudioFormat!
  @ObservationIgnored private var hasStartedPlayback: Bool = false
  @ObservationIgnored private var playbackCompletionContinuation: CheckedContinuation<Void, Never>?
  @ObservationIgnored private var drainContinuations: [CheckedContinuation<Void, Never>] = []
  @ObservationIgnored private var totalSamplesEnqueued: Int = 0
  @ObservationIgnored private var samplesPlayed: Int = 0
  @ObservationIgnored private var playbackStartTime: Date?

  private let sampleRate: Int

  // MARK: - Initialization

  /// Create an audio engine with the specified sample rate
  /// - Parameter sampleRate: Sample rate in Hz (default: 24000)
  public init(sampleRate: Int = 24000) {
    self.sampleRate = sampleRate
    setup()
  }

  // MARK: - Setup

  private func setup() {
    engine = AVAudioEngine()
    playerNode = AVAudioPlayerNode()

    audioFormat = AVAudioFormat(
      standardFormatWithSampleRate: Double(sampleRate),
      channels: 1,
    )
    guard audioFormat != nil else {
      Log.audio.error("Failed to create audio format")
      return
    }

    engine.attach(playerNode)
    engine.connect(playerNode, to: engine.mainMixerNode, format: audioFormat)

    do {
      try engine.start()
      Log.audio.debug("Audio engine started successfully")
    } catch {
      Log.audio.error("Failed to start audio engine: \(error.localizedDescription)")
    }
  }

  // MARK: - Public API

  /// Play audio samples and wait for playback to complete
  /// - Parameters:
  ///   - samples: Float audio samples
  ///   - volumeBoost: Optional volume multiplier (default: 1.25)
  public func play(samples: [Float], volumeBoost: Float = TTSConstants.Audio.volumeBoostFactor) async {
    guard !samples.isEmpty else { return }

    // Stop any current playback (but don't restart engine)
    await stop()
    resetEngineIfNeeded()

    guard let buffer = createBuffer(from: samples, volumeBoost: volumeBoost) else {
      Log.audio.error("Failed to create audio buffer")
      return
    }

    scheduleBuffer(samples.count, buffer)

    playerNode.play()
    isPlaying = true
    hasStartedPlayback = true
    playbackStartTime = Date()
    samplesPlayed = 0
    totalSamplesEnqueued = samples.count

    // Start playback position timer
    startPlaybackTimer()

    // Await completion (resumed by buffer completion callback)
    await withCheckedContinuation { continuation in
      self.playbackCompletionContinuation = continuation
    }
  }

  /// Enqueue audio samples for streaming playback
  /// - Parameters:
  ///   - samples: Float audio samples to enqueue
  ///   - prebufferSeconds: Seconds to buffer before starting playback (0 for immediate)
  public func enqueue(samples: [Float], prebufferSeconds: Double = 0) {
    guard !samples.isEmpty else { return }

    resetEngineIfNeeded()

    // Initialize tracking on first enqueue (timer starts when playback actually begins)
    if !hasStartedPlayback {
      samplesPlayed = 0
      totalSamplesEnqueued = 0
    }

    totalSamplesEnqueued += samples.count

    // Schedule in small slices for smoother streaming
    let sliceSamples = max(1, Int(0.03 * Double(sampleRate))) // 30ms slices
    var offset = 0

    while offset < samples.count {
      let remaining = samples.count - offset
      let thisLength = min(sliceSamples, remaining)
      let slice = Array(samples[offset ..< (offset + thisLength)])

      guard let buffer = createBuffer(from: slice, volumeBoost: 1.0) else {
        offset += thisLength
        continue
      }

      queuedSampleCount += thisLength
      let decrementAmount = thisLength

      playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
        Task { @MainActor [weak self] in
          guard let self else { return }
          samplesPlayed += decrementAmount
          queuedSampleCount = max(0, queuedSampleCount - decrementAmount)

          // Note: playbackPosition is updated by timer using wall-clock time for accuracy
          // Callbacks are only used for tracking queue drain (playback completion)

          // When queue drains, mark playback complete and resume all waiting continuations
          if queuedSampleCount == 0, hasStartedPlayback {
            isPlaying = false
            hasStartedPlayback = false
            for continuation in drainContinuations {
              continuation.resume()
            }
            drainContinuations.removeAll()
          }
        }
      }

      // Start playback when prebuffer is reached
      if !hasStartedPlayback {
        let prebufferSamples = Int(prebufferSeconds * Double(sampleRate))
        if prebufferSamples == 0 || queuedSampleCount >= prebufferSamples {
          playerNode.play()
          hasStartedPlayback = true
          isPlaying = true
          playbackStartTime = Date()

          // Start wall-clock timer for accurate position tracking
          startPlaybackTimer()

          // Retry if playback didn't start
          if !playerNode.isPlaying {
            Task {
              try? await Task.sleep(for: .milliseconds(100))
              self.playerNode.play()
            }
          }
        }
      } else if !playerNode.isPlaying {
        playerNode.play()
      }

      offset += thisLength
    }
  }

  /// Stop playback and reset state
  public func stop() async {
    // Resume any waiting continuations before stopping
    playbackCompletionContinuation?.resume()
    playbackCompletionContinuation = nil
    for continuation in drainContinuations {
      continuation.resume()
    }
    drainContinuations.removeAll()

    // Dispatch to background QoS to avoid priority inversion
    // (audio threads run at Default QoS, so calling from Background avoids inversion)
    await withCheckedContinuation { continuation in
      DispatchQueue.global(qos: .background).async { [playerNode] in
        playerNode?.stop()
        playerNode?.reset()
        continuation.resume()
      }
    }

    isPlaying = false
    hasStartedPlayback = false
    queuedSampleCount = 0
    playbackPosition = 0
    samplesPlayed = 0
    totalSamplesEnqueued = 0
    playbackStartTime = nil

    Log.audio.debug("Audio playback stopped")
  }

  /// Wait for all enqueued audio to finish playing
  ///
  /// Call this after finishing `enqueue()` calls to wait for playback to complete.
  /// Returns immediately if no audio is queued or playing.
  /// Safe to call from multiple concurrent contexts.
  public func awaitDrain() async {
    // Quick check - if nothing playing, return immediately
    guard hasStartedPlayback, queuedSampleCount > 0 else { return }

    await withCheckedContinuation { continuation in
      // Re-check condition atomically when setting continuation
      if queuedSampleCount == 0 || !hasStartedPlayback {
        continuation.resume()
      } else {
        drainContinuations.append(continuation)
      }
    }
  }

  /// Reset the audio engine (useful after interruptions)
  func reset() async {
    await stop()

    // Reconnect components
    if playerNode.engine != nil {
      engine.detach(playerNode)
    }
    engine.attach(playerNode)
    engine.connect(playerNode, to: engine.mainMixerNode, format: audioFormat)

    // Restart engine
    do {
      try engine.start()
      Log.audio.debug("Audio engine reset successfully")
    } catch {
      Log.audio.error("Failed to restart audio engine: \(error.localizedDescription)")
    }
  }

  // MARK: - Private

  private func createBuffer(from samples: [Float], volumeBoost: Float) -> AVAudioPCMBuffer? {
    let frameCount = AVAudioFrameCount(samples.count)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
      return nil
    }

    buffer.frameLength = frameCount

    guard let channelData = buffer.floatChannelData else {
      return nil
    }

    let count = vDSP_Length(samples.count)
    let maxValue = TTSConstants.Audio.maxSampleValue

    // Use vDSP for vectorized volume boost and clipping (equivalent to scipy/numpy operations)
    samples.withUnsafeBufferPointer { samplesPtr in
      // Apply volume boost: output = samples * volumeBoost
      var boost = volumeBoost
      vDSP_vsmul(samplesPtr.baseAddress!, 1, &boost, channelData[0], 1, count)

      // Clip to [-maxValue, maxValue] range
      var minVal = -maxValue
      var maxVal = maxValue
      vDSP_vclip(channelData[0], 1, &minVal, &maxVal, channelData[0], 1, count)
    }

    return buffer
  }

  private func scheduleBuffer(_ sampleCount: Int, _ buffer: AVAudioPCMBuffer) {
    playerNode.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
      Task { @MainActor [weak self] in
        guard let self else { return }
        samplesPlayed += sampleCount
        // Update playback position based on samples played
        playbackPosition = Double(samplesPlayed) / Double(sampleRate)

        // Buffer playback complete - resume continuation
        // Note: playerNode.isPlaying may still be true until explicitly stopped,
        // but .dataPlayedBack guarantees the audio data has been rendered
        if hasStartedPlayback {
          isPlaying = false
          hasStartedPlayback = false
          playbackCompletionContinuation?.resume()
          playbackCompletionContinuation = nil
        }
      }
    }
  }

  private func startPlaybackTimer() {
    // Cancel any existing timer
    Task { [weak self] in
      await self?.stopPlaybackTimer()
    }

    // Start a timer to update playback position during playback
    Task { [weak self] in
      while self?.isPlaying == true {
        try? await Task.sleep(for: .milliseconds(50))
        if let startTime = self?.playbackStartTime {
          self?.playbackPosition = Date().timeIntervalSince(startTime)
        }
      }
    }
  }

  private func stopPlaybackTimer() async {
    // Timer will stop automatically when isPlaying becomes false
  }

  private func resetEngineIfNeeded() {
    if !engine.isRunning {
      do {
        try engine.start()
      } catch {
        Log.audio.error("Failed to restart audio engine: \(error.localizedDescription)")
      }
    }
  }
}
