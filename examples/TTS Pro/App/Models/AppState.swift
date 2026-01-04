// Copyright © Anthony DePasquale

import Kokoro
import MLXAudio
import SwiftUI

/// Central state management for the TTS application
@MainActor
@Observable
final class AppState {
  // MARK: - Dependencies

  let engineManager: EngineManager

  // MARK: - User Input

  var inputText: String = "This is a test of word highlighting in Chatterbox Turbo."

  /// Speech speed multiplier (Kokoro only)
  var speed: Float = TTSConstants.Speed.default

  // MARK: - UI State

  var showInspector: Bool = true

  var autoPlay: Bool = true

  var statusMessage: String = ""

  // MARK: - Word Highlighting

  /// Whether word highlighting is enabled for streaming (user preference)
  var highlightingEnabled: Bool = true

  /// Whether we're currently in highlighting mode (showing highlighted text instead of editor)
  private(set) var isHighlighting: Bool = false

  /// Current playback position in seconds (from engine - single source of truth)
  var playbackPosition: TimeInterval {
    engineManager.chatterboxTurboEngine.playbackPosition
  }

  /// Word timings for the current audio (for highlighting during playback)
  private(set) var wordTimings: [HighlightedWord] = []

  /// Display text that matches the character ranges in wordTimings
  /// This may differ from inputText if the aligner skips certain characters
  private(set) var highlightingDisplayText: String = ""

  /// The text that the current wordTimings correspond to
  /// Used to invalidate timings when user edits the text
  private var timingsSourceText: String = ""

  /// Currently highlighted word index (for efficient updates)
  private(set) var currentHighlightedWordIndex: Int? = nil

  /// Timer for polling playback position during highlighting
  private var highlightUpdateTimer: Timer?

  // MARK: - Highlight Theme Settings

  /// Selected theme preset
  var highlightThemePreset: HighlightThemePreset = .default

  /// Custom spoken color (used when preset is .custom)
  var customSpokenColor: Color = .green

  /// Custom current word color (used when preset is .custom)
  var customCurrentWordColor: Color = .yellow

  /// Custom upcoming color (used when preset is .custom)
  var customUpcomingColor: Color = Color.primary

  /// The active highlight theme based on preset or custom colors
  var highlightTheme: HighlightTheme {
    if highlightThemePreset == .custom {
      return HighlightTheme(
        spokenColor: customSpokenColor,
        currentWordColor: customCurrentWordColor,
        upcomingColor: customUpcomingColor
      )
    }
    return highlightThemePreset.theme
  }

  // MARK: - Generated Output

  private(set) var lastResult: AudioResult?

  /// Time to first audio in seconds (for streaming with highlighting)
  private(set) var timeToFirstAudio: TimeInterval = 0

  // MARK: - Delegated State (from EngineManager)

  var selectedProvider: TTSProvider { engineManager.selectedProvider }
  var isLoaded: Bool { engineManager.isLoaded }
  var isGenerating: Bool { engineManager.isGenerating }
  var isPlaying: Bool { engineManager.isPlaying }
  var isModelLoading: Bool { engineManager.isLoading }
  var loadingProgress: Double { engineManager.loadingProgress }
  var error: TTSError? { engineManager.error }
  var generationTime: TimeInterval { engineManager.generationTime }
  var lastGeneratedAudioURL: URL? { engineManager.lastGeneratedAudioURL }
  var streamingGranularity: StreamingGranularity { engineManager.currentEngine.streamingGranularity }

  var canGenerate: Bool {
    isLoaded && !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating
  }

  // MARK: - Initialization

  init() {
    engineManager = EngineManager()
  }

  // MARK: - Provider Management

  func selectProvider(_ provider: TTSProvider) async {
    guard provider != selectedProvider else { return }

    await engineManager.selectProvider(provider)
    lastResult = nil
    statusMessage = provider.statusMessage
  }

  // MARK: - Engine Operations

  func loadEngine() async throws {
    do {
      try await engineManager.loadEngine()
    } catch {
      statusMessage = error.localizedDescription
      throw error
    }
  }

  /// Generate audio from the current input text
  func generate() async {
    guard canGenerate else { return }

    // Exit highlighting mode for regular generation
    clearHighlighting()
    statusMessage = "Generating..."

    do {
      lastResult = try await engineManager.generate(text: inputText, speed: speed)

      if let result = lastResult {
        statusMessage = formatResultStatus(result)
      }

      if autoPlay, let result = lastResult {
        await engineManager.play(result)
      }
    } catch is CancellationError {
      statusMessage = "Stopped"
    } catch {
      statusMessage = error.localizedDescription
    }
  }

  /// Generate with streaming
  func generateStreaming() async {
    guard canGenerate else { return }

    statusMessage = "Streaming..."

    do {
      lastResult = try await engineManager.sayStreaming(text: inputText, speed: speed)

      if let result = lastResult {
        statusMessage = formatResultStatus(result)
      }
    } catch is CancellationError {
      statusMessage = "Stopped"
    } catch {
      statusMessage = error.localizedDescription
    }
  }

  /// Play the last generated audio
  func play() async {
    guard let result = lastResult else { return }
    await engineManager.play(result)
  }

  /// Stop generation and playback
  func stop() async {
    stopHighlightUpdateTimer()
    clearHighlighting()
    await engineManager.stop()
    statusMessage = "Stopped"
  }

  /// Clear the highlighting state
  func clearHighlighting() {
    isHighlighting = false
    wordTimings = []
    highlightingDisplayText = ""
    timingsSourceText = ""
    currentHighlightedWordIndex = nil
  }

  /// Exit highlighting mode and return to editing (keeps the text but clears highlighting state)
  func exitHighlightingMode() {
    isHighlighting = false
    // Keep the text and timings for potential re-use, just exit the mode
  }

  /// Generate with streaming and word highlighting (for Chatterbox Turbo)
  ///
  /// Uses true streaming - audio starts playing as soon as the first chunk is ready,
  /// reducing perceived latency compared to batch-then-play.
  func generateStreamingWithHighlighting() async {
    guard canGenerate else { return }
    guard selectedProvider == .chatterboxTurbo else {
      // Fall back to regular streaming for other providers
      return await generateStreaming()
    }

    // Enter highlighting mode
    isHighlighting = true
    currentHighlightedWordIndex = nil
    highlightingDisplayText = inputText
    timingsSourceText = inputText
    timeToFirstAudio = 0
    statusMessage = "Generating..."

    do {
      // Use true streaming - audio plays as chunks arrive
      // Use attention-based alignment for accurate word timings
      let timingsResult = try await engineManager.chatterboxTurboEngine.sayStreamingWithTimings(
        inputText,
        referenceAudio: chatterboxTurboReferenceAudio,
        useAttentionAlignment: true,
        onTimingsUpdate: { [weak self] timings in
          // Update word timings incrementally as chunks arrive
          self?.wordTimings = timings
        },
        onFirstAudio: { [weak self] ttfa in
          guard let self else { return }
          // Capture TTFA and start highlight timer
          timeToFirstAudio = ttfa
          statusMessage = "Playing..."
          Log.audio.debug("Time to first audio: \(ttfa.formatted(decimals: 2))s")
          // Start highlight update timer now that audio is playing
          startHighlightUpdateTimer(duration: 0) // Duration not needed, timer runs until playback stops
        }
      )

      // Playback complete - stop timer
      stopHighlightUpdateTimer()

      Log.audio.debug("Word timings: \(timingsResult.wordTimings.count) words, duration: \(timingsResult.duration.formatted(decimals: 2))s, TTFA: \(timingsResult.timeToFirstAudio.formatted(decimals: 2))s")

      // Create AudioResult with duration
      lastResult = .samples(
        data: timingsResult.allSamples,
        sampleRate: 24000,
        processingTime: timingsResult.processingTime
      )

      statusMessage = formatResultStatusWithTTFA(lastResult!, ttfa: timingsResult.timeToFirstAudio)

      // Exit highlighting mode and return to editor after playback completes
      isHighlighting = false
      currentHighlightedWordIndex = nil
    } catch is CancellationError {
      statusMessage = "Stopped"
      stopHighlightUpdateTimer()
      clearHighlighting()
    } catch {
      statusMessage = error.localizedDescription
      stopHighlightUpdateTimer()
      clearHighlighting()
    }
  }

  // MARK: - Highlight Update Timer

  /// Start a timer to poll playback position and update highlight index
  /// Uses ~60fps (16ms) for smooth word transitions that align with audio
  private func startHighlightUpdateTimer(duration: TimeInterval) {
    stopHighlightUpdateTimer()

    highlightUpdateTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] _ in
      Task { @MainActor in
        guard let self = self else { return }

        // Stop if text was edited
        guard self.inputText == self.timingsSourceText else {
          self.stopHighlightUpdateTimer()
          return
        }

        // Update the current highlighted word index based on playback position
        self.updateCurrentHighlightedWordIndex()
      }
    }
  }

  /// Stop the highlight update timer
  private func stopHighlightUpdateTimer() {
    highlightUpdateTimer?.invalidate()
    highlightUpdateTimer = nil
  }

  /// Update the current highlighted word index based on playback position
  /// Uses optimized search starting from current index since playback is sequential
  private func updateCurrentHighlightedWordIndex() {
    let position = playbackPosition
    guard !wordTimings.isEmpty else { return }

    // Optimization: start search from current index since playback is sequential
    let startIndex = currentHighlightedWordIndex ?? 0

    // First check if we're still in the current word
    if startIndex < wordTimings.count {
      let current = wordTimings[startIndex]
      if current.start <= position && position < current.end {
        // Still in current word, no update needed
        return
      }
    }

    // Check next word (most common case during playback)
    let nextIndex = startIndex + 1
    if nextIndex < wordTimings.count {
      let next = wordTimings[nextIndex]
      if next.start <= position && position < next.end {
        currentHighlightedWordIndex = nextIndex
        return
      }
    }

    // Fall back to linear search from current position
    for i in startIndex..<wordTimings.count {
      let timing = wordTimings[i]
      if timing.start <= position && position < timing.end {
        if currentHighlightedWordIndex != i {
          currentHighlightedWordIndex = i
        }
        return
      }
    }

    // Check if past last word
    if position > 0, let lastTiming = wordTimings.last, position >= lastTiming.end {
      let lastIndex = wordTimings.count - 1
      if currentHighlightedWordIndex != lastIndex {
        currentHighlightedWordIndex = lastIndex
      }
    }
  }

  /// Check if a word at a given index should be highlighted
  func isWordHighlighted(at index: Int) -> Bool {
    guard isHighlighting else { return false }
    return currentHighlightedWordIndex == index
  }

  /// Check if a word at a given index has been spoken (is before current position)
  func isWordSpoken(at index: Int) -> Bool {
    guard isHighlighting, let currentIndex = currentHighlightedWordIndex else { return false }
    return index < currentIndex
  }

  /// Get the highlight progress for the current word (0.0 to 1.0)
  func currentWordProgress() -> Double {
    guard isHighlighting,
          let index = currentHighlightedWordIndex,
          index < wordTimings.count else {
      return 0
    }

    let timing = wordTimings[index]
    let wordDuration = timing.end - timing.start
    guard wordDuration > 0 else { return 1.0 }

    let position = playbackPosition
    let progress = (position - timing.start) / wordDuration
    return max(0, min(1, progress))
  }

  // MARK: - Private Helpers

  private func formatResultStatus(_ result: AudioResult) -> String {
    let timeStr = result.processingTime.formatted(decimals: 2)

    if let duration = result.duration {
      let durationStr = duration.formatted(decimals: 2)
      if let rtf = result.realTimeFactor {
        let rtfStr = rtf.formatted(decimals: 2)
        return "Generated \(durationStr) sec. audio in \(timeStr) sec. (RTF: \(rtfStr)x)"
      }
      return "Generated \(durationStr) sec. audio in \(timeStr) sec."
    }

    return "Generated in \(timeStr) sec."
  }

  private func formatResultStatusWithTTFA(_ result: AudioResult, ttfa: TimeInterval) -> String {
    let timeStr = result.processingTime.formatted(decimals: 2)
    let ttfaStr = ttfa.formatted(decimals: 2)

    if let duration = result.duration {
      let durationStr = duration.formatted(decimals: 2)
      if let rtf = result.realTimeFactor {
        let rtfStr = rtf.formatted(decimals: 2)
        return "Generated \(durationStr)s audio in \(timeStr)s (TTFA: \(ttfaStr)s, RTF: \(rtfStr)x)"
      }
      return "Generated \(durationStr)s audio in \(timeStr)s (TTFA: \(ttfaStr)s)"
    }

    return "Generated in \(timeStr)s (TTFA: \(ttfaStr)s)"
  }
}

// MARK: - Voice Selection

extension AppState {
  var kokoroVoice: KokoroEngine.Voice {
    get { engineManager.kokoroVoice }
    set { engineManager.kokoroVoice = newValue }
  }

  var orpheusVoice: OrpheusEngine.Voice {
    get { engineManager.orpheusVoice }
    set { engineManager.orpheusVoice = newValue }
  }

  var marvisVoice: MarvisEngine.Voice {
    get { engineManager.marvisVoice }
    set { engineManager.marvisVoice = newValue }
  }
}

// MARK: - Engine-Specific Property Access

extension AppState {
  /// Quality level for Marvis
  var marvisQualityLevel: MarvisEngine.QualityLevel {
    get { engineManager.marvisEngine.qualityLevel }
    set { engineManager.marvisEngine.qualityLevel = newValue }
  }

  /// Streaming interval for Marvis
  var streamingInterval: Double {
    get { engineManager.marvisEngine.streamingInterval }
    set { engineManager.marvisEngine.streamingInterval = newValue }
  }

  /// Temperature for Orpheus
  var orpheusTemperature: Float {
    get { engineManager.orpheusEngine.temperature }
    set { engineManager.orpheusEngine.temperature = newValue }
  }

  /// Top-P for Orpheus
  var orpheusTopP: Float {
    get { engineManager.orpheusEngine.topP }
    set { engineManager.orpheusEngine.topP = newValue }
  }
}

// MARK: - Chatterbox Reference Audio

extension AppState {
  /// Current prepared reference audio for Chatterbox
  var chatterboxReferenceAudio: ChatterboxReferenceAudio? {
    get { engineManager.chatterboxReferenceAudio }
    set { engineManager.chatterboxReferenceAudio = newValue }
  }

  /// Whether reference audio is prepared for Chatterbox
  var isChatterboxReferenceAudioLoaded: Bool {
    engineManager.chatterboxReferenceAudio != nil
  }

  /// Description of current reference audio
  var chatterboxReferenceAudioDescription: String {
    engineManager.chatterboxReferenceAudio?.description ?? "No reference audio"
  }

  /// Prepare reference audio from a URL (local file or remote)
  func prepareChatterboxReferenceAudio(from url: URL) async throws {
    let ref = try await engineManager.chatterboxEngine.prepareReferenceAudio(from: url)
    engineManager.chatterboxReferenceAudio = ref
  }

  /// Prepare the default reference audio
  func prepareDefaultChatterboxReferenceAudio() async throws {
    let ref = try await engineManager.chatterboxEngine.prepareDefaultReferenceAudio()
    engineManager.chatterboxReferenceAudio = ref
  }

  /// Emotion exaggeration for Chatterbox
  var chatterboxExaggeration: Float {
    get { engineManager.chatterboxEngine.exaggeration }
    set { engineManager.chatterboxEngine.exaggeration = newValue }
  }

  /// Temperature for Chatterbox
  var chatterboxTemperature: Float {
    get { engineManager.chatterboxEngine.temperature }
    set { engineManager.chatterboxEngine.temperature = newValue }
  }

  /// CFG weight for Chatterbox
  var chatterboxCFGWeight: Float {
    get { engineManager.chatterboxEngine.cfgWeight }
    set { engineManager.chatterboxEngine.cfgWeight = newValue }
  }
}

// MARK: - Chatterbox Turbo Reference Audio

extension AppState {
  /// Current prepared reference audio for Chatterbox Turbo
  var chatterboxTurboReferenceAudio: ChatterboxTurboReferenceAudio? {
    get { engineManager.chatterboxTurboReferenceAudio }
    set { engineManager.chatterboxTurboReferenceAudio = newValue }
  }

  /// Whether reference audio is prepared for Chatterbox Turbo
  var isChatterboxTurboReferenceAudioLoaded: Bool {
    engineManager.chatterboxTurboReferenceAudio != nil
  }

  /// Description of current reference audio
  var chatterboxTurboReferenceAudioDescription: String {
    engineManager.chatterboxTurboReferenceAudio?.description ?? "No reference audio"
  }

  /// Prepare reference audio from a URL (local file or remote)
  func prepareChatterboxTurboReferenceAudio(from url: URL) async throws {
    let ref = try await engineManager.chatterboxTurboEngine.prepareReferenceAudio(from: url)
    engineManager.chatterboxTurboReferenceAudio = ref
  }

  /// Prepare the default reference audio
  func prepareDefaultChatterboxTurboReferenceAudio() async throws {
    let ref = try await engineManager.chatterboxTurboEngine.prepareDefaultReferenceAudio()
    engineManager.chatterboxTurboReferenceAudio = ref
  }

  /// Temperature for Chatterbox Turbo
  var chatterboxTurboTemperature: Float {
    get { engineManager.chatterboxTurboEngine.temperature }
    set { engineManager.chatterboxTurboEngine.temperature = newValue }
  }

  /// Top-P for Chatterbox Turbo
  var chatterboxTurboTopP: Float {
    get { engineManager.chatterboxTurboEngine.topP }
    set { engineManager.chatterboxTurboEngine.topP = newValue }
  }

  /// Top-K for Chatterbox Turbo
  var chatterboxTurboTopK: Int {
    get { engineManager.chatterboxTurboEngine.topK }
    set { engineManager.chatterboxTurboEngine.topK = newValue }
  }
}

// MARK: - CosyVoice2 Speaker

extension AppState {
  /// Current prepared speaker for CosyVoice2
  var cosyVoice2Speaker: CosyVoice2Speaker? {
    get { engineManager.cosyVoice2Speaker }
    set { engineManager.cosyVoice2Speaker = newValue }
  }

  /// Whether speaker is prepared for CosyVoice2
  var isCosyVoice2SpeakerLoaded: Bool {
    engineManager.cosyVoice2Speaker != nil
  }

  /// Description of current speaker
  var cosyVoice2SpeakerDescription: String {
    engineManager.cosyVoice2Speaker?.description ?? "No speaker"
  }

  /// Prepare speaker from a URL (local file or remote)
  func prepareCosyVoice2Speaker(from url: URL, transcription: String? = nil) async throws {
    let speaker = try await engineManager.cosyVoice2Engine.prepareSpeaker(from: url, transcription: transcription)
    engineManager.cosyVoice2Speaker = speaker
  }

  /// Prepare the default speaker
  func prepareDefaultCosyVoice2Speaker() async throws {
    let speaker = try await engineManager.cosyVoice2Engine.prepareDefaultSpeaker()
    engineManager.cosyVoice2Speaker = speaker
  }

  /// Generation mode for CosyVoice2
  var cosyVoice2GenerationMode: CosyVoice2Engine.GenerationMode {
    get { engineManager.cosyVoice2Engine.generationMode }
    set { engineManager.cosyVoice2Engine.generationMode = newValue }
  }

  /// Instruct text for CosyVoice2 instruct mode
  var cosyVoice2InstructText: String {
    get { engineManager.cosyVoice2Engine.instructText }
    set { engineManager.cosyVoice2Engine.instructText = newValue }
  }

  /// Sampling parameter for CosyVoice2
  var cosyVoice2Sampling: Int {
    get { engineManager.cosyVoice2Engine.sampling }
    set { engineManager.cosyVoice2Engine.sampling = newValue }
  }

  /// Number of flow matching timesteps for CosyVoice2
  var cosyVoice2NTimesteps: Int {
    get { engineManager.cosyVoice2Engine.nTimesteps }
    set { engineManager.cosyVoice2Engine.nTimesteps = newValue }
  }

  /// Whether source audio is loaded for voice conversion
  var isCosyVoice2SourceAudioLoaded: Bool {
    engineManager.cosyVoice2Engine.isSourceAudioLoaded
  }

  /// Description of the loaded source audio
  var cosyVoice2SourceAudioDescription: String {
    engineManager.cosyVoice2Engine.sourceAudioDescription
  }

  /// Prepare source audio for voice conversion
  func prepareCosyVoice2SourceAudio(from url: URL) async throws {
    try await engineManager.cosyVoice2Engine.prepareSourceAudio(from: url)
  }

  /// Clear source audio
  func clearCosyVoice2SourceAudio() async {
    await engineManager.cosyVoice2Engine.clearSourceAudio()
  }

  /// Generate voice conversion
  func generateCosyVoice2VoiceConversion() async throws -> AudioResult {
    try await engineManager.cosyVoice2Engine.generateVoiceConversion(
      speaker: engineManager.cosyVoice2Speaker
    )
  }
}

// MARK: - CosyVoice3 Speaker

extension AppState {
  /// Current prepared speaker for CosyVoice3
  var cosyVoice3Speaker: CosyVoice3Speaker? {
    get { engineManager.cosyVoice3Speaker }
    set { engineManager.cosyVoice3Speaker = newValue }
  }

  /// Whether speaker is prepared for CosyVoice3
  var isCosyVoice3SpeakerLoaded: Bool {
    engineManager.cosyVoice3Speaker != nil
  }

  /// Description of current speaker
  var cosyVoice3SpeakerDescription: String {
    engineManager.cosyVoice3Speaker?.description ?? "No speaker"
  }

  /// Prepare speaker from a URL (local file or remote)
  func prepareCosyVoice3Speaker(from url: URL, transcription: String? = nil) async throws {
    let speaker = try await engineManager.cosyVoice3Engine.prepareSpeaker(from: url, transcription: transcription)
    engineManager.cosyVoice3Speaker = speaker
  }

  /// Prepare the default speaker
  func prepareDefaultCosyVoice3Speaker() async throws {
    let speaker = try await engineManager.cosyVoice3Engine.prepareDefaultSpeaker()
    engineManager.cosyVoice3Speaker = speaker
  }

  /// Generation mode for CosyVoice3
  var cosyVoice3GenerationMode: CosyVoice3Engine.GenerationMode {
    get { engineManager.cosyVoice3Engine.generationMode }
    set { engineManager.cosyVoice3Engine.generationMode = newValue }
  }

  /// Instruct text for CosyVoice3 instruct mode
  var cosyVoice3InstructText: String {
    get { engineManager.cosyVoice3Engine.instructText }
    set { engineManager.cosyVoice3Engine.instructText = newValue }
  }

  /// Sampling parameter for CosyVoice3
  var cosyVoice3Sampling: Int {
    get { engineManager.cosyVoice3Engine.sampling }
    set { engineManager.cosyVoice3Engine.sampling = newValue }
  }

  /// Number of flow matching timesteps for CosyVoice3
  var cosyVoice3NTimesteps: Int {
    get { engineManager.cosyVoice3Engine.nTimesteps }
    set { engineManager.cosyVoice3Engine.nTimesteps = newValue }
  }

  /// Whether source audio is loaded for voice conversion
  var isCosyVoice3SourceAudioLoaded: Bool {
    engineManager.cosyVoice3Engine.isSourceAudioLoaded
  }

  /// Description of the loaded source audio
  var cosyVoice3SourceAudioDescription: String {
    engineManager.cosyVoice3Engine.sourceAudioDescription
  }

  /// Prepare source audio for voice conversion
  func prepareCosyVoice3SourceAudio(from url: URL) async throws {
    try await engineManager.cosyVoice3Engine.prepareSourceAudio(from: url)
  }

  /// Clear source audio
  func clearCosyVoice3SourceAudio() async {
    await engineManager.cosyVoice3Engine.clearSourceAudio()
  }

  /// Generate voice conversion
  func generateCosyVoice3VoiceConversion() async throws -> AudioResult {
    try await engineManager.cosyVoice3Engine.generateVoiceConversion(
      speaker: engineManager.cosyVoice3Speaker
    )
  }
}

// MARK: - OuteTTS Speaker Profile

extension AppState {
  /// Current speaker profile for OuteTTS
  var outeTTSSpeaker: OuteTTSSpeakerProfile? {
    get { engineManager.outeTTSSpeaker }
    set { engineManager.outeTTSSpeaker = newValue }
  }

  /// Whether a speaker profile is loaded for OuteTTS
  var isOuteTTSSpeakerLoaded: Bool {
    engineManager.outeTTSSpeaker != nil
  }

  /// Description of current speaker profile
  var outeTTSSpeakerDescription: String {
    if let speaker = engineManager.outeTTSSpeaker {
      let wordCount = speaker.words.count
      return "Custom speaker (\(wordCount) words)"
    }
    return "Default speaker"
  }

  /// Load speaker profile from a URL (local file)
  func loadOuteTTSSpeaker(from url: URL) async throws {
    let speaker = try await OuteTTSSpeakerProfile.load(from: url.path)
    engineManager.outeTTSSpeaker = speaker
  }

  /// Load the default speaker profile
  func loadDefaultOuteTTSSpeaker() async throws {
    // Setting to nil means the engine will use the bundled default
    engineManager.outeTTSSpeaker = nil
  }

  /// Create and load speaker profile from an audio file
  func createOuteTTSSpeaker(from url: URL) async throws {
    let speaker = try await engineManager.outeTTSEngine.createSpeakerProfile(from: url)
    engineManager.outeTTSSpeaker = speaker
  }

  /// Temperature for OuteTTS
  var outeTTSTemperature: Float {
    get { engineManager.outeTTSEngine.temperature }
    set { engineManager.outeTTSEngine.temperature = newValue }
  }

  /// Top-P for OuteTTS
  var outeTTSTopP: Float {
    get { engineManager.outeTTSEngine.topP }
    set { engineManager.outeTTSEngine.topP = newValue }
  }
}


