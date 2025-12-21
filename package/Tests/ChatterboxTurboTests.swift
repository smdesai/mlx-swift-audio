// Copyright © Anthony DePasquale

import Foundation
import MLX
import Testing

@testable import MLXAudio

// IMPORTANT: See ChatterboxTurboTestHelper.swift for memory management notes.
// Run only ONE test suite at a time to avoid loading multiple ~2GB models.
//
// These tests are DISABLED by default to avoid memory issues when running benchmarks.
// To run these tests, remove the .disabled trait below.

@Suite(.serialized, .disabled("Disabled to avoid loading multiple models during benchmarks"))
struct ChatterboxTurboTests {
  @Test @MainActor func chatterboxTurboModelLoads() async throws {
    print("Testing ChatterboxTurboModel.load()...")

    let model = try await ChatterboxTurboTestHelper.getOrLoadModel()

    print("Model loaded successfully (shared)")

    // Check model components are initialized
    _ = model.t3
    _ = model.s3gen
    _ = model.ve
    _ = model.s3Tokenizer
    #expect(model.textTokenizer != nil, "GPT-2 tokenizer should be loaded")
    #expect(model.sampleRate == ChatterboxTurboConstants.s3genSr, "Sample rate should be 24kHz")

    print("ChatterboxTurboModel load test passed!")
  }

  @Test @MainActor func chatterboxTurboHasPrecomputedConds() async throws {
    print("Testing pre-computed conditionals...")

    let model = try await ChatterboxTurboTestHelper.getOrLoadModel()

    // Check if model has pre-computed conditionals (should be bundled with model)
    if let conds = model.conds {
      print("Pre-computed conditionals found")
      #expect(conds.t3.speakerEmb.shape[0] > 0, "Speaker embedding should exist")
      if let tokens = conds.t3.condPromptSpeechTokens {
        #expect(tokens.shape[1] > 0, "Conditioning tokens should exist")
      }
      print("Pre-computed conditionals test passed!")
    } else {
      print("No pre-computed conditionals (this is OK, model can still work with reference audio)")
    }
  }

  @Test @MainActor func chatterboxTurboGeneratesWithBuiltinConds() async throws {
    print("Testing generation with built-in conditionals...")

    let model = try await ChatterboxTurboTestHelper.getOrLoadModel()

    guard model.conds != nil else {
      print("Skipping test: No built-in conditionals available")
      return
    }

    let testText = "Hello world."
    print("Generating speech for: \"\(testText)\"")

    let startTime = CFAbsoluteTimeGetCurrent()
    let audio = model.generate(
      text: testText,
      temperature: 0.8,
      topK: 1000,
      maxNewTokens: 200
    )
    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

    audio.eval()
    let sampleCount = audio.shape[0]

    print("Generated \(sampleCount) samples in \(String(format: "%.2f", processingTime)) sec.")
    #expect(sampleCount > 0, "Should generate some audio samples")

    let audioDuration = Double(sampleCount) / Double(model.sampleRate)
    let rtf = processingTime / audioDuration
    print("Audio duration: \(String(format: "%.2f", audioDuration)) sec., RTF: \(String(format: "%.2f", rtf))")

    print("ChatterboxTurbo generation test passed!")
  }

  @Test @MainActor func chatterboxTurboTTSActorLoads() async throws {
    print("Testing ChatterboxTurboTTS actor load...")

    let tts = try await ChatterboxTurboTTS.load(quantization: .q4)

    #expect(await tts.sampleRate == ChatterboxTurboConstants.s3genSr, "Sample rate should be 24kHz")

    print("ChatterboxTurboTTS actor load test passed!")
  }

  @Test @MainActor func chatterboxTurboTTSGenerates() async throws {
    print("Testing ChatterboxTurboTTS generation...")

    let tts = try await ChatterboxTurboTTS.load(quantization: .q4)

    let testText = "Hello world."
    print("Generating speech for: \"\(testText)\"")

    let result = await tts.generate(
      text: testText,
      temperature: 0.8,
      topK: 1000,
      maxNewTokens: 200
    )

    print("Generated \(result.audio.count) samples at \(result.sampleRate)Hz in \(String(format: "%.2f", result.processingTime)) sec.")
    #expect(result.audio.count > 0, "Should generate some audio samples")
    #expect(result.sampleRate == ChatterboxTurboConstants.s3genSr, "Sample rate should be 24kHz")

    let audioDuration = Double(result.audio.count) / Double(result.sampleRate)
    let rtf = result.processingTime / audioDuration
    print("Audio duration: \(String(format: "%.2f", audioDuration)) sec., RTF: \(String(format: "%.2f", rtf))")

    print("ChatterboxTurboTTS generation test passed!")
  }
}
