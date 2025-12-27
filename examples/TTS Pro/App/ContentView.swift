// Copyright © Anthony DePasquale

import SwiftUI

struct ContentView: View {
  @State private var appState = AppState()

  @Environment(\.horizontalSizeClass) private var horizontalSizeClass

  var body: some View {
    Group {
      if horizontalSizeClass == .regular {
        RegularLayoutView()
      } else {
        CompactLayoutView()
      }
    }
    .environment(appState)
    .onTapGesture {
      dismissKeyboard()
    }
  }

  private func dismissKeyboard() {
    #if os(iOS)
      UIApplication.shared.sendAction(
        #selector(UIResponder.resignFirstResponder),
        to: nil,
        from: nil,
        for: nil
      )
    #elseif os(macOS)
      NSApp.keyWindow?.makeFirstResponder(nil)
    #endif
  }
}
