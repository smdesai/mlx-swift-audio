// Copyright © Anthony DePasquale

// MARK: - Async Stream Transform Helpers

/// Internal iterator wrapper for simple stream transformations.
/// Marked @unchecked Sendable because the iterator is accessed sequentially by design.
@usableFromInline
final class AsyncStreamTransformIterator<Input, Output>: @unchecked Sendable {
  @usableFromInline var iterator: AsyncThrowingStream<Input, Error>.Iterator
  @usableFromInline let transform: (Input) -> Output

  @usableFromInline
  init(_ stream: AsyncThrowingStream<Input, Error>, transform: @escaping (Input) -> Output) {
    iterator = stream.makeAsyncIterator()
    self.transform = transform
  }

  @usableFromInline
  func next() async throws -> Output? {
    guard let value = try await iterator.next() else { return nil }
    return transform(value)
  }
}

// MARK: - Public Functions

/// Transform an async throwing stream using the pull-based (unfolding) pattern.
///
/// This helper eliminates boilerplate for simple stream transformations while
/// ensuring proper async behavior. The transform closure is called for each
/// element as it's pulled from the stream.
///
/// - Parameters:
///   - stream: The source async throwing stream
///   - transform: A closure that transforms each element
/// - Returns: A new async throwing stream with transformed elements
@inlinable
func mapAsyncStream<Input, Output: Sendable>(
  _ stream: AsyncThrowingStream<Input, Error>,
  transform: @escaping (Input) -> Output
) -> AsyncThrowingStream<Output, Error> {
  let box = AsyncStreamTransformIterator(stream, transform: transform)
  return AsyncThrowingStream { try await box.next() }
}
