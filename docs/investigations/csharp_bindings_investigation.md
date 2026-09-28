# C# bindings investigation

In-repo .NET bindings over the engine C ABI (`crates/inference-ffi`, ABI 0.0.10), following the conventions of the
AudioCpp bindings: source-generated `LibraryImport` declarations in `Native/NativeMethods.cs` that interpret nothing,
wrapper types that own lifetimes and errors, a library resolver, and CTest-style console tests (0 pass, 1 fail, 77
skip).

## Run 1 - 2026-09-27 18:55

Question: can every ABI entry point be bound and driven from C#, including streams, blobs, media and skill arrays,
and host callbacks, with the engine test running on the tiny checkpoint?

Design:
- `InferenceEngine` is JSON in, JSON out, one method per entry point. Inputs are pinned as UTF-8 for the call (a
  spare byte keeps an empty input's pointer non-NULL, which the ABI requires); media and skill files are copied into
  unmanaged memory for the call.
- The native engine is an `EngineHandle` (`SafeHandle`). Every call and every open `EngineStream` adds a reference,
  so disposing the engine on one thread cannot free it under another, and the engine outlives its streams. The
  host callbacks' `GCHandle` targets are freed in the handle's release, after `inference_engine_free`, because a
  stream keeps the native engine (and so its callbacks) alive until it too is freed.
- Host callbacks are `[UnmanagedCallersOnly]` trampolines; a handler exception becomes `inference_callback_result_fail`
  with its message. The load-time tool descriptions are freed after load (the library copies them).
- `NativeMethods.AbiVersion` is checked on first use; the coverage test checks it equals the header's version.
- `crates/inference-ffi/examples/tiny_checkpoint.rs` writes the Rust tests' tiny checkpoint for other languages.
- `scripts/local_ci.sh --bindings` builds the cdylib with `--workspace` (the tests' feature unification, so it reuses
  their artifacts; `-p inference-ffi` alone rebuilt inference-server-core and inference-ffi with other features).

First build: the engine wrapper passed native pointers through lambdas over a delegate; C# refuses fixed locals and
mixed explicit and implicit lambda parameters there, so each entry point is an explicit method over two small helpers
(`Utf8Input`, `NativeArray<T>`).

Result: coverage binds all 62 entry points at ABI 0.0.10. Engine test on the tiny checkpoint: blocking and streamed
chat agree; unknown model is NotFound with `Code` "model_not_found"; malformed JSON is InvalidRequest; file upload,
content round trip (bytes and MIME), delete then NotFound, empty file upload; skill upload, list, and a
frontmatter-less SKILL.md refused; an engine loads with a host tool and search callback and chats; a malformed tool is
InvalidArgument. Not covered: a host callback actually invoked (needs a model that calls tools; the Rust unit tests
cover the C side), layout detection from C# (needs the layout checkpoint), and CUDA.

Review fixes (Run 1):
- Callback targets could be freed while the engine still called them: detached agent loops outlive
  `inference_engine_free` (it waits at most 10 s, and a loop keeps its own reference to the callbacks), so a
  trampoline could run `GCHandle.FromIntPtr` on a freed handle. Handlers now sit in a registry keyed by an id passed
  as user_data; the engine's release removes them, and a late call with a removed id fails cleanly. The header's
  user_data contract now says a callback may run shortly after the engine is freed.
- Streams and layout models were raw pointers: a stream dropped without Dispose held its engine reference forever
  (the whole model leaked), and Dispose racing a call was a use-after-free. Both are SafeHandles now; a stream's
  handle takes the engine reference and, if the engine was disposed meanwhile, frees the native stream instead of
  leaking it. Calls hold a `Lease`, and a disposed engine refuses calls even while streams keep it loaded (the engine
  test caught that SafeHandle alone still lets new references in while one is outstanding).
- `DetectBatch` frees every result even when reading one fails, and returns early for no images.
- Inputs are pinned in place instead of copied (a 64 MiB upload was copied twice), `NativeArray` disposal is
  idempotent, and partial construction of media, skill files and callbacks cleans up.
- `InferenceException.Code` tolerates non-object error JSON; trampolines never let an exception escape (a failure to
  report a failure reports it without detail); `round` parsing tolerates non-numbers; a negative timeout other than
  infinite throws, and sub-millisecond ones round up; `TryNext` has `[NotNullWhen(true)]`.
- The coverage test also checks parameter counts against the header and that the built library exports every bound
  symbol (through the resolver's own search; `NativeLibrary.TryLoad` did not reach the resolver). `--bindings` pins
  `INFERENCE_NATIVE_DIR` to the dev build, so a stale release library cannot be tested instead, and cleans up its
  temporary checkpoint on failure.
- Engine test additions: an attached image decodes like the same image as a data URL; completion and its stream;
  Anthropic Messages and a stream from message_start to message_stop; Responses stored, fetched, deleted and
  streamed to response.completed; the model list; unknown model status and unknown approval NotFound; a deleted
  file's content NotFound; a stream runs to its end after its engine is disposed, and the disposed engine refuses
  calls.
