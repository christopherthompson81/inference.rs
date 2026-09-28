# Python bindings investigation

A pure-Python `inference_rs` package over the engine C ABI (`crates/inference-ffi`, ABI 0.0.10), the replacement for the
pyo3 package (#19). It mirrors the C# bindings' design (`docs/investigations/csharp_bindings_investigation.md`).

## Run 1 - 2026-09-27 19:21

Question: can ctypes bind the whole ABI, streams, blobs, media and skill arrays and host callbacks, with no compiled
extension, and pass the same engine checks as the C# bindings on the tiny checkpoint?

Design:
- `_native.py`: one `SIGNATURES` table (restype, argtypes) for every entry point and `Structure`s for the
  `#[repr(C)]` types, bound on first use by a lazy library object that also checks the ABI version, so importing the
  package never needs the library. The library is found through `INFERENCE_NATIVE_DIR`, then the checkout's
  `target/release` / `target/debug`, then the platform search.
- Inputs pass as `c_char_p` from bytes, which is never NULL for `b""` (the ABI requires a non-NULL pointer).
- The engine's native handle is reference-counted in Python: calls and open streams hold a reference, `close()` stops
  new calls, and the native engine is freed after the last reference goes. A stream opened while another thread
  closes the engine is freed rather than leaked.
- Host callbacks are two module-level `CFUNCTYPE` trampolines over a handler registry keyed by the id passed as
  user_data, as in C#, so a late call after the engine is freed fails cleanly. Handler exceptions become
  `inference_callback_result_fail`.
- ctypes releases the GIL for foreign calls, so blocking engine calls let other threads run, and callbacks from
  engine worker threads take the GIL themselves.
- Tests are stdlib `unittest` (no pytest in this environment, and none needed): `test_coverage.py` (every header entry
  point declared with its parameter count, the header's ABI version, and exported by the built library) and
  `test_engine.py`. `scripts/local_ci.sh --bindings` runs them after the C# tests.

Result: 11 tests pass in 0.45 s on the tiny checkpoint: blocking and streamed chat agree; an attached image decodes like
the same image as a data URL; completion and its stream, Anthropic Messages and its stream, Responses create, get,
delete and stream; unknown model NotFound with code model_not_found, malformed JSON InvalidRequest, unknown approval
NotFound, the model list; file round trip, deleted content NotFound, empty upload; skills stored, a frontmatter-less
SKILL.md refused; a stream runs to its end after its engine is closed, and the closed engine refuses calls; an engine
loads with a host tool and search callback, and a malformed tool is InvalidArgument.

Formatting: ruff (the repo's Python formatter; installed into a scratch virtualenv, as it is not on this machine)
reformatted the package and rewrote `Optional[X]` as `X | None`, which needs Python 3.10, so `requires-python` is
`>=3.10` (3.9 is past end of life). Three blind `except Exception` catches stay, marked: nothing may raise through a C
callback.

Not covered: a host callback actually invoked (needs a model that calls tools), layout detection from Python (needs
the layout checkpoint and an image), and packaging the library into a wheel.

Review fixes (Run 1):
- Two memory-safety holes in the layout API: nothing checked that `pixels` covered the image, and the library reads
  `stride * (height - 1) + width * bpp` bytes it cannot bound, so a short buffer was an out-of-bounds read from pure
  Python; `LayoutImage` now refuses it (and a stride shorter than a row). `LayoutModel.close()` could free the model
  under a concurrent `detect`; engines, streams and layout models now share one reference-counted `Handle` with a
  `Lease` per call.
- `__del__` finalizers are `weakref.finalize` (safe at interpreter shutdown); the Engine docstring says a host
  callback must not hold the last engine reference.
- Trampolines caught `Exception` but called `str(exception)` outside the guard, so a raising `__str__` (or a
  `BaseException`) escaped ctypes, which only prints it and left the result unset. They catch `BaseException` and
  `_fail` guards `str()`. The unit tests drive the trampolines directly with a NULL result (which the setters ignore)
  and fail if anything reaches `sys.unraisablehook`.
- `Stream.next`: an infinite timeout waits, a huge one clamps instead of wrapping in c_int64, and sub-microsecond
  ones round up. NUL-terminated inputs refuse an embedded NUL; paths may be path-like; bytes are passed without a
  copy (media and pixels were copied into ctypes buffers).
- The first-use library load takes a lock; the library search stops at the checkout root (a `Cargo.toml`) instead of
  probing every parent of an installed package.
- `setuptools>=77` for the PEP 639 license string. `--bindings` runs the Python tests first, and a missing .NET SDK
  fails the C# tests without skipping them.
- Coverage now checks each parameter's and return's C type against its ctypes binding (a swapped pointer and size
  would pass a count check), and that the package version equals the workspace's. 21 tests pass.
- The package shares the import and distribution name `inference_rs` with the pyo3 package, which still builds.
  Decision: keep the name; the next change retires pyo3 and ports its examples, stub and release scripts, so the clash
  lasts only between the two.

## Run 2 - 2026-09-27 19:54

Question: can the package be strongly typed without hand-written classes that drift from the server?

Design:
- `scripts/generate_types.py` (stdlib) writes `inference_rs/types.py` from `docs/openapi.json`, which utoipa generates
  from the Rust types and a server test keeps current. String enums become `str` Enums, objects become dataclasses
  (required fields first; a single-valued `type` Literal is a variant's tag and defaults to its value), `oneOf`
  becomes a `Union` alias, and inline tagged variants get classes named by their tag (`ResponseFormatJsonSchema`).
  Aliases are emitted after the classes, in dependency order, since an alias is evaluated when defined. 163 schemas,
  1.9k lines, ASCII.
- `_codec.py`: `to_data`/`to_json` drop None fields (the server's defaults apply); `from_data` reads JSON into the
  classes, ignoring unknown fields and reading a missing required field as None (servers leave some out when empty),
  except when choosing a union's variant, where the first variant that fits exactly wins and data no variant fits is
  kept as it came.
- `Engine` is typed: requests are the classes (or their JSON), responses and stream event data are parsed (chat and
  completion chunks, OpenResponses events). The JSON-string engine is `JsonEngine`, reachable as `engine.json`.
  Responses the schema lacked (image generation, delete confirmations) came back as dicts; see the fixes below.
- A test regenerates the file and compares; `types.py` is excluded from ruff, which would reformat it away from the
  generator's output.

First run: a hand-built response without `system_fingerprint` failed to parse as required; required fields now read
as None outside union matching. Result: 27 tests pass, including real chat responses and chunks, the Responses stream
from `OpenResponsesStreamEventResponseCreated` to `...Completed` with its `ResponseResource`, model status, file
metadata and typed NotFound errors.

Not typed yet: the engine spec (`ModelSelected` has no schema), Anthropic stream events, and the responses above.

Review fixes (Run 2):
- Every optional field is a `X | None` union, and unions were always read strictly, so one extra field or a new enum
  value anywhere inside an optional object turned the whole object (or a whole `response.completed` event) into a
  dict, silently. A union now drops None and, with one member left, reads it with the caller's leniency; otherwise it
  picks the variant by its `type` tag, and falls back to strict first-fit only for untagged unions. Lenient reading is
  consistent: anything off-shape, including null or a wrong type in a required field, is kept as it came; only a
  union's exact matching raises. Tests: a newer field and reason inside optional objects still parse; real chat,
  chunk, Responses event, model and file payloads parse with no dict left where a class was expected, and serialize
  back to their null-stripped JSON.
- Tags that reference a one-value enum (`Tool.type`, `ToolCall.type`, ...) default to it, like inline tags, so
  `Tool(function=...)` works; dataclasses are keyword-only, so a regenerated field order cannot move positional
  arguments.
- Generator hazards: quotes in descriptions are escaped, `null` in an inline enum becomes `| None`, and two properties
  that map to one attribute are an error. The `_wire` rename (unused by the schema today) has a codec test.
- Schema drift: `GET /v1/files` documented a bare array but returns `{object, data}`; the file lists, both delete
  confirmations and the image generation response now have schemas (`FileListObject`, `ContainerFileListObject`,
  `FileDeleted`, `ResponseDeleted`, `ImageGenerationResponse`), so those methods are typed too.
- `force-exclude` keeps ruff off `types.py` even when a path names it. 32 tests pass.
