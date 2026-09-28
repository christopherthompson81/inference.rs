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

## Run 3 - 2026-09-27 20:29

Question: can the engine spec be typed from the same schema as requests?

Change:
- `EngineSpec` and its parts derive `ToSchema` and are registered in the OpenAPI document (not a route body: the spec
  the C ABI and bindings load from). `ModelSelected` and its field types (`ModelDType`, the loader-type enums,
  `IsqOrganization`, `LoraAdapterSpec`, `LoraRuntimeConfig`, `AgentPermission`) derive it under the `utoipa`
  feature; `inference-nn` gains that feature, enabled by core's. utoipa has no `PathBuf` schema, so the 32 path
  fields say `value_type = String`, and `write_uqff` is described by `UqffWriteSpec` (see the fixes below).
- `ModelSelected` is externally tagged (`{"Plain": {...}}`). The generator names each such variant's fields class
  after its tag (`ModelSelectedPlain`) with an `_external` key; the codec wraps and unwraps that layer, and a union
  picks an externally tagged variant by its key.
- `Engine` takes an `EngineSpec` (or its JSON as a dict or string); the typed tests load their engine from one.
- The package's ruff line length is 120, the repo's width (ruff's default 88 had rewrapped every edit).

Result: 34 Python tests pass, including a typed spec loading the tiny checkpoint and the external tags round-tripping.

Review fixes (Run 3):
- The loader-type enums published their Rust variant names (`"Qwen3"`), which serde rejects, so a spec with `arch` from
  the generated enum failed to load. The macros take each CLI name as a `literal` fragment, which reaches the derive
  wrapped in an invisible group utoipa does not read as a rename; they take `tt` now. A core test checks every
  loader enum's schema lists exactly the names serde writes, and that each parses back.
- `write_uqff` takes a path or `{output, types?, base_model?, repo_id?}`; its private deserializer enum is now the
  public `UqffWriteSpec`, which both parses and describes it (the object variant titled `Config`, and the generator
  names union variants by schema title when they have no tag).
- A dict spec holding classes serializes (`to_json` for every spec form); the single-value tag default applies only
  to `type` (it had made `ModelSelectedSpeech.arch`, an enum with one member today, silently optional).
- Tests: a misspelled or padded external tag stays as it came leniently and fits nothing strictly; loader enum
  values serialize as the engine's names. 35 Python tests pass.

## Run 4 - 2026-09-27 (evening)

Question: can every pyo3 example and Python docs snippet move to the ctypes package, and what does the engine spec lack?

Ported 64 examples, both notebooks and 30 guide pages (four agents on examples, three on guides, each validating its
snippets by building and serializing every spec and request through `ir.to_json` against a stub engine; no model run).
The Python reference is now rendered from the package (`docs/scripts/render_python_api.py`, grouped into engine,
spec, chat, responses, anthropic, management and layout pages) instead of the pyo3 `.pyi`.

Findings:
- Only `Plain` and `Run` gave `dtype`, `max_seq_len` and `max_batch_size` serde defaults; every other `ModelSelected`
  variant required them, so a GGUF or X-LoRA spec failed to construct without spelling out AUTO/4096/1. All variants
  default them now, as do Lora's `adapters` and `runtime_config` and `LoraRuntimeConfig`'s fields.
- No equivalent in the engine spec (dropped, docs say so): paged-attention pool/block size, `num_device_layers`,
  MTP speculative decoding, `enable_search` (reranking), default-model get/set, session export/import, the approval
  callback (approvals now arrive as stream events answered by `resolve_approval`).
- Still on pyo3 until the spec grows them: AnyMoE (3 examples), MCP client, code execution (2), shell (2), online
  calibration. `tests/test_examples.py` pins that list and fails once one of them is ported without leaving it.
- Two old examples named the wrong architecture (Mistral-Small-3.1 as Gemma3, Phi-3.5-MoE as Mistral); fixed.

Review fixes (Run 4), from running every example and guide snippet against a stub engine that sent each spec to the
real `inference_engine_load` and each request to a tiny engine:
- Seven examples and both notebooks sent `model="<nickname>"` (pyo3 ignored it); the engine answers `model_not_found`
  for anything but `default` or the real id. `test_multi_model.py` unloaded `default`, which the model calls reject.
- Stream loops read `event.data.choices` on every event, so an `error` event raised `AttributeError` over the real
  error; they check `event.name` now.
- The schema carried no default for serde `default = "fn"` fields; `schema(default = fn)` puts them in, and the type
  generator now writes a schema's scalar or enum default as the field's Python default (`max_seq_len: int | None =
  4096`), so the reference shows it and a round trip writes it out.
- `inference-ffi` only had `cuda`/`metal`; it now passes through every accelerator and functional feature pyo3 had.
- Stale prose: README, the from-source page and the NVFP4 note still gave pip/maturin steps, and the reference said
  enum members have no `.value`.

`tests/test_examples.py` resolves names through every import form (`import inference_rs.types as t`, `from
inference_rs.types import X`, `ir.types.X`), follows engines bound by `with`, assignment or annotation to check method
names, checks every `t.X(...)` keyword construction, and covers the Python snippets of every guide page that imports
the package (27 pages, 102 sources). It does not check attributes of responses or positional constructions.

Next: engine spec fields for AnyMoE, MCP, code execution/shell and calibration so the last nine examples port; then
wheels bundling `libinference_ffi`; then removing the pyo3 crate.

## Run 5 - 2026-09-28

Question: what does the engine spec need so the last pyo3 examples port, and what does it cost to expose it?

Finding: almost nothing new was needed below the spec. `InferenceRsForServerBuilder` already took MCP, code execution,
shell, MTP, device layers and paged-cache sizing (the CLI sets them), and `ModelLoaderConfig` already had an
`overrides.anymoe` slot that wraps the loader; the server builder passed `Default::default()`. The spec gained
`runtime.{device_layers, paged_cache, mtp}`, `agentic.{search, mcp, code_execution, shell, sandbox}` and `anymoe`,
and the eight examples plus four guides ported. Only `online_calibration.py` stays on pyo3: calibration is runtime
operations (begin, status, apply), not configuration, so it needs ABI entry points.

Review fixes:
- A bare `code_execution`/`shell` config ran model-written code unsandboxed with no approval, where the CLI
  sandboxes by default. `agentic.sandbox` (auto/on/off, honoring `INFERENCE_RS_SANDBOX`, the resolution now shared
  with the CLI in `inference_sandbox::SandboxMode`) gives a policy-less config the developer profile.
- The embedded configs accepted unknown keys, so `"sandbox_polcy"` parsed and ran unsandboxed; code execution,
  shell, sandbox policy and AnyMoE configs deny unknown fields (MCP stays lenient: its reference JSON has `_comment`
  keys).
- `device_layers` parsing panicked on a bad entry (an internal error across the ABI); it returns an error, checked at
  spec time. Two paged-cache sizes were silently resolved by order, all three turned paged attention off; the spec
  refuses more than one.
- `McpServerConfig`'s container `serde(default)` made utoipa evaluate `Default`, writing a random UUID into the schema
  (and so into `types.py`) on every regeneration; it now has field defaults only.
- `PagedCacheType` deserialized only `"Auto"`, though the TOML reference documents `auto`; its serde names are
  lowercase now.

Next: calibration, re-ISQ, tokenize/detokenize, session and default-model operations as ABI entry points, which is
what the pyo3 `Runner` still has over the ABI; then wheels; then removing pyo3.
