# Slimmer JSON decoding (and the panic boundary)

serde's derived `Deserialize` writes a full visitor per type, and every concrete deserializer it meets (serde_json's
`StrRead`, `SliceRead`, `Value`, `serde_path_to_error`'s wrappers) compiles that visitor again in the crate that calls
it. The phonemizer (vernacula-phonemizer, `rust/vernacula-json-derive` and `src/core/from_json.rs`) replaced it with a
`#[derive(FromJson)]` that decodes a parsed `serde_json::Value` through helpers generic only over the field type, and
kept about a fifth of its decoding code. cuda_binary_size Run 62 measured our side by binary size (~5 MiB of serde
decoding in the library, ~2.8 MiB of it our own types); this log measures it by build cost.

## Run 1 - 2026-10-29 10:50

**Question:** how much of the heavy crates' IR is serde, whose types is it, and what serde features would a
replacement have to cover?

**Command:** `cargo llvm-lines -p <crate> --lib` (CPU test features) for each heavy crate, functions grouped as
deserialize (`serde_core::de`, `Deserialize`), `serde_path_to_error`, serialize and utoipa schemas; deserialize then
split by the crate that owns the decoded type. A scan of `#[derive(...)]` and `#[serde(...)]` under `crates/`.

**Finding:**

| crate | IR lines | deserialize | path_to_error | serialize | utoipa | serde share |
|---|---|---|---|---|---|---|
| inference-api | 1,916,996 | ~275k | 314,107 | ~80k | 119,133 | 35% serde + 6% schema |
| inference-core | 1,767,510 | 241,426 | - | 64,243 | - | 17% |
| inference-server-core | 1,170,517 | 197,788 | 13,069 | 159,320 | 6,245 | 32% |
| inference-models-other | 630,478 | 160,079 | - | 6,878 | - | 26% |
| inference-webui | - | 42,471 | 40,552 | 31,768 | - | - |
| inference-nn | 839,182 | 55,990 | - | 2,498 | - | 7% |
| inference-quant | 867,100 | 48,659 | - | 7,517 | - | 6% |
| inference-selection | - | 20,141 | - | 8,357 | - | - |
| inference-agent | - | 9,091 | - | 3,423 | - | - |

- Owners of the deserialize code: in core, inference_core 57k, serde_json's own 47k, inference_protocol 33k,
  inference_nn 30k, inference_quant 16k, models-speech 11k, inference_mcp 9k; in server-core, inference_protocol 65k,
  serde_json 58k, inference_selection 22k, inference_layout 10k. Nearly all of it is our own types; external types
  (tokenizers, hf_hub) are a small part here.
- A type's decoder lands in every crate that names a concrete deserializer for it: protocol's request types are
  decoded in api (twice: plain and path-to-error), server-core and core; `ModelSelected` alone is 67k in api
  (12 variants x 2 access flavors).
- In the model crates the decoders are config structs read once per load from `config.json`
  (`serde_json::from_str` in each loader), e.g. hunyuan_v1_moe 3.3k, granite 3.2k, deepseek3 3.0k lines.
- Derives: 715 `Deserialize` (protocol 156, api 90, nn 55, core 52, models-other 48, speech 42, llama 33, qwen 31,
  cli 27, webui 25, quant 23, mcp 23, ...), 565 `Serialize`, 257 `ToSchema` (protocol 130, api 122).
- `#[serde]` attributes: `default` 1,437, `skip_serializing_if` 327, `rename` 250, `rename_all` 97, `alias` 84,
  `skip` 37, `deny_unknown_fields` 37, `untagged` 35, `tag` 28, `flatten` 10, `with` 9, `deserialize_with` 6,
  `transparent` 3, `content` 1. The tagged, untagged and flattened ones sit mostly in protocol (`openai.rs` 18,
  `responses_types/items.rs` 8, `tools/request.rs` 5, `responses_types/content.rs` 5).
- Non-JSON formats: the CLI's TOML config (`toml::from_str`, `cli/config`), axum `Query` extractors in server-core
  and webui (URL-encoded). These keep serde, or go through `serde_json::Value` first.

**Implication:** a Value-based decoder removes most of ~1.2M deserialize lines and all of the 368k path-to-error
lines, which sit in api, core and server-core, the crates that compile alone at the end of a build. The
path-to-error copy exists only to name the failing field in a 400; the phonemizer's `JsonError` builds that key
path itself (`unitWords: [1]: invalid type...`), so it goes away without losing the message.

## Run 2 - 2026-10-29 11:00

**Question:** the C ABI turns panics into `INFERENCE_ERR_INTERNAL` with `catch_unwind` (`inference-ffi/src/lib.rs`,
`guard`), which is why `panic = "abort"` was ruled out (cuda_binary_size Run 13). What does unwinding cost in IR and
compile time?

**Command:** `cargo rustc -p inference-api --lib -- --emit=llvm-ir -C panic={unwind,abort}`, then the same without
`--emit`, timed after a `touch`, twice each, load ~3.

**Finding:**
- IR file: unwind 3,293,162 lines (24,564 landing pads, 82,700 `invoke`s), abort 2,774,687 (-16%).
- Rebuild of inference-api: unwind 17.3 / 17.9 s, abort 15.8 / 15.8 s (-11%).
- Cargo builds tests with `panic = "unwind"` whatever the profile says, and the libraries a test links are built the
  same way, so abort could only apply to release and bundle builds, never to the edit-test loop.
- `guard` itself is small: one `catch_unwind` per entry point, the panic payload copied into `inference_last_error`.

**Implication:** abort would cut ~10% of release compile time and the unwind tables (7.1 MiB in the bundle), but a
panic would then kill the host process (and the server, where tokio now contains a panicking request to its task).
That is the safety the boundary buys; the slimmer path is fewer panics, not a different boundary. Not pursued.

## Design - 2026-10-29

A drop-in derive, so migrating a type is a change to its derive list:

- `inference-json` (new, depends on serde_json only): `FromJson` with `absent()`, `JsonError` carrying the key path,
  `from_slice` / `from_str` / `from_value` (parse once into `Value`, then decode), the field helpers and the leaf and
  container impls (strings, numbers, bool, `Option`, `Box`, `Arc`, `Vec`, sets, maps with string or numeric keys,
  tuples, arrays, `Value`, `PathBuf`, `Duration` as needed).
- `#[derive(FromJson)]` in a proc-macro crate (`inference-json-derive`, or beside the tool macros in
  inference-macros), emitting `::inference_json::...` paths and reading `#[serde(...)]` attributes itself, so utoipa's
  `ToSchema` and serde's `Serialize` keep reading the same renames and the OpenAPI document cannot drift from the
  decoder. Serialize-only attributes (`skip_serializing_if`, `skip_serializing`) are ignored.
- Beyond the phonemizer's set: `alias`, `rename_all` in snake_case / kebab-case / SCREAMING_SNAKE_CASE / UPPERCASE,
  `tag = "..."` (internally tagged, struct and newtype variants), `tag` + `content`, externally tagged enums with
  data, `untagged` with unit and struct variants, `transparent`, generics, `skip_deserializing`. `with` /
  `deserialize_with` take a `fn(&Value) -> Result<T, JsonError>` path.
- Behaviour pinned against serde: each migrated crate's types get a test that decodes the same JSON through both and
  compares, run once while serde's derive is still there, then the derive goes.
- Out of scope: `Serialize` (cheaper per type, and on the streaming path), TOML and URL-encoded inputs.

Order, each step measured with `scripts/housekeeping.sh --ir`:

1. The runtime, the derive and their tests, plus `ModelSelected` (67k in api alone).
2. Request bodies: `request_body::parse_json` decodes through `FromJson`; the protocol and api request types switch.
   Removes `serde_path_to_error` from api, server-core and webui.
3. Model configs in the five family crates, speech and diffusion (`config.json` readers).
4. The rest of core, nn, quant, selection, mcp, layout.

## Run 3 - 2026-10-29 11:28

**Question:** step 1: does the derive decode as serde does, and what does moving `ModelSelected` and its field types
to it save?

**Command:** `cargo test -p inference-json` (each test type defined twice by a macro, once deriving serde's
`Deserialize` and once `FromJson`, the same JSON through both: equal `Debug` output or both errors);
`scripts/local_ci.sh --tests`; `make docs-regen` (OpenAPI and Python types); `cargo llvm-lines -p <crate> --lib`.

**Finding:**
- The derive matched serde on every case tried: renames and aliases, `default` (field, path, container), `skip`,
  `deny_unknown_fields`, `flatten`, `transparent`, newtype and tuple structs, generics, externally / internally /
  adjacently tagged and untagged enums (unit, newtype, tuple and struct variants), `other`, number range checks.
  Error paths read `unitWords[1]: invalid type: integer `2`, expected a string`, as `serde_path_to_error` printed.
- Migrated: `IsqType` (quant); `ModelDType`, `LoraAdapterSpec`, the normal and multimodal loader types (nn);
  `IsqOrganization`, `UqffWriteConfig` (its hand-written `Deserialize` becomes `#[serde(from = "UqffWriteSpec")]`),
  the embedding, speech, transcription and diffusion loader types, `LoraRuntimeConfig` (core); `ModelSelected`,
  `SpeechGenerationSpec`, `MmprojSelection` (selection). Each is a derive-list change.
- CPU suite: 3000 passed. `docs-regen` leaves `docs/openapi.json` and the Python types unchanged.
- IR lines:

| crate | before | after | change |
|---|---|---|---|
| inference-api | 1,916,996 | 1,800,499 | -116,497 (-6.1%) |
| inference-server-core | 1,170,517 | 1,130,591 | -39,926 (-3.4%) |
| inference-core | 1,767,510 | 1,768,834 | +1,324 |
| inference-nn | 839,182 | 842,749 | +3,567 |
| inference-quant | 867,100 | 867,755 | +655 |

  `ModelSelected`'s decoder is now 13,980 lines, once, in inference-selection; what remains of it in api is 5,440
  (its `Clone`, `Serialize` and the bridge).

**Implication:** one type took ~150k lines out of the two crates at the end of the build. Next: step 2, the request
bodies, which removes `serde_path_to_error` (252k still in api, 41k in webui, 13k in server-core).

## Run 4 - 2026-10-29 11:30

**Question (user):** keeping the boundary, is there a slimmer way to catch and report a panic than unwinding?

**Finding:** the catch and the report are already small: `guard` is one `catch_unwind` per entry point plus copying
the payload into the thread's last error. The bulk is what makes a catch possible at all: the landing pads in every
function that owns something with a destructor (24,564 in inference-api alone, Run 2) and their unwind tables. Any
way of returning to the caller after a panic has to run those destructors, so it needs the same tables.
- `panic = "abort"` plus a panic hook can still report: the hook runs before the abort, so it can hand the message
  and location to a host-registered callback or write it to the last-error slot and stderr. The host learns what
  failed but does not survive it.
- `longjmp` out of a panic skips destructors (undefined behaviour across Rust frames that own anything).
- A child process per engine survives anything, at the cost of copying every request and response across a pipe.

**Implication:** keeping a host alive after a panic means keeping unwinding; the only slimmer reporting is
report-then-abort. Not pursued unless the boundary's promise changes.

## Run 5 - 2026-10-29 13:03

**Question:** step 2: request bodies through `FromJson`, dropping `serde_path_to_error`. What does it take and save?

**Change:**
- `request_body::parse_json` parses once with `inference_json::parse_slice` and decodes with `FromJson`; a parse
  failure stays `malformed_json`, a decode failure `invalid_request_body` with the path. `serde_path_to_error` leaves
  inference-api's dependencies.
- Every `Deserialize` derive in inference-protocol, inference-api and inference-mcp swapped, plus the types they
  reach in core, nn, sandbox, code-exec and selection (found by compiling: the swap tool reruns until no
  `FromJson` bound is missing). Five types with hand-written decoders get `inference_json::via_serde!`
  (`AdapterGenerationId`, `HfConfigOverrides`, `PagedCacheType`, `PagedCacheLayers`, `ReasoningEffort`).
- The derive grew what the protocol uses: variant-level `#[serde(untagged)]` (tried after the tagged variants, as
  serde does), `with` / `deserialize_with` on tuple fields, `either::serde_untagged(_optional)` decoded natively,
  `#[json(serde)]` for a foreign field type (llguidance's grammar), and serde's own `Either` encoding
  (`{"Left": ...}`, which saved agent sessions use). `NonZero` integers in the runtime.
- The web UI's 17 JSON handlers take a `JsonBody<T: FromJson>` extractor with axum's statuses (415 / 400 / 422),
  since axum's `Json` decodes through `serde_path_to_error` too.
- Response types holding `object: &'static str` (8) drop the derive: their `Deserialize` could only ever read
  `'static` input.
- URL query types stay on serde's `Deserialize` (`SkillListQuery`, `ListLoraAdaptersQuery`, `CalibrationTarget`):
  serde_urlencoded reads numbers from strings, and a `Value` keeps the last of a repeated key where serde's derive
  rejects it (`chat_route::adapter_routes_use_the_openai_error_envelope` caught `?model=a&model=b` answering 404 for
  400). The two that are also JSON bodies get `via_serde!`.

**Finding:**
- CPU suite 3000 passed; lint clean; `docs-regen` leaves `docs/openapi.json` and the Python types unchanged.
- IR lines:

| crate | master | step 1 | step 2 | change from master |
|---|---|---|---|---|
| inference-api | 1,916,996 | 1,800,499 | 1,402,113 | -514,883 (-26.9%) |
| inference-server-core | 1,170,517 | 1,130,591 | 998,138 | -172,379 (-14.7%) |
| inference-webui | - | 470,201 | 422,812 | -47,389 (-10.1% from step 1) |
| inference-core | 1,767,510 | 1,768,834 | 1,717,207 | -50,303 (-2.8%) |

  The decoders that replace them: `FromJson` code is 75k lines in api, 20k in protocol, 14k in mcp, 13k in core.
- Behaviour change: a JSON body with a repeated key now takes the last value instead of failing with "duplicate
  field" (as `serde_json::Value` and most JSON parsers do).

- Build time. A `touch` rebuild (workspace check / test build, s) barely moves: nn 8 / 15-16 (Run 93: 8 / 16), core
  8 / 10 (9 / 10), models-other 8 / 11 (8 / 12), api 6 / 7. A touch costs the dependents' re-check and the links;
  rustc's incremental cache makes the touched crate itself nearly free, and a comment-only edit is free too (1.6 s).
  What the IR buys shows in the crate compiled whole, alone (`cargo rustc -p <crate> --lib -- -C incremental=<new
  dir>`, dependencies built, same tree otherwise; master and this branch switched in place): inference-api 17.7 ->
  13.0 s (-27%), inference-server-core 11.0 -> 9.9 s (-10%). That is the serial tail of a build after an edit below
  them, the part the user saw core and api compiling alone. `scripts/housekeeping.sh --crate-time` now measures it.
- Tests for what the derive gained (variant-level `untagged`, both Either adapters and serde's `Either` encoding,
  `NonZero`, `#[json(serde)]`) decode as serde does.

**Next:** step 3, the model configs.
