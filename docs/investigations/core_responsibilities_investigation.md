# inference-core responsibilities investigation

What `inference-core` still owns beyond the pipeline infrastructure and the engine, how much of its IR each part
costs, and which parts can leave. Follows the loader series (`loaders_investigation.md`), which took core from
3,025,589 to 2,508,386 LLVM IR lines.

## Run 1 — 2026-09-29 18:34

**Question:** which non-essential concerns remain in core, and how big is each one?

**Commands:**
- Source lines per top-level module: `find <mod> -name '*.rs' | xargs cat | wc -l` over `crates/inference-core/src`.
- IR by module: `cargo llvm-lines -p inference-core --lib --features cuda`, saved from the loader series' final
  build as `irbuild.fns8.txt` in the scratchpad. Each function counts toward the first `inference_core::<module>`
  path in its name, so a generic instantiation over a core type is charged to that module. Crate hashes are stripped
  first.

**Raw finding:** of the 2,508,386 total, 1,296,615 lines name a core module. The other 1,211,771 are generic code
from other crates with no core type in the name: `core` 199k, `alloc` 168k, `serde_json` 157k, `tokenizers` 99k,
`hashbrown` 74k, `serde` 57k, `tokio` 46k, `inference_nn` 44k, `std` 43k, `html5ever` 40k, `candle_core` 35k,
`html2text` 35k, `inference_quant` 32k, `inference_protocol` 26k.

Grouped by concern:

| Concern | Modules | Source lines | IR lines |
|---|---|---|---|
| Agent and tool layer | `engine/{agentic_loop, tool_dispatch, agentic_session, file_tools}`, `search/`, `remote_fetch`, `chat_collector`, `agent_approval`, plus the HTML crates | ~5.5k | ~240k |
| GGUF format | `gguf/`, `pipeline/{gguf, ggml}` | ~16k | ~140k |
| Model selection and hardware fit | `selection/`, `tuning`, `diagnostics`, `resource_plan` | ~6k | ~75k |
| Adapters and AnyMoE | `adapter/`, `pipeline/amoe` | ~3k | ~46k |
| Chat templates and tokenizers | `pipeline/{chat_template, tokenizer, tiktoken}` | ~2k | ~57k |
| Speech and diffusion pipelines | `pipeline/{speech, diffusion}` | small | ~12k |

Largest core-owned modules that are arguably essential: `sequence` 86.6k (from 3.1k source lines), `engine` 59.5k,
`inference_rs` 49.4k, `distributed` 45.1k (from 854 lines), `speculative` 44.8k.

The agent layer's ~240k, by part: `agentic_loop` 46,085, `html5ever` 39,974, `tool_dispatch` 39,292, `html2text`
35,134, `search` 31,268, `markup5ever` 11,188, `agentic_session` 9,415, `inference_mcp` 7,992, `tendril` 4,486,
`remote_fetch` 4,147, the MCP and code-exec setup in `inference_rs` 3,140, `chat_collector` 2,794, `file_tools` 2,578,
`bm25` 1,023, `agent_approval` 740, `selectors` 519, `scraper` 199. That is about 9.6% of core, before its share of
the unattributed `serde_json`/`tokio` instantiations.

**What the agent layer touches in the engine** (mapped in `engine/agentic_loop.rs`, `engine/tool_dispatch.rs`,
`engine/add_request.rs` and `inference_rs/`):
- Entry: `Engine::handle_request` sends a chat request to `agentic_loop(Arc<Engine>, NormalRequest)` when search,
  registered tools, `max_tool_rounds`, `tool_dispatch_url` or input files are set, and the request isn't one of the
  loop's own inner requests (`AGENTIC_LOOP_REENTRY_SENTINEL`).
- Inner requests: the loop sends each round's request back through `engine.tx` as a `Request::Normal`, then reads the
  response channel. It never calls the pipeline to generate.
- Pipeline reads: modalities (to decide whether image tool results are allowed) and the tokenizer (to size search and
  extraction results).
- Engine state it uses: `tool_callbacks`, `search_callback`, `search_pipeline` (the RAG embedding model, loaded by
  core's `EmbeddingLoaderBuilder` and run as a `Pipeline`), `session_store`, `file_store`, and `handles` (it registers
  its spawned task there).
- Configuration: `InferenceRsBuilder` and `EngineConfig` carry `search_embedding_model`, `search_callback` and
  `tool_callbacks`. `InferenceRs::init_external_tool_callbacks` starts the MCP client and the code-exec and shell
  managers and merges their callbacks in. `inference_rs/sessions.rs` exposes the session and file stores.
- Request types in core (`request.rs`) hold `inference_mcp` option and notifier types, so core keeps
  `inference-mcp` as a dependency whichever way the loop moves.
- Outside core: `inference-api` (`engine_chat.rs`, `media_source.rs`, `inference_for_server_builder.rs`),
  `inference-ffi` (`callbacks.rs`), the SDK (`lib.rs`, `messages.rs`) and the CLI's interactive mode use
  `ChatResponseCollector`, the approval types and the search callback types.

**Implication:** the loop is a client of the engine. It only drives the engine through the request channel and reads
metadata, so it can move to a crate above core. There are two possible seams:
- A. A core trait the engine holds (like `MultimodalProcessorFactory`): `handle_request` hands agentic requests to
  an `Arc<dyn AgentRunner>` installed through the builder. The engine channel's behavior stays the same for every
  consumer. Each builder above core (api, SDK) must install it.
- B. The loop runs above the engine, at the api's request dispatch, and the engine only generates. That is the
  cleanest layering, but anyone who sends on the raw engine sender loses agentic requests unless the sender is
  wrapped.

## Run 2 — 2026-09-29 19:40

**Question:** how much IR leaves core when the agent layer moves to `inference-agent` behind seam A (the
`AgentRunner` trait)?

**What moved:** `engine/{agentic_loop, tool_dispatch, file_tools}`, the web fetch, HTML parsing and search tool
definitions from `search/mod.rs`, and the chunking, BM25 and ranking from `search/rag.rs`. Core keeps these parts of
the seam:
- In `engine/agent.rs`: the `AgentRunner` trait, the tool-name predicates, the loop constants and the engine accessors.
- In `search/`: the search types and `SearchEmbedder` (the embedding model's load and embed half of the old
  `SearchPipeline`).
- The session store.

`scraper`, `html2text`, `bm25` and `urlencoding` leave core's dependencies. `RebootState` now holds an
`EngineConfig` instead of copying its eight fields at five sites. `Engine::new` takes an `EngineParts` struct instead
of 15 arguments.

**Commands:** `CARGO_TARGET_DIR=<scratch> cargo llvm-lines -p inference-core --lib --features cuda`, and the same
for `-p inference-agent --lib --features inference-core/cuda`.

**Raw finding:**
- `inference-core`: 2,508,386 → 2,230,180 lines (-278,206, -11.1%). Function copies: 52,879 → 46,762.
- `inference-agent`: 366,048 lines, 9,081 copies. That includes 118,168 lines from `html5ever`, `html2text` and
  `markup5ever`, which Run 1 undercounted (only functions with those crates first in the name were counted there).
- The agent crate is larger than what left core because its generic instantiations (serde_json, tokio, alloc) are
  compiled again in the new crate rather than shared with core's copies.

**Implication:** the agent crate sits between core and `inference-api`, which depends on it, not beside it. Its codegen
runs alongside api's front end (cargo pipelines on `.rmeta`), so what it adds to the critical path is its own front end.
Core's serial monomorphisation and lowering shrink by 11%. A cold-build timing, run once CI is green, settles the
net effect.

## Run 3 — 2026-09-29 19:05

**Question:** does the agent crate move change the cold CUDA test build's wall time?

**Command:** the cold build from `cuda_build_time_investigation.md` Run 39:
`CARGO_TARGET_DIR=<scratch> cargo test --no-run --features cuda --workspace --lib --bins --tests --timings`.
Load was 3.35 at the start and 17.8 at the end, which matches that doc's loaded Run 49 (candle-kernels build script
66.4 s here, 65.8 s there) rather than its quiet rerun (61.2 s).

**Raw finding** (all times in seconds; loaded Run 49 in brackets):
- Totals: wall 245.2 (243.4), 2,292 unit-seconds (2,262), 794 units (792).
- `inference-core` lib: 66.8 (70.5), from t=123.4 to 190.2.
- `inference-core` lib test: 85.3 (94.5), from t=155.5 to 240.8.
- `inference-agent` lib: 16.7, from t=170.6 to 187.2. `inference-api` starts at t=172.2, so waiting on the agent
  crate's metadata costs it 1.6 s.
- `inference-api` lib: 33.9 (31.4). The api lib test takes 31.9 (42.1) and ends at t=224.2.
- The last unit is the CLI test binary: 21.0 (32.3), t=224.2 to 245.2. It starts only when the api lib test ends.

**Implication:**
- Core's own units shrink by 5 to 10%, but wall time doesn't move. The tail is the downstream chain: core's metadata,
  then `inference-api`, its lib test, and the CLI test binary.
- `inference-api` and `inference-server-core` are the next wall-time targets; by the same per-crate `cargo llvm-lines`
  survey that gave Run 1 its core figure, they are 1.37M and 1.15M IR lines.
- Moving GGUF out of core (step 2) would shorten core's lib test and metadata, but on its own won't cut wall time
  while the api chain sets the end.

## Run 4 — 2026-09-29 19:40

**Question:** where do `inference-api` (1,371,600 IR lines from 15.1k source lines) and `inference-server-core`
(1,148,114 from 6.6k) get their IR? They now end the cold build (Run 3). Core manages about 22 IR lines per source
line; these two produce 90 and 175.

**Command:** `CARGO_TARGET_DIR=<scratch> cargo llvm-lines -p <crate> --lib --features cuda`, grouped by module, by the
kind of code, and by the deserializer or serializer type a serde function is instantiated for.

**Raw finding:**
- **`inference-api`:** 571,098 lines name an api module and 800,502 are generic code from other crates. Top modules:
  `blocking` 168,270 (from a 254-line file), `anthropic` 76,286, `responses` 66,153, `engine` 62,812.
- **`inference-server-core`:** 202,660 lines name a server-core module and 945,454 don't, of which 80,358 are
  `inference_api` generics instantiated again here (for example `engine_chat::parse_request::{closure#0}`, 9,583).
  Top modules: `openapi_doc` 41,089 (`ApiDoc::openapi` alone 19,248) and `handlers` 38,922.
- **`blocking`:** each of its 24 methods spawns its own future type, so tokio's task harness, task core and
  scheduler code are instantiated once per method: 7.3k lines for a plain call and 13k for a streaming one, 190,708 in
  total. Tokio task and scheduler code is 258,002 lines in the api, 143,924 of it from `BlockingEngine`.
- **serde is 607,639 lines in the api and 683,116 in server-core.** The same protocol request types are deserialized
  through different deserializers in the two crates:

  | Instantiated for | `inference-api` | `inference-server-core` |
  |---|---|---|
  | serde_json `SliceRead` (the api's `parse_json`, the C ABI path) | 233,518 | 17,349 |
  | `serde_path_to_error` (inside axum's `Json<T>`, one per request extractor) | 0 | 176,599 |
  | serde_json `Value` | 31,578 | 50,442 |
  | Untagged-enum buffering (`Content`) | 30,686 | 28,517 |
  | serde_json `StrRead` | 18,917 | 4,208 |
  | Serializers | 16,513 | 44,941 |
  | Other serde | 197,629 | 228,643 |

  `ChatCompletionRequest::visit_map` appears three times in server-core, under three deserializers.
  `ResponseResource::serialize` appears three times, under three serializers.
- **`ModelSelected`'s `Deserialize`:** 14 instantiations in the api, about 40k lines for one enum.
- **utoipa (OpenAPI schemas):** 78,798 lines in the api and 132,417 in server-core.

**Implication, in order of confidence:**
1. Spawn one type-erased job in `blocking` (a boxed `dyn Future<Output = ()>` that sends its result back), so the
   tokio harness is instantiated once. Expect about 180k lines out of the api.
2. Give the HTTP server and the C ABI one request-parse path: a non-generic parse function per request type in the
   api, which server-core calls from a bytes extractor instead of axum's `Json<T>`. It needs to keep axum's status split
   (415 wrong content type, 400 bad syntax, 422 bad data) and the field path in the error. Expect most of
   server-core's 177k `path_to_error` lines out.
3. Find why api generics such as `engine_chat::parse_request` are instantiated again in server-core (80k).
4. Collapse the 14 `ModelSelected` deserializer instantiations.

## Run 5 — 2026-09-29 20:30

**Question:** how much IR does erasing `blocking`'s spawned futures save?

**Change:**
- Every `BlockingEngine` call now spawns the same `Pin<Box<dyn Future<Output = Box<dyn Any + Send>>>>`, and `run<T>`
  downcasts its output.
- A first cut passed the result back through `std::sync::mpsc::sync_channel`: 1,243,332 lines. That still
  instantiated std's channel (the zero, list and array flavors, about 18k lines) for each result type.
- A second cut used an `Arc<Mutex<Option<T>>>` slot: 1,201,091 lines.
- The `dyn Any` output, suggested in review, is the simplest of the three and the smallest.

**Command:** `CARGO_TARGET_DIR=<scratch> cargo llvm-lines -p inference-api --lib --features cuda`.

**Raw finding:** `inference-api` went from 1,371,600 to 1,195,272 lines (-176,328, -12.9%). Function copies went
from 31,081 to 25,363.

**Implication:** step 1 of Run 4 is done. Next is step 2, one request-parse path shared by the HTTP server and the
C ABI.

## Run 6 — 2026-09-29 19:41

**Question:** what is the disk cost of sharing generics across crates, with debug info off? Stable rustc shares them
only at opt-level 0, 1, `"s"` or `"z"`, so the test puts the workspace crates at opt-level 1 and keeps dependencies
at 3.

**Commands:** in one scratch target, with `--config profile.dev.debug=false`:
- `cargo build -p inference-cli --features cuda` as the profile stands (everything at opt-level 3).
- The same with `--config profile.dev.opt-level=1 --config 'profile.dev.package."*".opt-level=3'`.

**Raw finding:**

| Measure | All crates at 3 | Workspace at 1, dependencies at 3 | Change |
|---|---|---|---|
| Workspace rlibs | 305.4 MB | 278.8 MB | -8.7% |
| `inference` CLI binary | 177.2 MB | 154.7 MB | -12.7% |
| Build wall time | 172.5 s | 192.2 s | +11% |

- Rlibs by crate, opt-level 3 → 1 (MB): core 53.5 → 53.6, inference-nn 36.9 → 35.7, inference-quant 26.2 → 30.2,
  inference-api 24.8 → 22.5, inference-server-core 24.8 → 18.7, models-llama 18.8 → 15.8, models-gemma 15.5 → 12.1,
  models-speech 5.1 → 1.3.
- The exported instantiations are outweighed by opt-level 1's smaller code: only quant, mcp, audio and paged-attn grow.
- The second build recompiled 344 of the first build's 520 crates, dependencies included (all rlibs
  1,046.6 → 1,910.2 MB, the old set kept beside the new), so both builds are close to cold.

**Implication:** sharing doesn't cost disk. Rlibs and the binary both shrink. The unexplained number is the wall time:
+20 s despite less optimization. The second build ran at a higher load, so the next step is a timed pair on a quiet
machine, plus the test-suite runtimes, before any profile change.

## Run 7 — 2026-09-29 21:10

**Question:** how much IR does one shared request-parse path save?

**Change:**
- `inference_api::request_body::parse_json` runs `serde_path_to_error` over serde_json's byte-slice deserializer and
  produces axum's messages and codes (`malformed_json` for a syntax error, `invalid_request_body` for a data error).
- `JsonRequest` has one concrete impl per request type, so each deserializer compiles in the api only. The engine's
  `*_json` methods (the C ABI path) parse through it.
- server-core's handlers extract `ApiJson<T>`, which checks the content type, reads the bytes and calls
  `T::from_json`, instead of axum's `Json<T>`. Only the MCP endpoint keeps `Json`, since it maps syntax errors to
  JSON-RPC codes.

**Command:** `CARGO_TARGET_DIR=<scratch> cargo llvm-lines -p <crate> --lib --features cuda`.

**Raw finding:**

| Crate | Before | After | Change |
|---|---|---|---|
| `inference-server-core` | 1,148,114 | 923,139 | -224,975 (-19.6%) |
| `inference-api` | 1,195,272 | 1,258,892 | +63,620 |
| Both | 2,343,386 | 2,182,031 | -161,355 |

- `serde_path_to_error` code is now 17,258 lines in server-core (was 176,599) and 171,872 in the api (was 0).
- The api's plain `SliceRead` instantiations fell from 233,518 to 122,780. So the request types cost about 110k lines
  under a bare `from_slice` and 172k under `serde_path_to_error`, about 1.55 times as much.
- The remaining `SliceRead` code is mostly the engine spec: `ModelSelected`'s 14 deserializer instantiations, about
  40k lines.

**Implication:**
- Parsing with a bare `from_slice` would save another ~60k, but errors would lose the failing field's path, which
  the HTTP API reports today. The path stays.
- The C ABI's errors improve: they now carry the path, and a syntax error reports `malformed_json`, as HTTP does.
- Next is step 4, `ModelSelected`'s deserializer.

## Run 8 — 2026-09-29 22:00

**Question:** why are the api's async fns compiled again in server-core, and what does boxing them at the crate
boundary save?

**Finding first:** Run 4's step 4 was wrong. `ModelSelected`'s "14 copies" are its 14 variants' generated visitors,
not 14 deserializer types. It has one deserializer, and its size (about 50k lines) comes from serde deriving a map and a
sequence visitor for each variant. There is nothing to deduplicate.

**Cause:**
- A non-generic `async fn` body is a coroutine polled through the generic `Future` impl. Without shared generics
  (the dev profile's opt-level 3; see Run 6), every crate that awaits it compiles its own copy, along with everything
  it awaits in turn.
- `engine_chat::parse_request::{closure#0}` is 9,330 lines in the api and 9,583 in server-core.
- In total, server-core held 98,529 lines of api and core async bodies, headed by `parse_request` 13,378,
  `load_adapter` 13,078, `embed` 11,191 and `decode_video_ffmpeg` 4,748 (awaited by `parse_request`).

**Change:**
- Each entry point other crates await is now a plain `pub fn` returning `BoxFuture<'a, T>`, whose body is
  `Box::pin(<name>_inner(..))`. The unchanged body becomes a private `async fn <name>_inner`. The unsizing to
  `dyn Future` happens in a non-generic api function, so the coroutine compiles there only.
- A first cut wrapped each body in `Box::pin(async move { ... })` directly. It gave the same IR, but re-indenting the
  bodies rewrapped about 1,200 lines in `engine_chat.rs` alone.
- Converted: `engine_chat::{parse_request, ChatEngine::prepare, collect_chat}`,
  `lora_adapters::{load_adapter, unload_adapter, list_adapters}`, `engine_embeddings::embed`,
  `responses::{prepare_response, collect_response}`, `engine_completion::{prepare_completion, collect_completion}`,
  `anthropic::{prepare_messages, collect_messages}`, `generation::{generate_image, generate_speech}`,
  `models::reload_model` and `operations::calibration`.
- The cost is one allocation per request-level call.

**Command:** `CARGO_TARGET_DIR=<scratch> cargo llvm-lines -p <crate> --lib --features cuda`. The webui was measured
without features (it has no `cuda` feature) and the CLI with `--bin inference`. Master and the change were measured in
the same scratch target.

**Raw finding:**

| Crate | Master | Boxed entry points | Change |
|---|---|---|---|
| `inference-server-core` | 923,269 | 677,037 | -246,232 (-26.7%) |
| `inference-webui` | 319,629 | 305,646 | -13,983 |
| `inference-api` | 1,258,892 | 1,260,648 | +1,756 |
| `inference` CLI binary | 1,323,945 | 1,320,324 | -3,621 |
| `inference-ffi` | 104,003 | 104,003 | 0 |

- Server-core's saving is well beyond the 98k of async bodies. Those bodies also pulled in the serde, channel and
  futures code they instantiate.
- 19,419 lines of api and core async bodies remain in server-core. The largest is core's
  `InferenceRs::do_reload_model` (2,861), now reached through the boxed api `reload_model`; the rest are under 1k each.

**Implication:** box an async fn at the crate boundary whenever another workspace crate awaits it. Core's own public
async API, which the api awaits, is the next place to look.

## Run 9 — 2026-09-29 21:20

**Question:** does the Run 8 pattern pay one level down, for core's public async fns that the api, the SDK and the
CLI await?

**Change:** the same wrapper pattern for core's externally awaited entry points:
- `InferenceRsBuilder::build` and `ModelLoaderConfig::load`.
- `InferenceRs::{add_model, reload_model, send_request_async, shutdown}`.
- The LoRA list, status, load and unload calls. For the two `*_with_policy` loaders, the wrapper converts the
  `impl Into` arguments before boxing, so the inner fn is non-generic.
- `selection::quant::{resolve_model_quant, resolve_quant, read_existing_uqff_report}`.
- `pipeline::hf::{list_model_files, read_model_file_range}` and `remote_fetch::{fetch_limited, fetch_url}`.

**Commands:**
- IR: `cargo llvm-lines` per crate, master and the branch, in one scratch target.
- Build time: in one scratch target with `CARGO_INCREMENTAL=0`, a warm-up build of the branch, then
  `cargo test --no-run --features cuda --workspace --lib --bins --tests --timings` on master and then on the branch.
  Only core and its dependents rebuild. Load was about 14 during both builds.

**Raw finding, IR lines:**

| Crate | Master | Branch | Change |
|---|---|---|---|
| `inference-core` | 2,230,180 | 2,316,051 | +85,871 (+3.9%) |
| `inference-api` | 1,260,648 | 1,093,479 | -167,169 (-13.3%) |
| `inference` (SDK) | 142,393 | 77,368 | -65,025 (-45.7%) |
| `inference` CLI binary | 1,320,324 | 1,151,826 | -168,498 (-12.8%) |
| `inference-server-core` | 677,037 | 676,823 | -214 |

Net: -315k lines. Core grows because a `pub async fn` that nothing in core awaits was never compiled in core. Its body
was compiled only in each crate that awaited it. Boxed, it compiles once, in core.

**Raw finding, rebuilding core and everything downstream:**
- Wall time: master 80.8 s, branch 75.3 s (-6.8%). Unit-seconds: 393 and 338 (-14%).
- Core's lib: 38.9 s → 40.2 s. Core's lib test: 68.3 s → 61.4 s.
- `inference-api`: 29.4 s → 27.6 s. Its lib test: 30.1 s → 28.3 s.
- The SDK: 8.8 s → 5.0 s.
- The CLI test binary, the last unit in both: 29.5 s → 19.8 s.

**Implication:** moving an awaited body into its defining crate pays even when that crate is on the critical path,
because the api, the CLI and the SDK each compiled their own copy. The rule from Run 8 holds for every workspace
crate boundary.


## Run 10 — 2026-09-29 21:40

**Question:** what would moving GGUF out of core take, and what would it save?

**Command:** the Run 9 IR of `inference-core` at eb07616c (2,316,051 lines), grouped by `gguf::*` and
`pipeline::{gguf, ggml}`. The dependencies were mapped by reading each `gguf/` file's `use crate::` lines.

**Raw finding:**
- The GGUF-to-HF translation in `gguf/` (16.8k source lines) is 131,591 IR lines, 5.7% of core:
  `normal_config` 44,100, `normal_bindings` 12,031, `qwen_multimodal_bindings` 11,807, `gguf_tokenizer` 11,399,
  `multimodal_bindings` 7,715, `gemma3_config` 6,861, `multimodal_binding_utils` 5,691, `gemma3n_bindings` 5,262,
  `normal_registry` 4,850, `metadata` 4,802, `base_model` 3,855, the remaining bindings under 3k each.
- The pipelines that run GGUF and GGML models, `pipeline::gguf` 26,754 and `pipeline::ggml` 11,251, implement core's
  `Pipeline` and stay in core.
- What the translation needs from core:
  - `NormalLoaderType`, used throughout: the registry maps each canonical GGUF architecture to its compatible
    loaders, and `normal_bindings` branches on loader type for tensor names.
  - `MultimodalLoaderType` (the vision registry and the Qwen bindings).
  - The Gemma 3 and Gemma 3n vision config types, which live in `inference-models-gemma`.
  - `inference-nn` items (`Content`, `GGUFArchitecture`, the device-map loader trait, attention constants), and
    `RopePairing` in the other direction, which core's loaders read.
- The loader enums' variants are not feature-gated; only `loader()` dispatch is (`#[cfg(feature = ...)]` per row
  of `normal_loader_types!`). The enum, its names, parsing and HF-class detection are pure data.

**Implication:** the translation can move to a crate below core if the loader enums move with it or below it. Their
data half would come from one table macro exported from the lower crate, and core would generate its `loader()`
dispatch from the same table, so there is still one source of truth. The Gemma 3 and 3n pieces need the gemma family
crate, so the new crate would take it as an optional dependency behind `models-gemma`, or those two bindings would
move into the gemma crate. The expected saving is up to about 130k lines of core's serial codegen, compiled in
parallel with the family crates instead.

## Run 11 — 2026-09-29 22:15

**Question:** what do moving the loader enums to `inference_nn::loaders` and the GGUF translation to `inference-gguf`
change in IR and build time?

**Change:**
- `NormalLoaderType`/`MultimodalLoaderType` are expanded in inference-nn from the exported `normal_loader_table!`
  and `multimodal_loader_table!`. Core expands its feature-gated `loader()`/`get_processor()` dispatch from the same
  rows into `NormalLoaderTypeExt`/`MultimodalLoaderTypeExt`.
- `gguf/` became `crates/inference-gguf`, which core reaches as `crate::gguf`. The Gemma 3 and 3n bindings sit behind
  its `models-gemma` feature, which core forwards; tests build them regardless, through the dev-dependency.
- Tests gated on all five core family features became `#[cfg(test)]` over dev-dependencies. The crate keeps all 106
  of its tests (103 run, 3 ignored).

**Commands:**
- IR: `cargo llvm-lines` per crate.
- Build time: two trees in one scratch target, the branch and a `git worktree` of master, both warmed. Then each
  tree's `inference-nn/src/lib.rs` was touched and it was rebuilt with
  `cargo test --no-run --features cuda --workspace --lib --bins --tests --timings` and `CARGO_INCREMENTAL=0`, so
  everything from inference-nn down rebuilds. Load was about 15 for both.

**Raw finding:**
- IR: `inference-core` went from 2,316,051 to 2,163,788 (-152,263, -6.6%). `inference-gguf` is 206,760.
  `inference-nn` is 1,214,204.
- Rebuild wall time: master 113.1 s, branch 114.3 s. Unit-seconds: 664 and 698.
  - Core's lib: 55.9 s → 52.8 s (starts at 24.2 s instead of 20.4 s, after the slower inference-nn: 23.9 s against
    22.1 s).
  - `inference-gguf`: 15.9 s, from 18.0 s to 33.9 s, alongside the family crates.
  - Core's lib test, the last unit in both: from 46.1 s + 66.9 s on master to 43.0 s + 71.2 s on the branch.
  - The CLI test binary: 28.7 s → 28.8 s.

**Implication:**
- No measurable wall-time gain. Core's lib gets 3 s faster, but the tail (core's lib test and the api → CLI chain)
  is unchanged within the noise at this load. Unit-seconds rise with the new crate and its test binary.
- The value is structural: the GGUF format lives beside the model code rather than in the engine, and core's IR falls
  by 6.6%.
- For wall time, what's left is core's lib test and the api → CLI test chain.


## Run 12 — 2026-09-29 23:00

**Question:** can an internal parser over the model's own token bytes replace `openai-harmony` without changing what
the engine sees?

**Change:**
- `reasoning_parsers::harmony` gets a small incremental parser that mirrors the crate's `StreamableParser`:
  - It works from each token's decoded bytes. The Harmony strategy already receives them, special tokens included,
    because the chat template's format makes them visible.
  - A token whose bytes are exactly `<|start|>`, `<|message|>`, `<|end|>`, `<|call|>` or `<|return|>` counts as that
    special token.
  - Headers use the crate's own rules: channel extraction, `<|constrain|>` spacing, `to=` recipients and role
    detection.
- `HarmonyContext` no longer loads an encoding. That removes the o200k vocabulary download through
  `reqwest::blocking` and core's `prewarm_harmony_encoding` workaround, which ran inside `block_in_place`.

**Command:** a throwaway scratch crate, deleted afterwards (not committed). It holds the old `HarmonyContext` from
git (on `openai-harmony` 0.0.8) and the new one. Each text is encoded with Harmony's o200k tokenizer and fed token by
token, the old context by id and the new one by `decode_bytes([id])`. The comparison covers, after every token, the
current recipient, the reasoning, final and grammar-activation deltas, and at the end the reasoning, final content
and tool calls. Inputs: 9 hand-written conversations (channels, tool calls with the recipient before and after the
channel, `<|constrain|>`, multibyte text, stray text, malformed headers, no end marker) and 3,000 random
concatenations of Harmony fragments.

**Raw finding, three rounds:**
1. 2,707 same, 302 different. Two causes: the crate parses strictly by default, so a stop token inside a header is
   header text; and after a failed header, the pending assistant role stays until a header parses.
2. After matching the first: 2,957 same, 52 different.
3. After matching the second: 3,002 same, 7 different.

All 7 remaining differences are one bug in the old wrapper, not in the parser. Its per-channel lengths
(`last_analysis_len` and the like) carried across messages, so a later message on the same channel was sliced at the
earlier message's length and dropped or garbled. Examples:
- `...analysis<|message|>abcdef<|end|><|start|>assistant<|channel|>analysis<|message|>de` gave `abcdef`
  (new: `abcdefde`).
- A second final message `user` came out as `er`.

The new context resets the lengths when a message starts.

The review found a second, older wrapper bug: back-to-back calls to the same function merged into one, because the
"same call" check compared only recipients. Calls are now keyed by message too. Tests now cover this, the pending-role
quirk, a stop token inside a header, a recipient-only header, the GPT-OSS `assistant to=functions.x<|channel|>`
header form, and a Harmony stream through `ToolCallState`.

**Implication:** the replacement matches the crate on every stream except the ones the old wrapper got wrong.
`openai-harmony` leaves the dependency graph, and 49 packages leave the lockfile, among them `ravif`, `rav1e` and
`av1-grain` (the AVIF encoder chain behind issue #120). `reqwest` 0.12 remains only through `hf-hub` 0.4.

## Run 13 — 2026-09-29 23:30

**Question:** how much of core's lib test (the last unit in Run 11, 71.2 s starting at 43.0 s) is test code that
could move out?

**Commands:**
- `CARGO_TARGET_DIR=<scratch> cargo llvm-lines -p inference-core --lib --profile test --features cuda`, and the same
  without `--profile test`.
- Functions present only in the test build, grouped by their `...::tests` module.
- Source lines inside `#[cfg(test)]` modules and `tests.rs` files.

**Raw finding:**
- Source: 22,243 test lines in 59 files, 576 tests. The largest are `scheduler/paged_scheduler/tests.rs` 3,131,
  `pipeline/multimodal.rs` tests about 2.6k, `pipeline/inputs_processor.rs` 1.9k,
  `loaders/multimodal_loaders/tests.rs` 1,523, `sequence.rs` 1.1k, `prefix_cacher.rs` 1.1k, and
  `loaders/normal_loaders/tests.rs` 1,000.
- IR: the lib test is 2,428,564 lines and the lib 2,159,228. Functions only in the test build total 436,669 (18% of
  the lib test). The other 1,988,592 are core compiled again under `cfg(test)`.
- Largest test-only blocks: `scheduler::paged_scheduler::tests` 52,799, `selection::model_selected` 30,822 (tests
  deserialize `ModelSelected` through a second deserializer), `pipeline::cuda_graph::tests` 22,259,
  `loaders::multimodal_loaders::tests` 20,863, `sequence::tests` 14,675, `pipeline::isq::tests` 14,582,
  `prefix_cacher::tests` 13,665, `isq_flow::online::tests` 13,129, `loaders::normal_loaders::tests` 11,531.
- The big test modules are written against internals: `use super::*` inside private modules (`scheduler`,
  `sequence`, `prefix_cacher`, `pipeline` are all private in core). Moving them to integration tests would mean making
  those modules public.
- Scheduling: a test binary links, so cargo starts it only after every dependency's codegen, not just its metadata.
  That is why core's lib test starts at 43.0 s in Run 11, when the slowest family crate (`inference-models-qwen`,
  35.7 s) finishes, while core's lib starts at 24.2 s.

**Implication:**
- Moving every movable test would take at most about 18% (about 13 s) off core's lib test. The rest is core compiled
  again, which stays as long as core has any unit test.
- In Run 11 the CLI test binary ended 2.5 s before core's lib test, so the wall-time gain from shrinking core's lib
  test alone is capped near 2.5 s until the api to CLI chain also shrinks.
- The lib test's start is set by the slowest family crate's codegen, which is a separate lever.

## Run 14 — 2026-09-29 23:20

**Question:** does serving the committed OpenAPI document, instead of building it with utoipa, move the api to CLI
chain?

**Change:**
- The 43 `#[utoipa::path]` annotations became `#[cfg_attr(test, utoipa::path(...))]`, and the `#[derive(OpenApi)]`
  generator moved into a `#[cfg(test)] mod generated` in `openapi_doc.rs`. The staleness and regenerate tests still
  use it.
- At runtime, `get_openapi_doc(base_path) -> serde_json::Value` parses `include_str!` of `docs/openapi.json` and
  prefixes its path keys, and Swagger UI serves it with `external_url_unchecked`. serde_json's `preserve_order` is now
  declared in server-core, so the committed key order is kept explicitly rather than through minijinja.
- utoipa can't deserialize our document (an untagged `RefOr`). So embedders now serve it beside their own document
  instead of merging it into theirs; the `lib.rs` example shows how.

**Commands:**
- IR: `cargo llvm-lines -p inference-server-core --lib --features cuda`.
- Build time: two trees in one scratch target (the branch and a `git worktree` of master), both warmed. Each tree's
  `inference-api/src/lib.rs` was touched and it was rebuilt with the cold-build test command, `--timings` and
  `CARGO_INCREMENTAL=0`. Load was about 14.

**Raw finding:**
- IR: `inference-server-core` went from 676,823 to 544,498 (-132,325, -19.6%). utoipa code in it fell from 132,417
  to 9,639 (its own types' `ToSchema` derives).
- Rebuild from inference-api down: master 30.9 s, branch 30.6 s. Unit-seconds: 99 and 97.
  - server-core's lib: 11.9 s → 10.7 s.
  - The CLI test binary, last in both: from 15.6 s + 15.2 s on master to 14.6 s + 15.9 s on the branch.

**Implication:**
- IR falls, and the server no longer builds its document at startup, but the wall-time gain is within noise.
- After Runs 5, 7, 8 and 9, the crates below the CLI test binary are small enough that trimming them barely moves the
  chain.
- The cold build's end is now set by core: its lib (about 53 s), and its lib test (about 71 s), which starts only when
  the slowest family crate finishes codegen (Run 13).


## Run 15 — 2026-09-29 23:35

**Question:** what does core's second compile (its lib test) cost on its own, and how much of that is LLVM
optimization at the dev profile's opt-level 3?

**Commands:** in a scratch target with dependencies built and `CARGO_INCREMENTAL=0`, touch
`inference-core/src/lib.rs` and time:
- `cargo test --no-run --features cuda -p inference-core --lib --timings`, with
  `--config profile.dev.package.inference-core.opt-level=3`, then with `=1`.
- `cargo build --features cuda -p inference-core --lib`, the same two ways.

Load was 9 falling to 7, so the machine was mostly idle.

**Raw finding:**

| Unit | opt-level 3 | opt-level 1 |
|---|---|---|
| Core lib test | 28.7 s (unit 28.2 s) | 27.6 s (unit 27.1 s) |
| Core lib | 25.8 s | 23.0 s |

**Implication:**
- In the cold workspace build, core's lib test takes 71 s (Run 11) against 28 s alone. The difference is CPU
  contention: the build is saturated from start to end, so the lib test gets a share of the cores, not all of them.
- Optimization is a small part of either compile (1 to 3 s). The time is front-end work: type checking, borrow
  checking and monomorphization. A lower opt-level for core buys almost nothing.
- Removing the second compile would save about 28 CPU-seconds per cold build, out of about 2,100 unit-seconds. Moving
  all 576 unit tests out of core would take that, plus widening core's private modules to make the tests reachable.
  The cost is out of proportion to the gain.
- In a saturated build, wall time follows total CPU work more than any one chain.

## Run 16 — 2026-09-29 23:50

**Question:** where does core's single-threaded compile time go, pass by pass and item by item?

**Commands:** in a scratch target with dependencies built, `CARGO_INCREMENTAL=0`, core touched each time:
- `RUSTC_BOOTSTRAP=1 cargo rustc --features cuda -p inference-core --lib -- -Z time-passes` for the phases.
- `... -- -Z self-profile=<dir> -Z self-profile-events=default,args`, read with `summarize`.
- `crox --minimum-duration 100` to a Chrome trace. Each `evaluate_obligation` event was attributed to the enclosing
  `typeck_root`, `mir_borrowck` or `optimized_mir` event on the same thread, and its predicate recorded.

**Raw finding, before the change (at eb07616c plus #146 to #148):**
- Phases (32.8 s total): type checking 6.6 s, borrow checking 6.1 s, metadata generation 5.6 s, coherence 1.8 s,
  lowering to LLVM IR 7.0 s, LLVM passes 11.9 s (parallel), waiting on LLVM 5.0 s.
- Self time: `evaluate_obligation` 5.43 s over 158,587 queries, more than `typeck_root` itself (1.64 s self, 2.25 s
  total). `mir_borrowck` is 4.27 s total but 1.10 s self.
- Attributed trait-solving time: about 0.19 s each under the `mir_borrowck` and `optimized_mir` of every `#[async_trait]`
  sampling wrapper on `Pipeline` (`sample_causal_gen`, `try_sample_causal_gen_batched`,
  `try_sample_speculative_causal_gen`, `sample_block_gen`). There are 20 of them across 8 pipelines, even though their
  bodies only await a shared function that already returns a `BoxFuture`.
- The predicates are the auto traits of `Sequence`'s type graph: `Sequence: Send`, `Sampler: Send + Sync`,
  `Tokenizer: Send + Sync`, the FlashInfer workspace maps, and so on.

**Cause:** `async_trait` wraps each body in an async block boxed as `dyn Future + Send`, so the compiler proves the
block's captured state `Send`, including `&mut [&mut Sequence]`. The trait's lifetime bounds sit in each method's
environment, so every canonical query is distinct and nothing is cached across wrappers. Each wrapper re-proves the
whole graph.

**Change:**
- `Pipeline` drops `#[async_trait]`. Its four async methods become plain `fn`s returning `BoxFuture<'a, _>`.
- Forwarding impls return the shared function's future (`sample_and_add_toks`, `finalize_block_gen`) directly.
- `try_sample_causal_gen_batched` maps that future to `true`.
- Defaults and unsupported pipelines return `std::future::ready`.
- Five async blocks remain: the speculative method in the normal and multimodal pipelines, because the future borrows
  a local cache view, and AnyMoE's three methods, which hold the target's lock across the await.

**Raw finding, after:**
- Self time: `evaluate_obligation` 5.43 s → 1.49 s. Phases: borrow checking 6.1 s → 4.4 s, metadata 5.6 s → 4.0 s,
  total 32.8 s → 29.6 s.
- Rebuilds of core's lib test (`cargo test --no-run --features cuda -p inference-core --lib`, two each): master 29.5
  and 29.8 s (load 7.6 and 6.7), branch 26.3 and 26.0 s (load 10.0 and 8.6), about -12%.

**Implication:**
- About 3.5 s of single-threaded work comes off each of core's two compiles, so every cold build and every edit to
  core gains.
- What's left in the front end is spread thin: `typeck_root` 1.6 s self, `mir_borrowck` 1.1 s self, and the MIR
  passes. The larger serial items now scale with the amount of code: LLVM IR generation (`codegen_module` 5.4 s) and
  metadata (4.8 s).

## Run 17 — 2026-09-30 01:00

**Question:** Model selection and hardware fit (`selection/`, `tuning`, `diagnostics`, `resource_plan`) answer "which
model and quant fits this machine", which is an API question. Can they leave core, and what does core save?

**Coupling found:**
- `tuning`, `diagnostics` and `resource_plan` had no users inside core; they were only re-exported.
- `selection` used only core's public loader builders, plus `UqffWriteSpec` (schema only), `build_api_with_cache`
  and `get_device_layers_for_loader`, which were `pub(crate)`.
- The one real tie: `ModelLoaderConfig` held a `ModelSelected`, and core's unload/reload path rebuilt the loader from
  it through `LoaderBuilder`.
- The SDK depends on core but not on inference-api, so selection can't live in inference-api.

**Change:**
- New crate `inference-selection` above core holds `ModelSelected`, `LoaderBuilder`, model metadata, quant/GGUF/UQFF
  discovery, auto-tuning, paged KV planning and the doctor.
- Core's seam is `trait LoaderSource { fn build_loader(&self, &ModelLoaderConfig, no_kv_cache) }`.
  `ModelLoaderConfig.source: Arc<dyn LoaderSource>` replaces `model_selected`, and `ModelSelected` implements it.
  AnyMoE wrapping stays in core.
- `EmbeddingLoaderType::config_arch` moved into core beside its inverse, `from_causal_lm_name` (orphan rule).
- The doctor reports compiled-in backends, so the crate forwards cuda/metal/cutile/... to core. Consumers forward to
  it; inference-api gained a `cutile` feature so FFI's reaches it.
- Core dropped `sysinfo`, `walkdir`, `num-traits` and `candle-metal-kernels`.

**Raw finding:**
- `cargo llvm-lines --lib -p inference-core`: 2,163,788 → 1,840,443 (-323k). The survey estimated ~75k for this
  group, so most of the drop is generic instantiations (serde, hf-hub, sysinfo) the moved code pulled into core.
- `cargo llvm-lines --lib -p inference-selection`: 245,499. Net over both crates: about -78k.
- `-Z time-passes` on core's lib (touched lib.rs): total 29.6 s (Run 16) → 24.6 s. borrow checking 4.4 → 4.2 s,
  metadata 4.0 → 3.5 s, `codegen_crate` 8.0 s, LLVM passes 8.9 s.
- First CI run: every test passed, but a server-core doctest still imported `inference_core::ModelSelected`.
- Review: FFI's `cutile` did not reach the new crate (inference-api had no `cutile`), so an FFI cutile build would
  report `cutile: false` and compile out the doctor's cuTile check. The Makefile's supported-models regen targets
  still named `-p inference-core` and matched nothing. Both fixed.
- Second CI run: 287 s, all passing (2208 CPU, 2528 CUDA).
- cuTile check with CUDA 13.4 in a scratch target (`cargo clippy -p inference-selection -p inference-ffi -p
  inference-api --tests --features inference-selection/cutile,inference-ffi/cutile -- -D warnings`): clean, 18.5 min,
  2.3 GB, deleted afterwards.

**Implication:** Code that uses core only through its public surface is cheap to lift out, and each move takes its
monomorphized dependencies with it, which is where most of the IR was. The chat templates (minijinja) are the next
candidate of this kind, moving down to inference-protocol.

## Run 18 — 2026-09-30 09:30

**Question:** Run 16 found `#[async_trait]` wrappers re-proving `Send` over `Sequence`'s type graph. Does the same
pattern (a boxed `dyn Future + Send` whose proof can't be cached) survive anywhere, in core or the crates above it?

**Commands:** a CPU-only scratch target (`CARGO_INCREMENTAL=0`), each crate touched and rebuilt with
`RUSTC_BOOTSTRAP=1 cargo rustc -p <crate> --lib -- -Z self-profile=<dir> -Z self-profile-events=default,args`, read
with `summarize` and with `crox --minimum-duration 20`. A script charges each top-level trait-solving event
(`evaluate_obligation`, `type_op_prove_predicate`, `codegen_select_candidate`, normalization) to the innermost
enclosing `typeck_root`, `mir_borrowck`, `optimized_mir` or mono-item collection event and groups it by trait and
self type.

**Raw finding, per crate (self time; CPU-only, so not comparable with Run 16's CUDA build):**
- inference-core: `evaluate_obligation` 2.49 s, `type_op_prove_predicate` 0.54 s, `codegen_select_candidate` 0.53 s;
  3.20 s of top-level solving in all.
- inference-api 0.39 s, inference-server-core 0.22 s (+0.84 s `codegen_select_candidate`), inference-agent 0.11 s.
- LLVM object emission dominates all four (core 33 s, api 18 s, agent 11 s, server-core 9 s summed over 16 CGUs).
- Core by item: `speculative::driver::try_sample_speculative_causal_gen` 0.64 s (1054 queries, all
  `<C as SpeculativeCacheAccess>::Guard: Send/Sync`), `sample_and_add_toks` 0.38 s, `finalize_block_gen` 0.32 s,
  `submit_step` 0.18 s, the normal and multimodal speculative wrappers 0.19 s. All are `Send`/`Sync` proofs of an
  async fn's coroutine witness and `Sequence`'s graph.
- Every one of those queries' environments includes `OutlivesPredicate('^c_0, '^c_1)`: the explicit `'b: 'a` that
  #149 kept on the `BoxFuture` signatures (`fn f<'a, 'b: 'a>(seqs: &'a mut [&'b mut Sequence]) -> BoxFuture<'a, _>`).
- Server-core's cost sits under mono-item collection: projections through axum's `Handler`, tower's `MapResponse` and
  the handler fn types. That is axum monomorphizing each route, not a `Send` proof.

**Cause:** canonicalizing a query turns the free regions of an outlives where-clause into region variables, and
rustc's selection context won't use the crate-wide evaluation cache for an environment that holds inference
variables. So the region bound sent every nested proof to a per-query cache and each wrapper re-proved the whole graph.
Region-free trait bounds don't do this: the driver keeps `C: SpeculativeCacheAccess + Sync` and still got fast. This
finishes Run 16's diagnosis, which blamed the methods' lifetime bounds; moving off `async_trait` kept them. The bound is
redundant: `&'a mut [&'b mut Sequence]` implies `'b: 'a`, and implied bounds are not where-clauses.

**Change:** `<'a, 'b: 'a>` becomes `<'a>`, with `&'b mut Sequence` elided to `&mut Sequence` (clippy flags `'b` once it
has no bound), on 25 signatures: the `Pipeline` sampling methods, their shared functions,
`submit_step`, the speculative driver and `report_pipeline_forward_error`. Nothing else changes; the implied bound
still types the futures.

**Raw finding, after:**
- Core self-profile: top-level solving 3.20 s → 0.76 s. `evaluate_obligation` 2.49 → 0.54 s, `typeck_root` 3.31 →
  1.45 s self, `mir_borrowck` 2.28 → 0.90 s self. The speculative driver fell 0.64 → 0.013 s without touching its
  generic, so making it concrete over `PagedSpeculativeCacheAccess` isn't needed.
- `-Z time-passes`, two runs each: type checking 2.28/2.33 → 2.04/2.06 s, borrow checking 1.86/1.89 → 1.76/1.77 s,
  total 19.24/19.62 → 18.84/18.97 s, about -0.5 s. The self-profile overstated the gain roughly fivefold: its
  per-event recording inflates query-heavy passes most.

**Implication:** Explicit outlives bounds on functions that prove auto traits over large graphs cost real time, and
the self-profile makes that cost look bigger than it is; confirm with `-Z time-passes`. What remains of trait solving
in these crates is small. The next lever is codegen volume: LLVM emission dominates every crate, and server-core's
remaining solver time is axum's per-route monomorphization, which only erasing handlers (boxed services) would cut.

## Run 19 — 2026-09-30 11:00

**Question:** before moving the chat templates to inference-protocol, how much of core's IR is templating, and what
else is large?

**Command:** `cargo llvm-lines --lib -p inference-core`, with function names grouped by substring, then the
`Deserialize`/`Visitor` instantiations grouped by the type deserialized and its crate.

**Raw finding:**
- Of core's 1,840,443 lines, names containing `chat_template` total 47k and `minijinja` 30k (overlapping): about 2.5%.
  The template move would pay less than the selection move did.
- Names mentioning `serde_json` total 482k, over-counted by any signature that names it. Grouping the deserialize
  instantiations properly: 284k lines, of which 101k deserialize tokenizers' own types (`NormalizerWrapper`,
  `PreTokenizerWrapper`, `DecoderWrapper`, BPE, `Metaspace`, `Split`, ...), through both `serde_json` and serde's
  untagged `ContentRefDeserializer`.
- Source: `pipeline/tokenizer.rs` calls `Tokenizer::from_bytes`. In tokenizers 0.23 `from_bytes<P: AsRef<[u8]>>` and
  `from_file<P: AsRef<Path>>` are generic, so the whole tokenizer deserializer is compiled in the calling crate. Its
  `FromStr` impl is not generic and compiles once in tokenizers. inference-models-diffusion's FLUX stepper calls
  `from_file` twice; the other calls were in tests (core's `pipeline/tokenizer.rs` and llg, inference-gguf). The
  duplication happens because the dev profile's opt-level 3 turns share-generics off.

**Change:** `inference_nn::utils::tokenizer::{tokenizer_from_bytes, tokenizer_from_file}`, non-generic helpers over
`Tokenizer::from_str`, replace every `from_bytes`/`from_file` call in the workspace.

**Raw finding, after:**
- `cargo llvm-lines --lib`: inference-core 1,840,443 → 1,671,740 (-169k, -9%); inference-models-diffusion 396,710 →
  204,967 (-48%); inference-nn 793,881 → 794,882 (+1k for the helpers).
- Core's remaining deserialize IR: 147k, mostly its own request and template types and the protocol/MCP types inside
  them, which must instantiate wherever `NormalRequest` is deserialized. inference-nn's configs (`PreProcessorConfig`
  12k, `XLoraConfig`, `SamplingParams`) are about 20k more that non-generic `from_json`s in inference-nn would remove.
- `-Z time-passes` on core's lib (CPU, scratch target, `CARGO_INCREMENTAL=0`): total 19.87 → 18.93 s, `codegen_crate`
  5.86 → 5.26 s, LLVM passes 11.24 → 10.51 s. One steady run each; the first run of each pair was inflated
  (23.3 and 25.3 s) for both versions alike.
- A first attempt timed in the main target was meaningless: it is incremental, so the repeat reused everything.

**Implication:** A generic constructor in a dependency is compiled into every crate that calls it, however little
the caller adds. Grouping llvm-lines by the instantiated type's crate finds these; tokenizers' `from_file`/`from_bytes`
was the largest. Candidates the review found: tokenizers' generic `encode`/`encode_batch` (called in most model
crates), PNG encoding generic over the writer in core, and inference-nn's config deserializers. The chat-template
move is still worth doing for layering, but it is a small build-time item.

## Run 20 — 2026-09-30 13:00

**Question:** Run 19's review listed more generic-instantiation candidates. How large are they, and which are worth
moving for separation of concerns rather than build time?

**Command:** `cargo llvm-lines --lib -p <crate>`, grouped by the defining crate of each function and, for tokio, by
runtime module and spawned future.

**Raw finding:**
- tokenizers-defined IR per crate: qwen 4,097, llama 4,097, gemma 4,071, gguf 4,289, agent 4,258, phi 4,192, core
  19,329. The review's estimate for non-generic `encode`/`encode_batch` helpers (10 to 30k per crate) was far off:
  tokenizers' wrapper types keep the work inside tokenizers. Not worth doing.
- Core by defining crate (1.67M): its own code 29.2%, `core`/`alloc`/`std` 32.7%, serde_json 8.3%, tokio 5.5% (92k),
  hashbrown 4.6%, inference_nn 2.5%, png 0.6% (10.6k), minijinja 0.5%.
- tokio's runtime machinery per future is about 70k: the four distributed daemon replicators about 3.8k each,
  `tokio::fs` operations each spawning their own blocking task (about 15k), the rest spread thin.

**Separation-of-concerns reading of the candidates:** `encode` helpers and tokio spawn erasure are call-pattern
changes with no responsibility to move. inference-nn's configs already live in the right crate; only their parsing
happens in core. Chat templates are a protocol concern (messages to prompt text). Image encoding was the clearest
boundary problem: for `response_format: url` the pipeline wrote `image-generation-<uuid>.png` into the process's
working directory and returned that path, so the engine chose format, location and name.

**Change (image encoding):** the engine returns pixels, `Response::ImageGeneration(GeneratedImages { created,
images })`, as `Response::Speech` returns PCM. `RequestMessage::ImageGeneration` loses `format` and `save_file`.
`inference_protocol::images::{encode_png, image_generation_response}` encodes them for inference-api, the Rust SDK
(same public signature) and the CLI, and agent tool images use `encode_png`. Where url images are stored is
unchanged, now decided above the engine.

**Raw finding, after:** inference-core 1,671,740 → 1,661,023 IR lines; inference-protocol 149,483 → 173,978, since
the PNG encoder now compiles there, off the critical path beside the kernel builds. Core drops its `uuid` dependency.

**Implication:** the remaining build-time items are each 1 to 4% of core. The next architecture items are the
chat-template move and choosing where url images should be stored (the files store, a configured directory).
