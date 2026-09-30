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
