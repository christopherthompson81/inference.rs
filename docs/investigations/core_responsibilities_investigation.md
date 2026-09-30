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
