# SDK on inference-api: dependency cost investigation

Question: should the Rust SDK (`crates/inference`) depend on `inference-api`, so it can call the engine API's chat
collection and generation code instead of keeping its own copies? The plan (step 5, phase 4) made this conditional
on what the dependency costs an SDK-only build.

## Run 1 - 2026-09-28 18:00

Commands, on master `cddc32de`, dev profile, 16 threads:

```bash
cargo tree -p inference -e normal --prefix none | sort -u    # before and after adding `inference-api.workspace = true`
cargo build -p inference --timings                            # before
cargo build -p inference --timings                            # after adding the dependency (same target dir)
```

Dependency tree: 600 unique entries before, 609 after. New crates: `inference-api`, `utoipa`, `zip` (with `zlib-rs`,
`zopfli`, `bumpalo`), `chrono`, `data-url`. `inference-core` is also rebuilt, since `inference-api` turns on its
`utoipa` feature.

Timings (from each report's unit data):

| unit | before | after |
|---|---|---|
| inference-core | 50.3 s | 56.0 s |
| inference-api | - | 31.0 s |
| inference (SDK) | 10.0 s (both) | 10.0 s |
| wall, first run | 1 m 48 s | 1 m 34 s (rebuild of core and above only) |

`inference-api` depends only on `inference-core`, and the SDK would depend on it, so it lands on the critical path
between core and the SDK. An SDK-only build pays about 31 s for the api crate plus about 6 s of `utoipa` derive
expansion in core, roughly +37 s over a 1 m 48 s build. In workspace builds the api crate is built anyway, so they pay
nothing.

Decision: don't add the dependency. The one real gap it would have closed is that the SDK's `send_chat_request`
dropped the agentic tool-call records and files the API attaches. That code (progress folding, file-id stamping, the
per-response file cap) only needs core types, so it moves to `inference_core::ChatResponseCollector`, and both
`collect_chat` and the SDK's `final_response` use it. The SDK's other overlaps with the api (generation, operations)
are thin wrappers over `InferenceRs` or have different output types (raw PCM, `save_file`), so there is no more
duplication to remove through this dependency.
