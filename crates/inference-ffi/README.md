# inference-ffi

C ABI for inference.rs, built as `libinference_ffi` (`.so` / `.dylib` / `.dll`). The header is
[`include/inference.h`](include/inference.h), and the contract at its top is normative: opaque handles, status codes plus
a thread-local `inference_last_error()`, inputs copied, outputs borrowed from their handle (or returned as an owned
`inference_string`), no callbacks, and no panic crossing the boundary.

Current modules:

- **Engine** (`inference_engine_*`, `inference_chat*`, `inference_stream_*`): load an engine from a JSON spec and run
  OpenAI-style chat completions on it, blocking or streamed by polling. Requests and responses are the JSON the HTTP
  server takes and returns; errors carry the OpenAI error body in `inference_last_error()`. It is a shim over
  `inference-api`, the same engine surface the HTTP server is built on.
- **Layout** (`inference_layout_*`): PP-DocLayoutV3 document layout detection. It returns class, label, score and
  bounding box per region, in predicted reading order, and takes RGB/BGR/RGBA/BGRA/gray 8-bit images with arbitrary row
  stride.

## Build

```bash
cargo build --release -p inference-ffi                   # CPU
cargo build --release -p inference-ffi --features cuda   # adds the "cuda" backend
```

The library is written to `target/release/`. rustc exports only the crate's `#[no_mangle]` functions, and
`tests/export_surface.py` checks that they match the header exactly.

## Versioning

`inference_abi_version()` returns `(major << 16) | (minor << 8) | patch`, matching the `INFERENCE_ABI_VERSION_*` macros
in the header. The ABI is not stable yet: while it is 0.0.x, every change bumps the patch number and may break
callers, so bindings should require an exact match. Compatibility rules (minor versions add entry points, patch
versions fix behaviour) start at 0.1.0.

## Tests

```bash
inference-ffi/tests/run.sh [--features cuda --backend cuda] [model image]
```

The script:

1. Builds the library.
2. Checks that its exports match the header (`tests/export_surface.py`).
3. Runs the Rust ABI tests, modules of one binary under `tests/integration/`: `layout_abi.rs` (error paths, pixel
   formats, batching, handle lifetimes), `engine_abi.rs` (chat and streaming on a tiny random-weight model built at test
   time) and `header.rs` (the header declares exactly the exports and compiles as C99; these also run in
   `scripts/local_ci.sh --tests`).
4. Compiles the C99 consumer (`tests/c/layout_test.c`) with `-Wall -Wextra -Werror -pedantic`.
5. Diffs the consumer's detections against the Rust `pp_doclayout_v3_detect` example on the same page.

Without a model directory and image, the model-backed steps are skipped. The Rust tests also read
`INFERENCE_TEST_LAYOUT_MODEL` / `INFERENCE_TEST_LAYOUT_IMAGE`. Converting the page to PPM for the C consumer needs
ImageMagick's `convert`.
