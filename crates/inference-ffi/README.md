# inference-ffi

C ABI for inference.rs, built as `libinference_ffi` (`.so` / `.dylib` / `.dll`). The header is
[`include/inference.h`](include/inference.h), and the contract at its top is normative: opaque handles, status codes plus
a thread-local `inference_last_error()`, inputs copied, outputs borrowed from their handle, no callbacks, and no panic
crossing the boundary.

Current modules:

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
in the header. A different major version is incompatible. Minor versions only add entry points, and patch versions
change behaviour only.

## Tests

```bash
inference-ffi/tests/run.sh [--features cuda --backend cuda] [model_dir image]
```

The script:

1. Builds the library.
2. Checks that its exports match the header (`tests/export_surface.py`).
3. Runs the Rust ABI tests (`tests/layout_abi.rs`: error paths, pixel formats, batching, handle lifetimes).
4. Compiles the C99 consumer (`tests/c/layout_test.c`) with `-Wall -Wextra -Werror -pedantic`.
5. Diffs the consumer's detections against the Rust `pp_doclayout_v3_detect` example on the same page.

Without a model directory and image, the model-backed steps are skipped. The Rust tests also read
`INFERENCE_TEST_LAYOUT_MODEL` / `INFERENCE_TEST_LAYOUT_IMAGE`. Converting the page to PPM for the C consumer needs
ImageMagick's `convert`.
