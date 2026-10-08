---
title: Deep checks and parity references
description: Real-checkpoint tests, the paths they read, and how to rebuild the llama.cpp references and test GGUFs.
---

`scripts/local_ci.sh` runs on tiny checkpoints built at test time. Tests on real weights, and parity checks against
other implementations, run on request.

## Deep checks

`scripts/deep_checks.sh` runs the `deep` nextest profile (`.config/nextest.toml`). Each test skips unless its path is
set in `~/.cargo/config.toml` under `[env]` (build scripts track those variables, so keep them out of the command line):

| Variable | Points at | Read by |
|---|---|---|
| `INFERENCE_TEST_PADDLEOCR_VL_MODEL` | PaddleOCR-VL Hugging Face directory | `paddleocr_vl` |
| `INFERENCE_TEST_QWEN3_5_MODEL` | Qwen3.5-0.8B Hugging Face directory | `qwen3_5_mtp` |
| `INFERENCE_TEST_QWEN3_5_GGUF` | a Q8_0 GGUF of the same model | `qwen3_5_mtp` |
| `INFERENCE_TEST_IQ4_XS_GGUF` | a Qwen3.8-27B IQ4_XS GGUF file | `gguf_iq::iq4_xs_greedy_continuation_matches_llama_cpp` |
| `INFERENCE_TEST_LAYOUT_MODEL` | PP-DocLayoutV3 safetensors directory | `inference-ffi` `detections` tests |
| `INFERENCE_TEST_LAYOUT_IMAGE` | a document page image | `inference-ffi` `detections` tests |
| `INFERENCE_TEST_FLUX_DIR` | a single-file FLUX layout | `flux` (CUDA only) |

`iq4_xs_greedy_continuation_matches_llama_cpp` compares against a continuation recorded with mainline `llama-simple
-n 32` on the same file, so it needs no reference build.

## GGUF perplexity parity

`scripts/gguf_perplexity_parity.sh <llama-perplexity> <gguf-dir> [text] [tolerance]` scores every `*.gguf` in a
directory with the reference's `llama-perplexity` and with inference.rs, and fails past a relative drift (2% by default).

The references and the scored GGUFs are rebuilt from pinned inputs:

1. `scripts/build_gguf_references.sh [dir]` builds llama.cpp and ik_llama.cpp at the commits the recorded results used,
   with CUDA when `nvcc` is found (for the GPU it finds, so build where the GPU is visible), and a Python venv with mainline's pinned conversion requirements. `dir` defaults to
   `$INFERENCE_REFERENCE_DIR`, else `~/.local/share/inference-rs/references`; the binaries land in
   `<dir>/llama.cpp/build/bin` and `<dir>/ik_llama.cpp/build/bin`, the venv in `<dir>/venv`.
2. `scripts/make_gguf_test_dirs.sh <qwen3.5-0.8b hf dir> <out> [dir]` converts the checkpoint to F16 (without its MTP
   layer), computes an imatrix with each reference (ik cannot read mainline's format), and writes:

   | Directory | Types | Reference |
   |---|---|---|
   | `<out>/iq` | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | mainline |
   | `<out>/kt` | IQ1_KT to IQ4_KT, `--pure --token-embedding-type q8_0` | ik_llama.cpp |
   | `<out>/iqk` | IQ2_K to IQ6_K and IQ2_KT, in ik's default mixes | ik_llama.cpp |
3. Score each directory against its reference, from the repository root:

   ```bash
   refs=${INFERENCE_REFERENCE_DIR:-~/.local/share/inference-rs/references}
   scripts/gguf_perplexity_parity.sh $refs/llama.cpp/build/bin/llama-perplexity <out>/iq README.md 0.05
   scripts/gguf_perplexity_parity.sh $refs/ik_llama.cpp/build/bin/llama-perplexity <out>/kt
   scripts/gguf_perplexity_parity.sh $refs/ik_llama.cpp/build/bin/llama-perplexity <out>/iqk
   ```

The iq directory gets 5%: IQ1_S and IQ1_M score near perplexity 300 to 800 here, where mainline's own CPU and CUDA
paths already differ by up to 3.2%. Results are recorded in `docs/investigations/gguf_reference_builds_investigation.md`.
