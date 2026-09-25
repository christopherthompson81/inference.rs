# Test suite speed investigation

Goal: run the whole test suite, CPU and CUDA, without long single-core stretches, rebuild churn or idle cores.

Machine: i7-10700K (8C/16T), RTX 3090 (24 GB, sm_86), rustc 1.98.1, CUDA 12.8.

## Run 1 - 2026-09-25 11:00

Question: why do test runs have long single-core stretches?

Method: `touch` one inference-core file, then `cargo test -p inference-core --lib --no-run --timings`, with 1 s CPU
sampling.

Finding:

- inference-core's test build is one 44 s unit. Its first ~10 s is the single-threaded rustc front end, which is
  inherent to one very large crate on stable; codegen then fans out to ~10 cores.
- The real waste was rebuilds. Flipping `CC` between invocations (`CC="ccache gcc"` on the CUDA commands, unset
  elsewhere) reran the ring/aws-lc-sys build scripts and rebuilt everything above them, 59-77 s per direction.
- Each distinct `-p X --features Y` / `--workspace` combination also got its own artifacts.

Change:

- CC/CXX/NVCC and the `INFERENCE_TEST_*` model paths live in `~/.cargo/config.toml` `[env]`.
- `scripts/local_ci.sh` has fixed `--lint`/`--tests`/`--cuda`/`--docs` modes.
- The test modes build `--lib --bins --tests`, since linking ~60 example binaries per run cost minutes and tens of GB.

## Run 2 - 2026-09-25 12:30

Question: can the 63 `#[ignore = "requires CUDA"]` GPU tests run by default under `--features cuda`?

Change: they call `skip_without_cuda!()` instead, which skips without a device.

Finding: running them all in one libtest process segfaulted. The CUDA tests share process-global state (memory
pools, graph scopes), and the memory-pool tests had never run concurrently before. A process-wide lock in the
macro stopped the crash. The memory-accounting test still failed, though, because unguarded CUDA tests in the same
binary allocated concurrently. `--test-threads=1` passed but serialized the entire workspace.

The same runs surfaced tests the old CI never ran:

- a pyo3 fixture missing `config.json`;
- a stale `openapi.json`;
- two prefill-workspace tests whose expectations assume the SM90 FA3 FP8 paged build;
- four FP8 tests (cuBLASLt returns `NOT_SUPPORTED`, and the blockwise FP8 MMA GEMV returns garbage) that need sm_89+.
  Production gates the latter on `MIN_COMPUTE_CAPABILITY = 89`; only the tests lacked the gate.

## Run 3 - 2026-09-25 13:30

Question: can the CUDA tests be isolated without serializing everything?

Change: cargo-nextest runs each test in its own process, so each GPU test gets its own context and pools.

Finding:

- The CUDA suite (2450 tests) ran in 86 s fully parallel. Two things failed.
- The PaddleOCR paged tests passed their assertions and then segfaulted at process exit.
  - `Drop for InferenceRs` sent `Terminate` to the engine threads but never waited for them. A thread still inside
    CUDA raced the context teardown.
  - libtest hid this because a binary's tests share one process, which exits once.
  - Fix: drop now joins the engine threads, with a 10 s bound so a wedged engine can't hang it.
- `rlimit_nproc_caps_processes` failed, because `RLIMIT_NPROC` counts every thread of the concurrently running test
  processes. It now runs alone (`threads-required = "num-test-threads"`).

## Run 4 - 2026-09-25 14:00

Question: what bounds the wall time once nothing crashes?

Finding:

- The slowest tests were the PaddleOCR CPU f32 parity tests, at 50-90 s each. They were slowed by running
  concurrently, each with a rayon pool over every core. Total work was ~1500 test-seconds, i.e. ~94 s ideal on 16
  threads, and one of those tests ran four fixtures back to back.
- Changes:
  - The four-fixture test becomes one test per fixture.
  - Under `--features cuda`, the PaddleOCR parity tests run on the GPU in bf16 (the goldens match f32 exactly).
  - The model tests go in a nextest group. Its size comes from VRAM: each peaks at 1.9-2.4 GB (nvidia-smi at 100 ms),
    so 6 fit the 24 GB card with ~6 GB left for kernel tests and the desktop.
- Result: `--cuda` runs 2453 tests in 62 s, green twice. Total work is 940 test-seconds (58 s ideal), so the machine
  is saturated. The slowest single test is 15 s, and there is no single-core tail left.
- `--tests` (CPU) runs 2133 tests in 64 s, plus 56 doctests.
