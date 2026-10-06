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

## Run 5 - 2026-09-26

- Question: the CPU suite still has a long tail at 1-2 running tests. Is it scheduling or test length?
- Timeline (junit timestamps), before: the `gpu-model` group (max 6, sized for VRAM) also throttled CPU runs, so two
  PaddleOCR f32 tests waited for a slot and `two_images_in_one_message_match_transformers` ran alone for the last ~15 s.
- Change: moved the group override to a `cuda` nextest profile (`local_ci.sh --cuda` passes `--profile cuda`), and
  gave the PaddleOCR, layout ABI and tiktoken tests `priority = 100` in the default profile.
- Result: `cargo nextest run --workspace --lib --bins --tests` = 59.8 s (was 60 s). All 9 long tests now start at
  t=0, but the run is bounded by the longest one: two_images 59 s, page_01 50 s, page_00 49 s, ffi detections 33 s.
- Alone, the same tests take 6.2 s (text_only), 17.9 s (page_00) and 28.6 s (two_images). With 8 f32 model tests
  plus the rest sharing 16 cores, each runs ~2-3x slower, so the pole is CPU oversubscription, not order.
- Implication: scheduling can't beat ~max(single-test time under load). The CPU f32 decodes check the same goldens
  as the bf16 CUDA run, so the real lever is whether CPU `--tests` should run model-backed tests at all.

## Run 6 - 2026-09-26

- Change: nextest `default-filter` on the default profile leaves out the real-checkpoint tests (paddleocr_vl binary,
  ffi `detections_through_the_abi`). A `models` profile runs only those, and the `cuda` profile runs everything with the
  VRAM group. New `local_ci.sh --models` runs the model tier on CPU.
- Result (warm): `--lint --tests` = 23 s end to end (2122 tests plus doctests and smoke; was ~80 s). `--models` = 57 s
  for 8 tests (the CPU f32 pole, now opt-in). `--cuda` = 58 s, 2449 tests, green.

## Local CI phase overlap — 2026-09-30 00:30

**Question:** where does `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep` leave the cores
idle?

**Command:** the full mode set under `PS4='+ $(date +%s.%N) ' bash -x` for each command's start time, with
`vmstat -n 1` sampling CPU use, after `cargo clean -p inference-core` so every clippy configuration re-checks core
cold.

**Raw finding, cold core, 228 s in total:**

| Phase | Time | CPU in use |
|---|---|---|
| Six `--slim` clippy checks, one after another | 52 s | ~12% (core's lib and lib test on two cores) |
| CUDA-feature workspace clippy | 13.4 s | 21% |
| CPU workspace clippy | 12.7 s | 26% |
| Python binding tests | 10.2 s | 8.5% |
| C# build and tests | ~4 s | 7-16% |

With core warm (139 s in total), there is also a 27 s wait on the GPU-bound CUDA suite at 47%. One `--slim` check
after a one-line edit to core takes 2.6 s, because incremental compilation reuses the rest.

**Constraint:** every cargo command takes the target directory's lock, so cargo phases can't overlap within one target
directory, and a second target directory costs gigabytes of duplicated artifacts. Only work that isn't cargo can run
beside them: the CUDA suite's execution, and the Python and C# binding tests.

**Change:**
- The binding tests run in the background after the bindings library is built. The slim lint and the docs check run
  in the foreground meanwhile, overlapping them and whatever remains of the CUDA suite.
- The rlimit test still runs last and alone. One shared EXIT trap kills background jobs and removes temp files.
- `--slim` is skipped when `scripts/slim_needed.py` finds no changed file (against the merge-base with origin/master,
  plus untracked files) in inference-core or any workspace crate it depends on, a workspace-wide file, or these
  scripts. It uses `cargo metadata`, so new crates are covered. A crash or a missing origin/master runs the lint.
- `--sweep` saves the slim replay's artifact list in `target/debug/.slim-artifacts.json` and feeds it back on skipped
  runs, so the slim configurations' artifacts aren't deleted and later rebuilt cold.

**Raw finding, after:** cold core 215 s (-13 s, the binding tests now overlap the slim lint). With core warm the full
set takes 87 s. A change that stays above core (the api, server, webui, FFI, CLI, SDK, agent, bindings or docs) skips
the slim lint's six checks, 16 s warm or 52 s cold.

**Implication:** the slim checks' remaining serial time is core's front end run six times. With one target directory,
only fewer configurations or a faster core front end shorten it.

## Run 7 - 2026-10-06 12:01

Question: full local CI had grown to ~145 s even with nothing changed, and the CUDA suite peaked at ~16 GB of VRAM.
Which tests drive that, and which of them belong in it?

Phase timeline of `local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep` on master with nothing changed
(`bash -x` with a timestamped `PS4`): the CUDA suite ran in the background from 22 s to 136 s and was the long pole;
everything else (CPU clippy and tests, doctests, smoke builds, bindings, docs, the sweep replays) fit inside it. Total
145 s.

The real-checkpoint tests held most of that: IQ4_XS 27B (21 s, `threads-required = num-test-threads`, ~15 GB),
real-weight FLUX (20 s, alone, ~18 GB), three Qwen3.5 MTP runs (15-22 s each), and the two PaddleOCR-VL checks the
CUDA profile kept. They are deep checks (parity with recorded reference outputs, end-to-end runs on real weights),
not per-change tests, so they move to a `deep` nextest profile that `scripts/deep_checks.sh` runs on request; the
default and CUDA profiles exclude them, and `local_ci.sh --models` is gone.

CUDA suite after, with a VRAM sampler (`nvidia-smi --query-compute-apps` every 0.3 s, pid mapped to the test name
through `/proc/<pid>/cmdline`):

```
nextest --profile cuda: 2787 tests, 58 s wall (was ~114 s)
peak device memory: 6128 MiB, of which 1574 MiB is the idle desktop
largest test processes: 564 MiB (engine retention), 526 / 490 MiB (tiny Qwen3.5 MTP), most tiny-model tests 350-400 MiB
```

So the 16 GB peak was the IQ4_XS and FLUX checks; a CUDA context and its loaded modules account for most of each
remaining process, and no tiny-model test sizes its caches off free memory (the SDK's default paged KV budget is a
fixed context, `DEFAULT_PAGED_CONTEXT`).

Full CI on this branch: 146 s, CUDA suite 74 s inside it. The long pole is now `--slim`: six serial clippy runs, 67 s
to 136 s. It ran because the branch touches `scripts/local_ci.sh`, which `slim_needed.py` always counts, and most
code changes reach a crate inference-core depends on. Next: run the CPU suite in the background after its build, as
the CUDA suite is, so the slim lint overlaps it (CPU tests ~29 s, slim ~69 s).
