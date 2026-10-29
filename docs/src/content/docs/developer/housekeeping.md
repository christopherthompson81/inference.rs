---
title: Housekeeping
description: The recurring round that keeps build time, test time and duplication down.
---

Housekeeping is a recurring round, run between feature work, that keeps the edit-build-test loop short and the code
base from growing copies of itself. Each round measures, ranks, and lands the best items one PR at a time.

## Measure

`scripts/housekeeping.sh` gathers everything but the CI timing:

```bash
# Duplicated code and unused dependencies (builds nothing)
scripts/housekeeping.sh

# LLVM IR lines per heavy crate, and incremental rebuild times after touching nn, core and a model family
scripts/housekeeping.sh --ir --incremental

# Slowest tests and test modules of a full local CI run
/usr/bin/time -f "wall %e s" scripts/local_ci.sh --lint --tests --cuda --bindings --size --sweep 2>&1 | tee ci.log
scripts/housekeeping.sh --slow-tests ci.log
```

- **Duplication** (`scripts/dup_scan.py`): 12-line windows of normalized lines (whitespace collapsed; comments, blank
  and brace-only lines dropped) found in more than one place under `crates/`, without `third_party`. It prints the
  total, the split by crate and the file pairs that share the most windows. `scripts/dup_scan.py --pair A B` lists the
  spans two files share as line ranges in each, which is where checking a candidate starts.
- **Unused dependencies**: `cargo machete`. Check each hit before removing it: a dependency used only from a macro or
  under a feature can look unused.
- **IR lines**: `cargo llvm-lines` per crate. Generic code compiled once per caller is what makes a crate slow to build
  and large to ship; the top functions of `cargo llvm-lines -p <crate> --lib` show where it is.
- **Incremental rebuilds**: seconds for `cargo check --workspace --tests` and the nextest test build after a `touch` of
  a crate low in the stack. An edit low in the stack is what everyday iteration pays for.
- **Test time**: per nextest run, the slowest tests and the modules whose tests sum the most. A module of many medium
  tests matters as much as one slow test.

## Rank

Pick items by what they save: IR lines and build time first, then duplicated lines, then test time.

- Verify a duplication candidate by reading both spans (`--pair`): code that matches line for line but differs in
  meaning (another model's constants, another dtype's kernel) is not a candidate.
- Some repetition is deliberate: per-architecture SIMD kernels (AVX, NEON, the k-quants), the dtype arms of a kernel
  launch, and code derived from candle that keeps its upstream shape.
- Sharing must not drop a feature, a model or a kernel variant. A slow test is cut by a smaller checkpoint or fewer
  repeats, never by losing what it checks.

## Record

Each round adds a dated run to the investigation logs, with the commands, the raw numbers and what was chosen:

- `docs/investigations/codebase_size_investigation.md`: duplication, IR lines, unused dependencies, and the ranked
  list.
- `docs/investigations/test_time_investigation.md`: the CI wall time, each suite's time and the slowest tests.
- `docs/investigations/cuda_build_time_investigation.md`: cold and incremental build timings.

Compare against the previous run's numbers. A suite that got slower without new tests to account for it is the first
thing to explain.
