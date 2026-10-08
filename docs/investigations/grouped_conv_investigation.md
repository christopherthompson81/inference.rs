# #264: grouped and depthwise convolutions run one convolution per group

## Run 1 - 2026-10-08 09:48

Question: what does a native grouped kernel buy on CUDA, and where does the per-group split stay faster?

Code before: `Tensor::conv1d/conv2d` with `groups > 1` chunk input and weight per group and run im2col + cuBLAS GEMM
per chunk, then `cat` (`crates/inference-tensor/src/conv.rs`). The CUDA backend also carried direct conv kernels
(`conv1d_*`/`conv2d_*` in `inference-tensor-kernels/src/conv.cu`), reached only under `cudnn` for non-contiguous
weights; the conv1d one computed the input position as `(stride * l + offset) * dilation` (wrong for dilation > 1).
inference-layout had its own f32 depthwise conv2d kernel (`depthwise_conv2d_f32`).

Change: the direct kernels take a `groups` argument (output channel `c` reads group `c / (c_out / groups)`) and the
conv1d indexing is fixed; `ParamsConv{1,2}D` carry `groups`; CUDA runs a grouped conv in one launch of the direct
kernel when nothing tracks gradients (the conv backprop ops carry no group count).

Command: `cargo nextest run --profile cuda --features cuda --workspace --lib -E
'test(bench_depthwise_conv_split_vs_direct)' --run-ignored only --no-capture` (RTX 3090, BF16, 20 iterations each).

Raw, per-group split vs one direct launch:
- conv2d depthwise 3x3, 640 ch, 32x32: 16.3 ms vs 0.061 ms
- conv1d depthwise k15, 1024 ch, L1500: 26.3 ms vs 0.162 ms
- conv2d 3x3, 512 ch, 2 groups (256 per group): 0.198 ms vs 2.133 ms
- conv2d 3x3, 512 ch, 8 groups (64 per group): 0.275 ms vs 0.410 ms
- conv2d 3x3, 512 ch, 32 groups (16 per group): 0.942 ms vs 0.128 ms

The direct kernel loops over a group's input channels per output, so a few wide groups lose to per-group GEMMs. The
split costs a launch pair per group, so many narrow groups lose to the direct kernel. `groups >= c_in / groups`
separates all five cases; with it as the routing rule the routed timings are 0.061, 0.163, 0.198, 0.281 and 0.125 ms.

Every model call site with groups > 1 is depthwise (gemma3n vision, gemma3n/gemma4 audio, Phi-4-MM conformer, LFM2,
PP-DocLayout-v3), so all of them take the direct kernel on CUDA.

## Run 2 - 2026-10-08 09:48

Question: can inference-layout's depthwise CUDA kernel go, with its CUDA path calling `conv2d` plus a bias add?

Command: `cargo nextest run --profile cuda --features cuda --workspace --lib --bins --tests -E 'package(inference-layout)'`.

Raw: `pp_doclayout_v3::model::tests::batched_forward_matches_single` failed on CUDA, logits rel err 2.05e-4 (tolerance
1e-4; master 5.6e-7), deterministic over four runs. With only the tensor-layer change it passed. A scratch check showed
the new depthwise conv exact and batch invariant (0 difference batched vs single and vs CPU). Probing the encoder
top-k: batched and single select the same 300 queries, but two of them score 0.033843584 and 0.033843573 and swap
ranks 175/176. The test compares queries by position, so the swap reads as a numeric difference; the new kernel's
outputs (taps then bias, the old kernel added bias first) only moved the scores enough to expose the near-tie.

Sorting queries by predicted box to align them failed (rel err 6.5e-4): random weights saturate boxes into ties. Aligning
by the encoder position each query was selected from (now recorded in `Intermediates::query_sources`) makes logits,
boxes and masks match; order logits are masked by rank (strict upper triangle), so they compare on cells kept in both.

Next: full CI, then the PR.
