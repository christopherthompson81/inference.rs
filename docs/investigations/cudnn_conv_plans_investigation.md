# #263: cache cuDNN convolution plans per shape

## Run 1 - 2026-10-08 16:06

Question: what makes the `cudnn` feature's convolutions slower than the default im2col + GEMM path, and does it
win once that is fixed?

Code before (`crates/inference-tensor/src/cuda_backend/cudnn.rs`): every call built four descriptors, ran
`pick_algorithm` (`cudnnGetConvolutionForwardAlgorithm_v7`), and allocated a zeroed workspace; no math type was set.
Depthwise and grouped convs no longer reach cuDNN (#369 sends them to the direct kernel under `cudnn` too), so the
question is the dense shapes.

Command: `bench_dense_conv` (`crates/inference-tensor/src/conv.rs`, ignored), 20 iterations after one warm-up between
synchronizes, on the default build (`--profile cuda --features cuda --workspace --lib`) and on the cuDNN one (`-p
inference-tensor --features cudnn --lib`, narrowed to that crate so the feature adds one copy of cudarc and
inference-tensor to `target/` instead of the workspace). The plan cache's variants were switched by temporary
environment variables. RTX 3090, cuDNN 9.26, ms per call:

| case | im2col | cuDNN before | cached, default math | cached, tensor math | cached, find + tensor |
|---|---|---|---|---|---|
| BF16 conv2d 3x3 256->256 @64x64 | 0.398 | 2.221 | 0.485 | 0.108 | 0.105 |
| BF16 conv2d 1x1 640->1280 @32x32 | 0.123 | 2.310 | 0.188 | 0.052 | 0.052 |
| BF16 conv2d patch14 3->1152 @896 | 0.345 | 0.512 | 0.504 | 0.403 | 0.372 |
| BF16 conv1d k3 128->512 L3000 | 0.099 | 1.980 | 0.125 | 0.056 | 0.076 |
| F16 conv2d 3x3 | 0.396 | 2.224 | 0.234 | 0.071 | 0.065 |
| F16 conv2d 1x1 | 0.125 | 1.763 | 0.112 | 0.048 | 0.050 |
| F16 conv2d patch14 | 0.348 | 3.078 | 3.030 | 0.324 | 0.323 |
| F16 conv1d k3 | 0.099 | 1.524 | 0.084 | 0.076 | 0.085 |
| F32 conv2d 3x3 | 0.639 | 0.542 | 0.236 | 0.192 | 0.194 |
| F32 conv2d 1x1 | 0.165 | 0.407 | 0.069 | 0.059 | 0.057 |
| F32 conv2d patch14 | 0.581 | 0.350 | 0.393 | 0.323 | 0.320 |
| F32 conv1d k3 | 0.160 | 0.376 | 0.093 | 0.079 | 0.081 |

Per-call planning was most of the cost (BF16 3x3 2.22 -> 0.49 ms with a cache alone). The rest was the math type: with
`CUDNN_DEFAULT_MATH` half inputs stay on CUDA cores; `CUDNN_TENSOR_OP_MATH` cuts them 2-10x more. Timing the
algorithms (`cudnnFindConvolutionForwardAlgorithm`, through a `ConvForward::find_algorithm` tried in the vendored cudarc
and removed again) picks the same or a marginally different algorithm than the heuristic, so the plan keeps the cheaper heuristic.
NHWC (step 3 of the issue) is not needed: NCHW with tensor math already beats im2col.

A new parity test (`dense_conv_on_cuda_matches_the_cpu`: strided input, padding, stride, dilation, all three dtypes)
failed on the cuDNN build in F32: 0.018 against a peak of 55 (3.3e-4 relative). That is TF32: the default math runs F32
convolutions on tensor cores on Ampere (this predates the change). F32 now sets `CUDNN_FMA_MATH`, matching the
im2col path's precision; it passes, and F32 timings become 0.188 / 0.156 / 0.368 / 0.102 ms, still at or under
im2col's.

Final (pick, tensor math for BF16/F16, FMA for F32), against im2col: BF16 3x3 0.134 vs 0.398, 1x1 0.058 vs 0.123,
patch14 0.467 vs 0.345 (cuDNN slower), conv1d 0.059 vs 0.099; F16 0.071 / 0.054 / 0.396 / 0.082 vs 0.396 / 0.125 /
0.348 / 0.099. cuDNN wins most dense shapes 1.8-3.7x; the 14x14 stride-14 patch embed in half precision is the
exception (about 15% slower). The cargo-features page now names `cudnn` the opt-in speed choice for conv-heavy
models. No conv-heavy checkpoint (gemma3n, conformer audio) is on disk, so the end-to-end check of step 4 is not run.

Review of the change: plans and their workspaces were unbounded per thread, so variable-length audio or image sizes
would each keep a workspace (FFT and Winograd ask for hundreds of MB). Now the plans hold only descriptors and the
algorithm, one workspace per device per thread grows to the largest plan used, and the plan map clears past 256
entries. Known caveat: a plan first used inside a CUDA graph capture allocates its workspace in the graph; decode
graphs hold no dense conv today (the short and GDN convs are depthwise and take the direct kernel).
