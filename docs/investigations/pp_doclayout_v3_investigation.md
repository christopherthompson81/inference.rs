# PP-DocLayoutV3 in candle (mistralrs-layout)

Goal: port PP-DocLayoutV3 (RT-DETR-style document layout detector, 25 classes, with reading-order and mask heads)
to candle so layout detection runs in-process in the fork, replacing the ONNX Runtime dependency used by the C#
docling pipeline.

References:
- Weights: `PaddlePaddle/PP-DocLayoutV3_safetensors` (downloaded to `/mnt/data/models/PP-DocLayoutV3_safetensors`)
- PyTorch reference: transformers 5.17.0 `models/pp_doclayout_v3/modeling_pp_doclayout_v3.py` + `hgnet_v2`
- Independent ground truth: Paddle ONNX export `/mnt/data/models/PP-DocLayoutV3.onnx`
- Test image: PaddleX public demo page `layout_demo.jpg` (1654x2339, two-column academic page, public sample)

## Run 1 - 2026-09-24 17:40

Question: is the HF implementation a trustworthy reference? The safetensors checkpoint has no `decoder.class_embed` /
`decoder.bbox_embed`; HF ties them to `enc_score_head` / `enc_bbox_head` via `_tied_weights_keys`. If the tie were
wrong, HF output would diverge from the Paddle ONNX export.

Command: `venv/bin/python ref_dump.py layout_demo.jpg ref_demo.safetensors` (scratchpad). Feeds the same
`pixel_values` (HF processor output) into both HF and ONNX.

Finding: 13 detections at threshold 0.5 in both. Every box and score matches to the printed precision
(e.g. text 0.9605 [337.1, 183.1, 895.0, 653.7] in both). ONNX output column 6 is a reading-order rank; it is not
identical to HF's `order_seq` (e.g. 51 vs 47, 288 vs 299) but the relative ordering of the 13 kept boxes is the same.

Other facts gathered:
- Checkpoint names differ from HF module names; HF renames at load (`out_proj`->`o_proj`, `layers.N.fc1`->
  `layers.N.mlp.fc1`, `encoder.encoder.N`->`encoder.aifi.N`). Rust will use checkpoint names directly.
- Backbone is the default HGNetV2-L layout (stem 3->32->48, stages 128/512/1024/2048, light blocks in stages 3-4).
- `mask_enhanced=True`: the initial reference boxes come from the encoder masks (`mask_to_box_coordinate`), not from
  `enc_bbox_head` + anchors. The anchors/enc_bbox_head path only feeds topk selection indirectly via scores.
- ONNX outputs: `[N,7]` (cls, score, x1, y1, x2, y2, order), `[N]` counts, `[N,200,200]` int masks.
- No `grid_sample` in candle; deformable attention needs a hand-written bilinear sampler.

Implication: dump intermediates from HF and port stage by stage (backbone -> encoder -> decoder -> post-process),
checking each against the dump.

## Run 2 - 2026-09-24 18:25

Question: does the first full candle port (`mistralrs-layout`, CPU, f32) match the HF dump stage by stage?

Command:
`cargo run --release -p mistralrs-layout --example pp_doclayout_v3_parity -- --cpu --model .../PP-DocLayoutV3_safetensors --reference ref_demo.safetensors --image layout_demo.jpg`
(model forward is fed HF's own `pixel_values`, so model parity is isolated from preprocessing.)

Finding (model): every intermediate cos >= 0.9999999 on the first run.
- backbone.0-3 max_abs <= 1.2e-4; encoder.pan.* <= 4e-4; mask_feat 2.3e-4; enc_score 6e-5
- init_reference_points (mask->box path) max_abs 4.8e-7
- decoder.hidden.0-5 max_abs <= 2.8e-4 (deformable attention sampler is right)
- logits 2e-4, pred_boxes 5.6e-6, order_logits 0.28 on values of magnitude ~6e3, out_masks 0.08 on ~46
- Post-processed: identical 13 detections to HF, same labels, scores to 4 dp, boxes to 0.1 px, same reading order.

Finding (preprocessing): NOT matching. Our resize of the JPEG vs HF `pixel_values`: 99,731 of 1,920,000 values
differ, max_abs 0.086 (22/255), cos 0.9999987. Too large to be round-half-even ties.

Finding (speed): CPU forward 4.2 s/page, far too slow; GPU not tried yet.

Implication: model port is correct. Next: isolate whether the pixel diff is JPEG decode (PIL vs `image` crate) or
the bicubic resize, by feeding a PIL-decoded PNG.

## Run 3 - 2026-09-24 18:50

Question: where does the preprocessing diff come from (JPEG decode vs resize), and what exactly does HF's resize do?

Commands (scratchpad): re-ran parity with a PIL-decoded PNG of the same page; then `emu_resize.py`, `emu2.py`,
`emu3.py` emulating candidate resize algorithms in numpy against HF `pixel_values`.

Findings, in order:
- PIL-decoded PNG through our float bicubic (A=-0.75, clamped taps, f32 accumulate, round-half-even): still 90,004
  values differ, max 22. So the resize is the main problem, not JPEG decode.
- `torchvision.v2.functional.resize(uint8, BICUBIC, antialias=False)` on the PNG reproduces HF exactly (0 diffs),
  while `torch.nn.functional.interpolate(float)` + round reproduces *our* 90k diffs. torchvision takes torch's
  native CPU uint8 resize path, which is PIL-style: separable, fixed-point, uint8 intermediate.
- PIL-style emulation (support-2 truncated taps renormalized, int16 weights): A=-0.75 -> 2,400 diffs (all in the last
  output column), A=-0.5 -> 214,942 diffs. Dead end on the tap layout.
- Right-edge variants on column 799: truncated+renormalized max 4; 4 taps with clamped index -> 0 diffs.
- Final recipe, 0 diffs on the whole page: torch-standard tap placement (`real = scale*(o+0.5)-0.5`, 4 taps, clamped
  indices), A=-0.75 weights computed in f64 (f32 gives 5,034 diffs), quantized to int16 with the torch precision rule,
  horizontal pass into u8 (`(acc + 2^(p-1)) >> p`, clamp), then vertical pass the same way.

After porting to `preprocess.rs`:
- PIL PNG input: 0 of 1,920,000 values differ (bit-exact).
- JPEG input: 14,131 values differ, all by exactly 1/255: JPEG decoder rounding (image crate vs libjpeg-turbo).

Implication: preprocessing is solved; the remaining JPEG delta is decoder-level and outside the model. Next: CUDA.

## Run 4 - 2026-09-24 19:15

Question: CUDA parity, and why is it slow? (RTX 3090, f32, batch 1, 800x800)

Commands: `cargo build --release -p mistralrs-layout --features cuda --example pp_doclayout_v3_parity` then the
parity binary; `nsys profile -t cuda` + `nsys stats -r cuda_gpu_kern_sum`; ORT CPU timing in Python.

Findings:
- CUDA parity: all stages cos >= 0.9999998; decoder.hidden.5 max_abs 1.6e-3, logits 1.2e-3; same 13 detections.
- Baseline speed: 220 ms/page CUDA, 4.2 s/page CPU. Paddle ONNX on ORT CPU (16 threads): 0.62 s/page.
- Cause: candle's `conv2d` with `groups > 1` chunks into one conv per group on every backend (conv.rs
  `conv2d_with_algo`), and HGNetV2 light blocks + downsample layers are depthwise over 128-1024 channels.
- Fix: `Depthwise` in `layers.rs`, a sum of k*k shifted taps (stride handled by padding to a stride multiple and
  reshaping to `(b, c, h/s, s, w/s, s)` so a strided tap is a phase select + narrow). CUDA 220 -> 69 ms,
  CPU 4.2 -> 1.78 s, parity unchanged.
- nsys after the fix (4 forwards): im2col_f32 24%, bmul_f32 17%, badd_f32 16% (the depthwise taps, ~850 of each per
  forward), ucopy 8%, sgemm ~15%. 18k kernel launches for 4 forwards.
- Dead end: `--features cuda,cudnn`. No faster (72 ms) and it breaks parity: backbone.3 max_abs 6.9e-2,
  pred_boxes cos 0.921, logits cos 0.997. Consistent with cuDNN using TF32 convolutions on Ampere; the drift flips the
  encoder top-k query selection. Keep plain f32 convs; this model's query selection is precision-sensitive.

Implication: correctness done on CPU and CUDA. Perf headroom is a real depthwise kernel (removes ~40% of GPU time)
and a cheaper dense 3x3 path than candle's im2col. Not blocking.

## Run 5 - 2026-09-24 19:35

Question: does parity hold beyond the one demo page, end to end (our decode + preprocess + model + post-process vs
HF with PIL), and is batching correct?

Test set (all public PaddleX demo images): the two-column academic page from Run 1; a dense multi-column newspaper
page with a photo and CJK text (RGBA PNG); a skewed phone photo of an open book with formulas and a figure; a short
receipt-like text crop (a JPEG with a `.png` name); two table crops.

Commands: `pp_doclayout_v3_detect` (new example, JSONL output) on all six, once per image and once with `--batch`
(B=6); `compare_dets.py` runs HF on the same files and matches detections by class + IoU.

Findings:
- Batch vs single: identical detection lists on every image, max score delta 4.8e-7.
- vs HF: identical detection counts and classes on all six (13, 31, 18, 8, 1, 1), no unmatched boxes either way,
  reading order identical. Worst IoU 0.988 (phone photo), worst score delta 0.031 (one table crop, JPEG).
- The misnamed JPEG crashed `image::open` (extension-based); example now sniffs the format.
- Wall time for 6 images incl. decode + preprocess: 553 ms single, 480 ms batched (CUDA).

Implication: functionally complete for boxes/classes/scores/reading order. Masks/polygons are computed by the model
but not yet exposed by post-processing (the C# pipeline only consumes boxes).

## Run 6 - 2026-09-24 20:10

Question: do the PR #1 review fixes keep parity?

Fixes: u32 cross-head gather offsets (f32 offsets lose exactness once b*heads*S > 2^24, ~160 pages per batch);
load-time bail when memory tokens < num_queries or > 2^24; config `batch_norm_eps` wired into the RT-DETR
`conv/norm` layers and `decoder_input_proj` (HF uses defaults elsewhere); bail on unsupported `mask_enhanced`,
`learn_initial_query`, `normalize_before`, `anchor_image_size`, `eval_size`; final masks only on request; mask->box
temporaries deduplicated; one host transfer per output in `detect_batch`; empty batches return `Ok(vec![])`;
the crate's `cudnn` feature was removed. Candle-wide feature unification can still turn cuDNN on if another crate in
the same build enables `candle-core/cudnn`, and that re-introduces the Run 4 drift. Keep layout builds cuDNN-free.

Commands: parity example on CPU and CUDA; `pp_doclayout_v3_detect --batch` on the six pages + `compare_dets.py`.

Findings: CPU pred_boxes max_abs 3.8e-6, CUDA 1.4e-5 (unchanged); six-page HF comparison identical to Run 5;
empty batch returns immediately; CUDA forward 70 ms (skipping final masks is not a measurable win on GPU).

# Performance investigation

Baseline after PR #1: CUDA (RTX 3090, f32, B=1, 800x800) 70 ms/forward; CPU (16 threads) 1.72 s/forward;
Paddle ONNX on ORT CPU 0.62 s. Kernel mix from Run 4: im2col 24%, depthwise bmul+badd 33%, ucopy 8%, sgemm ~15%;
total kernel time ~= wall time, so the GPU is busy, not launch-starved.

## Run 7 - 2026-09-24 20:30

Question: was the Run 4 cuDNN drift TF32, and is fp32 cuDNN a speed lever for the dense convs?

Command: build with `--features cuda,candle-core/cudnn`, run the parity example with `NVIDIA_TF32_OVERRIDE=1` and `=0`.

Finding:
- `=1`: backbone.3 max_abs 6.9e-2, pred_boxes cos 0.900; 75-77 ms.
- `=0`: backbone.3 max_abs 1.2e-4, pred_boxes max_abs 1.4e-5 (parity restored); 69-70 ms.
So the drift is TF32, confirmed. fp32 cuDNN is not faster than candle's im2col+sgemm here.

Implication: cuDNN is not the lever. Next: attribute time per op type, starting with 1x1 convs (which go through
im2col for no reason) and the depthwise taps.

## Run 8 - 2026-09-24 20:45

Question: how much do 1x1 convs lose by going through candle's im2col?

New tool: `examples/pp_doclayout_v3_bench.rs` (random input, model forward only, warmup 3, 10 iters, synced by the
host copy of the logits). Baseline with it: B=1 66.8 ms, B=4 58.0 ms/img.

Change: `Pointwise` (1x1, stride 1) = `w (o, c) @ x (b, c, h*w)` + bias. 66 of the 103 dense convs are 1x1.

Finding: B=1 60.5 ms, B=4 52.4 ms/img; parity unchanged (pred_boxes max_abs 1.7e-5 CUDA).

## Run 9 - 2026-09-24 21:00

Question: what does a real depthwise kernel buy over the shifted-tap sum?

Change: `depthwise.rs` CustomOp3 (`x`, `w`, `b`): CPU direct loop parallel over channel planes (rayon), CUDA one thread
per output compiled with NVRTC at first use (`cuda_kernels.rs`, cudarc's nvrtc through candle; no build.rs). The tap sum
stays as the Metal fallback.

Findings:
- CUDA B=1 46.5 ms, B=4 39.4 ms/img; CPU 1.72 -> 1.45 s. Parity unchanged.
- Kernel time 62.2 -> 45.7 ms/forward, launches 4514 -> 2994. The depthwise kernel is no longer in the top 16.
- Dead end in the process: my first CUDA unit test was silently not applied (a Python string replace did not match
  rustfmt'd code and I had not asserted it), and it "passed" in 0.01 s. nsys showed zero kernels. Rewritten by hand;
  it now runs the CUDA path (0.24 s).

## Run 10 - 2026-09-24 21:20

Question: why is im2col 26% of GPU time?

Command: `nsys stats --force-export=true -r cuda_gpu_trace` on the bench; paired each im2col with the following GEMM.

Findings:
- im2col costs 2-4x its GEMM: e.g. stage-1 3x3 (48ch, 200x200) im2col 469 us vs sgemm 147 us, which is 69 MB written
  at ~147 GB/s on a ~936 GB/s card. Candle's layout is `(b, ho*wo, c*k*k)`, so adjacent threads read input
  addresses one channel plane apart (uncoalesced), and the conv then needs a transpose copy of the output.
- Also: `nsys stats` reused a stale `.sqlite` export once; always pass `--force-export=true`.

Change: `im2col.rs` cols-last layout `(b, c*k*k, ho*wo)` (adjacent threads walk `ox`: coalesced reads and writes);
the conv becomes `w (o, c*k*k) @ cols`, landing directly in NCHW with no transpose.

Finding: CUDA B=1 40.5 ms, B=4 35.1 ms/img; CPU 1.37 s. Parity unchanged (pred_boxes max_abs 1.2e-5).

## Run 11 - 2026-09-24 21:40

Question: can the bias add ride along with the dense-conv GEMM, and is the im2col kernel division-bound?

Change: im2col now emits `(b, c*k*k + 1, ho*wo)` with a trailing ones row, and `w2d` carries the bias as its last
column, so the GEMM adds the bias. Kernel uses a 2D grid (row from `blockIdx.y`, grid-stride over rows past 65535)
instead of 6 divides per element.

Finding: B=1 40.5 -> 37.1 ms, B=4 35.1 -> 31.9 ms/img; im2col 8.35 -> 5.72 ms/forward. Parity unchanged.

## Run 12 - 2026-09-24 22:00

Question: where does the rest go? Synced per-stage timers (temporary, reverted) and an nsys API breakdown.

Findings:
- Synced stage times (they inflate the total to 61 ms, so they rank stages only): backbone 14.0, encoder 18.7,
  mask-init 7.4, decoder 14.3, rest < 1.5 ms.
- Fused `mask_to_box` (one block per query, shared-memory min/max reduction; CPU rayon) replaced about 15 full passes over
  `(1, 300, 40000)`. B=1 only 37.1 -> 36.5 ms but B=4 31.9 -> 29.4 ms/img: B=1 was host-bound, not GPU-bound.
- `cuda_api_sum`: 2678 `cuLaunchKernel` plus **1867 `cuMemcpyHtoDAsync` per forward** (7.4 ms host). Correlating
  each HtoD copy with the next launch in the nsys sqlite: they precede `bmaximum`/`bminimum` (clamp), `ge`/`le`
  (scalar compares), `ucopy`, `badd`, `affine`. candle uploads every scalar operand and strided-layout metadata as a
  tiny H2D copy. Nearly all of them come from the decoder's tensor-op bilinear sampler and `inverse_sigmoid`.
- Fused multi-scale deformable-attention sampler (`msda.rs`, thread per (b, q, head, channel), level shapes as
  kernel args; CPU rayon). The value table is used in its natural `(b, S, h, d)` layout, so no transpose. The old
  tensor-op sampler is kept for Metal and as the test oracle.

Result: CUDA B=1 36.5 -> 27.0 ms, B=4 29.4 -> 26.9 ms/img (B=1 == B=4 per image now: GPU-bound); CPU 1.37 -> 1.17 s.
Parity: pred_boxes max_abs 1.05e-5 CUDA / 1.5e-5 CPU. Unit test: fused vs tensor-op sampler < 1e-5 on CPU+CUDA
with out-of-bounds sample locations.

## Run 13 - 2026-09-24 22:15

Question: RepVGG re-parameterization, and how do we compare with ORT on the same GPU?

Change: `ConvNormSpec::load_with_1x1` adds the folded 1x1 branch into the centre tap of the folded 3x3 kernel, so each
encoder RepVGG block is one conv + SiLU (12 pointwise convs and 12 full-map adds gone). Exact up to fp rounding.

Findings:
- CUDA B=1 27.0 -> 26.1 ms, B=4 25.9 ms/img. encoder.pan.* max_abs <= 2.8e-4 (unchanged), pred_boxes 1.3e-5.
- Kernel mix at 27.7 ms (before this change): fp32 SIMT sgemm ~11.5 ms, im2col 5.8, badd 3.9 (237x), copy2d 1.0,
  relu/silu 1.3, launches 1210/forward (was 4514), H2D copies 648/forward (was 1867).
- Reference: Paddle ONNX on onnxruntime-gpu 1.30 CUDA EP (cudnn EXHAUSTIVE), same GPU, B=1: 36.5 ms with TF32,
  39.3 ms fp32. We are ~1.5x faster in strict fp32.

Implication: GPU is in good shape; CPU (1.17 s vs ORT CPU 0.62 s) is now the laggard.

## Run 14 - 2026-09-24 22:40

Question: where does CPU time go (1.17 s vs ORT CPU 0.62 s)?

Commands: synced per-stage timers (temporary); `RAYON_NUM_THREADS` sweep; throwaway micro-benchmarks of candle conv2d vs
raw GEMM; per-op-kind accumulators in `ConvNorm::forward` (temporary); FLOP count of the HF model via forward hooks.

Findings:
- Stages: backbone 547 ms, encoder 415 ms, decoder 120 ms.
- Threads: 16 (default, all logical CPUs) 1.11 s, 8 (physical cores) 1.03 s, 4 1.21 s. Hyperthreads hurt sgemm a little.
- `perf` unavailable (`perf_event_paranoid=4`).
- Dead end: `candle-core/mkl` does not link. Candle references `hgemm_`, which the MKL 2020 static build lacks. The
  repo's own `mkl` feature has the same problem.
- candle CPU conv2d is ~half GEMM speed: 3x3 256->256 @100x100 51 ms (229 GFLOP/s) vs the same GEMM 27 ms (441).
  On CPU the im2col copy costs as much as the GEMM.
- Model FLOPs (HF hooks, 800x800): 185 GFLOP total. Encoder dense 72.6, backbone pointwise 45.5, backbone dense 23.5,
  linear 21.9 (mostly the decoder's per-layer `value_proj` over all 13,125 memory tokens), encoder pointwise 19.5.
  ORT's 0.62 s is ~300 GFLOP/s effective.

Change: `cpu_conv.rs` implicit-GEMM conv. The input is padded once into stride-phase planes; each kernel tap is one
accumulating `gemm::gemm` (beta=1) over a shifted strided view; output is computed on the padded-width grid and cropped.
No im2col buffer. 256ch 3x3: 26 ms (449 GFLOP/s). 48ch: 10.5 ms vs 21.4 (im2col). Only loses for the 3-channel stem
(17 ms vs 5 ms), so convs with < 16 input channels keep im2col + one GEMM.

Result: CPU 1.17 -> 0.97 s (16 threads) / 0.87 s (8 threads). Per op kind at 8 threads: dense 412, pointwise 174,
depthwise 76, activations 35 ms.

Next CPU lever, not taken: MKL sgemm measured in a throwaway crate is 1.1-2x the `gemm` crate (256x10000x2304: 667 vs 410
GFLOP/s; 48x40000x48: 487 vs 238). It would need an optional feature calling `sgemm_` directly (bypassing candle's
broken mkl feature), and it is Intel-only.

## Run 15 - 2026-09-24 23:05

Question: is batching correct on CPU? (Run 5 only checked batching on CUDA.)

Finding: **no, and it was already broken on master (PR #1)**. CPU `--batch` on the six pages: image 1 correct, later
images lose or gain detections (e.g. 8 -> 0, 1 -> 0). CPU single-image and CUDA batched were correct.

Bisection:
- candle ops tested batched vs per-item on CPU: upsample bilinear/nearest, max_pool2d, pad, conv2d, batched matmul,
  broadcast matmul, matmul with a transposed rhs. All exact.
- Model-level (2-image batch vs single, all intermediates): backbone exact, divergence starts at `encoder.pan.*`.
- Root cause: **candle CPU batched matmul with a transposed-view lhs is wrong for batch items > 0**. `Linear` on
  `x.transpose(1, 2)` with shape (b, n, c): item 0 correct, item 1+ errors of 1e1-1e2, for every shape tried. AIFI fed
  the transposed feature map straight into `v_proj`. CUDA is unaffected.
- Fix: `.contiguous()` before AIFI. Every stage is then bit-identical batched vs single.
- Regression test `batched_forward_matches_single` (random-weight full model at 160x160, CPU + CUDA). First version
  passed even with the fix reverted: with 0.05-scale random weights the signal dies in the backbone and the outputs are
  bias-dominated. With fan-in-scaled weights and ~1 BN gammas it fails without the fix (rel err 8.9e-3) and passes with it.
- Six pages CPU batched after the fix: identical to HF (same as CUDA).

## Run 16 - 2026-09-24 23:15

Question: with the model at 26 ms, what bounds end-to-end GPU throughput?

Finding: 6 pages end to end 334 ms vs 6 x 26 ms model. The single-threaded bicubic resize of full-resolution pages was
a big part. Both resize passes now run parallel over rows, and `detect_batch` preprocesses images in parallel:
334 -> 269 ms, byte-identical detections, pixel values still 0/1,920,000 different from HF on the PNG.

## Summary (RTX 3090 / i7-10700K, f32, 800x800)

| Step | CUDA B=1 | CUDA B=4 per img | CPU |
|---|---|---|---|
| PR #1 baseline | 66.8 ms | 58.0 ms | 1.72 s |
| 1x1 conv as matmul | 60.5 | 52.4 | |
| depthwise kernel (NVRTC) | 46.5 | 39.4 | 1.45 s |
| cols-last im2col | 40.5 | 35.1 | 1.37 s |
| bias in GEMM, 2D im2col grid | 37.1 | 31.9 | |
| fused mask->box | 36.5 | 29.4 | |
| fused deformable-attn sampler | 27.0 | 26.9 | 1.17 s |
| RepVGG re-param | 26.1 | 25.9 | |
| CPU implicit-GEMM conv | | | 0.97 s (0.87 s @ 8 thr) |
| Paddle ONNX on ORT (reference) | 36.5 (TF32) / 39.3 (fp32) | | 0.62 s |

Parity after every step: pred_boxes max_abs <= 1.8e-5 vs HF; the six-page end-to-end comparison is identical to Run 5
on CPU and CUDA.

## Run 17 - 2026-09-25 00:10

Question: why is ORT faster on CPU (0.62 s vs our 0.87 s)? Is it MKL?

Commands: onnxruntime 1.30 CPU (`get_build_info`, `get_available_providers`), `SessionOptions.enable_profiling`,
per-op-type kernel time averaged over 4 runs; CUDA EP re-measured with IO binding (all tensors on device).

Findings:
- Not MKL: plain pip build, CPU EP only, so it uses MLAS. The optimized graph contains `ReorderInput`/`ReorderOutput`:
  convs run in MLAS's blocked NCHWc layout as direct convolutions with fused bias/activation (no im2col).
- ORT per-run kernel time 505 ms: Conv 206.5 ms (118 convs, incl. pointwise/depthwise and fused activations);
  everything else ~300 ms (Add 67, Where 47, Cast 31, Expand 21, Concat 20, MatMul/FusedMatMul 39, GridSample 7).
- Ours at 8 threads: convs ~697 ms (dense 412 + pointwise 174 + depthwise 76 + activations 35), everything else
  ~180 ms. So ORT's convs are ~3.4x ours; the non-conv rest of our model is already faster than ORT's graph.
- CUDA with IO binding (fair, no host copies): ORT fp32 34.3 ms, TF32 31.4 ms, vs ours 26.4 ms fp32 (1.30x / 1.19x).
  The earlier 39.4 / 36.5 ms ORT numbers included ~5 ms of host transfers.

Implication: MKL only accelerates the GEMM inside our conv path (est. ~697 -> ~450 ms) and cannot reach MLAS. The CPU
lever is the conv algorithm: an NCHWc-style direct conv with fused bias+activation (or oneDNN as an optional dep).
Cheap side win: depthwise is 76 ms for 0.6 GFLOP (naive scalar loop; vectorize over width). ORT TensorRT EP not measured.

# CPU conv investigation (reproducing MLAS)

Target from Run 17: ORT/MLAS convs 207 ms vs ours ~697 ms (8 threads). Plan: AVX2+FMA direct conv in NCHW with
register blocking (4 output channels x up to 3 vectors of 8 pixels = 12 ymm accumulators; per input channel and tap:
<= 3 input loads + 4 weight broadcasts for 12 FMAs), input padded + stride-phase split once so every tap is a unit-stride
vector load, bias preloaded into the accumulators and ReLU/SiLU applied before the store. Then an AVX2 depthwise kernel.
Runtime feature detection; the implicit-GEMM path stays as the non-AVX2 fallback.

## Run 18 - 2026-09-25 01:10

Question: does the AVX2 direct conv (NCHW, 4 oc x 3 pixel vectors, fused bias/act) beat implicit GEMM, and why not more?

Findings, in order (throwaway `zz_direct` micro-benchmark, 8 threads; GFLOP/s):
- First version: wins on 48-96 channel layers (x2) and the 3-channel stem (x5), loses on 256ch@100 (306 vs 424).
- Cause 1, cache footprint: each (tile, row) task swept all 256 channels x 3 rows x 9 taps (~98 KB input + 36 KB weights).
  Fix: input-channel blocks sized from the real footprint (distinct phase rows x 48-px segment vs a 24 KB L1 budget),
  resuming partial sums from the output row. 256ch@100: 306 -> 451.
- Checked for register spills via objdump: the XV=1 inner loop keeps accumulators in ymm5-8; capping tiles at 2 vectors
  (11 live registers) did not help, so spills were not the limit.
- Cause 2, work outside the kernel (timers inside the op): weight packing 4.4 ms per call on 256ch layers (single-threaded,
  div/mod per element), phase planes up to 4.7 ms. Fix: pack once at model load; range-based phase copy.
- Cause 3, no input reuse across output-channel tiles: tasks now own a group of tiles sized so the group's weights fit
  ~128 KB of L2, reusing each L1-resident input block across the group. 256ch@50: 263 -> 397; 256ch@100 s2: 168 -> 264.
- Dead end: glibc mmap threshold (page-fault hypothesis for the 41 MB phase buffers) made no difference.
- Row alignment: rounding the phase-plane row stride to 8 floats: 256ch@50 397 -> 507, 64ch@200 373 -> 461,
  256ch@100 s2 264 -> 335. Unaligned 32-byte loads that straddle cache lines cost a lot; most taps are still unaligned
  (kx shifts by 1-2 floats).

## Run 19 - 2026-09-25 01:30

Question: pointwise and depthwise with the same machinery; thread count.

Findings:
- AVX2 depthwise (fused bias/act): 76 -> 21 ms/forward.
- Pointwise through the direct kernel reading the input in place (only the final partial vector of a plane goes through a
  `c x 8` scratch, so no full padded copy): vs candle GEMM + bias + relu, 336->64@200 14.6 -> 5.3 ms, 64->128@200
  11.8 -> 2.6 ms, 256->512@100 13.1 -> 5.7 ms, 512->192@50 1.35 -> 1.04 ms. An earlier version padded-copied the
  whole input when `h*w % 8 != 0` and lost on 50x50 maps (16 MB copies); replaced by the tail scratch.
- 16 vs 8 threads on the full model: 0.78 vs 0.63 s. Hyperthreads hurt the FMA-bound kernels. `PPDocLayoutV3Detector`
  now runs CPU work in its own physical-core rayon pool unless `RAYON_NUM_THREADS` is set (host's global pool untouched).
- Correction: re-measured ORT CPU with 2 warmups + 15 iterations: **449 ms mean / 437 ms min**. The 0.62 s from Run 17
  came from a 3-run sample including warm-up effects. The machine is a loaded desktop (load avg ~4, powersave governor);
  whole-model numbers vary by +-50 ms run to run, so compare mins over 15 iterations.
- Status: ours 0.65 s mean / 0.61 s min (default threads). Stages: backbone ~285 ms (was 547), encoder ~250 (was 415),
  decoder ~105 (was 120). ORT is still ~1.4x ahead.

## Run 20 - 2026-09-25 02:00

Question: does vectorizing over output channels (MLAS-style: broadcast input scalars, two aligned weight vectors per step,
no split loads) beat the pixel-vectorized kernel inside NCHW?

Change (reverted): 16 oc x 6 px tile (12 accumulators + 2 weights + 1 broadcast = 15 ymm), 64-byte-aligned packed weights,
no channel blocking (a tile's input, 256 ch x 3 rows x 8 px = 24 KB, fits L1), transposed store once per tile.

Finding: correct, but slower where it matters: 256ch@100 324 vs 504 GFLOP/s, 256ch@50 335 vs 469, 256->64 310 vs 473;
equal on small-channel layers. Without channel blocking every 6-pixel tile streams its full 147 KB of weights from L2,
which costs more than the split loads it removes. Making this formulation pay needs MLAS's NCHWc activation layout
(weights and inputs both blocked by 8 channels) end to end, not just inside one conv. **Dead end within NCHW.**

Final state of the CPU pass (default threads = physical-core pool; loaded desktop, 2 warmups + 15 iterations):
- Model forward: 0.67 s mean / 0.63 s min (Run 15 end: 0.97 s at 16 threads / 0.87 s at 8). ORT CPU: 0.45 s mean.
- Six pages end to end on CPU (decode + preprocess + batch-6 forward + post-process): 6.1 -> 4.5 s.
- Parity: six-page HF comparison identical on CPU (batched) and CUDA; CUDA forward unchanged at 26.0 ms.
- Remaining gap to ORT (~1.5x) is kernel efficiency: our best direct-conv shapes reach ~500 GFLOP/s (~40% of the ~1.2
  TFLOP/s AVX2 peak), MLAS NCHWc ~65-70%. Levers not pursued: NCHWc layout end to end (conv + add + cat + upsample
  custom ops), avoiding the `Tensor::cat` copies in HGNetV2 blocks by writing layer outputs into the concat buffer,
  and the stride-2 phase-plane build (8 ms on the stem conv).

## Run 21 - 2026-09-25 03:00

Question: after the PR #2 review fixes, do the fallback paths work? The review found that `mask_to_box` had no
Metal path, and that custom ops would fail when candle's CUDA was on but this crate's `cuda` feature was off.

Change: one `has_kernels(device)` gate (CPU, or CUDA with this crate's `cuda` feature) routes every custom op;
everything else takes tensor-op fallbacks (candle conv, shifted-tap depthwise, tensor-op sampler, tensor-op mask->box).

Command: build with `--features candle-core/cuda,candle-nn/cuda` (i.e. without `mistralrs-layout/cuda`, so every custom
kernel is disabled on the GPU) and run the parity example on CUDA.

Finding: all fallbacks correct end to end on the GPU: init_reference_points max_abs 4.8e-7, pred_boxes 1.3e-5, same 13
detections; 60 ms/forward (vs 26 ms with kernels). Before the fix this configuration errored.
Normal builds unchanged: six-page HF comparison identical on CPU (batched) and CUDA; CPU 0.65 s, CUDA 26.1 ms.
