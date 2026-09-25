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
