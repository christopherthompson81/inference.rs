# PaddleOCR-VL batching investigation

Reading many page regions at once with PaddleOCR-VL (the web UI's Documents tab) stops scaling at a few concurrent
requests (see ocr_batching_investigation.md, Runs 1 and 5). Where does a batched step's time go, and what keeps the
regions from batching?

## Run 1 - 2026-10-13 19:20

**Question:** with 16 region reads in flight, is the GPU or the engine's CPU the limit?

**Command:** `nsys launch` the server (PP-DocLayoutV3 and PaddleOCR-VL-1.5, 8192-token cache, `max_seqs: 16`), one
warm-up pass, then `nsys start` / `nsys stop` around `bench.py` reading the 25 regions of a synthetic dense page 16 at
a time (0.9 s, 27.5 regions/s); export to sqlite.

**Raw finding:**
- 45,370 kernels over 1.04 s; GPU busy (merged kernel intervals) 0.31 s, 30%.
- One thread made 165,785 CUDA API calls (346 ms inside them) in the window; the next busiest 4,209.
- Top kernels by time: `Kernel2` 95 ms (6,529 launches), `flash_attn_ext_f16` 37 ms, `gemv_kernel_batched` 22 ms; about
  60,000 `cuMemAllocAsync` / `cuMemFreeAsync` calls.
- `im2col_cols_last_f32` (the vision patch embedding) ran 38 times, about 0.3 ms apart within a step: one vision-tower
  pass per image.
- 1,953 device-to-host copies on the engine thread, 1,572 of them 12 bytes.

**Implication:** the GPU idles 70% of the time; the engine thread is the limit. Each device-to-host copy waits for the
GPU to drain, so 1,953 of them a second keep the GPU starved whatever the batch.

## Run 2 - 2026-10-13 19:35

**Question:** what makes the 12-byte copies, and what else serializes the regions? (A code read of the path from
admission to decode, plus gdb as the server's parent with a breakpoint on `cuMemcpyDtoHAsync_v2` when `$rdx == 12`,
recording the first 80 hits by source line.)

**Raw finding:**
- All 80 hits are `grid_rows` (`paddleocr_vl/inputs_processor.rs:95`), a `to_vec2` of the cached image grid
  (`cached_img_thw`, three u32 on the device): from `expand_image_prompt` at admission (`add_request.rs:914`) and at
  each step (`process_inputs`, lines 218 and 238, through `step.rs:209`). The grid is read back for every image row
  of every step.
- Other serial costs the code read found:
  - prefill batches only image prompts of the same image token count (`requires_uniform_prompt_batch`, and the
    3-chunk split around the unsplittable image span), so regions of different sizes prefill in separate steps;
  - the vision tower and connector run once per image (`paddleocr_vl/mod.rs`, a loop over rows and images; plain full
    attention, no packing), each building its rope tables on the CPU;
  - image preprocessing (CPU resize, upload, re-tokenizing the expanded prompt) runs on the engine thread at
    admission, one request at a time;
  - every forward reads `input_ids` back for the M-RoPE index and the embedding merge; decode has no CUDA graphs.

**Implication:** the per-step grid reads are the largest single count of syncs, and the grid never changes once
preprocessed, so it can stay on the host. Next: keep it on the host and measure again.

## Run 3 - 2026-10-13 20:10

**Change:** the image grid is built on the host (`inputs_processor.rs`, preprocessing), so `grid_rows` reads it without
a device copy.

**Raw finding (Run 1's method):** device-to-host copies in the window 1,953 to 386 (the 12-byte ones gone); 16 in
flight 27.5 to 36.0 regions/s, 1 in flight 6.0 to 7.5; GPU busy still 32%.

**Correction:** that benchmark reads the same 25 crops in every pass, and the model's encoder cache (keyed by image
content) answers every pass after the first, so its repeated-pass numbers skip the vision tower. A temporary probe in
the forward (CPU time of the whole forward, of everything before the text decoder, and of the vision tower and
connector) counted 25 images encoded over three passes. The "4.3x at 16 in flight" of
ocr_batching_investigation.md Run 1 is cached-vision throughput, not a real document's.

**Fair benchmark (scratchpad `vis/bench2.py`, `vis/fair.sh`):** 280 regions laid out from 30 pages (200 dpi) of a
real born-digital book; each level reads the same 100 regions on a freshly started server (empty encoder cache),
after one warm-up read of a region the runs don't use. 11,727 tokens of text in all (117 a region).

| in flight | time | tokens/s | regions/s | forwards | forward CPU | before the text decoder | vision tower |
|---|---|---|---|---|---|---|---|
| 1 | 61.4 s | 191 | 1.63 | 12,000 | 46.7 s | 27.2 s | 14.7 s |
| 4 | 39.1 s | 300 | 2.55 | 3,600 | 32.4 s | 25.9 s | 14.4 s |
| 16 | 36.2 s | 324 | 2.76 | 2,200 | 30.8 s | 26.0 s | 14.5 s |

(Probe counters are cumulative from the server's start, so they include the warm-up read; the vision column is about
148 ms an image whatever the concurrency.)

**Implication:** on real regions the vision tower is the largest cost, 14.5 s of the 30.8 s of forward at 16 in
flight, and doesn't amortize: one image a pass. The rest before the text decoder is 11.5 s, about 5 ms a step, which
scales with steps and prompt length (the M-RoPE index rebuilt over every row's whole prompt each step, decode included,
and the `input_ids` read-backs). The text decoder itself is about 4.8 s. Next: split the vision tower's 148 ms into GPU
compute and CPU work (its rope tables and position interpolation are built on the CPU per image).

## Run 4 - 2026-10-13 20:40

**Question:** what does the vision tower's 148 ms an image consist of?

**Raw finding:**
- A temporary probe with device syncs around each image's vision tower and connector, one region in flight over Run
  3's 100 regions: 24.4 s of vision GPU time for 97 images, rising faster than the patches:

  | patches | images | median | per patch |
  |---|---|---|---|
  | 500-1,500 | 53 | 34.6 ms | 54 us |
  | 1,500-3,000 | 20 | 252.6 ms | 108 us |
  | 3,000-6,000 | 24 | 837.6 ms | 178 us |

- nsys of one cold 1276x1353 region: 922 ms of kernels, `softmax_f32` 301 ms, `cast_bf16_f32` 203 ms, `Kernel2`
  (cuBLAS batched matmul) 140 ms, `cast_f32_bf16` 110 ms; no fattn kernel. The attention is materialized, in f32.
- The tower's head dim is 72 (1152 / 16). fattn has a tile kernel for it and parity-tests it ("head dims 40 and 72
  (SigLIP) run the tile kernel"); a throwaway test had `inference_fattn::supported` and `flash_attn` accept the
  tower's shapes in bf16 and f16 from 256 to 8,192 tokens. The engine never asked: `flash_backend_supports_sdpa`
  gates on `FATTN_HEAD_DIMS = [64, 80, 96, 112, 128, 256]`, the dims fattn takes in every layout (paged and packed
  too), which 72 isn't.

**Implication:** the quadratic, materialized attention is most of the tower's cost. Dense attention (unpaged,
unpacked, no softcap or window) at the tile-only dims 40 and 72 can ask fattn, which still refuses a shape it can't
take; the paged and packed gates stay as they are.

## Run 5 - 2026-10-13 21:00

**Change:** `flash_backend_supports_dense_sdpa`: the dense path in `Sdpa::run_attention` also tries fattn at head
dims 40 and 72 without softcap or a window.

**Raw finding, Run 3's benchmark** (the baseline rerun without either change: 36.1 s at 16 in flight, the same as with
only Run 3's grid change, which so helps the repeated-crop benchmark but not real regions):

| in flight | before | dense flash at 72 |
|---|---|---|
| 1 | 61.4 s | 47.7 s |
| 4 | 39.1 s | 27.3 s |
| 16 | 36.1 s | 24.3 s (1.49x) |

- Text: 100 of 100 regions identical to the baseline's at 16 in flight; 99 of 100 identical between 16 and 1 in
  flight with flash (the other 0.997 similar).
- The synced probe: 12.5 s of vision for the 97 images (from 24.4 s); medians 29.7 ms (500-1,500 patches,
  45.6 us a patch), 144.6 ms (1,500-3,000, 61.8 us), 381.1 ms (3,000-6,000, 81.2 us).

**Implication:** the vision tower's time halves, and the run is 1.5x faster at 16 in flight. Its cost still grows with
the square of the patches, as attention's work does.

## Run 6 - 2026-10-13 21:15

**Question:** does padding the head dim 72 to 80 with zeros, so fattn's tensor-core (mma) kernel takes it, beat the
tile kernel? (The pad adds nothing to q.k or the output, and the scale is explicit.)

**Raw finding:** 48.2 s at 1 in flight and 25.1 s at 16 (tile kernel: 47.7 s and 24.3 s); text 100 of 100 identical.

**Implication:** no gain at these sizes; reverted. What's left at 16 in flight is about half the vision tower (now
GPU-bound) and half the per-step work before the text decoder (about 5 ms a step: the M-RoPE index rebuilt over every
row's whole prompt each step, and the `input_ids` read-backs). Packing several images into one tower pass would
mainly help the small regions (the 53 under 1,500 patches take about 30 ms each, 1.6 s in all); the per-step work is
the larger lever.

## Run 7 - 2026-10-13 21:50

**Change:** each row's prompt tokens travel on the host (`prompt_ids`), and `window_positions` builds only the step's
`[3, batch, len]` M-RoPE positions there (the prompt's own positions inside the prompt, index + delta past it) and
uploads them once. Gone: building `input_ids_full` on the device row by row, reading it back every step, and
uploading three full position rows per row.

**Raw finding, Run 3's benchmark:** 47.3 s at 1 in flight, 26.9 s at 4, 23.7 s at 16 (Run 5: 47.7, 27.3, 24.3).
Text: 98 of 100 identical to the baseline at 16 in flight, 99 of 100 at 1 (the others 0.997 similar). During a 16 in
flight run, `nvidia-smi` every 200 ms: GPU 72% busy on average, under 50% in 26% of samples; the engine thread used
54% of a core.

**Implication:** about 2.5%, not the 11 s Run 6 attributed to the per-step work: the probe's "before the text decoder"
CPU time included waiting for GPU work already queued. With real regions the run is now mostly GPU-bound. What's left
is smaller: packing several small images into one vision pass, prefilling image prompts of different sizes in one
step, and CUDA graphs for decode.

## Run 8 - 2026-10-13 22:10

**Question:** the engine's 5 s throughput log during Run 7's 16-in-flight run showed 6 to 9 sequences running and 5
to 10 waiting, and a 20% encoder cache hit rate on regions never seen before. Is the paged cache (8,192 tokens, the
web UI's default context) the limit?

**Raw finding, Run 3's benchmark at 16 in flight, OCR loaded with three cache sizes:**

| `paged_context_len` | time | running / waiting (log) | encoder cache hits |
|---|---|---|---|
| 8,192 | 23.8 s | 6-8 / 8-10 | 20-24% |
| 32,768 | 21.4 s | 15-16 / 0 | 2% |
| 65,536 | 22.2 s | 15-16 / 0 | 2% |

5.3 GB of VRAM with both models loaded at 65,536.

**Implication:** a region's prompt is often 750 to 1,500 image tokens, so 8,192 tokens hold only a few at once: the
rest wait, and some are preempted and re-prefilled (the "hits" are their images encoded again). At 32,768 all 16 run,
10% faster; past that nothing more. The web UI's OCR slot now offers 32,768 by default.
