# Calibration data and encoder refinements (#1013 items 4 and the encoder notes)

Item 4: our own calibration, chat-formatted and long-context, against the WikiText-plus-docs mix the recipes use now
(`calib_mix_4x`, ~260K tokens in 1024-token chunks; allocation investigation Runs 12-31). The encoder notes: choosing
levels against the block scale as stored, and re-fitting levels after a sub-block's scale is quantized.

## Run 1 - 2026-11-20 17:35

Question: do the encoder refinements lower KLD?

Where they already hold: the K-quants (ggml's reference code, which `k_quants.rs` ports byte for byte) pick every
level against the fp16-rounded `d` times the quantized 6-bit sub-scale (`block.d.to_f32() * sc`, then
`nearest_int(x / d)`), for Q2_K-Q6_K. Q8_0 has one scale and the nearest level, nothing to refit. So both notes
already describe what the K-quants do; the open case is IQ4_KS, the recipes' bulk (16.8 of 18.6 GiB on the 35B),
whose 7-bit block scales are odd multiples of an f32 row scale that is refit (`sum w x q / sum w q^2`) after the
indices are chosen, the indices never re-chosen against it.

Two prototypes in `quantize_iq4_ks_row`, Qwen3.5-2B, every block weight IQ4_KS (embedding Q6_K), weighed by the
Hessian diagonals (`--hessian-weights-only`, CPU encoder), ik `llama-perplexity` on 40 WikiText chunks (the 2B's
older base, floor 0.00065):

```
encoder                                                        KLD               same top
ik's (byte-exact)                                              0.05389 +- 0.00091  87.78%
+ each block's scale step searched l-4..l+4 x both codebooks   0.05385 +- 0.00080  87.98%
  (indices re-chosen for each, least weighted error kept)
+ 3 passes: indices re-chosen against the refit row scale,     0.05389 +- 0.00080  87.78%
  refit again, the pass with least weighted error kept
```

Neither moves KLD beyond noise (0.00004 and 0.00000 against +-0.0009). IQ4_KS's block scales are fine-grained
(127 steps) and its scale search already fits each block, so what rounding leaves is the 4-bit levels' own error,
which no choice of scale removes. Both reverted; the encoder notes close as negative, with ik's bytes kept.

## Run 2 - 2026-11-20 18:30

Question: does chat-formatted calibration (and a mix of it) beat the WikiText-plus-docs mix?

Chat data: first user turns from UltraChat 200k's test split (MIT) and MBPP tasks, shuffled (seed 0), 800 for
calibration and 200 held out; each answered by the 35B itself (the v2 recipe, served by our engine, temperature 0.6,
top-p 0.95, thinking on for 3 in 4, up to 2048 tokens), rendered through the model's own chat template (a reply cut
at the limit kept as generated: the prompt with its generation prompt, then the raw text). Calibration texts cut at
conversation boundaries to the baseline's 266K tokens: chat only, and half the mix + half chat.

Eval sets, all ik `llama-perplexity` against BF16 bases: WikiText A and B (c512, 40 chunks, as before), chat (the
200 held-out conversations, c1024, 40 chunks), long (WikiText test, c8192, 4 chunks). The chat base needs the
template's special tokens parsed (`<|im_start|>` is one token, 248045); ik tokenizes without, so the base was
written by a scratch copy of its perplexity tool with `parse_special` on (the base stores its tokens; scoring reads
them, so stock ik scores against it).

v2 recipe (alloc_v2_rules.txt), GPTQ, Hessians from each text (35B, 266K tokens, 1024-token chunks):

```
calibration                     wiki A    wiki B    chat      long c8192
mix (current recipes)           0.01399   0.01446   0.01112   0.00997
chat only                       0.01617   0.01596   0.00985   0.01093
half mix + half chat            0.01457   0.01507   0.00938   0.01052
```

(Single-file noise from route-flip reshuffles ~+-0.0004-0.0008, allocation Run 29.) Calibration follows the domain:
chat-only is 11% better on chat and 11-15% worse on WikiText; the half mix is 16% better on chat (better than chat
alone) at 4% worse on WikiText and 5% worse at long context. For a chat model in use, the half mix is the better
calibration.

## Run 3 - 2026-11-20 19:00

Question: does long-context calibration help long-context quality? The mix again, in 4096-token chunks instead of
1024 (a prototype `INFERENCE_CALIBRATION_CHUNK` override; same tokens).

```
calibration                     wiki A    wiki B    chat      long c8192
mix, 1024-token chunks          0.01399   0.01446   0.01112   0.00997
mix, 4096-token chunks          0.01459   0.01445   0.01118   0.00983
```

Every change within noise (long -1.4%, wiki A +4%, B and chat unchanged). The Hessians are means over rows; longer
chunks only add later positions' rows, which this model's inputs barely tell apart. No gain, so the override was
not kept.

## Run 4 - 2026-11-20 19:50

Question: why does collecting the 35B's Hessians take 575 s for 266K tokens (12 s a layer, ~550 tokens/s over the
model, against ik's 2560 tokens/s scoring forward)?

A layer's time scaled with tokens, not chunks (10.6 s a layer at 4096-token chunks, 12.4 s at 1024). The routed
experts' accumulation (`ImatrixLayerStats::process_routed`) read each forward's routing back to the host (a sync)
and then, for each of the 256 experts, uploaded an index list, gathered its rows and ran its Gram: ~1500 launches
and 512 uploads per chunk per layer. Now the forwards' rows are buffered (up to 128K routed rows) and flushed
together: one read-back, one upload, one gather in expert order, each expert's Gram over its run of it.

```
35B, 266K chat tokens                        total    a layer
per forward (before)                         578 s    12.4-11.3 s
buffered, flushed per 128K routed rows       361 s    9.2-8.2 s
```

Hessians equal the earlier ones to f32 summation order (relative difference 2e-7 to 7e-6 on sampled experts).

## Run 5 - 2026-11-20 20:45

Question: why is the GPU idle half of a Hessian collection after Run 4? Timed on a 64K-token slice of the chat text
(63 chunks), `nvidia-smi dmon` beside it (nsys hangs this process, with or without OS-runtime tracing; a 6 s window
started after loading worked):

```
step                                                        64K tokens   GPU busy   a layer
Run 4 (buffered routed Grams)                               134 s        35%        build 0.4 s, forward 1.5-2 s
BF16 GGUF weights uploaded as stored (no f32 round trip)    89 s         56%
8 chunks a layer call (GDN state and positions per batch)   53 s         61%        build 0.35 s, forward 0.65 s
```

- `perf` on the host: a third of its time converted each BF16 weight to f32 (`dequantize`) and back to BF16 (the
  layer's dtype), single-threaded, 1.6 GB a layer. A float GGUF tensor's bytes are its values, so they now go to the
  device as stored (`GgufWeightSource::materialize_binding`; value transforms such as `log` still run in f32).
  Hessians byte-identical (40 of 40 sampled files).
- Routing larger uploads through the pinned staging ring instead of the driver's pageable copy measured no change
  (91 s against 89 s): the driver time `perf` showed was waiting, not copying. Not kept.
- The kernel window: per 1024-token chunk ~1640 launches, ~1290 small host-to-device copies, ~640 `is_u32` masks and
  ~500 small GEMMs: the tracked experts run the gather path, which loops over all 256 experts every call (the CUTLASS
  grouped GEMM serves untracked BF16 experts only; forcing it changed nothing here). A forward of 8 chunks as one
  batch pays that once for 8; the streamed GDN layer's cache and the attention layers' positions and masks now take
  the batch size (`GdnLayerCache::with_batch`). Scoring passes stay one chunk a call (stacking moved their scores,
  allocation investigation Run 30).

Batched against one chunk a call: dense Hessians within 1e-5 (layer 0) to 5e-3 (the head after 40 layers); single
experts' down Hessians 1-36% apart (~2K routed rows each over 64K tokens, and near-tie routes flip with rounding).
The full chat collection (266K tokens) and the v2 recipe quantized from each:

```
chat Hessians                     collect   wiki A    wiki B    chat      long
one chunk a call (Run 2)          578 s     0.01617   0.01596   0.00985   0.01093
buffered, BF16 as stored, batched 133 s     0.01663   0.01571   0.00890   0.01114
```

Within the route-flip noise either way (+0.0005, -0.0003, -0.0010, +0.0002); 4.3x faster.
