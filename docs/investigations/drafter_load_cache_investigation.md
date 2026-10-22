# Drafter load cache

Goal: the external drafter (DFlash 2 for Qwen3.8 27B, a 3.8 GB BF16 safetensors checkpoint quantized to q4k at load
with `draft_isq`) is quantized again on every server start; keep the quantized weights so a start reuses them.

## Run 1 - 2026-10-22 01:42

Question: how much of a start is the drafter?

Command: the LAN configuration (`from-config`, recipe A GGUF + DFlash 2 `draft_isq` q4k, 262K q3 KV) started once
with temporary timers around `DFlashDraftModel::load` (not kept); weights in the page cache.

```
01:41:43.45  Loading model weights (target, 15 GB GGUF)
01:41:48.62  recurrent pool reserved                      target weights 5.2 s
01:41:49.26  attach_dflash start
01:41:56.76  drafter layers done                          7.5 s
01:41:57.56  drafter weights done (fc, selector, rope)    8.3 s
01:41:58.17  model loaded                                 total 14.7 s
```

The drafter's load is 8.3 s of a 14.7 s start, more than the 15 GB target's.

## Run 2 - 2026-10-22 01:44

Question: what inside the drafter's 7.5 s of projections costs the time?

Command: Run 1's start with temporary timers per projection (`linear_from_weight`) and inside `generate_isq`.

```
upload BF16 to the GPU      0.33 s total
quantize (apply_isq)        6.39 s total
per tensor, 178M elements:  GPU->host 40 ms, to f32 65 ms, Q4_K quantize 589 ms, upload 9 ms
```

Q4_K quantization itself is the cost: about 300M elements/s on the 16 threads `from_float` already spreads blocks over
(its search per block is the slow part), so the 1.9B-weight drafter needs about 6.3 s. Parallelism is not the fix; not
quantizing again is. The BF16 upload is a wasted round trip as well: `generate_isq` quantizes from a host copy, so
the drafter now hands it the host tensor and uploads only the packed weight.

## Run 3 - 2026-10-22 01:51

Question: with the quantized projections kept as one UQFF file (`IsqCache` in inference-quant: `serialize_uqff` on
write, `UqffReader::load_linear` on read, keyed by the checkpoint files' size and mtime, the weight type, dtype and
version; written in the background), what does a start cost?

Command: Run 1's start, twice, with the cache directory removed first.

```
cold (quantizes, writes 1.0 GB in the background)   16.5 s   (the write is done before the model reports loaded)
warm (reads the file)                                6.6 s   (was 14.7 s)
```

Greedy output of one chat request is identical between the cold and warm starts (180 tok/s decode, drafts accepted).
The tiny-model test `a_quantized_drafter_is_read_back_from_the_cache` fails when every load is made to miss.
