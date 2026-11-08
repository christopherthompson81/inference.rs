# Qwen GGUFs on Vulkan

Goal (user): Qwen models run on the Vulkan backend (3090 here; the 780M iGPU is the other target). Local GGUFs:
Qwen3.5-0.8B Q8_0 (small check), Qwen3.6-35B-A3B UD-IQ4_XS (the MoE the CUDA recipes use).

## Run 1 - 2026-11-08 04:20

Question: what stops them?

- RoPE: `inference_quant::rotary` dispatches CUDA and Metal and otherwise the CPU kernel, which refuses device
  tensors ("RoPE positions must be CPU u32", "unsupported CPU RoPE dtype F32 F32 F32"). Added a composite in tensor
  ops (gather the cos/sin rows, two multiply-adds over the halves or interleaved pairs, the rest passed through) that
  Vulkan takes; CPU tests pin it to the CPU kernel for both layouts, partial rotation, positions and every cache row form.
- Next: "paged decode of 8 heads over 2 kv heads, head size 256" (0.8B) and "16 heads over 2 kv heads, head size 256"
  (35B). Every Vulkan attention kernel stops at head size 128 (paged decode and prefill `MAX_HEAD4 = 32`, the
  matrix-core prefill and flash attention 128).

`--paged-attn off` (tensor-op attention, any head size), `inference bench --vulkan --prompt-len 512 --gen-len 32`:

```
Qwen3.5-0.8B Q8_0: prefill 681 tok/s, decode 20.9 tok/s
```

Greedy chat against CUDA: the first five tokens agree (logprobs within 0.03); token six is a near tie that flips
(" sunlight" -1.31 on CUDA, " Ray" -1.37 here), so the text then differs.

Next: a head-256 paged decode. Its shared memory at MAX_HEAD4 64 is ~49 KB with the 16-head group limit, over the
3090's 48 KB; head 256 allows at most 8 heads a kv head anyway (8 value heads x 256 / 256 threads), so a variant
with MAX_GROUP 8 is 47.2 KB.

## Run 2 - 2026-11-08 04:35

A wide paged decode: `paged_decode_main.glsl` takes `DECODE_MAX_HEAD4` 64 and `DECODE_MAX_GROUP` 8 from
`paged_decode_h256{,_f16,_quant}.comp`; the op allows head size 256 there (128 for prefill, unchanged).
`vulkan_paged_attention_matches_the_cpu` runs heads 8 and 16 over 2 kv heads at 256 too.

`inference bench --vulkan --prompt-len 512,2048 --gen-len 64` (paged, the default):

```
Qwen3.5-0.8B Q8_0:          prefill 704 / 765 tok/s, decode 29.3 tok/s
Qwen3.6-35B-A3B UD-IQ4_XS:  prefill 65 / 84 tok/s,  decode 7.5 tok/s
```

Greedy chats now give CUDA's text (both models; 35B first-token logprob -0.23 vs CUDA -0.10, the rest within ~0.05).

Profile (`INFERENCE_VULKAN_PROFILE=1`, 5 s windows of decode): GPU busy 456 ms (35B) and 501 ms (0.8B) of 5,000.
The 0.8B does ~640 `map_f32` dispatches a token: GDN's recurrence and causal conv have CUDA and Metal kernels
and otherwise run as tensor-op composites (`gdn/backend.rs`), one step at a time, so the host's dispatch
recording bounds decode and prompts loop per token. The 35B's profile has no expert matvecs: its IQ4_XS
experts have no Vulkan kernel and run on the CPU.

Next: GDN on Vulkan (the recurrence and causal conv as kernels), then IQ4_XS expert kernels for the 35B.

## Run 3 - 2026-11-08 05:10

GDN on Vulkan:
- `gdn_recurrence.comp`: the CPU pass's recurrence from the convolved rows, a workgroup a (value head, sequence),
  a thread a value column holding S[:, j] in registers (head_k and head_v up to 128), reading the checked-out
  key-major state and writing the advanced one (the cache commits it back as before).
- The causal conv is Mamba's conv step (`ssm_conv_step`: the state the last `kernel` inputs, no bias, SiLU), in
  place on the checked-out rows, for a prompt as for a step.
- RoPE as a kernel (`rope.comp`, a thread a rotated pair), the composite kept for other dtypes.
Parity tests (each fails when its kernel is broken on purpose): `vulkan_gdn_recurrence_matches_the_cpu` (Qwen3.6's
16/32 heads at 128, both layouts, a step and a 37-token prompt over two sequences), `vulkan_gdn_conv_matches_the_cpu`,
`vulkan_rope_matches_the_cpu_kernel`.

```
Qwen3.5-0.8B Q8_0, prefill (2048) / decode:
  composite RoPE + GDN       765 / 29.3 tok/s
  + GDN recurrence kernel   1189 / 68.0
  + conv step + RoPE kernel 2186 / 89.6
Qwen3.6-35B-A3B UD-IQ4_XS:   97 / 14.7 (from 84 / 7.5)
```

Greedy text matches CUDA's (35B's sixth token is a near tie that flips: " gas" -1.19 here, " molecules" -1.18 on CUDA).

The 35B's experts (gguf-py over the file): IQ3_S 78 tensors 8.38 GiB, IQ4_XS 39 tensors 5.18 GiB, Q6_K 3, Q3_K 2,
Q4_K 1; attention, GDN and shared experts Q8_0. Neither i-quant has a Vulkan kernel, so the experts compute
through the fallback. Next (user): IQ4_XS and IQ3_S kernels.

## Run 4 - 2026-11-08 06:00

The i-quant kernels (user: "make the needed kernels"):
- `iq4_xs_int8.glsl` and `iq3_s_int8.glsl` decode a 32-value sub-block to int8 lanes and a scale (no offset), the
  contract `q4_k_int8.glsl` keeps; the eight families (matvec, scalar, matrix-core and large-tile prompt matmuls,
  the MoE matvec and both MoE matmuls, embedding) take them through their `.comp` wrappers (the Q4_K ones with the
  include swapped). IQ3_S's grid is ggml's `iq3s_grid` as a 512-entry constant (`iq3s_grid.glsl`, generated from
  ggml-common.h).
- Gated MoE experts on Vulkan ran on the CPU's kernels for every type ("until it has its own"). A Vulkan backend
  now takes SiLU-gated GGUF experts of a Vulkan block type, uploaded once from their mapping: gate and up through
  the expert matvec, SiLU(gate) * up, down, the router's weights (`inference_tensor::vulkan::BlockWeight`).
- Raw GGUF weights (the i-quants) gain a Vulkan storage (`BlockWeight`) for dense matmuls and embedding rows.
- Paged prefill gains the head-256 build (four threads a (row, head), 4 rows a workgroup, tiles staged as f16), so a
  prompt chunk behind a cached prefix runs (the 27B's second chunk failed: "paged prefill of 25 rows, 24 heads over
  4 kv heads, head size 256").
Tests (each fails with its kernel broken on purpose): `vulkan_iq_block_weights_match_their_dequantized_rows` (both
types, expert decode and prompt routes, dense step and prompt, against ggml's reference dequantization),
`vulkan_raw_gguf_iq_weights_match_the_cpu`, `vulkan_paged_prefill_matches_each_row_scored_alone` at head 256.

Dead end: `const int IQ4_NL[16]` indexed per value put the table in local memory on NVIDIA: a 17408 x 5120 IQ4_XS
matvec took 497 us (95 GB/s; Q5_K streams ~540 GB/s). The 16 values as four uint words with selects: 194 us
(245 GB/s), still 2.4x Q5_K's time a byte.

```
inference bench --vulkan --prompt-len 512,2048 --gen-len 64, 3090:
Qwen3.6-35B-A3B UD-IQ4_XS: prefill 1445 / 1678 tok/s, decode 38.1   (Run 3: 86 / 97, 14.7)
Qwen3.8-27B IQ4_XS:        prefill  174 /  169 tok/s, decode 12.9   (did not load: "Iq4Xs weights are not supported")
```

Greedy text: the 35B gives CUDA's text word for word; the 27B's eighth token is a near tie (" scatters off
molecules" vs CUDA's "off the molecules").

The 27B's prefill: the GPU is busy ~1.6 s of each 5 s window, its matmuls at 25-35 TFLOPS (a 1632-token chunk's work
is ~2.9 s at 30 TFLOPS, so ~560 tok/s is the ceiling). perf sees our threads blocked and the NVIDIA driver's
`[vkrt] Analysis` thread spinning; `strace -c` counts 27K ioctls (82 us each) over the run. Not resolved here.
Decode at 12.9 tok/s is the IQ4_XS matvec's 245 GB/s.

## Run 5 - 2026-11-08 07:30

Question: why does the 27B's prompt leave the GPU idle? (User: the open gaps, in order.)

- Our per-dispatch profile summed ~1.7 s of GPU work per 5 s; `nvidia-smi dmon` showed SM 49-89% at ~230 W, and nsys
  (`--trace=vulkan`) showed the queue busy throughout, with 5.7 s of a 14 s prompt in `vkWaitForFences`. Attributing
  nsys batch time to our labels failed: 1,532 of the batches were empty submits, their spans queue time.
- Temporary counters (fence waits, allocations, staging reads and writes): each 2 s window read ~6.7 GB back to the
  host and wrote ~3.4 GB. A backtrace on reads over 16 MB: `apply_op2_no_bwd<FusedGlu>`. `inference_quant::fused_glu`
  is a custom op with CUDA and Metal kernels and none on Vulkan, so every dense MLP (and shared expert) read its gate
  and up activations to the host, multiplied there and uploaded the product, on prompts and on decode steps.
  Fixed: on Vulkan the activation and the product run as the device's elementwise kernels
  (`vulkan_fused_glu_matches_the_cpu`, every activation).
- Also found: `index_select` of an f16 tensor (the attention layers' KV cache, 69.7M values, 32 a prompt) fell back to
  the CPU, `select_b32` taking 32-bit types only. 16-bit rows of an even length now select as words
  (`vulkan_half_index_select_matches_the_cpu`).

```
inference bench --vulkan --prompt-len 512,2048 --gen-len 64, 3090 (prefill 2048 / decode):
Qwen3.5-0.8B Q8_0:          2186 / 90.0  ->  9632 / 191.0 tok/s
Qwen3.6-35B-A3B UD-IQ4_XS:  1678 / 38.1  ->  2038 /  55.7
Qwen3.8-27B IQ4_XS:          169 / 12.9  ->   476 /  15.7
```

Greedy text: the 27B now gives CUDA's text word for word; the 0.8B's sixth token is a near tie either way (CUDA
" sunlight" -1.31 vs -1.43, here " Ray" -1.34 vs -1.41).

Left: buffer pool churn. With the KV cache sized to the card, free memory sits at the 512 MB margin and `make_room`
evicts pooled buffers on each miss: ~800 evictions and ~0.5 s of `vkAllocateMemory` per 2 s of prompt (counters,
removed). A 4 GB activation reserve (KV 4256 -> 2208 MB) cut evictions to ~60 a window but prefill moved only
375 -> 460 / 294 tok/s (noisy), and the pool grows into whatever is free. Not changed: it trades KV capacity.

## Run 6 - 2026-11-08 08:10

Question: can the IQ4_XS matvec stream like the K-quants? (Run 4 left it at 245 GB/s, its table lookup four selects
and a shift a value, about ten instructions against the int8 dot's eight a block.)

A `TYPE_SHARED_INIT` hook at the start of every family's main fills a type's shared tables before any thread can
leave. IQ4_XS keeps a 256-entry table indexed by a whole byte, both nibbles' int8 values in one read (4 reads a word
instead of 8 select chains); IQ3_S copies `iq3s_grid` to shared memory, a constant array indexed per lane serializing
on NVIDIA.

```
mat_vecs_iq4_xs m=17408 k=5120: 194 -> 93 us (508 GB/s; Q5_K m=10240 k=5120: 58 us, 570 GB/s)
Qwen3.8-27B IQ4_XS:        prefill 476 / decode 15.7 -> 463 (+-59) / 25.7 tok/s
Qwen3.6-35B-A3B UD-IQ4_XS: 2038 / 55.7 -> 2508 / 61.7
```

## Run 7 - 2026-11-08 09:00

Question: which small dispatches bound the 0.8B's decode? `map_f32` dispatches now carry their op and size in the
profile (`Add n=1024`). Per token (~600 dispatches): rms_norm_q8 61, quantize_q8_1 48, the MLP's gate and up matvecs
48, residual `Add` 48, `Silu` + `Mul` 24 each (the GLU, two ops since Run 5), a GDN state `Copy` of 262,144 values
18.6, and Gated DeltaNet's output norm as ~7 elementwise ops a GDN layer (Sqr, mean, Sqrt, Div, weight Mul, Silu,
Mul).

Changes:
- `gated_rmsnorm` (Mamba-2's: norm of x * silu(g)) takes a mode for GDN's order (norm of x, weight, then silu or
  sigmoid of the gate), its int8-writing build too; GDN's `RmsNormGated` runs it on Vulkan with a group as wide as
  the head, the weight shared across heads (`vulkan_gated_gdn_norm_matches_the_cpu`, both gates, a decode row and
  a prompt's).
- GLU map ops (`SiluMul`, `GeluMul`, `ReluMul`, `GeluErfMul`, `SigmoidMul`: act(a) * b in one dispatch) for
  `fused_glu` and the Vulkan MoE experts.
- The GDN recurrence advances the checkout's own pool-row view in place (the CPU path's condition), so commit skips
  the copy (`vulkan_gdn_recurrence_advances_a_pool_row_in_place`: the row advanced, the others untouched).
Dead end: `inference bench`'s decode (64 tokens at depth 4) swings 180-196 tok/s run to run; a 3,000-token greedy
stream is steadier.

```
3,000-token stream, mean decode after the first 50 tokens:
Qwen3.5-0.8B Q8_0: 146 (Run 5) -> 172 (fused GDN norm) -> 181 tok/s (GLU op, in-place state)
Qwen3.6-35B-A3B UD-IQ4_XS: 73 tok/s
```

Greedy text: both give CUDA's text word for word. Left: residual adds into the next norm (48 a token) and the
gate and up matvecs as one; model code, ~8% each.

## Run 8 - 2026-11-08 09:40

Q3_K on Vulkan: `q3_k_int8.glsl` (ggml's dequantize_row_q3_K: high bit from hmask, low pairs, a 6-bit scale per 16,
the -4 as Q6_K carries its -32, through the activation sums), the eight family wrappers, `BlockType::Q3K` (110 bytes),
and the dtype maps (`GgmlDType::Q3K` for QTensors, `GgufType::Q3K` for raw blocks and the expert stacks). Tests: Q3_K
joins `vulkan_quantized_matmuls_match_the_dequantized_weight` (QTensor) and the block-weight test (renamed
`vulkan_gguf_block_weights_match_their_dequantized_rows`, its reference Candle's dequantization for the k-quant);
an off-by-one offset fails both.

The 35B's next-token-prediction layer (blk.40: Q3_K gate and up, Q4_K down) now runs on the device:

```
Qwen3.6-35B-A3B UD-IQ4_XS, 3,000-token stream:  plain 72.5 tok/s, --mtp (n_predict 2) 88.9 tok/s
```

Greedy text with --mtp: CUDA's, word for word.

## Run 9 - 2026-11-08 09:30

Question: does one matvec for the MLP's gate and up (ending in their GLU) cut decode time? (User: do the open items:
gate+up pairing, add+norm fusion, pool churn.)

`mul_mat_vecs_glu_<type>` (mat_vec_main with `GLU` defined): a second weight binding, every block decoded from both
weights and dotted with the same int8 activation block, `act(g) * u` written by thread 0. The type files' weight
reads go through a `W16()` accessor that the GLU build points at either buffer by a selector set to a constant before
each unpack (folded once inlined). `QuantMethod::vulkan_block_weight` hands out a dense weight's device blocks (QTensor
and raw i-quant GGUF layers, forwarded by the pending and LoRA wrappers); `quantized_ffn` tries it for decode rows on
Vulkan. Test: `vulkan_glu_matvec_matches_its_projections` (Candle's types and the raw i-quants, every activation, one
and three rows, against the separate matvecs within 1e-5 of the output's max; nine rows give none). Pointing the up
read at the gate's buffer fails it (Q8_0: 4187 of 988).
Dead end: comparing the GLU of CPU products to the device's within 2e-2 fails for sigmoid and relu (3-7%): the int8
activations' error in the gate turns into one of up's size, so the check is against the device's own matvecs.

```
INFERENCE_VULKAN_PROFILE, 0.8B decode per 5 s: GPU 1521 -> 1348 ms
  before: mat_vecs m=3584 8.2 us x2 + Silu 2.3 + Mul 1.9 a layer; after: glu_mat_vecs_q8_0 14.2 us
3,000-token stream (stream_rate.sh):
Qwen3.5-0.8B Q8_0:          181 -> 183.6, 173.7 tok/s (host-bound: GPU busy ~27% of the window)
Qwen3.6-35B-A3B UD-IQ4_XS:  72.5 -> 75.1 tok/s
Qwen3.8-27B IQ4_XS:         30.3 tok/s (no earlier stream; Run 6's 25.7 was `inference bench`)
```

Greedy text (greedy.sh): the 0.8B's sixth token is now CUDA's " sunlight" (-1.318).

## Run 10 - 2026-11-08 09:50

Question: does folding the residual add into the next norm cut decode dispatches? The Qwen3.5 layers already add
and norm through `forward_add_rms_norm` (CUDA fuses it); on Vulkan it was `Add n=1024` (48 a token on the 0.8B)
then `rms_norm_q8`.

`rms_norm_q8` takes an add mode (the residual bound beside x, the sum written as the row loads); `AddRmsNorm` records
it for a decode's rows (at most 8, whole 32-value blocks, up to 16,384 wide), and `rms_norm_forward_add` takes it on
Vulkan before the add-then-norm fallback (`vulkan_add_rms_norm_matches_the_cpu`: one, three and nine rows, the sum
exact and the norm within 1e-5 of the CPU's; a subtracted residual fails it).

```
profile: Add n=1024 23,104 a 5 s window -> gone (48 left a window, the prompt's)
3,000-token stream, five 0.8B runs: 172.5, 192.7, 190.5, 198.7, 197.5 (median 192.7; Run 7 181)
Qwen3.6-35B-A3B UD-IQ4_XS: 74.4   Qwen3.8-27B IQ4_XS: 30.8
```

The 0.8B swings ~20 tok/s run to run (clocks 1965 MHz, no thermal throttle bits); the gain is inside a run or two's
noise. Seen in passing: `rows_f32` takes ~250 us a token on the 0.8B (the 248K-wide logits' row op), ~5% of a step.

## Run 11 - 2026-11-08 10:00

Regression found while benchmarking the pool (from Run 7, merged in #865): a 2048-token prompt on the 27B fails with
`gated RMSNorm of 98304 rows of 1 groups of 128`. GDN's norm runs a row a head a token (2048 x 48), and the kernel
took rows down one grid dimension (at most 65,535); Run 7 only ran chat-length prompts. Rows now go down y and on
through z (`rows` in the push constants, the last slice's overhang idle); `vulkan_gated_gdn_norm_matches_the_cpu`
adds 4,200 tokens of 16 heads (67,200 rows).

## Run 12 - 2026-11-08 10:30

Question: what does the pool's churn cost with the KV cache sized to the card, and what helps without trading KV?
Temporary counters (misses, evictions, `vkAllocateMemory` time, missed and idle classes; removed) on the 27B's
2048-token prompt (`inference bench --prompt-len 2048`, KV 4846 MB):

```
a 2 s window: ~330 misses, ~330 evictions (~20 GB), alloc ~0.3 s; free at the 512 MB margin, 0.5-1.1 GB idle
missed classes (MiB): 160, 192, 96, 80, 48, 40 (the prompt's activations); idle: other classes
```

Change: a pooled device allocation short of room first waits for the oldest submitted batch (its retained buffers
return to the pool) and retries the pool, before `make_room` frees idle buffers. Evictions ~150 a window.
Dead end: the 4 GB reserve again (KV 4846 -> 2798 MB): evictions ~30 a window, alloc ~0.05 s, TTFT 496 / 572 tok/s
(+-81 / 44) against 465-505: at most ~10% for 42% of the KV cache. Not changed.

```
alternating pairs (TMP_NOWAIT toggle, removed), prefill 2048 / decode tok/s:
Qwen3.8-27B:  wait 497.9, 495.9, 492.2 (+-27-31)  vs  505.9, 504.8, 497.8 (+-81-89); TTFT 4150 vs 4207 ms
Qwen3.6-35B:  wait 2966.8, 2971.3  vs  2840.4, 2860.9
Qwen3.5-0.8B: wait 9905.7, 9908.0  vs  9909.8, 9918.8
```

The 27B is level (its spread narrower), the 35B's prefill ~4% faster; kept. The 27B's prompt stays GPU-bound with
the churn it has left (its allocations overlap the GPU's work, the CPU recording ahead).

Checks: lint, CPU (3149) and CUDA (3352) suites pass; the Vulkan set (655) passes, with a flake under the full
parallel run: one of `a_vulkan_weight_holds_its_blocks_once` or `vulkan_quantized_cache_scores_as_its_dequantized_rows`
(q4/q4 decode, max diff 0.0013) fails ~1 run in 10, each taking ~3.5 s against ~0.3 s alone. Master's code fails the
same way (1 of 15 runs, the storage test), and this branch with the pool wait off does too (1 of 15): not from here.
