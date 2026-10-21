# Qwen3.6 35B-A3B quant recipe for expert offload

Goal: a GGUF recipe for Qwen3.6-35B-A3B tuned for speed with `--moe-expert-cache` (experts in host memory, a VRAM slot
cache, CPU misses), in the way Recipe A was for Qwen3.8 27B: judged by KLD against BF16 logits and by engine speed,
never by bulk type alone.

## Run 1 - 2026-10-21 01:30

Question: what do the published sources say about this model's quantization sensitivity and about quant choices for
offloaded experts, before measuring anything?

The local file (Unsloth UD-IQ4_XS, 16.95 GiB) by tensor:

```
ffn_down_exps   IQ4_XS x37, Q6_K x3, Q4_K x1        5806 MiB
ffn_gate_exps   IQ3_S x39, IQ4_XS x1, Q3_K x1       4536 MiB
ffn_up_exps     IQ3_S x39, IQ4_XS x1, Q3_K x1       4536 MiB
everything else (Q8_0 attention / GDN / shared experts, Q8_0 embeddings, Q6_K head, F32 router)   about 2.5 GiB
```

So the experts are 14.9 of 16.95 GiB: they are what an offload recipe can trade.

Unsloth's Qwen3.5 GGUF benchmark page (same architecture, 35B-A3B; per-tensor sweeps, 121 configurations):

- `ssm_out` is very sensitive and saves almost nothing on disk; keep it high. Attention tensors are "especially
  sensitive" on the hybrid architecture. MXFP4 does much worse than Q4_K on `attn_gate`, `attn_q`, `ssm_beta/alpha`.
- `ffn_up_exps` / `ffn_gate_exps` are "generally ok to quantize to 3bit"; `ffn_down_exps` is slightly more sensitive.
  All three at about IQ3_XXS is called the best size compromise; 2-bit degrades clearly.
- imatrix helps everywhere, most at low bits. I-quants cost speed: IQ3_XXS tg128 85.7 vs Q4_K 90.4 (their GPU).
- Their Qwen3.5 35B-A3B table (mean KLD): IQ3_XXS 13.1 GB 0.0501, Q3_K_XL 16.1 GB 0.0308, Q4_K_M 18.5 GB 0.0192,
  Q4_K_XL 19.2 GB 0.0137, Q5_K_XL 23.2 GB 0.0069.

The Qwen3.6 repo ships `imatrix_unsloth.gguf_file` (192 MB) and BF16 GGUF shards (69.4 GB). No per-tensor recipe or
KLD numbers are on the Qwen3.6 card (their chart is an image).

Offload guides (Doctor-Shotgun's llama.cpp MoE guide, ik_llama.cpp README): routed experts to the CPU, everything else
on the GPU; prompt batches above a threshold copy experts to the GPU (llama.cpp 32 tokens, ik `32 * experts / active`),
which matches our split. ik warns against `-rtr` repacking with CPU experts; its CPU gains for IQ4_KS / IQ2_KS are
Zen4 prompt-side. None of them gives a measured quant-type choice for offloaded experts.

tqLLM's MoE (`--moe-host-mb N`) places the least-used experts in host RAM from a routing profile and copies routed host
experts to a VRAM stage for decode: the GPU computes everything, the CPU nothing. Its experts are k-quants (K4/K5/K6);
its MoE lane is unpromoted pending KLD and speed gates (`docs/MOE_STATUS.md`).

What this implies for an offload recipe here:

- Expert bytes set three costs at once: how many VRAM slots fit, the CPU's RAM reads for misses (decode is
  bandwidth-bound since #527) and PCIe staging for prompts. Fewer expert bits buy speed directly, and at some size the
  whole model fits with long context and no offload (resident decode 180 tok/s against 98 at 128 slots).
- Gate/up tolerate about 3 bits; down wants a little more; non-expert tensors stay Q8_0.
- The type also needs every engine path: GPU indexed matvec / grouped GEMM, CPU decode (`cpu_dot`), CPU prompt tiles.
  The CPU paths serve IQ3_S, IQ4_XS and Q2_K-Q6_K only; ik types (IQ3_KS, IQ4_KS, better per bit) would need CPU
  kernels before they can be offloaded.

Next: BF16 reference logits and the imatrix, then candidate expert mixes measured for KLD, size, and speed resident,
at 64 / 128 slots and CPU-only; tqLLM's `--moe-host-mb` on the same file for comparison.

## Run 2 - 2026-10-21 00:20

Question: the BF16 reference for KLD, saved once as in Recipe A.

Setup: Unsloth's BF16 shards (69.4 GB) and `imatrix_unsloth.gguf_file`, converted to the legacy `.dat` that ik's
`llama-quantize` reads with mainline `llama-imatrix --in-file ... --output-format dat` (ik's build has no such flag).

Command: `llama-perplexity` (ik_llama.cpp c446f1e0) on WikiText-2 test, `-c 512 --chunks 40 -ngl 99 -ot exps=CPU`
(experts in 61 GiB of pinned host memory), `--kl-divergence-base`; the LAN server stopped.

Result: PPL 5.8236 +/- 0.13865, 14.1 s per pass of 4 chunks; the logits file is 5.07 GB.

Next: KLD of Unsloth's UD-IQ4_XS, then the same recipe made locally (`quantize.sh`) to confirm the rules reproduce it
before changing the expert types.

## Run 3 - 2026-10-21 00:45

Question: does `quantize.sh` (Unsloth's exceptions, `--custom-q` on ik's `llama-quantize` with their imatrix)
reproduce UD-IQ4_XS, so later candidates differ from it only in expert types?

Command: `kld.sh` (the Run 2 settings, fully offloaded, `--kl-divergence`) on each file.

```
                      size        mean KLD             same top token    99% KLD   PPL(Q)/PPL(base)
Unsloth UD-IQ4_XS     16.96 GiB   0.03234 +/- 0.00093  92.28%            0.338     1.0192
ours UD-IQ4_XS        16.56 GiB   0.03275 +/- 0.00112  92.38%            0.361     1.0197
```

The same KLD within error, so the rules (and the f32 router names) apply. Ours is 16.56 GiB against 16.96: Unsloth's
BF16 shards hold 40 blocks (733 tensors) and no MTP layer, while their UD file has `block_count` 41 and
`nextn_predict_layers` 1 (753 tensors, blk.40 about 0.40 GiB). Sizes below are without MTP; the final file takes
blk.40 from Unsloth's file, since the engine runs Qwen3.5 MTP.
Against Qwen3.8 27B's IQ4_XS (0.0174) this file is noisier: the experts' 3.44-bit gate/up are most of its weight.

Next: a sweep over expert types (`sweep.sh`): each candidate is quantized, measured, and deleted (the disk has
114 GB free); finalists are made again for speed.

## Run 4 - 2026-10-21 01:40

Question: which expert types (gate/up, down) give the best KLD per GiB? Everything else stays as in Run 3.

Command: `sweep.sh` (quantize, `kld.sh`, delete) per candidate. Sizes without MTP.

```
gate/up   down      size        mean KLD             same top   99% KLD   PPL ratio
IQ2_KS    IQ3_KS    12.34 GiB   0.1076 +/- 0.0026    86.10%     1.114     1.0460
Q2_K      Q3_K      13.64 GiB   0.0840 +/- 0.0021    88.14%     0.881     1.0412
IQ3_XXS   IQ3_S     14.71 GiB   0.0484 +/- 0.0013    90.90%     0.480     1.0309
IQ3_S     IQ3_S     15.62 GiB   0.0419 +/- 0.0012    91.13%     0.414     1.0308
IQ3_KS    IQ4_KS    16.04 GiB   0.0316 +/- 0.0008    92.70%     0.318     1.0021
IQ3_S     IQ4_XS    16.56 GiB   0.0327 +/- 0.0011    92.38%     0.361     1.0197   (UD-IQ4_XS recipe)
Q3_K      Q4_K      16.85 GiB   0.0308 +/- 0.0008    92.47%     0.325     1.0147
IQ4_XS    IQ4_XS    18.54 GiB   0.0168 +/- 0.0006    94.43%     0.162     1.0072
```

Raw findings:

- Gate/up are where the error is. IQ3_S -> IQ4_XS on gate/up alone (+1.98 GiB) halves the KLD (0.0327 -> 0.0168);
  IQ4_XS -> IQ3_S on down alone (-0.94 GiB) adds about 0.009. Per GiB the two are close (0.008 vs 0.010), so
  "gate/up tolerate 3 bits" from the Qwen3.5 sweep holds only in the sense that 3-bit is usable, not that it is cheap.
- 2-bit gate/up is past the knee: IQ2_KS / Q2_K cost 2.5-3.3x the KLD of IQ3_S for 2-3 GiB.
- IQ3_KS / IQ4_KS beat the UD recipe at 0.5 GiB less, as with Qwen3.8 27B, but only by about the error bar.
- Q3_K / Q4_K matches the UD recipe at +0.3 GiB and is a k-quant the CPU path already serves.

Next: the 4-bit region (IQ4_KS, IQ4_KSS, Q4_K all through; IQ4_XS / IQ4_KS gate/up over 3-bit down; IQ3_KS all
through), then speed for the frontier in the engine.

## Run 5 - 2026-10-21 02:30

Question: the 4-bit region and the 3-bit ik types, against Run 4's frontier.

```
gate/up   down      size        mean KLD             same top   99% KLD   PPL ratio
IQ3_KS    IQ3_KS    14.78 GiB   0.0422 +/- 0.0011    91.66%     0.453     0.9995
IQ4_KS    IQ3_KS    17.38 GiB   0.0270 +/- 0.0008    92.97%     0.266     1.0005
IQ4_XS    IQ3_S     17.60 GiB   0.0259 +/- 0.0009    93.25%     0.259     1.0174
IQ4_KSS   IQ4_KSS   17.75 GiB   0.0216 +/- 0.0008    93.67%     0.226     1.0119
IQ4_KS    IQ4_KS    18.65 GiB   0.0167 +/- 0.0006    94.55%     0.175     1.0061
Q4_K      Q4_K      19.44 GiB   0.0159 +/- 0.0005    94.40%     0.163     1.0051
```

Raw findings:

- The frontier (size, KLD): IQ3_KS all 14.78 / 0.042, IQ3_KS+IQ4_KS 16.04 / 0.032, IQ4_KSS all 17.75 / 0.022,
  IQ4_XS all 18.54 / 0.017. Q4_K all buys 0.001 for 0.9 GiB more; IQ4_KS all equals IQ4_XS all.
- 4-bit gate/up over a 3-bit down (17.4-17.6 GiB) is worse than IQ4_KSS everywhere at the same size: lowering every
  tensor a little beats lowering down a lot.
- IQ3_KS all matches IQ3_S all (0.042) at 0.84 GiB less; IQ3_XXS gate/up (Run 4, 14.71) is worse at the same size.
- PPL ratios of the ik types sit near 1.000 while their KLD is no better: PPL is not a quality measure here.
- Among types the CPU path already serves, IQ4_XS all is on the frontier; the ik frontier points (IQ3_KS, IQ4_KSS)
  would need CPU kernels.

Bench: `inference bench` had no `--moe-expert-cache` (only `serve` did), so it was added to the bench options
(local branch `bench-moe-expert-cache`). CUDA build with `profile.dev.debug-assertions=false`, as the misses run on
the CPU.

Next: engine speed for UD-IQ4_XS, IQ4_XS all, IQ3_S all and Q3_K/Q4_K, resident / 128 / 64 slots / CPU-only.

## Run 6 - 2026-10-21 03:30

Question: how fast is each candidate in the engine, resident, offloaded and CPU-only?

Command: `speed.sh` = `inference bench --format gguf -f <file> --prompt-len 512 --gen-len 128 --iterations 5`, as is,
with `--moe-expert-cache 128` / `64`, and `--cpu` (`--gen-len 32`). CUDA dev build of this branch,
debug-assertions off. Bench prompts are synthetic token ids, so routing is not text-like; offloaded decode varies
+/- 10-16 tok/s between iterations.

```
                     KLD      size        resident          128 slots          64 slots          CPU only
                                          prompt  decode    prompt  decode     prompt  decode    prompt  decode
UD-IQ4_XS (Unsloth)  0.0323   16.96 GiB   4376    165.7     1115    104.0      798     77.8      70.8    9.3
IQ4_XS all           0.0168   18.54       4414    174.0     1030    104.6      723     71.3      78.4    10.0
IQ3_S all            0.0419   15.62       4172    158.1     1131    108.2      844     85.2      82.9    10.3
Q3_K / Q4_K          0.0308   16.85       4049    174.8     1106    103.4      796     81.1      40.6    10.0
```

(Sizes of ours are without MTP, 0.40 GiB less than with it.)

Raw findings:

- Resident, IQ3_S is the slow type on the GPU: the UD file (mostly IQ3_S gate/up) decodes 5% slower than IQ4_XS all
  with 1.6 GiB less to read, and IQ3_S all slower still.
- At 128 slots the four are within noise. At 64 slots more of the misses run on the CPU and bytes start to show:
  IQ3_S all 85 against IQ4_XS all 71 (both +/- 15).
- CPU-only decode is about 10 tok/s for every file: it is not bound by expert bytes alone (Run 14 of the offload doc:
  the GDN projections and LM head read as much per token).
- Q3_K / Q4_K prompts on the CPU run at half speed: the prompt tiles serve only IQ3_S and IQ4_XS.
- IQ4_XS all halves the UD file's KLD and is as fast or faster in every column but 64 slots.

`--mtp` on the UD file (it has the MTP block): resident decode 165.7 -> 185.4 (MTP accept 87%, 2.74 tokens a step on
these synthetic prompts), 128 slots 104.0 -> 139.0. A step verifies several tokens per expert fetch, so MTP is worth
more offloaded than resident.

The BF16 has no MTP block, so `graft_mtp.py` (gguf-py, beside the model files) copies blk.40 and its metadata
(`block_count` 41, `nextn_predict_layers` 1) from the UD file into IQ4_XS all: 753 tensors, 18.94 GiB.

Next: confirm the grafted file's KLD, then `--mtp` speed for it and for the UD file at 128 / 64 slots.

## Run 7 - 2026-10-21 04:00

Question: does the grafted file still measure as IQ4_XS all, and what does MTP give it?

KLD of the grafted file: 0.016848, the same as without the block (llama-perplexity does not run MTP).

`speed_mtp.sh` (Run 6 settings plus `--mtp`):

```
                        resident           128 slots          64 slots
                        prompt  decode     prompt  decode     prompt  decode
UD-IQ4_XS               4173    179.7      1085    140.3      766     104.7
IQ4_XS all + MTP block  4279    198.6      1004    150.8      703     104.5
```

(MTP decode varies +/- 18-39 tok/s between iterations, with acceptance.)

So the recipe for this card: IQ4_XS on every routed expert, Unsloth's other choices (Q8_0 non-expert tensors, Q6_K
head, F32 router, their blk.34/38/39 exceptions) and their MTP block, 18.94 GiB. It halves the UD file's KLD
(0.0168 against 0.0323), decodes 10% faster resident and 7% faster at 128 slots with MTP, and needs no new kernels.
The cost is 1.98 GiB: 128 slots of it take more VRAM than of the UD file, and its prompts copy more over PCIe
(1004 against 1085 tok/s at 128 slots).

Open: decode on real text rather than synthetic ids (routing locality sets the hit rate); IQ4_KSS all (17.75 GiB,
0.0216) would trade KLD for 0.8 GiB but needs CPU kernels; tqLLM's `--moe-host-mb` on the same file.

## Run 8 - 2026-10-21 04:35

Question: there is a DFlash 2 drafter for this model (incoai/Qwen3.6-35B-A3B-DFlash2, Apache-2.0, 1.05 GB BF16:
6 layers, block size 8, taps on target layers 1/6/11/16/22/27/32/37, the target's embedding and head). The card
reports accept length 4.3-5.3 against built-in MTP's 4.3-5.0 at 7 drafts. Does it beat MTP here, resident and
offloaded?

Command: Run 6 bench on the IQ4_XS + MTP-block file with `--mtp-model <drafter> --mtp-n-predict 7
--mtp-draft-isq q4k` (the Qwen3.8 27B serving setup).

```
                        resident           128 slots          64 slots
                        prompt  decode     prompt  decode     prompt  decode
built-in MTP            4279    198.6      1004    150.8      703     104.5
DFlash 2, 7 drafts      4361    249.3      1038    163.9      626     49.0
```

Raw findings:

- Resident, DFlash 2 gives +26% over MTP (accept 46.6%, 4.26 tokens a step on synthetic prompts). The engine
  loads it against the MoE target with no changes.
- Offloaded, the gain shrinks at 128 slots (+9%, +/- 75 between iterations) and turns into a loss at 64 (49 tok/s):
  an 8-token verify routes up to 64 distinct experts a layer, so a small cache thrashes and the misses' CPU work
  grows with the draft length.
- At 64 slots, 22 `MoE expert cache fill failed` errors (CUDA_ERROR_INVALID_VALUE, then NOT_PERMITTED) print after
  the bench finishes: fills still queued on the global `moe-expert-fills` thread run after the context is torn down.
  None at 128 slots. Log noise at exit, but it shows the fill queue backs up under speculation.

Implications: for this model on 24 GB, resident with DFlash 2 is the fast configuration (the 18.94 GiB file plus the
q4k drafter and KV). Offloaded, speculation wants fewer drafts as slots shrink (a planner input), or the misses of a
verify batch shared across its tokens.

Next: real-text decode (a served request) for resident DFlash 2 and for 128 slots, the memory left for context
resident, and draft count at 128 / 64 slots.

## Run 9 - 2026-10-21 05:30

Question: Runs 6-8 used synthetic ids. What does decode do on real text, per configuration, and how much context
does each leave?

Command: `mixbench.py` (scratchpad) against `inference serve -p 1240 --format gguf -f <IQ4_XS + MTP file> ...`,
one server per configuration (`mixrun.sh`): 4 wiki summaries (~2K-token prompts), 4 code tasks, 2 Chinese essays,
UI sampling (0.6 / 0.95 / 20, seed 7), 400 tokens, streamed, decode = (tokens - 1) / (last - first delta); median
per kind. Context is the engine's "available context length" at load (default 16 prefix caches, 4 sequences).

```
config                      wiki    code    zh      accept len   context
resident, none (BF16 KV)    170.8   172.5   172.4                85,664
resident, none, q4 KV       168.1   170.5   170.4                279,808
resident, MTP (2 drafts)    152.8   157.6   159.1   2.48         58,496
resident, MTP 1 draft       145.9   151.8   153.4   1.88         58,496
resident, DFlash 7          205.1   204.2   193.2   3.52         24,832
resident, DFlash 7, q4 KV   195.6   204.2   203.2   3.58         81,216
resident, DFlash 7, q3 KV   213.2   205.4   204.9   3.57         99,488
resident, DFlash 5, q4 KV   212.0   233.7   210.7   3.49         86,560
128 slots, none              94.9   102.8   101.7                502,752
128 slots, MTP              125.1   132.1   130.2   2.49         446,592
128 slots, DFlash 7         111.3   138.6   126.0   3.52         445,184
128 slots, DFlash 5         130.6   160.5   134.2   3.32
128 slots, DFlash 3         135.2   157.2   138.9   2.77         446,816
128 slots, DFlash 2         136.8   149.1   142.4   2.43
64 slots, none               56.6    60.3    61.7                715,744
64 slots, MTP                65.8    70.9    74.2   2.49         643,200
64 slots, MTP 1 draft        70.6    71.8    74.0   1.91
64 slots, DFlash 3           52.5    68.8    66.3   2.47         658,176
```

Raw findings:

- Synthetic ids overstated acceptance (MTP 2.74 there, 2.48 here; DFlash 4.26 there, 3.5 here) and MTP's gain.
- Resident, the built-in MTP head loses to no speculation (153 vs 171): on a MoE a k-token verify reads close to k
  tokens' worth of distinct experts, so a 1.5-2.5-token gain does not pay for it. DFlash 2 at 5 drafts wins (+24-37%)
  as its acceptance is higher; 7 drafts is no better than 5 (the 7 extra-long tails rarely land).
- Offloaded, the best draft count falls with the slots: 128 slots DFlash 2-5 (+35-55% over none, 5 best on code),
  64 slots MTP 1 draft (+15-25%); DFlash at 64 slots is no better than none. A planner choosing slots should choose
  the speculation too.
- KV q4 / q3 costs nothing measurable in speed here.
- No `cache fill failed` errors on any served run (Run 8's came at bench teardown).
- Offloaded 2K-token prompts take 2.4-4.3 s (first request cold) against 0.55 s resident.

Resident DFlash leaves only 486 MB of KV room (BF16 81K-token cache cut to 25K). The external-drafter reservation
(`reserve_external_mtp_memory_with_runtime`) counts the q4k drafter's weights (about 260 MB, the ISQ pack factor is
applied) plus its windowed KV: (live sequences + retained prefixes) x 2048-token window x 6 layers, about 50 MB each,
with retained prefixes = min(prefix caches, GDN snapshots) = 8. The GDN pool (capacity 9, about 63 MB a state)
reserves its snapshots either way.

```
resident, DFlash 5, q3 KV        wiki    code    zh      post-load reserve   context
16 prefix caches (default)       232.2   241.2   209.9   1,028 MB            106,048
--prefix-cache-n 4               231.2   237.9   208.1   830 MB              146,592
--prefix-cache-n 2               230.0   238.4   208.2   731 MB              166,880
--prefix-cache-n 2 --max-seqs 1  229.5   238.6   207.8   583 MB              197,184
```

Next: 262K resident wants about 330 MB more. IQ4_KSS all (17.75 GiB without MTP, KLD 0.0216) is 0.79 GiB smaller
and needs no CPU kernels when resident; measure its GPU decode and context.

## Run 10 - 2026-10-21 06:10

Question: does IQ4_KSS all (0.79 GiB smaller) reach the full 262K resident with DFlash, and at what speed?

`graft_mtp.py` now imports ik_llama.cpp's gguf-py (mainline's has no ik type ids; 146 = IQ4_KSS, file type 148).
The grafted IQ4_KSS file is 18.15 GiB, KLD 0.021642 as in Run 5. Same `mixrun.sh` / `mixbench.py`.

```
resident, q3 KV                    wiki    code    zh      accept len   context
IQ4_XS, none (q4 KV)               168.1   170.5   170.4                279,808
IQ4_KSS, none                      163.7   166.2   166.2                506,624
IQ4_XS, DFlash 5                   232.2   241.2   209.9   3.50         106,048
IQ4_KSS, DFlash 5                  211.9   231.8   188.7   3.58         269,888
IQ4_KSS, DFlash 5, p2, 1 seq       209.6   231.8   189.2   3.57         361,024
```

Raw findings:

- IQ4_KSS reaches the model's 262K at default serving settings (4 sequences, 16 prefix caches) with DFlash 5 and a
  q3 cache: the 0.79 GiB is 164K tokens of q3 cache (5.0 KB a token).
- It costs 2-3% decode without speculation but 4-10% with DFlash: the 6-row verify is where IQ4_KSS's MMVQ rows
  cost more than IQ4_XS's (`dp4a_max_rows` already gives the ik types 2 rows against IQ4_XS's 5, noting their rows
  cost 5.8x a k-quant's at 8 rows). The MoE matvec for a 6-token verify is the place to look.
- KLD 0.0216 against 0.0168.

So two resident recipes, both with Unsloth's non-expert choices and their MTP block, DFlash 2 at 5 drafts, q3 KV:

- IQ4_XS experts, 18.94 GiB: 232 / 241 / 210 tok/s, 106K context by default, 197K with `--prefix-cache-n 2
  --max-seqs 1`. Also the offload file (CPU paths serve it).
- IQ4_KSS experts, 18.15 GiB: 212 / 232 / 189 tok/s, 262K at default settings. Resident only until IQ4_KSS has CPU
  kernels.

Next candidates: the IQ4_KSS small-batch MoE matvec (would close the gap), the DFlash reservation (retained
drafter prefixes are about 50 MB each, the GDN snapshot pool about 570 MB at any prefix count), or a mixed file
(IQ4_KSS on some layers' experts) sized to reach 262K at default settings.

## Run 11 - 2026-10-21 08:30

Question: why are IQ4_KSS experts slower than IQ4_XS in a DFlash verify, when they read 6% fewer bytes?

Command: `nsys profile --cuda-graph-trace=node -t cuda` around `inference bench` (DFlash 5, q3 KV, 64-token prompt,
256 tokens, 2 iterations) on each file; `nsys stats --report cuda_gpu_kern_sum`.

```
MoE matvec, mean per launch          IQ4_XS (mmvq_gguf_iq4_xs_moe)   IQ4_KSS (mmvq_rows_moe_kernel<iqk_iq4_kss>)
gate/up (fused GLU)                  90.0 us                         106.2 us
down                                 61.8 us                         65.8 us
```

The two kernels share their shape (a block per routed pair, 2 rows a block, 4 warps); the difference is the block
dot. `vec_dot_iq4_kss_q8_1` and `vec_dot_iq4_ks_q8_1` pick the 32-entry `iq4k_values` half with each block's shift
bit, so threads of a warp read different table words; `vec_dot_iq4_k_q8_1` already used the uniform 16-entry table
plus a packed `+4` (the high half is the low half + 4 and no entry reaches 0xFC). The same for KS / KSS, with the
4-step loop unrolled:

```
IQ4_KSS                              before      after
gate/up                              106.2 us    87.9 us
down                                 65.8 us     60.7 us
```

Same trace (launch counts equal). `gguf::raw` tests on CUDA pass (indexed MMVQ, dequantization, kernels against the
dequantized weight, 15 of 15). Real text (Run 9 settings, q3 KV):

```
IQ4_KSS                    wiki    code    zh      accept len
none, before               163.7   166.2   166.2
none, after                168.6   171.2   171.3
DFlash 5, before           211.9   231.8   188.7   3.58
DFlash 5, after            227.6   245.8   199.3   3.34
IQ4_XS, DFlash 5 (Run 10)  232.2   241.2   209.9   3.50
```

IQ4_KSS no-speculation decode now equals IQ4_XS's; with DFlash this run accepted fewer tokens, and per verify step
(tok/s over accept length) it is 3% faster than IQ4_XS. Qwen3.8 27B recipe A (IQ4_KS, no speculation, `bench`
512 / 128): 49.5 -> 51.1 tok/s, though the old number is the LAN binary (debug assertions on).

So IQ4_KSS experts, 18.15 GiB with the MTP block, are the resident recipe: 262K context at default serving
settings with DFlash 5 and a q3 cache, at IQ4_XS's speed, KLD 0.0216. IQ4_XS (18.94 GiB, KLD 0.0168) stays the
offload file until IQ4_KSS has CPU kernels.

## Run 12 - 2026-10-21 10:30

Question: can the IQ4_KSS file run offloaded (CPU kernels for the misses) at IQ4_XS's speed?

`cpu_dot` gains IQ4_KSS (generic and AVX2 row dot, four-row dot, tile decode); `CpuExpertProjection::row_bytes`
and `decode_tile` count the f32 row scale that ik's row-scaled types put before each row. The shifted half of the
IQ4_K table is the IQ4_NL table plus 4, so one 16-entry `pshufb` serves both, as in Run 11. Tests: row dots against
the dequantized row (generic, AVX2, x4), tiles against row dots, and the routed / tiled expert forward against dense
experts with IQ4_KSS throughout; a shift of 0 in place of 4 fails three of the four.

`cpu_dot::bench` (one thread, debug assertions off), in steps:

```
IQ4_KSS                                          row dot      x4 dot (per token)   tile decode (8 x 2048)
scale code via movemask + scalar bit packing     1.52 GB/s    8.37 GB/s            2.21 us
scales in-lane, activations permuted             6.22         7.82                 2.74
weights permuted per sub-block pair              4.84         5.36
one sub-block a load (both lanes)                4.27         5.58                 2.12
block-wide scales (madd + hadd tree)             4.5          7.0                  3.1
  ... without the `[__m256i; 4]::map` closure    9.08         8.46                 2.2
  x4 weights permuted, written out               8.9          8.9
  tile decode written out (no array iterator)                                      1.05
IQ4_XS for reference                             5.0          13.5                 0.44 (8 x 512)
```

Raw findings: the closure passed to `[__m256i; 4]::map` inside a `#[target_feature]` function was not inlined (a
closure does not inherit the target features): `objdump` showed a call per block with the vectors through memory.
The step-two to step-four regressions were measured with that call in place, so they do not rank those layouts.
Iterating an array of vectors (`into_iter().enumerate()`) also kept them in memory.

Engine (`speed.sh`, synthetic ids, same session):

```
                     128 slots          64 slots           CPU only
                     prompt  decode     prompt  decode     prompt  decode
IQ4_KSS + MTP block  1034    110.0      737     76.5       78.9    9.4
IQ4_XS + MTP block   1002    98.2       708     66.1       77.3    9.1
```

Real text (Run 9 settings, IQ4_KSS): 128 slots DFlash 3 137.9 / 159.4 / 156.2, DFlash 5 131.7 / 156.3 / 125.8;
64 slots MTP one draft 71.4 / 73.1 / 77.3, none 55.0 / 62.4 / 64.5; no fill errors. Within noise of IQ4_XS's.

So the IQ4_KSS file (18.15 GiB, KLD 0.0216) now serves every configuration: resident 262K with DFlash 2, and
offloaded or CPU-only at least as fast as IQ4_XS.
