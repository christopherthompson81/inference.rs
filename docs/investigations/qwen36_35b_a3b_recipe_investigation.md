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
