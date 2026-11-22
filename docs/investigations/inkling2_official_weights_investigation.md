# Inkling 2 Mini from its own checkpoint

thinkingmachines' `Inkling-2-Mini` (about 20 GB of safetensors under `model.llm.*`, `hf_quant_config.json` naming each
tensor's precision) on the 3090, beside ggml-org's IQ4_NL GGUF that the CUDA work so far ran (llama.cpp's converter
dequantizes the released experts to f32 and requantizes them, so the GGUF is a requant of these weights).

## Run 1 - 2026-11-21 12:00

The checkpoint, read from the safetensors headers:

- Dense weights BF16 (3.0 GiB, the 201,024-row embedding and head 0.86 GiB each), the routers' and loop controllers'
  too; the depth-mix and message queries F32. The vision tower 0.6 GiB (not loaded: text only).
- The routed experts (14.6 GiB): layers 0-19 NVFP4 (`w13_weight` U8 `[128, 2304, 1152]`, E4M3 block scales a 16
  values, an F32 `amax` per expert over a shared `outer_scale_denominator`), layers 20-23 MXFP8 (E4M3 values, an E8M0
  scale a 32). `w13` interleaves the gate's rows (even) and the up's (odd).
- vLLM's reference (`vllm/models/inkling2/nvidia`) quantizes the expert activations per token (NVFP4 with the
  "fouroversix_mae" scale choice, MXFP8), sm90+ only. The 3090 has no FP4/FP8 tensor cores; here the experts run their
  stored weights against BF16 activations (NVFP4 through the GGUF NVFP4 kernels' 8-bit inputs).

Loading:

- `inkling2_official::bind` rebinds the var builder to the model's names (renames; the conv weights squeezed, the
  global layers' `static_w` reversed and transposed; the depth-mix queries as rows, block i's feeding layer i + 1; the
  looped layers' message queries from the cross-pass query rows) and attaches an expert weight source: NVFP4 as GGUF
  NVFP4 blocks (`modelopt_to_gguf_blocks`, the same codes and scale bytes) with each expert's second-level scale kept
  apart (`GgufRawMatMul::with_expert_scales`), MXFP8 as blockwise FP8 with 1x32 power-of-two scales.
- The fast MoE paths apply an expert's scale the raw kernels leave out: the down's folded into the routing weights,
  the gate's and up's on their rows before the softcapped activation (indexed matvec and grouped mmq both).
- Sizing: the inventory counted the 48 cache slots as layers (24 in the checkpoint) and FP8 at BF16's bytes, so device
  mapping fell back to a BF16-expert estimate and the paged cache got 0 blocks. The loader now reports its weight
  layers and the experts' bytes as loaded (`checkpoint_tensor_bytes`).
- Detection: `Inkling2LoopForConditionalGeneration` has no multimodal loader; a class without one now runs its
  `text_config`'s text loader. The config's ModelOpt `quantization_config` is dropped for these checkpoints (the
  experts' precisions are `hf_quant_config.json`'s).

Greedy on the printing-press prompt: "the 15th century, when Johannes Gutenberg invented a new way to print books..."
(the GGUF with grouped projections takes the same branch at its near tie).

```
inference bench (CUDA, paged + graphs)     official    IQ4_NL GGUF    llama.cpp (GGUF)
tg128                                      65.9        84.9           61.1
pp512                                      741         1926           1486
```

Decode reads BF16 dense weights where the GGUF reads Q8_0, and the MXFP8 layers run blockwise FP8's warp-per-output
kernel; prefill is the MXFP8 layers' that kernel (no grouped path).

Tests (`inkling2_tiny`): the tiny model at widths of 64 in the official layout, NVFP4 experts in layers 0-3 and MXFP8
in 4-5 drawn as random codes; on the CPU it scores as its decoded weights in the model's own layout (1e-4, f32 KV;
with the CPU's f16 KV the gap was 8.5e-4). On CUDA the NVFP4 kernels' 8-bit inputs move the tiny model's logprobs by
0.08-0.17 against f32 experts, so CUDA scores the checkpoint against itself with each expert's scale (a power of two)
folded into its block scales instead: prompt (the grouped kernels) and decode steps (the indexed matvec) within 1e-3;
with the gate's and up's scales dropped it fails. The blockwise FP8 fallback (f32 inputs, the CPU) dequantized only 2D
weights and broadcast a token's row to its routes wrongly; both fixed.

## Run 2 - 2026-11-21 13:00

KL divergence against the official weights: `quantize kld` streams a model a layer at a time, which a looped core does
not take, so `--resident` loads a model that fits whole and scores through prompt scoring (`prompt_logits`, its
reference divergence). Base: the official checkpoint over the wiki eval text, 40 chunks of 512
(`/mnt/data/models/.profile-tmp/kld/inkling2-official.kld`, 4.1 GB).

```
inference quantize kld --resident -m <model> --kld-base inkling2-official.kld
model                       mean KLD   median    p99      same top   PPL
official (itself)           0.00000    0.00000   0.0000   100.00%    6.7710 (base 6.7610)
IQ4_NL GGUF (ggml-org)      0.00995    0.00437   0.0877    96.04%    6.7877
```

The base's PPL reads its quantized records (u16 codes), ours the logits; the streamed scoring reports both the same
way. The GGUF's 0.00995 is the requant's cost (IQ4_NL experts from NVFP4, Q8_0 dense from BF16): the number a custom
recipe from these weights is measured against.

## Run 3 - 2026-11-21 19:30

Recipe A: the GGUF with the official checkpoint's NVFP4 experts as stored. llama.cpp's GGUFs carry NVFP4 expert
stacks with a per-expert `<weight>.scale` (`[experts]` F32, its converter's `_write_scales_tensor`); its Inkling
converter dequantizes the experts instead. A gguf-py graft (scratchpad `recipe/graft_nvfp4.py`) copied ggml-org's
IQ4_NL GGUF with layers 0-19's `ffn_gate_up_exps` / `ffn_down_exps` replaced by the official bytes (ggml NVFP4 blocks,
the checkpoint's interleaved gate/up rows stacked gate then up) and their scales; layers 20-23 (MXFP8 in the
checkpoint) keep the base's Q8_0, the dense weights its Q8_0. 17,627,963,584 bytes against the base's 17,627,940,768.
gguf-py's NVFP4 dequantization of a grafted row is the official row (the block values exact, the scale 4.5e-5).

Loading needed llama.cpp's per-expert scale: `GgufWeightSource::load_scale` now reads a `.scale` of one value per
stacked expert as the expert scales of #1040, and Inkling's bindings bind `ffn_*_exps.scale` (`parse_block_tensor`
splits off only `.weight` / `.bias`, so a scale arrives as the role `ffn_down_exps.scale`; unbound, the first score
was KLD 13.5).

```
quantize kld --resident against the official base     mean KLD   median    p99      same top   PPL
ggml-org IQ4_NL                                         0.00995    0.00437   0.0877   96.04%     6.7877
recipe A (official NVFP4 experts, rest as ggml-org's)   0.00374    0.00133   0.0379   97.46%     6.7706

inference bench --pa-context-len 8192        pp512   tg128
ggml-org IQ4_NL                              1926    84.5
recipe A                                     1557    78.6
```

Same size, 2.7x lower KLD; what remains is the Q8_0 dense weights and the MXFP8 layers requantized to Q8_0. The
NVFP4 experts run slower than IQ4_NL's: the per-expert scales are applied in separate ops around the raw kernels (the
gated activation path at decode, the explicit activation path at prefill instead of the fused GLU), and the NVFP4
matvec itself.

## Run 4 - 2026-11-21 21:30

`quantize gguf --nvfp4-from <checkpoint dir>` writes recipe A in Rust: an `Nvfp4Source` on the requantize plan
(`ModeloptExperts`, the checkpoint's NVFP4 stacks repacked into ggml NVFP4 blocks, `w13`'s interleaved rows stacked
gate then up) supplies each stack it holds whatever the rules say, and the per-expert `<weight>.scale` is written as
the entry after it. Inkling's mapping (GGUF stack to official stack) is `inference_gguf::official_nvfp4_experts`.

```
inference quantize gguf -m Inkling-2-Mini-IQ4_NL.gguf -o Inkling-2-Mini-NVFP4.gguf \
    --nvfp4-from /mnt/data/models/inkling2-mini-official
Wrote ... (16.41 GiB): 40 NVFP4 (10935.0 MiB), 186 Q8_0 (5724.2 MiB), ...   (28 s)
```

Against the gguf-py graft of Run 3: the same 744 tensors, every one byte-identical, the same metadata keys.

Decode profile of recipe A against IQ4_NL (nsys, ms a token under the profiler): the NVFP4 expert matvec 2.82 against
IQ4_NL's 2.43 (same bytes, 16% slower), and the per-expert scales' separate ops 0.6 (`bmul_f32` 136 a token,
`index_select` 132 more), which a scale applied where the matvec writes its outputs would remove.

## Run 5 - 2026-11-21 22:05

Per-expert scales applied in the kernels instead of separate ops: the expert matvec (both `mmvq_gguf.cuh`'s and
`mmvq_rows.cuh`'s MoE entries) multiplies an output by `scales[ids[pair]]`, the gate's and up's before the GLU, the
down's on its rows; the grouped GLU quantize (`quantize_mmq_q8_1_glu`) scales a row by its assignment's expert before
the activation. Decode no longer builds `ExpertScaled`'s per-route tensors at all.

```
recipe A (NVFP4 experts)                 KLD       pp512    tg128
before (Run 3)                           0.00374   1557     78.6
in-kernel scales                         0.00374   1540     81.7    (two runs: 1544/1540, 81.7/81.7)
ggml-org IQ4_NL (same session)           0.00995   1913     83.6

llama.cpp 44c20b1c0 llama-bench -fa 1    pp512    tg128
recipe A                                 1398     59.0
ggml-org IQ4_NL                          1487     61.1
```

Decode +3.1 tok/s, 1.9 short of IQ4_NL. Prefill is unchanged: Inkling's softcap SiLU is no GLU the fused kernels
take, so its prefill keeps the explicit path (`scaled_act`: 44 ms of `bmul_f32` and ~32 ms more `ucopy_f32` in a
pp512 profile, ~6%); the fused-GLU scales serve NVFP4 MoE models whose activation the kernels fuse. The prefill gap
is the NVFP4 mmq itself (nsys, pp512 + 1 token):

```
mul_mat_q<NVFP4 (40), 128>   528 launches   544.5 ms
mul_mat_q<IQ4_NL (20), 128>  528 launches   355.3 ms
```

llama.cpp's NVFP4 is 6% behind its IQ4_NL in prefill; ours is 20% behind, with our IQ4_NL 29% ahead of llama.cpp's
and our NVFP4 only 10% ahead. Next: what our IQ4_NL mmq path has that NVFP4's lacks.

## Run 6 - 2026-11-21 22:20

Whether the NVFP4 mmq gap is the format or our kernel: nsys kernel sums of a pp512 bench, ours against llama.cpp
44c20b1c0 (`llama-bench -p 512 -n 0 -fa 1 -r 3`), per launch:

```
                      NVFP4 (40)          IQ4_NL (20)        NVFP4 / IQ4_NL
ours (mmq_x 128)      1.031 ms (528)      0.673 ms (528)     1.53
llama.cpp (J 32)      0.989 ms (352)      0.739 ms (352)     1.34
```

Upstream's mmq is rewritten since ours was vendored (config per type, J and `ggml_prec` template parameters), and
its NVFP4 launch is 4% faster than ours; its IQ4_NL is 9% slower than ours. Most of the gap is NVFP4's scale per 16
values (the `q8_0_16` MMA path Q3_K takes, twice the scale work of IQ4_NL's one per 32), not a fix we lack. Porting
upstream's mmq rewrite for 4% of one kernel is not worth it; prefill stays ~20% behind IQ4_NL with NVFP4 experts.

## Run 7 - 2026-11-21 22:40

Where recipe A's KLD comes from. A lossless source GGUF: llama.cpp's converter (`conversion/inkling2.py`) run at
`--outtype bf16` with layers 0-19's NVFP4 stacks filtered out (MXFP8 dequantizes exactly into BF16), then recipe A's
NVFP4 stacks and scales merged in (gguf-py; the converter interleaves blocks' tensors, so a block's additions go after
its last). 22.9 GB: too big for the 3090 with working memory, and the auto map's CPU layers fail (the depth mix read
states across devices and dtypes, fixed by moving each to the current state's device in F32; then the fused attention
has no CPU path). So variants derived from it with `quantize gguf`:

```
quantize kld --resident, official base          size       mean KLD   p99      same top
V0  dense Q8_0, layers 20-23 experts Q8_0       16.41 GiB  0.00376    0.0384   97.56%   (recipe A: 0.00374)
V1  dense BF16, layers 20-23 experts Q8_0       17.78 GiB  0.00317    0.0308   97.64%
V3  as V1, layers 20-21 experts BF16            19.56 GiB  0.00316    0.0310   97.58%
V5  as V0, layers 20-23 gate_up Q6_K            15.79 GiB  0.00381    0.0374   97.55%
official checkpoint against its own base                   0.00000
```

Dense Q8_0 costs 0.0006; MXFP8 to Q8_0 costs nothing measurable (V3 = V1), and to Q6_K (gate_up; down's rows of 1152
take no K-quant) 0.00005 for 0.62 GiB. V5 benches as recipe A (pp512 1539, tg128 81.7). The rest, ~0.0031, is not
in the weights: V3 holds layers 0-21 exactly as the checkpoint does and still scores 0.0031, with both loads at BF16
activations. Some compute difference between the GGUF load and the checkpoint load, which every GGUF recipe pays
(IQ4_NL's 0.00995 included). Next: find it bottom-up (one layer's outputs, GGUF against checkpoint load).

Not the router: V6 (V0 with `ffn_gate_inp` at BF16, as the checkpoint computes it) scores 0.00376, as V0. Not the
tokens: the base stores them, and neither load has a BOS to put first. The checkpoint's `hf_quant_config.json` also
asks for NVFP4 activations (per-token amax, blocks of 16, `fouroversix_mae`) into the NVFP4 experts and MXFP8 ones
into the MXFP8 experts; both our loads quantize activations to Q8_1 there, so the base itself departs from that.

## Run 8 - 2026-11-21 23:30

Bottom-up: the residual stream entering each of the 48 layer visits (24 layers, the core 6-17 three times) and inside
the layers, dumped (an uncommitted env hook) for the first scored forward, one token, from the checkpoint load and
from V1 (layers 0-19 as the checkpoint stores them).

First divergence: layer 0's MoE (relative 4e-3), with its input, expert ids, routing weights and kernel path
identical. The per-expert NVFP4 scales differed: the GGUF loader read `<weight>.scale` at the model's dtype (BF16,
then F32), rounding each expert's scale by up to 2^-9 (6.1190956e-5 stored, 6.1035156e-5 = 2^-14 loaded). Fixed
(read at F32; test `a_bf16_model_keeps_its_experts_f32_scales`). Rescored:

```
                         before fix   after fix
recipe A                 0.00374      0.00377
V1 (dense BF16)          0.00317      0.00329
```

Not the floor. With the scales fixed, visits 0-27 match exactly; the first divergence is visit 27 (layer 15, second
pass), and V1 against itself is exact (deterministic). Inside it every input to the attention is identical (the
controllers' inputs and outputs, the messages), and the q group's projections already differ: q by 3.4e-6 relative at
visit 15 (absorbed by BF16 rounding downstream), the output gate by 1.4e-4 at visit 27, which survives. Both far under
a BF16 step (3.9e-3), so summation order: the checkpoint's BF16 linears run `UnquantLinear`'s matmul and the GGUF's
BF16 tensors the GGUF weight source's linear. The looped, routed model amplifies that (a near-tied expert flips,
everything after diverges) into the ~0.003 floor. Neither load is wrong; the floor is this model's sensitivity to
kernel noise, and recipe differences under ~0.0002 are inside it.

## Run 9 - 2026-11-22 00:00

Recipe B, built in one step from the ggml-org GGUF and the checkpoint: `--nvfp4-from` became `--checkpoint`, whose
source (`CheckpointSource`) also reads MXFP8 stacks as their values (E4M3 times the block's power of two, w13's
interleaved rows stacked gate then up) for the rules to quantize in place of the GGUF's tensor. No lossless BF16
intermediate needed.

```
inference quantize gguf -m Inkling-2-Mini-IQ4_NL.gguf -o Inkling-2-Mini-NVFP4-B.gguf \
    --checkpoint /mnt/data/models/inkling2-mini-official \
    --tensor-type 'blk\.2[0-3]\.ffn_gate_up_exps\.weight$=q6_k' \
    --tensor-type 'blk\.2[0-3]\.ffn_down_exps\.weight$=q8_0'
Wrote ... (15.79 GiB): 40 NVFP4 (10935.0 MiB), 182 Q8_0 (2970.2 MiB), 4 Q6_K (2126.2 MiB), 478 F32   (69 s)
```

Layers 20-23's eight stacks from the checkpoint's values are byte-identical to the same rules over the lossless
BF16 GGUF of Run 7 (`--only` overlays from both).

```
                      size        mean KLD   p99      same top   pp512   tg128
ggml-org IQ4_NL       16.41 GiB   0.00995    0.0877   96.04%     1913    83.6
recipe A              16.41 GiB   0.00377    0.0370   97.56%     1540    81.7
recipe B              15.79 GiB   0.00381    0.0377   97.53%     1531    81.7
(floor, Run 8)                    ~0.0031
```

Recipe B: 0.62 GiB under IQ4_NL at 2.6x lower KLD, within 0.0007 of the floor; the dense Q8_0 is most of what is
left (0.0006, 1.37 GiB to remove).

## Run 10 - 2026-11-22 06:00

The dense weights' bits, by KLD and speed (after the speed work of #1043-#1045). Variants of recipe B from the
lossless GGUF (Run 7's, rebuilt), each changing one group from Q8_0, layers 20-23's experts as recipe B's:

```
                                 size        mean KLD   same top   pp512   tg128
all Q8_0 (= recipe B)            15.79 GiB   0.00368    97.45%     1852    98.8
attention (q k v gate o) Q6_K    15.65 GiB   0.00463    97.10%     1833    95.0
attention Q5_K                   15.58 GiB   0.00737    96.30%     1838    98.7
head Q6_K                        15.69 GiB   0.00380    97.25%     1854    99.5
embedding Q6_K                   15.69 GiB   0.00374    97.42%     1860    98.8
embedding Q4_K                   15.58 GiB   0.00485    97.06%     1857    98.6
head + embedding Q6_K (C)        15.58 GiB   0.00387    97.27%     1863    99.2
```

The attention stays Q8_0: Q6_K costs 0.001 and decodes slower (its matvec at these shapes is not bound by the bytes
it saves), Q5_K doubles the excess KLD for no speed. The head and the embedding take Q6_K at the noise's edge
(+0.0002 together) for 0.21 GiB and +0.4 tok/s: recipe C (`Inkling-2-Mini-NVFP4-C.gguf`), recipe B's rules plus
`--tensor-type '^(output|token_embd)\.weight$=q6_k'` from a source whose head and embedding are BF16.
