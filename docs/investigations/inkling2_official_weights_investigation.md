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
