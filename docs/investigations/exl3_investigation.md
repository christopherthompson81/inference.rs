# EXL3 investigation

#943: load and run ExLlamaV3's EXL3 quants, then write them. Reference: exllamav3 1.6.6 (PyPI, prebuilt
extension) in `/mnt/data/exl3-venv` (`--system-site-packages` over the system torch 2.11+cu130, with a newer pydantic).

## Run 1 - 2026-11-13 10:20

Format, from exllamav3's `modules/quant/exl3.py`, `exllamav3_ext/quant/{reconstruct.cu,exl3_dq.cuh,codebook.cuh}`:

- An EXL3 linear stores `trellis` (i16, `(in/16, out/16, 16K)`, a 16x16 tile per entry; `16K + 8` words for a
  half-integer K), `suh` (f16, `in`), `svh` (f16, `out`), and optionally a scalar i32 marker `mcg` (holding
  0xCBAC1FED) or `mul1` (0x83DCD12D) naming the codebook; none means 3INST. The model's `quantization_config` is
  `quant_method: exl3` with `bits` (a float) and a per-tensor `tensor_storage` map; embeddings and norms stay f16.
- Each tile is a K-bit-per-weight stream read most significant bit first over the little-endian u32 pairs of its
  u16 words (so u16[1] before u16[0]); weight p takes the 16-bit window ending at bit (p+1)K, wrapping around the
  tile. Half-integer K alternates steps of K_A + 1 and K_A (MASK 0xAAAA): position 2m+1 ends at (m+1)(2K_A + 1),
  2m at K_A + 1 bits before that.
- Codebooks: 3INST `x*89226354 + 64248484`, `& 0x8fff8fff ^ 0x3b603b60`, the two f16 halves added in f16; mcg the
  multiply alone, then the same; mul1 `x*0x83DCD12D`, the byte sum over 0x6400 read as f16, then
  `x*0x1eee + 0xc931` with one f16 rounding (`__hfma`).
- Stream position p is tile element `perm[p]` (the tensor-core order in `tensor_core_perm`).
- The weight (exllamav3's `[in, out]`) is the tiles, then a 128-wide Sylvester Hadamard (scaled 1/sqrt(128)) down
  the rows, times `suh`, then the same along the columns, times `svh`.

numpy prototype against `LinearEXL3.get_inner_weight_tensor` (Qwen3-0.6B-exl3 3.0bpw, q_proj): reading u16 words in
natural order gave max abs diff 6.0 (sorted tile values differed too); with the u32-pair order, 0.0. The full
transform with exllamav3's own `get_hadamard(128)` (equal to Sylvester's): max rel 4.9e-4, f16 rounding of its
in-place f16 steps.

## Run 2 - 2026-11-13 16:00

Rust port (`inference-quant/src/exl3`, codebooks in `gguf/trellis.rs`), `scripts/exl3_parity.py` (exllamav3's
`get_weight_tensor` against the `exl3_reconstruct` example). Test tensors fetched by byte ranges from recent quants:
an mcg tensor (an LFM3 4.49bpw quant, K = 6), mul1 (a Qwen3-ASR decoder 3bpw quant, K = 3), half-integer K = 1.5
with mul1 (a Qwen3.8-27B 1.5bpw quant, 24 words a tile), and Qwen3-0.6B-exl3's 3INST tensors (K = 3, 4, 6):

```
mcg  words 96  max rel err 6.70e-04
mul1 words 48  max rel err 5.26e-04
mul1 words 24  max rel err 4.77e-04   (K = 1.5)
3inst words 48/64/96  4.67e-04 .. 6.88e-04
```

All within f16 rounding of exllamav3's f16 rotation.

Loading: `QuantizedConfig::Exl3` (recognized before the integer-`bits` parse), `checkpoint_linear` reconstructs each
EXL3 linear into a dense layer at load. `inference run -m Qwen3-0.6B-exl3/3.0bpw` answers ("Paris"), 465 tok/s.

Perplexity, README.md, first 512 tokens, second half scored (ours: the perplexity example `--llama-cpp-ctx 512`;
exllamav3: `model.forward` over the same tokens):

```
model                 exllamav3   ours bf16   ours f16   ours f32
Qwen3-0.6B (base)     3.1545      -           3.1686     -
EXL3 3.0bpw           3.8272      3.8486      -          -
EXL3 3.5bpw           3.5856      3.6197      -          -
EXL3 4.0bpw           3.3922      3.4427      3.4216     3.4204
```

The base model already differs by +0.45% between the engines; the quant's own cost is +8.0% (ours, f16) against
+7.5% (exllamav3) at 4.0bpw. The tensors agree to f16 rounding, so the rest is the engines' forward passes on a 0.6B
model. Next: trellis-native CUDA kernels (dense reconstruction holds the weights at f16 size).

## Run 3 - 2026-11-13 18:30

Kernel plan (the user chose vendoring exllamav3's kernels, MIT). The include graph of exllamav3 1.6.6's
`quant/exl3_gemm.cu`, `exl3_gemv.cu`, `hadamard.cu`, `reconstruct.cu`, `exl3_kernel_map.cu`, `exl3_devctx.cu`:

- torch-free kernel core: `exl3_gemm_kernel.cuh` (359 lines), `exl3_gemm_inner.cuh` (1005), `exl3_gemv_kernel.cuh`
  (491), `exl3_dq.cuh`, `codebook.cuh`, `hadamard_inner.cuh`, `exl3_kernel_map.cuh`, `exl3_devctx.cuh`,
  `coop_autotune.cuh`, `ptx.cuh`, `arch.cuh`, `compat.cuh` (plus ROCm headers, not needed).
- torch-bound: the launchers (`exl3_gemm.cu` 892 lines: autotune hash, kernel-map choice, per-device lock buffers,
  cooperative launches, CUDA-graph parameter recording; `exl3_kernel_map.cu`, `exl3_devctx.cu`, `hadamard.cu`,
  `reconstruct.cu`) and `util.h`/`util.cuh`/`graph.cuh`.
- instantiation: `comp_units/exl3_comp_unit_{1..8}_cb{0,1,2}.cu`, one translation unit per bitrate and codebook.

At run time y = x W with W = diag(suh) H Wi H diag(svh) (H the 128-block Hadamard, Wi the trellis tiles): the GEMM
and GEMV kernels take `suh`, an `A_had` scratch and `svh` and do both rotations inside (grid syncs between the
stages), so no separate Hadamard launch. Our part: C ABI launchers in place of the torch ones (the GEMV fast path for
small m, else the GEMM with the kernel map's default shape, no autotune), a device context (SM count, compute
capability, the lock buffer, the shared-memory limit), the build into a kernel library per arch, and an `Exl3Layer`
QuantMethod holding the trellis on the GPU.

## Run 4 - 2026-11-13 19:30

exllamav3's kernels vendored: `third_party/exllamav3` (1.6.6, MIT; headers pristine, the four torch-bound sources
trimmed and marked), built from `kernels/cuda/exl3/*_instance_exl3.cu` into their own `inferencequant_exl3` library
(24 bitrate x codebook units, the three half-integer ones, the half-integer GEMV, the kernel map, the device context,
the GEMV launcher and our C ABI launcher: 6.7 MB for sm86). `Exl3Layer` keeps the trellis, `suh` and `svh` on the
device; f16 input, f32 output, the rotations inside the kernels.

GPU test against the CPU reconstruction (random 512x512 trellises, every codebook, 40 words = K 2.5; m = 1 and 32):
max error 3.4e-4 .. 5.4e-4 of the largest output (the kernels accumulate in f16).

Qwen3-0.6B-exl3 on the kernels: perplexity 3.4226 at 4.0bpw (dense f16 3.4216), 3.8432 at 3.0bpw. Decode 207 tok/s
with zeroed scratch and output buffers, 250 with uninitialized ones (the kernels write them whole); the dense layers
ran 465. exllamav3's own generator on the same model: 170 tok/s. A 0.6B model's decode is launch-bound (196 EXL3
matmuls a token, ~20 us each); the comparison that matters is a large model. Decode graphs capture with the
cooperative launches.

## Run 5 - 2026-11-13 21:30

A 27B hybrid (Qwen3.8-27B-exl3, 3.5bpw, mul1 codebook, with its vision tower) on the 3090.

Loading first failed on the vision tower: its loaders build plain linears (`inference_nn::linear`), which found no
dense `weight`, and the trellises are padded to multiples of 128 (a 1152 -> 4304 projection is stored as 4352, with
`svh` and the bias padded too). Fix: `exl3::implied_config` turns a trellis into an EXL3 config where the model passed
none, and `exl3::dense_linear_in` reconstructs a plain linear from its trellis and crops the padding (`narrow`).

Decode, one short poem prompt:
- ours EXL3 kernels: 36.3 tok/s (756 tokens)
- ours GGUF recipe A (IQ4_KS bulk, 14 GiB): 50.6 tok/s
- exllamav3 1.6.6 generator: 45.5 tok/s (500 tokens in 11.0 s, prefill 0.04 s). An earlier reading of 63.4 counted
  the output by re-encoding its text (885 tokens for 500 sampled); its own `new_tokens` count is the right one.

So the EXL3 path trails exllamav3 by ~20% on the same weights. Suspects: per-linear bf16 -> f16 input and f32 -> bf16
output conversions, the default kernel shape without exllamav3's autotune, scratch allocations per call.

## Run 6 - 2026-11-13 23:00

Where the 27B's decode time goes (nsys, graphs off so every kernel shows; ~45 decode steps): 27.7 ms of kernels a
token, all of the step, so GPU-bound. The EXL3 GEMMs take ~80%: the 3-bit `exl3_gemm_kernel` ran at ~80 us for a
~33 MB tensor (~410 GB/s of the 3090's 936).

exllamav3's own generator under nsys (graphs traced) runs different kernels for the same decode:
`exl3_gemv_int8_sq_kernel` for every mul1 matmul at m = 1 (54% of its kernel time), `exl3_mgemm_kernel` for fused
projections. The int8 GEMV is `EXL3_INT8_GEMV`, default 2 (plain int8 activations, dp4a, both Hadamards and the
activation quantization in one launch) for mul1 tensors at K + half <= 5 (Ampere/Ada; 6 on Hopper/Blackwell).

First tried its autotuner (`coop_autotune.cu`, vendored without torch; shape and grid size timed once per
(m bucket, k, n), cached in `~/.cache/inference-rs/exl3_autotune_v1.bin`): decode unchanged at 35.5 tok/s. The tuned
choices are mostly the default shape over all 82 SMs. Kept: it costs one tuning pass per shape at first use, and a
shape first seen inside a capture takes the default.

Then the int8 GEMV (vendored; our instances only the f32-output plain forms, one and two rows plus the cooperative
fallback; workspace per stream, since its counters reset themselves and two models on one device would race):

```
27B 3.5bpw mul1, decode tok/s     ours before   ours int8 GEMV   exllamav3 1.6.6
poem prompt, ~250 tokens          35.5          47.0             45.5
```

Accuracy at decode (teacher-forced, each step the previous prompt plus one token through the prefix cache, 98.9%
hits so each scored step is an m = 1 decode; 6 passages x 96 tokens of README.md):

```
EXL3_INT8_GEMV=2 (int8)   ppl 2.88512
EXL3_INT8_GEMV=0 (f16)    ppl 2.88555
mean |d logprob| 0.00103, top-1 agreement 99.8%
```

Prefill perplexity cannot see it (m = 512 never takes the GEMV): 1.4134 both ways. The GPU test's int8 case
(random 512 x 512, mul1, m = 1 and 2) measures 6e-3 of the largest output against 3.4e-4 .. 8e-4 for the f16 kernels.
