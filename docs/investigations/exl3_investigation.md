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
