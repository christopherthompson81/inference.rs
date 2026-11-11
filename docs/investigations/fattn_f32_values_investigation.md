# fattn with f32 value accumulation

llama.cpp's mma FlashAttention kernel (vendored in `inference-fattn`) keeps the softmax-weighted value sums in f16 on
Turing and later (`T_C_VKQ = tile<16, 8, half2>`), also where the partial results of parallel warps are staged in
shared memory for the combine. Upstream (checked at 970b51ce7, 2026-10-22) still does, and ignores a precision request
for flash attention on CUDA. TRELLIS.2's DiT at 1024 attends diffusely over up to ~18K tokens with values up to ~200:
#903 scales the values against the overflow (NaN), but the precision stays f16's. Goal: an opt-in variant whose value
sums are f32.

## Run 1 - 2026-11-11

Question: how much of fattn's error on a long diffuse call is the f16 accumulator? The house job's failing self-attention
call (block 0, t = 375, 18171 tokens, 12 heads, d 128; dumped as `attn_dump.safetensors`, scratch) against PyTorch's
f32 SDPA on the same bf16 inputs, relative L2:

    fattn as it is (values scaled, #903):            2.87e-2
    P and V rounded to f16, sums in f32 (simulated):  1.38e-5
    the reference output rounded to bf16:             1.64e-3

So the inputs' f16 rounding costs nothing measurable; the f16 sums are all of it, ~2000x the f32 variant's error and
~18x a bf16 output's own rounding.

Plan:
- `mma_tile_sizes` gains an f32-values flag: `T_C_VKQ` becomes the float tile matching `T_C_KQ` (the same A/B tiles,
  so the mma overloads exist: `tile<16, 16, float>`, or `tile<16, 8, float>` for 8-column warps).
- The running rescale by the softmax max gets float branches (the AMD paths have them; Turing does not).
- The combine stages partial sums in half in shared memory, which would overflow again: with f32 values each warp
  writes its sums divided by its own row sum (bounded by max|v|), keeps that row sum in the meta padding (16 bytes a
  row, 8 used), and the final loop weights each partial by scale * row sum instead of scale, so the stream-k fixup and
  the final division are unchanged.
- Instances only where needed: head dim 128, ncols2 1 (no GQA packing), not the 8-column (decode) shapes.
- `FattnOptions` gains the flag; TRELLIS.2's DiT asks for it.

## Run 2 - 2026-11-11

Built as planned (`FattnOptions::f32_values`, `fattn_layout.f32_values`, `flash_attn_ext_f16<..., f32_values>`), with
three bugs on the way, each caught by `tests/integration/f32_values.rs`:

- The layout only reaches the kernel when Rust sends one (`uses_lay`), which plain dense calls did not: the flag was
  dropped and the f16 kernel ran (f32 and f16 errors identical, head 64 "supported").
- With one warp per column (np = 1) and a stream-k split, the original never stages the meta (max, row sum) in shared
  memory; the f32 weight reads the row sum from there, so it read garbage (inf). The meta is now always staged for
  f32 sums.
- The meta row of a thread's second column: `T_C_VKQ::get_i(threadIdx.x % 4)` is column c for a half2 tile but
  element 2c for a float one, so one row's meta was never written and another's twice (erratic inf in whole rows,
  counts in multiples of 1024 = 8 heads x 128). Fixed with `get_i(2*c)` for float.

Then a probe over batch 1/2, causal or not, 5 to 300 tokens: all finite, within f16 rounding (<= 0.002) of the f16
kernel on unit inputs. The cases: the reference within 2e-3 (f16) / 8e-3 (bf16) at 5 to 1000 Q rows, batch 1 and 2,
masked, implicit causal; an even row over 8192 keys valued ~1000 within 1e-3 of the mean (the f16 kernel gives NaN
there); a diffuse 8192-token case > 4x closer to the reference than f16 sums; head 64, GQA and softcap unsupported.
The existing parity cases pass unchanged.

Speed (`bench::f32_values_vs_f16`, bf16, 12 heads x 128, 3090):

    batch 2 x  8259: f16 sums 14.3 ms (58.6 TFLOP/s), f32 sums 17.9 ms (46.9), x1.25
    batch 1 x 18171: f16 sums 32.4 ms (62.6),          f32 sums 40.8 ms (49.8), x1.26
    batch 2 x  4096: f16 sums  3.8 ms (54.4),          f32 sums  4.6 ms (44.7), x1.22

GeForce Ampere runs f16-input mma with f32 accumulation at half the f16-accumulation rate, and P.V is half the FLOPs:
most of the 25% is that, not the registers.

TRELLIS.2 with its DiT attention on the f32 sums (temporary env switch), `every_pipeline_stage_matches_the_reference`
(rms against the reference; the same cell sets throughout):

    stage            f16 sums (values scaled)   f32 sums
    structure        0.82                       0.47
    shape 512        0.047 (max 3.76)           0.029 (max 0.64)
    shape 1024       0.099                      0.078
    texture          0.033                      0.035

End to end: robot 39.4 -> 42.1 s (+7%), the house (seed 1, 18171 tokens) 87.2 -> 102.8 s (+18%).

## Run 3 - 2026-11-11

Question: can a hybrid keep f16's speed: each KV tile's P.V products summed at the f16 rate in a half tile (a warp's
<= 64 keys of a tile cannot overflow at TRELLIS.2's values), then added into the f32 sums? Built (a zeroed
`tile<16, 8, half2>` per V column chunk, mapped into the float tile: half2 element l is float elements
4*(l/2) + 2*(l%2) + {0, 1}); the f32-value cases pass. Speed (`bench::f32_values_vs_f16`):

    batch 2 x  8259: f16 sums 14.3 ms, hybrid 30.7 ms (x2.15)
    batch 1 x 18171: f16 sums 32.4 ms, hybrid 73.6 ms (x2.27)
    batch 2 x  4096: f16 sums  3.8 ms, hybrid  7.6 ms (x2.00)

Slower than the plain f32 accumulation (x1.25). Every head-128 instance already sits at the 255-register cap with
stack (`cuobjdump --dump-resource-usage`, sm86): 16 / 32 / 64 columns at 256 / 336 / 512 bytes in f16, the hybrid
400 / 528 / 640; the temporary tile and its conversion push more of the hot loop to local memory than the f16-rate
mma saves. Capping f32 requests at 32 columns (less spill per thread, more warps per column) was worse still: x5.0
to x5.6. Both reverted; the plain f32 accumulation re-measured at x1.25 / x1.26 / x1.22.

## Run 4 - 2026-11-11

Decision (user): TRELLIS.2's DiT attention runs on the f32 sums by default. `dit::sdpa` calls fattn with
`f32_values` on CUDA when `supported` (always, at its head 128 without GQA, from Turing on), the shared `Sdpa`
otherwise. #903's value scaling (`value_abs_sums_bf16`, `value_scale_finish`, the scale arguments of
`head_norm_rope_bf16` and `gate_residual_bf16`, the cross-attention scale) is removed: f32 sums do not overflow there.
A GPU case: even attention over 8192 keys valued ~1000 through `sdpa` lands on the mean within bf16's rounding.

    every_pipeline_stage_matches_the_reference (rms, max):
      structure   0.50          (f16 sums scaled: 0.82)
      shape 512   0.029, 0.64   (0.047, 3.76)
      shape 1024  0.076, 1.31   (0.099, 3.33)
      texture     0.036         (0.033)
    robot 1024-cascade 41.5 s (39.4 s with f16 sums), GPU peak 6.4 GB; the house, seed 1: 97.6 s (87.2 s)

Bundle (`--size`): libinferencefattn.so +0.18 MiB for the three instances across archs; libinference_ffi.so +1.76 MiB
from the TRELLIS.2 work merged today (#896-#903, no size run then); baseline updated.
