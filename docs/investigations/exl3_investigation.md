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

## Run 7 - 2026-11-14 00:30

MoE experts: Qwen3-30B-A3B-exl3 3.0bpw (turboderp, 3INST; experts stored one linear each,
`mlp.experts.N.{gate,up,down}_proj.{trellis,suh,svh}`). `Exl3Experts` keeps each expert's trellis on the device
with a device table of their addresses, and multiplies through exllamav3's `exl3_mgemm_kernel` in indexed mode (slot
j: its row times expert `indices[j]`); the kernels were already in the library (the comp units instantiate gemm and
mgemm), only a C launcher was new. Graph-safe: no host reads.

Prefill on that path (one row a slot, so each slot reads its expert whole): 2598 tokens in 32.8 s, 79 tok/s. Fixes,
2600-token README prompts, fresh each time (the prefix cache survives restarts, so a repeated prompt measures
nothing; the server's own prompt rate also undercounts chunked prefill, so these are wall clock):

```
experts at prefill                                         tok/s
one row a slot (indexed mgemm)                             79
grouped on the host, one GEMM an expert (128 launches)     ~330
grouped, one mgemm, every group padded to the largest      ~330 (690 under nsys: 60% mgemm, 17% the row gather)
grouped, one mgemm, groups of at most 64 rows              1800
exllamav3 1.6.6                                            2700
```

Grouping reads the indices on the host, so it runs only outside a capture and past 64 slots; decode keeps the
indexed kernel.

Decode: ours 100 tok/s, exllamav3 123.5. Per token (nsys, graph nodes): ours 11.2 ms of kernels, exllamav3 ~7.3.
exllamav3 fuses q/k/v into one mgemm (2 launches a layer, 39 us, against our 4 at 63 us: ~1.2 ms a token), runs the
experts through fused cooperative MoE kernels (`exl3_moe_coop_a/b`: gate+up+activation, then down with the routing
weights, 59 us a layer against our 76 us plus the routing glue), and has none of our per-linear bf16/f16/f32 casts
(~1 ms).

Perplexity (README.md, 512 tokens, second half scored): ours 2.1382, exllamav3 2.1384.

## Run 8 - 2026-11-14 03:00

The fused decode paths, Qwen3-30B-A3B-exl3 3.0bpw, decode tok/s (poem prompt, ~150 tokens):

```
step                                                               ours    exllamav3 1.6.6
Run 7 (indexed mgemm per projection)                               100     123.5
+ exllamav3's fused MoE (exl3_moe_coop a/b: gate+up+silu, down)    123.5
+ f16 model dtype (no input/output casts around each linear)       ~126 (bf16 kept as the default)
+ q/k/v in one mgemm, per-matrix widths                            124     (35 us a layer against 48 apart)
+ the same, sliced into equal column slices                        142
```

The fused MoE (`exl3_moe_coop_launch`, vendored without its torch front ends) takes up to 8 tokens; our C ABI fills
its parameter block from a struct of our own. q/k/v: q, k and v each rotate their input with their own `suh`, so
their trellises cannot be concatenated into one; exllamav3's mgemm takes the three, in sliced mode (q cut into
k-wide column slices, each with its trellis and `svh` offset to its first column, the input rotated once per
source). The per-matrix-width mode left k and v's blocks idle while q ran (35 us); sliced, every slice the same
width. The output pointers change every call, so a one-thread kernel fills them from the group's device tables
(built on the first call, a prefill, since the decode graphs capture later and a capture cannot copy tables to the
device). Two traps on the way: the q/k/v layers arrive wrapped in ColumnParallelLayer, which had to forward the
EXL3 hook; and a group built only on a fusable (small) call never got its tables, as the first such call was inside
a capture. mul1 groups at K <= 5 stay unfused (the int8 GEMV alone is faster, exllamav3's own gate); gate/up of a
dense MLP fuse the same way.

Decode accuracy (teacher-forced, 6 x 96 README tokens through the prefix cache): fused 4.74172, unfused (master)
4.75968; mean |d logprob| 0.0165, top-1 agreement 99.8%. The 27B (mul1, int8 GEMV) is unchanged at 47 tok/s.

## Run 9 - 2026-11-14 05:00

Writing EXL3 (#943 steps 2-3), planned. Holding every layer's Hessian for a whole-model pass does not fit: H is
`in x in` f32 per input, ~1.2 GB for one 27B down_proj, ~90 GB over its 64 layers. exllamav3's converter goes a
module at a time: capture that module's Hessians from the current calibration state, quantize it, push the state
through the quantized module, next. About 4000 lines of Python (`conversion/convert_model.py`, `exl3_lib/
quantize.py`). In phases:

1. Hessian capture: `x^T x` per linear (f32, over every row, over the count; exllamav3 adds `sigma_reg` x the
   mean diagonal at quantization, not at capture), collected in passes that fit a budget, written as exllamav3's
   `--hessians` directory (`<key>.safetensors` holding `hin`), so exllamav3's own converter can check them.
2. The tensor encoder: rotation (`su`/`sv` signs, 128-wide Hadamard), LDLQ over the Hessian, Viterbi over the
   bitshift trellis, packing; tensor by tensor against exllamav3's quantizer.
3. Whole models: the module-at-a-time driver with the state propagated through quantized modules, output in
   exllamav3's layout, perplexity/KLD against exllamav3's conversion.

Phase 1 done here. `hessians_out` (SDK `with_hessians_out`; the CLI gets the converter in phase 3) asks a load
for a calibration file's Hessians in place of quantizing: every linear captured unquantized and left on its device
(`CaptureAllResident`; `CaptureAll`, the UQFF write's, moves captures to the host), one calibration pass per
4 GiB of Hessians. Traps: the parallel-layer wrappers forward hooks through a `delegate_to!` list that needed the
new one; an untied embedding records no rows (skipped). A tied one (Qwen3-0.6B) records the lm_head's rows under
`model.embed_tokens`.

Qwen3-0.6B, README.md (6272 tokens), against transformers' (bf16 both, forward pre-hooks, the same 1024-token
chunks; `scripts/exl3_hessian_parity.py`): rows equal (6272), relative Frobenius error 1.4e-3 .. 1.1e-2 over
layers 0, 14 and 27 (larger deeper, as the two engines' bf16 activations drift). Shared inputs (q/k/v, gate/up)
equal exactly. Collection of 485K tokens (exllamav3's standard calibration text, 350 KB of each part): 45 s.

Our Hessians through exllamav3's own converter (`-hess`, Qwen3-0.6B, 4.0bpw, 3inst), against its own calibration:
exllamav3 takes them ("Using precomputed Hessians"). A tied embedding's rows are the head's, so they are written as
`lm_head` (the key the converter reads; without it the head was quantized uncalibrated). Perplexity
(exllamav3's forward, docs/src/content/docs/reference/*.md, not in the calibration text):

```
                                   512 tokens (head uncalibrated)   4096 tokens
bf16                               6.7753                           11.0421
exllamav3, own calibration         6.8568                           11.4606  (+3.8%)
exllamav3, our Hessians            6.8099                           11.6339  (+5.4%)
```

The 512-token reading was noise. Ours are the unquantized model's activations, collected in one pass;
exllamav3's own are captured module by module from the state after the quantized modules before it, which lets
each layer's LDLQ see (and compensate) its inputs' quantization error. That state propagation is phase 3.

## Run 10 - 2026-11-14 08:00

Phase 2, the tensor encoder (`exl3::encode::encode`), after exllamav3's `quantize_exl3`:

- 2a: the Viterbi tile quantizer vendored (`quantize_tiles` with its dense specializations, and the half-integer
  rates), in its own library `inferencequant_exl3q` that only writing opens; `pack_tile` in Rust. Both rates pack
  as one most-significant-first stream of u32 words (whole: K bits a position; half: K and K + 1 alternating, the odd
  positions taking the extra bit). The quantizer's states, packed and decoded by our loader's decoder, give back
  exactly the values it reports.
- 2b: H damped (0.025 x the mean diagonal) and rotated by the input signs and blockwise Hadamard, Cholesky by
  cuSOLVER (`potrf`; a row-major symmetric matrix is its own column-major view, so an upper factor there is L in
  our lower triangle, the other triangle left as input: LDLQ masks each span's rows of L to the columns left of
  their own block instead of a k x k mask), the 16 x 16 diagonal blocks inverted on the host; the weight
  regularized (column rms over their mean, the output Hadamard, row rms over -1.2437, the input Hadamard, a global
  scale searched coarse-then-fine on a diagonal sample of tiles plus the extremes); LDLQ in 128-row spans of 16-row
  blocks; rotated back; the scales refit twice (output columns in closed form, input rows by a cuSOLVER solve of
  (Q Q^T) o H). Signs from our own seeded xorshift, not torch's RNG.

Against exllamav3 on Qwen3-0.6B tensors with our phase-1 Hessians (`scripts/exl3_encode_parity.py`; "ours" is
exllamav3's `LinearEXL3` reconstructing our tensors, which equals the error our encoder reports, so the format
matches), output error tr(E H E^T) / tr(W H W^T):

```
                    q_proj (layer 0)        down_proj (layer 14)    o_proj (layer 27)
2.0 3inst           0.009853 / 0.010065     0.045355 / 0.045345     0.038573 / 0.038562
4.0 3inst           0.000660 / 0.000669     0.002998 / 0.003003     0.002561 / 0.002564
2.5 mul1            0.004910 / 0.005009     0.022551 / 0.022604
3.0 mcg             0.002454 / 0.002493     0.011324 / 0.011327
4.0 mul1            0.000636 / 0.000645     0.002896 / 0.002895
```

(exllamav3 / ours). q_proj always a little above: within our own seed spread (4.0 3inst over seeds 1-6:
0.000651 .. 0.000669). Each tensor encodes in under a second.

## Run 11 - 2026-11-14 10:30

Phase 3, the converter (`inference quantize exl3`, SDK `with_exl3_out`, `exl3_convert` example): every linear
captured unquantized on its device; the modules grouped by layer, and for each layer in order one calibration pass
collecting its Hessians, each linear encoded (`exl3::encode`) and swapped (`PendingIsqLayer::replace`) for its
reconstruction, so the next layer's Hessians see the encoded layers before it (exllamav3's state propagation,
reached by re-running the model instead of streaming one module's state). The head last at 6 bits; a tied
embedding's rows are the head's, written as `lm_head` (the lookups keep the embedding). Output: each linear's
trellis/suh/svh/marker in place of its weight, every other source tensor copied (the source's own `lm_head.weight`
next to a tied embedding dropped: it doubled the output, 932 against 636 MB), the config with its
`quantization_config`, the tokenizer and side files.

Qwen3-0.6B, 4.0bpw 3inst, exllamav3's standard calibration text (485K tokens, 29 passes): 10.5 min. Perplexity:

```
                                     exllamav3's forward (4096 tokens)   ours (whole file, 512-token windows)
bf16                                 11.0421                             10.7283
exllamav3, own calibration           11.4606  (+3.8%)                    11.3686  (+6.0%)
exllamav3, our one-pass Hessians     11.6339  (+5.4%)                    11.3570  (+5.9%)
ours, layer by layer                 11.4159  (+3.4%)                    10.8944  (+1.5%)
```

Ours is at least exllamav3's own conversion by its own forward, and well ahead by ours (whose Hessians came from
our forward's activations, the ones it then runs). The CLI path: 3.0bpw mul1 from README.md alone, 2.3 min, our
PPL 11.3261. A tiny Llama (128 wide, 266-token vocabulary, the head kept dense) writes, loads and decodes.

Not yet: MoE experts (the converter bails on stacked experts), tensors not in 128 blocks (exllamav3 pads them),
models too large for the device in full precision (needs exllamav3's module streaming), and the L passes (an early
exit after each pass's own layer would halve them).

## Run 12 - 2026-11-14 11:30

Converter gap 1, widths not in 128 blocks. exllamav3 zero-pads such a linear on both sides (`Linear.pad_out`; the
padded inputs are zero, so H has zero rows there and the damping keeps it definite; the padded outputs get zero
scales), stores the trellis, `suh` and `svh` at the padded sizes, and pads the bias at load. Ours now does the same
in `encode`, and the loader takes a trellis up to one block wider than the linear: `Exl3Layer::with_features`
zero-pads the input and cuts the output (a padded fused group runs layer by layer); the CPU reconstruction is cut.

A 200 x 176 linear (padded to 256 x 256, 4.0 mcg): its cut decode carries the reported output error, and the kernels
layer at 176 -> 200 multiplies as that decode does (2e-3). The tiny Llama's 266-row head now writes as a 384-row
trellis, loads and decodes.

## Run 13 - 2026-11-14 12:00

Converter gap 2, the passes. A layer's Hessians need the model only up to it, so its pass now stops there:
`inference_nn::decoder::stop_after_layer` (thread-local, so another model's forward in the process runs whole) makes
the shared decoder stack's forward fail with `CALIBRATION_STOPPED` after that layer, and the calibration loop takes
that as a finished chunk. Models outside the shared stack run their whole forward, as before.

Qwen3-0.6B, 4.0 3inst, the same 485K tokens: 363 s against 625 s, and the written trellises are bit-identical to the
whole-pass run's (q_proj of layer 0, down_proj of 13, up_proj of 27, the head).

## Run 14 - 2026-11-14 13:30

Converter gap 3, MoE experts. A routed Hessian mode collects, per expert of a stacked `[E, out, in]` projection,
x^T x over the rows its router sent it, over their count (a unit Hessian for an expert no row reached: encoded
uncalibrated, exllamav3's fallback); the stack is stacked on the host when read, since one device allocation of
every expert's Hessian was the layer's largest. The converter encodes each expert as `<experts>.<e>.<proj>`, swaps
the stack for the reconstructions, copies none of the source's expert tensors, and keeps routers in full precision.

Two failures on the way. A tiny OLMoE panicked in CUDA `index_add` on one matrix of a stack: the backend copies the
destination contiguous from 0 but then sliced the copy with the source's offset; fixed (and covered). OLMoE-1B-7B ran
out of device memory in the experts' forward: collecting stats forces the gather path, where each slot read a
copy of its expert's weight (8192 slots of a 1024-token chunk at top-8, 4 MB each); `UnquantLinear` now groups the
rows by expert past 64 slots, outside a capture: one matmul an expert.

OLMoE-1B-7B-0125-Instruct (64 experts, top-8, 16 layers; 13.8 GB in bf16) at 4.0 mul1, 485K calibration tokens:
2408 s, 3.5 GB written, device memory peaking at 23.8 of 24 GB. exllamav3 has no OLMoE, so only our engine scores it
(512-token windows):

```
text                          f32       bf16      EXL3 4.0
docs reference pages          13.2266   13.4252   13.0388
README.md                     1.6408    1.6491    1.6612  (+0.7% on bf16)
pipeline/isq_flow/plan.rs     5.5253    5.4942    5.6502  (+2.8%)
```

The first text's reading below bf16 (and f32) is that text's luck; the other two put the encoding where 4 bits sit.

## Run 15 - 2026-11-14 14:30

Converter gap 4, models larger than the GPU: streaming. The converter now loads the model on the host in bf16 (`inference
quantize exl3` default; `--resident` keeps the old GPU load) and keeps the calibration text's hidden states there, one
tensor per 1024-token chunk. For each layer, the shared decoder stack rebuilds it on the GPU from the checkpoint
(`LayerStream::build_layer`, the stack keeping its spec and var builder; the rebuilt linears tracked under their checkpoint
keys), runs every chunk through it to collect the Hessians, encodes and swaps its linears, runs the chunks through the
encoded layer again for the next layer's inputs, and drops it. Then the head, over the final-normed states.

Tiny Llama, both ways: every layer tensor bit-identical between the resident and streamed writes. The head differs:
the resident calibration forward hands the head only each chunk's last row (the logits selection), so its Hessian was
one row per chunk; streamed, it takes every row, as exllamav3's does.

Qwen3-0.6B, 4.0 3inst, the same 485K tokens: 173 s against 363 s resident (one pass over the states per layer instead
of a pass through the model up to it), 7.9 GB peak RSS; all 702 layer tensors bit-identical to Run 13's, the head
not.

Found on the way: a tied model never loaded the `lm_head.trellis` written for it (ours or exllamav3's): the decoder
took the full-precision embedding as the head whenever `tie_word_embeddings`. An EXL3 checkpoint holding its own
`lm_head` trellis now loads it. Qwen3-0.6B, 200 KB of wikitext-2 test, our perplexity example:

```
bf16                                24.4875
EXL3 4.0, resident (one-row head)   25.5624
EXL3 4.0, streamed (every-row head) 25.5182
```

(the small eval text of the earlier runs put both within noise of the full-precision head's 24.205: 24.226 / 24.247).

## Run 16 - 2026-11-14 15:30

Speed-ups. Encode time by phase (temporary probe, random weights, 4.0 mul1):

```
shape (out x in)   factor   regularize   scale search   LDLQ     refit   pack (CPU)
1024 x 1024        3 ms     0.4 ms       36 ms          94 ms    6 ms    16 ms
3072 x 1024        3        1            95             261      8       46
5120 x 5120        32       9            167            1836     101     417
17408 x 5120       49       45           549            6308     316     1405
```

LDLQ is ~80%, and it is the trellis quantizer's own time: per row of tiles, ~20 us a tile at any count (64 tiles 1.4
ms, 320 tiles 6.3 ms, 1088 tiles 21 ms), so it is throughput-bound, not launch-bound. Consequences:

- Batching q/k/v (or gate/up) into one LDLQ, which is exact when they share the input signs and hence the factor
  (LDLQ is column-separable given L), would save nearly nothing: not done (it would also move the output off
  exllamav3's per-linear signs).
- Sharing the factorization saves 3-50 ms a group: not done, for the same reason.
- Reading the tile states back once at the end instead of after each row of tiles: no change (the sync was not the
  cost), kept since it costs nothing.
- Packing the tiles in parallel (rayon): 17408 x 5120 refit+pack 1.72 s -> 0.57 s. Kept; bit-identical.

On the streamed path, q/k/v and gate/up (dense, shared expert, stacked experts) read one input, so the first of each
collects the Hessian and the rest reuse it: 7 x^T x accumulations a dense layer become 3. And the host load now builds
no decoder layers at all (`inference_nn::decoder::stream_layers` scope; the forward refuses to run a stack loaded that
way): only the embedding, final norm and head load on the host.

Qwen3-0.6B, as Run 15: 164 s (from 173), 6.8 GB peak RSS (from 7.9), every tensor bit-identical to Run 15's. A tiny
OLMoE written streamed matches its resident write in every tensor but the head. At this size a layer is 4.5 s: ~2 s
encoding, the rest the two passes over 474 chunks (per-chunk overhead more than compute at 0.6B).

## Run 17 - 2026-11-14 16:30

The batching question again, for MoE: Run 16 timed dense shapes only. The quantizer by tile count (one launch,
mul1):

```
tiles    8      16     32     48     64     96     128    256    512    1024   4096
4 bit    1.34   1.29   1.35   1.32   1.38   2.63   2.65   5.24   9.33   17.7   68.4  ms
2 bit    1.99   2.08   2.25   2.43   2.61   4.71   4.70   8.42   14.5   27.0   104   ms
```

It runs in waves of 64 tiles: a row of tiles narrower than a wave costs the wave. Dense Qwen3 widths are multiples of
1024 (whole waves), so Run 16's conclusion holds for them; MoE experts are not: Qwen3-30B-A3B's gate/up rows are 48
tiles (768 wide). So experts now encode together (`encode::encode_experts`): each prepared alone (padding, damping,
factor, regularization, scale search), their LDLQs in lockstep with batched matmuls (L stacked `[e, k, k]`), one
quantizer launch per row of tiles for all of them, a batch capped at 2 GiB of LDLQ state. Bit-identical to encoding
each alone (unit test: every trellis word and the proxy error). And the scale search quantizes all ten coarse
scales in one launch and the five fine ones in another (15 launches -> 2), for every linear.

128 random experts, 4.0 mul1:

```
shape        alone     together
768 x 2048   32.9 s    25.1 s   (LDLQ 175 -> 110 ms an expert; the scale search ~61 ms an expert either way: it
2048 x 768   26.6 s    23.6 s    quantizes ~3600 sampled tiles, already whole waves)
```

The first OLMoE run of this ran out of device memory preparing the experts: each prepared expert held ~1 GB. Cause:
`Tensor::to_device` moved a view's whole storage (keeping the view's layout into it), so one expert's Hessian
`h.get(e)` of the host stack `[64, 2048, 2048]` f32 moved all 64 (1.07 GB). The one-expert-at-a-time path of Runs
14-16 did the same, freeing it each time: ~1 GB over PCIe per expert, 64 GB per projection per layer. Now a view
whose storage holds more elements than the view moves only its own (`force_contiguous` first; a broadcast view, which
holds fewer, still moves as it is).

OLMoE-1B-7B-0125-Instruct at 4.0 mul1, the same 485K tokens, streamed with all of the above: 1275 s against Run 14's
2408 s resident; 78 s a layer; device memory peaking at 7.9 GB (2.1 of it other processes) against 23.8. Perplexity
by our example, both measured now (512-token windows):

```
text                          resident (Run 14)   streamed
this repo's README.md         15.8509             15.7664
pipeline/isq_flow/plan.rs     5.7292              5.6960
```

(Run 14's README figure was another revision of the file.)

## Run 18 - 2026-11-14 18:00

Qwen3-14B (bf16, 29.5 GB: larger than the 3090's 24 GB), 4.0 mul1, streamed, the same 485K tokens. First attempt:
40 layers at 72-73 s (device memory peaking at 13.6 GB with 2.1 of other processes), then out of device memory
encoding the head. A layer: ~23 s collecting its Hessians (the pass over the states plus the x^T x accumulations,
~13 s of it, most down_proj's 17408^2), ~40 s encoding (gate 8.6, up 11.4, down 12.7, attention ~7), ~9.5 s
running the states on.

GPU utilization sampled every 0.5 s across a layer: 77% on average, and idle (0%) for 4-5 s after each MLP
projection's encoding, three times a layer. That was the swap's reconstruction: `dense_weight` decodes the trellis on
the host, single-threaded (89M weights for a 17408 x 5120 linear). The encoder already holds the decode on the GPU:
its LDLQ values rotated back with the stored scales (the refit's row and column scalings commute with the rotations,
so only the scales' f16 rounding separates it from the refit `q`). `Exl3Encoding::decoded` now carries it, and the
swap uses it; it matches `dense_weight` to 1e-5 of the weight's scale (asserted in the encoder's test).

The head (151936 x 5120, 3.1 GB a f32 copy): the encoder held the weight as given, its regularized copy and a
stacked copy, LDLQ's feedback, its rows and their join, and then the refit's `H W`, `H q` and products, all at full
size. Now the regularized weight is handed to LDLQ and dropped before its values join, LDLQ runs a chunk of 8192
columns at a time (columns are independent given L; a test pins the chunked run to one over all columns), and the
refit and output error take `H W`, `H q` and their reductions a chunk at a time. A head-sized encode at 6 bits: 82 s,
~17 GB of device memory (random weights, a random Hessian).

The rerun with both: 2518 s for the model (40 layers at 59.5-60.7 s, from 72-73; the head 90 s at 6 bits), device
memory peaking at 18.6 GB (2.1 of other processes; the head's encode is the peak), 8.2 GB written. Perplexity on the
first 200 KB of wikitext-2 test (our example, 512-token windows; bf16 does not fit the device, so Q8_0 ISQ stands in):

```
Qwen3-14B Q8_0 (ISQ)     10.0032
EXL3 4.0 mul1, ours      10.0635  (+0.6%)
```

Left open: the Hessian x^T x accumulation (~13 s of a 14B layer); the copy of unquantized tensors loads whole shards
(host RSS reached 42 GB); and a host-memory fallback for a layer or head whose encode does not fit the device (14B's
are not the largest a model will have): today device OOM ends the run, where it could retry the linear with smaller
column chunks or with its large intermediates (Hessian, LDLQ feedback, refit products) kept in system RAM.

## Run 19 - 2026-11-14 02:55

A host-memory fallback for encodes that do not fit the device. Where an encode's memory goes: the k x k tensors
(H, H damped, L, the factor's temporaries) and the k x n ones (the weight as given, its regularized copy, LDLQ's values,
the rotated-back `q`, the decode). A head grows the second kind (n = vocabulary), a wide MLP's down_proj the first.
The k x n work was already column-separable (LDLQ given L, the refit's sums, the output error), so the fallback parks
it on the host: the weight's regularization and the rotations run on the CPU (128-wide Hadamards are cheap there), H
and H damped wait on the host while L is used, and LDLQ, the refit and the output error take a chunk of 8192 columns
to the device at a time. L stays on the device. The encoder tries the device first; on an allocation failure a batch
of experts splits in half, and a linear alone retries parked.

First run, Qwen3-0.6B 4.0 mul1 with a second process holding all but 2.5 GiB of the 3090: the layers fit, the head
(151936 x 1024) ran out, retried, and then failed with `cuSOLVER could not create: CUSOLVER_STATUS_INTERNAL_ERROR`.
The encoder made a cuSOLVER handle a call; creating one allocates outside the memory pool, and the pool kept the
failed attempt's blocks. Each thread now keeps one handle a device.

Rerun, and the same conversion without the other process:

```
                       head         head proxy error   wiki PPL (200 KB)
on the device          20.7 s       0.000237           25.470
from host memory       24.4 s*      0.000237           25.497
* including the failed attempt on the device
```

All 899 layer tensors are bit-identical between the two; the head's differ (svh 6% equal, trellis 56%): the host's
rounding of the scales moves the quantizer's choices, not the error. A unit test pins a parked encode against one on
the device (proxy error within 0.1%, decodes within 5% of the weights' scale); an ignored one fills the device and
encodes a 32768 x 4096 linear that no longer fits (17.7 s, passes).

Also: every streamed run since #954 ended with `prompt step - Model failed ... holds none to run`, the engine's
warm-up request on a model whose layers were dropped. A streamed EXL3 load now skips it as a UQFF write does
(`loaded_only_to_write`).

Not covered: an allocation failure outside the encoder (building a layer on the device, its Hessians' x^T x, its
forward) still ends the run. A layer that does not fit at all would need its forward on the host.

## Run 20 - 2026-11-14 03:45

How often an encode outgrows the device. Per streamed layer the device holds the layer in bf16, its Hessians in
f32, and one encode. From the configs: dense up to 14B fits (measured); 70B-class (Llama 3 70B, Qwen2.5-72B, Mistral
Large: 28-29K MLP widths, a down_proj Hessian of 3.3 GB) was the open question; 405B-class (53248, 11.3 GB a Hessian)
and the 256-384-expert MoEs (DeepSeek V3, Kimi K2, Llama 4 Maverick: 22-34 GB a layer in bf16) cannot fit as built.

An ignored test encodes a 70B down_proj (8192 x 28672, bf16, full-rank Hessian). On an otherwise empty 3090 it ran
out of device memory 5.6 s in, the Run 19 fallback included: that only parks the k x n tensors, and this linear is
k x k bound. Counting from the code, about eight k x k were alive at once: the Hessian and its damped copy, the
rotation's steps (shadowed `let`s and chained temporaries live to the end of their statement; `had_left` copied the
transpose twice), cuSOLVER's copy, the block LDL's three chained copies; and later the refit's H, H damped, q q^T and
its product.

Changes, each measured by the pool's live high-water mark (CU_MEMPOOL_ATTR_USED_MEM_HIGH, reset per phase; nvidia-smi
shows reservations only), GiB:

```
                     rotate  factor  scale  LDLQ   refit
first fit             10.51   11.38  10.08  12.28  14.44
in-place products     10.51    8.75  10.08  12.28  11.81
H on host after LDL   10.51    8.75  10.08   9.22   8.75
H dropped for rotate   7.45    5.69   7.01   9.22   8.75
```

- "first fit": H damped rebuilt from H where used (not kept); the rotation as reassignments; `had_left` as one batched
  product over the 128-row blocks; cuSOLVER factoring the rotation in its own buffer; a retry re-rotates from H with
  the extra damping (same values as damping the previous try); q q^T as one product where q is on the device.
- "in-place products": the block LDL and (q q^T) o H damped rewrite their rows in place (`slice_set`), 4096 at a time.
- "H on host after LDL": H waits on the host from the factorization to the refit (LDLQ needs only L); the streamed
  writer holds a layer's Hessians on the host too, each moved to the device for its own linears.
- "H dropped for rotate": the device copy of H goes once the damped copy exists; a retry uploads it again.

The down_proj now encodes in 30.4 s at 9.22 GiB live (10.56 reserved), proxy error 0.0025393977923281012 at every
step (the changes kept it bit-identical). The test asserts under 12 GiB: beside a 70B layer (1.7 GB bf16) and its
activations, that fits a 24 GB device.

Qwen3-0.6B 4.0 mul1 against Run 19's: 900 of 902 tensors bit-identical; lm_head's suh and svh differ in 0.1% of
values (q q^T is one product now where the chunked sum was), its trellis identical. Wiki PPL 25.469 (was 25.470),
136.7 s (was 135.3).

Still not covered: 405B-class dense (one Hessian is half the device; the factorization needs two) and layers too large
to build on the device (the big MoEs).
