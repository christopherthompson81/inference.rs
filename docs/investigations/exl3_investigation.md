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

## Run 21 - 2026-11-14 05:10

The two left-over costs from Run 18: Hessian accumulation (~13 s of a 14B layer) and the host memory of a streamed
run (42 GB peak RSS on 14B).

Hessians: each 1024-row chunk added `x^T x` as an f32 matmul of the bf16 activations cast up, then an add over the
k x k sum. bf16 products are exact in f32, so one cuBLAS GemmEx over the bf16 rows (CUBLAS_COMPUTE_32F, f32 output,
beta = 1 into the running sum) is the same product but for the order of its sums, on the tensor cores and with no
temporary. Bench at 14B's down_proj width (17408, 1024 rows): 28.62 ms a chunk before, 9.69 after (~64 TFLOPS, near
the 3090's 71 for bf16 with f32 accumulation); 474 chunks 13.6 s -> 4.6 s; after 100 chunks the two sums differ by
6.5e-6 of the diagonal. Routed (expert) Hessians take the same path.

Qwen3-0.6B, 4.0 mul1: layers 4.0 -> 3.4 s, 137 -> 121 s. Its perplexity moved 25.470 -> 26.034, so noise was
measured: the same conversion with only the summation order changed (each chunk's product in two halves), once each
way:

```
Hessian path                   wiki PPL
f32 (as before)                25.470
f32, sums split in two         25.321
bf16 GEMM                      26.034
bf16 GEMM, chunks split        25.540
```

Equally good encodings of the 0.6B spread ~3% at 4 bits (every linear's trellis choices move with its Hessian's
rounding); per-linear proxy errors are unchanged (ratio 0.998 mean, 0.969-1.025). A probe in the run itself put the
two Hessians within 1e-5 of their scale. So 26.03 is a draw, not a defect; the 14B decides.

Unquantized copy: `copy_unquantized` loaded every shard whole (the encoded weights are most of it) and kept the few
others; it now maps the shard and reads only those.

Qwen3-14B 4.0 mul1, the same 485K tokens:

```
                  total     a layer       peak RSS (anon / file-backed)   wiki PPL
Run 18            2518 s    59.5-60.7 s   42.1 GB (not split)             10.0635
now               2108 s    49.9-50.7 s   41.0 GB (11.7 / 29.4)           10.0405   (Q8_0 10.0032)
```

The RSS was mostly the mapped checkpoint (29.4 GB file-backed: the whole 29.5 GB model, touched as each layer is
built), which the kernel reclaims under pressure; the process's own memory peaks at 11.7 GB (the calibration states
are 5 GB of it). Not a problem to chase.

## Run 22 - 2026-11-14 08:40

Decode headroom on Qwen3-30B-A3B-exl3 3.0bpw (3INST). Bandwidth floor: ~1.25 GB of trellis read a token (attention
3.9 + 3.1 MB, experts 9.4 + 4.7 MB a layer, 48 layers, plus a 233 MB 6-bit head), ~1.3 ms at 936 GB/s; we run 7 ms.
Where it goes: `prof2.sh` (serve, one warm-up request, nsys over a second 100-token poem request,
`--cuda-graph-trace=node`), 94 decode steps between head launches.

```
a step: 7.68 ms wall under nsys (6.9 without), 7.37 ms of kernels, 1208 kernels; gaps 0.31 ms (4%)

                                 us/token   each      exllamav3 1.6.6 (ref_moe trace, Run 7), us/token
exl3_moe_coop_a (gate+up+silu)   1890       39.4      1946
exl3_gemm (o_proj x48 + head)    1315       18.5/428  1326 + 278 head
exl3_mgemm (q/k/v, sliced)        954       19.9       969
exl3_moe_coop_b (down)            855       17.8       898
attention (mma + stream-k fixup)  811       15.3+1.6   848 (split + combine)
casts (f32->bf16 x5, bf16->f16 x3, f32->f16, u32->i64 a layer)   592
rms_norm x2, badd x2, router topk + gemv, qk norm/rope, cache write, slice ptrs   ~840
```

So GPU-bound: the step is its kernels. The four EXL3 matmuls are 5.0 ms (65%), the same kernels and times as
exllamav3's; glue is ~1.4 ms (20%; exllamav3 has its own ~0.9); attention 0.8 ms.

The matmuls under ncu (base clocks, one layer of a decode step):

```
                 grid x block   DRAM % peak   issue busy   occupancy (theor / achieved)   limit
moe_coop_a       96 x 512       24            35%          67 / 39%   (0.59 waves)         regs, smem
moe_coop_b       128 x 512      26            40%          67 / 49%   (0.78 waves)         regs
mgemm q/k/v      80 x 512       20            41%          33 / 33%                        92 KB smem, 128 regs
gemm o_proj      82 x 512       16            39%          33 / 33%                        92 KB smem, 114 regs
```

Neither DRAM (16-26%) nor issue (35-41%) is near its limit: latency-bound, one or two 16-warp blocks an SM with too
few loads in flight. ALU pipe ~50% of active cycles (the trellis decode), tensor pipe 5-9% (m padded to 16 rows).
The autotuner's shape/grid choices are already applied, so the headroom is in a decode-specific kernel (more
warps and loads in flight per SM at m = 1), not in knobs. At 60% of DRAM bandwidth the matmuls would be ~2 ms.

Ranked:
1. matmul kernels at m = 1: ~3 ms of 7 (new kernel work, the most uncertain)
2. glue fusion (casts into the EXL3 launches' input/output, residual add + norm, the router's gemv + topk + index
   cast): ~1 ms, our own code
3. attention at short context: 0.8 ms for ~250 keys a step (latency of the stream-k mma path), maybe 0.4 ms

## Run 23 - 2026-11-14 09:40

Glue, step one. Every bf16 <-> f16/f32 cast around the EXL3 launches (ten a layer: the q/k/v and expert inputs,
the q, k, v and expert outputs, the expert picks and weights, o_proj's input and output) cost 1.2 us each in the
trace. An f16 model (no casts) decoded at 154 against 143, so they were worth up to ~8%.

The kernels now read and write bf16 themselves where it matters most:
- fused MoE (`MoeCoopParams`): bf16 x, u32 picks, f32 routing weights read in place, bf16 output
- sliced q/k/v mgemm: bf16 A (`a_bf16`, the input Hadamard loads bf16 and rounds to f16) and bf16 outputs through
  a second pointer list (`C_bf_list`; the f32 rows stay the kernels' scratch, `had_fb_r_128_inner` rounds the
  rotated result)

The rounding matches the casts (bf16 -> f32 -> f16 in, f32 -> bf16 out), so both paths are bit-exact against the cast
route in the GPU tests; only the routing weights differ, now f32 where the cast route rounded them to f16. Left as
casts: o_proj and the head (`exl3_gemm` shares its argument list with the GEMV and int8 GEMV paths it may hand off
to), 2 a layer.

Then the pre-FFN residual add and norm as one launch (`cuda_add_rms_norm`, already used by Qwen3.5) in the shared
decoder's pre-norm layer: badd + rms_norm -> one kernel, for every model on that path.

Decode, Qwen3-30B-A3B-exl3 3.0bpw, poem prompt (`pwr.sh`: 3 x 400-token requests after a warm-up, nvidia-smi at
200 ms). The 3090 runs at its power cap here (~341 W, throttle reason SwPowerCap, 1950-1965 MHz), and readings drift
~3% between sessions (f16 read 154 earlier and 150 later), so the paths ran in alternating pairs off one binary (an
env toggle, since removed):

```
                                       tok/s (pairs)              kernels a step   busy a step (nsys)
casts (master)                         138.7  139.2  138.5        1208             7.37 ms
bf16 in place                          144.4  144.2  144.0         824             6.96 ms
casts + fused pre-FFN norm             140.2  140.1
bf16 in place + fused pre-FFN norm     148.0  147.7                                         (+6.6% on master)
```

Decode accuracy (teacher-forced, `dppl2.sh`, as Run 8): casts 4.74172 (Run 8's), bf16 + fused norm 4.74186; mean
|d logprob| 0.0056, top-1 agreement 99.8%.

Size: the EXL3 kernel library grows 7.29 -> 7.71 MiB (every mgemm instance carries both input loads and the bf16
output stage); baseline updated.

Next: the matmul kernels themselves (Run 22's first item).

## Run 24 - 2026-11-14 11:30

exllamav3's small-m GEMV (`exl3_gemv_kernel`, QTIP-style: warps split k with no block barriers, B streamed to
registers) runs only 4 bpw for the 3INST codebook (`K != 4 && cb == 0` refuses the rest, and there is no 3-bit
3INST instance); its comment says 3 bpw was measured only for mcg and mul1. Admitted 3INST at 3 bpw (one instance
grid, the same shape heuristic as mcg's 3 bpw: the narrow config when its grid is one co-resident wave). On the
30B-A3B that takes o_proj (4096 -> 2048) at decode; q/k/v stay in the sliced mgemm, the head (6 bpw) in the GEMM.

```
o_proj at m = 1                 18.5 us (exl3_gemm) -> 14.3 us (exl3_gemv)
decode tok/s, pairs             EXL3_GEMV=0: 145.0 145.4    default: 153.3 153.5     (+5.8%)
```

q/k/v as three GEMVs in place of the one sliced mgemm (a temporary toggle): 142 against 155 (fused). The mgemm
stays.

Decode accuracy (teacher-forced): EXL3_GEMV=0 4.74186, GEMV 4.75072 (+0.19%); mean |d logprob| 0.0077, top-1
agreement 99.8%: the GEMV accumulates in f16 and folds to f32 on a cadence, a different rounding order, within the
spread earlier kernel swaps showed (Run 8: 0.4%). The kernel test's 3INST 48-word case at m = 1 and 2 now runs the
GEMV, inside its f16 bound.

Readings drift: the fused-q/k/v runs read 155 then 164 in one sitting (throttle 0x404, power and thermal); only
alternating pairs are comparable.

The fused MoE's own geometry knobs, alternating with the default (tok/s, 3 requests each):

```
default                      163.0 165.2 165.0 | 166.3 165.1 160.3 | 164.2 163.5 162.9 | 165.2 166.1 166.3
EXL3_MOE_COOP_WIDE=0         160.3 164.9 165.3   (32-column tiles: 384 blocks a launch in place of 96)
EXL3_MOE_COOP_KSPLIT=2       164.4 166.0 154.9
EXL3_MOE_COOP_KSPLIT=4       155.4 155.9 158.0
```

None beats the default (wide tiles, no split, exllamav3's choice below Blackwell): more blocks do not buy
throughput there, so the coop kernels' 24-26% of DRAM bandwidth is their inner loop (3INST decode plus the m16 MMA
at one real row), not the grid. What remains in the matmuls needs a kernel of our own.

## Run 25 - 2026-11-14 14:30

A decode GEMV of our own (`exl3_dgemv_instance_exl3.cu`). Why: `decode_matmul_times_by_shape` (an ignored
benchmark, kernel times from nsys) showed exllamav3's GEMV paying a fixed ~8.5 us a launch (2048 x 512, 0.4 MB:
9.0 us; 2048 x 1536: 9.5 us; 4096 x 2048: 15.0 us): a cooperative launch, two grid syncs around the input and output
Hadamards, and a serial k loop per warp.

Design: no grid sync. A block per (128-column group, k-split); each warp one 16-column tile, its 8 or 16 trellis
words a lane issued before anything else while the block rotates its rows of x (`* suh`, Hadamard) into shared
memory; exllamav3's register window extraction and codebook decode (`dq8_regs_{2,3,4}bits`), one m16n8k16 MMA pair a
slice with f32 accumulation (exllamav3's GEMV accumulates in f16). Partials go to a row per split in a per-stream
workspace (16 MB, made with the first layer on the stream, outside any capture); the last block of a column group sums
them in split order, so the result is the same every run (an atomicAdd version was ~0.3 us faster at small shapes and
not reproducible), then rotates them out and resets its counter. Up to four matrices of one k in a launch (q, k, v),
2/3/4 bpw, every codebook; mul1 at <= 5 bpw keeps the int8 GEMV.

Rows a block: 128 below 16 blocks an SM, else 256 (512 never won):

```
k x n           MB     exllamav3          ours (KR 128 / 256 / 512)
2048 x 512      0.4    8.9  (gemv)        4.2 / 5.2 / 7.2
2048 x 1536     1.2    9.5  (gemv)        6.2 / 7.2 / 7.6
4096 x 2048     3.1    14.9 (gemv)        10.7 / 11.5 / 13.1
4096 x 8192     12.6   ~58  (gemm)        30.7 / 30.1 / 31.4
8192 x 8192     25.2   ~58  (gemm)        58.1 / 54.7 / 54.5
```

ncu at 8192 x 8192: DRAM 51%, issue slots 73% busy, stalls mostly math-pipe throttle and not-selected: the trellis
decode's instruction count is the limit there now, not latency.

In the model (Qwen3-30B-A3B 3.0bpw): o_proj alone moved decode 165.5 -> 166.4 (pairs; clocks fell 1942 -> 1920 MHz
at the same ~348 W, the 370 W cap's SwPowerCap active). With q/k/v as one three-job launch in place of the sliced
mgemm:

```
                         tok/s (pairs)       step under nsys    kernels a step
exllamav3 kernels        166.5 166.1          7.13 ms            776
ours (q/k/v, o_proj)     198.8 198.6          6.01 ms            632
```

q/k/v + o_proj: 1.67 -> 1.07 ms a token; the fill-pointers kernel and the casts around them are gone too. Decode
accuracy (teacher-forced): 4.74874 (Run 24's GEMV 4.75072, the mgemm/gemm 4.74186). Greedy output reads as before.

Next: the MoE experts the same way (stand-ins of their weight volume: 2048 x 12288 in 24.3 us against coop_a's 41,
768 x 16384 in 14.6 against coop_b's 20.7).

## Run 26 - 2026-11-14 16:00

The MoE experts for one token on the same blocks (`moe_a_kernel`, `moe_b_kernel`): A over (slot, gate/up, column
group, k-split), its last block of each slot's column group summing both projections' partials, rotating them out,
applying silu(gate) * up and rotating that (* down's suh, Hadamard) into the down projection's f16 input; B over
(slot, column group, k-split), its last block of a slot's group rotating out and weighting by the routing weight, the
last slot of a column group summing the slots in order into the token's row. Experts come from `sel` and the
projections' address tables on the device, so decode graphs capture it. More than one token keeps exllamav3's coop
kernels. The fused-MoE tests' one-token cases run these kernels (checked under nsys), the five-token ones coop.

```
Qwen3-30B-A3B 3.0bpw       tok/s (pairs)                      gate/up     down      step (nsys)
exllamav3 coop             179.3 195.0 200.9 | 195.3 180.0 200.0   40.0 us     18.2 us   6.01 ms
ours                       218.1 213.2 216.8 | 225.6 215.5 208.6   27.3 us     15.8 us   5.16 ms
```

Readings spread more now (clocks 1875-1920 MHz at the same ~352 W; the faster kernels draw more of the cap). Decode
accuracy (teacher-forced) 4.74964 (Run 25: 4.74874).

Per token now: gate/up 1.31 ms, q/k/v + o_proj 1.13, down 0.76, attention 0.64, the 6 bpw head (exllamav3's GEMM,
233 MB at ~525 GB/s) 0.44, glue ~0.8.

## Run 27 - 2026-11-14 16:45

Past 4 bpw (heads, mostly) a tile is more words than a warp has lanes: the decode GEMV now loads them in two passes
and decodes through shared memory with exllamav3's generic `dq_dispatch` (5, 6 and 8 bpw, 128 rows a block, the single
GEMV only). The 30B's 6 bpw head (2048 x 151936, 233 MB): exllamav3's GEMM 470.5 us, ours 325.5 us (716 GB/s). Step
under nsys 5.16 -> 5.03 ms. Decode accuracy (teacher-forced) 4.75024.

Decode in this session read 247-249 tok/s, but sessions drift (Run 26's pairs read ~216 for the same kernels bar the
head): nsys step times are the comparable measure here. Another `inference` process (a server from the user's
serve.toml, idle) shared the GPU from 10:22.

Qwen3-0.6B-exl3 4.0bpw (every linear on the new kernels at decode), teacher-forced decode: exllamav3's kernels 6.96042,
ours 6.96547; mean |d logprob| 0.0049, top-1 agreement 99.7%.

Since Run 22 (30B-A3B, step under nsys): 7.68 -> 5.03 ms.

Size: the EXL3 kernel library 7.71 -> 8.01 MiB (the decode GEMV at 2-8 bpw and the two MoE kernels at 2-4 bpw, each
codebook); baseline updated.

## Run 28 - 2026-11-14 12:30

Qwen3.6-35B-A3B next (the user's model; BF16 safetensors, `Qwen/Qwen3.6-35B-A3B`, 72 GB, downloading). Our converter
streamed only the shared decoder: Qwen3.6 is the bespoke Qwen3.5 stack (Gated DeltaNet + attention + MoE, loaded as
`Qwen3_5MoeForConditionalGeneration`), and the multimodal pipeline takes no EXL3 write. The user chose both routes:
exllamav3's converter first (it has `qwen3_5` and `qwen3_5_mtp`), ours alongside.

Ours:
- the Qwen3.5 stack builds a layer through one `LayerBuild` (the constructor's loop and a rebuild on any device
  share it) and implements `LayerStream` for plain stacks (RMS norms, no PLE, no QSA); a host load under
  `stream_layers` builds no layers. A streamed attention layer runs on paged attention from its prompt mask (the
  plain path needs a KV cache); a GDN layer from a fresh `GdnLayerCache` as a prefill.
- an EXL3 write of a multimodal checkpoint loads its text model through the text loader named by
  `text_config.model_type`, and the text loader reads a nested `text_config`.

Qwen3.5-0.8B (dense hybrid) streamed at 4 bpw: every proxy error NaN past layer 0. Bisected by skipping swaps: only
`linear_attn.in_proj_qkv`'s reconstruction broke the layer, with 14 NaN rows of 1024. Those rows are dead channels
(every value 1.18e-37, the same rows of q and k): their rms underflows, so the encoder gives them a zero output
scale, and the reconstruction's f16-rounding ratio `svh / sv` was 0 / 0. The stored trellis and scales were right
(svh 0 decodes to 0); only the reconstruction the converter swaps in for the next layer's inputs was poisoned, so
any model with dead channels was affected, streamed or not. The ratio is now 0 there (`dead_channels_reconstruct_as_zeros`).

Also: a streamed (layerless) Qwen3.5 model panicked at serving setup reserving GDN deferred state (indexing empty
layers); a layerless model reserves none.

Result, 200 KB calibration slice, teacher-forced decode (README): BF16 5.50146, our streamed EXL3 4.0 5.72816
(+4.1%; exllamav3's own Qwen3-0.6B 4.0 was +3.8% in Run 9). The output (multimodal config, EXL3 text weights, vision
tower copied unquantized) loads and decodes through our engine. `--resident` fails on hybrid models ("Hybrid
recurrent metadata is required"): the resident calibration drive never carried recurrent state, so hybrids convert
streamed only.

## Run 29 - 2026-11-14 14:55

exllamav3 1.6.6's conversion of Qwen3.6-35B-A3B at `-b 4.0` (defaults: head 6 bpw, mul1 codebook): 2h 15m, 18.07
GiB of tensors. Of that, routed experts 15.15 GiB, embeddings 0.95, attention/GDN/shared experts 0.71, head 0.36,
MTP 0.40, vision tower 0.52 (6 bpw); text without vision or MTP 17.16 GiB (the IQ4_KSS recipe file is 18.15 GiB
with its MTP block).

Loads and decodes through our engine as is: the multimodal pipeline, vision tower dequantized to dense at load (81
linears through `dense_linear_in`), routed experts through our MoE kernels.

KLD against the recipe's BF16 base (`perplexity --kld-base bf16-wiki-c512-ch40.kld`, wikitext-2, 40 x 512):

| | PPL (base 5.8237) | KLD mean | KLD p99 | same top-1 |
|---|---|---|---|---|
| exllamav3 EXL3 4.0 | 5.8694 | 0.02146 | 0.198 | 93.81% |
| IQ4_KSS recipe (ik, from the recipe log) | | 0.0216 | | |

Speed, `inference bench` 512-token prompt / 128-token decode, alternating in one session:

| | prefill tok/s | decode tok/s |
|---|---|---|
| EXL3 4.0 | 1618, 1618 | 147.3, 147.4 |
| IQ4_KSS GGUF | 4234, 4216 | 164.4, 163.3 |

A first request after the conversion took 64 s and the first KLD chunk 51.7 s. Neither came back: a rerun of the
server answered in 0.39 s, and a KLD rerun's first chunk took 1.4 s. The KLD run had been reading its 5 GB base file
cold. Profiling also showed the dense vision-tower dequantize runs at load, not in the first pass.

So EXL3 matches the GGUF recipe's quality in 1 GiB less memory, but decodes 10% slower and prefills 2.6x slower.
Next: profile decode per kernel against the GGUF run.

## Run 30 - 2026-11-14 15:05

Where Run 29's decode loses, nsys over a 100-token request (kernel time a token, 7.05 ms in all):
`exl3_gemv_int8_sq_kernel` 2.78 ms in 247 launches (11.3 us each), and the casts around it (`cast_f32_bf16`,
`cast_bf16_f16`, ~250 apiece) 0.65 ms. Every GDN, attention and shared-expert linear is mul1 at 4 bpw, and mul1
at <= 5 bpw still kept exllamav3's int8 GEMV. That gate predates our decode GEMV (Run 25), which decodes mul1 too;
Qwen3-30B-A3B is 3INST, so it never met the gate.

A/B with the gate overridden (temporary env, `inference bench` 512 / 128, alternating):

```
                       decode tok/s       teacher-forced decode PPL (README, 576 tokens)
int8 GEMV (as was)     147.6 147.4        3.39687
our GEMV               189.5 188.7        3.40265
```

The PPL moves by +0.17%, about what Run 24 saw between kernels on this short text. Our GEMV keeps f16
activations where the int8 one quantized them. Now one row always takes our GEMV (fused q/k/v included); the int8
gate holds only for 2-16 rows. The GPU test's int8 bound covers only m = 2.

The GDN's `in_proj_qkv` and `in_proj_z` share an input but ran as two launches: they now go through `fused_group`
(one two-job GEMV at one row) as gate/up and q/k/v do: 188.4 -> 192.6 tok/s. Kernel time a token is now 5.52 ms
(before the qkv/z fusion): our GEMV 1.83 ms (188 launches), moe_a 0.76, moe_b 0.50, the head 0.49.

Decode, then: EXL3 4.0 192.6 tok/s against IQ4_KSS's 164 (both without MTP), at KLD 0.0215 against 0.0216.

Prefill is the gap now (1628 against 4234 tok/s). One 512-token forward under nsys: a 520 ms span with 340 ms of
kernels. The MoE expert `exl3_mgemm` takes 126 ms (120 launches, 1.05 ms each), the dense `exl3_gemm` 77 ms (251,
305 us), index selects 53 ms (`is_u32_f16` 120 x 352 us, `is_u32_f32` 150 x 71 us: the expert grouping's row
gathers), copies and casts ~40 ms. The other ~180 ms is idle, while the host reads the router's picks to group rows
by expert.

## Run 31 - 2026-11-14 17:10

Our streamed converter on Qwen3.6-35B-A3B, exllamav3's settings (`inference quantize exl3 --bits 4.0`, head 6, mul1,
exllamav3's standard calibration text, 485K tokens in 473 chunks). A first try encoded the GDN's `in_proj_b` and
`in_proj_a` (2048 -> 32, padded to 128). exllamav3 keeps them in BF16 (`qmap = None` for the GDN's b/a, split or
fused), and a padded layer misses our decode GEMV. They now stay in full precision by name, as the router does. A
width rule (narrower than 128 kept) broke the tiny OLMoE test, whose 64-wide k/v are meant to encode padded.

1 h 50 min (158-163 s a layer), against exllamav3's 2 h 15 min. Text weights 17.155 GiB, the same as exllamav3's to
the MiB. Ours copies the MTP block (1.57 GiB) and the vision tower (0.83) in BF16, where exllamav3 encodes them at
the head rate (0.40, 0.52).

```
wikitext-2 40 x 512 against the BF16 base   PPL (base 5.8237)   KLD mean   KLD p99   same top-1
exllamav3 4.0                               5.8694              0.02146    0.198     93.81%
ours 4.0                                    5.8707              0.02210    0.217     93.57%
IQ4_KSS (recipe)                                                0.0216

inference bench 512 / 128, alternating      prefill tok/s       decode tok/s
exllamav3 4.0                               1646, 1645          197.1, 197.0
ours 4.0                                    1648, 1642          196.3, 196.4
```

Ours is 3% behind exllamav3 on KLD and level on speed. The 3% is not chased here: the calibration chunking
differs (exllamav3 runs its rows at its own lengths), and Run 9 measured the same order of gap on Qwen3-0.6B.
Not done: MTP and the vision tower are left in BF16 rather than encoded.

## Run 32 - 2026-11-14 18:25

Does exllamav3's all-experts calibration explain Run 31's KLD gap (ours 0.02210, exllamav3 0.02146)? exllamav3
(`capture_pass_experts`) runs every calibration row through every expert for the Hessians. Ours uses only the
rows the router sends each one: with top-8 of 256, about 15K of 512K tokens per expert on average, and far fewer
for rare experts.

The quick test used temporary switches (not committed; the patch is kept in the scratchpad). `TMP_ALL_EXPERTS`
feeds each expert's gate/up Hessian every token, and its down Hessian its own activations on every token.
`TMP_STOP_LAYER=10` encodes layers 0-9 only. The ten layers were spliced into a symlinked copy of our Run 31
output, and the shards' tensor names and shapes checked equal. A control re-encoded the same layers with routed
Hessians and only the encoder's per-tensor seed changed (`TMP_SEED_SALT`).

```
wikitext-2 40 x 512 vs the BF16 base       KLD mean   KLD p99   same top-1   PPL (base 5.8237)   s a layer
ours (Run 31)                              0.02210    0.217     93.57%       5.8707              160
layers 0-9 all-experts                     0.02197    0.209     93.75%       5.8896              213-230
layers 0-9 reseeded only (control)         0.02200    0.222     93.77%       5.8741              159-166
exllamav3                                  0.02146    0.198     93.81%       5.8694
```

Reseeding alone moves the mean KLD by 0.0001, the same as all-experts calibration. The all-experts p99 is lower
(0.209 against the control's 0.222), but the control moved p99 by 0.005 itself. So on ten layers, the effect is not
distinguishable from encoding noise. It is at most a small part of a 0.0006 gap. The proxy errors of the two runs
cannot be compared: each grades against its own Hessian, and the routed run's few-token Hessians flatter rare
experts.

Not ruled out: calibration layout (exllamav3 runs 250 rows of 2048 tokens, ours 473 chunks of about 1024), the
bf16 Hessian GEMM (#957), and all-experts calibration across all 40 layers, where a small per-layer effect could
add up.
The ~60 s a layer the test added is mostly the hack: 512 identical gate/up Hessians per chunk, where one shared
Hessian (the shared expert's) would serve every expert, and a per-expert gather loop for the down Hessians.

## Run 33 - 2026-11-14 19:40

Run 32's gap, on Qwen3.5-0.8B: the same GDN + attention stack without experts, minutes a conversion, several
seeds a setting. The base came from a BF16 GGUF (llama.cpp's `convert_hf_to_gguf.py`), with `llama-perplexity
--kl-divergence-base`, wikitext-2 40 x 512 (BF16 PPL 16.60). Every conversion is 4.0 mul1, head 6, scored by our
`perplexity --kld-base`.
Temporary switches (not committed; the patch is kept in the scratchpad):
- `TMP_SEED_SALT`: the per-tensor encoder seed.
- `TMP_CAL_CHUNK`: the calibration chunk length.
- `TMP_F16`: the calibration forward in f16. f32 can't run: the GDN's CUDA recurrence takes f16/bf16 only.
- `TMP_CAL_TOKENS`: calibration token ids read from a file.

exllamav3's seed, through a wrapper offsetting `make_quant_args`'s per-tensor seed.

```
KLD mean (3 seeds)                                   each                          mean     min. s
ours, our calibration text, bf16                     0.02332 0.02260 0.02273       0.0229   151
ours, 2048-token chunks                              0.02375 0.02308 0.02407       0.0236   146
ours, calibration forward in f16                     0.02254 0.02223 0.02296       0.0226
ours, exllamav3's exact 250 x 2048 rows              0.02265 0.02222 0.02236       0.0224
exllamav3 1.6.6 (seed offsets 0, 1000, 2000)         0.02159 0.02269 0.02250       0.0223   230
```

exllamav3's first conversion was a lucky draw. Over three seeds it is level with ours on its own calibration rows
(0.0223 against 0.0224), and either converter's seeds spread by about 0.001. Checked along the way, with the same
calibration rows:
- Per-tensor proxy errors equal exllamav3's within 2% in layers 0, 1, 12 and 23, and on the head (0.000100 /
  0.000099). Each converter grades against its own Hessian, but with the same rows layer 0's Hessians differ only
  by fp16 against bf16.
- Every text tensor is at the same rate (4 bpw).
- Both capture one pass per decoder layer, quantizing the whole layer before the next.
- The only file difference that is not a marker shape: exllamav3 stores the GDN's `in_proj_a`/`in_proj_b` in F16,
  ours BF16. Converted to F16, ours scores 0.02264 against 0.02265.

What is left between the converters is the calibration text: about 0.0005, inside the seed spread but in the same
direction across all three seeds. exllamav3's mix is weighted wiki 50, c4 20, code 20, multilingual 10, technical
10, tiny 5, and random-token rows 20 (about 15% of rows). It builds whole-article rows with shuffled lines, and BOS
on alternate articles. Ours is the same files taken in equal parts, as one stream. Some of that edge may be its wiki
weight (37% of rows), which leans toward the wikitext eval. Run 31's Qwen3.6 gap (0.0221 against 0.0215, one seed
each) fits the same reading: a seed draw plus the calibration mix.

## Run 34 - 2026-11-14 20:30

MoE prefill, Qwen3.6-35B-A3B exllamav3 4.0. A server prefill of 525 tokens under nsys took 309 ms: 248 ms of
kernels, 37 ms idle. The routed experts ran one projection at a time through the generic gather path:
- gate and up each copied the input once per top-k slot (8x) as f16;
- each projection read the routing indices on the host (three syncs a layer);
- each built its padded grouped input with its own gather, and gathered its f32 output back to slot order;
- each group was padded to the largest group's rows, capped at 64.

Now, past 8 tokens, outside a capture, past 64 slots, `fused_silu_moe` groups once a layer (`SlotGroups`):
- each group's token rows gathered straight from the hidden states;
- gate and up, then silu(gate) * up in f32, then down, in the grouped layout;
- the slots gathered back and weighted per token.

```
inference bench 512 / 128                      prefill tok/s   decode tok/s   KLD (wikitext-2 vs BF16 base)
before (generic path, 64-row groups)           1659            197            0.02146
generic, 16-row groups                         2120-2134                      0.02175
fused, 16-row groups                           2740            198            0.02190
fused, 32-row groups                           2500, 2489      198            0.02120
fused, 64-row groups                           2039                           0.02120
fused, 16 rows, EXL3_GEMM_H_ACC off            2535            198            0.02177
IQ4_KSS GGUF                                   4234            164            0.0216
```

16-row groups take exllamav3's single-block tiles (TILESIZE_M 16). On sm_86 those accumulate the MMAs in f16
(`EXL3_GEMM_H_ACC`), but turning that off recovers only 0.00013 of the 0.0007, and it slows the other GEMMs' 16-row
tiles. So the single-block path loses accuracy here for some other reason, not found. Multi-row tiles (32 and 64
rows land on the same variants: equal KLD to five digits) match the dense path. 32 rows is the default: faster and
more accurate than where this started. The fused path's f32 activation beats the generic path's bf16 one by 0.00026.

Left for prefill: the dense `exl3_gemm` (~67 ms for 525 tokens; exllamav3 reconstructs to f16 and runs an hgemm past
144 rows), the casts around the EXL3 calls, and the host read of the routing indices (one sync a layer now).

## Run 35 - 2026-11-14 21:15

Dense prefill matmuls through cuBLAS, as exllamav3 does past 144 rows. Its reconstruct kernel decodes the trellis to
the f16 weight in the rotated basis; ours rotates the input (`* suh`, Hadamard, f16), runs one `cublasGemmEx` (f16
in, f32 out, f32 compute), and rotates the output in place (Hadamard, `* svh`). Weights past 64M values (the head)
stay on the trellis GEMM.
Why: one 512-token chunk's dense projections (Qwen3.6, 250 GEMMs) take ~66 ms on the trellis GEMM, and torch's cuBLAS
f16 GEMM does them in 29.6 ms at 48.5 TFLOPS with f32 accumulation (28.1 ms with f16 accumulation: not worth it).

Error against the CPU reconstruction (relative rms):
- Random weights, 512 x 512: 0.00036-0.00050 for the reconstruct path at 200 rows, every codebook and the
  half-integer rate, against the trellis GEMM's 0.0006-0.0009 at 32 rows.
- Real weights (layer 0's `out_proj` and shared `down_proj`, layer 3's `o_proj`), 200 bf16 rows with an outlier
  channel at 300x: reconstruct 1.68e-3, trellis GEMM 1.82e-3 and 2.33e-3.

KLD against BF16 moved anyway: 0.02120 -> 0.02222. Both paths are deterministic (each scores 0.00000 against its own
earlier output). Scored against each other they differ by KLD 0.0054 (96.9% top-1 agreement). But that difference is
the same 0.0050-0.0056 whether the reconstruct path runs everywhere or only for k = 4096, 512 or 2048. It does not
add up layer by layer: any small numeric change perturbs this model's logits that much, likely through the GDN
recurrence. Against BF16 those three partial runs score 0.02170, 0.02220 and 0.02263.

So at 40 chunks, KLD against BF16 cannot rank kernels whose results differ by less than ~0.0007. That retracts Run
34's reading: its 16-row "3% loss" was this noise, and in isolation the 16-row tiles have the same error as 32 and
64. GROUP_ROWS goes back to 16. Kernels are judged by their error against an f32 reference. KLD at this length only
catches gross breakage. The converter comparisons of Runs 31-33 sit at the same scale. Run 33 used three seeds a
side and a 0.8B model, so its equivalence conclusion holds, but single-run KLD differences under ~0.001 mean little.

```
inference bench 512 / 128, Qwen3.6 exllamav3 4.0     prefill tok/s    decode tok/s   KLD
start (Run 33)                                       1660             197            0.02146
grouped MoE, 16 rows + dense through cuBLAS          3190, 3192       198            0.02209
IQ4_KSS GGUF                                         4234             164            0.0216
```

EXL3 library 8.01 -> 8.05 MiB.

## Run 36 - 2026-11-14 21:35

Dead end: grouping the slots by expert on the device, to drop the one host read of the routing ids a layer (16.7-25
ms of idle GPU in a 525-token prefill). exllamav3's indexed GEMM skips a group whose expert is -1, so one kernel (one
block) built the groups with room for the most any routing makes: slots / 16 plus one an expert, -1 past the last. It
counted each expert's slots, laid out its groups, and placed each slot in slot order.

```
512-token prefill, tok/s           host grouping   device, global ids   device, ids in shared (u16)
                                   3190            3004                 3050
```

The kernel took 143 us a call with the ids in global memory (11.5 ms over 80 calls): one thread an expert scans
every slot. Sizing for the worst case costs more again: ~512 groups where a layer used ~350. Every gather and
activation runs on 8192 rows instead of ~5600, and the cooperative mgemm steps through the idle groups. Reverted. A
win would need exact sizes without a host read, so groups made by the GEMM from offsets on the device, not a fixed
grid.

## Run 37 - 2026-11-14 22:15

Our own grouped expert kernels for prefill (`exl3_dgemv_instance_exl3.cu`), after the decode GEMV. That GEMV's
m16n8k16 MMAs used one row of A; here all 16 rows are an expert group's tokens. The block layout:
- a block per (group, 128 output columns), 8 warps of 16 columns, over every k, so no workspace and no grid sync;
- per 128-wide k step (one Hadamard span), the group's token rows are gathered and rotated (`* suh`, Hadamard) into
  shared memory, and each warp decodes its trellis tiles in registers (exllamav3's `dq8_regs_*`) into full MMAs;
- gate and up share a block. At the end both are rotated out, silu(gate) * up stays in f32, and it is rotated by
  down's suh into down's f16 input (the block's 128 columns are exactly one of down's Hadamard spans);
- down reads that input, runs the same loop, and rotates out by svh.

2 to 4 bpw, every codebook; other rates keep exllamav3's mgemm. Groups are fixed at 16 rows (host-built, exact
count).

Error against an f32 reference, 8 of layer 0's real experts, 100 bf16 tokens (outlier channel at 200x), top-2:
ours 1.71e-3 relative rms, exllamav3's mgemm route 2.81e-3. The fused-MoE test's 100-token case runs these kernels
at 3 and 4 bpw.

```
inference bench 512 / 128, Qwen3.6 exllamav3 4.0     prefill tok/s    decode tok/s   KLD
Run 35 (mgemm experts)                               3190             198            0.02209
our grouped kernels                                  3924, 3922       198, 198       0.02239 (in the band)
IQ4_KSS GGUF                                         4234             164            0.0216
```

nsys, 525-token prefill: 176 ms span, 140 ms of kernels. Experts 42.7 ms (gate/up 30.4, down 12.3), against ~67 ms
on the mgemm. cuBLAS dense ~26 ms, its reconstruct and rotations ~10 ms. Gathers and weighting ~16 ms, casts ~9 ms,
GDN ~14 ms. EXL3 library 8.05 -> 8.14 MiB.

## Run 38 - 2026-11-14 23:00

DFlash 2 (5 drafts) on the EXL3 model; real-text decode as the recipe's Run 9 measured it (wiki summary, code
task, essay; temperature 0.6, top_p 0.95, top_k 20, seed 7, 400 tokens; `rtbench.sh` in the scratchpad):

```
tok/s                               wiki     code     essay    tokens a step
EXL3, no speculation                200.6    200.7    200.8
EXL3, DFlash 2, exllamav3 kernels   299.0    324.1    210.3    3.42
IQ4_KSS GGUF, DFlash 2              285.9    305.6    198.7    3.57
EXL3, DFlash 2, ours (below)        337.3    407.7    258.7    3.51
```

A verify step (nsys, code task, ~4 tokens a step) took 13.7 ms of GPU time: exllamav3's coop MoE kernels 4.1 ms,
its small-m GEMV 3.1 ms (250 launches of 12.3 us), the drafter ~1.1 ms, and the trellis GEMM ~1.1 ms (likely the
head at 6 rows). Our m = 1 decode kernels now take up to 8 rows:
- the GEMV puts the rows in A's first 8 MMA rows (8-15 stay zero), with per-row partials in the workspace; one row
  a warp is rotated out by the last block. q/k/v and the single linears take it first, up to 8 rows, before the
  int8 gate;
- the MoE kernels give slot j token j / topk's row, and the last slot of each (token, column group) sums that
  token's slots in order into its row.

Each row gets the same bits it gets alone (`a_few_rows_get_each_rows_own_bits`: 2 to 6 bpw, every codebook, and the
experts at five tokens). Greedy (temperature 0) output with DFlash still differs from plain greedy decode: code at
char 320 of 1344, wiki at 103 of 1196. So does the GGUF's: identical on code, wiki at 135 of 1226. The tie
tolerance is off by default (`--mtp-greedy-tolerance` 0), and probabilistic drafting only changes what is proposed:
a greedy row accepts a draft only if it is the verify row's argmax. What differs is the verify forward's numerics
against single-token decode. The candidates: the head at 6 rows (past the GEMV workspace, so exllamav3's kernel),
the speculative GDN recurrence, and multi-query attention. EXL3 library 8.14 -> 8.16 MiB.

## Run 39 - 2026-11-14 23:40

Dead end: the 6 bpw head at a verify step's rows on our GEMV. At 6 rows its split partials need ~24M floats, past
the stream's 4M workspace, so it falls back to exllamav3's kernel. With 1940 column groups it has blocks enough
without splitting k, so a block took all 16 of its k steps in turn. Each step went from zero and was added to the
running sum in order: the same f32 sum as the split partials added in split order. A head-width case (5 rows, n
106496) gave each row its one-row bits.

```
head at 6 rows, us a call (nsys, code task)    exllamav3 trellis GEMM 466   ours, k in one block 522   + next-step prefetch 538
```

12% slower, twice a step (target and drafter), and greedy DFlash output diverged from plain greedy decode at the same
characters as before (code 320, wiki 103). The head was not the cause; that leaves the speculative GDN recurrence
and multi-query attention. Reverted.

## Run 40 - 2026-11-15 00:10

Where greedy DFlash output parts from plain greedy decode (Qwen3.6 EXL3, 5 drafts, temperature 0, top-2 logprobs
per token; `greedy_lp.sh`). The real-checkpoint MTP test (`qwen3_5_mtp.rs`) already allows this: the verify kernels
round differently from decode, so drafting may only swap a near-tied top two (TIE_MARGIN 0.25).

```
                 first differing token   plain top two            DFlash top two           max top-logprob shift before
code             70                      -0.8496 / -0.8496        -0.6994 / -0.9494        0.145
wiki             19                      -0.5824 / -0.8324        -0.6396 / -0.7646        0.053
```

Both are near ties; the code one is exact. The gaps (0.25, 0.125, 0) are bf16 steps at these logit magnitudes. An
EXL3 head gave bf16 logits, while GGUF heads already read f32 (`ForwardContext::lm_head`) so theirs do not round. EXL3
heads now read f32 too; decode in alternating pairs is 193.5/193.9 tok/s with bf16 logits against 193.6/193.9.

With f32 logits:

```
                 first differing token   plain top two            DFlash top two           max top-logprob shift before
code             40                      -0.7640 / -0.7688        -0.7431 / -0.8151        0.092
wiki             19                      -0.5998 / -0.8097        -0.6274 / -0.7785        0.056
```

Still near ties, and the logprobs move by up to 0.09 before the split: far past a kernel's rounding (~1e-3 a layer).
That is Run 35's sensitivity again: any small numeric difference anywhere in the forward grows to ~0.05-0.1 nats at
the output. No one component is the cause. It is the sum of the verify forward's parts that are not the
single-token decode's bits:
- the bf16 linears at several rows (router, shared-expert gate, the GDN's b/a);
- the speculative GDN recurrence;
- multi-query attention.

The EXL3 linears, q/k/v and experts already are (Run 38). Greedy output equal to plain decode would need all three
bit-identical to their one-token forms (matching reduction orders in flash attention and the recurrence kernels):
large, and it would only move which of a near-tied pair is picked. Not pursued.

## Run 41 - 2026-11-15 00:55

Toward Qwen3.8-Flash-Next on EXL3 (turboderp's quants: 1.66 bpw is 28.2 GiB of weights and an 18.5 GiB n-gram table;
2.05 bpw 33.2 + 24.4), which does not fit 24 GB. Can EXL3 layers leave the GPU yet? A layer the device map puts on
the CPU loads, but `linear_b`/`experts_b` decode it to a dense weight in the model's dtype. And any CPU layer turns
PagedAttention off for the whole model:

```
Qwen3.6 EXL3 4.0, inference bench 512 / 64           prefill tok/s    decode tok/s
all layers on the GPU                                3922             198
-n "0:36" (layers 36-39 on the CPU, as bf16)         17.2 (29.8 s)    11.4
```

Keeping them quantized off the GPU, step 1: with `--moe-expert-cache`, a CUDA layer's EXL3 experts load into
page-locked host memory mapped into the device (`Exl3Experts::new_in_host`): one allocation a projection, each
expert's trellis, suh and svh 256-byte aligned. The address tables point there. The kernels (decode, verify,
grouped prefill) read the address tables as before, so they are unchanged and read the experts over the bus: no
dequantization, no CPU compute, and decode graphs and PagedAttention stay. No device cache yet.

```
Qwen3.6 EXL3 4.0, every layer's experts in host memory    prefill tok/s    decode tok/s    KLD (40 x 512)
                                                          271              22.2            0.02214
```

Decode reads ~512 MB of experts a token (40 layers x 8 x 1.6 MB) in 45 ms: ~11 GB/s, PCIe 3 x16. KLD is in the band
of Run 35. `experts_in_host_memory_match_experts_on_the_device` pins the same bits as on the device at 1, 5 and 100
tokens.

Next: a device cache of hot experts per layer, so hits read VRAM and only misses cross the bus.

## Run 42 - 2026-11-15 01:40

Step 2: the device cache (`exl3/host_cache.rs`). Each offloaded projection has:
- `N` slots in VRAM (`--moe-expert-cache N`);
- a host copy of its address table;
- decayed use scores (0.9 a step);
- a mapped routing record.

Each fused MoE forward copies its picks into the record (a node decode graphs capture). `cache_step`, called from
`offload_activity` as each step starts (outside any capture), does three things:
- scores the last step's picks;
- points the table at the slots whose copies have landed;
- starts up to 16 copies (16 in flight at most) on a copy stream of its own.

An expert replaces a cached one only past 1.5x its score. The evicted expert's entry points back at host memory, and
an event recorded after that on the compute stream gates the copy that overwrites its slot. A new expert's entry
points at its slot only once its copy's event has completed. So a kernel reads valid bytes either way, and the
results are the same bits as on the device: `experts_in_host_memory_match_experts_on_the_device` now runs eight
rounds with a 3-slot cache stepped between them, and asserts experts landed.

```
inference bench 512 / 128 (synthetic tokens), decode tok/s     cache 64   cache 256
fills a step 1                                                 27.2       27.0
fills a step 4                                                 38.3       42.8
fills a step 16                                                39.7       68.8 (still warming within 128 tokens)

real text (rtbench, 400 tokens), 16 fills          wiki     code     essay
cache 64                                           50.5     41.6     38.0
cache 128                                          98.8     77.5     65.8
cache 256 (every expert)                           198.1    188.4    150.4
no cache (Run 41)                                  22.2 (bench)
GGUF IQ4_KSS, CPU misses, 128 / 64 slots (recipe Run 9): 95-103 / 56-62
```

At 128 slots EXL3 is near the GGUF's CPU-miss path. Below that the GGUF wins: its misses run at the CPU's memory
bandwidth, these over PCIe 3 (~11 GB/s). The loader does not pick this for EXL3 itself (the planner is GGUF's);
`--moe-expert-cache` asks for it.

## Run 43 - 2026-11-15 07:28

Question: does Qwen3.8-Flash-Next's EXL3 (turboderp 1.66bpw_h4_ng3: ring-coded token embedding and hashed n-gram
table, experts mostly K=1.5) load and score next to the IQ4_XS GGUF, and why did the first perplexity run run out of
host memory?

Loader work: the n-gram table ships as `ngram_embedding.safetensors` (128 shards of tail-biting mul1 rings, K=3, 31
words a row, plus per-head bias), added to the file list and kept out of the device-map inventory; `exl3::ring`
decodes rings (n-gram rows on gather, the embedding to dense once at load); embed_tokens and QSA's fused index
projection probe `.trellis` as well as `.weight`. The inventory counts per-layer expert bytes so `--moe-expert-cache`
plans EXL3 experts in host memory.

The out-of-memory run was the perplexity example, not the loader: its safetensors branch dropped
`--moe-expert-cache` (only the GGUF branch passed it), so the planner sized experts on the device and mapped layers
37-47 to the CPU, where EXL3 rebuilds experts dense (~5 GB a layer). Serve and bench runs had mapped 0-47 to cuda[0].
Fixed in the example, along with the paged cache sized to the scored window (without it the QSA context capped at 2051
and a 4096 window failed).

```
examples perplexity -m <exl3 1.66bpw> --moe-expert-cache 64 --pa-cache-type auto --file wiki.test.raw --llama-cpp-ctx 4096
examples perplexity --gguf <IQ4_XS 3 shards> (same flags)
  EXL3 1.66bpw  PPL 1.6935   load 50 s, lowest MemAvailable 64.0 GB of 125.7 GiB (pinned experts + ring table in RAM)
  GGUF IQ4_XS   PPL 1.3207   lowest MemAvailable 106.3 GB (experts pageable in the file mapping)
```

EXL3 is 28% above the GGUF at about 40% of its bits. One 4096 window; no exllamav3 number to compare (the model does
not fit its 24 GB card without offload), so quantization loss and a decode error in the new ring paths are not yet
separated. Next: check `exl3::ring` decodes against exllamav3's own codec on sampled embedding and n-gram rows.

## Run 44 - 2026-11-15 07:55

Question: warm real-text decode for Flash-Next EXL3 1.66bpw against the IQ4_XS GGUF, both with experts in host memory.
At 1.66 bpw the EXL3 is a speed test, not an accuracy one (Run 43).

```
rtbench.sh <tag> --moe-expert-cache <slots> -m <exl3 | gguf>   (serve, warm-up pass, then wiki/code/essay, 400 tokens)
                    wiki   code   essay  tok/s
  EXL3 64 slots     27.7   24.1   23.5
  GGUF 64 slots     26.7   28.7   29.6
  EXL3 128 slots    36.1   30.1   30.4
  GGUF 128 slots    37.6   41.0   42.4
  EXL3 256 slots    55.0   44.8   45.1   (all 48 layers on cuda[0])
```

At equal slots the GGUF wins: its misses run on the CPU at memory bandwidth, EXL3's are read over PCIe 3 by the
kernels (as on Qwen3.6, Run 42). EXL3's experts are ~2.5x smaller, so twice the slots fit: 256 slots (half of the 512
experts) beats the GGUF's best fit (160 slots already spilled layers 43-47 to the CPU in qwen4exp Run 12). The fit, not
the miss path, is where the low-bit EXL3 pays off.

## Run 45 - 2026-11-15 08:40

Question: does an EXL3 expert cache too large for the GPU shrink to fit, as a GGUF's does, instead of moving layers
to the CPU? It did not: the slot planner (`ExpertCachePlan::slots`) only read expert counts from GGUF and UQFF
sources. It now reads them from the checkpoint inventory too, which reports experts only when they are trellis-coded
(a BF16 checkpoint's experts have no host-memory path).

```
serve --moe-expert-cache 512 --pa-context-len 16384 -m <Flash-Next exl3 1.66bpw>
  WARN the requested MoE expert cache does not fit ...; caching fewer experts per layer requested=512 slots=352
  Layers 0-47: cuda[0]; KV cache q3 (offload policy under `auto`, as for GGUF)
```

352 slots is near the ~360 estimated from the GGUF's fit. EXL3 offload under `--pa-cache-type auto` now quantizes
the KV cache like GGUF offload; Runs 42-44 ran BF16 KV.

## Run 46 - 2026-11-15 09:35

Question: Flash-Next EXL3 2.68bpw_h6_ng8 (turboderp KLD 0.0249; model files only, the 1.66 branch's 3-bit n-gram table
linked in) decode at the largest expert cache that fits.

Per-tensor bits: experts 61440 tensors at K=2.5 and 12288 at K=3 (1.66: K=1.5 and 2); everything else 6.

```
rtbench.sh fnx268-c512 --moe-expert-cache 512 -m <exl3 2.68bpw>
  WARN ... requested=512 slots=200 context=16384      (Layers 0-47: cuda[0], KV q3)
  wiki 34.1   code 26.5   essay 26.9 tok/s
GGUF IQ4_XS, Run 44: 128 slots 37.6 / 41.0 / 42.4
```

200 slots, as estimated from file sizes. But decode trails the GGUF at 128 slots by ~30%, where 1.66 at 256 led it.
Rough miss budget: at 200 of 512 experts a layer, ~140 misses a token of ~1.5 MB each is ~210 MB over PCIe 3, ~20 ms of
a ~33 ms step, so the in-kernel bus reads look like the limit. Not yet profiled: the achieved bus rate of the mapped
reads (zero-copy reads by kernels often reach well under memcpy's ~12 GB/s), and the K=2.5 expert kernels' share.

## Run 47 - 2026-11-15 10:10

Question: where does Run 46's 2.68bpw decode step go, and do the kernels' reads of host experts run at the bus rate?

```
prof3.sh p268 --moe-expert-cache 512 -m <exl3 2.68bpw>   (200 slots; warm 100-token request under nsys)
  span 4190 ms, kernels busy 3997 ms, idle > 0.1 ms 58 ms
  exl3_moe_coop_a 1596 ms (3960 x 403 us)   exl3_moe_coop_b 809 ms (3960 x 204 us)   exllamav3's, the K=2.5 layers
  moe_a 343 ms (792 x 434 us)               moe_b 171 ms (792 x 216 us)              ours, the K=3 layers
  dgemv 384 ms, gemv_batched 177 ms; cache fills 6 MB in the window (warm, not churning)
nvidia-smi dmon -s t during rtbench decode: rx 7-10 GB/s a second (mean 8.7, peak 12.0); link Gen3 x16
```

The MoE kernels are ~70% of the step at ~600 us a layer either way, against tens of us were the layer's experts all on
the device, so the step is bus-bound, and ours and exllamav3's match: the kernel is not the lever. The reads already
run at ~70% of Gen3's practical ~12 GB/s. 8.7 GB/s over a ~33 ms step is ~290 MB a token, ~190 of the 480 picks
(top-10 x 48) missing with 39% of experts cached: routing is flat enough that hits track coverage closely.

Implication: staging misses at the full bus rate (a gather kernel with wide coalesced reads into a device buffer, the
MoE kernel then reading VRAM) caps out near +20-25% (~33-35 tok/s), still under the GGUF's 38-42 at 128 slots, whose
misses run on the CPU at memory bandwidth with nothing crossing the bus. On PCIe 3, EXL3 offload leads only where the
cache covers most experts (1.66 bpw at 256 slots).

## Run 48 - 2026-11-15 10:25

Question: where does the GPU sit idle in the streamed converter (Qwen3.8 27B BF16, 4.0 mul1, source shards on a USB
drive)?

```
nvidia-smi utilization every 0.5 s + iostat on the drive, 230 s across layers 2-6
  mean GPU util 81%, 17% of samples under 10%
  GDN layers 65 s: ~6 s idle at each layer's start while the drive reads ~140 MB/s (its sequential rate)
  attention layer 3 (82 s): ~28 s idle while the drive reads 11-54 MB/s
```

Every idle stretch is the next layer's weights coming off the drive on first touch of the mapped shards; the GPU is
busy from calibration through encoding. Fix: before a layer builds, a thread reads the next layer's tensor bytes
(runs from the safetensors headers, neighbours within 1 MiB merged, 8 MiB reads), so its pages are cached while the
GPU works; the next read-ahead waits for the last, keeping reads sequential. Expected ~10-15% of a conversion from
slow storage; not measured yet (the running conversion predates it).
