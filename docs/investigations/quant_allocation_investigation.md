# Quantization types allocated by measured sensitivity (#1013, item 1)

Goal: pick each tensor group's GGUF type from its measured KLD cost per byte saved, for a target size or KLD, instead
of by hand. The Qwen3.6-35B-A3B iGPU sweep (qwen36_35b_a3b_recipe_investigation Runs 16-17) found the groups' costs
add near-linearly, so a single-group sweep predicts combined recipes. Tool: `scripts/quant_sweep.py` (`sweep`, then
`allocate`); quantization with `inference quantize gguf` (ik_llama.cpp's search, #1008/#1011), KLD with the
perplexity example, so no llama.cpp tool is involved.

## Run 1 - 2026-11-18 21:30

Question: can the whole loop run on our tooling, on a small model of the 35B's architecture?

Setup: Qwen3.5-2B (hybrid GDN like Qwen3.6-35B-A3B, dense FFN, tied embeddings: `token_embd` is also the head, 509M
of its 1.9B weights) converted to a BF16 GGUF with llama.cpp's `convert_hf_to_gguf.py --no-mtp`. The perplexity
example gained `--kld-base-write` (a `llama-perplexity --kl-divergence-base` file written by our engine: the text
tokenized as llama-perplexity does, chunks of `--kld-ctx`, each chunk's second half recorded). WikiText-2 test, 40
chunks of 512, as the recipe runs.

Base written in 57 s. BF16 scored against its own base: KLD mean -0.00001, 100% same top token (the base's u16
encoding). All-Q8_0 (F32 `ssm_alpha/beta/conv1d`): 2,015,477,472 bytes, KLD 0.00150, 96.97% same top. One sweep
point costs ~48 s (3 s to quantize, 44 s to score).

Sweep: seven groups (`ffn_gate`+`ffn_up`, `ffn_down`, GDN `attn_qkv`, `attn_gate`, `ssm_out`, full attention
q/k/v/output, `token_embd`) each lowered alone from all-Q8_0 to Q6_K, Q5_K, Q4_K, IQ4_XS and Q3_K, no imatrix.

## Run 2 - 2026-11-18 22:10

Sweep results (bytes saved from all-Q8_0 at 2,015,477,472 B, KLD 0.00150; no imatrix):

```
group          Q6_K              Q5_K              Q4_K              IQ4_XS            Q3_K
ffn_gate_up    146 MB  0.00228   227 MB  0.00540   302 MB  0.01472   321 MB  0.01604   382 MB  0.06175
ffn_down        73 MB  0.00200   113 MB  0.00445   151 MB  0.01139   160 MB  0.01162   191 MB  0.04100
attn_qkv        55 MB  0.00182    85 MB  0.00273   113 MB  0.00747   120 MB  0.00696   143 MB  0.02441
attn_gate       18 MB  0.00161    28 MB  0.00210    38 MB  0.00335    40 MB  0.00354    48 MB  0.01111
ssm_out         18 MB  0.00218    28 MB  0.00529    38 MB  0.01482    40 MB  0.01177    48 MB  0.05020
attention       21 MB  0.00197    33 MB  0.00380    44 MB  0.00926    47 MB  0.00968    56 MB  0.03585
token_embd     123 MB  0.00233   191 MB  0.00398   254 MB  0.01235   270 MB  0.01420   322 MB  0.04747
```

Per byte, `ssm_out` is the dearest group and `attn_gate` the cheapest; Q3_K is a cliff in every group; IQ4_XS
beats Q4_K on the GDN projections and `ssm_out` while costing more on the FFN and the (tied) embeddings.

Allocation against uniform recipes at the same size (`allocate`'s exhaustive search, then `measure`, four models
scored at once):

```
recipe              bytes           KLD measured  predicted   99%     same top
uniform Q5_K        1,310,441,184   0.01831       0.01875     0.1049  92.43%
allocated (Q5_K)    1,308,671,712   0.01665       0.01708     0.0883  92.86%
uniform Q4_K        1,075,429,088   0.06880       0.06436     0.4000  85.58%
allocated (Q4_K)    1,073,397,472   0.05089       0.04728     0.2675  87.92%
uniform IQ4_XS      1,016,676,064   0.06458       0.06481     0.3624  86.93%
```

Allocated picks at the Q4_K size: IQ4_XS for `ffn_gate`/`ffn_up`, `attn_qkv`, `attn_gate` and `token_embd`, Q4_K for
`ffn_down`, Q6_K for `ssm_out` and the full attention. At the IQ4_XS size the best mix was all IQ4_XS (anything
smaller per group reaches Q3_K).

Findings: at the same size the allocation cuts KLD 26% at ~4.4 bits (p99 0.27 against 0.40, top token +2.3 points)
and 9% at ~5.4 bits. The sum of single-group deltas predicts within 2-8%; at the 4-bit points the measured KLD runs
~7% above the sum (the errors compound slightly). Five recipes scored in 4.5 min with `--jobs 4`: the scoring is CPU
bound (~400 CPU-s a point: the logits come back to the host and every row walks the 248K vocabulary), so jobs
mostly share the cores; moving the KLD onto the GPU beside the logits would be the next speedup.

Next: the 35B (an imatrix, its BF16 again, ~10 min a point), then Hessian error feedback (#1013 item 2).

## Run 3 - 2026-11-19 08:10

Question: does scoring on the GPU, beside the logits, take the scoring off the CPU?

Change: `PromptLogitsRequest::reference` (llama-perplexity's base encoding: per-row scale and floor, u16 codes) makes
the engine score `KL(reference || model)`, the top-token match and the reference log-prob of the next token where the
logits were computed, 64 rows at a time; only the codes go up and three numbers a row come back. The perplexity
example sends each chunk's base records that way.

Findings along the way:

- The first try ran at 3.2 s a chunk (CPU scoring: 0.8 s): raw logits move to the host per forward (step.rs
  `scatter`), so the tensor ops ran on the CPU. A request flag (`raw_logits_on_device`) keeps them on the device.
- With them kept, chunk 2 on read garbage (PPL 2512): the API thread read them on another stream before the step's
  kernels finished. The step now synchronizes before handing device logits over.
- Same top-1 85.49% against the CPU's 85.58%: a device argmax may take any tie, llama-perplexity the first index
  (bf16 logits tie at the top often enough). `first_max` picks the first.

Result (uniform Q4_K, 40 chunks): PPL 12.2411, KLD mean 0.06880, median 0.04845, p99 0.40003, same top 85.58%, equal
to the CPU scorer; the divergence takes ~30 ms a chunk, a whole scoring run 7.4 s (33 s before, 4.9 CPU-s against
57). The five Run 2 recipes re-measured equal to Run 2.

But the five recipes still take 178-220 s: quantizing now dominates (~44 s a k-quant recipe at ~7 cores, the dev
build's debug assertions on). Points run whole on each worker (quantize then score): jobs 1 220 s, 2 190 s, 3 178 s
(2,275 CPU-s: the quantizers contend). Next: reuse tensors across points (the base and each group/type quantized
once, a point's file assembled by copying), and a quantizer built without debug assertions.

## Run 4 - 2026-11-19 08:50

Question: with scoring off the CPU, can a sweep point stop re-quantizing the whole model?

Change: `inference quantize gguf --reuse <earlier.gguf>` (repeatable) copies a tensor an earlier output from the same
source, imatrix and quantizer holds at its planned type instead of quantizing it again (a requantize test pins that a
reusing run writes the identical file). `quant_sweep.py sweep` quantizes and scores the base once, keeps it while the
points run, and every point reuses it, so a point quantizes only its group; `measure` takes `--reuse` too.

Command: `sweep` over two of the seven groups (`ffn_gate_up`, `ssm_out`, five types each) with `--jobs 2`.

Result: base and 10 points in 103.5 s (594 CPU-s), ~9.4 s a point against ~48 s in Run 1; every KLD, p99 and top
equal to Run 2's. The full seven-group sweep is now ~6 min instead of ~29. The remaining time is the group's own
quantization (the FFN at ~5 bits is the largest) and the 7 s score.

## Run 5 - 2026-11-19 10:30

Question: the Qwen3.6-35B-A3B sweep on our tooling, and does allocation beat the hand recipe R1 there?

Setup: BF16 GGUF reconverted from the safetensors (71.1 GB), Unsloth imatrix, base = R0 (IQ4_XS experts, F32 router /
GDN small tensors, Q8_0 the rest; 20,273,121,216 B, KLD 0.01825). Nine groups, each point an overlay (#1017) of its
group scored over the base (27 s a point). The two IQ4_KSS expert points failed: `quantize gguf` could not write the
type (fixed since, ik parity investigation Run 5). The experts-down IQ4_KS point failed to score once (a test build
replaced the example binary mid-run) and was rerun.

Bytes saved (negative: added) and KLD added against the base:

```
group         Q6_K              Q5_K              Q4_K              IQ4_XS            other
exps_gate_up                                                                          IQ4_KS -43 MB -0.00108, IQ3_KS +2902 MB +0.01604
exps_down                                                                             IQ4_KS -86 MB -0.00028, IQ3_KS +1419 MB +0.01067
shexp          +31 MB +0.00023   +48 MB +0.00075   +64 MB +0.00176   +69 MB +0.00219
attn_qkv      +122 MB +0.00003  +189 MB +0.00070  +252 MB +0.00203  +267 MB +0.00359
attn_gate      +61 MB -0.00022   +94 MB +0.00046  +126 MB +0.00176  +134 MB +0.00108
ssm_out        +61 MB -0.00014   +94 MB +0.00073  +126 MB +0.00288  +134 MB +0.00285
attention      +73 MB +0.00028  +112 MB +0.00116  +150 MB +0.00416  +159 MB +0.00349
token_embd    +123 MB +0.00029                    +254 MB +0.00048
output        +123 MB +0.00027  +191 MB +0.00145  +254 MB +0.00568
```

Readings: IQ4_KS experts beat IQ4_XS for 43 / 86 MB more (gate/up -0.0011 KLD, the largest single gain); token
embeddings at Q4_K are nearly free (+0.0005 for 254 MB); the untied output head is not (+0.0057 at Q4_K, top-1 -1.7
points); Q6_K on `attn_gate` / `ssm_out` scoring below the Q8_0 base is within noise (~0.0003 here).

Allocated picks (`allocate`): at the base's size IQ4_KS experts + Q6_K `attn_qkv`/`attn_gate`/`ssm_out` (predicted
20.16 GB, KLD 0.0166); at 19.5 GB IQ4_KS gate/up, Q5_K `attn_qkv`/`attn_gate`, Q6_K the rest, Q4_K embeddings
(predicted 0.0195; R1, hand-made at the same size, measured 0.0194 with ik's tool); at 18.5 GB IQ3_KS experts-down
(predicted 0.0280). Their measurement was stopped: each needed ~14 min of CPU expert quantizing, which Run 6 removes.

## Run 6 - 2026-11-19 12:00

Question: where does a sweep point's time go now, and can the quantizing leave the CPU?

CPU: the quantizer ran on rayon's global pool (8 physical cores). Four layers of gate/up experts at IQ4_KS with the
imatrix: 8 threads 79.7 s, 16 (SMT) 63.5 s, identical bytes. `quantize gguf` now runs in the logical-CPU pool. Batching
experts per CPU call gained nothing (852 s against ~780 s for a whole group): the search was already compute-bound.

GPU (`kernels/cuda/gguf_quantize`, IQ4_KS first): the CPU port's every product, sum and fused multiply-add as an
explicit rounding intrinsic, so nvcc contracts nothing. First version (a thread per block, a thread per row): 9.6 s for
the four layers, kernels 6.2 s of it (the row pass serial over 2,048 values per row, 512 threads a launch). Then: a
warp per block for the search (the CPU's eight interleaved partial sums chained lane j - 8 to lane j by shuffles), a
weights pass, the row pass split (row scale, per-block indices and refit terms, a serial add of the terms per row);
experts batched ~1 GiB a call with a per-expert imatrix; the codebook tables in shared memory (constant memory
serialized the lanes' differing indices); bf16 uploaded and widened on the GPU.

```
four layers of gate/up experts, IQ4_KS, imatrix   wall
CPU, 16 threads                                   58-62 s
GPU, per expert, first kernels                    9.6 s
GPU, chunked + split kernels                      8.2 s (kernels 3.5 s, search 2.8 s)
GPU, tables in shared memory                      7.8 s
GPU, bf16 widened on the GPU                      4.6 s   (a Q8_0 run of the same tensors, mostly I/O: 3.2 s)
```

Every version wrote the CPU's bytes: GPU tests compare f32, multi-expert and bf16 inputs against `iq_quantize`, and the
four-layer outputs `cmp` equal. A whole expert group drops from ~13 min to ~45 s; `--cpu` keeps the CPU path.

Next: the same for IQ4_XS, IQ4_KSS, IQ3_KS and the K-quants the sweeps use, then rerun Run 5's measurements.

## Run 7 - 2026-11-19 13:10

Question: the other types the sweeps quantize on expert tensors (IQ4_KSS, IQ4_XS, IQ3_KS) on the GPU, byte-exact.

Shared parts moved to `gguf_quantize/iq4k_common.cuh` (tables, rounding, f16 bits, weights, row scale). Per type:

- IQ4_KSS: a thread per block for the search (one serial fit chain per try, the parity pruning inside it); the row
  refit is a fused chain across the row, so the block pass stores each value's level and a row pass chains them in the
  CPU's order. Matched the CPU on the first run.
- IQ4_XS: no row scale; a thread per block for weights (raw imatrix, sigma2 summed per lane), the search from both
  codebook ends and the step-by-step refinement; a thread per super-block for the f16 scale and indices. First run:
  1,001 bytes differing, all in the f16 scales' low byte (CPU 0x02af, GPU 0x0200): the library builds with
  `--use_fast_math`, whose flush-to-zero makes `__float2half_rn` drop subnormal f16 results. A bit-level conversion
  (the `half` crate's) fixed it; IQ3_KS's f16 row scale uses it too.
- IQ3_KS: a warp per block as IQ4_KS (8-value codebooks), the row's f16 scale, a thread per super-block for the packed
  bits (its eight blocks share bytes) and levels, the row refit chained per row. Matched on the first run.

```
four layers of gate/up experts, imatrix   CPU (16 threads)   GPU     identical
IQ4_KS                                    58 s               4.6 s   yes
IQ4_KSS                                   188 s              13.1 s  yes
IQ4_XS                                    61 s               3.4 s   yes
IQ3_KS                                    44 s               5.9 s   yes
```

GPU tests compare all four with the CPU ports on f32 rows (plain and imatrix, three shapes, a zeroed block and row),
per-expert imatrix slices and bf16 rows. K-quants stay on the CPU: the sweeps use them only on the small non-expert
groups.

## Run 8 - 2026-11-19 13:45

Question: Run 5's stopped measurements, now that the experts quantize on the GPU; does allocation beat R1?

The two IQ4_KSS expert points and the allocated recipes as overlays over the base (six points in 13.4 min with GPU
quantizing; two more in 5.6 min):

```
point                                   bytes            KLD predicted  KLD measured  p99     same top
base (R0)                               20,273,121,216                  0.01825       0.1835  94.38%
exps_gate_up IQ4_KSS                    19,628,246,976                  0.02023       0.2026  93.83%
exps_down IQ4_KSS                       20,015,171,520                  0.01962       0.1942  94.27%
allocated at the base's size            20,158,302,144   0.01656        0.01686       0.1816  94.61%
R1 (hand-made, as an overlay)           19,968,231,360                  0.01995       0.2086  94.15%
allocated at R1's size                  19,904,022,464   0.01704        0.01829       0.1968  94.42%
allocated at 19.5 GB                    19,490,736,064   0.01945        0.02006       0.2160  93.71%
allocated at 19.5 GB, IQ4_KSS down      19,436,922,816   0.01896        0.02019       0.2023  94.03%
allocated at 18.5 GB (IQ3_KS down)      18,399,315,904   0.02799        0.02947       0.3178  92.78%
```

(R1's size differs from its 19.50 GB file of Run 17: this base keeps the MTP layer's experts at IQ4_XS.)

Findings: at R1's size the allocation measures 8% lower KLD (0.0183 against 0.0200), with a lower p99 and a higher
top-1 match. Predictions ran 2-5% under the measurement for the first four recipes and 6-7% for the two with Q4_K
embeddings: Q4_K `token_embd` cost +0.0005 alone but +0.0014 on top of the 20.27 GB recipe (compounding, or partly
noise at ~0.0003). IQ4_KSS gate/up was the cheapest single saving per byte (+0.0020 for 645 MB), but IQ4_KSS down
inside the 19.5 GB recipe measured no better than without it. A pick should be measured before it is trusted; with
GPU quantizing that is ~1-3 min a recipe.

## Run 9 - 2026-11-19 14:30

Question (#1013 item 2): does GPTQ-style error feedback, kept inside the GGUF formats, lower KLD?

Method: full input Hessians (`X^T X / n`) of every text linear of Qwen3.5-2B, collected with forward hooks in a
transformers prototype (scratch `collect_hessians.py`, 128 chunks of 512 tokens, keyed by llama.cpp's GGUF names;
the 35B's V-head reorder of `ssm_out`'s input columns applied when key and value heads differ). `quantize gguf
--hessians` then quantizes a matrix a block of columns at a time with the type's own byte-exact quantizer (the
Hessian's diagonal as its importances) and carries each block's error onto the columns not yet quantized:
`W_rest -= E U_bb^-1 U_b,rest`, `U` the inverse Hessian's upper Cholesky factor (cuSOLVER). `--hessian-weights-only`
keeps the diagonal weighting without the feedback, the baseline. Only types whose blocks carry their own scales for
now (the K-quants, Q4_0-Q8_0, IQ4_NL, IQ4_XS); about a minute for the 2B.

First run, WikiText-train calibration, scored on WikiText test (same domain), then on the repo's docs (another):

```
2B, uniform type      none              Hessian weights   GPTQ (damp 0.01)
Q4_K   wiki / docs    0.0688 / 0.0598   0.0414 / 0.0405   0.0298 / 0.0443
IQ4_XS wiki / docs    0.0646 / 0.0593   0.0505 / 0.0442   0.0317 / 0.0462
```

The feedback fit the calibration's domain (-28% / -37%) and lost on the other (+9% / +5%): GPTQ's calibration
overfitting. Held-out sets for the next grid: WikiText test, the docs' even-numbered files (docs B), and 400 Rust
sources from the repo (never calibrated on); a second calibration mixes WikiText train with the docs' odd files in
4 KB pieces. Q4_K:

```
calibration, method          wiki     docs B   rust
none                         0.06880  0.05461  0.04984
wiki, Hessian weights        0.04143  0.03819  0.03349
wiki, GPTQ damp 0.01         0.02975  0.04371  0.04214
wiki, GPTQ damp 0.1          0.03036  0.03835  0.03667
wiki, GPTQ damp 1.0          0.03173  0.03618  0.03237
mixed, Hessian weights       0.04234  0.03596  0.03080
mixed, GPTQ damp 0.01        0.03346  0.02269  0.02226
mixed, GPTQ damp 0.1         0.03209  0.02319  0.02208
```

Findings: with a varied calibration the feedback lowers KLD 24-37% against the same Hessians used as weights only, on
every held-out set, Rust included; damping trades in-domain gain for robustness when the calibration is narrow (1.0
on WikiText alone still beat the weights everywhere). `--gptq-damp` defaults to 0.1.

Next: Hessian collection in the engine (the EXL3 writer's per-linear Hessians) rather than transformers; the row-scale
types (IQ4_KS, IQ4_KSS, IQ3_KS: the row scale has to be fixed before the first block); MoE experts, whose per-expert
gate/up Hessians would not fit on disk for the 35B (16 MB each, 10,240 of them), so the feedback would run as the
engine collects, layer by layer, as the EXL3 writer does.

## Run 10 - 2026-11-19 15:20

Question: GPTQ feedback for the row-scaled types (IQ4_KS, IQ4_KSS, IQ3_KS), whose row scale the CPU quantizers fix
from all of a row's block scales before encoding any block.

Change: each split into a block search and a block encoder, the row functions rebuilt from them (ik parity script 0
bytes for all three; IQ3_KS now pinned in `the_ik_quantizer_writes_ik_llamas_bytes` too). GPTQ takes a row's scale
from an ordinary first pass, then per 256 columns searches each block's scale on the columns as fed forward, encodes
against the fixed row scale (decoding to `d * level`), carries the error on, and refits the row scale at the end.

Qwen3.5-2B, every matrix at the type, mixed calibration (Run 9's), KLD:

```
type, method                 wiki     docs B   rust
IQ4_KS none                  0.06277  0.05663  0.04521
IQ4_KS Hessian weights       0.04354  0.03499  0.02997
IQ4_KS GPTQ                  0.03170  0.02208  0.01987
IQ3_KS none                  0.32068  0.27427  0.26983
IQ3_KS Hessian weights       0.20556  0.15577  0.14163
IQ3_KS GPTQ                  0.15954  0.09677  0.09432
```

-27 to -37% (IQ4_KS) and -22 to -38% (IQ3_KS) against the same Hessians as weights only. The row-scaled path runs on
the CPU a super-block at a time (~90 s for the 2B).

Next: Hessians from the engine instead of transformers (the safetensors loader's linears already collect them, the
EXL3 writer's path), keyed by GGUF name through `gguf_tensor_names`; for MoE, one pooled Hessian per layer for the
gate/up experts (they share the layer input; ~16 MB a layer instead of 4 GB) and per-expert ones for down (1 MB each).

## Run 11 - 2026-11-19 15:45

Question: Hessians from the engine instead of the transformers prototype, in a form the 35B's experts fit.

Change: `inference quantize hessians -m <checkpoint> --calibration-file <text> -o <dir>` loads the checkpoint's text
model on the host (a multimodal one's alone, as an EXL3 write does) and streams it through the GPU a layer at a time
(the EXL3 writer's loop): each layer's Hessians over the calibration states, written as `<linear>.safetensors`
(`hin`, exllamav3's layout), q/k/v and gate/up sharing one; the head's over the final normed states. Stacked gate/up
experts read the layer's shared input, so they take the shared expert's Hessian (their pooled second moment); other
stacks write one per expert (a 35B down expert's is 1 MB). `quantize gguf --hessians <dir>` binds each file to the
tensor its weight loads from (`<stack>.<e>.<proj>` to expert e, `lm_head` to the output or, tied, `token_embd`), the
GDN output projection's Hessian reordered as the converter orders its V heads; GPTQ runs per expert for stacks.

Qwen3.5-2B, the mixed calibration's first ~65K tokens: 151 Hessians in 30 s (5.4 GB). GPTQ, KLD:

```
type, Hessians                 wiki     docs B   rust
Q4_K, prototype (Run 9)        0.03209  0.02319  0.02208
Q4_K, engine                   0.02749  0.01941  0.01859
IQ4_KS, prototype (Run 10)     0.03170  0.02208  0.01987
IQ4_KS, engine                 0.02597  0.01755  0.01724
```

The engine's do better mainly because they include the head's: the tied `token_embd` now gets the feedback too.
Against no calibration at all (Q4_K 0.0688 / 0.0546 / 0.0498) that is -60 / -64 / -63%.

## Run 12 - 2026-11-19 18:40

Question: GPTQ on Qwen3.6-35B-A3B with the engine's Hessians.

`quantize hessians` over the same ~65K mixed tokens, the model streamed from the host (67 GB of BF16): 40 layers in
49 min (~60 s a layer; ~10 min of loading before), 10,571 files, 17 GB (dense Hessians, a pooled one for each gate/up
expert stack from the shared expert, 256 per-expert down ones a layer). `quantize gguf --hessians` bound 371 of them.

The allocated recipe at R1's size (Run 8: IQ4_KS experts, Q6_K `attn_qkv`/`attn_gate`/`ssm_out`, Q4_K `token_embd`,
Q8_0 the rest), WikiText KLD:

```
calibration, method                        KLD      p99     same top
Unsloth imatrix, no feedback (Run 8)       0.01829  0.1968  94.42%
our Hessians' diagonals, no feedback       0.01718  0.1785  94.22%
our Hessians, GPTQ                         0.01672  0.1666  94.61%
```

The feedback gives 2.7% here against 24-37% on the 2B; the calibration alone 6% over Unsloth's imatrix. Suspects:
the gate/up experts take the pooled (all-token) Hessian rather than their own routed rows', and each down expert's
Hessian comes from ~2K rows (65K tokens x 8 of 256 experts).

Two false starts on the way: the first run left 169 tensors BF16 (the run script's `while read` dropped the rules
file's unterminated last line, `\.weight$=q8_0`); and the first GPTQ took 66 s a layer at 7.5 of 16 threads and 3%
GPU, the experts one at a time. Experts sharing a Hessian are rows of one matrix to GPTQ, so a stack now goes in one
pass, and experts with their own Hessians run side by side: 37 min for the recipe.

## Run 13 - 2026-11-19 19:25

GPTQ's row-scaled loop on the GPU (each row's scale from the type's kernels, then per super-block the block search,
the encoding against that scale with its levels, the row's fit sums continued in the CPU's order; the error and its
carry already ran there). A GPU test checks the bytes equal the CPU loop's for IQ4_KS, IQ4_KSS and IQ3_KS.

The 35B recipe: 252 s against 2,219 s (8.8x), KLD 0.01672 / p99 0.1666 / 94.61%, the same file's.

## Run 14 - 2026-11-20 01:00

Where a 35B Hessian dump's 64 s a layer goes (the 4x calibration run, `quantize hessians` from the safetensors on an
external USB drive): the drive reads at 71 MB/s at 100% utilization, ~1.7 GB a layer; the process at one core, the GPU
idle between bursts, `memmove` high in the profile. The host load also built every layer (126 GB) though the
streamed collector rebuilds each on the GPU, the states crossed to the host and back for each layer, and each layer
ran twice (collect, then propagate).

Changes: the states stay on the GPU when they fit (4 GiB; the 35B's 4x set is ~1.1 GB); one pass a layer collects and
propagates (the layer as loaded, so nothing is lost); a Hessian dump skips the host layer build as an EXL3 write does;
and `quantize hessians -m <file>.gguf` collects from a full-precision GGUF through its native bindings, read-ahead
ranges from the GGUF's tensor table, so the BF16 GGUF on NVMe serves and the safetensors are not needed.

2B check, Q4_K GPTQ scored on WikiText: Hessians from the GGUF 0.02731, from the safetensors 0.02749 (the Hessians
differ by 0.04% median, 0.2% worst, down projections: bf16 rounding of the GGUF's stored norms). The states-on-GPU,
one-pass change alone gave bit-identical Hessians on the 2B.

35B 4x collection from the GGUF: 419 s total, 10-13 s a layer (was 61-67 s), peak RSS 71 GB.

## Run 15 - 2026-11-20 01:30

Experiment A: the alloc-r1size recipe (Run 12) with GPTQ from Hessians over 4x the calibration tokens (1,080,000
characters of the WikiText + docs mix), gate/up still pooled.

```
Hessians                 KLD mean   p99     same top-1
1x (Run 12/13)           0.01672    0.1666  94.61%
4x                       0.01680    0.1738  94.33%
```

No gain: the token count was not what held the 35B back (down experts' ~2K rows each included). That leaves the
pooled gate/up Hessians as the suspect; experiment B collects each expert's own.

## Run 16 - 2026-11-20 02:40

Experiment B: each gate/up expert's own routed Hessian (`INFERENCE_PER_EXPERT_HESSIANS`, collected from the GGUF over
the 4x tokens; up binds to the gate's files, one input). First stored in bf16 to halve the 168 GB: 255 of layer 0's
256 experts then failed to factor ("the damped Hessian is not positive definite"), min eigenvalue of H + 0.1 mean
diag -0.04 to -0.18 (mean diag ~0.3). Each expert sees a few thousand rows of 2048 inputs, so its Hessian is near
singular and bf16's rounding outweighs the damping. Stored instead as f32 packed upper triangles (`hin_upper`): exact,
96 GB with the down experts, collected in 12 min (17-19 s a layer).

```
Hessians                        KLD mean   p99     same top-1
pooled gate/up, 1x (Run 13)     0.01672    0.1666  94.61%
pooled gate/up, 4x (Run 15)     0.01680    0.1738  94.33%
per-expert gate/up, 4x          0.01699    0.1784  94.54%
```

No gain either: an expert's own few thousand rows fit GPTQ no better than the pooled second moment. Neither suspect
from Run 12 explains the 35B's small feedback gain.

The quantize took 991 s (pooled: ~230 s) with 8 s between GPU bursts: a CPU triangular inverse per 256-column block
per expert (column-strided, 45% of samples), then the 256 Hessian files of a stack read and unpacked one at a time
(64%). Row-wise back substitution for the inverse, the files loaded in parallel and the triangle mirrored in 64x64
tiles: 470 s, KLD 0.01699 unchanged.

## Run 17 - 2026-11-20 03:30

Why GPTQ's feedback moves the 35B so little (Run 12: diag-only 0.01718, GPTQ 0.01672, -2.7%; the 2B -27% to -34%).
`INFERENCE_GPTQ_REPORT=1` logs each GPTQ tensor's output error over the calibration inputs, `tr(dW H dW^T) /
tr(W H W^T)`, summed over experts, and the Hessians' off-diagonal share of their Frobenius norm. Both modes of each
model quantized with it (the 2B from its GGUF Hessians, the 35B recipe from the pooled 4x ones); medians per kind:

```
2B, Q4_K / IQ4_KS        gptq/diag err   diag rel_err   offdiag
attn_* / attn_qkv/gate   0.52-0.56       7e-4 - 3e-3    0.80
attn_output              0.35            8.8e-3         0.98
ffn_gate / ffn_up        0.55            1.4e-3 / 4e-3  0.81
ffn_down                 0.49            4.4e-3         0.93
ssm_out                  0.59-0.60       4.7e-3         0.91

35B recipe
attn_* (Q8_0)            0.27-0.35       4e-6 - 7e-5    0.78-0.96
attn_qkv / gate (Q6_K)   0.58 / 0.63     8e-5 / 2e-5    0.80
ssm_out (Q6_K)           0.61            2.6e-4         0.92
shexp (Q8_0)             0.61-0.67       1e-5 - 3e-5    0.35-0.96
ffn_gate/up_exps (IQ4_KS) 0.72           3.0e-3/4.8e-3  0.96
ffn_down_exps (IQ4_KS)   0.86            4.6e-3         0.75
```

The error sits in the routed experts (rel_err 3-5e-3 against <= 2.6e-4 for everything else), and there GPTQ cuts it
by 28% (gate/up) and 14% (down), against ~45-50% on every 2B tensor. Damping is not the cause: the 35B's expert input
Hessians are flatter, not heavier-tailed (layer 20: mean/median diagonal 1.21 vs the 2B's 1.95, max/mean 47 vs 250,
0.29% of eigenvalues below the 0.1 mean-diagonal damping vs 12.4%); a near-isotropic input leaves the feedback little
correlation to trade error along. Down experts' inputs are flatter still (expert 0: max/mean 10.9, none below).

And the cut that is there barely reaches KLD: a 14-28% lower error on the tensors carrying nearly all of it gave
-2.7% KLD, where the 2B's ~50% gave ~30%. Measured in-sample (the calibration's own Hessians), so the next check is
the same error over Hessians of the evaluation text: if the cut holds out of sample, the loss is between layer error
and KLD (propagation, routing among 256 experts) and sequential GPTQ / router agreement come next; if it vanishes,
the feedback is fitting the calibration and is done for this model's experts.

## Run 18 - 2026-11-20 03:15

Out of sample: Run 17's two 35B files (same bytes, checked) with the report measured over Hessians of the KLD's own
text (`INFERENCE_GPTQ_REPORT_HESSIANS`, the head of wiki.test.raw, ~21K tokens, collected in 109 s):

```
gptq/diag err            in sample (calibration)   out of sample (eval text)
ffn_gate/up_exps         0.72                      0.78
ffn_down_exps            0.86                      0.99
shexp                    0.61-0.67                 0.65-0.67
attn_* / attn_qkv/gate   0.27-0.63                 0.29-0.67
ssm_out                  0.61                      0.64
output                   0.36                      0.42
```

Two findings. The down experts' cut vanishes out of sample: each has its own Hessian over a few thousand routed rows
(133K tokens x 8 / 256 ~ 4K, for 512 inputs), and the feedback fits those rows' noise. The same reason per-expert
gate/up Hessians (Run 16) did worse than pooled ones. Every pooled or dense Hessian (gate/up from the shared input,
shexp, attention, ssm_out) keeps most of its cut: gate/up 22% out of sample.

That 22% still reaches KLD weakly. Rough share: gate/up IQ4_KS -> IQ3_KS (about 4x the error) added 0.0171 KLD in the
sweep, so their part at IQ4_KS is ~0.0057 and a 22% cut predicts ~-0.0013; the whole recipe moved -0.0005. So some
loss between layer error and output remains (propagation through 40 layers, routing), on top of the down experts'
feedback being noise.

Next: the down experts' Hessians shrunk toward their stack's mean (more effective rows), or their feedback off; then
sequential GPTQ (each layer's Hessians collected with the layers before it quantized) for the propagation gap.

## Run 19 - 2026-11-20 04:30

GPTQ quantize speed, the 35B recipe with the pooled 4x Hessians (a pass of ~214 s). Per kind: the expert stacks 156 s
(gate/up 1.1-1.4 s each, down 1.1 s), Q6_K attn_qkv/attn_gate/ssm_out 38 s, the head and embedding 11 s. Timers in a
stack: the IQ4_KS fixed-scale encode 0.80 s (8 launches over 131K rows), the carry 15 ms, reading and widening the
BF16 0.18 s. ncu on `iq4_ks_search`: L1/TEX 99.7%, DRAM 1.5%, issue slots 38%: bound by its shuffles (a warp per
32-value block, four chained sums of ~11 shuffles per scale try, 31 tries).

Changes, each checked byte for byte (the full recipe's file identical to the one before, `cmp`; the GPU-vs-CPU and
batched-vs-alone tests):
- `iq4_ks_search`: eight lanes per block, lane k running the CPU's partial sum k over values k, k + 8, ... in
  registers, the eight added in `hsum_float_8`'s order by three xor shuffles (IEEE addition commutes, so the pairs
  are the same): 38 ms to 7.5 ms a call.
- `iq4_ks_blocks`: a thread per value pair instead of per block (coalesced).
- the super-block bytes copied into the output in parallel, not stripped byte by byte then scattered serially.
- `quantize_blocks` / `from_float_imatrix`: tasks sized to fill every thread (a GPTQ call encodes one block column,
  2 tasks of 1024 blocks on 16 threads for ssm_out).
- per-expert Hessians: the stack's experts in one batched GPU pass (per-expert importance through the kernels'
  rows-per-importance, the carry a batched matmul over stacked factors, up to 2 GiB a batch).
- `quantize gguf` turns off cudarc's per-buffer event tracking (one stream): 37,440 event creates and destroys, ~20 s
  of contended API time over four down stacks, half their setup.
- BF16 widening in parallel; the values copy before upload only when a column is dead.

```
                          before   after
whole recipe              214 s    127 s
ffn_gate/up_exps (41)     110 s    44 s
ffn_down_exps (41)        46 s     44 s
attn_qkv/gate, ssm_out    38 s     22 s
```

Left: a down stack reads 256 Hessian files (0.16 s) and its weights (0.19 s) before the GPU starts; prefetching the
next tensor during the current one's GPU pass would hide that (~25 s). The IQ4_KSS and IQ3_KS searches still use
the chained warp sums.

## Run 20 - 2026-11-20 04:25

First probe of the propagation gap (Run 18): GPTQ with each layer's Hessians collected from the activations of a
model whose earlier layers are already quantized ("true sequential" GPTQ's inputs), done cheaply as one round of a
fixed point: `quantize hessians` over the quantized GGUF itself (its earlier layers quantized with full-precision
Hessians, so one iteration of the sequential scheme), then the BF16 quantized again with those Hessians. Needed:
Hessian tracking on quantized GGUF layers (`GgufMatMul::begin_track_hessian`, input statistics on `GgufRawMatMul`,
the ik types' layer), and `weight_rank` reading a quantized layer's shape.

```
2B, Q4_K on the linears (embedding BF16)   wiki     docsB    rust
Hessians from the BF16 model               0.02233  0.01582  0.01596
from the quantized model                   0.02195  0.01637  0.01632

35B recipe (pooled 4x calibration)         KLD mean  p99     same top-1
Hessians from the BF16 model               0.01680   0.1738  94.33%
from the quantized model                   0.01710   0.1746  94.31%
```

No gain on either (the 2B mixed within ~3%, the 35B +1.8%). On the 35B the routed down experts' Hessians did not
collect from the quantized model (the fused GGUF expert path does not call the routed-stats hook), so they came from
the BF16 dump; their feedback is noise out of sample anyway (Run 18).

Sequential inputs alone fit `Q X_q` to `W X_q`, the drifted input times the original weight, which does not undo the
drift. The objective that does is the asymmetric one (GPTAQ): `min ||W X_fp - Q X_q||`, solved as GPTQ on the target
`W C H_q^-1` with `C = X_fp X_q^T`, which needs each linear's full-precision and quantized inputs side by side (both
models streamed together), and does not apply to routed experts whose rows route differently in the two models.

## Run 21 - 2026-11-20 04:50

Run 19's leftover: each GPTQ tensor's Hessians and source values loaded before its GPU pass (a down stack 0.16 s of
Hessian files and 0.19 s of weights). A second thread now reads the next GPTQ tensor's Hessians and values (its own
reader, widened to f32) while the GPU quantizes the current one, handed over a one-deep channel.

```
35B recipe, pooled 4x Hessians   before   after
whole                            127 s    106 s   (peak RSS 6.0 GB; same bytes, cmp)
ffn_down_exps (41)               44 s     36 s
ffn_gate/up_exps (82)            44 s     35 s
```

## Run 22 - 2026-11-20 05:15

Question (after a literature pass: GPTAQ is dense-only; MoEQuant, EAQuant, VSRAQ, GEMQ, ProQuant treat routing):
how much does the quantized 35B's routing drift from the BF16 model's?

`INFERENCE_ROUTE_DUMP=<file>` makes `moe_router_topk` append each call's expert ids and weights. Both models streamed
through `quantize hessians` over the evaluation text (the head of wiki.test.raw, ~21K tokens; BF16 GGUF 103 s, the
GPTQ recipe file 20 s), each on its own states, so the drift is end to end: every layer routes on the hidden state
its own quantized predecessors produced.

```
layer   top-8 Jaccard   same top-8 set   same top-1   BF16 route weight on dropped experts
0       0.930           69.6%            94.1%        0.022
10      0.925           68.2%            94.6%        0.025
15      0.883           54.2%            91.3%        0.047
20      0.841           43.0%            88.7%        0.062
25      0.804           33.7%            81.5%        0.084
30      0.807           35.7%            82.4%        0.087
35      0.832           40.8%            87.5%        0.072
39      0.875           52.2%            87.8%        0.052
all     0.865           50.7%            89.4%        0.055
```

Routing drift is large and grows with depth to layers 25-30: two tokens in three change at least one of their eight
experts there, one in five its top expert, though the flips are mostly low-weight experts (5.5% of the routing weight
on average). Whether it is a large share of the KLD needs the quantized model run with the BF16 routes forced.

Two false starts: the BF16 model in the perplexity example maps layers 13-39 to the host, where the first MoE layer
failed (`dtype mismatch in matmul, lhs: F32, rhs: BF16`: host layers run in f32 while an expert stack keeps its bf16,
fixed: the unquantized gather takes its input in the weight's dtype), and after the fix it ran ~3.5 min a chunk
(27 host layers) and stopped with an internal error two chunks in. The streamed collector does the same in 103 s.

## Run 23 - 2026-11-20 06:10

A streamed KLD scorer, `inference quantize kld -m <model> --kld-base <llama-perplexity base>`: the model streamed a
layer at a time through the GPU (as `quantize hessians`) over the base's own chunks, each chunk's second half scored
against the base's records by the engine's own divergence code (`LogitsReference`, `divergence`, now in
`inference_nn::logits_reference` with a base reader), f32 logits. The 35B's BF16 scores in 115 s, a quantized file in
~20 s; the engine could not run the BF16 35B (27 host layers, ~3.5 min a chunk, then an error).

Validated on a small MoE that fits the GPU in BF16 (OLMoE-1B-7B, 64 experts top-8; BF16 GGUF from llama.cpp's
converter, base from llama-perplexity, 40 x 512 of wiki.test):

```
KLD vs llama.cpp BF16 base     engine     streamed
OLMoE BF16                     0.00495    0.00495
OLMoE Q4_K                     0.10956    0.10957
Qwen3.5-2B Q4_K (our base)     0.02233    0.02205
35B GPTQ recipe                0.01680    0.01665
35B BF16                       -          0.00556
```

A false start found the cause of every streamed-vs-engine mismatch so far (Runs 20, 22 and the first forced-route
numbers): the streamed path put BOS at a chunk's first position for any model with a BOS token, where llama-perplexity
(and the engine's scoring) swaps it in only when encoding adds one. Qwen3.6 adds none, so the streamed 35B read a
different first token in every chunk: PPL 7.54 against 5.82, KLD 0.316 for its BF16. Found bottom-up: OLMoE's routes
differed already at layer 0, at position 0 entirely; a per-layer dump showed the layer-0 input (the embedding) off in
row 0 only. (`quantize hessians` still prepends the BOS to its calibration chunks; Runs 14-22's Hessians and route
dumps ran on BOS-prefixed chunks.)

Our BF16 35B is 0.00556 from llama.cpp's BF16: a third of the recipe's 0.0167 is between the two engines, not the
quantization.

Forced routing, measured right (`INFERENCE_ROUTE_DUMP` on the BF16 run, `INFERENCE_ROUTE_FORCE` on the quantized,
the same calls in the same order):

```
35B                     own routes   BF16 routes forced
GPTQ recipe             0.01665      0.01168
Q4_K experts            0.01767      0.01229
```

Above the 0.0056 floor, forcing the BF16 routes takes the recipe from 0.0111 to 0.0061: routing flips are ~45% of
the KLD the quantization adds. Routing agreement over these chunks: top-8 Jaccard 0.903, same top-8 set 60.7%, same
top-1 92.9% (layer 25: 0.871 / 51.2% / 91.2%), weight on dropped experts 3.5%. Router consistency is worth pursuing.

## Run 24 - 2026-11-20 07:40

Where the routing flips come from: each recipe group quantized alone (the rest BF16, the router and GDN small tensors
F32), scored with `quantize kld` against the llama.cpp BF16 base with its own routes and with our BF16's routes
forced. Excess over our BF16 (0.00556):

```
group (alone)              own routes   BF16 routes forced   routing share
exps gate/up (IQ4_KS)      0.00702      0.00393              44%
exps down (IQ4_KS)         0.00508      0.00259              49%
token_embd (Q4_K)          0.00213      0.00011              95%
ssm_out (Q6_K)             0.00204      0.00054              74%
attn_qkv (Q6_K)            0.00174      0.00023              87%
attn_gate (Q6_K)           0.00099      0.00005              95%
full attention (Q8_0)      0.00046      0.00008              83%
shexp (Q8_0)               0.00025      0.00006              76%
output (Q8_0)              0.00002      0.00002              -
```

Outside the routed experts nearly all of a group's harm is through the routing: an error in the residual stream
(the embedding, GDN, attention, the shared expert) costs little at its own outputs and much through the routes it
flips downstream. The routed experts' is half and half. So router consistency is the lever for every group; the
allocation sweep (Run 5) priced the non-expert groups by an effect that is mostly routing.

## Run 25 - 2026-11-20 08:00

GPTAQ on the 35B recipe, its statistics from the twin stream (Run 24's commit): the BF16 model over the 4x
calibration with the GPTQ recipe file as its twin, each dense input paired (`hin` the twin's Hessian, `dxx` the
cross moment), 462 s; the routed down experts' Hessians from the BF16 dump. The twin stream reproduces the two-pass
forced-route KLD exactly (0.01168), so the pairing is right; a GPU test pins GPTAQ below GPTQ on synthetic drift.

```
recipe, quantize kld vs llama.cpp BF16          KLD mean
GPTQ (BF16 Hessians)                            0.01665 / 0.01695 (BOS-free Hessians)
twin Hessians, no residual term (scale 0)       0.01674
GPTAQ, twin routed as BF16, scale 1 / 0.5 / .25 0.01873 / 0.01736 / 0.01728
GPTAQ, twin on its own routes, scale 1 / 0.5    0.01900 / 0.01747
```

A cheaper router-aware fix, the routers refit in closed form so that on the quantized model's inputs they give the
BF16 model's logits (`W' = W (C + H) H^-1` at the MoE input, patched into the F32 `ffn_gate_inp`; the shared-expert
gate too in the first try):

```
statistics           router + shexp gate   router only   router only, half
twin routed as BF16  0.01770               0.01758       -
twin own routes      -                     0.01785       0.01654
```

and routing agreement after the refit barely moves (top-8 Jaccard 0.9025 to 0.9039, same set 60.7% to 61.2%).

Every linear correction of the input drift loses (one within noise). The drift each layer sees is not a linear
function of its input that a weight or router change can undo: much of it is downstream of discrete routing flips,
and a least-squares fit to it reshapes the softmax weights more than it saves routes. One-shot GPTAQ also corrects for
the GPTQ file's drift rather than its own (the paper quantizes sequentially); a fused sequential build could still
differ, but the router refit, which has no such mismatch, does not help either. Next lever for routing: reducing
the drift itself, or a learned (gradient) router-consistency objective (EAQuant, VSRAQ), a larger build.

## Run 26 - 2026-11-20 09:10

Question: how does the allocated GPTQ recipe compare with Unsloth's published quants of about its size, on one
scorer? ik's `llama-perplexity --kl-divergence` (WikiText, c512, 40 chunks, the BF16 base of every run here) on all
three; the recipe rebuilt from `hessians-4x-nobos` (the Run 12 rules, 104 s, 19,904,022,464 B).

```
file                         bytes            KLD                  p99     same top   PPL(Q)/PPL(base)
Unsloth UD-IQ4_NL_XL         19,500,506,080   0.02329 +/- 0.00063  0.2458  93.39%     1.0141
ours, allocated + GPTQ       19,904,022,464   0.01599 +/- 0.00041  0.1712  94.60%     1.0042
Unsloth UD-Q4_K_S            20,893,015,008   0.01604 +/- 0.00047  0.1739  94.28%     1.0037
```

Ours matches UD-Q4_K_S (KLD within error, lower p99, higher top-1 match) at 989 MB (4.7%) smaller, and is 31% under
UD-IQ4_NL_XL at 2% larger. llama-perplexity scores the recipe 0.01599 where our engine scored the same recipe 0.0167-0.0170:
the engine's BF16 floor against this base (0.0056, Run 23) shows up as about 0.001 on a quantized file.

Next: the drift-reduction allocation (bits where routing flips start) to widen the margin at a fixed size.

## Run 27 - 2026-11-20 09:40

Question: does a router fit with the published router-consistency objectives (EAC-MoE's TopK-MSE, VSRAQ's rank and
boundary-margin terms) recover the routing that Run 25's full least-squares refit did not?

`INFERENCE_STREAM_ROUTER_DUMP` (new, with a twin): each layer's MoE input in the BF16 model (`xf`) and the GPTQ
recipe twin on its own routes (`xq`), over the 1x calibration (65K tokens, 40 layers in 3 min, 21 GB). A scratch
fitter (PyTorch) fits each F32 router on 80% of the tokens to the BF16 logits `W x_fp` and reports top-8 Jaccard on
the other 20%:

```
method                                         held-out Jaccard (mean of 40 layers)   same set
none                                           0.9055                                 0.627
full least squares (Run 25's objective)        0.9081                                 0.638
per-row TopK-MSE (k 16), closed form, ridge    0.8802                                 0.558
TopK-MSE + rank + margin, Adam, 400 steps      0.9005                                 0.613
same, 2000 steps, weaker anchor                0.8823                                 0.553
```

The per-row fit lets an expert's logits rise freely on the tokens it should not take (train agreement falls at
layer 30); the gradient fits overfit (train up, held-out down). Why nothing fits, per layer:

```
layer  input drift  logit drift sd  BF16 top-8 margin p10/p50  tokens with a flip  drift R2 from x_q (train/held)
0      5.2%         0.032           0.008/0.054                20%                 0.896/0.882
1      5.8%         0.027           0.007/0.050                21%                 0.611/0.566
2      6.1%         0.034           0.008/0.051                23%                 0.465/0.405
5      6.8%         0.038           0.007/0.045                28%                 0.229/0.143
10     8.9%         0.053           0.008/0.055                30%                 0.122/0.040
20     14.4%        0.086           0.006/0.041                48%                 0.089/0.014
30     14.4%        0.093           0.007/0.046                40%                 0.113/0.024
39     12.2%        0.053           0.006/0.046                33%                 0.106/0.019
```

Past the first layers 1-4% of the router logits' drift is linearly predictable from the quantized input: it is
noise to any router, at a spread (0.05-0.09) the size of the median routing margin, so a third to a half of tokens
flip an expert. Layers 0-4 looked fixable (held-out Jaccard 0.945 -> 0.961 with full least squares), but refitting
only those scores worse with llama-perplexity (recipe 0.01599; layers 0-4 0.01629, layers 0-2 0.01618), and on
WikiText layer 0's agreement moves 0.9161 -> 0.9174 only (all layers 0.9037 -> 0.9054): the early drift is each
token's own embedding error, learned for the calibration's tokens (the random held-out split shares them).

Router-side fixes have nothing to work with on this model at this bit width; the published gains are at W4A4, 3 bits
and NVFP4, and VSRAQ's 4-bit hybrid (Nemotron-3-Nano) moved WikiText PPL 10.79 -> 10.70. The lever is the drift
itself. Next: Hessians weighting each expert row by its routing weight squared (how its error reaches the output).

## Run 28 - 2026-11-20 10:05

Question: do Hessians weighting each routed expert row by its routing weight (MoEQuant's affinity guidance; w^2 is
how a routed expert's error reaches the output) beat the unweighted ones? An env switch scaled each slot's row by
w^(p/2) before the routed Hessian; the 4x calibration again, the recipe quantized from each set, ik llama-perplexity:

```
down experts' Hessians     KLD      p99     same top
unweighted (recipe)        0.01599  0.1712  94.60%
weighted by w              0.01680  0.1807  -
weighted by w^2            0.01700  0.1904  94.61%
```

Both lose (only the down experts change: gate/up take the shared expert's Hessian). Weighting concentrates each
expert's ~2K rows onto its strongly routed few, and these statistics already overfit (Run 17). Switch removed.

## Run 29 - 2026-11-20 10:40

Question: priced inside the recipe rather than alone, which groups should move? Each group's type changed in the
GPTQ recipe (`--reuse` of the rest), ik llama-perplexity. The sweep's prices were isolated, and Run 8 had shown
compounding (Q4_K `token_embd` +0.0005 alone, +0.0014 in the recipe). A second BF16 base on later WikiText text
(40 chunks from byte 600,000; experts on the host, 213 s) checks what the first set finds.

```
move (from the recipe, 19,904,022,464 B)   bytes          KLD set A   KLD set B
none                                       19,904,022,464  0.01599     0.01565
token_embd q4_k -> q6_k                    20,035,135,424  0.01600
token_embd q4_k -> q8_0                    20,158,302,144  0.01568
attn_qkv q6_k -> q8_0                      20,025,919,424  0.01679     0.01593
attn_qkv q6_k -> q5_k                      19,837,175,744  0.01752
attn_gate q6_k -> q8_0                     19,964,970,944  0.01626
attn_gate q6_k -> q5_k                     19,870,599,104  0.01728
ssm_out q6_k -> q8_0                       19,964,970,944  0.01527     0.01530
ssm_out q6_k -> q5_k                       19,870,599,104  0.01905
shexp q8_0 -> q6_k                         19,872,786,368  0.01735
attention q8_0 -> q6_k                     19,831,392,192  0.01654
output q8_0 -> q6_k                        19,780,855,744  0.01615
exps down iq4_ks -> iq4_kss                19,560,089,536  0.01889
exps gate/up iq4_ks -> iq4_kss             19,216,156,608  0.02012
ssm_out q8_0, output q6_k                  19,841,804,224  0.01544     0.01547
  + token_embd q6_k                        19,972,917,184  0.01399     0.01446
  + token_embd q8_0                        20,096,083,904  0.01385     0.01454
  + token_embd q8_0, attention q6_k        20,023,453,632  0.01402
Unsloth UD-Q4_K_S                          20,893,015,008  0.01604     0.01580
Unsloth UD-IQ4_NL_XL                       19,500,506,080  0.02329     0.02152
```

Two upgrades measure worse (`attn_qkv` and `attn_gate` to Q8_0, the first on both sets): a change reshuffles the
near-tie flips of Run 27, so one move's KLD carries about +/-0.0008 of that on 40 chunks, and only larger effects
read. Moves compound: `token_embd` Q6_K alone gives nothing, with `ssm_out` Q8_0 beside it -0.0014 / -0.0010 (two
drift sources feeding the same flips). `ssm_out` is the most sensitive group per byte both ways (+0.0031 for 33 MB
down, -0.0007 / -0.0004 for 61 MB up); `output` Q6_K is a cheap saving.

The new recipe (`ssm_out` Q8_0, `output` Q6_K, `token_embd` Q6_K, the rest as Run 12) measures 0.01399 / 0.01446:
8-12% under the recipe for 69 MB (0.35%), and 8-13% under UD-Q4_K_S at 920 MB smaller. Allocation should be priced
in context (greedy moves from a recipe, measured jointly) and checked on a second text, not added from isolated costs.
