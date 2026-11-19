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
