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
