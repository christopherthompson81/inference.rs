# Nemotron-H text KLD against llama.cpp

Our runs of the Nemotron 3.5 Transcribe text GGUF score 3x llama.cpp's KL divergence from the same BF16 base. Setup:
the perplexity example (`examples/rust/advanced/perplexity`) with `--kld-base
/mnt/data/models/nemotron-transcribe-gguf/bf16-wiki-c512-ch40.kld` (ik_llama.cpp's `llama-perplexity
--kl-divergence-base` over wikitext-2, 40 chunks of 512, the second half of each scored: 10,200 positions), on
`Nemotron-3.5-Transcribe-text-IQ4_NL-down-Q5_0.gguf` unless noted. The base is read the same way by both tools: both
report PPL(base) 12.198913 (below the 12.7692 llama.cpp printed while writing the base, since log-probs more than 16
below a row's top are stored at that floor).

## Run 1 - 2026-11-02 06:35 (local clock)

**Question:** is the gap ours, and on which device and precision?

**Commands:** `perplexity --file README.md --gguf <IQ4 GGUF> --kld-base <base> [--dtype f32] [--cpu]` (the `--dtype`
and `--cpu` flags added to the example for this), master with chunked SSD (#771) and with SSD forced off.

**Findings:**

| run | PPL | mean KLD | median KLD | p99 KLD | same top-1 |
|---|---|---|---|---|---|
| llama.cpp CUDA (`kld_down_q5_0.log`, nemotron_transcribe Run 17) | 12.6669 | 0.02498 | 0.00936 | - | 93.22 % |
| ours CUDA, BF16 activations, SSD | 12.7827 | 0.07516 | 0.02778 | 0.689 | 88.57 % |
| ours CUDA, BF16 activations, sequential scan | 12.6964 | 0.07435 | 0.02699 | 0.629 | 89.02 % |
| ours CUDA, F32 activations, SSD | 12.7439 | 0.07525 | 0.02746 | 0.684 | 88.61 % |
| ours CPU, sequential scan | 12.6787 | 0.07477 | - | - | 88.81 % |

The gap is the same on both devices, both scans and both activation precisions, so it is not CUDA kernels, the SSD
change or BF16 rounding: either our model computes something llama.cpp's does not, or we decode some weight type
differently. `--cpu` on the BF16 GGUF with the default dtype was killed for memory (F32 weights, ~118 GB of 125).

**Next:** the BF16 GGUF on the CPU with `--dtype bf16`: near zero means the quantized weights' decode; ~0.05 means
the model's math.

## Run 2 - 2026-11-02 07:00 (local clock)

**Question:** what in the model's math differs from llama.cpp's?

**Reading:** llama.cpp's `nemotron_h_moe` graph (`src/models/nemotron-h.cpp`, `mamba-base.cpp`) against ours: the
same router (sigmoid, bias for selection only, normalized, x2.5), ReLU-squared experts and shared expert, grouped
RMSNorm after `y * silu(z)`; but its `ggml_ssm_scan` applies softplus with no clamp on dt, while ours clamped prompts'
dt at `time_step_min` (0.001; decode at 0), following transformers 5.19's port (`modeling_nemotron_h.py`:
`self.time_step_limit = (config.time_step_min, float("inf"))`). NVIDIA's own `modeling_nemotron_h.py` (the Nemotron
3 Nano repo's remote code) clamps with `config.time_step_limit` ([0.0, 1e30] in this config) on both paths and uses
`time_step_min` only to draw the dt bias at init. This model's dt runs far below 0.001 (dt + bias down to -15 before
softplus), so the floor raised many heads' steps by orders of magnitude.

**Test (prompt floor forced to 0, CUDA):** PPL 12.6166, KLD mean 0.02672, median 0.00946, p99 0.260, same top-1
93.18 %, llama.cpp's numbers (0.02498 / 0.00936 / 93.22 %).

**Change:** the Mamba dt clamp is the config's `time_step_limit` for prompts and decode alike, (0, max) when absent
(GGUF loads), and `time_step_min` is no longer read. Unit case
`the_dt_clamp_is_the_time_step_limit_not_the_init_minimum`.

**Result:**

| run | PPL | mean KLD | median KLD | p99 KLD | same top-1 |
|---|---|---|---|---|---|
| llama.cpp CUDA | 12.6669 | 0.02498 | 0.00936 | - | 93.22 % |
| ours CUDA (SSD), fixed | 12.6166 | 0.02672 | 0.00946 | 0.260 | 93.18 % |
| ours CPU, fixed | 12.6691 | 0.02670 | 0.00949 | 0.292 | 92.76 % |

The BF16-on-CPU run (`--dtype bf16`) was stopped once this explained the gap. The audio KLD figures so far
(nemotron_h_cuda_speed Run 4, 0.0049 / 0.0055) were against our own BF16 CPU reference, which had the same floor, so
they need a new reference before they guide the audio recipe.
