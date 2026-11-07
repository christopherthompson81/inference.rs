# Falcon-H1 investigation

Issue #578: TII's Falcon-H1 (`FalconH1ForCausalLM`, GGUF `falcon-h1`), dense hybrids from 0.5B to 34B where every layer
runs attention and a Mamba-2 mixer side by side on one norm, then a SwiGLU MLP, with muP multipliers on the
activations.

## Run 1 - 2026-11-07 06:00

Question: can Falcon-H1 run as Granite's hybrid model (the Mamba-2 path Nemotron-H uses, with its CUDA SSD prefill,
decode graphs and mid-chunk captures), and what does it need beyond it?

Reference read: transformers 5.19 `models/falcon_h1` and llama.cpp 970b51ce7 `conversion/falcon_h1.py`.

- Layer: `input_layernorm`; `mamba(h) * ssm_out` plus `self_attn(h * attention_in) * attention_out` into the residual;
  `pre_ff_layernorm`, `feed_forward` (SwiGLU with the gate scaled by `mlp_multipliers[0]` inside the activation, the
  down projection by `mlp_multipliers[1]`). Keys scaled by `key_multiplier`, softmax at 1/sqrt(head_dim), RoPE
  NeoX-style at theta 1e11. Embeddings times `embedding_multiplier`, logits times `lm_head_multiplier`.
- Mamba: Mamba-2 with `mamba_d_ssm` as its width (not `expand * hidden`), the input scaled by `ssm_in` and its
  projection's z, x, B, C and dt slices by the five `ssm_multipliers`. With `mamba_rms_norm` false (the 0.5B) the output
  is `y * silu(z)`, no gated norm.
- llama.cpp folds every multiplier into the weights at conversion, with no metadata for them; it ropes NeoX-style and
  does not permute q/k. TII's GGUFs match: against the 0.5B checkpoint times its multipliers, every tensor compared
  within BF16 rounding (`ssm_in` rel 0.0017, the rest 0.0000; `ssm_a` is `-exp(A_log)`). `blk.N.ffn_norm` has no
  `.weight` suffix. The tokenizer's pre-type is `falcon-h1`, which llama.cpp reads as Llama 3's split.

Built:

- A `Parallel` block in Granite's hybrid model: attention and a Mamba block on one norm, the attention output added
  beside the Mamba output. The hybrid cache gives each layer one kind of state, so a parallel block pages its
  attention at its own index and keeps its recurrent state in a cache layer after the stack's (the paged config lists
  those as unpaged); the loader maps the stack's layers to devices, the cache's slots follow them. Forward-time muP
  scales (skipped at 1): attention input, keys and output; the Mamba projection as one vector over z, x, B, C, dt and
  its output; the MLP gate and down projection. The Mamba gated norm is optional.
- GGUF: `falcon-h1` registered (HalfSplit pairing, the SSM layout), config synthesis from the metadata (the multipliers
  at one, `mamba_rms_norm` from an `ssm_norm` tensor), bindings renaming to Falcon's names and the Granite SSM
  conversions, `falcon-h1` / `falcon3` tokenizer pre-types as Llama 3's.

```
serve -m falcon-h1-0.5b                       (safetensors, BF16, CUDA)
  "The capital of France is Paris."; a docstringed is_prime; ~260 tok/s decode
serve --format gguf -m falcon-h1-0.5b -f Falcon-H1-0.5B-Instruct-BF16.gguf
  before: "This isntations / This isntations ..."
```

The GGUF route read the checkpoint's `config.json` beside the file (trace log: `embedding_multiplier: 5.656854`,
`ssm_multipliers: [...]`), so the folded weights got the multipliers a second time. A GGUF loaded alone synthesizes its
config, but any checkpoint config beside it (or `--tok-model-id`) would break it. Fix: a checkpoint config given to a
`falcon-h1` GGUF drops the nine multiplier keys. After: "The capital of France is Paris.", the same is_prime.

## Run 2 - 2026-11-07 06:40

Question: does the 0.5B match transformers, and how does the 7B GGUF run on the 3090 against llama.cpp?

```
PYTHONPATH=bindings/python INFERENCE_NATIVE_DIR=target/debug python3 scripts/falcon_h1_parity.py falcon-h1-0.5b
  198 tokens (past the 128-token Mamba chunk): mean KL 1.55e-07 (worst 2.44e-06), top-1 agreement 1.000,
  max logit diff 0.0092; greedy 24 tokens identical
```

7B Instruct Q4_K_M (`tiiuae/Falcon-H1-7B-Instruct-GGUF`, 4.3 GiB; 44 parallel blocks, Mamba head dim 128, state 256,
the gated Mamba norm): loads with 262K of context on the GPU, decode graphs captured; "Canberra ... chosen as the
capital in 1908 ...". Decode 115.7 - 116.0 tok/s over three 400-token runs; llama.cpp 970b51ce7 `llama-bench -fa 1`
tg128 112.33 tok/s. Prompt: 4,126 tokens in 1.2 s end to end (3,484 tok/s with the request around it); llama.cpp
pp4096 4,054 tok/s.

A third prompt (4,263 tokens) failed: "invalid CUDA top-1 output: max_logit=NaN". Not the prefix cache
(`--prefix-cache-n 0` failed the same), not the prompt chunking (one 8K chunk failed too), not the length (208 tokens
failed, 143 and 291 passed); the 0.5B (BF16 and its GGUF) never failed. Bottom-up, with per-block finite checks (the
first used `max`, which skips NaN; a sum catches it): every block finite until one Mamba output; inside it the scan's
inputs (x, B, C, dt, the state) finite and its output state NaN, from the chunked SSD path (prompts of 128 tokens or
more; shorter ones take the sequential scan).

Cause: the 7B's A reaches -2.4e6 (layer 38) and -2.3e8 (layer 43). The SSD kernel's mask takes
`exp(A * (cs[t_out] - cs[t_in]))` over a chunk's prefix sums of dt, which its tree-ordered sum rounds; where the steps
between two positions are near zero (softplus of a very negative dt), the later sum can land a few ulp under the earlier
one, and `A` of -2e6 times -6e-5 is exp(+120), inf in f32, times C.B gives NaN. The same holds for each position's
decay to the chunk's end. Fix: both exponents clamped at 0, which they are in exact arithmetic. The Vulkan shaders sum
`A * dt` sequentially, which keeps the sums monotone, so they need no change.

Test (`cuda_chunked_ssd_scan_holds_a_steep_decay`): A of -2e8 and -2.4e6 among ordinary heads, every other step's dt
about 1e-13 and the rest random around 3, 300 tokens against the CPU scan. Without the clamp it failed three runs of
three ("state is not finite" twice, "state: max error 5.0 at scale 5.6"); with it five of five pass. The 7B then served
208, 2,811, 4,126, 4,263, 6,727 and 12,499-token prompts, and 300 tokens at 115.9 tok/s.

Tiny tests (`falcon_h1_tiny`, F32 on the CPU): a checkpoint's multipliers applied on the activations score as its
weights with them folded in (with and without the gated norm), and a gated-norm model decodes a cached repeat the same.
The first run of the second failed: "kv-cache: requested capacity (547) above max seq len (512)": the hybrid cache grows
its KV cache by a fixed step past the prompt, beyond a 512-position config; the fixture now has 4,096.
