# State-space and recurrent models investigation

Issue #588: attention-free or mostly attention-free language models with constant-size recurrent state: Mamba 1 and 2,
Falcon-Mamba, Jamba, RWKV 6/7 and their Qwen-backbone conversions. Planned as three steps: Mamba-2 on the Mamba-2
layer Granite, Nemotron-H and Falcon-H1 already run; Mamba 1, Falcon-Mamba and Jamba on a new selective scan; RWKV.

## Run 1 - 2026-11-07 10:40

Question: does a pure Mamba-2 stack (`Mamba2ForCausalLM`) run as Granite's hybrid model with only Mamba blocks, and
does it match transformers and llama.cpp?

Reference read: transformers 5.19 `models/mamba2`, llama.cpp 970b51ce7 `conversion/mamba.py` (`Mamba2Model`) and
`src/models/mamba-base.cpp`.

- Layout: `backbone.embeddings`, `backbone.layers.N.{norm, mixer}`, `backbone.norm_f`, `lm_head` (tied on the small
  conversions). The mixer is Nemotron-H's: `in_proj` to z, x, B, C, dt; conv1d with bias; per-head A, D, dt_bias;
  `n_groups` (8 on Mamba-Codestral); a gated RMSNorm; `out_proj`. No attention or feed-forward anywhere.
- The gated norm runs after the gate and per group, whatever `norm_before_gate` says (Mamba-Codestral's config says
  true): mamba_ssm's fused kernel and llama.cpp both do. transformers' unfused fallback (its CPU path) normalizes the
  whole inner width instead of per group, so for `n_groups` above one a CPU transformers run is not a reference; the
  130M (one group) is.
- `time_step_limit` is `[0.0, Infinity]`, bare in the config (Python's json); state-spaces' HF conversions write no
  `architectures`, only `model_type`.

Built: `Mamba2` (`mamba2.rs` and its loader), the `mamba2` GGUF (config from the ssm metadata, Nemotron-H's tensor
bindings under `backbone`). Fixes the first loads needed:

- The companion-model check parsed `config.json` without rewriting non-finite literals: "expected value at line 1
  column 270" (the `Infinity`). It now reads them as the loader does.
- Auto-detection required `architectures` ("missing field `architectures`"); a config without them now resolves by
  `model_type`, as transformers' Auto classes do.
- The paged-attention planner divided by the per-token KV bytes, zero with no attention layer ("attempt to divide by
  zero" at paged_attention/mod.rs:691). Mamba-2 reports no paged-attention support; its state is the hybrid cache's.

```
PYTHONPATH=bindings/python INFERENCE_NATIVE_DIR=target/debug python3 scripts/causal_lm_parity.py mamba2-130m
  183 tokens: mean KL 1.74e-08 (worst 5.75e-07), top-1 agreement 1.000, max logit diff 0.0005; greedy 24 identical
scripts/gguf_perplexity_parity.sh llama-perplexity mamba-codestral-gguf          (README.md, ctx 512, CUDA)
  Q4_0  reference 1.2958  ours 1.2922  (-0.0028)
  Q8_0  reference 1.2989  ours 1.2939  (-0.0038)
```

(`scripts/falcon_h1_parity.py` is now `scripts/causal_lm_parity.py`; nothing in it was Falcon's.)

Mamba-Codestral 7B Q4_0 (gabriellarson's GGUF) on the 3090, against llama.cpp 970b51ce7 `llama-bench`:

| | ours | llama.cpp |
|---|---|---|
| decode, 400 tokens | 127.5 / 127.8 tok/s | tg128 129.40 |
| prefill | 5,500 tokens: 3,951 / 4,023 tok/s | pp4096 3,514.99 |

Four requests at once decode as each does alone (two identical prompts give the same text as a lone request).

Seen on the way, not Mamba's: a completion's leading whitespace is trimmed (`Sequence::get_final_delta` and the
streaming delta trim the first piece), so a code completion that starts with an indent loses it ("    if n == 0"
comes back "if n == 0"); llama.cpp's server keeps it.

Next: step 2, Mamba 1 / Falcon-Mamba (a scan with a full `A` of d_inner x d_state and per-channel dt), then Jamba.

## Run 2 - 2026-11-07 11:30

Question: why did the full CUDA suite fail `cuda_chunked_ssd_scan_holds_a_steep_decay` (the Falcon-H1 NaN test from
#844, not touched here), and is the clamp #844 added enough?

```
scripts/local_ci.sh --lint --tests --cuda
  case cuda_chunked_ssd_scan_holds_a_steep_decay: FAILED: state: max error 4.0908213 at scale 5.4774537
INFERENCE_TEST_CASE=steep_decay ... cuda_cases, 8 runs
  2 of 8 failed ("max error 4.97 at scale 4.97", "max error 5.26 at scale 5.73")
```

The test's inputs are random per run, so #844's five passing runs were luck. The clamp stopped the overflow (no NaN);
what remains is precision. The kernel takes each position pair's decay as exp(A * (cs[t_out] - cs[t_in])) over f32 prefix
sums of dt. With a chunk's sum in the hundreds, one ulp is ~3e-5, and times A = -2e8 an exponent of -6000: a span of
near-zero steps, whose true decay is ~1, comes out 0 when the rounding puts the later sum one ulp above the earlier.
The sequential scans (CPU, decode, llama.cpp's ssm_scan) multiply per-step factors and never form the difference.

Fix: the prefix sums carry their rounding error (two-sum, `__fadd_rn` / `__fsub_rn` so fast-math cannot contract or
reorder them), stored beside the sums (`cs` is now [k][h][2][L]); the mask, the decay to the chunk's end, the chunk's
decay and the finish all subtract (hi, lo) pairs, ~48 bits. The clamp at zero stays for what rounding is left.

```
20 runs of the SSD CUDA cases: 0 failures
Mamba-Codestral Q4_0 prefill, 5,500 tokens, no prefix cache: 3,898 / 3,988 / 3,991 tok/s (before: 3,951 / 4,023)
gguf_perplexity_parity.sh mamba-codestral-gguf: Q4_0 1.2927 (ref 1.2958), Q8_0 1.2940 (ref 1.2989)
```

`--size`: +0.77 MiB against a baseline last written at #812, 33 merges back (`libinference_ffi.so` .text +0.58 MiB;
the kernel library +0.01 MiB); rewritten here.

## Run 3 - 2026-11-07 12:20

Question: step 2's first half. Do Mamba 1 (`MambaForCausalLM`) and Falcon-Mamba (`FalconMambaForCausalLM`) run on
Granite's Mamba layer, and what does their scan need?

Reference read: transformers 5.19 `models/mamba` and `models/falcon_mamba`, llama.cpp 970b51ce7 `conversion/mamba.py`
(`MambaModel`) and `src/models/mamba-base.cpp`.

- The mixer: `in_proj` to x and z, the conv over x alone, `x_proj` to dt (`time_step_rank` wide), B and C, `dt_proj`
  (with bias) back to the channels, softplus, the scan, `y * silu(z)`, `out_proj`. A is `(channels, d_state)`, one
  rate per state item, and there is one B and C per token for all channels. Falcon-Mamba normalizes dt, B and C with
  weightless RMSNorms (`mixer_rms_eps`, 1e-6; llama.cpp uses the layer-norm eps, 1e-5) before `dt_proj`.
- Falcon-Mamba-7B's config says `expand` 16; its width is `intermediate_size` (8192 = 2 x 4096).
- llama.cpp writes both as `mamba`, Falcon-Mamba with `ssm.dt_b_c_rms` true, and `ssm_dt` as `dt_proj`'s weight and
  bias (Mamba-2's `ssm_dt` is the bias alone).

Built: Mamba 1 as Granite's Mamba layer with a `Selection` (x_proj, dt_proj, the optional norms): each channel a head of
width one in one group, so the conv and SSM state, the recurrent pool, prefix snapshots and the decode paths are
Mamba-2's. The scans (CPU, CUDA sequential) take A as `(heads, d_state)`; the chunked SSD path, Vulkan's step kernel and
Metal's scan take one A per head and are skipped for it. `mamba.rs` and its loader serve both classes; the `mamba` GGUF
binds to transformers' names.

```
causal_lm_parity.py mamba-130m          183 tokens: mean KL 9.23e-09, top-1 1.000, max logit diff 0.0014; greedy 24 identical
causal_lm_parity.py falcon-mamba-tiny   180 tokens: mean KL 8.10e-09, top-1 1.000 (the dev checkpoint repeats one token)
gguf_perplexity_parity.sh falcon-mamba-7B-instruct-Q4_K_M:  reference 1.4251, ours 67.9467
gguf_perplexity_parity.sh mamba-130m-f32 (llama.cpp's convert):  reference 1.7708, ours 88.9686
```

The 130M's GGUF failed where its safetensors matched: the binding, not Falcon's norms. Every GGUF tensor, converted as
the binding does, equals the checkpoint's (A's max diff 1.2e-7), and the loaded A, D, dt bias and conv matched under
both loads. Probing each projection with one input: `dt_proj` came out 2581.6 apart, the dt bias's sum. A GGUF linear
carries the bias its file has whatever the loader asks, so the bias was added in `dt_proj` and again in the scan.
`dt_proj` now keeps its bias on both routes and the scan's dt bias is zero, as transformers' slow path has it.

```
mamba-130m-f32   reference 1.7708   ours 1.7600   (-0.0061)
falcon-mamba-7B  reference 1.4251   ours 7.0476   (+3.9453)
```

Falcon-Mamba still off: its layers loaded without the norms. The synthesized config named `falcon_mamba` as its model
type, which the reload identity then overwrote with the loader's (`mamba`); the registry sees GGUF keys, not values, so it
cannot pick the loader by `dt_b_c_rms` either. The config now carries the flag (`dt_b_c_norm`). Also needed: the GGUF
tokenizer's `falcon` pre-type, llama.cpp's three regexes.

```
falcon-mamba-7B   reference 1.4251   ours 1.4291   (+0.0028)
falcon-mamba-tiny-f32 (convert)   reference 125367.9   ours 125430.0   (+0.0005)
```

Speed (Falcon-Mamba 7B Instruct Q4_K_M, 3090, no prefix cache) against llama.cpp 970b51ce7 `llama-bench`:

| | decode, 400 tokens | prefill, 3,901 tokens |
|---|---|---|
| warp per channel (Mamba-2's sequential kernel, 16 of 32 lanes) | 90.0 / 91.9 | 1,945 / 1,973 |
| a thread per channel, B and C shared per step | 100.1 / 100.1 | 3,948 / 3,997 |
| and the dt/B/C norms in the model's dtype (no f32 casts) | 109.0 / 109.8 | |
| llama.cpp | tg128 107.46 | pp4096 3,323.99 |

The norms without the casts: perplexity 1.4296 (1.4291 with them). "What is the capital of France?" -> "The capital of
France is Paris."

Next: Jamba (Mamba 1 layers with dt/B/C norms that have weights, attention without RoPE, MoE).

## Run 4 - 2026-11-07 13:15

Question: step 2's second half. Does Jamba (`JambaForCausalLM`) run on what Runs 1-3 built?

Reference read: transformers 5.19 `models/jamba`, llama.cpp 970b51ce7 `conversion/jamba.py`.

- Layer `i` is attention when `i % attn_layer_period == attn_layer_offset`, Mamba 1 otherwise; its feed-forward is a MoE
  when `num_experts > 1` and `i % expert_layer_period == expert_layer_offset`, a SwiGLU MLP otherwise. Norms
  `input_layernorm` and `pre_ff_layernorm`, `final_layernorm`, `model.embed_tokens`.
- Attention has no position embedding (the RoPE helper is imported but never applied).
- The Mamba 1 mixer normalizes dt, B and C with weighted RMSNorms (`dt_layernorm`, `b_layernorm`, `c_layernorm`, at
  `rms_norm_eps`), where Falcon-Mamba's have no weights.
- The MoE: `router` (no bias), a softmax top k with no renormalizing, SwiGLU experts, no shared expert.
- llama.cpp writes no periods: attention layers are those with KV heads in a per-layer `head_count_kv`, MoE layers
  those with stacked `ffn_*_exps`; `ssm_dt_norm`, `ssm_b_norm` and `ssm_c_norm` are the three norms.
- Checkpoints: Jamba-tiny-dev (8 experts), AI21-Jamba-Reasoning-3B and Jamba2-3B (one expert, so a dense MLP on every
  layer; an official Q4_K_M GGUF). Jamba Mini 1.7 (52B) is gated, so not tried.

Built: `jamba.rs` and its loader; the hybrid MoE takes Jamba's routing (`MoeRouting::Jamba`: the `router`, softmax top
k unnormalized, SwiGLU experts, no shared expert, no selection bias) beside Nemotron-H's; Mamba 1's norm is weighted or
not (`DtBcNorm`); the `jamba` GGUF binds through the Mamba 1 binding with attention and feed-forward names, its periods
from the KV heads and the expert tensors.

```
causal_lm_parity.py jamba-tiny-dev            184 tokens: mean KL 7.35e-09, top-1 1.000; greedy 24 identical
gguf_perplexity_parity.sh jamba-reasoning-3b-Q4_K_M:   reference 1.9099, ours 1.7667 (-0.0750)
```

Ours under the reference by 7.5% is the better side, but large. Tokenizing 3 KB of README.md: llama.cpp 1,130 tokens,
ours 1,107 with no BOS, the model's own tokenizer.json 1,108 with its BOS, otherwise ours exactly. llama.cpp's SPM path
(the vocabulary's scores, `add_space_prefix` false) splits `media="` into three tokens where the tokenizer.json has one,
so llama.cpp scores another text. The missing BOS was ours: the GGUF's SentencePiece BPE built no post-processor, so
`add_bos_token` (true here) was ignored. It now adds the BOS as the BPE path already did (a test beside it).

```
gguf_perplexity_parity.sh jamba-reasoning-3b-Q4_K_M:   reference 1.9099, ours 1.4743 (-0.2281)
gguf_perplexity_parity.sh mamba-codestral-gguf (SPM, add_bos):  Q4_0 1.2950 (ref 1.2958), Q8_0 1.2983 (ref 1.2989)
causal_lm_parity.py jamba-reasoning-3b (BF16 checkpoint, F32, CPU):  184 tokens: mean KL 5.47e-09, top-1 1.000,
  max logit diff 0.0045; greedy 24 identical
gguf_perplexity_parity.sh jamba-tiny-dev-f32 (llama.cpp's convert, stacked experts):  reference 1.6847, ours 1.6870
```

The 3B matches transformers, so the gap to llama.cpp is its tokenization. Jamba Reasoning 3B Q4_K_M on the 3090: 262K
of context, decode graphs captured; it reasons through "How many positive divisors does 360 have?" coherently.

| | ours | llama.cpp `llama-bench -fa 1` |
|---|---|---|
| decode, 400 tokens | 333.1 / 332.8 tok/s | tg128 255.19 |
| prefill, 3,902 tokens (no prefix cache) | 8,788 / 8,808 tok/s | pp4096 9,258.39 |

The tiny test (`mamba_tiny`) takes a Jamba kind: Mamba 1 + MoE, NoPE attention + MLP, Mamba 1 + MoE, which decodes as
it prefills, resumes from the prefix cache, and scores on CUDA as on the CPU.

Next: step 3, RWKV 6/7.
