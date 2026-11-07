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
