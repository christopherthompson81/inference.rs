# EXAONE on real checkpoints

EXAONE (#571) on the shared decoder, checked against llama.cpp (970b51ce7, `build_cuda`) and transformers 5.19 on the
RTX 3090. EXAONE 4.0 first: post-norm layers (OLMo 2's names), per-head q/k RMSNorms before RoPE, llama3 RoPE
scaling, and with a sliding window ("LLLG", 4096 on the 32B) the global layers skip RoPE; the 1.2B has no window and
ropes every layer. llama.cpp's `exaone4` GGUFs write the llama3 factors as a `rope_freqs` tensor and the window as
`sliding_window` plus a per-layer `sliding_window_pattern`; the `exaone4` pre-tokenizer is GPT-2's split.

What the decoder needed: a NoPE layer now applies its per-head q/k norms (the RoPE step that carried them is
skipped), and the chat templates' `{% continue %}` (minijinja's `loop_controls`).

## Run 1 - 2026-11-16 17:40

**Question:** does EXAONE-4.0-1.2B (LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF Q8_0, 1.3 GiB) decode as llama.cpp does?

Greedy chat, "Explain in two sentences why the sky is blue.", the same mode on both (`chat_template_kwargs`):

```
thinking off  prompt 23 tokens on both; 24 of 46 equal, max |d logprob| 0.095
              step 24 a near tie: llama.cpp ' than' -1.143 / ' strongly' -1.234, ours ' strongly' -1.155 / ' than' -1.277
thinking on   prompt 19 tokens on both; 31 of 173 equal, max |d logprob| 0.060; step 31 a near tie (' my' / ' it')
```

Left to its defaults the request differs: our `enable_thinking` defaults on, llama.cpp renders this template with
thinking off.

`scripts/gguf_perplexity_parity.sh` (README.md, c=512): llama.cpp 1.7369, ours 1.7806, +2.5% (over its 2%). Against
llama-perplexity's `--kl-divergence-base` (wikitext-2 test, c=512, 20 chunks): PPL 113.31 against 113.34, KLD mean
0.0200, median 0.0085, same top-1 92.45%, where other models sit at ~0.001 between the engines (ERNIE 4.5, Nemotron-H).

So against transformers (bf16, CUDA, the same 20 chunks' tokens; no BOS swap, the GGUF does not add one):

```
against transformers bf16                  KLD mean   median   same top-1   PPL (transformers 112.1)
llama.cpp Q8_0                             0.0277     0.0112   91.2%        113.6
ours Q8_0                                  0.0241     0.0119   91.0%        113.2
ours bf16 (the safetensors checkpoint)     0.0088     0.0056   94.9%        111.9
transformers bf16 on the CPU (5 chunks)    0.0196     0.0034   94.7%
```

The model itself swings that far on rounding: transformers' CPU and CUDA runs differ by more than ours does from its
CUDA run. Our Q8_0 is as near the reference as llama.cpp's, so the 0.020 between the engines is this model's spread,
not a bug. The 1.2B decodes correctly.

## Run 2 - 2026-11-16 18:00

**Question:** does EXAONE-4.0-32B (Q4_K_M, 18.0 GiB; sliding layers beside NoPE global ones) decode as llama.cpp does,
window included?

All 64 layers on the card at 4096 tokens (decode ~40 tok/s). Greedy chat as in Run 1:

```
thinking off  prompt 23 tokens; 48 of 48 equal (the last the end of turn, printed '' by llama.cpp), max |d logprob| 0.144
thinking on   prompt 19 tokens; 200 of 200 equal, max |d logprob| 0.107
```

KLD against llama-perplexity's base (wikitext-2 test). At c=8192 (the window dropping keys) our perplexity example runs
out of memory: it scores a chunk in one pass, 8192 x 102400 f32 logits (3.3 GB) beside the weights, `--cached-prefix
4096` too. So the window was cut to 256 in the file (`gguf_set_metadata.py exaone4.attention.sliding_window`, which
llama.cpp reads over its 32B default), scored at c=2048, and set back to 4096:

```
c=2048, 4 chunks                     PPL ours   PPL llama.cpp   KLD mean   median   same top-1
window 4096 (never drops a key)      7.9386     7.9414          0.00258    0.00104  97.85%
window 256                           8.9663     8.9503          0.00580    0.00200  97.19%
ours at 256 against llama.cpp at 255                            0.00826             97.12%
ours at 256 against llama.cpp at 257                            0.00900             96.55%
```

The window doubles the KLD, but ours at 256 sits nearest llama.cpp's own 256 (an off-by-one in either direction scores
worse), so it is the cut-down model's spread, not a mask that keeps one key more or less.

Tool call (one `get_weather` tool), thinking off and on: `get_weather({"city":"Lyon","unit":"celsius"})`, finish
`tool_calls` (thinking on also writes a lead-in sentence as content, and its reasoning comes back split). With the
result sent back the reply gives Lyon at 17 degrees with light rain (342 prompt tokens).

EXAONE 4.0 decodes correctly.

## Run 3 - 2026-11-16 18:15

**Question:** does EXAONE 3.5 (LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct; a Llama decoder under `transformer.h.N.attn.attention`,
llama3 RoPE, tied embeddings) decode as llama.cpp does, from its `exaone` GGUF (Q8_0, 2.6 GiB) and its checkpoint?

The GGUF's tensors carry llama.cpp's standard names; they are renamed to the checkpoint's as Qwen 1's are (one table
per loader), `rope_freqs` under the stack's root. Its `exaone` pre-tokenizer is llama.cpp's starcoder split.

```
greedy chat (thinking n/a)  prompt 30 tokens on both; 91 of 91 equal (the last the end of turn), max |d logprob| 0.153

wikitext-2 test, c=512, 20 chunks, against llama-perplexity --kl-divergence-base (Q8_0)
                          PPL ours   PPL llama.cpp   KLD mean   median    p99      same top-1
ours, the GGUF            12.3913    12.3770         0.00145    0.00072   0.0133   98.00%
ours, the bf16 checkpoint 12.4020                    0.00199    0.00113   0.0174   96.98%
```

The engines agree at the level other models do (Run 1's 1.2B is the outlier, its own spread). EXAONE 3.5 decodes
correctly.
