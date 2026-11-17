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

## Run 4 - 2026-11-16 19:00

**Question:** does EXAONE 4.5 (LGAI-EXAONE/EXAONE-4.5-33B-GGUF: Q4_K_M text, 18.7 GiB, F16 projector, 2.4 GiB) read
images as llama.cpp and transformers do?

EXAONE 4.5 is Qwen2.5-VL's windowed vision tower with GQA attention (32 query heads, 8 kv) beside the EXAONE 4 text
stack, on plain 1D positions (no M-RoPE), with `<vision><|image_pad|></vision>` placeholders. It runs on the Qwen-VL
path: the tower moved to `inference_nn::vision` and took GQA, the Qwen-VL inputs processor takes the delimiters and
whether to build M-RoPE positions from its spec, and the model ropes and masks as a text stack does. The GGUFs bind as
Qwen2.5-VL's projector under `model.visual`, the text as EXAONE 4 under `model.language_model`, the `nextn` block as
the checkpoint's root-level `mtp.*`.

Probe: the repository's sample photo (3744x5616, a person in a hanbok making kimchi) resized to 448x672 (384 image
tokens), "Describe this image in one sentence.", greedy, thinking off and on. llama.cpp needs `--no-mmproj-offload`
(the projector does not fit beside the weights), ours spills layers 60-63 to the CPU.

```
                prompt tokens (transformers' processor: 403 off / 401 on, 384 pads)
llama.cpp       402 / 400
ours            403 / 401

thinking off    5 of 34 equal, max |d logprob| 0.226; step 5 a near tie on both sides:
                llama.cpp ' ' -0.607 / ' traditional' -0.878, ours ' traditional' -0.704 / ' ' -0.778
                llama.cpp: "A person in a light blue hanbok is preparing kimchi in a dark bowl, while a plate of freshly
                           made, vibrant kimchi sits in the foreground on a white tablecloth."
                ours:      "A person in a light blue traditional Korean hanbok is preparing kimchi in a dark bowl, while a
                           plate of freshly made, vibrant red kimchi sits in the foreground on a white tablecloth."
thinking on     19 of 80 equal, max |d logprob| 0.108; step 19 a near tie (' foreground' -0.675 / ' image' -0.714)
```

Our prompt is transformers' token for token in count; llama.cpp's is one shorter (its media marker leaves out a
newline the template writes), so the image's surroundings already differ there. The text side alone (the same GGUF as
a text model, wikitext-2 c=2048, 4 chunks): PPL 7.4759 against llama.cpp's 7.4640, KLD 0.00281, same top-1 97.75%.

What the run turned up: the GGUF's `exaone-moe` pre-tokenizer (now read), its 65-block pattern covering the MTP
block (the text config keeps the stack's 64), the vision delimiters marking each image's span for the prefix cache
(Qwen's were looked up), and drafting through the Qwen-VL wrapper, which built its own masks for the verify step's
rows: a flat-position model now masks as its text stack does, and the tiny MTP test keeps greedy output.

## Run 5 - 2026-11-16 19:55

**Question:** does EXAONE 4.5's MTP layer draft on the real checkpoint?

`--mtp` needs every layer on the card (PagedAttention), which the Q4_K_M text GGUF with the projector misses; the
IQ4_XS text GGUF (16.8 GiB) fits with it at 2048 tokens. Greedy output with `--mtp --mtp-n-predict 2` equals plain
(46 of 46 text tokens, 36 of 36 image tokens), but no draft is accepted: 0 of 86, and 0 of 43 at one draft a step.

The head is vLLM's `Exaone4_5MultiTokenPredictor` (vllm/model_executor/models/exaone4_5_mtp.py): `fc` over
`concat(pre_fc_norm_embedding(embed(token p+1)), pre_fc_norm_hidden(h_p))`, one `Exaone4DecoderLayer` (post-norm, its
first layer, so sliding with RoPE), `mtp.norm`, the shared `lm_head`, `h_p` the target's final-normed output. The
drafts are on topic but Korean (' 산란' "scattering" where ' by' follows "scattered", ' 파장' "wavelength" for
' violet'); with the hidden half zeroed they turn to English function words, so the hidden half carries them.

Checks, none of which moved acceptance off zero:

```
GGUF blk.64 against the checkpoint's bf16 mtp.* (shard 2's byte range)   cos 0.997 (IQ4_XS), norms 1.0000
the dumped hiddens through the head                                        the target's next tokens (last row exact)
a numpy reference of vLLM's head over those hiddens, 39 rows of plain English text:
  as specified (GGUF weights or the checkpoint's bf16 MTP weights)        0/39
  pre-norm hidden, current-token embedding, halves swapped, norms swapped  0/39
  no attention to earlier rows                                             3/39
  a pre-norm reading of the layer                                          0/39
```

The engine computes what vLLM's code describes; that description, over these weights and hiddens, predicts nothing
here. Every probe above ran with thinking off or on raw text, which turns out to be the point: a self-distillation
project on the stock head (github.com/gnu-gnu/exaone45-mtp-selfdistill, vLLM, FP8, 3 drafts) measures about 30%
acceptance on thinking requests and 0.8% on non-thinking ones, where MTP runs at 0.85x of plain. The head was trained
on reasoning traces.

With thinking on (the sky prompt, 400 tokens, IQ4_XS with the projector, 2048 tokens):

```
                  accepted          decode tok/s
plain                               38.5
--mtp, 1 draft    161 of 235 (69%)  57.2  (1.49x)
--mtp, 3 drafts   156 of 408 (38%)  51.6  (1.34x)
```

Drafted output parts from plain at near ties (164 of 382 and 82 of 297 tokens equal; the splits 0.04 apart), as
batched verify rows round apart from one-by-one decode. EXAONE 4.5's MTP works as the references describe; it pays
only while the model thinks, so `--mtp` suits thinking requests and costs speed on others.

## Run 6 - 2026-11-16 22:40

**Question:** does K-EXAONE (mradermacher/K-EXAONE-236B-A23B-i1-GGUF, i1-IQ3_XS, 90.2 GiB) run as llama.cpp does?

The file holds more than the 24 GiB card, so ours keeps the experts in host memory (20 a layer cached on the GPU, the
CPU running the rest) and llama.cpp runs with `--cpu-moe`. Its `sliding_window_pattern` lists the 48 decoder layers
only, where EXAONE 4.5's also covers the MTP block; the config builder took the block count and refused the file, and
now takes either length.

Greedy chat, the sky prompt (24 tokens thinking off, 22 on, the same on both):

```
ours, KV cache auto (q3: the plan quantizes it when experts leave the GPU)
thinking off    14 of 160 equal, max |d logprob| 0.146; ours ' although' -1.18, llama.cpp ' which' -0.73
thinking on     23 of 240 equal, max |d logprob| 0.244
ours, --pa-cache-type none (bf16)
thinking off    119 of 160 equal, max |d logprob| 0.107; step 119 a near tie (' because the' -0.876 / '.' -1.050)
thinking on     24 of 240 equal, max |d logprob| 0.027; step 24 a near tie (' it' -0.676 / ' it has' -0.721)
```

The early split was the q3 cache alone; with the model's dtype the engines part only at near ties.

KLD, wikitext-2 test, c=512, 20 chunks, against llama-perplexity's base (experts on the GPU through its op offload),
our KV cache bf16:

```
                                              PPL      KLD mean   median    p99      same top-1
llama.cpp, the base's own run                 14.4694
ours, 20 experts a layer cached               14.7832  0.03752    0.00853   0.469    91.71%
ours, 4 cached (most experts on the CPU)      14.8113  0.04217    0.00975   0.487    91.00%
llama.cpp --cpu-moe --no-op-offload           14.7158  0.03980    0.01019            91.18%
```

(The example prints the base's PPL from its stored log-probs, 14.0444; llama-perplexity's own count gives 14.4694.)
0.0375 is far above ERNIE 4.5 21B's 0.0011, but llama.cpp against itself, only its expert kernels moved from the GPU
to the CPU, spreads as far: at IQ3_XS this model's routing turns rounding into different experts. Ours sits inside
that spread. A first run with `--kl-divergence --n-cpu-moe 44` scored KLD 0 exactly: op offload copies host experts
to the GPU for large batches, so it ran the base's kernels.

MTP (`--mtp --mtp-n-predict 1`, bf16 cache):

```
                  accepted        decode tok/s
plain                             3.6-4.1
thinking off      61 of 96 (64%)  3.6
thinking on       97 of 140 (69%) 3.9
```

Unlike EXAONE 4.5's head, K-EXAONE's drafts without thinking too. It buys nothing here: decode is bound by reading
experts from host memory, and a verify step's two rows touch more of them. Drafted output parts from plain at near
ties (step 4, 0.05 apart; step 104, 0.03), the batched verify rounding of Run 5. The MTP layer is NoPE, as the
config's `mtp_layer_types` gives it (llama.cpp does not run the `nextn` block).
