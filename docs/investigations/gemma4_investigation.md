# Gemma 4 on real checkpoints

Gemma 4 had only tiny random-weight tests. Before adding an EAGLE-3 drafter for it (#584), its decoding is checked
against llama.cpp (970b51ce7, `build_cuda`) on the same GGUF, greedy, on the RTX 3090. Target:
google/gemma-4-31B-it-qat-q4_0-gguf (Apache 2.0; 16.4 GiB text, 1.1 GiB projector).

Probes: a raw completion ("The capital of France is Paris. The capital of Germany is", 24 tokens) and a chat request
("Explain in two sentences why the sky is blue.", 80 tokens, top-2 logprobs).

## Run 1 - 2026-11-16 11:40

**Question:** does the 31B decode as llama.cpp does, and does it load as llama.cpp loads it?

```
llama.cpp  raw  ' Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of ...'  13 prompt tokens
           chat '*   Topic: Why the sky is blue. ...'                                                     26 prompt tokens
```

What we did, in order:

1. **A text-only GGUF is refused** ("supported only as a multimodal model"): it needs `--mmproj`. llama.cpp loads
   the text file alone. Not fixed here; the projector loads beside it.
2. **The device map put layers 45-59 on the CPU**, though weights and projector (19.0 GB as the planner counted
   them, against 18.85 GB of files) fit the card. The planner sized the paged KV cache from the loader's config,
   which gave every layer 16 KV heads at the global layers' head dim (512): 3840 MB at 2048 tokens. The model
   allocates per layer (sliding: 16 x 256, global: 4 x 512), ~1.8 GB. The loader now reports the same per-layer
   config the model builds its cache with (one `gemma4_kv_model_config`). After: all 60 layers on the GPU at the
   default context (1.6 GB of KV). At `--pa-context-len 8192` 5-6 layers still spill: every layer's cache holds the
   full context (~7.4 GB); llama.cpp keeps only the 1024-token window on sliding layers. Not done here.
3. **A spilled layer failed** ("unsupported CPU RoPE dtype F32 BF16 BF16"; gpt-oss hit the same): CPU layers run
   F32 beside the model's BF16 RoPE tables. The CPU kernel now converts the rows the call's positions read. Then
   "cuda rms_norm ... got F32 and BF16": Gemma 4 moved its activations back to the GPU with `to_device`, keeping the
   CPU layers' F32; it now goes through the mapper's `cast_nm_device`. After: a spilled load decodes the same text
   (10.3 tok/s with 6 CPU layers).
4. **Raw completions had no BOS** (12 prompt tokens, output `<|channel>` repeated). The GGUF's base-model assets
   (google/gemma-4-31B-it) supply the tokenizer, and its `tokenizer.json` leaves BOS to the tokenizer class
   (transformers' `GemmaTokenizer` adds it); the GGUF says `add_bos_token`. An external tokenizer for a GGUF that
   asks for BOS now gets a `<bos> $A` post-processor when it adds none. Chat stays at 26 tokens: the template writes
   `<bos>` and the chat path leaves the tokenizer's off then.

Decoding, all layers on the GPU:

```
raw   ' Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Poland is ...' (13 tokens)
chat  43 tokens equal to llama.cpp's, |d logprob| <= 0.09; step 43 a near tie on both sides:
      llama.cpp ' contains' -1.103 / ' consists' -1.235, ours ' consists' -1.095 / ' contains' -1.197
```

The 31B decodes correctly. Our `top_logprobs` carry token ids where OpenAI's carry the token text; that is the
response type everywhere, not Gemma 4's.

## Run 2 - 2026-11-16 12:40

**Question:** does the 26B-A4B (MoE) decode as llama.cpp does, and do the two agree beyond short chats?

google/gemma-4-26B-A4B-it-qat-q4_0-gguf (13.45 GiB text, 1.1 GiB projector). Its 80-token chat matches llama.cpp
token for token (|d logprob| <= 0.09).

A text-only `gemma4` GGUF now loads: without `--mmproj` it takes the native multimodal path with no projector, its
base model's assets fetched as for a projector load, and the config's `vision_config` / `audio_config` dropped. That
also lets `scripts/gguf_perplexity_parity.sh` and the perplexity example score it:

```
gguf_perplexity_parity.sh (README.md, c=512)     llama.cpp    ours     change
gemma-4-26B_q4_0-it.gguf                          20.4464     21.5542  +0.0542
gemma-4-31B_q4_0-it.gguf                           5.2652      4.5164  -0.1422

wikitext-2 test, c=512, 20 chunks, against llama-perplexity --kl-divergence-base
        PPL ours   PPL llama.cpp   KLD mean   KLD median   KLD p99   same top-1
26B     1233.9     1018.3          0.1263     0.0354       1.437     83.47%
31B      336.8     1578.5          0.1168     0.0537       1.012     85.43%
```

Raw text puts both engines far off (wikitext PPL 300-1900 for a 31B), and there the two disagree well past kernel
rounding. In chat form they agree: a 422-token prompt's 60-token reply on the 31B is equal token for token, mean
|d logprob| 0.009, max 0.147. So the chat path (what a drafter serves) matches llama.cpp; which engine is nearer the
model on raw text needs a transformers reference (bf16), not run here.

## Run 3 - 2026-11-16 16:30

**Question:** Run 2's raw-text disagreement (KLD ~0.12 between the engines): which of ours and llama.cpp's is nearer
the model?

Reference: transformers 5.19, google/gemma-4-31B-it in bf16 on the CPU, over the first 3 chunks of the wikitext base
(c=512, first token swapped for BOS as llama-perplexity does), scored rows as llama-perplexity scores them; ours and
llama.cpp's log-probs from their `--kl-divergence-base` files of the same chunks (`g4ref/` in the session scratchpad:
a reader for that format, the reference, the comparison).

```
against transformers bf16 (3 chunks)     KLD mean   KLD median   same top-1   PPL
transformers bf16                                                              10858
llama.cpp q4_0                           1.194      0.660        60.5%          948
ours q4_0                                1.275      0.724        59.1%          174
```

On raw wikitext the instruction-tuned 31B itself is lost (PPL ~10^4 in bf16), and there both q4_0 engines are as far
from the reference as each other, KLD ~1.2: rounding swings a distribution this flat. Run 2's 0.12 between the engines
is that, not a bug on either side; raw-text perplexity does not measure this model.

In its chat form, the prompt both render (26 tokens, thinking on, `<bos><|turn>system\n<|think|>...`), transformers'
greedy reply matches ours for 34 of 40 tokens and parts at "all colors (spectrum)" against our "all colors." —
where llama.cpp's reply is ours too (Run 1: 43 tokens equal). The q4_0 weights part from bf16 there, in both engines.
Gemma 4 decodes correctly.

## Run 4 - 2026-11-19 10:00

**Question:** does the 26B-A4B run on Vulkan on a 780M (7840U, 64 GB), and how near llama.cpp's Vulkan build is it?

Same GGUF (gemma-4-26B_q4_0-it.gguf), `inference bench --vulkan --pa-context-len 8192`, performance profile; llama.cpp
Vulkan `llama-bench` on the same file.

What we did, in order:

1. **The experts failed** ("moe experts forward"): Gemma's fused `ffn_gate_up_exps` with GELU had no Vulkan path, and
   the direct loader never tried Vulkan. Vulkan experts now take a fused gate/up stack (one matvec, then the halves)
   and GELU as well as SiLU.
2. **Global layers' attention failed** (head size 512): a 512-wide paged decode kernel (8 query heads per KV head,
   two dims per thread) and a 512-wide build of the matrix-core prompt kernel (four subgroups per row tile).
3. **Sliding layers copied their cache to the CPU every token**: Vulkan decode and the three prompt kernels took no
   window. They now take one and skip tiles before it. A CUDA+Vulkan build already decoded on Vulkan there and
   ignored the window; that was wrong past 1024 tokens. Windowed prompts and first chunks use the paged prompt kernel.
4. **Host round trips**: the router with Gemma's per-expert scale, the unused per-layer-input masks and the logit
   softcap each synced; all three stay on the device now. Decode is GPU-bound (~93% busy).
5. **Long prompts drifted** (3979-token prompt, 21 tokens equal to the CPU's, mean |d logprob| 0.088): RoPE position
   tables were a rank-1 f32 product run on matrix cores, which round inputs to f16 (integers exact to 2048). f32
   products with k < 16 now run exact, and MoE router logits run exact f32 (llama.cpp keeps routers f32).

```
                      first run     now          llama.cpp Vulkan
decode (tok/s)        5.2           22.5-24.9    27.8
prefill 512 (tok/s)   343           393          527
prefill 2048 (tok/s)  166           337
3979-token prompt     21/120 tokens equal to CPU   111/120 equal to CPU, 120/120 equal to llama.cpp
```

Per decode token (44.5 ms): Q6_K head 9.7 ms (61 GB/s), MoE gate/up 7.1 ms, MoE down 4.8 ms (k = 704), ~300 RMS norms
1.2 ms, windowed attention 1.85 ms. Not done here: the default context plans a 262k-token cache (56 GB, OOM on this
machine) with full-context sliding layers; `--pa-context-len` is needed.
