# BERT-family encoders (#586)

Classic BERT / RoBERTa / XLM-R / CamemBERT encoders as embedding models: learned absolute positions, post-LayerNorm,
GELU FFN, token type embeddings, then the repo's sentence-transformers Pooling / Normalize modules.

Checkpoints used (public sentence-transformers repos, safetensors only):

| Checkpoint | Class | Tokenizer | Pooling |
|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | `BertModel` | WordPiece | mean + normalize |
| BAAI/bge-small-en-v1.5 | `BertModel` | WordPiece | CLS + normalize |
| intfloat/multilingual-e5-small | `BertModel` | SentencePiece Unigram (XLM-R vocab) | mean + normalize |
| sentence-transformers/all-distilroberta-v1 | `RobertaForMaskedLM` (weights unprefixed) | byte-level BPE | mean + normalize |

## Run 1 - 2026-10-27 08:54

Question: does the new `bert` embedding loader reproduce transformers' `AutoModel` last hidden state, pooled the way
each repo's `modules.json` says?

Command: `target/debug/inference serve --cpu --dtype f32 --no-ui -p 8095 -m <checkpoint>` per checkpoint, then
`ref.py <checkpoint> 8095` (scratchpad): four short English / French prompts through `/v1/embeddings`, against
transformers 5.19 `AutoModel` in F32 with the repo's pooling (CLS or mean over all tokens) and L2 normalize.

```
all-MiniLM-L6-v2      cos 0.33-0.46
bge-small-en-v1.5     cos 1.000000, max_abs 1.4e-7
multilingual-e5-small cos 1.000000, max_abs 1.2e-7
all-distilroberta-v1  cos 0.35-0.62
```

Finding: the encoder itself is right (bge's CLS pooling and e5's mean pooling match to F32 rounding), and the two
failing repos are the two whose `tokenizer.json` carries `"padding": {"strategy": {"Fixed": 128}}` and a 128-token
truncation. `get_tokenizer` kept both, so every prompt was padded to 128 tokens and the mean ran over the pad rows.
transformers ignores a saved padding / truncation unless the call asks for it.

Implication: drop the saved padding and truncation in `get_tokenizer` (the engine fits prompts to the context itself),
and take sentence-transformers' `max_seq_length` from `sentence_bert_config.json` as the embedder's prompt limit
(256 for MiniLM, below its 512 positions).

Also found on the way: `Pooling` summed for `pooling_mode_mean_tokens` (invisible behind Normalize, wrong without it)
and left CLS / max / mean-sqrt-len `unimplemented!()`; bge's pooling config also omits the newer
`pooling_mode_lasttoken` / `include_prompt` fields, which were required.

## Run 2 - 2026-10-27 08:56

Question: with the tokenizer padding dropped, do all four match?

Command: as Run 1, after rebuilding.

```
all-MiniLM-L6-v2      worst cos 1.000000, max_abs <= 2.1e-7
bge-small-en-v1.5     worst cos 1.000000, max_abs <= 1.4e-7
multilingual-e5-small worst cos 1.000000, max_abs <= 1.2e-7
all-distilroberta-v1  worst cos 1.000000, max_abs <= 2.7e-7
```

Finding: all four match transformers to F32 rounding, including the RoBERTa position offset (`pad_token_id + 1`)
and the unprefixed weights of a checkpoint that declares `RobertaForMaskedLM`.

Implication: the CPU path is right; next, the CUDA path in BF16 and the tiny-checkpoint tests.

## Run 3 - 2026-10-27 08:57

Question: does the CUDA path (flash attention, non-causal) hold in BF16 and F32, and does in-situ quantization work?

Command: as Run 1 without `--cpu`, with `--dtype bf16` / `--dtype f32`, then `--isq q8_0` / `--isq q4k` (BF16).

```
                      bf16 worst cos  f32 worst cos  q8_0 worst cos  q4k worst cos
all-MiniLM-L6-v2      0.99993         1.000000       0.99957         0.967
bge-small-en-v1.5     0.99992         1.000000
multilingual-e5-small 0.99997         1.000000       0.99976         0.988
all-distilroberta-v1  0.99994         1.000000
```

Finding: the first `--isq q8_0` run failed every request with `dtype mismatch in add, lhs: BF16, rhs: F32` in
`GgufMatMul::add_bias`: the GGUF ISQ branch stores the bias in F32, and the CUDA quantized matmul returns the BF16
activation dtype. It is not BERT-specific: any biased projection outside the fused-QKV path would hit it on a BF16
model. `add_bias` now casts the bias to the output dtype (a no-op clone when they match). Q4K's loss on 384-wide
models is ordinary quantization error.

Implication: Q8_0 is a usable ISQ for these encoders; Q4K is not worth recommending at this width.

## Run 4 - 2026-10-27 09:15

Question: BAAI/bge-m3 (`XLMRobertaModel`, 8194 positions, CLS pooling) ships only a single `pytorch_model.bin`. Does it
load and match?

Command: as Run 3 on `/mnt/data/models/bge-m3`, `--dtype f32` then `--dtype bf16`.

Finding: two load failures before any embedding. `PICKLE_MATCH` only matched sharded
`pytorch_model-0000N-of-0000M.bin`, so a single `pytorch_model.bin` was never picked. With that widened, the load
failed with `header too large`: `load_safetensors` maps every file as safetensors on its default mmap path, and only
`INFERENCE_RS_NO_MMAP=1` reached the pickle backend, so the existing pickle fallback never worked by default. A
checkpoint with a `.pth` / `.pt` / `.bin` file now takes the per-file loader (a first cut that switched on any
non-safetensors file broke the LoRA adapter tests, whose staged files have no extension). After both:

```
bge-m3 f32   worst cos 1.000000, max_abs <= 4.0e-5
bge-m3 bf16  worst cos 0.999906
```

Implication: bge-m3's dense embedding is served; its sparse (`sparse_linear.pt`) and ColBERT (`colbert_linear.pt`)
heads are outside sentence-transformers' modules and not loaded.

## Run 5 - 2026-10-27 10:40

Question: do the `*ForSequenceClassification` heads and pair token types score (query, document) pairs as
transformers does?

Command: `inference serve --dtype {f32,bf16} -m <reranker>` on GPU, then `rerank_ref.py <reranker> 8095`
(scratchpad): one query and four documents through `/v1/rerank`, against transformers
`AutoModelForSequenceClassification` on the tokenizer's pair encoding, sigmoid of the single logit.
Checkpoints: cross-encoder/ms-marco-MiniLM-L6-v2 (`BertForSequenceClassification`, two token types, `bert.` prefix,
pooler + classifier) and BAAI/bge-reranker-base (`XLMRobertaForSequenceClassification`, one token type, `roberta.`
prefix, dense + out_proj).

```
                       f32 worst |diff|  bf16 worst |diff|  order
ms-marco-MiniLM-L6-v2  1.6e-6            2.3e-4             same
bge-reranker-base      5.7e-5            1.1e-3             same in f32; bf16 swaps two docs tied at 0.000037
```

Finding: MiniLM's F32 match means the segment ids reach the model (its type 1 row differs from type 0, so dropping
them would move every score). bge-reranker-base's F32 gap was larger than rounding, so Run 6 checks it on CPU.

## Run 6 - 2026-10-27 10:43

Question: is bge-reranker-base's F32 gap a model error or GPU F32 arithmetic?

Command: as Run 5 with `--cpu --dtype f32`.

```
bge-reranker-base      worst |diff| 5.7e-8
ms-marco-MiniLM-L6-v2  worst |diff| 8.3e-9
```

Finding: exact on CPU, so the GPU F32 gap (about 1.3e-3 in the logit over 12 layers of 768-wide matmuls) is the
GPU path's arithmetic, not the head.

## Run 7 - 2026-10-27 10:55

Question: do the tiny-checkpoint rerank tests hold (pair segment ids reach the model; a `bert.`-prefixed cross-encoder
reloads from its UQFF with the same scores)?

Command: `cargo nextest run -p inference -E 'test(/^bert_embedding_tiny::/)'`.

Finding: two failures, both real.
- The segment-id test saw all-zero type ids. tokenizers 0.23.2's `encode_fast` builds tokens on its offset-free path
  with a hardcoded type id 0 (`tokenizer/pre_tokenizer.rs`, `into_encoding`, `OffsetType::None`), so a pair's second
  segment loses its ids unless a post-processor (BERT's `TemplateProcessing`) assigns them again. That is why
  ms-marco-MiniLM still matched in Run 5; the tiny tokenizer has no post-processor. The pair now goes through
  `encode`.
- The Q8_0 cross-encoder failed with `input tensor is not contiguous` (shape [2, 32], stride [576, 1]): the head's
  first-token slice is a strided view, which the quantized matmul rejects. The slice is now made contiguous.

After both: 11 of 11 pass.

## Run 8 - 2026-10-27 12:03

Question: do the RoPE encoders (Nomic BERT, ModernBERT) match transformers, including ModernBERT's sliding-window
layers?

Command: as Run 1 (`--cpu --dtype f32`) and Run 3 (`--dtype bf16`, GPU), with a fifth, 402-token prompt so ModernBERT's
sliding layers (128-token window, |i - j| <= 64) mask. Checkpoints: nomic-ai/nomic-embed-text-v1.5 (`NomicBertModel`,
post-norm, fused unbiased QKV, RoPE base 1000, SwiGLU with `fc11` up and `fc12` gate, mean pooling) and
ibm-granite/granite-embedding-english-r2 (`ModernBertModel`, pre-norm, bias-free LayerNorms, GeGLU, global layer
every 3rd with RoPE theta 80000, the rest sliding with theta 10000, CLS pooling). Reference: transformers 5.19's
native `nomic_bert` and `modernbert` modules.

```
                               cpu f32 worst cos  max_abs   gpu bf16 worst cos
nomic-embed-text-v1.5          1.000000           <= 7.2e-6   0.99992
granite-embedding-english-r2   1.000000           <= 7.6e-6   0.99994
```

Finding: both match at every length, 402 tokens included. Nomic's v1.5 config carries both its GPT-2 names (`n_embd`,
`n_layer`, `n_head`, `n_inner`) and the BERT names, so serde aliases collided (`duplicate field hidden_size`); the
config now reads either. ModernBERT's max_abs is larger than BERT's (7.6e-6 against 2e-7) over 22 layers of a
checkpoint stored in BF16, with the cosine at 1.0.

Implication: both load. The tiny ModernBERT test fails when the sliding mask is dropped (checked by mutation).
Left for later in #586: nomic-bert-moe, Jina v2 (ALiBi), Jina v3 (XLM-R with RoPE and task LoRA), Jina v5 / EuroBERT,
NeoBERT, the Llama bidirectional embedder.

## Run 9 - 2026-10-27 12:48

Question: do the Llama-style encoders (EuroBERT on the shared decoder stack with a bidirectional mask, NeoBERT on the
RoPE encoder block) match their references, with no `modules.json` (mean pooling, as sentence-transformers falls
back to)?

Command: as Run 8 on EuroBERT/EuroBERT-210m (transformers' native `eurobert`) and chandar-lab/NeoBERT, whose only
reference is its own remote `model.py`. That file was read first (it defines the model and touches no files or
network) and run with a stand-in for `xformers.ops.SwiGLU` (`w12` split in halves, `silu(first) * second` into `w3`),
which is not installed here (`neobert_ref.py`, scratchpad). NeoBERT differs from the Nomic / ModernBERT block in two
ways the block now takes as options: its fused QKV is laid out per head (`view(b, s, heads, 3 * d).chunk(3)`), and
its rotary is Llama 1's complex-pair (interleaved) form. Its SwiGLU is 2/3 of `intermediate_size`, rounded up to a
multiple of 8.

```
                 cpu f32 worst cos   gpu bf16 worst cos
EuroBERT-210m    1.000000            first run: HTTP 500 at 381 tokens
NeoBERT          1.000000            0.99923
```

Finding: both match in F32. The BF16 failure was `CUDA_ERROR_MISALIGNED_ADDRESS` in inference-fattn, and it is not
EuroBERT's: ModernBERT on the GPU embeds 200 tokens and fails at 201 (and every request after, the context being
poisoned). fattn's mma kernel copies mask rows with 16-byte `cp.async`, so every mask row has to start on a 16-byte
boundary; llama.cpp only ever hands it masks padded to its KV stride, and ours had the key length as its row stride
(201 or 381 f16 = 402 or 762 bytes). `fattn_mask` now gives the mask a row stride padded to 8 f16, keeping the
logical key length. A GPU test runs fattn at key lengths 201 and 381 (full and windowed masks, head dim 64) against
the CPU, and fails with the padding removed.

After the fix, EuroBERT BF16 worst cosine is 0.99951 over the same prompts, 381 tokens included. That is BF16
arithmetic on a base MLM's raw mean-pooled states, not the port: transformers' own BF16 against its F32 gives
0.99987 and 0.99959 on two of the prompts.

Implication: EuroBERT, NeoBERT and Llama bidirectional embedders load. The 8B llama-embed-nemotron is not parity
checked here (a 16 GB BF16 download); it runs the same bidirectional decoder stack as EuroBERT, with GQA and Llama 3
rope scaling, which the tiny checkpoint covers. Left in #586: nomic-bert-moe, Jina v2 (ALiBi), Jina v3 and v5 (task
LoRA adapters), and GGUF loading for embedders.

## Run 10 - 2026-10-27 13:23

Question: does nomic-embed-text-v2-moe (Nomic BERT with GELU FFNs and an 8-expert top-2 MoE every other layer) match
Nomic's own code?

Command: as Run 8, first with `ref.py` (transformers' native `nomic_bert`), then with `nomic_remote_ref.py`: Nomic's
remote `modeling_hf_nomic_bert.py` (read first; its `os` use is only local weight paths), with transformers 4's
`get_extended_attention_mask` supplied since transformers 5 dropped it.

```
ref.py (native nomic_bert)       cpu f32 worst cos 0.007
nomic_remote_ref.py, v2-moe      cpu f32 1.000000 (max_abs <= 1.9e-7)   gpu bf16 0.99983
nomic_remote_ref.py, v1.5        cpu f32 1.000000                       gpu bf16 0.99993
```

Finding: the first comparison was against the wrong reference. transformers' native `nomic_bert` has no MoE, so
AutoModel built the expert layers from random init; the remote code is the real one. Semantics from it: the router
softmaxes in F32 over the experts and keeps the top 2 without renormalizing; expert e is
`gelu(x @ w1[e]^T) @ w2[e]` with both stacks `[experts * 3072, 768]`; one bias is added after the weighted sum; layer
i is MoE when `i % moe_every_n_layers == 1`. Ours runs the experts densely: `gelu(x @ w1^T)` over every expert,
scaled per expert by the router weight (zero outside the top k), then one matmul with `w2`. That is exact, with no
gather or host sync, at 4x the expert FLOPs of a top-2 gather. (The unquantized `gather_forward` copies each
selected expert's matrices per token, about 1.9e9 floats for a 400-token prompt, which rules it out for an encoder.)
The first GPU run failed with `no cuda implementation for argsort`; the k-th largest router weight now comes from k
masked maxima. v1.5 agrees with the remote reference too (its max_abs differs only because the script normalizes
and v1.5 ships no Normalize module).

## Run 11 - 2026-10-27 13:30

Question: does Jina BERT v2 (ALiBi in place of positions, GeGLU FFN with an inner LayerNorm) match Jina's own code?

Command: as Run 8 on jinaai/jina-embeddings-v2-base-en with `jina2_ref.py`: Jina's remote `modeling_bert.py`
(jina-bert-implementation), run with stand-ins for what transformers 5 removed (`find_pruneable_heads_and_indices`,
`prune_linear_layer`, `transformers.onnx.OnnxConfig`, `get_extended_attention_mask`, `get_head_mask`, and
PretrainedConfig defaults such as `is_decoder`).

```
cpu f32   worst cos 1.000000 (max_abs <= 5.6e-8)
gpu bf16  worst cos 0.99997
```

Finding: matches, a 402-token prompt included. Each head adds `-slope_h * |i - j|`; the slopes are the
power-of-two series with the next power's odd terms filling a non-power head count (12 heads: 2^-1 .. 2^-8, then
2^-0.5, 2^-1.5, 2^-2.5, 2^-3.5), the same as ggml's `max_bias = 8`. The bias is a per-head additive mask, which
fattn declines (it takes one mask for every head), so ALiBi layers run eager attention; fattn's `max_bias` with one
shared `-|i - j|` mask (how llama.cpp runs jina-bert-v2) would avoid the per-head tensor at long inputs. A tiny test
first compared a sequence with its reversal, which symmetric ALiBi cannot tell apart (every |i - j| is kept); it now
swaps two tokens, and fails with the bias dropped. jinaai/jina-reranker-v1-tiny-en is a classifier checkpoint whose
config names the base `JinaBertModel`, so it loads as an embedder, not a reranker.

## Run 12 - 2026-10-27 14:05

Question: with a per-request `task` (the user's choice over a load-time task), does Jina v3 match Jina's own code
for every task, and with no task?

Command: `inference serve --cpu --dtype f32` (then `--dtype bf16` on GPU) on jinaai/jina-embeddings-v3, then
`jina3_ref.py`: Jina's remote `XLMRobertaLoRA` (xlm-roberta-flash-implementation, read first) on the checkpoint,
called with `adapter_mask` set to the task and the task's instruction prepended (as its `encode` does), mean pooled
and normalized, against `/v1/embeddings` with the same `task`.

Semantics, from `modeling_lora.py`: a linear layer's task weight is `W + (alpha / rank) * B_t A_t`
(`lora_A` `[tasks, rank, in]`, `lora_B` `[tasks, out, rank]`); an embedding table's is `E + (alpha / rank) * A_t B_t`
(`lora_A` `[tasks, vocab, rank]`, `lora_B` `[tasks, rank, hidden]`); no task leaves the base weights. The encoder
is Nomic BERT's layout (fused `mixer.Wqkv`, GELU `mlp.fc1` / `fc2`, post-norm), rotary base 20000 from position 0.

```
                     cpu f32 worst cos (max_abs)       gpu bf16 worst cos
no task              1.000000 (<= 3.9e-7)              0.99974
retrieval.query      1.000000 (<= 5.1e-7)              0.99987
retrieval.passage    1.000000 (<= 8.0e-7)              0.99984
text-matching        1.000000 (<= 3.9e-7)              0.99991
```

Finding: every task matches, at 9 to 434 tokens. Ours adds each row's adapter as a low-rank delta
(`x A_t^T B_t^T`, and `A_t[ids] B_t` for the tables) rather than merging weights, so one batch can hold rows of
different tasks; a row with no task scales its delta by 0. The first load failed with `missing field model_type`:
Jina v3's config names none, and the BERT config now defaults it. Nothing in Jina v3 is quantized in place (its
base weights load from `parametrizations.weight.original`), so ISQ leaves it as it is.

Implication: Jina v3 is served. Jina v5 puts its task adapters (separate PEFT LoRAs) inside Llama-style layers on
the shared decoder stack, which would need the same per-row delta there; left for later.
## Run 13 - 2026-10-27 14:35

Question: do encoder GGUFs, as llama.cpp's converter writes them (`bert`, `nomic-bert`), load and embed as the
safetensors checkpoints do?

Command: `convert_hf_to_gguf.py --outtype f32` (and `q8_0`) at llama.cpp 970b51ce7 on all-MiniLM-L6-v2,
bge-small-en-v1.5 and nomic-embed-text-v1.5 (multilingual-e5-small does not convert: `BertModel` there carries
XLM-R's SentencePiece vocabulary, which the converter's BERT path rejects), then
`inference serve --cpu --dtype f32 --format gguf -m <dir> -f <file>` against `ref.py` (transformers on the original
checkpoint) and against `llama-embedding` (rebuilt at 970b51ce7; the build_cuda binary was from August and missing its
library) on the same GGUF.

```
                          vs transformers (f32 GGUF)   vs llama-embedding (f32)   vs llama-embedding (q8_0)
all-MiniLM-L6-v2          1.000000 (max_abs 1.6e-7)    0.999998                    0.99975
bge-small-en-v1.5         1.000000 (max_abs 1.4e-7)
nomic-embed-text-v1.5     1.000000                     1.000000                    0.99216
```

Finding: the F32 GGUFs match transformers to F32 rounding, so the WordPiece tokenizer rebuilt from the GGUF
(llama.cpp stores word-initial pieces with a U+2581 prefix and continuations bare; control tokens as they are) and
the tensor bindings are right. llama.cpp's own F32 MiniLM differs from transformers by about 2.6e-4. The Q8_0 gap
with llama.cpp on Nomic is ours: against the transformers F32 reference, our CPU Q8_0 GGUF gives 0.9911 to 0.9967
where llama.cpp's gives 0.9981 to 0.9986, while our CUDA Q8_0 gives 0.9981 to 0.9985 (llama.cpp's level). ISQ Q8_0
on the safetensors checkpoint shows the same split (CPU 0.9904, GPU 0.9982), so it is the CPU quantized matmul, not
the GGUF path.

The cause: the x86 repacked matmul (`repack_x86::select`, Q8_0 added in #523) quantizes the activations of every
repacked type to Q8_K, one scale per 256 values. ggml pairs Q8_0 weights with Q8_0 activations (one scale per 32);
Q8_K is the right pairing only for the K-quants. Nomic's FFN inputs carry outliers, so one scale over 256 values
costs it about 0.008 in cosine; MiniLM barely moves (0.99954 against llama.cpp's 0.99962).

Implication: a CPU accuracy issue for every Q8_0 model on x86 (GGUF or ISQ), outside #586. Either Q8_0 leaves the
repacked path (ggml-exact, slower prefill), or its tiles take per-32 activation scales.

## Run 14 - 2026-10-27 15:05

Question: does an XLM-R GGUF (llama.cpp's `bert` architecture with a `t5` SentencePiece Unigram vocabulary) load
and embed as the original does?

Command: `convert_hf_to_gguf.py --outtype f32` on BAAI/bge-m3 (the converter needed the `sentencepiece` package,
installed to the user site), then `inference serve --cpu --dtype f32 --format gguf` against `ref.py` on the
original checkpoint (CLS pooling, normalized), a French prompt and a 422-token one included.

```
bge-m3 f32 GGUF vs transformers: worst cos 1.000000, max_abs <= 4.2e-7
```

Finding: matches. The vocabulary converts to a Unigram model over the GGUF's scores, normalized by its
`precompiled_charsmap` (and `remove_extra_whitespaces` collapsing runs of spaces), split on Metaspace with the space
prefix `add_space_prefix` asks for, and wrapped as `add_bos` / `add_eos` / `add_sep` say: `<s> $A </s>`, with the
XLM-R pair `<s> $A </s> </s> $B </s>`. T5's own GGUFs carry the same `t5` model without `add_bos`, which the flags
keep right. The converter trims XLM-R's position table to start at 0 (`position_embd` is 8192 rows here), so the
GGUF runs as BERT with no position offset. The synthesized `vocab_size` comes from `token_embd.weight`'s rows, not
the token count, since a converter may pad the table.

Implication: GGUF embedders load for `bert` and `nomic-bert`. Not yet: GGUF rerankers (`pooling_type` 4, the `cls`
head), and the other encoder architectures llama.cpp writes. The CPU Q8_0 accuracy gap of Run 13 is #655.

## Run 15 - 2026-10-27 15:33

Question: do the other encoder GGUFs llama.cpp's converter writes (`modern-bert`, `jina-bert-v2`, `neo-bert`,
`nomic-bert-moe`) and its rerankers (`pooling_type` 4) load and score as the originals?

Command: `convert_hf_to_gguf.py --outtype f32` (llama.cpp 970b51ce7) on a ModernBERT embedder (granite r2), Jina
Embeddings v2, NeoBERT, nomic-embed-text v2 MoE, bge-reranker-base and an ms-marco MiniLM cross-encoder, then
`inference serve --cpu --dtype f32 --format gguf` against the reference scripts on the originals (their own remote
code where the model has it) and against llama.cpp's `llama-embedding` on the same GGUF.

```
modern-bert      vs transformers         cos 1.000000
jina-bert-v2     vs Jina's code          cos 1.000000
neo-bert         vs NeoBERT's code (CLS) cos 1.000000   (llama.cpp on the same GGUF: 0.02)
nomic-bert-moe   vs llama.cpp            cos 0.99999
nomic-bert-moe   vs Nomic's code         cos 0.887
bge-reranker-base GGUF logit 4.71547     transformers 4.71546
ms-marco MiniLM GGUF                      no `cls.weight` (pooler dropped)
```

Finding: the four architectures bind under the converter's names, with configs from its metadata: ModernBERT's
alternating window from `attention.sliding_window`, `sliding_window_pattern` and `rope.freq_base_swa`; NeoBERT's
`feed_forward_length` is the SwiGLU width, 2/3 of `intermediate_size`; Nomic MoE's `ffn_up_exps` / `ffn_down_exps`
stacks reshape back to megablocks' `w1` / `w2`. Encoder BPE vocabularies (`gpt2` model, ModernBERT and Jina v2) need
CLS/SEP wrapping from `add_bos` / `add_eos`, and `modern-bert` joins the gpt-2 pre-tokenizer group. The token-type
count comes from `token_types.weight`'s rows: the metadata default of 2 is wrong for Nomic MoE's single row.

Three conversion losses upstream, for reporting there:
- the converter drops `mlp.experts.bias` for `nomic-bert-moe`, the whole 0.887 gap (zeroed in the source, ours
  matches the GGUF to F32 rounding);
- the converter drops BERT's `pooler.dense`, which `BertForSequenceClassification` scores through, so BERT
  rerankers score wrong (RoBERTa's `classifier.dense` survives as `cls`); we refuse such a GGUF with a pointer to
  the safetensors checkpoint;
- llama.cpp's `build_qkv` reads NeoBERT's `qkv` as a stacked Q/K/V block while the converter keeps NeoBERT's
  per-head interleave, hence its 0.02; our encoder reads the per-head layout as NeoBERT's own weights have it. EuroBERT does not convert
  (`EuroBertForMaskedLM` unsupported).

Tests: `gguf_embedding_tiny.rs` writes tiny GGUFs under llama.cpp's names for all six architectures (F32 and Q8_0
against the safetensors source by cosine) and checks the pooler-less reranker refusal.

Implication: GGUF loading covers every encoder architecture llama.cpp writes. Left in #586: Jina v5.

## Run 16 - 2026-10-27 15:59

Question: how is Jina v5 built, and do its merged per-task repos already load?

Command: read the remote code and configs of `jina-embeddings-v5-text-small` (Qwen3-0.6B base) and `-text-nano`
(EuroBERT-210m base); `jina5_merged.py` compares each `-retrieval` repo with its base plus the retrieval adapter
(weights, and the last-token embedding against Jina's `encode`); `jina5_ref.py` saves Jina's own `encode` (PEFT,
F32) per task and prompt on a short and a 1.7K-character text; then `inference serve --cpu --dtype f32` on each
`-retrieval` repo.

```
small: merged weights vs base + B A: max abs 3.755e-03   embed cos 0.999997
nano:  merged weights vs base + B A: max abs 1.927e-03   embed cos 0.999996
serve small-retrieval: Unsupported Hugging Face Transformers model class `Qwen3Model`
serve nano-retrieval:  missing field `rope_theta`
```

Finding: the base repos are `JinaEmbeddingsV5Model`, a `PeftMixedModel` over a plain Qwen3Model / EuroBertModel
(unprefixed `layers.N` keys) with one standard PEFT LoRA per task under `adapters/<task>/` (r 32, alpha 32, on
q/k/v/o and gate/up/down; adapter keys `base_model.model.layers.N...`). Pooling is the last unpadded token,
L2-normalized; texts get a `Query: ` or `Document: ` prefix (`document` by default), for every task. The Qwen3
tokenizer adds no special tokens, so the last token is the text's own; the EuroBERT one appends `<|end_of_text|>`,
and EuroBERT attends bidirectionally. Each merged repo is the base plus its adapter merged (the gap is bf16
rounding), with `modules.json` last-token pooling and the prefixes left to the caller. Two load gaps: `Qwen3Model`
has no embedding mapping, and the nano repos' transformers 5 config keeps `rope_theta` under `rope_parameters`.

Implication: fix the two config gaps, which loads the merged repos; for the base repos, a `JinaEmbeddingsV5Model`
loader that registers the decoder stack's linears for dynamic LoRA, preloads the four adapters, and scopes each
embedding batch to its rows' tasks (`retrieval.query` / `retrieval.passage` share the retrieval adapter).

## Run 17 - 2026-10-27 16:12

Question: with the two config gaps fixed and a `JinaEmbeddingsV5Model` path, do the merged and base v5 repos embed
as Jina's code does, per task?

Command: `inference serve --cpu --dtype f32` on each `-retrieval` repo (`jina5_cmp.py --merged`: `Query: ` /
`Document: ` prefixed by hand, no task) and on each base repo (`jina5_cmp.py`: every task by `task`), against Run 16's
saved `encode` outputs; then the base repos on CUDA at the default BF16, the tasks of each text sent at once.

```
small-retrieval (merged) f32 CPU   worst cos 0.999995    nano-retrieval (merged) f32 CPU   worst cos 0.999993
small base, 5 tasks      f32 CPU   worst cos 1.000000    nano base, 5 tasks      f32 CPU   worst cos 1.000000
small base, 5 tasks     bf16 CUDA  worst cos 0.999946    nano base, 5 tasks     bf16 CUDA  worst cos 0.999944
unknown task: "unknown task `nope`; this model's tasks are: retrieval.query, retrieval.passage, text-matching,
clustering, classification"
```

Finding: exact on the base repos. Config preparation (after any `--hf-overrides`) lifts transformers 5's
`rope_parameters` into `rope_theta` / `rope_scaling` when those are absent (the root and `text_config`; per-layer-type parameters are left alone), and
`Qwen3Model` maps to the Qwen3 embedder. The base repos route by `is_decoder` (false: the bidirectional Llama
embedder, else Qwen3); `adapters/<name>/` folders become the embedding paths' LoRA adapters, which the dynamic LoRA
runtime preloads (4 x rank 32); each forward is scoped to its rows' adapters, so mixed tasks batch. With no Pooling
module in `modules.json`, the last token is pooled and normalized. The merged repos' residual (5e-6) is their bf16
merge, not ours.

Implication: commit with a tiny-checkpoint test (adapters checked against the merged weights, mixed tasks in one
batch) and docs.

## Run 18 - 2026-10-27 16:47

Question: after reopening #586 for the scope it lists that was still open, does DistilBERT (embedder and
`DistilBertForSequenceClassification` cross-encoder) run on the BERT encoder under its own names?

Command: `inference serve --cpu --dtype f32` on `sentence-transformers/multi-qa-distilbert-cos-v1` against `ref.py`
(five texts, one of 402 tokens) and on `ml6team/cross-encoder-mmarco-german-distilbert-base` (one label) against
`rerank_ref.py`.

```
multi-qa-distilbert-cos-v1: worst cos 1.000000, max_abs <= 2.5e-7
mmarco german distilbert reranker: worst score diff 9.7e-8, same order
```

Finding: DistilBERT is BERT's post-LayerNorm block with no segment table, under other names (`transformer.layer.N`,
`attention.{q,k,v,out}_lin`, `sa_layer_norm`, `ffn.lin1/lin2`, `output_layer_norm`; `distilbert.` under the task
heads) and config keys (`dim`, `n_layers`, `n_heads`, `hidden_dim`, `activation`). The BERT encoder reads them
through a per-architecture name table and serde aliases. Its classifier is `pre_classifier`, ReLU, `classifier` on
the first token, with no pooler. Tests: a tiny BERT renamed by hand to DistilBERT embeds as the BERT (segment table
zeroed), and the classifier's logit equals that head applied by hand to the first-token state.

Implication: DistilBERT done. Next on #586: the ModernBERT and NeoBERT classifier classes.

## Run 19 - 2026-10-27 17:04

Question: do the ModernBERT and NeoBERT classifier classes #586 lists score as their references, and what is
`ModernBertDecisionModel`?

Command: `inference serve --cpu --dtype f32` on `Alibaba-NLP/gte-reranker-modernbert-base`
(`ModernBertForSequenceClassification`, mean pooled) against `rerank_ref.py`, and on
`tomaarsen/reranker-NeoBERT-gooaq-bce` against `neobert_rerank_ref.py` (NeoBERT's own
`NeoBERTForSequenceClassification`, each pair alone); read llama.cpp 970b51ce7 `conversion/bert.py` for the decision
model.

```
gte-reranker-modernbert-base: worst score diff 2.95e-6, same order
reranker-NeoBERT-gooaq-bce:   worst score diff 2.05e-7, same order
```

Finding: both match. The five heads (BERT's pooler, RoBERTa's `classifier.dense`, DistilBERT's `pre_classifier`,
ModernBERT's `head`, NeoBERT's `dense`) are one shape: pool (first token, or mean for ModernBERT's
`classifier_pooling`), a dense layer, tanh / ReLU / the classifier activation, an optional LayerNorm (ModernBERT's
`head.norm`), the label projection. They now share one `ClassifierHead`, each encoder naming its weights.

`ModernBertDecisionModel` is not a transformers class: llama.cpp's converter names it for two RL decision checkpoints
(`convaiinnovations/laya`, `SupersonicLabs/Julia-1`), a ModernBERT encoder with extra head blocks over `[MASK]`
option slots, a decision prompt template and per-option-type temperatures. It is a decision-serving feature, not an
embedder or reranker.

Implication: classifiers done. Next on #586: the `jina-bert-v3`, `eurobert` and `llama-embed` GGUF architectures;
the decision model needs its own scope.
