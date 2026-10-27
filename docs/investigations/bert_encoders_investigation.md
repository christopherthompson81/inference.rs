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
