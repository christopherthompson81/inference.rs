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
