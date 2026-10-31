# Qwen 1 (#562)

The original QWen series (`QWenLMHeadModel`, GGUF `qwen`), which ships remote code instead of a transformers class and
`qwen.tiktoken` instead of `tokenizer.json`, as a spec on the shared decoder (`crates/inference-models-qwen/src/qwen1.rs`).
What the decoder lacked: its modules under other names (`transformer.h.N.attn.c_attn`, `mlp.w1`/`w2`/`c_proj`,
`ln_1`/`ln_2`/`ln_f`; `DecoderSpec::module_names`). What core lacked: a `qwen.tiktoken` reader and a chat template
for a family whose checkpoints carry none (`NormalModelLoader::default_chat_template`).

## Run 1 - 2026-10-30 18:12

**Question:** do our model, tokenizer and chat prompt match Qwen's own code, from safetensors and llama.cpp's GGUF?

**Command:** `scripts/qwen1_tiny_parity.py --llama-cpp ~/Programming/llama.cpp` (transformers 5.19.0, tiktoken
0.14.0, llama.cpp 970b51ce7), on CPU in f32, on CUDA in bf16, and with `--device cuda:0 --dtype f32`. The remote code of
Qwen/Qwen-7B-Chat at 93a65d34 builds a seeded 3-layer model over the real 151,936-token vocabulary (64 wide, 4 heads of
16, MLP 96, norms and the `c_attn` bias perturbed). Scored on 20 ids (one `<|im_start|>`), a 12-token greedy run, the
README and two special-token strings through both tokenizers, and a system + one-turn chat against `make_context`.

**Finding:**
- The remote code needs two patches under transformers 5: it imports `transformers_stream_generator` for
  `chat_stream` (an old package, left uninstalled), and calls `PreTrainedModel.get_head_mask`, which is gone. Its
  `generate` no longer runs with the new cache API, so the reference decodes by full forwards.
- The first run matched the model (mean KL 2.8e-7, top-1 1.000, the same greedy tokens) but 2 of 3 tokenizations:
  the README came to 5,648 tokens against 4,829 (`<a` as `<`, `a`; `\tand` as `\t`, `and`). Our tiktoken readers ran
  the rank file's pattern and then `ByteLevel` with `use_regex`, which splits again with GPT-2's regex; transformers'
  tiktoken converter turns it off. The same line served Mistral's `tekken.json`: against tiktoken over Mistral-Nemo's
  tekken.json, the README tokenized to 5,827 tokens against 4,939 and this repository's `tokenizer.rs` to 3,944
  against 3,267. With `use_regex` off, all three texts match tiktoken for both files.
- `tokenize_chat` then failed: "The selected model does not define a chat template". The new loader hook was not
  forwarded by `AutoNormalLoader`, which a `Plain` load without an architecture goes through.
- After both fixes, CPU f32: safetensors and GGUF at mean KL 2.8e-7, top-1 1.000, max logit diff 0.0025, the same 12
  greedy tokens, 0 of 3 tokenizations differing, and the chat's 46 tokens as `make_context` builds them.
- CUDA bf16: KL 6.8e-5, top-1 0.900 (2 of 20 positions); CUDA f32: KL 1.8e-8, top-1 1.000, max logit diff 0.0000, so
  bf16's two are near ties.
- `qwen1_prefill` fails with the gate and up projections swapped; the tokenizer test
  (`a_qwen_tiktoken_file_merges_by_rank_and_ids_its_specials_after_the_ranks`) fails with `use_regex` back on
  (`[258, 60, 97, 32, 98]` against `[258, 256, 32, 98]`).

**Next:** a published quantized GGUF against llama.cpp's perplexity.

## Run 2 - 2026-10-30 18:12

**Question:** does a published quantized Qwen 1 GGUF load and score as llama.cpp scores it?

**Command:** `scripts/gguf_perplexity_parity.sh ~/Programming/llama.cpp/build_cuda/bin/llama-perplexity <dir>`
(README.md, ctx 512, one chunk) on a community Q8_0 of Qwen-1.8B-Chat (1.8 GiB); then one sampled chat through the C
ABI.

**Finding:**
- The load first failed: `Standalone qwen config requires GGUF metadata qwen.attention.head_count_kv`. The file
  predates llama.cpp writing it for Qwen 1, which has no grouped KV; llama.cpp reads a missing head_count_kv as the
  head count for every architecture, and our standard GGUF fields now do too.

| file | llama.cpp | ours | drift |
|---|---|---|---|
| Qwen-1.8B-Chat Q8_0 | 3.1236 | 3.1288 | 0.0017 |

- "Name three primary colors, comma separated." (temperature 0.7): "Red, Blue, Green.", finish `stop` after 7
  tokens; the 27-token prompt carries the default system turn.

**Next:** none. `use_dynamic_ntk` and `use_logn_attn` only act past `seq_length`, so a safetensors load caps the
context there; a GGUF carries llama.cpp's `context_length` (`max_position_embeddings`) and runs past it as llama.cpp does,
without either.
