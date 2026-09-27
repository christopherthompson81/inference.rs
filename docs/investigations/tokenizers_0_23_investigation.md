# Owning the llguidance byte tokenizer to move to tokenizers 0.23

`tokenizers` built twice: we used 0.21.4 (shared with `toktrie_hf_tokenizers`, whose latest release still pins 0.21)
and the pinned candle rev needs 0.23.2. `toktrie_hf_tokenizers` is a ~330-line adapter that turns a Hugging Face
tokenizer into the per-token byte strings llguidance builds its grammar trie from, so it moved in-tree
(`inference-core/src/pipeline/llg/byte_tokenizer.rs`) and the workspace moved to 0.23.

## Run 1 - 2026-09-27 00:30

- Question: does the in-tree port on tokenizers 0.23 produce exactly what the crate produced on 0.21? A difference
  would show up as wrong grammar masks, not as a failure.
- Command: a throwaway binary linking both `tokenizers` 0.21.4 + `toktrie_hf_tokenizers` 1.4.0 and `tokenizers`
  0.23.2 + the port, run over seven local `tokenizer.json` files. For each it compares `TokRxInfo` (EOS, end of turn,
  unk, pad, vocab size), every token's byte string, and the added-token table `(id, content, special)`.
- Inputs: byte-level BPE (Llama 3.2, Qwen3 and a Qwen-derived TTS model, Granite-Docling, Whisper, with 26 to 1609
  added tokens) and SentencePiece-style byte-fallback BPE with a `Replace` decoder (PaddleOCR-VL 1.5 and 1.6, 1041
  added tokens).
- Result: all seven identical: 0 differing tokens across 885k, equal info, equal added-token tables.
- Implication: the port and the 0.23 bump change nothing llguidance sees. Remaining API changes were mechanical:
  `add_tokens` / `add_special_tokens` take owned tokens and, with `with_normalizer`, return `Result`.

## Run 2 - 2026-09-27 06:00

- Question: does llguidance / toktrie 1.4.0 -> 1.8.0 change the token masks our constraints produce?
- Command: two throwaway binaries (Cargo cannot hold two semver-compatible versions in one graph), each pinning one
  version and building the parser factory exactly as `build_llg_factory` does with the in-tree byte tokenizer. Each
  walks three grammars (a regex, a JSON schema with a required object/array, a small Lark grammar) for up to 24
  steps, recording the allowed-token mask at every step and consuming a pick fixed by the step index, then prints a
  digest per mask. Same seven `tokenizer.json` files as Run 1.
- Result: all 400 mask digests identical between 1.4 and 1.8 (`diff` empty). The regex and JSON walks run all 24
  steps on every tokenizer; the Lark grammar completes after 2-5 tokens.
- Implication: the bump is safe for constrained decoding; the workspace compiles with no API changes and the 174
  grammar/constraint/tool tests pass.
