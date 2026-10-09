# Gemma 3 GGUF with an mmproj: vision, tokenizer and projector loading

Found while timing #252 on gemma-3-4b-it (GGUF Q4_K_M, F16 mmproj): images were misread, long prompts decoded blank
lines, and neither common mmproj file loaded. Same on master; llama-server on the same files was the reference.

## Run 1 - 2026-10-08 20:50

Question: why does Gemma 3 GGUF describe a page of text as "colorful, intricate patterns"?

Commands, in order, each answering the question before it:
1. A synthetic image (a red square and a blue circle on white), "What shapes and colors are in this image?":
   llama-server answers "Shapes: Square, Circle"; we answered "a colorful, abstract design". The tower or its wiring is
   wrong, not the model's reading of dense text.
2. A temporary dump (not committed) of our pixel values, tower output and projected features, against a transformers
   `SiglipVisionModel` and Gemma 3's projector built from the mmproj's own tensors and fed our pixel values: tower
   cosine 0.963 (ours BF16, reference F32), features 0.995. Pixel values are right (white 1.0, the dark channels
   -0.84). The encoder is not the problem.
3. The GGUF vocabulary has 262144 tokens and no `<image_soft_token>`, which is id 262144 (`image_token_index`) in
   transformers' 262145-token tokenizer. The loader drops the base model's tokenizer.json for that one-token mismatch
   and converts the GGUF's own vocabulary, so the processor's expanded `<image_soft_token>` text tokenizes into plain
   pieces: no position carries the image id, the features are never placed, and the model describes an image it never
   saw.

Fix: for Gemma 3 with a projector, the GGUF tokenizer gets `<image_soft_token>` as a special token at
`image_token_index` when it is the first id past the vocabulary (an error otherwise), and the model maps image-token ids
to 0 before the embedding lookup (the GGUF embedding table has no row for them; their rows are replaced by image
features), as transformers does when the image id is past the vocabulary.

After: the shapes answer matches llama-server word for word, paged and unpaged; the page is read as a text page.

## Run 2 - 2026-10-08 20:50

Question: the 6212-token image prompt still decodes blank lines. Why?

Raw: the same prompt is 3678 tokens in llama-server and 6212 for us; README text alone is 3423 against 5847, and our
summary spelled "LLM" as "LLL". The GGUF tokenizer conversion turned `tokenizer.ggml.model = llama` (SentencePiece)
into a Unigram model over `tokenizer.ggml.scores`. Gemma's (and Llama 2's) SentencePiece models are BPE: the scores
rank merges, they are not piece log-probabilities, and Unigram's best path over them splits rare long pieces into many
short ones.

Prototype (Python, `tokenizers` 0.22, against the base model's tokenizer.json): Unigram 5801 tokens for the README
against HF's 3407; BPE with merges recovered from the scores, ordered as transformers' `SentencePieceExtractor` orders
them (every split of a piece into two vocabulary pieces, by the merged piece's score, stable on piece ids), plus the
control and user-defined pieces as added tokens: 3407 tokens, identical ids. A first try that ranked ties by the left
piece's id, or drew merges from normal pieces only, missed by 1-2%.

Fix: `llama` and `gemma` GGUF vocabularies convert to that BPE; `replit` keeps Unigram. `add_space_prefix` defaults to
true when the GGUF names none (llama.cpp's default; the Unigram path always prepended).

Check (a temporary Rust test, not committed, the Rust conversion against tokenizer.json on three files):
- Gemma 3: README 5071/5071, a docs page 1238/1238, a Rust source file 17537/17537, identical ids throughout.
- TinyLlama (Llama 2 SentencePiece): the docs page identical (1440); README 5863/5863 with one tie split differently
  (two-space then `|` where transformers has a space then space-`|`; llama-server's tokenize splits it as we do); the Rust file 19486 against
  19450 (llama-server 19502). The old Unigram missed both files by more.

After: the README prompt is 3423 tokens, as in llama-server, and the summary opens with llama-server's words; the image
plus README prompt (3679 tokens) answers with a summary instead of blank lines.

Also: an mmproj with no `general.type` (ggml-org's converter) or `general.type = clip-vision` (unsloth's) was refused;
components now merge with `mmproj`, `clip-vision`, or no type and `general.architecture = clip`, as llama.cpp loads
them. #252's Gemma 3 timing ran on the 6212-token tokenization; it measured the attention path on that prompt length,
which is unaffected.

Review follow-ups: the Gemma 3 processor now fails a request with images when the tokenizer has no image token, where
it used to drop the images and answer anyway (the reason this went unnoticed); the image-token remap compares against
a device scalar made at load, so a decode step copies nothing to the GPU; an untyped `clip` component must name a
`clip.projector_type`; merge sorting folds -0.0 into 0.0 as Python's sort does (Gemma 3's vocabulary has one -0.0
piece). `sentencepiece_local_gguf_matches_hf_tokenizer` (ignored; `INFERENCE_RS_SPM_GGUF` and
`INFERENCE_RS_SPM_TOKENIZER_JSON`) passes on the Gemma 3 pair; on TinyLlama, whose GGUF scores are all 0.0, it stops at
"hello  world", a tie split differently from transformers, as expected. Not checked, no local file: Gemma 3n GGUFs
from before llama.cpp wrote its full 262400-token vocabulary would lack `<image_soft_token>` and `<audio_soft_token>`
the same way; the processor check above does not cover Gemma 3n.
