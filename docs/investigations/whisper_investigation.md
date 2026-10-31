# Whisper (#737, encoder shared with #564)

The standalone Whisper speech recognizer (encoder-decoder), served as a transcription model beside Parakeet. Its
encoder is the one the Whisper-encoder audio LLMs (#564) need, so it lives in inference-nn.

## Run 1 - 2026-10-31 11:21

**Question:** what do the cached checkpoints say the port must read from config rather than hard-code?

**Command:** `config.json` and `generation_config.json` of openai/whisper-large-v3-turbo, -large-v3, -small,
-base.en (all cached).

**Finding:**
- Shapes: turbo d_model 1280, 32 encoder / 4 decoder layers, 20 heads, FFN 5120, 128 mels, vocab 51866; large-v3 the
  same with 32 decoder layers; small 768 / 12 / 12, 80 mels, vocab 51865; base.en 512 / 6 / 6, 80 mels, vocab
  51864. All: 1500 source positions, 448 target positions, GELU, no embedding scale.
- The special ids move between versions, so they come from generation_config: `no_timestamps_token_id` 50364 (v3)
  / 50363 (v1/v2 multilingual) / 50362 (.en), timestamps start one past it; `task_to_id` transcribe 50360 / 50359,
  translate 50359 / 50358; `decoder_start_token_id` 50258 (multilingual) / 50257 (.en); EOS 50257 / 50256;
  `begin_suppress_tokens` [220, EOS]; `suppress_tokens`; `lang_to_id` (100 languages in v3, 99 in v1/v2, none
  for .en); `alignment_heads` (turbo [[2,4],[2,11],[3,3],[3,6],[3,11],[3,14]]) for DTW word times;
  `max_initial_timestamp_index` 50.
- Files: transformers layout with `tokenizer.json`, one `model.safetensors` (turbo and large-v3 F16).

**Next:** the encoder in inference-nn, the decoder and decoding in inference-models-speech.

## Run 2 - 2026-10-31 11:46

**Question:** does the port (shared encoder, decoder with greedy timestamp decoding, 30 s windows, DTW word times)
match transformers and OpenAI's reference on whisper-large-v3-turbo?

**Command:** `inference serve -m openai/whisper-large-v3-turbo`, `/v1/audio/transcriptions` `verbose_json` with word
granularity, on synthetic espeak-ng clips (3.4 s, 20.1 s, and 44.5 s of three passages with 2 s pauses, two windows);
transformers 5.19 `pipeline(..., return_timestamps="word")`; OpenAI `whisper` 20250625 (scratch venv)
`transcribe(word_timestamps=True, temperature=0, condition_on_previous_text=False, fp16=True)`.

**Finding:**
- Text equals both references on all three clips (the 44.5 s one across a window boundary).
- First cut of word times: within 0.02 s of OpenAI on the 3.4 s and 20 s clips. transformers differs from both at
  sentence ends ("dog." ends 3.34 s there, 2.92 s in OpenAI and ours): it takes word times its own way.
- On the 44.5 s clip two words were off: the first word after a 2 s pause started 1.56 s early (in the silence), and
  the first word of window 2 by 0.4 s. OpenAI's `add_word_timestamps` caps words at sentence ends and after pauses,
  holds a segment's first and last word to its timestamps, merges punctuation keeping the word's own times, and
  resumes the next window at the last word's end; ported (`timing::place_words`, the word-timestamp seek).
- After the port: max deviation 0.04 s (two encoder positions) over 105 words, 0.02 s on the other clips.
- Our verbose_json segments differ from OpenAI's at one comma: the API builds segments from words (sentence ends and
  pauses, as for Parakeet), not from the model's timestamp pairs. transformers' tokens confirm the decoder's pairs
  (`<|9.10|><|9.36|>`).

## Run 3 - 2026-10-31 11:46

**Question:** do the other checkpoint generations (special ids, mel bins, depths) work?

**Command:** as Run 2 on openai/whisper-base.en (English-only ids, 80 mels), -small (v1/v2 ids), -large-v3 (32
decoder layers), against OpenAI's same-named models.

**Finding:**
- Text equals OpenAI on every checkpoint and clip.
- Word times: small and large-v3 within 0.02 s on all words; base.en within 0.02 s on the short clip, but 17 of 109
  words past 0.02 s (max 0.18 s) on the 44.5 s clip.
- base.en with `--dtype f16` (OpenAI's precision; the default here is BF16): 9 words, max 0.18 s; with F32
  attention scores, 7 words, max 0.12 s. OpenAI's own FP16 and FP32 runs agree within 0.02 s, so this is not
  precision noise alone.
- Our per-window timestamp segments and seek points equal OpenAI's no-word-timestamp segments exactly (window 1: 6
  segments ending 28.36 s; window 2 from frame 3000), and the alignment heads equal OpenAI's built-in table. The
  inputs to the alignment are the same; the gap is inside it, on the smallest model only. Left open; the scores stay
  FP16 as OpenAI computes them.

**Next:** language detection, unit and engine tests, docs.

## Run 4 - 2026-10-31 11:56

**Question:** language identification and forcing, the parity script, and English-only checkpoints.

**Command:** synthetic Spanish and German espeak-ng clips on large-v3-turbo; `language=en` on the Spanish clip,
`language=xx`; `scripts/whisper_parity.py openai/whisper-large-v3-turbo turbo <4 clips>` (FP16 both sides);
base.en with no language and with `language=de`.

**Finding:**
- Spanish and German are identified (`spanish`, `german`), transcripts equal OpenAI's, word times within 0.02 s.
  `language=en` on Spanish gives Whisper's English rendering; `xx` is an invalid request.
- The parity script passes on all four clips at FP16: max |dt| 0.020 s on each (BF16 had reached 0.04 s on the long
  one).
- base.en first accepted `language=de` and ignored it (no language tokens); it now refuses any language but
  English and reports `english`.

**Next:** full local CI. Follow-ups: the translate task and `/v1/audio/translations`, beam search, temperature
fallback, `prompt` as previous text, the audio.cpp `crisperwhisper` GGUF (#747), the encoder for #564.
