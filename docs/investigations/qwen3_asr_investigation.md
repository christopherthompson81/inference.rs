# Qwen3-ASR (#560)

Qwen3-ASR is the Qwen3-Omni audio encoder (AuT) in front of a dense Qwen3 decoder. The encoder is shared with
Qwen3-Omni (#559), so it goes in once, where both can use it.

## Run 1 - 2026-10-31 09:10

**Question:** what do the references (transformers 5.19 `qwen3_asr`, llama.cpp 970b51ce7 `qwen3a.cpp`) and the
published GGUFs say the port needs?

**Command:** read `transformers/models/qwen3_asr/{modeling,feature_extraction,processing}_qwen3_asr.py`, llama.cpp
`tools/mtmd/models/qwen3a.cpp`; gguf-py over `Qwen3-ASR-1.7B-Q8_0.gguf` and `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`
(ggml-org); tensor names of the Qwen/Qwen3-ASR-1.7B safetensors index.

**Finding:**
- Features: Whisper's log-mel (n_fft 400, hop 160, 128 slaney mels to 8 kHz, log10 clamped to max - 8, then
  (x + 4) / 4), clips under 8000 samples zero-padded, mel frames right-padded to a multiple of 100 with a frame mask.
- Encoder: mel cut into 100-frame chunks; three conv2d (3x3, stride 2, pad 1, 480 channels) + GELU per chunk;
  `conv_out` (no bias) over channels x freq (480 x 16); sinusoidal positions 0..12 restarted per chunk; padded
  positions dropped; 24 pre-norm layers (LayerNorm, biased q/k/v/out, GELU FFN 4096), `ln_post`; projector
  linear(1024, 1024), GELU, linear(1024, 2048). 13 tokens per full chunk.
- Attention is windowed: `n_window_infer` 800 mel frames = 8 chunks = 104 tokens per window (block-diagonal,
  bidirectional), the last window shorter. llama.cpp's graph (`build_vit` over every position) attends over the whole
  clip, which matches only up to 8 s of audio; to confirm against transformers on a longer clip.
- Decoder: Qwen3 dense (28 layers, hidden 2048, 16/8 heads, head_dim 128, q/k norms, tied embeddings, theta 1e6).
  The config's interleaved M-RoPE sections [24, 20, 20] reduce to 1-D RoPE, since audio positions are the same on
  all three axes; the transformers `-hf` config says plain `rope_type default`.
- Prompt: ChatML; the audio is `<|audio_start|>` + one `<|audio_pad|>` (151676) per encoder token +
  `<|audio_end|>` in the user turn. The model answers `language <NAME><asr_text><text>`; prefilling the assistant
  turn with `language <NAME><asr_text>` forces the language. EOS `<|im_end|>`.
- Checkpoint names (original repo): `thinker.audio_tower.{conv2d1-3, conv_out, layers.N.*, ln_post, proj1, proj2}`,
  `thinker.model.*`, `thinker.lm_head`; the `-hf` repo renames the projector `multi_modal_projector.linear_{1,2}`.
- GGUF: text `qwen3vl` arch, `n_deepstack_layers` 0, `rope.dimension_sections` [24, 20, 20, 0], ChatML template,
  BOS = EOS = `<|im_end|>`; mmproj `clip.audio.projector_type qwen3a`, `a.conv2d.N`, `a.conv_out`,
  `a.position_embd` (stored), `a.blk.N.{ln1, ln2, attn_q/k/v/out, ffn_up/down}`, `a.post_ln`, `mm.a.mlp.N`.
  Neither file stores `n_window` or `n_window_infer`.

**Next:** map how an existing audio-in LLM (Granite Speech) is wired here: features, placeholders, loader, GGUF
mmproj, serving.

## Run 2 - 2026-10-31 09:29

**Question:** does the port (shared Whisper log-mel and encoder layer, the AuT encoder, Qwen3 text) reproduce
transformers on the BF16 checkpoint, short of the attention window and across it?

**Command:** two synthetic espeak-ng clips (3.4 s, and 20.1 s to cross the 8 s window), resampled 22.05 -> 16 kHz and
stored as 16-bit WAV; transformers 5.19 `Qwen3ASRForConditionalGeneration` with the original repo's weights renamed
into its layout (`thinker.audio_tower.proj{1,2}` -> `multi_modal_projector.linear_{1,2}`, `thinker.model` ->
`model.language_model`), greedy; ours through `inference serve` on the same weights (a local dir with a generated
`tokenizer.json` and the GGUF's ChatML template), `audio_url` data URL, temperature 0.

**Finding:**
- transformers cannot load the original repos itself: every tensor reports missing (it expects the `-hf` layout) and
  the nested `thinker_config` falls back to defaults (`output_dim` 3584).
- Placeholders: 45 and 262 audio tokens; our prompts are 60 and 277 tokens, the same 15 around them.
- Transcripts match word for word: "The quick brown fox jumps over the lazy dog." and the 61-token long clip, each
  prefixed `language English<asr_text>`.
- First-token top 5: same ids in the same order (`language` at -0.0); the tail at -14 to -17 within 0.5 nats
  (short: ours -14.875 / -15.875 / -17.125 vs -14.375 / -15.75 / -16.75), two BF16 steps of a top logit near 39.
- Our chat route takes audio as `audio_url`, not OpenAI's `input_audio` part; a gap of its own.

**Next:** the GGUF path (`qwen3vl` text + `qwen3a` mmproj), checked the same way, and against llama.cpp on the long
clip, where its full attention should differ.

## Run 3 - 2026-10-31 09:32

**Question:** does the GGUF pair (ggml-org's `Qwen3-ASR-1.7B-Q8_0.gguf` + `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`) load and
transcribe as the BF16 checkpoint does, and does llama.cpp's encoder really attend across the 8 s window?

**Command:** `inference serve -m /mnt/data/models -f Qwen3-ASR-1.7B-Q8_0.gguf --mmproj ... --tok-model-id <local dir>`
with the Run 2 clips and client; llama.cpp 970b51ce7 `llama-mtmd-cli -p "<__media__>" --temp 0 -ngl 99` on the same
WAVs.

**Finding:**
- The GGUF family binds as `Qwen3Asr` (`qwen3vl` text + `clip.audio.projector_type = qwen3a`); both transcripts
  match Run 2 word for word, prompts 60 and 277 tokens. Decode about 230 tok/s (BF16: 150).
- llama.cpp gives the same two transcripts. Its preprocessor splits the mel into 800-frame windows
  (`tools/mtmd/mtmd-audio.cpp:724`, "matching n_window_infer=800") and encodes each on its own (3 audio chunks
  for the 20 s clip), so its full attention never crosses a window: the Run 1 suspicion was wrong.

**Next:** tiny random-weight engine test, binding and sizing pins, docs.

## Run 4 - 2026-10-31 09:42

**Question:** does the documented GGUF command work with the original repo as `--tok-model-id` (it has
`vocab.json`/`merges.txt` and a `tokenizer_config.json` without a template, no `tokenizer.json`)?

**Command:** `HF_HUB_OFFLINE=1 inference serve --format gguf -m /mnt/data/models -f Qwen3-ASR-1.7B-Q8_0.gguf
--mmproj ... --tok-model-id Qwen/Qwen3-ASR-1.7B`, then the Run 2 client.

**Finding:**
- The tokenizer comes from the GGUF (gpt2 BPE, 151,936 tokens), but chat requests were refused: "Received messages
  for a model which does not have a chat template". The GGUF's template was taken only when the repo had no
  `tokenizer_config.json` at all; this repo has one without `chat_template`.
- With the GGUF's template standing in whenever no repo file holds one (a `.jinja`, a `chat_template` key, or an
  explicit template file), both clips transcribe as in Run 2.

**Next:** full local CI. Left for a follow-up: the transformers `-hf` layout (flat config, `model.audio_tower`,
`model.multi_modal_projector`, `model.language_model`), `/v1/audio/transcriptions` for an audio-in decoder, the
forced-aligner head.

## Run 5 - 2026-10-31 10:23

**Question:** can `/v1/audio/transcriptions` serve an audio-in decoder, and does the transformers `-hf` layout load?

**Change and command:**
- A loader may name an `AudioTranscriber` (the whole prompt around one clip, with the answer prefix a language
  forces, and the answer's parse). inference-api turns a transcription request for such a model into a
  `MultimodalTokens` generation, greedy, budgeted 64 + 16 tokens per second of audio.
- The GGUF pair through `serve`; curl with `response_format` `json`, `verbose_json`, `srt`, and `language` `en`,
  `zh`, `xx`.
- The `-hf` repo's `config.json`, `chat_template.jinja` and safetensors header (by byte range), not its weights.

**Finding:**
- Both clips transcribe as in Run 2. `verbose_json` reports `english`, identified by the model, with duration
  20.12 s. `language=zh` on the English clip: the answer no longer starts with the model's own
  `language English<asr_text>` (the prefix took), and the text stays English. `xx` is refused naming the language.
- The model gives no word timings, so segments were empty and `srt`/`vtt` blank. A transcript without words is now
  one segment over the clip: `1 / 00:00:00,000 --> 00:00:03,439 / The quick brown fox jumps over the lazy dog.`
- The tiny SDK model first answered "Received a request incompatible for this model's category":
  `AutoMultimodalLoader` forwards each trait method by hand and returned the default `None`.
- `-hf` layout: a flat config (`rope_theta` under `rope_parameters`, no audio start/end ids), one
  `model.safetensors` with `model.audio_tower.*`, `model.multi_modal_projector.linear_{1,2}`,
  `model.language_model.*`, no `lm_head` (tied). The real config parses (Transformers layout, theta 1e6, tied). The
  tiny checkpoint renamed into this layout transcribes byte for byte as the original.
- The `-hf` template writes `<|audio_start|><|audio_pad|><|audio_end|>` per audio content part and drops the user's
  text, so a flattened message held no placeholder: "The number of `<|audio_pad|>` tokens (0) must match the number
  of audio clips (1)". A template that contains the placeholder now gets the content parts (prompt 70 tokens on the
  tiny model); ChatML keeps the flattened text.

**Next:** the forced-aligner head (Qwen3-ForcedAligner-0.6B) for word timestamps; OpenAI's `input_audio` part.
