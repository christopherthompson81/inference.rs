# Whisper-encoder audio LLMs (#564)

Voxtral, Ultravox, Qwen2-Audio, Audio Flamingo 3, GLM-ASR and MERaLiON-2 share one recipe: the bidirectional Whisper
encoder (now `inference_nn::audio::whisper::WhisperEncoder`, #748), a projector, and a text decoder we already have.

## Run 1 - 2026-10-31 12:59

**Question:** what does Voxtral (offline, mistralai/Voxtral-Mini-3B-2507) need beyond the shared encoder, and what do
its GGUFs and prompt look like?

**Command:** the repo's `config.json` / `preprocessor_config.json` / weight index; transformers 5.19
`models/voxtral`; gguf-py over ggml-org's `Voxtral-Mini-3B-2507-Q4_K_M.gguf` and `mmproj-...-Q8_0.gguf`; llama.cpp
970b51ce7 `tools/mtmd/models/whisper-enc.cpp`; mistral_common 1.x `encode_chat_completion` /
`encode_transcription` with a 2 s clip.

**Finding:**
- Encoder: the Whisper encoder exactly (`audio_tower.{conv1,conv2,embed_positions,layers.N,layer_norm}`, 32 layers,
  1280 wide, 128 mels). Projector: four encoder frames stacked (1280 x 4 = 5120), `multi_modal_projector.linear_1`
  (5120 -> 3072, no bias), GELU, `linear_2` (3072 -> 3072, no bias). Text: Llama layout (30 layers, 3072, 32/8 heads,
  head_dim 128, theta 1e8, vocab 131072) under `language_model.`.
- Features: each clip padded to a multiple of 30 s, one Whisper log-mel over the padded clip (one peak), cut into
  3000-frame chunks; each chunk is 1500 positions, 375 audio tokens.
- Prompt (mistral_common): chat `[BOS][INST][BEGIN_AUDIO]` + `[AUDIO]` x 375 per chunk + text + `[/INST]` (ids 1, 3,
  25, 24, ..., 4; no spaces); transcription `... [/INST]lang:en[TRANSCRIBE]` (id 34). `audio_token_id` 24.
- The HF repo has no `tokenizer.json` or template (transformers uses mistral_common); the GGUF has a gpt2/tekken
  tokenizer with these specials, but its chat template is Unsloth's Devstral one, so the loader must bring its own.
- mmproj: `clip.projector_type voxtral`, `clip.audio.projector.stack_factor 4`, `a.conv1d.N`, `a.position_embd`,
  `a.blk.N.{ln1,attn_q/k/v/out,ln2,ffn_up/down}`, `a.post_ln`, `mm.a.mlp.N` (Q8_0).

**Next:** the Voxtral model and its GGUF path, checked against llama.cpp on the Q4_K_M GGUF.

## Run 2 - 2026-10-31 17:32

**Question:** what does loading ggml-org's Voxtral GGUF do today, and where does the chat template come from?

**Commands:** `inference run --cpu --format gguf -m /mnt/data/models/voxtral-mini-3b -f Voxtral-Mini-3B-2507-Q4_K_M.gguf`
(projector beside it); gguf-py over the text GGUF's metadata; reading the GGUF/multimodal template code.

**Finding:**
- The load stops before any template: `multimodal GGUF requires its original config.json, but the GGUF files do not
  identify one unambiguous Hugging Face base model`. Neither GGUF carries `general.base_model.*`; the text GGUF is
  `general.architecture llama` with no name, so only the projector (`clip.projector_type voxtral`) says Voxtral.
- With `--tok-model-id mistralai/Voxtral-Mini-3B-2507` it would still fail: `VoxtralForConditionalGeneration` is in no
  loader table. The existing `voxtral` loader is Voxtral Realtime (`VoxtralRealtimeForConditionalGeneration`,
  Mistral-native consolidated weights), a different model.
- The embedded template (9838 chars) is Unsloth's Devstral one; it opens every chat with Devstral's OpenHands agent
  system prompt. Precedence today: a template file or an explicit template, else the GGUF's metadata template, and a
  loader's `default_chat_template` only when none of those exists, so a loader can never correct a wrong GGUF template.
- The native multimodal GGUF path (`resolve_native_multimodal_gguf`, keyed by projector type and text architecture)
  needs a `config.json`; `synthesize_normal_config` builds one for text GGUFs, nothing does for audio projectors.

**Implies:** the template fix lands with the offline Voxtral port: (1) the model (shared `WhisperEncoder`, stack-4
projector, the shared decoder as `Mistral`), (2) its audio inputs processor, (3) loader + table row (`voxtral-chat`, module
`voxtral_chat`; renaming Realtime would touch 19 files for no new capability), (4) a loader template that wins over a GGUF's embedded one,
(5) GGUF: `("voxtral", "llama")` in the native multimodal registry, projector bindings, and a config synthesized from
the GGUF metadata so no base repo is needed, (6) tests, then parity against llama.cpp's mtmd on the same Q4_K_M.

## Run 3 - 2026-10-31 18:05

**Question:** with the model, loader, template override and GGUF config synthesis in place, does ggml-org's Q4_K_M
pair load with no base repo, and does it match llama.cpp?

**Commands:** `inference run --cpu --format gguf -m /mnt/data/models/voxtral-mini-3b -f Voxtral-Mini-3B-2507-Q4_K_M.gguf`
(text, then `--audio` with the synthetic two-voice garden dialogue, 28.7 s); `inference serve --cpu` + greedy
`/v1/chat/completions` with an `input_audio` part; llama.cpp 970b51ce7 `llama-mtmd-cli -ngl 0 --temp 0` on the same
files (the GPU holds the LAN server, so both on the CPU).

**Finding:**
- Loads with no base repo or `--tok-model-id`: the config is synthesized from the GGUF metadata. Text prompt is 16
  tokens (`<s>[INST]...[/INST]`), so the Devstral template is gone. Audio prompt 390 tokens (375 for the one chunk).
- Text-only greedy ("Name three rivers in Europe..."): ours and llama.cpp agree for ~30 tokens ("Sure! Here are three
  rivers in Europe and a fact about each: 1. **Danube River (Europe's second-longest river)**"), then drift, as two
  Q4_K CPU kernels do.
- Audio greedy: ours "The conversation is about planning a garden project. The speakers are discussing the garden
  plans they received...", llama.cpp "This conversation is about a garden project. The speakers are discussing the
  garden plans, deciding on the location for the vegetable beds...". Same with llama.cpp forced to
  `--chat-template mistral-v7-tekken` (its default is the GGUF's Devstral template). Our first-token top-2: 1784 at
  0.572, 4380 at 0.381, a close call.
- Same prompt framing in both (`mtmd.cpp`: marker first, `[BEGIN_AUDIO]` + embeddings); both projectors use erf GELU.
- llama.cpp's Whisper preprocessing (`mtmd-audio.cpp`) is openai-whisper's: 30 s of zeros appended and the log-mel
  normalized over that span; transformers (what mistral's processor uses) pads only to the next 30 s boundary. On a
  28.7 s clip the peak and the padded frames differ, enough to flip a 0.57/0.38 call. llama.cpp is not the reference
  for Voxtral's features. Dead end for parity.
- `llama-completion` in build_cuda segfaults at start (with and without the GPU); `llama-mtmd-cli` reading the prompt
  from stdin works as a text-only runner.

**Next:** transformers on the BF16 checkpoint as the reference (also the only way to check our safetensors path):
greedy answers and first-token probabilities, ours BF16 vs transformers BF16.

## Run 4 - 2026-10-31 18:00

**Question:** does our Voxtral match transformers (the reference mistral's processor runs through) on the original
weights?

**Commands:** `hf download mistralai/Voxtral-Mini-3B-2507 --include "*.safetensors"`; `scripts/voxtral_parity.py
mistralai/Voxtral-Mini-3B-2507 <wavs>` (transformers 5.19 + mistral_common 1.13 in a scratch venv; both F32 on the CPU,
greedy, 48 tokens; the question "What is said in this recording? Answer in two sentences." per clip, plus a text-only
prompt).

**Finding:**
- First start failed: `missing field audio_config`. The repo ships `config.json` (HF names, `model-*.safetensors`) and
  Mistral's native `params.json` (`consolidated.safetensors`); the safetensors paths took `params.json` whenever
  present, a rule for Voxtral Realtime, whose loader reads the native config. Now `params.json` wins only without a
  `config.json` or when it names an architecture whose loader reads the native config
  (`VoxtralRealtimeForConditionalGeneration`). Voxtral Realtime's and Small's repos ship both too.
- Exact match on every case: text-only (192 chars), the synthetic garden dialogue 28.7 s (258), a short synthetic clip
  (151), a 44.5 s two-chunk clip (246), a 74.5 s three-chunk concatenation (250), a 9.3 s Spanish clip (132).
- So the features (one log-mel over the clip padded to whole 30 s chunks), the encoder, the stacking and the projector
  agree with transformers; the GGUF-vs-llama.cpp first-token split in Run 3 is llama.cpp's openai-whisper-style
  padding, not ours.

**Next:** docs (supported models, GGUF support), then local CI. Transcription mode (`lang:xx[TRANSCRIBE]` through the
transcriptions endpoint) is a separate step.

## Run 5 - 2026-10-31 19:34

**Question:** does Voxtral's transcription mode, served through `/v1/audio/transcriptions`, match transformers'
`apply_transcription_request`?

**Commands:** mistral_common `encode_transcription` with and without a language;
`scripts/voxtral_parity.py mistralai/Voxtral-Mini-3B-2507 <garden dialogue> <short> <multi 44 s>` (transcripts with
`language=en`, max 192 tokens; the chat cases again).

**Finding:**
- Prompt: `<s>[INST][BEGIN_AUDIO][AUDIO]x375[/INST]lang:en[TRANSCRIBE]`, or `...[/INST][TRANSCRIBE]` without a language;
  the answer is the bare transcript.
- Exact match: transcripts of 406, 44 and 591 chars (the 44 s clip across two chunks), and all chat cases again.
- The endpoint already refuses `task=translate` for every audio-in decoder.

**Next:** Qwen2-Audio (transformers `qwen2_audio`), then GLM-ASR, Audio Flamingo 3; Ultravox and MERaLiON-2 are
remote code only.

## Run 6 - 2026-10-31 20:50

**Question:** does our Qwen2-Audio (Qwen/Qwen2-Audio-7B-Instruct, BF16 shards, 16 GB) match transformers?

**Commands:** transformers 5.19 `models/qwen2_audio` read for the encoder (the Whisper encoder; avg-pool by 2 *before*
the final layer norm; a padding mask over the clip's real frames) and processor (one 30 s window per clip, truncated;
`Audio N: <|audio_bos|><|AUDIO|><|audio_eos|>\n` per clip from the processor's own template; the repo's tokenizer
template is text-only ChatML); `scripts/audio_llm_parity.py` (generalized from voxtral_parity.py: the transformers
class comes from `architectures`, transcripts only for a model with `apply_transcription_request`), F32 on the CPU.

**Finding:**
- The released `text_config` omits hidden size, layers, heads and activation; transformers' `Qwen2Config` defaults
  fill them (4096, 32, 32 heads and 32 KV heads, SiLU, untied). Our config applies the same defaults.
- Truncating the encoder input to the clip's real positions equals transformers' padding mask: every pooled token it
  keeps averages two real positions.
- First run: text, the 28.7 s dialogue and the 44 s clip (cut to 30 s) match; the short clip did not: same first
  sentence, then transformers stopped and ours went on. With `repetition_penalty` 1.0 in both, an exact match; with
  the checkpoint's 1.1, ours 0.45 " This" vs 0.29 `<|im_end|>`, transformers `<|im_end|>` 19.71 vs " This" 19.09
  (processed logits).
- Cause: our prompt was 134 tokens to transformers' 125. `tokenizer.json` lists 3 added tokens and
  `tokenizer_config.json`'s `added_tokens_decoder` 3288, the audio markers among them; transformers merges the
  latter, we read only `tokenizer.json`, so `<|audio_bos|>`/`<|audio_eos|>` became text pieces (`<|AUDIO|>` worked
  only because the processor registers its placeholder). The penalty saw different tokens; at 1.0 the extra
  pieces happened not to change the answer.
- Fix (engine-wide): `get_tokenizer` merges `added_tokens_decoder` entries missing from `tokenizer.json` at their
  declared ids. Prompt now 125 tokens.

- Full parity again, the checkpoint's 1.1 penalty in both: exact match on text (205 chars), the short clip (83), the
  dialogue (258), the 44 s clip (246) and a Spanish clip (169).

**Next:** GLM-ASR (its own encoder: rotary positions, pre-norm layers, no learned
positions), Audio Flamingo 3; Ultravox and MERaLiON-2 are remote code only.

## Run 7 - 2026-10-31 21:55

**Question:** does our GLM-ASR (zai-org/GLM-ASR-Nano-2512, 4.5 GB) match transformers?

**Commands:** transformers 5.19 `models/glmasr` read; `scripts/audio_llm_parity.py zai-org/GLM-ASR-Nano-2512 <short>
<garden dialogue> <multi 44 s>` (chat and transcription mode), F32 on the CPU.

**Finding:**
- The tower is Whisper's layers with rotary positions (half-split, the first 32 of each head's 64 dims) in place of the
  table, under Llama-style names (`input_layernorm`, `o_proj`, `post_attention_layernorm`, `mlp.fc1`/`fc2`, final
  `norm`). The shared `WhisperEncoder` now takes `WhisperLayerNames` and `WhisperPositions::Rotary` instead of a second
  copy of the layer.
- Each 30 s window is its own feature extraction (own log-mel peak) and the tower reads every window whole, padding
  included (no mask); four positions stack into one 5120-wide input; the projector widens to 4096 and back to 2048;
  each window keeps `((positions - 4) / 4) + 1` tokens. Up to 21 windows (655 s).
- The repo's template writes each audio part itself, so GLM-ASR's prefixer adds nothing (with one, the placeholder
  appeared twice: the prefixer writes into the text part, which a parts-writing template keeps).
- Transcription: the processor's `apply_transcription_request` prompt (`Please transcribe this audio into text`); no
  language option in the reference, so a requested language is ignored.
- Exact match on the first run: text (185 chars), chat about three clips (44, 212, 235), transcripts (44, 406, 599, the
  last across two windows).

**Next:** Audio Flamingo 3 (noncommercial licence), then Ultravox and MERaLiON-2 (remote code only).

## Run 8 - 2026-10-31 23:30

**Question:** does our Audio Flamingo 3 (nvidia/audio-flamingo-3-hf; the sharded files only, 16 GB, the repo also
ships a duplicate single file) match transformers?

**Commands:** transformers 5.19 `models/audioflamingo3` read; `scripts/audio_llm_parity.py nvidia/audio-flamingo-3-hf
<short> <garden dialogue> <multi 44 s>` (chat and transcription mode), F32 on the CPU (7B in F32 does not fit the
3090 even without the LAN server).

**Finding:**
- AF3 is Qwen2-Audio's tower (masked to each window's real positions, pooled by two before the final norm) over 30 s
  windows as GLM-ASR cuts them (each its own log-mel, up to 600 s), a two-layer MLP projector with biases, and a
  Qwen2.5 decoder whose config uses transformers 5's names (`rope_parameters`, `hidden_size` in the audio config).
  One `PooledAudioSpec` now drives both models (projector kind, window limit); the loaders come from one macro.
- The repo's template writes `<sound>` for each audio part (one per message); the processor's transcription prompt
  is `Transcribe the input speech.` with no language.
- First run stopped on the transcript step (no transcriber yet); the script now prints each case as it finishes.
- Exact match: text (191 chars), chat about three clips (65, 39, 82), transcripts (80, 439, 621; the last across two
  windows).

**Next:** Ultravox and MERaLiON-2 have no transformers implementation; a reference means running their repos' remote
code, which needs the user's go-ahead.

## Run 9 - 2026-11-01 00:20

**Question:** how does Ultravox (fixie-ai/ultravox-v0_5-llama-3_2-1b) load, and what does its remote code do?

**Commands:** the Hub's file lists; the repo's `ultravox_model.py`, `ultravox_processing.py` (read, not run); gguf-py
over ggml-org's `ultravox-v0_5-llama-3_2-1b-GGUF` (`Llama-3.2-1B-Instruct-Q8_0.gguf` + `mmproj-...-f16.gguf`).

**Finding:**
- The safetensors repo holds only the tower (Whisper large-v3-turbo's encoder) and the projector; the decoder comes
  from `text_model_id` (meta-llama/Llama-3.2-1B-Instruct, gated) and the config has no `text_config`. The GGUF pair
  holds both halves, so the GGUF path needs no second repository; the port loads from GGUF only.
- Projector (`clip.projector_type ultravox`, `stack_factor 8`): zero-pad the encoder positions to a multiple of 8 and
  stack them (1280 x 8 = 10240), RMSNorm (`mm.a.norm_pre`, eps 1e-6), `mm.a.mlp.1` 10240 -> 4096 (no bias), SwiGLU
  with the *second* half gated (`silu(gate) * x`, 4096 -> 2048), RMSNorm (`mm.a.norm_mid`), `mm.a.mlp.2` 2048 -> 2048.
- Audio: one log-mel over the whole clip (WhisperFeatureExtractor, `padding="longest"`, no 30 s pad), cut into 3000-frame
  chunks; the first chunk is not padded, continuation chunks are zero-padded to 3000 and masked to their real frames.
  `audio_lens` comes from the waveform mask sliced every hop (`ceil(len / 160)`); each chunk takes
  `ceil(audio_lens / 16)` tokens, filled from its stacked frames `[:token_len]`.
- The placeholder `<|audio|>` is not in the vocabulary; the processor writes the tokenizer's EOS (`<|eot_id|>`) per
  token, which the turn ends use too. Placement goes by position, so we mark the run with an unused Llama special
  token; no repetition penalty in the generation config, so the id the run carries changes nothing.
- The encoder is Whisper's with positions cut to the input length (`embed_positions[:T]`) and a key mask for padded
  continuation chunks: padded positions are computed (their queries attend to real keys only) and can reach the last
  kept token. Exact parity needs key-limited attention in the shared encoder.

**Next:** key-limited encoding, the model, GGUF bindings and config synthesis (as Voxtral's), then the remote code as
reference (transformers 4.48-era code; F32 on the CPU; ours on the Q8_0/F16 GGUFs, so agreement, not exact match).

## Run 10 - 2026-10-31 22:30

**Question:** does our Ultravox on ggml-org's GGUF pair (Q8_0 text, F16 projector) agree with the repo's remote code
(fixie-ai/ultravox-v0_5-llama-3_2-1b with meta-llama/Llama-3.2-1B-Instruct, F32 on the CPU)?

**Commands:** `inference serve --cpu --format gguf -m /mnt/data/models/ultravox-1b -f Llama-3.2-1B-Instruct-Q8_0.gguf`;
a scratch script running `AutoModel`/`AutoProcessor` with `trust_remote_code` (transformers 5.19, plus accelerate and
peft for its imports), greedy, 64 tokens, the text prompt and "What is said in this recording?" after `<|audio|>` with
no separator, over the synthetic short clip, the Kokoro garden dialogue (28.7 s) and the 74 s two-chunk clip.

**Finding:**
- First load failed: `shape mismatch for multi_modal_projector.linear_1.weight, expected [10240, 10240], got [4096,
  10240]`. The archive reports shapes as (out, in), not ggml's (in, out); the projector width now comes from the
  first dimension.
- The remote code is transformers 4 era. Under 5.19 it needs two shims in the reference script: the removed
  `get_extended_attention_mask` (a 2-D mask to the additive 4-D form) and Whisper layers returning a tuple (5.x
  returns the tensor, and the remote loop takes `[0]`, the batch's first row). It also wants
  `openai/whisper-large-v3-turbo`'s `preprocessor_config.json` and the repo's `ultravox_config.py`.
- The text prompt came out 43 tokens for us, 41 for the reference: our renderer defines `date_string`
  (`"%d, %B, %Y"`, "31, October, 2026"), so the Llama 3 template skips its `strftime_now("%d %b %Y")` branch
  ("31 Oct 2026") that transformers takes. That affects every Llama 3.x template and predates this port; left alone
  here. With `date_string` passed to the reference too, both sides render identical prompts.
- Audio token counts match the reference exactly: 23 (short), 189 (dialogue), 468 (two chunks, 3000 + 4400 frames).
- Greedy outputs agree word for word through the opening sentence or two, then part where the Q8_0 text model departs
  from F32: the rivers (Volga vs Seine as the third), the short clip after the pangram explanation, the dialogue at
  "specifically mention the" vs "specifically to the South", the two-chunk clip after its first sentence. Both
  describe the second chunk's content, so the continuation chunk's padding and key limit work.

**Next:** commit, checks, ask before pushing. The `date_string` difference is a separate fix to raise.

## Run 11 - 2026-11-01 10:40

**Question:** does our MERaLiON-2 (MERaLiON/MERaLiON-2-3B, safetensors) match its repo's remote code?

**Commands:** the repo's `modeling_meralion2.py`, `processing_meralion2.py` and configs read; the GGUF repo's
`llama_cpp_meralion.patch` read; `inference serve --cpu --dtype f32 -m MERaLiON/MERaLiON-2-3B`; a scratch script running
`AutoModelForSpeechSeq2Seq`/`AutoProcessor` with `trust_remote_code`, greedy, 64 tokens, F32 on the CPU; the text
prompt, "What is said in this recording?" and the card's transcription instruction over four synthetic clips (short,
the 28.7 s dialogue, the 74 s clip, a Spanish one).

**Finding:**
- Model: Whisper large-v3's encoder (unmasked over each padded 30 s chunk), `ln_speech` LayerNorm, 15 positions stacked
  (19200), `mlp_adapter` 19200 -> 6400 + SiLU, `silu(gate_proj) * pool_proj` at 6400, `out_proj` -> 2304, into a Gemma 2
  2B decoder (no `lm_head`: tied, though our `gemma2::Config` defaults untied). Gemma scales `inputs_embeds` by
  sqrt(hidden) after the audio is scattered in, so the adapted audio is scaled by 48 too.
- Processor: `len // 480000 + 1` chunks (a clip of whole chunks ends in an empty one), at most 10; every chunk takes 100
  `<SpeechHere>` tokens; each chunk is zero-mean unit-variance normalized over its real samples, then zero-padded and
  given its own log-mel. The model was trained on `Instruction: <text> \nFollow the text instruction based on the
  following audio: <SpeechHere>`; the prefixer writes it.
- `generation_config.json` sets `no_repeat_ngram_size: 6`, which we did not support; added as a logits processor over
  prompt plus output (transformers' rule), filled from generation config like the other defaults.
- The remote code imports `HybridCache`, gone in transformers 5; the reference runs in a scratch venv with transformers
  4.50.1 (the version the config names).
- First text-only run differed from the first list item (ours `* **The Danube**`, theirs `1. **Rhine River**`): the
  remote code loads with SDPA, and transformers' Gemma 2 applies the attention-logit softcap only under eager
  attention. With `attn_implementation="eager"` the reference matches ours exactly; the softcap is the model's design,
  so the reference runs eager.
- Exact match on every case: text (271 chars), chat about four clips (46, 122, 160, 129), transcripts (56, 143, and the
  dialogue and 74 s clip, where the reference's 64-token cap stops early and its text is an exact prefix of ours).
  Prompt token counts agree (131, 231, 331).
- Transcripts carry speaker labels (`<Speaker1>: ... <Speaker2>: ...`); kept as the model writes them.
- The official GGUF needs its patched llama.cpp (`clip.projector_type meralion`, `mm.a.mlp.0..3`, `mm.a.norm_pre`)
  and a Gemma 2 text binding the multimodal GGUF path lacks: left for a follow-up.

**Next:** commit, checks, ask before pushing; then GGUF for MERaLiON-2, Qwen2-Audio (`qwen2a`), GLM-ASR (`glma`).
