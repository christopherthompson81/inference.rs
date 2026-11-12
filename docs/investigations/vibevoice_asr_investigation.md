# VibeVoice-ASR (#739)

`microsoft/VibeVoice-ASR-HF` (MIT, bf16, 8 shards, 16.5 GB): two speech-tokenizer encoders over raw 24 kHz audio
(acoustic, 64-d latents; semantic, 128-d), a two-branch projector summed into one Qwen2-width embedding per 3200
samples (7.5 Hz), and a Qwen2.5-7B decoder that answers with a JSON array of `{Start, End, Speaker, Content}`.
Reference: transformers 5.19's native `VibeVoiceAsrForConditionalGeneration` (the system python). Test audio: a
synthetic two-voice dialogue from Qwen3-TTS 1.7B CustomVoice (two preset voices, three turns, 12.7 s), and it repeated
to 64 s and 305 s.

## Run 1 - 2026-11-11 21:20

Question: what does the reference compute?

Read transformers' `models/vibevoice_asr` and `models/vibevoice_acoustic_tokenizer`, the repo's config, template and
processor config.

- Encoders: a causal conv stem (32 filters, kernel 7), six strided causal convs (ratios 2, 2, 4, 5, 5, 8; kernel twice
  the stride; channels doubling to 2048), ConvNeXt-1d stages (depths 3 x6 then 8: RMSNorm over channels, depthwise
  causal conv, a GELU FFN of 4x, each branch scaled by a learned gamma), a head conv to the latent width. Causal means
  left padding of `kernel - stride`.
- The model encodes in chunks of 1,440,000 samples (60 s), each conv carrying its last `kernel - stride` inputs into
  the next chunk: one causal pass over the whole clip, cut where memory needs.
- Projector: per branch Linear, RMSNorm (1e-6), Linear to 3584; the two summed.
- The acoustic latents get `vae_std * randn(1) * randn_like(latents)` added at inference, `vae_std` 0.625.
- Feature extractor: scale to -25 dB FS RMS, then to a peak of 1 if over it; pad to a multiple of 3200.
- Prompt (the repo's template): system "You are a helpful assistant that transcribes audio input into text output in
  JSON format.", then the user turn: `<|object_ref_start|>`, one `<|box_start|>` per hop (rounded up),
  `<|object_ref_end|>`, "This is a {seconds:.2f} seconds audio, please transcribe it with these keys: Start time, End
  time, Speaker ID, Content". The template ends with the user turn's `<|im_end|>\n`: the model writes its own
  `<|im_start|>assistant\n`.

## Run 2 - 2026-11-11 21:50

Question: do ours match in F32?

Reference on the CPU with `vae_std` 0; ours: `examples/vibevoice_audio_parity.rs` for the audio half, then `inference
serve --cpu --dtype f32` for the whole model.

```
reference: 156 prompt tokens; 109 generated in 111 s:
  <|im_start|>assistant\n[{"Start":0,"End":2.62,"Speaker":0,"Content":"Did you finish the report ..."},
  {"Start":2.83,"End":7.8,"Speaker":1,...},{"Start":8.03,"End":12.7,"Speaker":0,...}]<|im_end|>\n<|endoftext|>
ours, audio: clip max diff 2.98e-8; embeddings 344064 values, cosine 1.000000
ours, chat: 156 prompt tokens; the same text; 107 generated (stops at <|im_end|>, before the reference's last two)
ours, /v1/audio/transcriptions verbose_json with diarization_model=default: the same three segments, times and speakers
```

## Run 3 - 2026-11-11 22:00

Question: the default setup (bf16 on the 3090), and longer audio?

```
12.7 s: 2.4 s; the same segments, times within 0.02 s (2.81 for 2.83, 7.81 for 7.8, 8.01 for 8.03)
64 s: CUDA out of memory
```

The weights take 16.5 GB and the KV pool the rest; a 60 s encoder chunk needs about 2 GB of activations (1.44M
samples through the stem's 128-wide FFN, several live at once). Since the carried context makes any whole number of
hops give the same output, ours encodes 10 s at a time (75 hops): the 12.7 s clip, now two chunks, still gives cosine
1.000000.

```
64 s: 10.2 s, 11 segments, every turn with its voice's speaker
305 s: 48.8 s, 49 segments, speakers 0 and 1 only, every line of the second voice speaker 1
```

## Run 4 - 2026-11-11 22:10

Question: what does leaving out the reference's latent noise change?

The reference in bf16 on CUDA over the 64 s clip, `vae_std` 0 against 0.625 with seeds 0 to 2:

```
seed 0, 1, 2: 11 segments each, the same text and speakers as without noise; starts shifted by at most 0.02 s
```

The noise moves only times, by a frame or two of rounding; ours leaves it out, so a transcript repeats exactly.

Not done: the streaming checkpoints (`VibeVoiceForASRStreamingTraining`, 1.5B and 7B), hotword context in the
prompt ("with extra info: ..."), multi-pass decoding of recordings past the KV budget, and the third-party GGUF.

## Run 5 - 2026-11-11 22:40

Question: what do the streaming checkpoints (`microsoft/VibeVoice-ASR-Streaming-1.5B` and `-7B`) compute?

They declare `VibeVoiceForASRStreamingTraining` (`model_type` `vibevoice`, Microsoft's layout, which transformers does
not have). Reference: Microsoft's `microsoft/VibeVoice` repo (`modeling_vibevoice_asr.py`, `streaming_generate`), in a
venv over the system torch with transformers 4.57.3 (`/mnt/data/vibevoice-venv`).

- The same encoders and projector under other names: `model.{acoustic,semantic}_tokenizer.encoder.downsample_layers.0`
  is the stem conv and `.1..6` the strided ones, `stages.N.M` the ConvNeXt blocks (their mixer one `.conv` deeper),
  `head.conv.conv`; `model.{acoustic,semantic}_connector.{fc1,norm,fc2}`; the decoder at `model.language_model` with
  `lm_head` at the root (the 1.5B ties it). The config lists the encoder ratios last-first ([8, 5, 5, 4, 2, 2]) and
  its depths as "3-3-3-3-3-3-8"; kernel 7, FFN 4x and exact GELU are fixed in the code.
- `preprocessor_config.json`: `normalize_audio` false (no loudness scaling, unlike the offline model), windows of 22
  frames (70,400 samples, 2.93 s) and 4 of lookahead (0.53 s), each window encoded alone (no context carried), the last
  zero-padded to the whole window.
- Loop: the prompt "You are a helpful assistant that transcribes audio input into text output. Please transcribe the
  following audios streamingly with these keys: speaker, content\n" (31 tokens, no chat template) prefilled; per window
  `<|object_ref_start|>`, its 26 embeddings, `<|object_ref_end|>`, greedy to `<|text_chunk_end|>` or EOS (256 at
  most), then `<|text_chunk_end|>` fed so the cache ends on it; the cache runs the whole recording.
- Text per window, a turn opening with `\n Speaker N:`; 1.5B on the dialogue, f32 CPU, noise off:
  `[1/5] ' \n Speaker 0:Did you finish ...'`, `[2/5] ' \n Speaker 1:Almost. I just need to add the budget '`,
  `[3/5] 'numbers from last quarter. \n Speaker 0:Great. '`, `[4/5] 'Send it to me ... I will read '`, `[5/5] 'it
  tonight.'`.

## Run 6 - 2026-11-11 22:55

Question: do ours (`streaming.rs`: the offline audio half under the upstream names, the Qwen2 stack through
`DecoderStack::forward_cached` with its own KV cache) match?

`examples/vibevoice_streaming_parity.rs` against the reference's per-window token ids:

```
first try: every window "4.5 million." (ids 0 of 15 equal)
  --parts: prompt ids equal (31); window 0 embeddings cosine 1.000000
```

The audio and prompt were exact, so the decoder: `forward_cached` (new, moved from Qwen3-TTS's talker) built its mask
only for the first call, as a prompt chunk; every later call with several tokens (a window's 28) attended to its own
future. The talker never fed more than one token after its first call, so it never showed. The mask is now built at
any offset.

```
1.5B, f32 CPU: windows 15/15, 15/15, 13/13, 13/13, 3/3 ids equal (26 s)
1.5B, bf16 CUDA: the same (0.73 s); f16 CUDA: the same (0.76 s)
7B, f32 CPU: 15/15, 15/15, 13/13, 14/14, 3/3 (86 s); bf16 CUDA: the same (1.7 s)
```

Exact, both sizes. bf16 is kept on CUDA (the release dtype; f16, though equal here, is unmeasured on long Qwen2 runs).

## Run 7 - 2026-11-11 23:05

Question: served, whole and live?

`inference serve -m VibeVoice-ASR-Streaming-1.5B` (CUDA, bf16):

```
dialogue, /v1/audio/transcriptions verbose_json: 0.6 s; turns 0-2.93, 2.93-8.80, 5.87-12.70 (window times);
  with diarization_model=default, speakers 0, 1, 0
305 s: 12.9 s, 49 turns, speakers 0 and 1, every line of the second voice speaker 1; 7.0 GB on the GPU
/v1/realtime?intent=transcription, the dialogue pushed in 0.2 s pieces at real time: a delta per window at 3.8,
  6.7, 9.9, 13.1 s and the last on commit; the completed transcript the same
```

A turn's times are its windows': it starts at the start of the window its mark is in and ends with the last window
it reaches into, so adjacent turns can overlap by up to a window.

## Run 8 - 2026-11-11 23:15

Question: does the reference's latent noise (`fix_std` 0.5 on the streaming checkpoints) matter?

The reference over the 64 s clip, 1.5B, f32 CPU: noise off, then seeds 0 to 2 with it on.

```
22 windows each; every seed's windows identical to the run without noise
```

Ours leaves it out here as offline.

## Run 9 - 2026-11-11 23:50

Question: hotwords (the request's `prompt`) and recordings past the cache budget?

Context: both models take it in their prompt as Microsoft's `context_info`. Offline, transformers'
`apply_transcription_request(prompt=...)` renders "This is a 2.00 seconds audio, with extra info: <context>\n\nPlease
transcribe it with these keys: ..." (capital P), as ours now does; streaming, "... keys: speaker, content and extra
info: <context>\n". `scripts/vibevoice_asr_parity.sh --streaming --context "budget, quarterly numbers"` on the 1.5B:

```
windows 15/15, 15/15, 13/13, 14/14, 3/3 ids equal (window 3 is 14 with the context, 13 without)
```

Passes: past `with_pass_positions` (the decoder's maximum by default: 65,536 positions on the 1.5B, about 74 min at
~44 per window; 131,072 on the 7B) a window starts a new pass, its cache fresh from the prompt, and its speakers are
numbered after every earlier pass's. 1.5B on the 305 s clip at 1,500 positions:

```
4 passes (new at 85, 170, 255 s); speakers 0-1, 2-3, 4, 5-6; 51 turns in 12.6 s
the third pass, starting cold mid-conversation, labeled both voices 4 until its second window pair
```

A cold pass loses who is who, as expected; one pass over the same clip kept both voices apart (Run 7), so the default
stays the whole context.

Not done: GGUF (the audio.cpp container, #747).
