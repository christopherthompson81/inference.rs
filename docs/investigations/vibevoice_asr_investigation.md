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
