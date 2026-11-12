# Cohere Transcribe (#738)

`CohereLabs/cohere-transcribe-03-2026` (gated, Apache 2.0, 2B, bf16 safetensors): a 48-layer FastConformer (d 1280,
8 heads, dw_striding subsampling by 8, rel_pos attention, untied position biases) projected 1280 -> 1024 into an
8-layer pre-LN NeMo transformer decoder (ReLU FFN 4096, a stored sinusoidal position table, a LayerNorm over the
embedding) and a 16384-way head with bias. Reference: transformers 5.19's native `CohereAsrForConditionalGeneration`
(the system python), checkpoint in `/mnt/data/models/cohere-transcribe-03-2026`.

## Run 1 - 2026-11-11 18:25

Question: what layout does the checkpoint use, and how much of Parakeet and Whisper carries over?

Read the repo's `config.json`, the safetensors header and transformers' `models/cohere_asr/` (modeling, feature
extractor, processor).

Findings:
- Tensor names are NeMo's (`encoder.layers.N.self_attn.linear_q`, `encoder.pre_encode.conv.N`,
  `transf_decoder._decoder.layers.N.first_sub_layer.query_net`, ...). The config's `encoder:` and `preprocessor:`
  sections are exactly NeMo's, so `NemoEncoderConfig::parakeet()` and `parakeet_encoder_name` load the encoder
  through Parakeet's `Encoder` unchanged, and `NemoPreprocessorConfig::mel()` gives the frontend (`NemoMel`).
- The decoder is Whisper's shape (pre-norm self-attention, cross-attention, two-layer FFN, final LayerNorm) with
  different names, a key bias, ReLU, an embedding LayerNorm and an untied head: Whisper's decoder becomes the shared
  `text_decoder.rs`, configured by a names table.
- The prompt is `U+2581 <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|xx|> <|xx|> <|pnc|> <|noitn|>
  <|notimestamp|> <|nodiarize|>` (ids `13764 7 4 16 62 62 5 9 11 13` for English), then greedy to `<|endoftext|>`
  (3), 256 new tokens at most.
- The model does not identify the language: the reference's `transcribe()` requires one of its 14 codes.
- Long audio: past 35 s (`max_audio_clip_s`), chunks of at most 35 s, each ending at the quietest 1600-sample window
  (RMS) of its last 5 s (`overlap_chunk_second` is a search window, not an overlap); texts joined with a space, none
  for ja and zh.
- The feature extractor adds a 1e-5 dither from torch's generator seeded by the clip's length; off for parity.
- The stored `preprocessor.featurizer.fb` and `window` are bf16; the reference ignores them and computes f32 ones.

## Run 2 - 2026-11-11 18:33

Question: does ours match the reference in f32?

`scripts/cohere_asr_parity.sh /mnt/data/models/cohere-transcribe-03-2026 <lang> <clip>`: the repo's own demo clip
(English, 5.4 s, a public VoxPopuli sample), the Qwen3-TTS clips of the language ID work (de, fr, es, it, ja, zh;
one weather sentence each) and a 70 s English clip (the English sentence 9 times with 0.6 s gaps).

```
demo en: features 544 frames, max diff 1.50e-4; encoder cosine 1.000000; 16 of 16 ids equal
de 32/32, fr 37/37, es 28/28, it 31/31, ja 37/37, zh 33/33 ids equal; encoder cosine 1.000000 on each
long en: 3 chunks (0-30.6, 30.6-61.8, 61.8-70.2 s), 112/112, 112/112, 27/27 ids equal
de on the CPU: cosine 1.000000, 32/32 equal (2.6 s)
```

Exact, chunk splits included. The reference's encoder output is a frame longer (its features carry one masked
padding frame, and a batch pads to its longest chunk); the valid frames are the ones compared.

## Run 3 - 2026-11-11 18:40

Question: does reduced precision hold (the pipeline's `auto` is f16 on CUDA)?

The same script with `COHERE_DTYPE=bf16` and `f16`, on de, ja and the long English clip:

```
bf16: encoder cosine 0.99993-0.99995; every chunk's ids equal
f16:  encoder cosine 0.999996-0.999999; every chunk's ids equal
```

Both decode the reference's ids exactly; f16 keeps the encoder closer.
