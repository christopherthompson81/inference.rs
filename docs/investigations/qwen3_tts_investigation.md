# Qwen3-TTS investigation

Porting Qwen3-TTS (12 Hz, issue #561): the talker (a Qwen3 decoder over codec codebook 0), the code predictor (a small
Qwen3 decoder filling codebooks 1-15 per frame), the 12 Hz codec decoder, and for the Base checkpoints the speaker
encoder. Reference: the `qwen-tts` 0.1.1 package (transformers 4.57.3), run from a venv at `/mnt/data/qwentts-venv` that
reuses the system torch.

## Run 1 - 2026-11-11 12:50

Question: what does the reference compute, and can the talker and code predictor reuse the shared decoder?

Read `qwen_tts/core/models/modeling_qwen3_tts.py` and `tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`.

- Talker: Qwen3 layers (q/k norm before RoPE), interleaved M-RoPE sections [24, 20, 20]. For one unpadded sequence the
  three position rows are all `cumsum(mask) - 1`, so the interleaved tables equal plain 1-D RoPE: a standard Qwen3 stack.
  Its embedding is `talker.model.codec_embedding` (3072 rows), its head `talker.codec_head`; the text track is a separate
  `text_embedding` (151936 x 2048) through `text_projection` (fc1 2048->2048 + SiLU + fc2 2048->hidden, with biases).
- Prompt (CustomVoice, non-streaming, the default for `generate_custom_voice`): text_proj(`<|im_start|>assistant\n`);
  then codec [think, think_bos, language, think_eos, speaker, pad] each plus tts_pad, except the last of them plus
  tts_bos; then text_proj(text tokens) and tts_eos, each plus codec_pad; then tts_pad + codec_bos. Every decode step adds
  tts_pad to the frame's summed embeddings. Language `auto` swaps the first four for [nothink, think_bos, think_eos].
  Speakers with a dialect (eric, dylan) switch the language id to the dialect's when the language is Chinese or auto.
- Decode step: the code predictor runs on [talker's last normed hidden, codec_embedding(code0)] then 14 more steps, step
  k reading `code_predictor.model.codec_embedding[k-1]` and scoring with `lm_head[k]`. The talker's next input is the sum
  of all 16 codes' embeddings (code0 through the talker's table, codes 1-15 through the predictor's) plus tts_pad.
  `small_to_mtp_projection` maps talker width to predictor width (identity at 0.6B, a linear at 1.7B), applied to
  every predictor input, so the predictor's tables are in the talker's width.
- Sampling defaults (generation_config.json): talker top-k 50, temperature 0.9, repetition penalty 1.05 over generated
  code0s, tokens 2048..3071 suppressed except EOS 2150, EOS suppressed for the first 2 steps; predictor top-k 50,
  temperature 0.9, no penalty.
- Codec decoder: split RVQ (1 semantic + 15 acoustic codebooks of 2048 x 256, `embedding_sum / cluster_usage`, 1x1 conv
  out 256->512, summed), causal conv k3 512->1024, transformer (input_proj 1024->512, 8 layers: RMS, attention with no
  q/k norm, RoPE theta 1e4, sliding window 72, layer scale on both residuals, SiLU MLP 1024; norm, output_proj ->1024),
  two (causal transposed conv x2, ConvNeXt) upsamplers, causal conv k7 1024->1536, four blocks (SnakeBeta, causal
  transposed conv x8/x5/x4/x3 halving channels, residual units at dilations 1/3/9), SnakeBeta, conv k7 ->1, clamp to
  [-1, 1]. 1920 samples per frame at 24 kHz; decoded in chunks of 300 frames with 25 frames of left context.

Implication: talker = `CausalLm` from a Qwen3 `DecoderSpec` (embed `codec_embedding`, head `codec_head`). The code
predictor is a `DecoderStack` whose inputs are always given as embeddings; its tables are in the talker's width (2048 at
1.7B), so the stack must not load a hidden-width token embedding of its own. The codec transformer has layer scale,
which `DecoderSpec` does not have: a small module of its own.

## Run 2 - 2026-11-11 12:56

Command: `qwentts-venv/bin/python -P qwentts_ref.py Qwen3-TTS-12Hz-0.6B-CustomVoice <out>` (fp32 on CUDA, speaker Ryan,
English, greedy talker and predictor, repetition penalty 1.0, a two-sentence public-domain-style English text).

Question: is greedy decoding deterministic and finite, for a parity target?

Finding: 87 frames (codes `[87, 16]`), EOS reached; wav 167040 samples = 87 x 1920 at 24 kHz. Saved prompt ids, codes and
wav to `greedy.safetensors` under `/mnt/data/models/.profile-tmp/qwen3tts-ref`.

Implication: greedy codes are the parity target for the talker + predictor; the codec decoder is checked on those codes
against the reference wav.

## Run 3 - 2026-11-11 13:08

Command: `target/debug/examples/qwen3_tts_parity --model Qwen3-TTS-12Hz-0.6B-CustomVoice --dump greedy.safetensors
--text <run 2's text> --voice ryan --language english [--bf16 | --cpu]` (CUDA build).

Question: do the prompt, the talker + predictor and the codec decoder match the reference?

Finding:
- prompt ids: 27 = 27, equal (the tokenizer built from `vocab.json` + `merges.txt` + `added_tokens_decoder`).
- codec on the reference's codes: 167040 = 167040 samples, SNR 64.5 dB (f32 everywhere; the codec always runs f32).
- greedy, f32 on CUDA: 87 frames, all 87 x 16 codes equal the reference, 2.0 s.
- greedy, bf16 on CUDA: first 3 frames equal, first 7 first codes; then it runs 137 frames (no EOS until later).
- greedy, f32 on CPU: first 28 frames equal, 95 frames, 26.2 s (dev build with debug assertions).

Implication: the port is exact in f32; greedy decoding is chaotic under any change of arithmetic, so bf16 is judged on
sampled output, not on codes.

## Run 4 - 2026-11-11 13:12

Command: the same with `--bf16 --seed 1 --sampled-wav sampled_bf16_s1.wav` (default sampling), then
`parakeet_transcribe --model parakeet-tdt-0.6b-v3` on each wav.

Question: is bf16 speech at the default sampling intelligible?

Finding:
- reference greedy wav: "The quick brown fox jumps over the lazy dog. It was a bright cold day in April."
- our f32 greedy: identical transcript.
- our bf16 greedy: 11.0 s, only "The quick brown fox jumps over the lazy dog." (the rest is not words).
- our bf16 sampled, seed 1: 6.56 s of audio in 1.08 s (6x real time), "The quick brown fox jumps over the lazy dog. It
  was a bright, cold day in April."

Implication: bf16 is fine for the default (sampled) decoding; greedy is not the model's mode (the reference defaults to
sampling too). Next: the engine loader, then the 1.7B CustomVoice and VoiceDesign checkpoints (instructions).

## Run 5 - 2026-11-11 13:18

Command: `qwentts_ref.py --model Qwen3-TTS-12Hz-1.7B-CustomVoice --speaker Vivian --language English --instruct "Speak
slowly, in a warm and cheerful tone."` (greedy, fp32), then the parity example on the same; and a bf16 sampled run
(seed 3) transcribed with Parakeet.

Question: does the 1.7B layout (talker 2048 wide, predictor 1024 through `small_to_mtp_projection`) and the instruction
prompt match?

Finding: prompt ids equal; the reference's greedy decode never reached EOS (599 frames at the 600 cap); ours equals it
on all 599 x 16 codes. The codec over 599 frames (two chunks with 25 frames of left context) reaches 65.5 dB. The bf16
sampled run: 6.40 s in 1.27 s, transcript exact.

Implication: CustomVoice is done at both sizes; VoiceDesign shares the instruction path (no speaker id).

## Run 6 - 2026-11-11 13:27

Command: `qwentts_ref.py --model Qwen3-TTS-12Hz-1.7B-VoiceDesign --language English --instruct "A deep, gravelly male
voice, an old sea captain telling a story."` and the parity example with the same; a bf16 sampled run (seed 5).

Question: does VoiceDesign (instruction, no speaker id) match?

Finding: prompt ids equal; greedy 599 x 16 codes equal (the reference again ran to the 600 cap); codec 53.1 dB on its
codes. bf16 sampled: 7.76 s in 1.65 s, Parakeet transcript exact.

Implication: the three supported variants (0.6B/1.7B CustomVoice, 1.7B VoiceDesign) are exact in f32. Base (voice
cloning) needs the speaker encoder and, for in-context cloning, the codec's encoder: not in this change.

Tiny engine test (`crates/inference/tests/integration/qwen3_tts_tiny.rs`): voices, seeds, languages, instructions,
voice design, detection and request errors, 5 tests in under 0.1 s each on CPU.

## Run 7 - 2026-11-11 13:58

Question: what does voice cloning (Base) add, and does it match?

Read `generate_voice_clone`, `create_voice_clone_prompt`, `generate_icl_prompt`, the speaker encoder and transformers'
`MimiModel.encode`:
- x-vector: an ECAPA-TDNN (TDNN 128->512 k5, three SE-Res2Net blocks k3 at dilations 2/3/4, scale 8, aggregation over
  their outputs 1536 k1, attentive statistics pooling, 3072->enc_dim; every conv 'same' with reflect padding) over a
  log-mel (n_fft 1024, hop 256, 128 Slaney mels to 12 kHz, reflect padding (n_fft - hop) / 2 each side, uncentered,
  `log(clamp(mel, 1e-5))`). enc_dim is the talker width (1024 at 0.6B, 2048 at 1.7B). It takes the speaker id's place
  in the codec tags.
- In-context (a transcript given): the clip's codes from the codec's Mimi encoder (causal SEANet, 8-layer transformer,
  replicate-padded stride-2 downsampler, split RVQ keeping 16 of 32 codebooks, nearest centroid per residual), the
  clip's text before ours, bos then the clip's frames beside them; text past the frames trails one row per decoded
  frame. The output is decoded after the clip's frames and cut at `ref_frames / all_frames` of the samples.
- Cloning defaults to streaming text (`non_streaming_mode=False`): x-vector mode puts only the first text token beside
  bos, the rest trail. CustomVoice and VoiceDesign default to non-streaming.
- Mimi's encoder transformer gets a full causal mask: transformers' `create_causal_mask` ignores the config's
  `sliding_window` (250), so the reference never windows it.

Command: `qwentts_clone_ref.py --model Qwen3-TTS-12Hz-0.6B-Base --ref-audio greedy.wav [--ref-text <run 2's text>]`
(the clip is run 2's synthetic Ryan output), then the parity example with `--reference-audio [--reference-text]`.

Finding (first pass): x-vector mode exact (cosine 1.000000, 399 greedy frames equal; the reference ran to the 400 cap),
codec SNR only 33.7 dB. In-context: clip codes 81 of 87 frames equal, the six others first differing at codebooks
3-14; greedy equal for 15 frames, every first code equal.

## Run 8 - 2026-11-11 14:05

Question: are the six clip frames ours or the reference's numerics?

Command: the reference tokenizer's `encode` of the same clip on CPU, CUDA with TF32 off, and CUDA with TF32 on (the
default for convolutions: `torch.backends.cudnn.allow_tf32 = True`).

Finding: CPU 81 and CUDA without TF32 81 frames equal to the dump, CUDA with TF32 85 (TF32 is not even
deterministic run to run). Ours against the CPU reference: 87 of 87, on CUDA and on CPU.

Implication: the dumps must run with TF32 off. Regenerated: 0.6B in-context clip codes 87/87, greedy 399/399, codec
97.2 dB; x-vector 399/399, codec 79.8 dB (the 33.7 dB was TF32 in the reference's codec convs too). 1.7B Base
in-context: x-vector cosine 1.000000, clip 87/87, greedy 399/399, codec 84.5 dB.

## Run 9 - 2026-11-11 14:15

Command: the parity example with `--bf16 --seed 4 --sampled-wav ... --similar-to greedy.wav cv17_vivian.wav` for
0.6B and 1.7B Base in both modes, then Parakeet.

Question: does sampled bf16 cloning speak the text in the clip's voice?

Finding (x-vector cosine of the output to the clip's voice / to another voice, transcript):

| model | mode | audio / time | to clip | to other | transcript |
|---|---|---|---|---|---|
| 0.6B | x-vector | 3.68 s / 0.65 s | 0.990 | 0.929 | exact |
| 0.6B | in-context | 4.16 s / 0.84 s | 0.993 | 0.931 | exact |
| 1.7B | x-vector | 4.48 s / 0.93 s | 0.992 | 0.946 | exact |
| 1.7B | in-context | 4.40 s / 1.03 s | 0.991 | 0.945 | exact |

Implication: the clone is nearer the clip's voice than another in every case (the encoder's own space compresses the
gap). Next: the request fields and engine tests.

## Run 10 - 2026-11-11 14:40

Question: do llama.cpp's GGUFs (ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF: talker `qwen3tts` + mmproj with
`qwen3tts_spkenc` and `qwen3tts_gen`) load, and how close are they?

Read `conversion/qwen3tts.py` (llama.cpp 970b51ce7): the talker's text projection is folded into `token_embd` (text
rows) with the codec embedding appended (codec rows, named `<|codec_*|>` after their config keys, so the codec ids,
languages and voices come back from the token names; which voices speak a dialect does not); SnakeBeta is stored as
`e^alpha`, `1 / (e^beta + 1e-9)`; codebooks are stored divided by their usage; the code predictor's 15 tables and heads
are stacked; the codec transformer's heads, head size, RoPE base and window are hardcoded, not stored. No codec encoder
is converted, so a GGUF cannot clone in-context. The bindings undo the folds (log; -log; usage of ones) except the text
projection, which the talker now skips when the checkpoint has none.

Command: the parity example with `--model Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf` (f32 compute) against
`base17_xvec.safetensors` (the release in f32: x-vector cosine 1.000000, codec 116.4 dB, greedy 399/399).

Finding (GGUF bf16 file): prompt ids equal (tokenizer rebuilt from the GGUF's tokens and merges), x-vector cosine
1.000000, codec 48.3 dB (the GGUF stores the codec's matrices in BF16, the release in F32), greedy 48 frames equal,
71 first codes, then EOS at 181 frames (the folded text table is rounded to BF16 after folding).
bf16 sampled, seed 4: bf16 GGUF 4.24 s in 0.90 s, Q8_0 4.80 s in 0.85 s; both transcribe exactly and are nearer the
clip's voice (0.991) than another (0.943-0.944).

Implication: GGUF loading is right to within the file's own rounding; it is a BF16-class checkpoint.

## Run 11 - 2026-11-11 14:55

Commands: `scripts/qwen3_tts_parity.sh` (now with TF32 off in the reference, and Base clips) on 0.6B Base in-context and
0.6B CustomVoice; the tiny engine tests (`qwen3_tts_tiny`); `inference serve` with the 1.7B Base Q8_0 GGUF and with the
1.7B Base release, `/v1/audio/speech` with a base64 `reference_audio` (run 2's synthetic clip).

Finding: Base in-context x-vector 1.000000, clip 87/87, greedy 599/599, codec 94.3 dB; CustomVoice greedy 87/87, codec
114.7 dB (the 64.5 dB of run 3 was the reference's TF32). The tiny GGUF, written as llama.cpp's converter writes one,
speaks within 2 PCM16 steps of its checkpoint (the SnakeBeta log/exp round trip). Over HTTP: the Q8_0 GGUF clones
(200, transcript exact) and answers `reference_text` with a 400 naming the missing codec encoder; the release continues
the clip (200, transcript exact).
