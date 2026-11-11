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
