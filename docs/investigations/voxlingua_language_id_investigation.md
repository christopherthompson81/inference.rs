# VoxLingua107 language identification (#745)

SpeechBrain's `speechbrain/lang-id-voxlingua107-ecapa`: a 60-mel Fbank with sentence mean normalization, an ECAPA-TDNN
(channels 1024 x4 then 3072, attention 128, embedding 256) and `Xvector.Classifier` (LeakyReLU, BatchNorm, one
512-wide linear block with LeakyReLU and BatchNorm, then 107 logits and log-softmax). Reference: SpeechBrain 1.1.2 in
`/mnt/data/speechbrain-venv` (over the system torch 2.11+cu130), checkpoint in
`/mnt/data/models/lang-id-voxlingua107-ecapa`.

## Run 1 - 2026-11-11 16:30

Question: what exactly do the checkpoint and SpeechBrain compute, and how much of the Qwen3-TTS speaker encoder
(`qwen3_tts/speaker.rs`, an ECAPA-TDNN ported from SpeechBrain) carries over?

Read `speechbrain/lobes/models/ECAPA_TDNN.py`, `Xvector.py`, `lobes/features.py`, `processing/features.py`,
`nnet/CNN.py` and the checkpoint's keys (`torch.load` of the three `.ckpt` files).

Findings:
- The classifier is `Xvector.Classifier`, not ECAPA's cosine `Classifier` the issue guessed: `act` (LeakyReLU 0.01) on
  the embedding, `norm`, `DNN.block_0.{linear,act,norm}`, `out` (107 with bias), `softmax` (log). Keys
  `norm.norm.*`, `DNN.block_0.linear.w.*`, `DNN.block_0.norm.norm.*`, `out.w.*`.
- TDNN blocks are conv, ReLU, then BatchNorm: the norm cannot fold into the conv (the ReLU sits between). Applied as a
  per-channel scale and shift; `asp_bn` sits between pooling and `fc`.
- SpeechBrain's `Conv1d` wraps the torch conv: `blocks.0.conv.conv.weight` where Qwen3-TTS's port has
  `blocks.0.conv.weight`; norms are `<block>.norm.norm.*`, `asp_bn.norm.*`. Otherwise the graph is the same as the
  Qwen3-TTS encoder (reflect 'same' padding, Res2Net chunking, SE mean over time, global-context attentive pooling
  with the 1e-12 variance clamp). One backbone (`ecapa.rs`) with a layout switch serves both.
- Fbank: `n_fft` 400, 25 ms (400) Hamming window (torch's periodic), hop 160, `center=True` with zero padding,
  power spectrum (`spectral_magnitude` power 1 is |X|^2), HTK mel points `linspace(mel(0), mel(8000), n_mels + 2)`
  with triangles whose two sides both use the left band (`band = hz[1:] - hz[:-1]`, `band[:-1]`), `10 log10(max(x,
  1e-10))`, then a floor of the utterance's max minus 80 dB. `InputNormalization(norm_type=sentence,
  std_norm=false)` subtracts the per-mel mean over time. No deltas, no context.
- `normalizer.ckpt` is a global normalizer the hyperparams never use (`mean_var_norm` is sentence-level).
- `hyperparams.yaml` uses `!new:` and `!ref` tags; serde-saphyr reads the needed fields through them (unit test).

Next: write `ecapa.rs` shared by both models, the VoxLingua module, and a parity script.

## Run 2 - 2026-11-11 16:44

Question: does our VoxLingua107 match SpeechBrain on the checkpoint's own sample (a public Thai reading, 44.1 kHz,
13 s), resampled once to 16 kHz by torchaudio and fed to both?

`scripts/voxlingua_parity.sh /mnt/data/models/lang-id-voxlingua107-ecapa /mnt/data/speechbrain-venv/bin/python
/mnt/data/models/lang-id-voxlingua107-ecapa/udhr_th.wav` (reference on CUDA with TF32 off; ours on CUDA, f32)

```
reference: [('th: Thai', 0.9888), ('lo: Lao', 0.0112), ('la: Latin', 0.0)]
fbank: 78000 vs 78000 values, max diff 1.24e-3
embedding: cosine 1.000000
posterior: max diff 2.14e-7; ours th 0.9888, reference th (0.09 s for 13.0 s)
```

Exact on the first run. The fbank's 1.2e-3 is in dB (values span tens of dB): f32 FFT and mel sums.

Next: more languages, and the CPU.

## Run 3 - 2026-11-11 16:46

Question: does parity hold across languages and on the CPU?

Six synthetic espeak-ng clips (en, de, fr, es, ru, hi; one weather sentence each, about 5 s), same script; then the
Thai clip with `VOXLINGUA_CPU=1`.

```
de: reference cy 0.3937 / pl / nl   embedding cosine 1.000000, posterior max diff 7.81e-6
en: reference cy 0.8033 / sv / vi   cosine 1.000000, max diff 1.97e-6
es: reference la 0.9667 / ia / ca   cosine 1.000000, max diff 1.79e-7
fr: reference fr 0.9686 / zh / lb   cosine 1.000000, max diff 4.77e-7
hi: reference nl 0.1940 / cs / cy   cosine 1.000000, max diff 8.27e-7
ru: reference pl 0.4827 / war / lb  cosine 1.000000, max diff 1.97e-6
th (CPU): cosine 1.000000, posterior max diff 2.98e-7, 0.39 s for 13 s
```

Our top-1 equals the reference's on every clip. The model itself misreads espeak's rule-based voices (English as Welsh,
Spanish as Latin), with low top-1 probabilities on most: rule-based synthetic speech is far from the corpus. Not a parity
question, but it shows why the response should carry an ambiguity mark when top-1 is low. Natural speech checks need
a neural TTS (Run 4).

## Run 4 - 2026-11-11 17:03

Question: does the shared ECAPA-TDNN (`ecapa.rs`, now under the Qwen3-TTS speaker encoder too) leave Qwen3-TTS Base
cloning as it was, and how does language ID do on natural-sounding speech?

`scripts/qwen3_tts_parity.sh /mnt/data/models/Qwen3-TTS-12Hz-0.6B-Base /mnt/data/qwentts-venv/bin/python <a
Qwen3-TTS-generated clip>` (x-vector mode):

```
x-vector: cosine 1.000000 to the reference's
codec on reference codes: 1150080 vs 1150080 samples, SNR 79.9 dB
greedy: 649 frames ours, 599 reference; first 599 frames equal
```

Unchanged from before the move. Then eight clips from Qwen3-TTS 1.7B CustomVoice (one speaker, the same weather
sentence in each language, 5 to 10 s) through `scripts/voxlingua_parity.sh`:

```
de 0.9882   en 0.9998   es 0.9852   fr 0.9925   it 1.0000   ja 1.0000   ru 0.8724 (be 0.1276)   zh 0.9994
embedding cosine 1.000000 on all eight; posterior max diff at most 1.92e-6
```

All eight right, and exact against SpeechBrain. Neural TTS reads as its language where espeak's rule-based voices did
not (Run 3).

## Run 5 - 2026-11-11 17:09

Question: does it work served, alone and as a transcription's `language_model`, with the real checkpoint?

A scratch server (`inference from-config`, port 1299) with the VoxLingua107 directory (auto-detected from its
`config.json` and files), `openai/whisper-small` and `openai/whisper-base.en`; the Run 4 clips.

```
POST /v1/audio/language (de, top_k 3): de 0.988, iw 0.006, yi 0.005; ambiguous false; duration 6.15
  return_embedding: 256 values
POST /v1/audio/transcriptions model=whisper-small language_model=<lid>, verbose_json:
  de -> language "german", the German sentence (one word off)
  fr -> language "french", the French sentence      ja -> language "japanese", the Japanese sentence
  whisper-base.en with language_model on the German clip: transcribes as English (the identified `de` is not a
  language it takes, so it runs as unset) instead of failing
  language_model naming a Whisper model: 400, param language_model, code language_id_failed
```

Whisper reports its own language name (`german`), which the response prefers over the identified code, as it already
did for a requested language. Nothing left that changes the design.
