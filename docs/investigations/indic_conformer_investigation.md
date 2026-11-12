# IndicConformer 600M (#740)

AI4Bharat's 22-language hybrid RNN-T/CTC FastConformer. Weights from `ai4bharat/IndicConformer` (`IndicConformer.nemo`,
2.5 GB F32, NeMo 1.23.0rc0 from AI4Bharat's fork); reference: the same model's ONNX export,
`ai4bharat/indic-conformer-600m-multilingual` (TorchScript preprocessor plus onnxruntime sessions, CPU), driven as its
`model_onnx.py` does. A CTC-only repackaging of that export, `christopherthompson81/indicconformer-600m-onnx` (made for
Vernacula), was a cross-check of the vocabulary layout. Test audio: one FLEURS test clip each for hi, ta, bn, mr and
ur (google/fleurs, CC-BY-4.0, public), fetched as Vernacula's `fetch_indic_test_audio.py` does.

## Run 1 - 2026-11-11 19:05

Question: which checkpoint loads, and what layout does it use?

- The multilingual HF repo ships ONNX only: an encoder whose external weights are mostly unnamed
  (`onnx__MatMul_8067` ...), per-language joint post-nets, `language_masks.json` and `vocab.json`. Not loadable here.
- `ai4bharat/IndicConformer.nemo` is the same model with named weights (24 layers, d 1024, 8 heads, dw_striding by 8,
  rel_pos, xscaling, 80 mels, `EncDecHybridRNNTCTCBPEModel`): `NemoEncoderConfig::parakeet()` and the existing
  renames load the encoder and predictor (2-layer LSTM 640, embedding 5633 rows).
- New: `joint.joint_net.2.<lang>` is one 257-wide output layer per language (256 tokens, local blank 256), keys in
  `joint.language_keys` order (as, bn, brx, doi, kok, gu, hi, ...; hi starts at 1536, matching `language_masks.json`
  and the Vernacula repack's `language_spans.json`). `ctc_decoder.decoder_layers.0` is the shared 5633-way CTC head.
- New: the tokenizer is `type: multilingual`, one 256-piece SentencePiece model per language.
- The reference RNN-T loop starts the predictor from SOS 5632 and feeds it the joint's local ids (0..255), and reads
  text from the language's own vocabulary.

## Run 2 - 2026-11-11 19:40

Question: does ours (multisoftmax joints in Parakeet's transducer) match the reference?

`scripts/indic_conformer_parity.sh IndicConformer.nemo indic-conformer-600m-multilingual hi hi.wav`:

```
features: 936 frames ours, 937 reference, max diff 4.22e-2
encoder: 117 frames ours, 118 reference, cosine 0.998976
rnnt: 38 ids ours, 38 reference, first 38 equal
ours: two short words of the 11 (transcribe())
```

Two findings:
- The ids already match, but `transcribe()` wrote almost nothing: one tokenizer over all 22 languages' pieces collapses
  the pieces the languages share (each has its own `<unk>`; Devanagari languages share many), so most ids decode to
  nothing. Fixed with one tokenizer per language over its local ids, as the reference reads text.
- The features differ at the edges and by a frame: the reference's TorchScript (AI4Bharat's fork, NeMo 1.x) pads the
  STFT by reflection and counts `len / hop + 1` frames valid; ours follows NeMo 2.x (zeros, `len / hop`). Not dither:
  two runs of the reference preprocessor are identical. NVIDIA's own 1.17 `.nemo` (the hybrid of the NeMo
  investigation, Run 20) matched NeMo 2.7's frontend, so the frontend follows the code that runs a checkpoint, not
  its version: the reflect frontend (`NemoMelConfig::reflect_pad`) is used for multisoftmax checkpoints only, which
  only AI4Bharat's fork makes.

## Run 3 - 2026-11-11 19:55

Question: exact now, across languages and on CUDA?

The same script, each FLEURS clip in its language (F32, CPU), and hi with `INDIC_CUDA=1`:

```
hi: features max diff 3.87e-5, encoder cosine 1.000000, 38/38 RNN-T ids equal
ta: 4.82e-5, 1.000000, 112/112      bn: 3.50e-5, 1.000000, 111/111
mr: 5.46e-5, 1.000000, 100/100      ur: 1.04e-4, 1.000000, 62/62
hi on CUDA: cosine 1.000000, 38/38 (0.05 s)
```

Every transcript equals the reference's RNN-T text. Against the FLEURS references they differ in punctuation (the model
writes none) and a few words (Tamil: 2 of 24; Hindi: both join "skiing route" into one word, where FLEURS writes two).

## Run 4 - 2026-11-11 20:23

Question: served, at the default dtype, with a language identifier naming the language?

A scratch server with the `.nemo` directory (auto-detected) and VoxLingua107; `language_model` set, `verbose_json` with
word timestamps, CUDA at the default dtype (f16):

```
hi, ta, bn, mr: each identified correctly; transcripts equal the F32 parity runs; 10, 24, 28, 28 timed words
ur: identified as hi; the right words, in Devanagari rather than Urdu script
no language: 400, "this model does not identify the language; name one of as, bn, ... (or a `language_model`)"
```

Spoken Urdu and Hindi are close enough that VoxLingua107 reads the Urdu clip as Hindi, so the transcript is right in
the wrong script; naming `language=ur` gives the Urdu one. A limit of identification, not of the recognizer.

Not done: the CTC head (the reference's default `decoding='ctc'`; RNN-T is the hybrid's primary head and what loads
here), and a GGUF.
