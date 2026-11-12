# audio.cpp GGUF container (#747)

audio.cpp publishes GGUFs for about a hundred audio models (`audio-cpp/audio.cpp-gguf`, one folder per model, plus a
few standalone repos). They are not llama.cpp GGUFs: `general.architecture = audiocpp`, the original checkpoint's
tensors under their own names and its other files embedded byte for byte. One reader serves every family whose
loader we have.

## Run 1 - 2026-11-12 00:20

Question: what exactly is in the container, as audio.cpp (a01e18c9) writes and reads it?

`src/framework/assets/tensor_source.cpp` (`convert_tensor_sources_to_gguf` and the GGUF tensor source), `docs/gguf.md`,
and llama.cpp's gguf-py reader over `VibeVoice-ASR-Streaming-7B` q4_k:

```
audiocpp.model_spec.family        vibevoice_asr_streaming
audiocpp.model_spec.json          the family's package spec (sources, required files)
audiocpp.weight_type              q4_k
audiocpp.tensor_name_format       native
audiocpp.tensor_sources.names     ['']   (paths ['model.safetensors.index.json'])
audiocpp.embedded_files.names     13 files: config.json, tokenizer files, preprocessor_config.json, README, figures/...
audiocpp.embedded_files.offsets   14 u64 (n + 1, into one blob)
audiocpp.embedded_files.data      12,435,097 u8
audiocpp.tensor_names             1177 logical names, one per GGUF tensor in file order
audiocpp.tensor_ranks / _shapes   1177 ranks, 1739 dims: each tensor's PyTorch shape (rank 0 for a scalar)
tensors: 857 BF16, 319 Q4_K, 1 F16; 198 stored as `_audiocpp.<i>`
lm_head.weight Q4_K [3584, 152064] (GGML ne order); model.acoustic_connector.fc1.weight BF16 [64, 3584]
```

- A tensor is stored under its own name unless that is 64 bytes or longer (`GGML_MAX_NAME`) or repeats; then it is
  `_audiocpp.<i>`, its name at index i of `tensor_names`.
- Data is GGML's layout (dims reversed, quant blocks along the last PyTorch dim). Quantized: rank-2 `.weight`s whose
  row is a multiple of the block, except embeddings and codebooks (names containing `embed` or `codebook`, stored F16);
  everything else keeps the source dtype (BF16 here).
- A rank-0 scalar is stored as one element; the exact shapes restore it.
- Several sources (Chatterbox's `ve`, `s3gen`, `t3_*`) prefix each name with `<source>/`.
- Our GGUF reader already leaves top-level arrays over 2^20 elements in the file (`Content::large_arrays`), so the
  12 MB blob is read only when unpacked.

Plan: open the file as a `GgufArchive`, bind each logical name to its tensor (a reshape where the recorded shape
differs) for `GgufWeightSource`, unpack the embedded files once into a cache directory, and let a loader take the
package where it takes a checkpoint directory.

## Run 2 - 2026-11-12 00:55

Question: does the reader serve VibeVoice-ASR Streaming from audio.cpp's GGUFs, and what do the quants cost?

`inference-gguf::audiocpp` (each logical name bound to its stored tensor, the embedded files unpacked under
`~/.cache/inference-rs/audiocpp/<stem>-<hash>/`); the auto loader and the transcription loader read an audio.cpp GGUF
(or a directory holding one) as the checkpoint directory it unpacks to. `inference serve -m <file>.gguf`, CUDA,
`/v1/audio/transcriptions` verbose_json with `diarization_model=default`, the 12.7 s dialogue and the 305 s clip, each
against the same checkpoint's bf16 safetensors. GPU memory is the card's total less the 1.1 GB it holds idle.

```
7B   bf16 safetensors  17.0 GB  305 s in 35.4 s
7B   q4_k (audio.cpp)   7.2 GB  305 s in 17.7 s  864/864 words, 1 edit ("Great." -> "Great!" once); 49/49 turns, speakers equal
1.5B bf16 safetensors   5.3 GB  305 s in 13.0 s
1.5B q8_0               3.9 GB  305 s in 11.1 s  identical text, turns and speakers
1.5B q4_k               3.2 GB  305 s in 10.2 s  identical text, turns and speakers
```

The first try failed routing ("missing `config.json`"): a local file id goes through the auto loader's Hub-style config
read, not `local_config_artifacts`; both now unpack it.

## Run 3 - 2026-11-12 01:40

Question: the offline VibeVoice-ASR GGUF (`VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf`)?

It embeds Microsoft's own checkpoint (`microsoft/VibeVoice-ASR`, class `VibeVoiceForASRTraining`, the streaming
checkpoints' names: `model.language_model.*`, `model.*_connector`, a root `lm_head`), not transformers' port that we
load (`microsoft/VibeVoice-ASR-HF`). It has no `preprocessor_config.json` (the processor's defaults: -25 dB FS
normalization) and adds Qwen2.5's tokenizer files, which Microsoft's repo leaves to `Qwen/Qwen2.5-*`.

Changes: `VibeVoiceAsrConfig::parse` reads either config (the upstream encoder config shared with streaming); the
model takes its decoder under `model.language_model` with `lm_head` at the root when the layout is Microsoft's; the
loader row lists both classes. Every safetensors loader reads an audio.cpp GGUF through two hooks: file lookups
(`hf::get_file`, `try_get_file`, `list_repo_files`) resolve to the unpacked directory, and the weights
(`get_model_paths`, `WeightFiles::load`) are the GGUF through the reader.

Serving each, 12.7 s dialogue and 305 s clip, `diarization_model=default`, CUDA:

```
transformers bf16 (-HF)        305 s in 49.1 s; 21.9 GB with its 7.2 GB paged KV pool
Microsoft's bf16 (safetensors) identical to -HF: every segment's text, times and speaker, both clips
audio.cpp q8_0                 305 s in 29.4 s; 17.1 GB with the same pool; text and speakers identical, times moved in
                               48 of 49 segments: +0.01 to +0.05 s (mostly +0.02), always later
```

The bf16 runs agree exactly, so Microsoft's layout loads as transformers' does; the late drift is q8_0's (the
reference's own noise moves times by up to 0.02 s too, Run 2 of vibevoice_asr_investigation.md). Microsoft's repo
has no tokenizer: from the Hub it needs `--tok-model-id Qwen/Qwen2.5-7B` (the GGUFs embed theirs).

Tests: a package written as audio.cpp writes one (an aliased name, a scalar, embedded files) reads back
(`audiocpp::tests`); the tiny streaming checkpoint repacked as an audio.cpp GGUF transcribes as its safetensors, by
file and by directory (`vibevoice_streaming_tiny::an_audiocpp_gguf_transcribes_as_its_checkpoint`). Unpacks whose GGUF
is gone or changed are pruned on the next unpack.

## Run 4 - 2026-11-12 01:55

Question: do the families we already run come along through the same hooks? Two local GGUFs (downloaded 2026-09,
possibly older than the current uploads), `inference serve -m <file>`:

```
qwen3-asr-0.6b-q8_0          routes as Qwen3-ASR, then: `tokenizer.json` not found (the package embeds vocab.json and
                             merges.txt only)
parakeet-tdt-0.6b-v3-q8_0    reading the header: unknown dtype for tensor 27
```

Phase 2 work: a tokenizer built from `vocab.json` + `merges.txt` where a package has no `tokenizer.json`, the GGML
types our reader lacks (tensor 27 of the Parakeet file), and each family's names checked against its loader.

## Run 5 - 2026-11-12 07:15

Question: what do the current uploads hold? The September copies predate audio.cpp's 2026-10-19 refresh ("Refresh
embedded GGUF specs for v0.9.2"), so the q8_0 (q4_k for Voxtral) of each family we run was fetched again:
`hf download audio-cpp/audio.cpp-gguf <folder>/<file>...` and `audio-cpp/Nemotron-3-Diarization-GGUF`.

```
Parakeet-TDT-0.6B-v3       transformers layout (ParakeetForTDT): config, processor_config, tokenizer.json; 24 I64
                           tensors (BatchNorm's num_batches_tracked); the last tensor's padding left off the file
Nemotron-3.5-ASR-Streaming transformers layout (Nemotron3_5AsrForRNNT) with its tokenizer.json
Cohere-Transcribe          the release's files and names
Qwen3-ASR 0.6B / 1.7B      Qwen's repo: vocab.json + merges.txt, no tokenizer.json
Qwen3-ForcedAligner-0.6B   Qwen's original repo (no tokenizer.json, the time classes in lm_head)
Qwen3-TTS (4 packages)     two sources, `model_weights/` and `speech_tokenizer_weights/` (the codec, encoder included)
Kokoro-82M q8_0            `audiocpp` now, one source `kokoro/` over the .pth names, config and voices/*.bin embedded
Kokoro-82M multilingual    still the older `kokoro_tts` architecture, which we already read
Voxtral-Mini-4B-Realtime   transformers' names (audio_tower, language_model, multi_modal_projector) beside Mistral's
                           params.json and tekken.json
Sortformer-Diar-4spk-v1    the offline v1 (SortformerOffline), not the streaming v2.1 we run
Nemotron-3-Diarization     audio.cpp's own conversion from the .nemo: its own config (chunk 264, no FIFO), NeMo names
```

Each failure on the way, in order: `Content::read` refuses the I64 type (GGML 27), so the reader takes the header
alone (`gguf_file::Header`, raw type ids) and binds no integer-typed tensor; the archive refused the unpadded last
tensor ("padded storage ... ends at byte 915735776, beyond file length 915735760"), which llama.cpp's loader accepts,
so the storage range now ends at the file; the transcription loader still bailed for every family but VibeVoice.

## Run 6 - 2026-11-12 07:50

Question: does each family transcribe from its package as from its release? `/v1/audio/transcriptions` verbose_json,
the 12.7 s dialogue and the 305 s clip (the 36-word dialogue 24 times over, so 864 words is all of it), CUDA, bf16.
Speech loaders now take a `VarBuilder` from core (`load_with`): plain-`VarBuilder` models (Parakeet, Cohere's
encoder, Kokoro) read the package dequantized tensor by tensor (`dense_var_builder`), the rest through
`GgufWeightSource` with quantized matrices kept.

```
Parakeet q8_0 vs nvidia/parakeet-tdt-0.6b-v3   dialogue identical; 305 s: the package has all 864 words, the
                                               release 838 (one 26-word sentence missing from a window)
Nemotron-3.5 q8_0 vs the release               dialogue identical; 305 s: 2 words apart, both toward the script
                                               ("a" -> "the", a dropped "Great" back)
Cohere q8_0 vs CohereLabs release              identical on both
Qwen3-ASR 0.6B q8_0 vs Qwen/Qwen3-ASR-0.6B     identical on both (792 words each)
Qwen3-ASR 1.7B q8_0 vs Qwen/Qwen3-ASR-1.7B     identical on both (4233 words each: both loop on the 305 s clip in
                                               one pass)
```

Qwen's repos carry no `tokenizer.json`: a `Qwen2Tokenizer` now builds from `vocab.json` + `merges.txt` + the config's
added tokens, as transformers converts it (NFC, Qwen's split, byte-level BPE; settings checked against
`AutoTokenizer`'s `backend_tokenizer`). That also loads `Qwen/Qwen3-ASR-0.6B` and the original
`Qwen/Qwen3-ForcedAligner-0.6B` from their own repos, which failed before ("Could not find `tokenizer.json`").

Qwen3-ForcedAligner q8_0 against `Qwen/Qwen3-ForcedAligner-0.6B-hf`, beside Qwen3-ASR 0.6B, word timestamps: equal
on the dialogue (36 words) and the 105 s reading (147 words). On the 305 s clip 628 of 792 equal, and from word 685
both break down: the transcript has lost repeats there, the reference collapses the words onto 216.64 s and the
package spreads them 0.15 s apart. The aligner first failed with "expected `,` or `}` at line 1 column 5": the loader
handed it the tokenizer file's text as `tokenizer.json`; it now gets the loaded tokenizer serialized.

## Run 7 - 2026-11-12 08:10

Question: the speech models. `/v1/audio/speech`, each output transcribed by Parakeet.

```
Kokoro q8_0 vs hexgrad/Kokoro-82M (af_heart)   6.67 s vs 6.72 s, same transcript; two reference runs agree only to
                                               20 dB SNR (decoder noise), so no sample-level comparison
Kokoro multilingual (ff_siwis, French)         transcribes back exactly
Qwen3-TTS, seed 7, q8_0 package vs release:
  1.7B CustomVoice (vivian)                    6.24 s vs 6.56 s   both say the text
  1.7B VoiceDesign (instructions)              5.92 s vs 6.96 s   both say the text
  0.6B Base (clip + reference_text)            5.04 s vs 4.80 s   both say the text
  1.7B Base (clip + reference_text)            5.28 s vs 5.68 s   both say the text
GPU, package vs release: 5.2 vs 6.5, 5.1 vs 6.6, 5.2 vs 5.7, 6.3 vs 7.7 GB
```

The packages keep the codec encoder, so cloning with `reference_text` works, which llama.cpp's Qwen3-TTS GGUFs
cannot. Sampled audio diverges between q8_0 and bf16 from the first frames, so only the transcripts compare.

Not loaded, and why: Sortformer's package is the offline v1, a model we do not run; Nemotron-3 Diarization's is
audio.cpp's own conversion (NeMo names, its own config and streaming settings) of a 0.1 GB model, little gained
over the release. Voxtral Realtime needs transformers' layout (next run).

## Run 8 - 2026-11-12 09:30

Question: Voxtral Realtime from its package, whose weights are transformers' `model.safetensors` (the repo ships it
beside `consolidated.safetensors`)? The first load failed on `mm_streams_embeddings...conv_layers.0.conv.weight`.

Range reads of both files' layer 0 (`scratchpad` script over the Hub, no full download) showed the layouts hold the
same values except every Q and K projection, encoder and decoder, Q's bias included: transformers' rows are each
head de-interleaved (`consolidated == model.reshape(h, 2, d/2, ...).swapaxes(1, 2)`), the HF half-split order. So the
model reads either layout from one names table (`voxtral::names`, picked by which encoder conv the checkpoint holds,
since ISQ hides the embedding from `contains_tensor`) and rotates half-split pairs for transformers'.

`/v1/chat/completions` "Transcribe this.", greedy, against `consolidated.safetensors` in bf16:

```
                     q4_k package          consolidated bf16
12.7 s dialogue      identical text
105 s reading        189 words identical   (bf16 ran out of memory here at first: see below)
305 s clip           864 words identical (all of it)
                     26.8 s, 11.1 GB       47.2 s, 16.3 GB
```

Both with `--pa-context-len 16384`: by default the KV cache holds one sequence at the full 131,072 context fitted to
the free memory, and the bf16 load then has no room for the encoder on a 105 s clip (CUDA out of memory); the q4_k
package fit there in 19.3 GB. Tests: a tiny checkpoint renamed and re-laid as transformers' decodes as the
consolidated one (and fails with adjacent RoPE), an ISQ load of it writes a UQFF that reloads alike (residual names),
and its audio.cpp repack decodes as its safetensors (`voxtral_tiny`).
