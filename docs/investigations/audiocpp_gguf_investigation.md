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
