# NeMo speech models (#391): Parakeet, Sortformer, Silero VAD

## Run 1 - 2026-10-09

**Survey, to plan #391.**

inference.rs today:
- There is no transcription surface: no `/v1/audio/transcriptions`, request or response type, C ABI entry or binding. `ModelCategory::Audio` exists but is unused.
- The one speech-to-text path is chat with Voxtral.
- The pipeline template to follow is speech generation (`pipeline/speech.rs`, `RequestMessage::SpeechGeneration`, `Response::Speech`).
- `crates/inference-audio` decodes audio but has neither resampling nor mel features; each model does its own.
- The Phi conformer has NeMo `dw_striding` subsampling. Its relative positions are a T5-style bias, not Transformer-XL `pos_bias_u/v` with rel_shift.
- Kokoro's `BiLstm` is single-layer and stateless.

audio.cpp (the fork is newer, 2026-09-24):
- Parakeet-TDT 0.6B v3: from HF safetensors or its own GGUF; mel → 8x dw_striding → 24 FastConformer layers → 2-layer LSTM predictor and joint → greedy TDT → SentencePiece words.
- Nemotron 3.5 streaming ASR: cache-aware RNN-T.
- Sortformer v1 and v2.1: streaming AOSC speaker cache.
- Nemotron-3 diarization.
- Silero VAD v5: STFT conv, 4 conv1d, LSTM cell; `get_speech_timestamps` post-processing.
- No beam search, and no CTC for Parakeet.

References on this machine:
- transformers 5.19 has `ParakeetForCTC`/`ForRNNT`/`ForTDT`, a torch reference without NeMo, which is not installed.
- `nvidia/parakeet-tdt-0.6b-v3` ships HF `model.safetensors` (723 tensors, F32, HF names), `.nemo` and a q8_0 GGUF.
- Local checkpoints: Sortformer v2.1 `.nemo` (HF cache), Nemotron-3-Diarization `.nemo` with references, the Nemotron 3.5 ASR streaming GGUF, and Silero as ONNX only.

**Plan:** one PR per item.
1. Parakeet (HF safetensors; TDT, RNN-T and CTC heads) with the NeMo mel frontend and the whole transcription surface (protocol, core, inference-api, server `/v1/audio/transcriptions`, C ABI, bindings), plus a parity script against transformers and a tiny-checkpoint engine test.
2. Silero VAD, which also gates long audio.
3. Sortformer v2.1.
4. Nemotron cache-aware streaming, `.nemo` archives and GGUF.

**Environment fixes along the way:**
- `hf download` failed with "process() takes no keyword arguments": the user-site httpx2 needs Brotli 1.2+, and only the system's 1.1 was installed. Upgraded in the user site.
- transformers' Parakeet feature extractor needs librosa; installed in the user site.
- `python3 -I` also drops the user site, which holds torch. Use `-P` for scripts in the scratchpad.

## Run 2 - 2026-10-09

**Reference transcript.** `nvidia/parakeet-tdt-0.6b-v3` through transformers on CPU, on audio.cpp's LibriSpeech fixture (7.43 s, 16 kHz mono):
- 0.68 s.
- Text: "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait."
- The first token timestamps: W 0.32-0.40, ell 0.40-0.56, "," 0.56-0.56, " I" 0.64-0.80.
- The grid is 80 ms per encoder frame (hop 160 x factor 8 at 16 kHz). Punctuation is pinned to the previous token's end.

**Reference details to match:**
- Mel:
  - preemphasis 0.97 (masked past the audio's length);
  - `torch.stft` centered with zero (constant) padding, n_fft 512, win 400, hop 160, symmetric Hann;
  - power spectrum, librosa Slaney mel (128 bins, 0 to 8 kHz, slaney norm);
  - `ln(x + 2^-24)`;
  - per-feature normalisation over the valid frames, with variance over N-1 and the epsilon 1e-5 added to std, then valid frames masked.
  - Valid frames are `(len + 2*(n_fft/2) - n_fft) / hop`.
- Subsampling: conv2d(1→256, k3 s2 p1), ReLU, then twice [depthwise conv2d k3 s2 p1, pointwise 1x1], ReLU. Flatten (C, F/8) to 4096, then a linear layer.
- Relative positions: `pos = [T-1 .. -(T-1)]`, inv_freq over hidden 1024, sin/cos interleaved per pair.
- Attention:
  - `matrix_bd = (q + bias_v) @ relative_k_proj(pos)ᵀ`, rel_shift, keep the first T, scale;
  - `scores = (q + bias_u) kᵀ * scale + matrix_bd`.
- Block: macaron FF with 0.5 weight, attention, conv module (pw1, GLU, depthwise k9 with symmetric padding, BatchNorm, SiLU, pw2), FF2 at 0.5, norm_out.
- Decoders:
  - TDT: the token argmax over the first 8193 logits and the duration argmax over the last 5. A blank with duration 0 becomes 1.
  - The predictor LSTM (2 layers) advances only on a non-blank token; it starts from the blank id (8192).
  - Joint: `relu(enc_proj + dec_proj)` → head.
  - RNN-T: a blank (or `max_symbols_per_step` symbols at one frame) advances one frame.
  - CTC: a 1x1 conv head, argmax, merge repeats, drop the pad/blank.

## Run 3 - 2026-10-10

**Parakeet port, parity against transformers.** `scripts/parakeet_parity.sh <parakeet-tdt-0.6b-v3> <wavs> --cpu`, F32.

Implementation:
- `inference_audio::nemo::NemoMel`, computed in f64. The Slaney filterbank moved from Voxtral into `inference_audio::mel`, with Voxtral's arithmetic unchanged.
- `parakeet::{encoder, decoder}`.
- BatchNorm is folded into the depthwise conv.
- The conformer's depthwise conv is written as 9 shifted multiply-adds, because inference-tensor's grouped conv on CPU splits into one conv per channel.
- The masked trailing feature frame is dropped before the encoder instead of masking the attention. That is equivalent: the zero conv padding equals the masked frame, and padded keys only ever feed padded rows.

```
ok 2086-149220-0033.wav (LibriSpeech, 7.4 s):  features cos 1.000000, encoder cos 1.000000, emissions 44/44, text matches, 3.41 s
ok de_20s.wav (German read speech, 20 s):      features cos 1.000000, encoder cos 1.000000, emissions 96/96, text matches, 6.87 s
ok es_15s.wav (Spanish read speech, 15 s):     features cos 1.000000, encoder cos 1.000000, emissions 23/23, text matches, 5.29 s
```

- **Exact on the first run.** Emissions compare token, frame and span.
- **First Spanish sample:** both implementations emitted nothing on it. It is loud throughout (RMS 0.22 in every 30 s window), so probably music, not speech. The second sample was used.
- **Dead end:** the first attempt passed the samples whole (10 and 6.5 minutes). Full attention over about 7,500 frames needs several GB per attention matrix in both implementations, so the run was stopped and the clips cut to 15 to 20 s. Long audio needs a plan of its own: a bound on attention memory, a duration cap, and VAD segmentation in PR 2.
- **Timing:** the per-clip times include a second encode inside `transcribe`, in a dev build with debug assertions on. Not a perf number yet.

## Run 4 - 2026-10-10

**Engine plumbing, CUDA, long audio, timing.**

Transcription surface:
- `RequestMessage::Transcription`, `ModelCategory::Transcription`, which replaces the never-produced `Audio`.
- `TranscriptionPipeline` with auto-detection from `model_type`, and `ModelSelected::Transcription`.
- `Engine::transcription(_json)`, `POST /v1/audio/transcriptions` (multipart), `inference_transcription` in the C ABI, and Python and C# wrappers.
- `TranscriptionModelBuilder` in the SDK, and a CLI interactive mode.
- Response formats json, text, srt, vtt and verbose_json. Segments split at sentence ends and at pauses of 1.5 s or more.

Bugs and changes found along the way:
- **CUDA:** `matmul is only supported for contiguous tensors ... rstride [95232, 128, 1, 1024]`. The key, after the head split and transpose, is not a batched-GEMM layout. The CPU path accepted it, so CPU parity passed. Fixed by making the per-head tensors contiguous.
  - A new default-device tiny test (CUDA under `--features cuda`) covers this; every earlier tiny test forced CPU.
- **Timestamps past the audio:** a TDT token predicted near the end can span frames beyond the audio, and the reference does not clamp. Clamped to the duration; the tiny engine test caught it with a word ending after the clip.
- **Long audio:** attention queries are split into blocks of 512 rows. Rows a..b read the position rows from T - b, and the same pad-and-reshape shift then aligns the block. Checked numerically in numpy against the direct `offset = i - j` scores for blocks [0,4), [4,8), [8,11) and [3,10) of T = 11, all equal.
  - Audio over 24 minutes (NeMo's full-attention bound) is refused.
- **Resampling:** the shared resampler is Voxtral's (and Gemma 3n's, the same code).
  - Gemma 4's own differs: it flushes to the expected length. It deliberately keeps rubato's delay; its comment says trimming it drifted from HF's librosa/soxr path. It stays separate, as does Phi's 8 kHz handling.
  - I first thought the untrimmed delay was a bug; Gemma 4's measurement says otherwise.
- **Disk:** the root filesystem filled (916 GB, 219 MB free) during a test build. Cleared `target/debug/incremental` (28 GB, regenerable).

Parity (exact everywhere):

| clip | CPU | CUDA |
|---|---|---|
| LibriSpeech 7.4 s | 44/44 emissions, cos 1.000000 | 44/44 |
| German 20 s | 96/96 | 96/96 |
| Spanish 15 s | 23/23 | 23/23 |
| German 60 s (750 frames, 2 attention blocks) | 245/245 | 245/245 |

Warm `transcribe` time (CPU with dev debug assertions off, 8 threads), against transformers in torch (8 threads):

| clip | ours CPU | torch CPU | ours CUDA |
|---|---|---|---|
| 7.4 s | 1.17 s | 0.53 s | 0.06 s |
| 60 s | 6.87 s | 3.54 s | 0.43 s |

CPU is about 2x slower than torch, a follow-up. CUDA runs at about 140x real time.

## Run 5 - 2026-10-10

**Review fixes, the bundle size, and disk space.**

Review findings, all fixed:
- **One clip failed its whole batch.** A clip the model refuses (too short, over 24 min) failed every sequence batched with it, as a 500 `model_error` that lost the message.
  - `ForwardInputsResult::Transcription` now carries one `Result` per sequence. A refused clip answers only its own request, with a 400 `ValidationError` carrying the model's message.
  - New test: three concurrent requests, the middle one 10 ms long. The middle fails "too short"; the other two succeed and agree.
- **A lost client failed its batch.** A disconnected client's send error aborted the others' responses. The transcription sender now logs and continues. Speech and image have the same pattern; left alone, out of scope.
- **Length check placement.** The 24-minute cap now runs before resampling, so a 3-hour decode is not copied first.
- **Subtitle segments** are capped at 30 s, so unpunctuated CTC/RNN-T transcripts still split.
- **Form checks:** `stream=true` is refused with a 400, a duplicate `file` is refused, and a multipart error names its field. `verbose_json` echoes the request's `language`.
- **Constants:** the resampler settings, the metadata length and the max-symbols default are now named.

Noted, not changed:
- The shared resampler leaves rubato's filter delay in. Content lands about 16 ms late from 8 kHz input and about 3 ms late from 44.1 kHz, the same amount is cut at the end, and parity never exercises it (transformers refuses a rate mismatch).
- Gemma 4's own resampler keeps the delay on purpose, for HF parity.
- Both effects are under the 80 ms timestamp grid. Revisit if a reference that resamples disagrees.

**Bundle size** (`--size`, sm_86):

| | `libinference_ffi.so` |
|---|---|
| Baseline (set at #337) | 100.42 MiB |
| master | 106.41 MiB (+6.00) |
| This branch | 107.08 MiB (+0.66 over master) |

- The +6.00 MiB came from #389-#406 (layout GGUF, Kokoro, vernacula-phonemizer), merged without `--size`.
- The user accepted the update. The baseline now reads 112,280,688 bytes.

**Disk** (root filesystem, 916 GB):
- It filled twice during CI ("No space left on device").
- `target/debug` was 47-67 GB:
  - incremental cache: 19-28 GB;
  - 234 test binaries: 22.7 GB. About half of each is line-table debug info: in a 245 MB binary, `.text` is 60 MB and the debug sections about 120 MB;
  - rlibs and rmeta: 9.7 GB.
- No `--sweep` had completed: two CI runs died before it, and this session's ad-hoc builds added duplicate trees.
- The first completed sweep freed 45.2 GiB: `target/debug` went to 13 GB, with 89 test binaries (6.6 GB). The disk now has 50 GB free.
- The user wants the incremental cache kept (fast rebuilds).

**Dead end:** to measure master's bundle I built it from a git worktree into the same `target/`.
- Cargo gives path crates the same artifact names in either checkout, and judged master's newer artifacts fresh against the branch's older sources.
- The branch's bundle then compiled `inference-api` against master's `inference-protocol` ("no `TranscriptionRequest` in `openai`").
- Fixed by deleting `target/bundle`. Never build another checkout into this `target/`.
