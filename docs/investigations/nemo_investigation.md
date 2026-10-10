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

## Run 6 - 2026-10-10

**Silero VAD: weights, architecture, parity.**

Weights:
- The `silero-vad` PyPI package (6.2.3, MIT) ships `silero_vad.jit` (TorchScript, archive "VADr_v6_10_25") and `silero_vad_16k.safetensors`.
- They are different weight sets. The STFT basis is equal; conv1, the LSTM and the final conv are not.
- Over the 7.4 s LibriSpeech clip, the safetensors differs from the jit by up to 0.33 in probability (178 vs 179 frames over 0.5). It is most likely the v5 weights, kept for the package's tinygrad port.
- audio.cpp's bundled `silero_vad_16k.safetensors` is byte-equal to the package's, so it is v5.
- `mlx-community/silero-vad-v6` (MIT, "silero_vad PyPI 6.2.1") holds the jit's v6 weights: kernels transposed to (out, k, in), and the LSTM biases summed (`bias == bias_ih + bias_hh`, checked).

Architecture:
- A torch port of the package's tinygrad model with the jit's weights matches the jit exactly (max diff 0.0 over 233 chunks).
- Per 512-sample chunk with 64 samples of context: reflect pad 64, STFT conv (258, 256, stride 128, so 4 frames), magnitude, conv1-4 (strides 1, 2, 2, 1, so 1 frame), an LSTM cell (128), ReLU, a 1x1 conv, sigmoid.

Distribution:
- The user asked for a small GGUF, as a separate model.
- `examples/silero_vad_gguf.rs` converts either layout to `silero_vad`: F32, PyTorch names, 1.24 MB, with the framing and `SegmentOptions` defaults as metadata.

Implementation:
- The encoder is batched over all chunks; the LSTM and head run as a host loop on every device, one step per chunk.
- `speech_segments` ports `get_speech_timestamps_from_probs`, including the max-duration split at the longest silence and the padding pass's int truncation.
- **Dead end:** `max_speech_duration_s = inf` serialized to JSON `null`, which did not read back ("invalid type: null, expected f64"). It is now `Option`, where unset means unbounded.

Parity (`scripts/silero_vad_parity.sh <gguf> <wavs> --cpu`) against the jit plus `get_speech_timestamps`:

| clip | chunks | max prob diff | segments (default / 5 s cap / 2 s cap) | ours, CPU dev |
|---|---|---|---|---|
| LibriSpeech 7.4 s | 233 | 2.0e-6 | 3 / 3 / 4, exact | 0.04 s |
| German 20 s | 625 | 6.9e-6 | 4 / 6 / 11, exact | 0.09 s |
| Spanish 15 s | 469 | 3.3e-6 | 2, exact | 0.07 s |
| German 60 s | 1875 | 3.4e-6 | 12 / 16 / 34, exact | 0.27 s |

Both sides' cuts of the reference's own probabilities match too, so the cutting port is verified apart from the model.

## Run 7 - 2026-10-10

**Silero VAD in the engine, and Parakeet's long form.**

Surface:
- `ModelSelected::VoiceActivity` (a GGUF file, a directory holding one, or an HF repo holding one); local GGUFs are auto-detected.
- `ModelCategory::VoiceActivity`, with per-sequence results as for transcription.
- `Engine::voice_activity(_json)` and `POST /v1/audio/vad` (multipart; the reference parameters, plus `return_probabilities`).
- `inference_voice_activity` in the C ABI, with Python and C# wrappers.
- `VoiceActivityModelBuilder` in the SDK, and a CLI interactive mode.
- The multipart reader is now shared by both audio routes. Each route declares its number, flag and list fields, so a text field is never coerced.

Segmentation defaults moved out of the GGUF: they are the reference's constants, and requests override them field by field.

Long form:
- `ModelSelected::Transcription { vad_model_id }` loads the VAD beside Parakeet.
- Audio over 5 minutes is cut with `max_speech_duration_s = 120` into windows of at most 2 minutes, from a run of segments' first start to its last end. Each window is transcribed alone and its tokens and words are shifted by the window's offset.
- Stretches without speech are skipped, and the 24-minute cap does not apply.

Silero parity on CUDA, same clips as Run 6: max probability diff 3.34e-6 to 6.85e-6, and segments exact both with defaults and with the 5 s cap.

**Long form on a real recording** (a 10-minute conversational German recording, CUDA F32, `examples/parakeet_transcribe.rs`):

| run | time | words |
|---|---|---|
| full context, no VAD | 6.52 s | 1452 |
| windowed with the VAD | 6.05 s | 1472 |

- 1280 words are in common (88.2% of the full-context transcript).
- The differences are mostly filler words, both ways.
- One clause the full-context run has is missing from the windowed run, consistent with quiet speech the VAD classed as silence between windows. Unconfirmed: there is no reference transcript for this file, so neither run is known to be the more accurate.
- Follow-up: measure on audio with a reference transcript, and consider padding windows past the VAD's edges.

Tests:
- `silero_tiny`: thresholds 0 and 1.1 give one segment and none whatever the weights; chunk counts and probability ranges; undecodable audio is a 400.
- Long form with an always-speech VAD: a 301 s clip transcribes in windows, and words after 120 s carry their offset.
- Segment-cutting unit cases pinned to the reference function's own outputs. (I first wrote one expecting a 96 ms gap to split; the reference bridges it, being under 100 ms.)
- The mlx-layout GGUF round trip, the `/v1/audio/vad` form (including a field-level 400), and the Python and C# wrappers.

## Run 8 - 2026-10-10

**Diarization survey.** The user pointed to `nvidia/Nemotron-3-Diarization`, the newer replacement for Sortformer v2.1, and asked to do both.

- Both are NeMo `SortformerEncLabelModel`s with the same streaming modules: an arrival-order speaker cache (AOSC) plus a FIFO.
- The encoders differ:
  - Nemotron-3: feature stacking x8, then a 31-layer RoPE transformer (d512), a subpixel upsampler and an 8-speaker head (tf 192).
  - v2.1: a FastConformer (NEST) plus a transformer, 4 speakers.
- References:
  - transformers 5.19 has `nemotron3_diarization` (`Nemotron3DiarizationForAudioFrameClassification`, about 820 lines plus a 267-line processor). The HF repo ships the transformers layout (`config.json`, `processor_config.json`, `model.safetensors`), plus `.nemo` and q8_0 GGUF. License: openmdw-1.1.
  - transformers also has `nemotron_asr_streaming` and `nemotron3_5_asr`, which bear on #391's last part.
  - v2.1 has no transformers port; it exists only as a `.nemo` (in the HF cache).
- Local material (`/mnt/data/models/nemotron3_diarization`):
  - NeMo prediction dumps for 6 clips (`ref/`, `ref_vmel/`);
  - ground-truth RTTMs for AMI and VoxConverse clips;
  - an earlier C# implementation's predictions for both models.

Plan: Nemotron-3 first, against transformers, with the diarization API surface. Then Sortformer v2.1 from `.nemo`, reusing the surface and the speaker cache.

## Run 9 - 2026-10-10

**Nemotron-3 Diarization port, parity against transformers.** `scripts/nemotron_diarization_parity.sh <ckpt> <wavs> --cpu`, F32, on 6 clips (a 97.6 s demo clip, three 300 s VoxConverse dev clips, two 300 s AMI SDM clips).

Implementation (`diarization::{nemotron3, cache}`):
- Features are `NemoMel` without normalisation, and the masked trailing frame is dropped.
- 8x frame stacking, then 31 pre-norm layers with half-split RoPE, using cos/sin tables built at load.
- `proj`, then the subpixel upsampler conv (192 to 1536, k3) and the relu-dense-relu head.
- The speaker cache ported from transformers: scores and selection on the host in f32, embeddings gathered on the device.
- Offline runs use the top-level chunking: 340 + 40 lookahead, FIFO 40, update period 300.

**First run:** every segment exact, but probabilities off by up to 0.027 and 0.094. Locating the first frame past 1e-4 put it at 9752 and 29992: the final 8 frames, the last encoder group, only.
- Cause: transformers keeps the masked trailing feature frame. That gives one extra, all-padding encoder group, masked as a key but still computed. Its upsampler (k3) then reads that group's hidden state for the last real group's frames, where ours sees zero padding.
- Kept ours (zero padding past the audio). The parity report shows the final 80 ms apart.

**Second run:** four clips within 2.4e-5 before the final group. Two diverged mid-stream (from frames 5440 and 19040; frame 5440 is encoder frame 680, a chunk boundary).
- The first update already pops 300 frames into the 264-slot cache, so it compresses from the start, and its top-k keeps whichever of two near-equal frames the float noise favours.
- **Check:** transformers on CPU against transformers on CUDA, the same F32 weights. It parts the same way on the same clips (max diffs 5.76e-2 and 9.24e-2; first past 1e-4 at frames 764 and 16328). This is the model's sensitivity, not a port bug.
- Such clips are judged on speaking-decision agreement (at least 99.5% of frame x speaker decisions).

| clip | max prob diff (before last group) | decisions agree | segments exact | ours, CPU dev |
|---|---|---|---|---|
| demo 97.6 s | 1.56e-5 | 100.000% | 29/29 | 6.7 s |
| VoxConverse c | 2.09e-6 | 100.000% | 13/13 | 19.8 s |
| AMI a | 5.76e-2 (cache flip) | 99.983% | 229/250 | 20.3 s |
| AMI b | 2.42e-5 | 99.999% | 128/129 | 19.2 s |
| VoxConverse a | 2.04e-5 | 100.000% | 70/70 | 19.8 s |
| VoxConverse b | 9.61e-2 (cache flip) | 99.997% | 14/18 | 19.7 s |

- The NeMo dumps in `ref/` differ from transformers by up to 0.13. NeMo ran its own chunking (264 frames, FIFO 0) and dither, so they are not a parity target.
- **Timing:** CPU takes 19.5 s for 300 s, against 5.2 s in torch. A follow-up.

## Run 10 - 2026-10-10

**Nemotron-3 Diarization in the engine; CUDA parity.**

Surface:
- `ModelSelected::Diarization`, auto-detected from `model_type: nemotron3_diarization`.
- `ModelCategory::Diarization`, with per-sequence results.
- `Engine::diarization(_json)` and `POST /v1/audio/diarization` (multipart: `threshold`, `return_probabilities`, `response_format` json or rttm).
- `inference_diarization` in the C ABI (a blob carrying JSON or RTTM), with Python and C# wrappers.
- `DiarizationModelBuilder` in the SDK, and a CLI interactive mode that prints RTTM.

The tiny checkpoint (fixture configs: cache 16, chunk 10 + 2, FIFO 4) fills and compresses the cache within a 6 s clip. Tests cover:
- frame counts, and thresholds 0 and 1.01 (one segment per speaker, and none);
- RTTM lines;
- repeats on the default device;
- the HTTP form;
- the Python and C# wrappers.

CUDA parity, same 6 clips as Run 9:
- Every result matches the CPU run: the same two cache flips, and the same agreement (99.983% and 99.997%).
- Max probability diff 1.2e-5 to 3.6e-5 where the cache stays in step.
- 300 s in 1.47 s (about 200x real time), against 19.5 s on CPU in a dev build.

## Run 11 - 2026-10-10

**Review fixes, and BF16 measured.**

The review found that a BF16 load failed every request, because the cache read F32 out of BF16 logits. Auto picks BF16 on sm80+ GPUs, so `serve -m nvidia/Nemotron-3-Diarization` would have failed. The cache now scores in F32 whatever the model dtype.

`scripts/nemotron_diarization_parity.sh ... --bf16` on CUDA, the same 6 clips (the reference stays F32):
- Speaking decisions agree on 99.986%, 99.997%, 99.922%, 99.968%, **99.399% (FAIL)** and 99.994%.
- Only 7/13 to 160/250 of segments match exactly, against nearly all in F32.
- Max probability diff is up to 0.95.
- Speed: 1.39-1.44 s per 300 s, against 1.50-1.57 s in F32.

BF16 moves speaker decisions for no real speedup, so `ModelSelected::Diarization` resolves `auto` to F32, as the VAD does. An explicit `bf16` still loads.

Other fixes:
- Embeddings are built per step from the features slice, instead of one `(1, groups, hidden)` tensor for the whole clip. That drops the full copy of the features and the clip-sized device tensor; F32 parity is unchanged.
- `threshold` must be within [0, 1]; NaN is rejected too.
- The cache-budget rates are parsed as f64, as Python floors them.

Left as follow-ups:
- Mel features are still computed for the whole clip (about 9 GB of f64 at 24 h).
- `return_probabilities` JSON grows with audio length (about 30 MB per hour).
- The RTTM file ID is fixed as `audio`.

## Run 12 - 2026-10-10

**Sortformer v2.1: what the checkpoint holds, and where a reference comes from.**

The `.nemo` (`nvidia/diar_streaming_sortformer_4spk-v2.1`) is a plain tar holding `model_config.yaml` and `model_weights.ckpt`, a torch zip of 990 tensors, all F32 apart from batch-norm counters.

Architecture, from the config and the weight names:
- **Mel:** 128 mels, n_fft 512, window 400, hop 160, no normalization. The filterbank ships in the checkpoint.
- **FastConformer:** 17 layers, d512, 8 heads, conv kernel 9, dw_striding x8, xscaling, rel_pos with biases. It is Parakeet's encoder under NeMo names:
  - `pre_encode.conv.N` -> `subsampling.layers.N`, `pre_encode.out` -> `subsampling.linear`
  - `linear_{q,k,v,out}` -> `{q,k,v,o}_proj`, `linear_pos` -> `relative_k_proj`
  - `pos_bias_{u,v}` -> `bias_{u,v}`, `conv.batch_norm` -> `conv.norm`
- **Projection:** `encoder_proj` 512 -> 192.
- **Transformer:** 18 post-LN layers (d192, FF 768, ReLU), no positional encoding.
- **Head:** relu, `first_hidden_to_hidden`, relu, then `single_hidden_to_spks` (4). `hidden_to_spks` is unused.
- **Checkpoint schedule:** chunk 188, left and right context 1, FIFO 0, cache 188, period 188, 3 silence frames.
- **Silence embedding:** a running mean of low-probability popped frames, not a learned parameter (it is learned in Nemotron-3).

Sources compared:
- **audio.cpp-fork's C++ port:** no NeMo dumps for v2.1; only synthetic tests.
- **The Vernacula ONNX work** (`/mnt/data/Programming/vernacula`, `scripts/nemo_export/`, `docs/investigations/sortformer_*`) is the useful one:
  - `.venv-nemo-export` has NeMo 2.7.1 working.
  - `sortformer_fidelity_der.py` takes the reference from NeMo's own `forward_streaming`. Its rule: never take the reference from a transcription of the loop. A wrong-signed boost survived three rounds of checking against one.
  - A different schedule needs `SortformerModules` rebuilt from the cfg; mutating attributes doesn't reconfigure it.
  - `sortformer_topk_ties_investigation.md`: torch's top-k tie order is a quickselect artifact. It differs across torch builds, so tie-driven divergence (saturated sigmoid rows) is not a defect to drive to zero. That matches what Nemotron-3 showed in Runs 9-10.
  - The `.v21.preds.f32` files are from the ONNX model at Vernacula's 124/124/124 schedule, with no context and a whole-clip mel. They are a sanity check, not a parity target.
- A scratch venv built here with NeMo 2.6 imported, but was deleted in favour of Vernacula's.

Next: a `scripts/sortformer_parity.sh` that dumps NeMo `forward_streaming` at the checkpoint's own schedule, and the port: Parakeet's encoder under a key rename, plus the transformer and head, and the cache with left context and the running silence mean.

## Run 13 - 2026-10-10

**Sortformer v2.1 port, parity against NeMo's `forward_streaming`.**

Command: `NEMO_PYTHON=<vernacula .venv-nemo-export>/bin/python scripts/sortformer_parity.sh <.nemo> <6 wavs> [--cpu]`. NeMo 2.7.1 runs on CPU with the checkpoint's own schedule (chunk 188, context 1/1, FIFO 0, cache 188). Its dumps hold the preprocessor's mel and the streaming probabilities, trimmed to `ceil(mel/8)` frames.

The port:
- `.nemo` read in place: a tar member is a byte range of the file, and `PthTensors::in_range` opens the torch zip there.
- NeMo FastConformer names renamed to the Parakeet encoder's, so that encoder loads unchanged. It is split into `subsample` and `encode` so the cache and FIFO go in between, before the xscale.
- New: the 18 post-LN layers and the head.
- The shared cache gains a left-context offset, encoder-rate probabilities, and a running-mean silence slot (`Silence::Popped`).

Results, CPU and CUDA (same 6 clips as Run 9; the reference is NeMo on CPU both times):

| clip | mel max diff | prob max diff CPU / CUDA | decisions | segments |
|---|---|---|---|---|
| 97.6 s demo | 1.35e-3 | 5.96e-6 / 4.08e-6 | 100% | 29/29 |
| VoxConverse c | 4.74e-4 | 4.17e-6 / 3.68e-6 | 100% | 14/14 |
| AMI a | 1.94e-4 | 1.77e-5 / 2.43e-5 | 100% | 261/261 |
| AMI b | 1.24e-4 | 1.14e-5 / 1.03e-5 | 100% | 126/126 |
| VoxConverse a | 4.99e-4 | 5.60e-6 / 3.78e-6 | 100% | 54/54 |
| VoxConverse b | 3.53e-4 | 6.29e-6 / 5.81e-6 | 100% | 19/19 |

The mel differs by up to 1.35e-3 in the log domain: NeMo's preprocessor is f32 and ours f64, and quiet frames magnify it. The probabilities show the difference does not matter. The tolerance is 5e-3.

No cache flips: unlike Nemotron-3 (Runs 9-10), no clip reaches a top-k near-tie.

Speed for 300 s: about 1.75 s on CUDA (about 170x real time), and about 29 s on CPU in a dev build.

Nemotron-3, rerun on CUDA after the cache refactor: identical to Run 10 (1.93e-5; 99.983% and 99.997% on the flip clips).

## Run 14 - 2026-10-10

**Sortformer v2.1 in the engine.**

Loading:
- `DiarizationLoader` takes the transformers layout where a `config.json` is present (Nemotron-3). Otherwise it takes the one `.nemo` the id names (the file, a directory, or a Hub repo), which must be a Streaming Sortformer.
- The test: `target: ...SortformerEncLabelModel` over an `encoder._target_` ending in `ConformerEncoder`. Nemotron-3's own `.nemo` has the same target over a `TransformerEncoder`, so it is refused rather than misread.
- Auto-detection takes a local `.nemo` after that check, and a Hub repo of a lone `.nemo` from its listing; the loader then checks it.

End to end with the debug CLI, `inference run -m nvidia/diar_streaming_sortformer_4spk-v2.1` given the 97.6 s demo clip:
- logs "Detected a Streaming Sortformer `.nemo`";
- prints 29 RTTM lines, the 29 segments of Run 13.

Tests:
- A tiny random-weight `.nemo` is built at test time. Its config is a committed `model_config.yaml`, and its weights are written by a new `inference_tensor::pickle::write_pth`, read back by `PthTensors`. Its schedule (chunk 6 + 1/1, FIFO 4, cache 16) compresses within a 6 s clip. The engine output is 75 frames of 80 ms, the same from the file and from its directory.
- Unit tests:
  - a torch zip read in place inside a larger file;
  - `write_pth` round-trip;
  - NeMo-to-Parakeet name renames;
  - archive config and weights;
  - the running-mean silence: popped silent frames average into the slots compression fills;
  - auto-detection: Sortformer `.nemo`, transformer `.nemo` refused, Hub listing.

Dependencies:
- `tar` 0.4 (default features off) is now direct; it was already in the lockfile.
- YAML through the workspace's existing `serde-saphyr` 0.0.16, which topology parsing already uses.

## Run 15 - 2026-10-10

**Review fixes.**

Review found no streaming-math defects. Fixed:
- **Auto-detection:** a Hub `.nemo` routes to diarization only when its name or the repo id says "sortformer". Otherwise any `.nemo`-only repo (an ASR model) would download several GB and then be refused.
- **Silence mean:** kept in F32, as NeMo's float32 zeros keep it. In BF16, `mean * frames` over thousands of silent frames outruns the mantissa.
- **Loader layout test:** now "`config.json` resolves" instead of "the listing names it". A listing that failed with the network down sent a cached Nemotron-3 down the `.nemo` path.
- **Small fixes:**
  - `FileWindow::read` saturates past the window end;
  - `write_nemo` builds the zip in memory, with no temp file;
  - the class is matched by suffix, so aliases of the module path pass;
  - a gzipped (old) `.nemo` gets a clear error.
- **Tests:**
  - tiny weights carry NeMo names, so the rename runs end to end;
  - the fixture's silence threshold makes every popped frame silent, so the mean is non-zero;
  - the cache test's second batch has a silent frame, pinning observe-before-compress.

Not taken: opening the torch zip once per load instead of once per tensor. `ZipArchive` clones only over a `Clone` reader, and loading the 990 tensors already takes well under a second next to inference.

CUDA parity after the fixes is identical to Run 13 (2.43e-5 max, all 503 segments exact).
