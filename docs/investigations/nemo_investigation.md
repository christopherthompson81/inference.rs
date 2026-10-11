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

## Run 16 - 2026-10-10

**Nemotron streaming ASR: what is on disk.**

Remaining #391 scope: the cache-aware streaming Nemotron ASR; Parakeet from `.nemo`, including the hybrid TDT/CTC (`nvidia/stt_en_fastconformer_hybrid_large_pc`, only its refs dir is cached); GGUF where converters write it; beam search.

The local GGUF, `nemotron-3.5-asr-streaming-0.6b-q8_0.gguf` (931 MB), was read with `gguf-py`'s `GGUFReader`:
- Written by audio.cpp, not llama.cpp: `general.architecture = audiocpp`, `audiocpp.model_spec.family = nemotron_asr`.
- It embeds `config.json`, `processor_config.json` and `tokenizer.json`, and keeps transformers' tensor names (`audiocpp.tensor_name_format = native`).
- 655 tensors:
  - Q8_0: the large linears;
  - F16: the 13088-token embedding;
  - F32: norms, convs and the LSTM.
- Layout against Parakeet:
  - `encoder.subsampling.conv_in` plus `layers.N.{depthwise,pointwise}_conv`, where Parakeet has `layers.N` indices;
  - the subsampling linear takes 4352 = 256 x 17 inputs, which suggests causal padding over 128 mels (128 -> 65 -> 33 -> 17);
  - a `prompt_projector` (1152 -> 2048 -> 1024) that Parakeet lacks;
  - the decoder and joint are the RNN-T shape Parakeet has, with 2 LSTM layers.

Next: map transformers' `nemotron_asr_streaming` and `nemotron3_5_asr` against our Parakeet port; that comparison is running.

## Run 17 - 2026-10-10

**Nemotron streaming ASR, offline, parity against transformers.**

Checkpoints (HF, transformers layout, F32):
- `nvidia/nemotron-speech-streaming-en-0.6b`: `nemotron_asr_streaming`, vocab 1025, window 71, lookahead 13.
- `nvidia/nemotron-3.5-asr-streaming-0.6b`: `nemotron3_5_asr`, vocab 13088, window 57, lookahead 3, and 128 language prompts.

Both use a `nemotron_asr_streaming_encoder`: 24 layers, d1024, 128 mels, no biases, no xscaling. NVIDIA's repos also ship a `.nemo` and a q8_0 GGUF. That GGUF (`general.architecture = asr`) uses NeMo tensor names and the NeMo config as `asr.*` metadata, unlike audio.cpp's `audiocpp` GGUF from Run 16.

Encoder against Parakeet's:
- **Causal subsampling:** convs padded `(k-1, s-1)` in time and frequency (128 -> 65 -> 33 -> 17 bins); names `conv_in` and `layers.N.{depthwise,pointwise}_conv`.
- **Causal depthwise conv:** padded `k-1` on the left only, followed by a LayerNorm in place of the folded BatchNorm.
- **Chunked-limited attention:** chunk = lookahead + 1, and a query sees its own chunk plus `left // chunk` chunks behind.
- **Attention over a block plan:** each 512-row query block reads only its key window, the rel-shift sliced as `T - b + c`, with an additive mask. Full attention is the same plan with every key and no mask. That makes attention linear in length for these models.
- **The mel is not normalized.**

Nemotron-3.5 joins a one-hot language prompt (128 slots) to every encoder frame, through `prompt_projector` (linear, ReLU, linear), before the encoder projector:
- The request's `language` maps through `processor_config.json`'s `prompt_dictionary`; a language outside it is refused per request.
- With no language, the model emits a `<xx-XX>` tag (an added special token, stripped from the text), which is reported as the transcript's language.

`scripts/parakeet_parity.sh` gains both classes. For 3.5 it compares the prompt-fused encoder output under the auto prompt. Results, CPU and CUDA:

| model | clips | features / encoder cos | emissions | text |
|---|---|---|---|---|
| en-0.6b | 3 LibriSpeech, 97.6 s meeting, 7 s synthetic Spanish | 1.000000 / 1.000000 | 694/694 (CPU), 731/731 (CUDA) | all match |
| 3.5-0.6b | 2-3 LibriSpeech, synthetic Spanish and French, meeting | 1.000000 / 1.000000 | 953/953 (CPU), 892/892 (CUDA) | all match |
| parakeet-tdt-0.6b-v3 (regression) | the 4 CUDA clips | 1.000000 / 1.000000 | 695/695 | all match |

In auto mode, 3.5 identifies the synthetic clips as `es-ES`, `fr-FR` and `en-US` (the last is British-voiced English).

Speed: the 97.6 s clip takes about 2.0 s on CUDA, about 16 s (en) and 21 s (3.5) on CPU in a dev build.

Tests:
- The tiny Parakeet set gains both streaming variants (window 5, chunk 2) across every format and the repeat test.
- A 3.5 language test: the named language is reported, it changes the transcript, and an unknown one is refused.
- A unit test pins the query-block windows and masks, including a block past the first.

Left for the next PR:
- true chunked streaming (K/V and conv caches, an incremental API);
- `.nemo` and NVIDIA `asr` GGUF loading for the whole family;
- the hybrid TDT/CTC checkpoint;
- beam search.

## Run 18 - 2026-10-10

**Review fixes.**
- **The 24-minute cap stays for streaming models.** Attention is linear now, but the causal subsampling's first stage alone holds a `(1, 256, T_mel/2, 65)` activation: about 12 GB per hour in F32. A 64 MiB compressed upload holds hours of audio, so uncapped it could take a CPU server out through the OOM killer. Longer audio goes through the VAD long-form path, as for Parakeet.
- **`language: auto`** (the 3.5 default prompt) is identify mode: the response reports the identified tag, not "auto".
- **A streaming encoder config without `sliding_window` or `default_num_lookahead_tokens` is an error.** It used to fall back silently to full attention over causal convs.
- **Unit test:** an unaligned case, a chunk of 3 that neither the 512-row block nor the 5-frame left context divides. The real English model's chunk of 14 does not divide 512 either.
- **Follow-up, not done:** a chunked encoder reads only offsets within about +/-(512 + left + chunk), yet the full `(2T - 1, hidden)` position table is still projected in every layer.

## Run 19 - 2026-10-10

**Parakeet family from `.nemo`.**

The `.nemo` configs come from ranged reads of the archives' first 1.5 MB (`model_config.yaml` follows the tokenizer members):

| checkpoint | class | encoder | head |
|---|---|---|---|
| `parakeet-tdt-0.6b-v3` | `EncDecRNNTBPEModel` | per_feature mel, `use_bias: false` | TDT (`decoding.model_type: tdt`, durations in `model_defaults`) |
| `stt_en_fastconformer_hybrid_large_pc` | `EncDecHybridRNNTCTCBPEModel` | 17 x d512, 80 mels, xscaling, biases | RNN-T plus `ctc_decoder`; only ships as a `.nemo` |
| `nemotron-speech-streaming-en-0.6b` | `EncDecRNNTBPEModel` | `chunked_limited`, `causal_downsampling`, `conv_context_size: causal`, `layer_norm` | RNN-T; `att_context_size` lists `[70,13] [70,6] [70,1] [70,0]` (first is the default) |
| `nemotron-3.5-asr-streaming-0.6b` | `EncDecRNNTBPEModelWithPrompt` | | prompt weights not mapped: refused, pointing to the transformers layout |

Mapping:
- **Encoder renames:** the Sortformer table, plus the streaming pre-encode. NeMo keeps dw_striding's `conv.{0,2,3,5,6}` indices there, while transformers names them `conv_in` and `layers.N.{depthwise,pointwise}_conv`.
- **Head renames:**
  - `decoder.prediction.{embed,dec_rnn.lstm}` -> `decoder.{embedding,lstm}`
  - `joint.pred` -> `decoder.decoder_projector`
  - `joint.enc` -> `encoder_projector`
  - `joint.joint_net.2` -> `joint.head`
  - `(ctc_)decoder.decoder_layers.0` -> `ctc_head`
- **Vocabulary:** NeMo's `vocab_size` plus the blank. CTC models read `num_classes`.
- **Tokenizer:** the archive's SentencePiece `tokenizer.model` is read by a minimal protobuf walk (field 1 = pieces, field 1 within = text). A decode-only BPE is built over the pieces: Metaspace decoder, byte fallback, `<...>` pieces special. That is transformers' conversion: its added tokens are the 264 angled pieces (special) plus the 10 digits (not special).
- **Weights:** read in place through `PthTensors::in_range`, as for Sortformer.

`scripts/parakeet_parity.sh <hf dir> <wavs> --cpu --nemo <the repo's .nemo>` keeps the transformers reference and loads ours from the `.nemo`, on parakeet-tdt-0.6b-v3:
```
ok librispeech clean 0000: features cos 1.000000, encoder cos 1.000000, emissions 15/15, text matches
ok librispeech other 0001: 1.000000 / 1.000000, 17/17, text matches
ok synthetic Spanish:      1.000000 / 1.000000, 44/44, text matches
ok 97.6 s meeting:         1.000000 / 1.000000, 619/619, text matches
```
The tokenizer built from the SentencePiece pieces decodes every transcript identically to `tokenizer.json`.

## Run 20 - 2026-10-10

**`.nemo` recognisers through the engine; hybrid against NeMo itself.**

`scripts/nemo_asr_parity.sh <.nemo> <wavs> [--cpu]` (with `NEMO_PYTHON` set to the Vernacula NeMo 2.7.1 venv) handles checkpoints with no transformers port. It dumps NeMo's preprocessor mel, its `encoder()` output, and its greedy RNN-T hypothesis: `rnnt_decoder_predictions_tensor(..., return_hypotheses=True)`, whose `y_sequence` and `timestamp` give tokens and frames. The parity example replays them, loading ours with `--nemo`.

`nvidia/stt_en_fastconformer_hybrid_large_pc`: 17 x d512, 80 mels, xscaling, biases, RNN-T + CTC, decoded with its RNN-T head:

| clip | CPU | CUDA |
|---|---|---|
| LibriSpeech clean 0000 | 1.000000 / 1.000000, 18/18, text matches | same |
| LibriSpeech other 0000 | 1.000000 / 1.000000, 16/16 | - |
| LibriSpeech other 0001 | 1.000000 / 1.000000, 19/19 | same |
| 97.6 s meeting | 1.000000 / 1.000000, 641/641 | same, 1.29 s |

(features cos / encoder cos, emissions with frames, text)

`nemotron-speech-streaming-en-0.6b.nemo` against the transformers reference, CUDA: 1.000000 / 1.000000; emissions 18/18, 21/21, 53/53 and 639/639; every text matches. This confirms the streaming pre-encode renames (`pre_encode.conv.{0,2,3,5,6}` -> `conv_in` and `layers.N.*`).

Engine:
- `TranscriptionLoader` takes the transformers layout where a `config.json` resolves, else the one `.nemo` (checked by `is_asr`: an `EncDec*` class over a Conformer with an RNN-T or CTC decoder).
- Auto-detection, for a local `.nemo`: Sortformer -> diarization, recogniser -> transcription. For a Hub repo with only a `.nemo`: a "sortformer" name -> diarization, any other -> transcription; the loader then checks the class.

Tests:
- The tiny Parakeet checkpoint is rebuilt as a `.nemo`: NeMo YAML, weights under NeMo names (`nemo_name`, the inverse rename), and a SentencePiece model of its pieces. For each of TDT, RNN-T, CTC and streaming, the `.nemo` and the transformers layout give byte-identical `verbose_json`.
- Unit tests: head and streaming renames round-trip; the auto-detect routes.

## Run 21 - 2026-10-10

**Review fixes for `.nemo` recognisers.**

- **Silently wrong output:** a regular (non-streaming) encoder with a finite `att_context_size` is refused. NeMo bands its attention there, and the loader ran full attention.
- **TDT:** detected as NeMo's decoder detects it, from a non-empty `decoding.durations`, not `decoding.model_type`. A multi-blank RNN-T (`big_blank_durations`) is refused.
- **Tokenizer:** the protobuf walk now keeps varint fields, so it reads each piece's type. Special = UNKNOWN or CONTROL. Data, parakeet-tdt-0.6b-v3's SentencePiece model against its `tokenizer.json` (read with the NeMo venv's `sentencepiece_model_pb2`):
  - 7918 normal, 273 user-defined and 1 unknown piece, `byte_fallback` false;
  - transformers marks 272 of the user-defined pieces non-special (`<|pnc|>` and the like stay text), and `<unk>` special.

  So the `<...>` rule from Run 19 would have stripped user-defined pieces that transformers keeps. A byte piece (byte fallback) is refused rather than stripped. A tokenizer other than `bpe` (SentencePiece) gets a clear error, and `is_asr` needs a `tokenizer:` section.
- **Config:** `normalize` must be `per_feature` or `NA`; the conv context causal or centred; `reduction` unset.
- **Loader:** takes the transformers layout only for a `config.json` that is a Parakeet config. When neither layout loads, the error carries both causes.
- **Hub auto-detect:** a lone `.nemo` routes to transcription only when its name or the repo's says parakeet, `stt_`, fastconformer, asr or speech. A TTS or speaker-ID `.nemo` is not downloaded only to be refused.

The three real checkpoints rerun on CUDA after the fixes, all exact:
- parakeet-tdt-0.6b-v3 `.nemo`: 15/15, 17/17, 619/619;
- streaming English `.nemo`: 18/18, 21/21, 639/639;
- the hybrid against NeMo: 18/18, 19/19, 641/641.

Environment: a reboot cleared the scratchpad under `/tmp`, taking the Kokoro-made synthetic clips and the CI memory wrapper. The wrapper was restored from the session record; the clips are not needed for these runs.

## Run 22 - 2026-10-10

**NVIDIA's `asr` GGUF.**

The format, from `parakeet-tdt-0.6b-v3.q8_0.gguf` (714 MB, 701 tensors, 47 keys):
- `general.architecture = asr`.
- The NeMo config as `asr.encoder.*`, `asr.preprocessor.*`, `asr.rnnt.*` and `asr.tdt.durations`. The streaming context is in `asr.encoder.{att_context_style, offline_left_ctx, offline_right_ctx, conv_context, conv_norm}`; there is no `causal_downsampling` key.
- The tokenizer as `asr.tokenizer.vocab`: pieces only, no piece types.
- Tensors keep NeMo names, apart from:
  - the LSTM, as `lstm.ih_l0.weight` against PyTorch's `lstm.weight_ih_l0`;
  - the conformer's pointwise convs, stored as matrices without the kernel axis.
- The converter's own `encoder.pos_enc.pe` and `preprocessor.fb` are skipped.
- Precision: Q8_0 for every large matrix, F16 for the convs and embedding, F32 for norms and biases.

Is the converter's quantization the only difference? Checked with the NeMo venv's `torch` and `gguf-py`, against each model's `.nemo`:
- **parakeet-tdt-0.6b-v3:** all 272 Q8_0 tensors are byte-identical to `gguf.quants.quantize(F32, Q8_0)` of the `.nemo` weights.
- **nemotron-speech-streaming-en-0.6b:** all 651 tensors match their source (Q8_0 byte-exact, F16 within F16 rounding).
- **Worst per-tensor dequantization error:** `joint.pred.weight` at 1.06% relative; the encoder matrices are around 0.6%.

So the loader is right by construction, and every difference from the F32 reference is the 8-bit weights.

`scripts/parakeet_parity.sh <hf dir> <wavs> --gguf <file> --min-cosine 0.95`, CUDA, against the transformers F32 reference:

| model | LibriSpeech clean 0000 | other 0000 | other 0001 | 97.6 s meeting |
|---|---|---|---|---|
| tdt-0.6b-v3 | 0.99991, text exact | 0.97150, text exact (emissions move from token 11) | 0.99982, text exact | 0.99987, WER 2.08% |
| streaming-en | 0.99975, text exact | 0.99991, text exact | 0.99984, text exact | 0.99954, WER 3.18% |

(encoder cosine, then the transcript)

The meeting clip's differences are 8-bit drift on the hardest audio:
- Parakeet: a proper-noun spelling, a British/American spelling, and a contraction written out (two words);
- the streaming model: one casing change, one homophone swap, and a few words gained or lost in the deliberately overlapped crosstalk.

So GGUF mode passes on cosine plus a WER bound (5% by default) rather than identical emissions.

Engine:
- `TranscriptionLoader` tries transformers, then `.nemo` (its F32 weights), then `asr` GGUF.
- Auto-detection takes a local `asr` GGUF.

Tokenizer rule refined: transformers' specials are `<unk>` and `<pad>` (the pad token it registers), plus control pieces (3.5's language tags); `<blank>` lies past the vocabulary. So `<pad>` is special by name in both readers. The GGUF reader marks `<unk>` and `<pad>`, having no types. The tiny equivalence test caught this: an F32 `asr` GGUF of the tiny streaming model emitted `<pad>` into its text.

Test: an F32 `asr` GGUF of the tiny checkpoint (converter names, flattened pointwise convs, `asr.*` metadata) transcribes byte-identically to the transformers layout, for TDT, RNN-T and streaming.

Follow-up, raised by the user: a better quantization, keeping the sensitive tensors at higher precision. The plan:
- a per-tensor precision policy in our own `asr` GGUF writer;
- a sensitivity sweep: one tensor group at F32 at a time, the rest Q8_0, measuring encoder cosine and WER against F32.
- `joint.pred` (largest error, tiny) is the first candidate.

## Run 23 - 2026-10-10

**Review fixes for the `asr` GGUF.**

Confirmed against both real files:
- **Value types:** `u32` ints, `i32` contexts, `f32` floats, `bool` flags, an `i32` durations array.
- **Head sizes:** the TDT joint has 8198 rows = 8193 + 5 durations. The vocab includes the blank, `blank_id = vocab - 1`, and the piece lists hold `vocab - 1`.
- **Streaming context:** `offline_*` is 70/13, the `.nemo`'s first `att_context_size` pair; `train_*` is unset (0/0) for Parakeet.
- **Shapes:** only `pointwise_conv{1,2}` are flattened.

Fixed:
- **Causal pre-encode:** inferred from `encoder.pre_encode.out.weight`'s input width (channels x bins: 4352 causal against 4096 centred for 128 mels), not assumed from the attention style. A mismatched file now fails the `parakeet()` checks instead of a shape mismatch.
- **Defaults:** `pos_emb_max_len`, `preemph` and `max_symbols_per_step` default as on the `.nemo` path, and the contexts accept any integer type.
- **Loader:** a `.gguf` path loads as the GGUF directly, so its errors are its own.
- **Hub auto-detect:** a repo whose ASR-named checkpoint is a `.gguf` now counts too.
- **Parity tooling:** `--nemo` and `--gguf` conflict, and the script forwards `--max-wer`.
- **This log:** Run 22 described the clip's differing words concretely; they are now abstracted.

Both GGUFs rerun on CUDA: identical to Run 22.

Left: one Hub repo holding several `asr` GGUFs (q8_0 and f16) needs the file path. NVIDIA's repos ship transformers files, which win, so their GGUF loads by its path.

## Run 24 - 2026-10-10

**A mixed-precision `asr` GGUF: the writer, and a sensitivity sweep.** Raised by the user: keep the sensitive tensors finer than Q8_0.

The writer: `parakeet::write_asr_gguf(nemo, out, policy)`, driven by `examples/asr_quantize.rs`, writes NVIDIA's layout from a `.nemo`:
- the same `asr.*` keys and value types;
- NeMo names with the converter's LSTM naming, and flattened pointwise convs;
- NVIDIA's precision defaults: norms, biases and `pos_bias_*` F32; convs and the embedding F16;
- the matrices in `--matrices`, unless a `--keep fragment=type` names them. A row that doesn't fill whole blocks drops to Q8_0, then F16.

`load_nemo` was split so the writer reads the same parsed config (`read_recogniser`).

Check, `--matrices q8_0` on parakeet-tdt-0.6b-v3: all 699 tensors are byte-identical to NVIDIA's `q8_0.gguf` (272 Q8_0, 30 F16, 397 F32). The file is 642 MiB, against NVIDIA's 681 MiB, which adds its own `pos_enc.pe` and `preprocessor.fb` tables. The NVIDIA-only keys (`pos_emb_max_len`, train contexts, frontend flags, joint and embed dims, `file_type`) are now written too.

The sweep:
- **Clips:** 8 English clips, 3 LibriSpeech plus the 97.6 s demo, two 300 s AMI and two 300 s VoxConverse.
- **Reference:** transformers F32 once, kept with `parakeet_parity.sh --keep-dumps`.
- **Metric:** total WER (edits over reference words, all clips), plus encoder cosine min and mean.
- **Policies:** uniform Q8_0, then one group at a time at F16 over Q8_0 (attention, position projection, feed-forward, conv pointwise, pre-encode out, LSTM, joint, `joint.pred`, first and last layer), then uniform F16, Q6_K, Q5_K and Q4_K for size.

## Run 25 - 2026-10-10

**Sweep 1, parakeet-tdt-0.6b-v3.** One group at F16, the rest Q8_0, scored against transformers F32 on the 8 clips (2928 reference words):

| policy | size | WER | encoder cos min / mean |
|---|---|---|---|
| Q8_0 (= NVIDIA's file) | 642 MiB | 1.195% | 0.97150 / 0.99609 |
| attention q,k,v,o,pos F16 | 754 | 1.400% | 0.96871 / 0.99581 |
| position projection F16 | 664 | 1.195% | 0.97214 / 0.99618 |
| **feed-forward F16** | 1002 | **0.478%** | **0.99609 / 0.99935** |
| conv pointwise F16 | 709 | 1.298% | 0.98962 / 0.99837 |
| pre-encode out F16 | 646 | 1.195% | 0.97437 / 0.99645 |
| LSTM F16 | 648 | 1.195% | 0.97150 / 0.99609 |
| joint F16 | 647 | 11.100% | (encoder unchanged) |
| `joint.pred` F16 | 642 | 18.955% | (encoder unchanged) |
| layer 0 F16 | 664 | 1.195% | 0.97426 / 0.99639 |
| layer 23 F16 | 664 | 1.161% | 0.97149 / 0.99611 |
| F16 everywhere | 1197 | 0.000% | 0.99988 / 0.99998 |
| Q6_K | 501 | 4.850% | 0.96997 / 0.99201 |
| Q5_K | 424 | 16.325% | 0.95658 / 0.98622 |
| Q4_K | 351 | 20.014% | 0.94742 / 0.97047 |

**The joint rows are not a precision effect.**
- With `joint.pred` F16, two 300 s clips (AMI b, VoxConverse a) lose half their emissions. On AMI b, 204 words are dropped in one stretch before decoding resumes: TDT's blank attractor, where the predictor reaches a state that scores blank on every frame.
- Is it ours? In transformers, with the F32 model's matrices rounded through Q8_0 as NVIDIA's converter stores them (2-D weights and pointwise convs) and `decoder_projector` through F16 instead:

  | transformers configuration | words on AMI b (of 420) |
  |---|---|
  | all Q8_0 | 419 |
  | all Q8_0, projector F16 | 214 |
  | projector F16 alone | 420 |

  This reproduces our 214 words exactly. Our implementation matches the reference even in this failure; the attractor is the model's.
- So WER on long unsegmented clips is chaotic in the decoder's weights, and the decoder rows say nothing about precision. The encoder cosine is the smooth measure of what each group costs.

Reading the encoder:
- The feed-forward matrices carry most of the 8-bit error (min cos 0.9715 -> 0.9961, mean 0.9961 -> 0.9994), at +360 MiB. They are 2 x 2 x 4M parameters per layer, the bulk of the model.
- The other groups barely move it.
- Uniform K-quants fall off fast.

Next round:
- split the feed-forward (macaron halves 1 and 2, `linear1` and `linear2`);
- keep the small decoder at F16 to take the chaotic path out of comparisons;
- try K-quant bases with the sensitive matrices at Q8_0, for a smaller file.

## Run 26 - 2026-10-10

**Sweep 2 (full clips), and sweep 3 (20 s pieces).**

Sweep 2, on the full 8 clips: WER was chaos again. A full-precision decoder alone read 11.1%, and every K-quant base 13-16%, whatever its encoder quality. The 300 s clips fall into the TDT blank attractor depending on exact weights (Run 25). The encoder cosine localized the error:

| policy | size | encoder cos min / mean |
|---|---|---|
| Q8_0 | 642 MiB | 0.97150 / 0.99609 |
| `feed_forward1` F16 | 833 | 0.99860 / 0.99950 |
| `feed_forward2` F16 | 833 | 0.96874 / 0.99581 |
| `linear1` (both halves' up-projections) F16 | 833 | 0.99895 / 0.99959 |
| `linear2` F16 | 833 | 0.96857 / 0.99578 |

So the encoder's 8-bit error sits in the up-projections, mostly the first macaron half's.

A WER protocol that measures precision rather than chaos: the 8 clips cut into 68 pieces of 20 s at fixed points, with a transformers F32 reference per piece. Production transcription windows long audio (VAD, at most 2 min) anyway. A deletion now stays within its piece.

Sweep 3, on 68 pieces (3200 reference words):

| policy | size | WER (edits) | encoder cos min / mean |
|---|---|---|---|
| Q8_0 (= NVIDIA's) | 642 MiB | 0.781% (25) | 0.97150 / 0.99865 |
| **decoder + joint F16** | **653** | **0.281% (9)** | 0.97150 / 0.99865 |
| `linear1` F16 | 822 | 1.156% (37) | 0.99184 / 0.99968 |
| `feed_forward1.linear1` F16 | 732 | 1.062% (34) | 0.99163 / 0.99953 |
| `feed_forward1` F16 | 822 | 0.906% (29) | 0.99155 / 0.99955 |
| feed-forward F16 | 1002 | 1.219% (39) | 0.99165 / 0.99958 |
| `linear1` + decoder F16 | 833 | 0.344% (11) | 0.99184 / 0.99968 |
| Q6_K, `linear1` Q8_0 | 548 | 1.031% (33) | 0.93801 / 0.99672 |
| Q6_K, `linear1` F16 | 728 | 0.969% (31) | 0.94193 / 0.99687 |
| Q6_K, feed-forward Q8_0 | 594 | 0.969% (31) | 0.95568 / 0.99727 |
| Q5_K, feed-forward Q8_0 | 568 | 3.594% (115) | 0.91117 / 0.99462 |
| Q6_K | 501 | 2.875% (92) | 0.93358 / 0.99510 |
| F16 everywhere | 1197 | 0.000% (0) | 0.99878 / 0.99998 |

Reading:
- **The decoder sets WER.** Keeping the predictor (embedding already F16, LSTM, projector) and joint at F16 costs 11.5 MiB (1.8%) and removes about two thirds of the 8-bit word errors. The joint's logits decide the close calls between tokens, so its quantization error lands straight in the transcript.
- **Encoder precision barely moves WER at this level.** `linear1` F16 lifts the cosine (min 0.9715 -> 0.9918) without fewer edits. The counts here (9-39) are small: a few edits either way is noise. The decoder effect (25 -> 9, and 37 -> 11 with `linear1`) holds across both bases.
- **Q6_K with the feed-forward at Q8_0** is smaller than NVIDIA's file (594 MiB) at near its WER. Plain K-quants are worse.

Next: K-quant bases with an F16 decoder, and the same policies on the streaming English model (its own reference over the pieces), to see whether the decoder effect generalizes.

## Run 27 - 2026-10-10

**Sweep 4, and the streaming model: the decoder effect holds.**

parakeet-tdt-0.6b-v3, 68 pieces (3200 words):

| policy | size | WER (edits) |
|---|---|---|
| Q6_K, decoder F16 | 513 MiB | 2.938% (94) |
| **Q6_K, feed-forward Q8_0, decoder F16** | **606** | **0.531% (17)** |
| Q6_K, `linear1` Q8_0, decoder F16 | 559 | 0.688% (22) |
| Q5_K, feed-forward Q8_0, decoder F16 | 580 | 3.375% (108) |
| Q4_K, feed-forward Q8_0, decoder F16 | 555 | 8.812% (282) |

nemotron-speech-streaming-en-0.6b, same 68 pieces, its own transformers reference (3188 words):

| policy | size | WER (edits) | encoder cos min / mean |
|---|---|---|---|
| Q8_0 (= NVIDIA's) | 628 MiB | 1.694% (54) | 0.99868 / 0.99970 |
| **decoder F16** | **636** | **1.098% (35)** | 0.99868 / 0.99970 |
| Q6_K, feed-forward Q8_0, decoder F16 | 588 | 1.192% (38) | 0.99703 / 0.99889 |
| F16 everywhere | 1180 | 0.000% (0) | 0.99987 / 0.99998 |

Conclusions:
- **Decoder at F16.** Keeping the decoder (predictor LSTM and projector) and joint at F16 is the best use of bytes: +1.2 to 1.8% in size, removing a third (streaming) to two thirds (Parakeet) of the 8-bit word errors.
- **Feed-forward at Q8_0.** Of the encoder, it is the part that needs 8 bits: with it at Q8_0 and the rest at Q6_K, both models come out smaller than NVIDIA's files (-6%) and more accurate. Below Q6_K the rest of the encoder falls apart.
- **Q8_0 and below.** The streaming model's encoder is far less sensitive to Q8_0 than Parakeet's (min cos 0.9987 against 0.9715); its words are lost in the decoder.

The presets on `QuantPolicy`, exposed as `asr_quantize --preset`:
- `q8_0`: uniform, NVIDIA's file byte for byte;
- `q8_0-decoder`: `q8_0` with the decoder and joint at F16;
- `q6_k-mix`: Q6_K, feed-forward Q8_0, decoder F16. Its file reproduces sweep 4's 605.7 MiB.

Tests:
- a unit test pins the per-tensor precision rules, including the fallback for rows that fill no K-block (640-wide to Q8_0);
- a tiny `.nemo` written as an `asr` GGUF (F32 matrices, F16 convs) transcribes byte-identically to the `.nemo`, for TDT, RNN-T and streaming.

Caveat: the edit counts are small (9 to 54 per policy). The decoder effect is consistent across both models and every base, but single-digit differences between neighbouring policies are within noise.

## Run 28 - 2026-10-10

**Review fixes for the writer and the sweep tooling.**

- **`parakeet_parity.sh --keep-dumps`:** reuse was keyed on the dump count, so the same clips under another model would have been scored against the first model's reference.
  - This sweep's results stand: each model and clip set had its own directory (`v3_ref`, `v3_pieces_ref`, `stream_pieces_ref`), and the two subset copies matched their clip lists.
  - The directory now records a manifest (model directory and ordered clip list). It is reused only on an exact match and cleared and refilled otherwise.
- **`general.file_type`:** an exhaustive llama.cpp mapping (BF16 32, Q4_1 3, Q5_1 9, Q2_K 10, and so on).
- **`asr.rnnt.joint_dim`:** read off `joint.pred.weight` (NeMo's own `joint_hidden`), not taken as the predictor width. The missing-predictor case is an error instead of a zero default.
- **Writer rules:** rank-0 tensors go to F32 (no underflow); the decoder fragment is `decoder.prediction.`, so a hybrid's `ctc_decoder` cannot match it.
- **Round-trip test:** also loads a `q6_k_mix` file.

Rechecked: `--preset q8_0` is still byte-identical to NVIDIA's file for all 699 tensors. Every key NVIDIA writes is now present (`joint_dim` 640).

## Run 29 - 2026-10-10

**Beam search: a port of NeMo's default algorithm, and what beam search is worth.**

Ported NeMo's `default` beam searches:
- `BeamRNNTInfer.default_beam_search`: scores as Python floats;
- `BeamTDTInfer.default_beam_search`: scores in float32, `(duration, token)` pairs taken from torch's `cartesian_prod` order, duplicate merging by log-add, a lone zero duration replaced by the shortest non-zero one.

Both rank the n-best by score per symbol (`score_norm`). The beam size reaches them as `TranscriptionOptions.beam_size`, exposed as the `beam_size` request field (1 to 16).

Parity, `scripts/nemo_asr_parity.sh <.nemo> <wavs> --beam 4` against NeMo 2.7.1 on CPU, tokens and text:
- **hybrid RNN-T:**
  - 3 LibriSpeech clips exact;
  - 97.6 s demo exact (639 tokens; greedy gives 641);
  - 300 s AMI exact (998; greedy gives 1023).

  So the beam is exercised and matches.
- **parakeet-tdt-0.6b-v3 TDT:** exact as well (16, 505 and 2 tokens). But 2 tokens on 300 s, against 1207 greedy: NeMo's own default TDT beam collapses on long audio, and we reproduce it.

How NeMo's TDT strategies behave on long audio (beam 4, words):

| strategy | 300 s AMI | 97.6 s demo |
|---|---|---|
| greedy | 681 | 289 |
| `beam` / `default` | 1 | 233 |
| `maes` | 161 | 237 |
| `malsd_batch` | 633 | 279 |

Quality against ground truth: `libri_dummy.parquet` (73 LibriSpeech utterances with references), lowercase, punctuation stripped, jiwer WER, in NeMo:

| model | greedy | default beam | `malsd_batch` |
|---|---|---|---|
| hybrid (RNN-T) | 2.26% | 2.35% | - |
| parakeet-tdt-0.6b-v3 (TDT) | 3.83% | 3.65% | 3.48% |

Conclusions:
- Without an LM, beam search buys little on these models.
- The RNN-T default beam is not better than greedy here.
- For TDT, only `malsd_batch` both helps (9% relative) and survives long audio. The default and `maes` searches drop speech.
- So shipping the ported TDT default beam would be harmful.

Options:
1. Ship the RNN-T beam and refuse TDT beam.
2. Port `malsd_batch` for TDT.
3. Drop beam search.

## Run 30 - 2026-10-10

**Beam search: MALSD for TDT, the default search for RNN-T.** The user chose to port `malsd_batch` for TDT, keep the RNN-T default beam, and refuse the default TDT beam.

The MALSD port (`Transducer::malsd`) follows NeMo's `ModifiedALSDBatchedTDTComputer.modified_alsd_torch` without fusion models.
- `beam` slots step together, each at its own frame.
- Per step, each active slot scores its top `beam` (label, duration) pairs over all labels with blank, flattened label-major. A blank with duration 0 is forced to the shortest non-zero duration.
- A finished slot carries over at its score in its first candidate.
- A slot past `max_symbols` at one frame gets only a forced blank, scored with the best duration.
- All `beam x beam` candidates are pruned to `beam`, and slots on one path recombine by log-sum-exp, the first best keeping the score. A path is the same transcript hash, last label, token count and next frame.
- The predictor runs batched over the slots, keeping a source's state where the label is blank.
- Scores are float32, and the best slot is chosen by score over (tokens + 1).
- NeMo uses the duration index as the duration; with v3's durations `[0..4]` that is the same, and ours uses the value.

The predictor now runs `n` rows at once (`start`/`advance` over a token slice). The RNN-T default beam and greedy decoding use one row.

Parity, `nemo_asr_parity.sh --beam 4` (MALSD for TDT, default for RNN-T), NeMo 2.7.1, tokens and text:
- **parakeet-tdt-0.6b-v3:** exact on 3 LibriSpeech clips (15/15, 16/16, 17/17), on the 97.6 s demo (612/612) and on the 300 s AMI clip (1135/1135; greedy gives 1207, NeMo's default TDT beam gives 2).
- **hybrid RNN-T, rerun after the predictor change:** exact (16/16, 639/639, 998/998).

Speed, CUDA: the 300 s clip takes 42 s with MALSD (beam 4), against about 6 s greedy. One host round trip per step: joint logits to the host for the top-k, then the predictor. A follow-up could keep the top-k on the device.

API:
- `beam_size` on the transcription request (form field, integer kind), 1 to 16, validated in inference-api;
- carried in `TranscriptionOptions`;
- a CTC head refuses a beam over 1 per request.

Tests:
- an engine test: beam 4 on the tiny TDT, RNN-T and streaming heads decodes and repeats exactly; 17 is refused; CTC is refused;
- a unit test pins recombination: log-sum-exp into the first best, other frames and transcripts kept apart.

## Run 31 - 2026-10-10

**Review fixes for beam search.**
- MALSD returns no emissions for an empty encoder output, instead of underflowing.
- RNN-T's state cache is trimmed after each frame to sequences at least as long as the shortest kept hypothesis. Hypotheses only lengthen, so no evicted state is asked for again, and the output is unchanged. On 300 s it no longer holds every state ever expanded.
- A beam on a CTC head is refused before the encoder runs.
- MALSD's transcript hash is NeMo's `hash_text` (`h * 6364136223846793005 + 1 + label`, wrapping).
- The final pick takes the first best slot, as Python's `max` does.
- Constants hoisted, comments one line, and kept hypotheses cloned only on a break.
- Beam parity now compares frames too: our frame plus span for TDT against MALSD's stamp, and the frame for RNN-T against the default beam's. Exact on both models:
  - TDT: 16/16, 612/612, 1135/1135;
  - RNN-T: 16/16, 639/639, 998/998.

The user's read of Runs 29-30: greedy performs best, and segmentation by VAD or diarization plus greedy gives reasonable WER. So greedy stays the default, `beam_size` stays opt-in, and MALSD's speed (42 s per 300 s on CUDA) is not worth optimizing.

A follow-up worth doing: speaker-attributed transcription, with diarization turns segmenting greedy recognition.
