# Kokoro investigation

Goal (#390): Kokoro-82M text-to-speech in `inference-models-speech`, from the PyTorch release and from GGUF, with voice packs per request; then a Rust port of vernacula-phonemizer as its G2P.

## Run 1 - 2026-10-09 09:22

Question: what are we porting, and how is it laid out?

Commands: read the upstream reference (`pip install --user --break-system-packages --no-deps kokoro` 0.9.4 plus `loguru`; its `__init__` imports misaki, so reference scripts import `kokoro.model` directly); open the `.pth` with torch; parse the multilingual GGUF header by hand; count the phonemizer sources.

Findings:
- **Checkpoint.** `kokoro-v1_0.pth` is a dict of five state dicts (`bert`, `bert_encoder`, `predictor`, `decoder`, `text_encoder`), every key prefixed `module.`, all F32, 81.76M parameters.
  - inference-tensor's `PthTensors::new(path, Some(key))` reads one sub-dict, so the release loads without conversion.
  - Convolutions are weight-normed (`weight_g`/`weight_v`); they are folded at load as Dia's DAC does.
- **Voices.** Each `voices/*.pt` is a (510, 1, 256) F32 tensor. The pipeline uses row `len(phonemes) - 1`. Columns 0..128 are the decoder style, 128..256 the prosody style.
- **Architecture** (`kokoro/model.py`, `modules.py`, `istftnet.py`, ~750 lines):
  - an ALBERT (12 layers sharing one layer's weights, `gelu_new`, post-LN, eps 1e-12) and a 768-to-512 projection;
  - a prosody predictor: 3 x (BiLSTM + AdaLayerNorm), then a BiLSTM with a 50-bin duration head (sigmoid sum / speed, rounded, min 1), then the shared BiLSTM with F0/N AdaIN residual stacks;
  - a text encoder: embedding, 3 x (conv k5, LayerNorm, LeakyReLU 0.2), BiLSTM;
  - an iSTFTNet decoder:
    - AdaIN res blocks at 1090 channels;
    - a generator with two transposed-conv upsamples (x10, x6) and Snake res blocks;
    - an NSF harmonic source with 9 harmonics at 24 kHz;
    - output by inverse STFT (n_fft 20, hop 5).
- **Randomness.** The source module draws a random initial phase per harmonic and Gaussian noise per sample. Parity therefore has to inject the reference's draws, and our model needs a seed or an injected-noise path.
- **Missing ops.** inference-tensor has no LSTM, STFT or reflection pad. Dia's DAC already has weight-normed convs and a Snake (with a 1e-9 epsilon Kokoro lacks); share those rather than copy them.
- **Multilingual GGUF on disk** (`kokoro-82m-q8_0.gguf`, 932 MB): audio.cpp's format, architecture `kokoro_tts`.
  - Tensors are numbered `kokoro.N`, with names in `kokoro.tensor_names` and shapes in `kokoro.tensor_shape.*`.
  - Types: 360 F32, 111 Q8_0, 77 BF16; weight-norm pairs are kept unfolded.
  - It embeds 781 MB of files: config.json, `voices/*.bin` (the 54 packs), espeak-ng-data, unidic and g2p data.
  - Loading it means reading the embedded voices and remapping the names.
- **Phonemizer** (vernacula-phonemizer): about 127k lines of TypeScript over 180 language directories, 145 MB of data, and a C# port of about 103k lines. That is a multi-PR port on its own.

Plan:
1. The model from the `.pth` release, driven by Kokoro phoneme strings. Parity against the reference with its noise injected, via `scripts/kokoro_parity.sh`.
2. Engine wiring: `SpeechLoaderType::Kokoro`, a per-request voice, a `phonemes` request path, a tiny-checkpoint engine test, then GGUF (audio.cpp's file).
3. The phonemizer port: core and English first, mapped to Kokoro's vocabulary, then the other languages.

## Run 2 - 2026-10-09 09:38

Question: does the port match the reference on fixed phoneme input?

Command: `scripts/kokoro_parity.sh /mnt/data/models/Kokoro-82M-source af_heart 1e-3 --cpu`. The reference records its source noise (`torch.randn_like` with 9 harmonics) and dumps ids, style, noise, durations and audio per case. `examples/kokoro_parity.rs` replays them. There are four invented phoneme strings: short, sentence, question, and one at speed 1.3.

Finding:
- The first try failed with `index-select only supports contiguous tensors`: `d` comes out of a cat. Fixed with `.contiguous()`.
- Then durations matched exactly in all four cases, so the ALBERT, the LSTMs, AdaLayerNorm and the duration head are right.
- The waveform did not match: SNR 23-25 dB, max |diff| 9-13% of peak.

Next: bisect the generator.

## Run 3 - 2026-10-09 09:38

Question: where in the generator does the waveform diverge?

Method:
- Dump the reference's generator inputs (`gen_x`, style, F0), its source signal, its STFT magnitude and phase, the `conv_post` output and the audio.
- In a throwaway test, compare each host DSP step and the network fed each stage of the reference's data. That test was deleted afterwards.

Finding:
- `istft` on the reference `conv_post` output: max diff 3e-7. STFT magnitude: 1.2e-7.
- STFT phase was off by exactly 2pi in places. The DC and Nyquist bins of a real FFT are exactly real, but the direct DFT leaves +-1e-17 imaginary parts, so `atan2` returned -pi where torch has +pi. Forcing those to 0 lifted SNR to 34-43 dB.
- One flip remains, at bin 3 frame 0 (magnitude 0.046, angle at the +-pi cut). It is FFT rounding, not a bug.
- The network fed the reference's STFT is exact: 96.6 dB, max diff 7e-6.
- Our STFT of the reference source: 76 dB; with that frame-0 flip undone, 96.6 dB.
- So everything left is the harmonic source, max diff 2.5e-4 from the reference. Every step checks out:
  - the downsampled phase increment equals the per-frame value exactly (0.0 difference);
  - a double-accumulated cumsum equals torch's;
  - but the upsampled phase reaches about 1e5 rad, where an f32 ulp is about 0.008 rad, so lerp/FMA order and the SIMD `sinf` move `0.1 * sin(phase)` by up to about 4e-4.

Next: measure how close the reference is to itself.

## Run 4 - 2026-10-09 09:38

Question: what waveform agreement is achievable at all?

Command: the reference on torch CPU against torch CUDA, with identical seeded noise (`torch.randn_like` replaced by a CPU-seeded draw).

Finding: durations are equal, but CPU against CUDA gives only 25.0-27.4 dB, with max |diff| 10-13% of peak. Our port against reference CPU (34-44 dB) is closer than the reference is to itself.

Decision: the parity check requires exact durations and SNR >= 30 dB (`--min-snr`). The network's exactness was shown separately in Run 3.

Run on CUDA (`scripts/kokoro_parity.sh /mnt/data/models/Kokoro-82M-source`): pass.

| case | tokens | SNR (dB) | max diff / peak | time |
|---|---|---|---|---|
| short | 14 | 38.6 | 3.1% | 0.13 s for 1.65 s of audio |
| question | 32 | 44.0 | 1.4% | 0.17 s |
| sentence | 74 | 35.9 | 5.4% | 0.35 s for 4.4 s |
| fast (speed 1.3) | 41 | 34.2 | 5.1% | 0.28 s |

CPU (dev profile) takes 2.5-9 s for the same cases, which needs profiling.

## Run 5 - 2026-10-09 09:57

Question: why is CPU synthesis 6.5x slower than torch (8.1 s against 1.23 s on 8 threads for the 4.4 s sentence)?

Command:
- Stage timers in `synthesize_ids` and the generator; `perf` is unavailable because `perf_event_paranoid` blocks it without root.
- The torch reference timed on CPU.

Finding:
- About 80% of the time is the generator's convolutions. The last stage alone, 128 channels at 120 frames per F0 frame with kernels up to 11 and dilation up to 5, took 5.5 s.
- `inference-tensor`'s CPU `conv1d` builds a (l_out, c_in * k) im2col single-threaded, multiplies, then copies the (l_out, c_out) result back through a strided transpose.

Change:
- The column buffer is now (c_in * k, l_out), one row per (channel, tap), filled in parallel as strided slices.
- `kernel @ col` then lands in (c_out, l_out) directly, with no transpose copy.
- A new test, `cpu_conv1d_matches_a_direct_loop`, checks it against a direct loop (padding, stride, dilation, taps past both ends, transposed input, batch 2).

Finding 2:
- The new layout first failed its own test. The cause is a bug in the shared CPU matmul, not the conv.
- `MatMul` folds batches into one GEMM when the lhs is broadcast across the batch (`a_skip == 0`) by widening n to b * n. That is only valid for a single output row whose rhs columns are evenly spaced across batches.
- Its other fold, for a broadcast rhs, likewise assumed contiguous lhs rows.
- So any CPU batched matmul with a broadcast lhs (b > 1) and m > 1 returned wrong values.
- Fixed by gating both folds on their real conditions. `batched_matmul_with_broadcast_or_transposed_operands` fails on the old conditions ("case 0 batch 1: 2.75") and passes now.

Result: sentence 8.1 s to 3.25 s; short 2.5 s to 1.28 s. Parity unchanged (34-42 dB, durations exact). Still about 2.6x torch on CPU; remaining:
- the generator's last stage, 1.6 s;
- the predictor and text-encoder LSTMs plus ALBERT, about 0.6 s, per-step in the LSTMs.

A later pass can take those.

Side note: the session's machine froze during the full-workspace rebuild that followed the first `inference-tensor` edit. The previous boot's journal shows the compositor hanging and VS Code crashing at 09:46, with no OOM kill or GPU fault; the box has no swap. Rebuilds now run with 8 jobs under a watchdog that kills cargo below 10 GiB available. The same rebuilds since bottomed out at about 103 GiB available, so the cause is unconfirmed.

## Run 6 - 2026-10-09 10:24

Question: what does audio.cpp's Kokoro (`~/Programming/audio.cpp`), whose GGUF we have, tell us about the file format and about efficiency?

Method: read its writer `tools/prepare_kokoro_gguf.py`, its loader `src/models/kokoro_tts/assets.cpp`, and its predictor, decoder and session code.

GGUF format:
- Tensor i is `kokoro.i`. `kokoro.tensor_names[i]` is the PyTorch name with `module.` stripped (sorted). `kokoro.tensor_shape.<name>` holds the PyTorch-order shape as INT64.
- Types: F32 for anything with a dim of 1 or rank < 2. Otherwise BF16, and Q8_0 only for rank-2 tensors whose last dim divides by 32 (linears, LSTMs, embeddings). Conv kernels are not reshaped.
- Weight norm is stored raw. audio.cpp folds it with `sqrt(sum v^2 + 1e-12)`; torch has no epsilon, and we follow torch.
- Embedded files: `audiocpp.embedded_files.{names (sorted), offsets (n+1, from 0), data}`.
  - The packs are `voices/<id>.bin`: raw f32 [rows, 256], the `[rows, 1, 256]` source squeezed.
  - Also embedded: `voices.json` with rows and cols, `vocab.tsv`, the original `config.json`, and `g2p/*`.
- Loading it therefore needs:
  - the name remap;
  - BF16/Q8_0 dequantization to F32;
  - the embedded config and voices, read with `VoicePack::from_f32_le`.

Semantics that match ours:
- Style row = phoneme count - 1, with the decoder style first.
- Ids are `[0, ..., 0]`.
- Durations: `round(sum(sigmoid)/speed)`, clamped [1, max_dur].
- SineGen and the STFT run on the host. Its initial-phase draws are discarded; we found the same, that they never reach the output.
- Reflect pad 1 on the left in the last stage. Resblocks are averaged.

Efficiency ideas for later:
- The ups ConvTranspose (k = 2 * stride) becomes a precomputed "phase-shuffle" k=3 Conv1d with out * stride channels.
- The stride-2 depthwise pool becomes a dense diagonal; we already use two interleaved phases instead.
- The long frame-rate shared LSTM is unrolled in 64-frame blocks.
- im2col for the F32 convs is multithreaded with memcpy rows, like the conv1d rewrite in #397.
- The only CPU number it publishes is relative: 16 threads 18-23% faster than 8.

Divergences from us:
- It has no phoneme bypass. Its embedded model spec lists a `phonemes` option, but the code always runs its G2P, and over 510 phonemes is an error.
- It splits long text before G2P (240 codepoints, preferring sentence then clause breaks).
- It reseeds the noise per chunk.

## Run 7 - 2026-10-09 10:44

Question: does Kokoro work through the engine, the HTTP server and the error paths?

Wiring done:
- `SpeechLoaderType::Kokoro`, detected from `config.json` (Kokoro checked before Dia).
- An optional architecture on the speech selection; the CLI's `speech` mode now detects it instead of forcing Dia.
- Per-request `SpeechOptions` (voice, speed, phonemes, seed) carried on the sequence. The diffusion params slot became a `OneShotParams` enum.
- A `Pipeline::validate_speech_options` hook, so request errors are validation errors and not "The model failed".
- Weights from `.pth` or `model.safetensors`; voices from `.pt` or raw `.bin`.
- Long phoneme input packed into chunks of at most 510 characters at spaces.

Commands:
- The tiny-checkpoint test (`kokoro_tiny`). It uses random weights recorded through a plain `VarBuilder` and two raw voice packs.
- `inference serve -p 18777 speech -m /mnt/data/models/Kokoro-82M-source` (CUDA dev build) with curl.

Finding:
- Tiny test passes: seeds repeat and differ, the default voice, a blend, speed 0.5 is longer, chunking is longer, WAV = PCM + 44 header bytes, and the errors fire.
- On the first try the unknown-voice case returned the generic "The model failed to process the request." That is what led to the validation hook.
- The real model over HTTP: 200 `audio/wav; rate=24000; channels=1`, 3.7 s, RMS 0.046, peak 0.33.
- A missing `phonemes` returns a 400 `invalid_request_error` with the reason. An unknown voice is rejected with the list of the release's 54 voices.

Also fixed in the speech guide: the Dia Rust example called `generate_speech("...")`; the SDK method takes a `SpeechGenerationRequest`.

## Run 8 - 2026-10-09 10:52

Review fixes, then the same checks again.

Fixes:
- A local checkpoint without `voices/` now gets the loader's clear "needs a voice pack" error, not a bare OS error.
- A speech config for the other architecture fails at load.
- Dia refuses `phonemes`. It keeps ignoring `voice` and `speed`, which OpenAI clients always send.
- Error and schema text no longer name planned work.
- Duration divides by `speed` as torch does, instead of multiplying by its reciprocal.
- Two voice packs with one name are refused.
- New tests: `DepthwiseUp2` against the grouped transposed conv, plus config auto-detection through `ModelBuilder` on the tiny checkpoint, which gives the same audio as the named loader.

Not changed:
- `generation` still needs an explicit `arch`, since its fields are per-architecture; this is now documented on the field.
- The `.pth` and `.pt` paths stay covered only by `scripts/kokoro_parity.sh`, which needs the release.

Finding:
- 7/7 Kokoro tests pass.
- Parity on CUDA is unchanged: durations exact, SNR 34.2/44.0/35.9/38.6 dB.

## Run 9 - 2026-10-09 11:19

Question: can we load audio.cpp's multilingual `kokoro-82m-q8_0.gguf`, and how close does it sound to the release?

Change:
- The shared GGUF reader parsed every metadata array element into a 32-byte `Value`, so the 781 MB embedded-files blob would have cost about 25 GB of memory. Top-level arrays of fixed-size values longer than 1M elements are now located instead (`Content::large_arrays`) and read by element range (`read_large_array`).
- `kokoro::gguf` maps `kokoro.N` to its `kokoro.tensor_names` entry, dequantizes to F32 in the `kokoro.tensor_shape.*` shape, and reads only `config.json` and `voices/*.bin` from the embedded table.
- The speech loader takes a `.gguf` file, or a directory holding one GGUF and no `config.json`. The auto loader recognizes it when there is no config.

Commands:
- The parity example with `--gguf`.
- A temporary test comparing every GGUF tensor with the `.pth` tensor of the same name (deleted afterwards).
- Spectral SNR (STFT magnitudes, 512/128) added to the parity example, with the same metric on torch CPU against CUDA.
- `inference serve` in `speech` and auto modes on the GGUF.

Finding:
- Loading takes 1.0 s with 600 MB resident.
- Every tensor shape matches, and the worst relative error is 4.1e-3, all from Q8_0. The loader is exact up to the file's quantization.
- Quality (spectral SNR, dB):

| case | torch CPU vs CUDA | our F32 vs torch CPU | Q8_0 GGUF vs torch CPU |
|---|---|---|---|
| short | 37.7 | 45.8 | 34.8 |
| question | 43.7 | 51.2 | 24.1 |
| fast | 37.4 | 42.1 | 22.8 |
| sentence | 35.7 | 43.1 | durations differ: 945000 samples become 955800 |

- Waveform SNR for the GGUF is 3-24 dB; it is not meaningful under weight quantization because of phase drift.
- The GGUF therefore sounds different from the release in a measurable way, though it is the same voice. That is the file's quantization (audio.cpp's default package), not our loader.
- Serving: `serve speech -m <gguf>` and `serve -m <gguf>` (auto) both return 200 `audio/wav` with the `bf_emma` voice read from the GGUF, byte-identical to each other.
- Tiny test: an F32 audio.cpp-layout GGUF built from the tiny checkpoint speaks byte-identically to that checkpoint, through both the named and the auto loader.

Dead end: the first version of that test wrote the GGUF into the checkpoint directory before generating the reference audio. The "one GGUF in a directory" rule then loaded the GGUF on both sides, and the test passed trivially. Now a directory with `config.json` always loads the release files, and the test generates its reference first.

## Run 10 - 2026-10-09 11:28

Review fixes:
- `is_kokoro_gguf` peeks at the header and first key (`gguf_file::peek_architecture`) instead of parsing the whole file, so auto mode on a config-less text GGUF no longer reads all its strings first.
- A Kokoro GGUF given with a different explicit arch is refused by name.
- A missing numbered tensor names its PyTorch name.
- One shared `GGUF_EXTENSION` constant.

Test additions:
- The tiny GGUF now embeds a 1M + 1 byte filler file, so the Kokoro load goes through `read_large_array` and the filter that skips files the model doesn't read.
- A directory holding just the GGUF is auto-detected.

Finding:
- 8/8 tests pass (Kokoro, speech crate, gguf_file).
- The real Q8_0 GGUF in auto mode still detects through the peek (audio.cpp writes `general.architecture` first), and its output is byte-identical to the earlier run.

## Run 11 - 2026-10-09 11:44

Question: where does CPU synthesis time go now (sentence case: 3.25 s, against torch's 1.23 s on 8 threads)?

Command: temporary `eprintln!` stage timers (not committed); `kokoro_parity --cpu` on the four dumps.

Finding (sentence case, 4.4 s of audio, CPU dev profile, 8 threads):

| part | time |
|---|---|
| pre-decoder (ALBERT, BiLSTMs, predictor, text encoder) | 0.54 s |
| decoder AdaIN blocks | about 0.2 s |
| generator stage 0 | 0.75 s |
| generator stage 1 | 1.58 s |
| STFT + iSTFT | 0.16 s |

My first reading of the timers blamed the spectral head (0.87 s). That was wrong: the timer was cumulative, and the iSTFT itself is 73 ms.

Micro-benchmarks on one stage-1 activation (128 ch x 21240):
- conv k3: 20 ms; k7 d3: 44 ms; k11 d5: 66 ms. That totals about 1.2 s over stage 1's 24 convs.
- AdaIN + Snake as tensor ops: 15.5 ms, about 0.37 s over the stage.
- torch's conv k11 d5 on the same shape: 23 ms.

The GEMM is the limit:
- `gemm` 0.19 reaches 160-190 GFLOPS on 8 threads and 47-52 GFLOPS on 1 thread.
- torch (MKL) reaches 420 GFLOPS on 8 threads and 98 on 1.
- The i7-10700K's AVX2 FMA peak is about 150 GFLOPS per core.

Dead ends:
- A conv1d as one GEMM per tap over a padded input, with no im2col buffer: 61 ms against 56 ms for k11. im2col was not the bottleneck. Reverted.
- `codegen-units=1` for the gemm crates: no change (45-47 GFLOPS per core).

## Run 12 - 2026-10-09 11:53

Change, CPU only:
- AdaIN fused with its activation, Snake or leaky ReLU, as one `CustomOp1` pass per channel row. That replaces about 14 tensor passes.
- The LSTM recurrence runs as host loops: one batched input projection, then an axpy row update per step. The two directions run in parallel.

Result:

| case | before | after |
|---|---|---|
| sentence | 3.25 s | 2.57 s |
| question | 1.49 s | 1.13 s |
| short | 1.23 s | 1.02 s |

- Pre-decoder: 583 ms to 266 ms. Stage-1 residual blocks: 1.58 s to 1.33 s.
- Parity unchanged: durations exact, SNR 34-42 dB, spectral 42-51 dB.
- What remains is the convolution GEMM, at about 2.3x MKL's throughput gap.

## Run 13 - 2026-10-09 12:08

Question: does the CPU work leave CUDA untouched?

Command: `scripts/kokoro_parity.sh /mnt/data/models/Kokoro-82M-source` (CUDA).

Finding: no.
- The parity check failed, differently on every run: durations differed in 1-3 of the 4 cases, and once a case dropped to -2 dB.
- Cause: `rayon::join` ran the LSTM's two directions on two threads for every device. The forward and reverse tensor ops then raced on the one CUDA device, apparently through its single stream and shared state.
- Full CI had passed anyway, because the tiny Kokoro tests force the CPU.

Fix:
- Directions run in parallel only for the host loops on the CPU.
- New test `kokoro_repeats_exactly_on_the_default_device`: the tiny model on the build's device (the GPU under `cuda`) must give the same waveform 5 times for one seed.
  - With the race restored it failed 2/2.
  - With the fix it passes 2/2 on CUDA and also on CPU.
- CUDA parity is back, identical across two runs (SNR 34.2/44.0/35.9/38.6 dB).

Lesson: inference-tensor's CUDA backend is not safe to drive from two threads at once on one device.

Review fixes:
- The fused AdaIN falls back to the tensor path when the style batch differs from the input batch.
- `RES_SLOPE` is now `f32`.
- The fused test feeds an input at a non-zero offset.

## Run 14 - 2026-10-09 12:22

Question: can our own AVX2/FMA sgemm close the GEMM gap from Run 11?

Command:
- A BLIS-style packed sgemm (6x16 FMA tile, KC 256, MC 96, NC 4096) in `inference-tensor`, used by the CPU MatMul for f32. It matched a naive product over edge shapes, transposes and accumulation.
- GFLOPS benchmarks at 1 and 8 threads.

Finding 1: on the dev profile it was no faster: 48-52 GFLOPS on one thread, the same as `gemm`. Its disassembled micro-kernel had the right shape (12 FMAs on 12 register accumulators), with compare-and-branch noise in the loop.

Finding 2, the real cause: the dev profile is opt-level 3 but keeps `debug-assertions` and `overflow-checks` on, which put checks in every inner loop of both GEMMs. With `--config profile.dev.debug-assertions=false --config profile.dev.overflow-checks=false` (release-like):

| GFLOPS | gemm, 1 thread | gemm, 8 threads | own sgemm, 1 thread | own sgemm, 8 threads | MKL, 8 threads |
|---|---|---|---|---|---|
| 2048^3 | 76 | 307 | 92 | 270 | 428 |
| 128x1408x21240 | 82 | 285 | 58 | 227 | - |
| 21240x1408x128 | 66 | 290 | 62 | 78 | - |

Runs 11-12 measured `gemm` at "half of MKL" through debug checks. Release-like, it is 70% of MKL, and the first own sgemm is slower on most shapes (tall-M and gemv especially). The own sgemm is dropped.

Finding 3: Kokoro release-like (`gemm`, the fused AdaIN and the host LSTM from Run 12):

| case | time |
|---|---|
| sentence | 2.13 s (torch 1.23 s) |
| question | 1.08 s |
| short | 0.86 s |

Parity unchanged.

Finding 4: the conv now splits roughly half GEMM, half im2col. For k11 d5 the GEMM is about 27 ms of 48 ms; torch takes 23 ms for the whole conv. A per-tap conv (one GEMM per tap over a padded input, no im2col) beats im2col only for wide kernels:

| kernel | im2col | per-tap |
|---|---|---|
| k3 | 13 ms | 18-21 ms |
| k5 | 22 ms | 25-27 ms |
| k7 | 31 ms | 27-31 ms |
| k11 | 47 ms | 36-40 ms |

That is worth about 5% of Kokoro. Not taken yet.

Implications:
- CPU numbers measured in the dev profile overstate GEMM-bound costs about 1.6x.
- The remaining Kokoro gap to torch (1.7x) is mostly conv structure (im2col traffic), not GEMM speed.

## Run 15 - 2026-10-09 12:35

Question: does a direct conv1d without the im2col buffer close the conv gap to torch?

Change: `cpu_backend/conv1d_direct.rs`, an AVX2/FMA kernel.
- A 6-output-channel x 16-step register tile; each (input channel, tap) does 12 FMAs.
- The padded input is read in place: two 8-wide loads per tap.
- Weights are packed as [6-channel block][i][k][6] and broadcast.
- Tasks are 64-step time blocks, each running every channel block, so the input window stays in L2.
- It handles f32, stride 1, one group, kernels of 5 and up on AVX2+FMA CPUs; everything else keeps im2col.

Command: conv benchmarks on (1, 128, 21240), release-like (debug assertions and overflow checks off), plus torch on 8 threads; then Kokoro parity on CPU.

Finding (ms):

| kernel | direct | im2col | torch |
|---|---|---|---|
| k1 | 11-12 | 11-12 | 6.6 |
| k3 | 15 | 13-14 | 16 |
| k5 | 17 | 22-24 | 22 |
| k7 d3 | 20 | 31 | 19 |
| k11 d5 | 24-25 | 46-48 | 28 |

The direct path only takes kernels of 5 and up, since im2col's single GEMM wins at k3.

Kokoro, release-like:

| case | im2col | direct | torch |
|---|---|---|---|
| sentence | 2.06 s | 1.49 s | 1.23 s |
| short | 0.91 s | 0.70 s | - |

Parity is unchanged: durations exact, the same SNRs.

New test `cpu_wide_kernel_conv1d_matches_a_direct_loop`: k5/k7/k11 with dilation and padding past both ends, partial channel and time tiles, several time blocks, batch 2, and a transposed input and kernel.

## Run 16 - 2026-10-09 12:45

Review fixes:
- `c_in == 0` falls back to im2col, because rayon rejects a zero chunk size.
- `Tensor::conv1d` rejects a padded input shorter than the dilated kernel. Before, `l_out` underflowed on both paths.
- The direct kernel is compiled out under `mkl` and `accelerate`, which keep their own measured GEMM path.
- The timing figures moved out of the `MIN_KERNEL` comment, which would otherwise go stale.
- When (batch, time block) tasks are fewer than the pool's threads, each task's channel blocks are split as well, so short outputs parallelize.
- Tests add a one-step output, an offset input, two groups at k5 (each group's kernel chunk arrives at an offset), and the too-short error.

Finding: Kokoro release-like CPU, after the split.

| case | time |
|---|---|
| sentence | 1.28 s (torch 1.23 s) |
| question | 0.64 s |
| fast | 0.77 s |
| short | 0.66 s |

Run 15 had 1.49 s for the sentence. Parity is unchanged.

Overall, on the dev profile with debug checks off: 3.25 s -> 1.28 s, about torch's speed. Most of the original gap was the conv's im2col traffic plus per-op overhead in AdaIN and the LSTM; GEMM speed was secondary.

## Run 17 - 2026-10-09 14:54

Question: can Kokoro read plain text through the Rust vernacula-phonemizer (local branch `rust-port-scaffold`; en and en-GB, byte-identical to the TS goldens per that session)?

Steps and findings:
1. **A first mapping of my own** (canonical IPA to misaki), scored against misaki's `us_gold`/`gb_gold` lexicons over 3,000 words each: US 58.6% exact (CER 7.8%), GB 44.8% (11.9%) after two GB heuristics (drop secondary stress; small schwa before final -n/-l/-m).
   - Most residuals were the two dictionaries disagreeing (ɪ/ə reduction, secondary stress), not notation.
   - One useful finding: Kokoro v1.0 runs misaki without its 2.0 version setting, so the flap is `T`.
2. **The user pointed to vernacula's own translator** (/mnt/data/Programming/vernacula, `Vernacula.Tts.Base`): `KokoroFormat`, `KokoroPhonemizer`, `KokoroChunker`. These are measured, listener-tested and verified against goldens.
   - Differences from mine:
     - a word-final unstressed flap becomes the tap `ɾ` (Kokoro's duration predictor over-allocates `T` there);
     - GB keeps æ and secondary stress;
     - the de-/re- prefix ᵻ becomes ə, keyed on the source word through the trace;
     - non-English arms that keep phonemic contrasts English collapses;
     - Mandarin tone placement, and NFD decomposition (Portuguese nasals).
   - My mapping was replaced by a faithful port:
     - `kokoro/g2p.rs`: the renderer, the voice-prefix language table, the group-to-word map over the trace's UTF-16 spans, and the prefix rule;
     - `kokoro/chunker.rs`: paragraph, sentence, clause and word packing at 460/508 Kokoro tokens.
   - My GB heuristics were dropped.
3. **Tests:** vernacula's `KokoroFormatTests` vectors (every language arm) and `KokoroPhonemizerTests` (end to end through the real phonemizer) pass byte for byte.
   - The tiny engine test's voices are now af_/bm_/jf_-prefixed, and text and its own phonemes give identical audio for both English voices.
4. **The real model over HTTP**, a 186-character passage with a date and money, seed 1:

| voice | audio | first-request time |
|---|---|---|
| af_heart | 15.5 s | 1.7 s (about 1 s of it the phonemizer's data load) |
| bf_emma | 14.7 s | 1.1 s |

   The user listened to both: correctly phonemized and rendered.

Open: `vernacula-phonemizer` is a path dependency on the local, unpushed branch. A pushable PR needs a git dependency, which needs that branch pushed.

## Run 18 - 2026-10-09

Review of the text-input change, before the PR.

- Phoneme-string packer (`tts.rs` `chunks`): after cutting at the last pause, the carried words plus the next word could still pass `max`. `chunks("a, bcdefg hijklmn", 9)` gave `["a,", "bcdefg hijklmn"]` (14 > 9), so a caller's `phonemes` with an early comma in a long stretch could exceed 510 and fail in synthesis. Fixed: the carry is flushed too when it still doesn't fit; the case is pinned in the unit test.
- Text chunker cut an over-long single word (a digit run, URL) into characters and rejoined them with spaces, so it was read character by character. vernacula's C# `PackToBudget` does the same; changed here so character-level packing rejoins with no separator (and the separator costs nothing). New test: 1200 digits split into pieces that concatenate back to the input.
- vernacula's per-language `WordSegmentation` is not ported; whitespace words are right for en/en-GB, the only languages the phonemizer reads. Noted at `whitespace_words`; needed before ja/cmn turn on.
- Each piece is phonemized several times while chunking (as in C#). Dropped the per-call `readable` probe (validate already checks once per request); count caching left out.
- Dropped defensive code that couldn't fire (a span clamp, a word-index bound), and moved non-ASCII regex paraphrases in comments to words.
- Engine finding: a generation-time error (for example blank text, "nothing to speak") reaches the client only as "The model failed to process the request."; only `validate_speech_options` errors keep their text. The blank-text test asserts the error only.
- New end-to-end case: 20 copies of the test sentence (past 510 phonemes) synthesize through the multi-piece path.

`kokoro_tiny` 5/5 and the speech crate's kokoro tests pass; full CI rerunning.
