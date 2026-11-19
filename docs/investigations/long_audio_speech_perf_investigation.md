# Long recordings through transcription and diarization: where the time goes

Goal: a long upload (a 53-minute call, AAC in MP4, 48 kHz stereo) through Parakeet TDT 0.6B v3 (Q8_0 GGUF, Silero
VAD) and the Nemotron-3 diarizer used the GPU in bursts; find what keeps it idle. Server as the web UI loads the models
(`/v1/manager/load`), RTX 3090, CUDA dev build.

## Run 1 - 2026-11-18 20:30

Question: how busy is the GPU, and what is the CPU doing meanwhile?

`nvidia-smi` at 100 ms over the request, `perf record -F 499 -g` on the server:

```
request                      wall     GPU util mean / median
transcription                23.4 s   20% / 1%
diarization                  20.3 s   10% / 1%
transcription + diarization  29.2 s   25% / 0%
```

CPU seconds by thread (transcription): the request thread 10.1 s (rubato's sinc resampler 5.9 s, the AAC decoder
3.3 s), then the engine thread 12.2 s (resampling again 4.1 s, VAD 0.5 s, the rest the model). Uploads were decoded at
their own rate capped at 24 kHz (#1007, for VibeVoice-ASR), so a 48 kHz file was resampled to 24 kHz on decode and
again to 16 kHz in every 16 kHz model; and the decode ran on a tokio worker, not a blocking thread.

## Run 2 - 2026-11-18 21:10

Question: what do decoding at 16 kHz, and a faster resampler, each buy?

Decoding at 16 kHz (a quick patch): transcription 16.8 s, with diarization 18.4 s. The GPU timeline: ~9 s at 2%
(decode: resample 4.1 s, AAC 3.3 s, one thread), then ~9 s at 50-94% with both models.

rubato's `FftFixedIn` against our sinc stream (256 taps, 256x oversampling) on 53 minutes, 48 to 16 kHz: 1.7 s
(1024-frame chunks, 2 sub-chunks; larger chunks were slower) against 4.6 s. Not taken: it changes every resampled
input's bytes for a 2.7x on one stage.

## Run 3 - 2026-11-18 21:50

Change: speech pipelines report their rate (`GeneralMetadata::speech_sample_rate`); uploads decode straight to the
named models' rate (the highest when they differ, today's 24 kHz cap for any other model), off the runtime; the
decoder hands its mono blocks to the resampler on a second thread (same samples, same order, same output).

```
request                      before    after
transcription                23.4 s    13.5 s
transcription + diarization  29.2 s    15.0 s
```

The GPU now idles ~5 s at the start (the AAC decode, the resampler overlapped with it), then 50-90% for ~10 s.
Transcript as before (53:12, ~8,400 words; two speakers with the diarizer).

Left: overlapping the decode with the models (windows handed to the engine as they decode) would hide the first
~5 s; Opus / AC-3 uploads still pass through the server's 24 kHz WAV conversion, one extra resample for a 16 kHz model.
