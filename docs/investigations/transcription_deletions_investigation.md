# Transcription deletions: words a Parakeet transcript leaves out

Test clip throughout: a real 10-minute, 16 kHz mono conversational telephone recording (two speakers, frequent
overlapping backchannels) with a reference SRT (1483 words after normalizing). Scoring: lowercase, punctuation
stripped, Levenshtein alignment; a "deletion run" is consecutive reference words the hypothesis lacks. The app
("parakeet - C#" session) transcribes with the VAD plan and `max_window_seconds` 15 and diarizes separately; its
transcript builder assigns every word to a speaker and drops none, so missing words are missing from the engine.

## Run 1 - 2026-11-09 15:30

Question: what does the ONNX pipeline (vernacula: Sortformer diarization, then Parakeet TDT v3 fp32 on each speaker
segment, overlapping segments transcribed separately) produce that ours lacks?

Command: `dotnet src/Vernacula.CLI/bin/Release/net10.0/Vernacula.CLI.dll --audio <clip> --export-format json --benchmark`
(CPU, 78 s). 132 segments, 94 pairs of overlapping segments of different speakers. Segment text only, so words are
spread evenly over each segment and sorted by time before aligning.

| | words | sub | del | ins | WER |
|---|---|---|---|---|---|
| vernacula (per speaker segment) | 1603 | 115 | 18 | 138 | 18.3% |
| ours, 15 s windows (app's run, Vulkan) | 1481 | 60 | 34 | 32 | 8.5% |

Reference words we delete that vernacula keeps: 24. Ten are one sentence of the main speaker (about 121-126 s); the
rest are single backchannels and fillers under the other speaker, plus one 4-word stretch at about 160-165 s.
Vernacula deletes 8 we keep. Its extra insertions are mostly the same overlapped stretch transcribed under both
speakers, and short segments in another language (one-second backchannels read as Spanish, "aja").

Implication: the overlap backchannels are what per-speaker segmentation buys, at the cost of insertions and
code-switching in short segments. The 10-word sentence is a separate case: one speaker, inside one of our windows.

## Run 2 - 2026-11-09 15:45

Question: is the dropped sentence our window plan, or the decode?

Command: `scratchpad/lost/slices.py` (debug FFI, CPU, f32), each slice transcribed alone.

- our window (118.46-130.99 s) alone: keeps the sentence.
- vernacula's segment for that speaker (118.33-129.76 s): drops it.
- 119.0-130.99 and 120.2-130.99: keep it.

So it's a near-tie: the same audio keeps or drops the sentence depending on a fraction of a second of context.

## Run 3 - 2026-11-09 16:00

Question: the CPU slice kept it; does the whole stream on each device?

Command: `scratchpad/lost/stream.py <device> out.json` (VAD plan, max_window_seconds 15, 48 windows), CPU on the
debug FFI, Vulkan on the app's bundle build at a6187c13f; `dtype` set in the spec.

| device / dtype | words | sub | del | ins | WER | 121-126 s sentence |
|---|---|---|---|---|---|---|
| CPU, auto (f32 on this AVX2 box) | 1498 | 62 | 27 | 42 | 8.8% | kept |
| Vulkan, f32 | 1498 | 62 | 27 | 42 | 8.8% | kept |
| Vulkan, bf16 (what `auto` picks on a discrete GPU since #880) | 1481 | 60 | 34 | 32 | 8.5% | dropped |

Vulkan f32 matches the CPU word for word. The sentence goes with bf16: its rounding flips the near-tie of Run 2. WER
is about level (bf16 has fewer insertions), but bf16's errors include a whole sentence, which the app's user notices
more than a stray filler. Remaining deletions in f32 (27): backchannels under the other speaker, the 160-165 s
stretch, and the clip's last seconds.

Next: decide the speech dtype default on GPUs (f32 for Parakeet costs #880's gain: 2.30 -> 2.56 s per 10 min and
about 0.8 GB more peak), and whether overlapped backchannels are worth a second, per-speaker pass.

## Run 4 - 2026-11-09 16:20

Question: the user asked whether bf16 can be made more accurate at its speed, as was done before. Where does Parakeet's
bf16 drift on Vulkan come from?

Is it the sentence? No: the window alone on Vulkan keeps it in bf16 too (and bf16 keeps it on vernacula's segment,
where f32 drops it). Inside the stream it is one near-tie that bf16 tips; a drift measure is needed, not one sentence.

Setup (scratch branch `expt-bf16-drift`, not for merge): env switches in the Parakeet encoder; `EXPT_DUMP` appends each
window's encoder output; `EXPT_HEAD_F32` runs the decoder/joint in f32; `EXPT_WROUND` rounds the f16-halved weights
through bf16 first; `EXPT_SITES=a,b` rounds activations through bf16 (or f16 with `EXPT_F16`) at named points in an
f32 run. Vulkan FFI (`cargo build -p inference-ffi --no-default-features --features vulkan`), `run.py` over five
real ~10-minute recordings (three en conversational telephone, one de, one fr), 15 s windows. Drift = word spans that
differ from the f32 run (`diff.py`); encoder error = |x - f32| / |f32| per window, clip 1.

```
variant                            enc err mean / max    spans vs f32 (en1 en2 en3 de fr)
bf16 (auto on a discrete GPU)      0.0248 / 0.0650       10 10  8 21  2
bf16, decoder+joint f32            -                      8 12  9 18  0     (decoder isn't it)
f32, weights through bf16          0.0055 / 0.0382        0  0  1  3  0
f32, residual stream through bf16  0.0106 / 0.0272        1  4  3  9  0
```

Per site (bf16 in an f32 run, clip 1, enc err mean / max): scores (content, position, their sum) 0.0143 / 0.0839;
res 0.0106 / 0.0272; qkv (q, k, v, pos projections, biased queries) 0.0103 / 0.0801; dw (depthwise running sum per
tap) 0.0064 / 0.0164; conv (gate, norm, silu) 0.0049 / 0.0187; ff hidden 0.0030 / 0.0124; attn out 0.0013; probs
0.0010; all sites 0.0203 / 0.0647 (the rest of real bf16's 0.0248 is the weights). Leaving the attention logits in f32
(all but scores, qkv) still 0.0154: no single site carries it.

The matrix cores round every product's inputs to f16 even in f32 mode, so f16 storage is what f32 mode's products see.
Same sites rounded through f16 instead: enc err 0.0032 / 0.0236; spans vs f32 0 3 0 1 0 (bf16 at the same sites:
0 14 7 16 3). Largest magnitudes at the sites over four clips: res 7209-8666, qkv 448 (every clip: a fixed channel),
scores 208-281, dw 121-197, ff 159-167, conv 120-127: f16's 65504 leaves ~7x on the residual stream.

Implication: f16 activations (with the f16 weights f32 mode already keeps) would hold Parakeet about as close to f32
as f32 mode itself, at bf16's memory and bandwidth. It needs f16 builds of the kernels #880 gave bf16 builds (map, rows,
conv2d, matmul_cm, the small products, conversions; the 2-byte copy already exists) and `auto` picking f16 for the
ASR on a discrete Vulkan GPU. CUDA's auto bf16 Parakeet presumably drifts the same way; f16 there needs no kernels.

## Run 5 - 2026-11-09 17:00

Question: do real f16 kernels hold up as Run 4's emulation said, at bf16's speed and memory, and does CUDA drift the
same way?

Change (branch `vulkan-f16-speech`): the bf16 kernels' bodies read and write through `half.glsl`, whose
specialization constant `HALF_F16` (constant_id 0) picks f16 bits (`unpackHalf2x16` / `packHalf2x16`) over bf16 at
pipeline creation, so one SPIR-V serves both (no new shader binaries; the f32 builds unchanged): map, rows, conv2d,
the small and dot products, matmul_cm in both tiles, and the conversion. `PipelineSpec::constant` sets it;
`inference_vulkan::Half` picks the pipeline. The backend computes over F16 wherever it computed over BF16. `auto` for
the speech loaders is f16 on a discrete Vulkan GPU with matrix cores, and where it would be bf16 on CUDA.
`vulkan_half_ops_match_the_cpu` runs the bf16 checks over f16 too (f16 ties, overflow to inf, a subnormal: bit for bit
against the CPU's).

Word drift, same five clips, spans vs the same device's f32:

```
                    en1 en2 en3 de fr   total
Vulkan bf16          10  10  8  21  2    51
Vulkan f16            0   2  1   3  0     6
CUDA bf16             5  11  3  25  2    46
CUDA f16              1   3  1   1  0     6
```

Vulkan f16 clip 1 is word for word f32 (the 121-126 s sentence back; 27 deletions, WER 8.8%). CUDA f32 and Vulkan f32
agree word for word on clip 1. CUDA `auto` after the change matches CUDA f16 word for word (en2, de).

Speed and peak (`bench.py`: one warm-up, then three 10-minute streams, 15 s windows; nvidia-smi every 50 ms, MiB over
the idle desktop), RTX 3090:

```
Vulkan f32    2.41-2.59 s   +3485-3498
Vulkan bf16   2.40-2.58 s   +1801-1821
Vulkan f16    2.35-2.62 s   +1799-1844
CUDA f32      2.65-2.70 s   +2864-2990
CUDA bf16     2.01-2.06 s   +2005-2057
CUDA f16      1.99-2.07 s   +2005-2052
```

Diarizer (Nemotron-3, `auto` stays f32, untouched): asked for 16 bits, f16 moves 0.06-0.42 s of frames from f32 over
three clips, bf16 0.67-1.17 s.

One f16 bench process aborted once with glibc's `corrupted double-linked list` (its output lost in the pipe buffer);
not reproduced in 60 runs after (f16, bf16, f32; with and without nvidia-smi polling). Noted, not chased.

Remaining deletions against the reference in f32 and f16 alike: backchannels under the other speaker, the 160-165 s
stretch, the clip's last seconds. Per-speaker segments (vernacula) recover the backchannels at the cost of
insertions and short-segment code-switching; not done.

## Run 6 - 2026-11-09 19:40

Question (from the app's session, after #891): listening still finds dropped speech, and ONNX output has 112 words in
39 runs that ours lacks. Which of those are real drops, and are they the window cuts, the dtype, or overlap?

Scripts in `scratchpad/drops/`, Vulkan RTX 3090, bundle `libinference_ffi` at 39a693374, clip 1.

- The 112 is mostly the alignment. ONNX lists an overlapping speaker's segment after the segment it sits in, so a
  sequence alignment against our time-ordered words drops whatever moved. Counting per ONNX segment instead (its
  non-filler words with no match in ours within 1 s, `bag.py`): 94, almost all 1-3 word differences (spelling, a
  repeated word, one-second segments that ONNX reads as Spanish). One 7-word "run" is in our stream word for word.
- f16 and f32 give the same words on every slice tried (`slices.py`): the remaining drops are not the dtype.
- Real drops, heard by the model when given less context: a 4-word repetition under the other speaker inside a
  window (~2:42); a 6-word phrase across the cut at 516.2 s (neither window alone has it, the two joined do); a
  4-word phrase of the quieter speaker at ~9:26 (the window from 563.6 s drops it, from 565.0 s keeps it); most of
  the last window (584.6-598.2 s, 14 words, ~18 more heard per segment).
- Overlapping windows (`overlap.py`: each window widened by N s, words kept by midpoint in their own span):

```
overlap   WER     sub del ins
0 s       9.1%    63  29  44    (the stream word for word)
2 s      11.9%    81  26  70
4 s      10.8%    63  42  55
```

  Word times move with context, so a midpoint merge duplicates and drops words at the seams. Not the fix.
- Second pass over overlapped speech (`second.py`: each of our diarizer's 147 segments that overlap another
  speaker's, 322 s of audio, transcribed alone with 0.2 s pad; new words = ones the stream lacks within 1 s): 145 new
  words, 64 confirmed by ONNX or the reference nearby. It recovers each real drop above (and 9 words of the last
  window) but also adds short-segment Spanish and repeated fillers, as vernacula's output does.

Implication: what is left is overlap and context near-ties, which a second, per-speaker pass reaches at ~50%
precision as it stands (it would need a filter, e.g. segment length and runs of 3+ new words, and costs about half
again the ASR time for this clip). Window cuts and dtype are ruled out as the main cause.

## Run 7 - 2026-11-09 20:00

Question (user): one-second overlaps are the ones that code-switch; does a minimum segment length make the second
pass clean?

Command: `second.py` again, each segment's new and confirmed words kept, then cut by segment length.

```
min length  segments  seconds  new  confirmed
0           147       322      145  64 (44%)
1.0 s        81       288      103  48 (47%)
1.5 s        64       267       86  45 (52%)
2.0 s        54       249       81  43 (53%)
3.0 s        38       210       71  37 (52%)
```

From 1.5 s up no Spanish is left. The unconfirmed words there are English, and the two biggest blocks (13 words at
~74.5 s, 9 at ~589 s) fill gaps where the stream has no words at all for 4 s and 6 s, so they look like speech the
reference leaves out rather than insertions. What remains as noise is mostly a repeated filler or one word beside one
the stream has. Confirmed-by-ONNX-or-reference understates the precision of a 1.5 s cutoff.

Implication: a second pass over overlapped segments of 1.5 s or more (64 here, 267 s, so about +45% ASR time), its
new words merged into the timeline, recovers the audible drops with little code-switching.

## Run 8 - 2026-11-09 20:30

Question: the same second pass through the engine, as the app would run it: one transcription stream with `spans`
(branch `transcription-spans`: each `[start, end]` a window transcribed alone, in place of the VAD plan).

Command: `scratchpad/drops/spans_pass.py vulkan:0 f16 <clip> <stream> <diarization> 1.5 <out>`: our diarizer's
segments of 1.5 s or more that overlap another speaker's, 0.2 s pad, as `spans`; a span's word is added unless the
stream has the same word within 1 s.

```
                     words  WER    sub del ins   deleted runs >= 4   ONNX words unmatched within 1 s
stream (f16)         1498   9.1%   63  29   44   2                   94
stream + spans pass  1583  12.3%   56  13  114   0                   71
```

64 spans (292 s with padding) in 1.70 s on the 3090, on top of the stream's ~2.4 s. Deletions more than halve and
no long run is left; the insertions are the uncounted speech of Run 7 plus repeated fillers, so WER against this
reference rises.

## Run 9 - 2026-11-09 21:10

Question (user): keep the second pass as small as possible. Which overlapped turns does it need?

Command: `scratchpad/drops/select.py` (the 64 spans of Run 8, each span's new words kept with its features: the
share of the turn the other speaker overlaps, the stream's words per second in it, its longest stretch with no
stream word).

```
rule                     spans  seconds  new non-filler words  confirmed by ONNX or reference
all (Run 8)               64     292      79 (100%)            40 (100%)
stream gap >= 1.0 s       33     163      64 ( 81%)            29 ( 72%)
stream gap >= 1.5 s       17      93      53 ( 67%)            23 ( 57%)
stream gap >= 2.0 s       10      68      45 ( 57%)            20 ( 50%)
overlapped >= 50%          8      22       6 (  8%)             2 (  5%)
stream < 2 words/s        17      62      48 ( 61%)            18 ( 45%)
```

Turns sitting under the other speaker add almost nothing. The words come from long turns the other speaker cuts
into, where the main pass goes quiet: a gap of 1 s or more with no stream word finds half the spans (and 56% of
the seconds) with 81% of the new words.

Implication: diarize alongside the main pass (as `diarization_model` does now), then transcribe only the overlapped
turns of 1.5 s or more that hold a 1 s gap in the main pass's words. The gap needs the main pass's words, so the
spans come after it; batching them makes that cheap.

## Run 10 - 2026-11-09 22:00

Question: the second pass in the engine (`diarization_model` on a transcription, `overlap_pass` on unless false):
turns of 1.5 s or more that another speaker overlaps and that hold a 1 s gap in the first pass's words, as `spans`.
Does it hold Run 8's gain, at what cost?

Command: `scratchpad/drops/engine_pass.py vulkan:0 <clip> <out> [off]` (one engine with Parakeet + VAD and
Nemotron-3 added, stream with `max_window_seconds` 15 and `diarization_model`; one warm-up, three timed).

```
                       words  WER    sub del ins   deleted runs >= 4   time
overlap_pass false     1498   9.1%   63  29   44   2                   2.36-2.40 s
overlap_pass (on)      1566  11.9%   61  16  100   0                   3.27-3.47 s
app's pass (Run 8)     1583  12.3%   56  13  114   0                   +1.8 s in the app
```

The gap rule keeps most of Run 8's recovered words for half its spans: +0.9 s against the app's +1.8 s.

Side check, from a review of the Sortformer accuracy work in vernacula: its one fix our cache lacks is merging a
speaker's gaps under 0.5 s (it reported mean DER 17.2 -> 15.4% on far-field meeting audio). Applied by hand to our
Nemotron-3 turns on this clip: 254 -> 189 segments, 121 -> 101 of 1.5 s or more. The second pass over the merged
turns: 35 spans, 236 s, 56 new non-filler words (29 confirmed), against 33 spans, 163 s, 64 (29) unmerged.
Neutral for the pass (longer spans, same confirmed words); its case rests on diarization accuracy, which this
clip's reference (no speakers) cannot score.

## Run 11 - 2026-11-09 22:30

Question (user): diarize first and plan the ASR windows from its turns, in place of the VAD; no transcription until
the turns are known.

Command: `scratchpad/drops/diar_first.py vulkan:0 <clip> df` (one engine, Parakeet without VAD and Nemotron-3; the
plan: every speaker's turns joined, gaps under 0.3 s closed, runs over 15 s cut at the latest turn end inside 15 s,
grouped up to 15 s, each reaching up to 1 s into its gaps; run as `spans`; then the Run 10 second pass).

```
                              WER    sub del ins   deleted runs >= 4      time (sequential)
VAD plan, no pass (Run 10)    9.1%   63  29   44   2 (4, 6 words)         2.4 s (VAD + ASR, beside diarization)
diarization plan, no pass     8.9%   64  29   39   1 (13 words, last 6 s)  0.65 diarize + 1.85 ASR
VAD plan + pass (Run 10)     11.9%   61  16  100   0                      3.3-3.5 s
diarization plan + pass      11.0%   59  15   90   0                      0.65 + 1.85 + 0.90 = 3.4 s
```

The diarizer's plan transcribes about as well as the VAD's, slightly fewer insertions, and costs about what the VAD
did (0.65 s against the VAD's share of the stream). One clip; the 13-word run is the clip's last seconds, which the
pass recovers. With the pass the totals level: sequential diarization costs the concurrency the VAD plan had.
What it buys: the overlapped turns are known before any transcription, so windows and spans can go in one batch,
and the gap rule (which needs the first pass's words) is the only thing keeping them apart.
