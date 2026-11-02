# Nemotron 3.5 Transcribe word timestamps

The checkpoint bundles a CTC word timestamp head (`ctc_timestamp.decoder.*`: a 2-layer transformer over the speech
encoder's states upsampled x2, a classifier over a 23,862-piece SentencePiece vocabulary and the blank) and its
tokenizer (`alignment_tokenizer/`). Reference: transformers `cc1451df3` (`models/nemotron3_5_transcribe`:
`get_word_timestamp_logits`, the processor's `align_word_timestamps`, a port of NeMo's
`MultiSpeakerSOTWordTimestampAligner`), installed in a scratch venv over the system torch 2.11 (torchaudio, scipy).

## Run 1 - 2026-11-02 15:03 (local clock)

**Question:** do our word times match transformers' on the same audio and transcript?

**Setup:** synthetic clips only: `clip00` (39 s, one Kokoro voice), `clip05`, and `mix2` (41.5 s: alternating 10 s
pieces of two voices' clips with 0.5 s of silence between). Transcripts from our server (GPU, the text GGUF
IQ4_NL-down-Q5_0 and the BF16 audio GGUF), word times through `/v1/audio/transcriptions` (`verbose_json`,
`timestamp_granularities[]=word`); transformers loads the BF16 checkpoint on the CPU (43 s, ~85 GB peak) and aligns
the same transcript with its own head and speaker logits.

**Implementation:** the head beside the audio tower (the transposed convolution with kernel = stride as one linear per
tap, `NemoTransformer` with an optional frame embedder for the transformer, a kernel-1 convolution as the classifier),
run on the encoder's chunks, its log-softmax over only the transcript's pieces and the blank (an hour would be 8.6 GB
of full logits); the alignment (`nemotron_transcribe/word_timestamps.rs`): the transcript's words per `<spk:N>` tag,
the tags matched to diarization speakers by IDF-weighted cosine of greedy piece counts and the Hungarian assignment,
torchaudio's `forced_align` (its window and tie-breaking) on each speaker's frames, the snap to diarization segments.
The loaders carry the tokenizer inside the model's config (`alignment_tokenizer`): from the checkpoint directory, or
written into the audio GGUF's config by `quantize nemotron-transcribe-audio`, which now also writes the head (the BF16
audio GGUF 1.37 -> 1.55 GiB). An engine request (`WordTimestamps`) runs the head when a response needs timings.

**Finding 1:** an untagged transcript (our default verbatim prompt) aligned with the speaker logits, as transformers'
processor does when given them, put `mix2`'s whole transcript on one diarization speaker's frames: the first word at
10.75 s. transformers' documented use passes the speaker logits only with the speaker-tagged prompt, so ours aligns an
untagged transcript on every frame, as one speaker.

**Result (`scripts/nemotron_transcribe_word_timestamps_parity.sh`):**

| clip | prompt | words | exact (start and end) | largest gap |
|---|---|---|---|---|
| mix2 | verbatim | 90 | 90 | 0 |
| clip00 | verbatim | 112 | 112 | 0 |
| clip05 | verbatim | 79 | 79 | 0 |
| mix2 | speaker tags (a temporary local prompt switch) | 120 | 120, tags to speakers `{0: 0, 1: 1}` as transformers | 0 |
| clip00 | speaker tags | 112 | 111 | 10 ms (a word snapped to a diarization segment edge, our diarizer's logits) |

The CPU's word times for `clip00` equal the GPU's (112 of 112). The alignment adds ~2.4 s to a 39 s clip on the CPU
(28.3 s against 25.9 s for `json`). The 3090's paged pool with the head loaded: ~92K tokens (BF16 audio GGUF) and
~183K (Q8_0), alternating twice; ~107K and ~200K before, the head's ~100 MB at BF16.

**Next:** speaker attribution through the API (the speaker-tagged prompt, words carrying their tags); the head's
classifier at BF16 in the audio GGUF (F32 now, 122 MB).

## Run 2 - 2026-11-02 15:42 (local clock)

**Question:** can the head run quantized, and at what cost to the word times?

**Change:** the head's matrices (the transformer's qkv, output and FFN, the classifier) load through the tower's
`tower_projection`, so a GGUF that stores them quantized keeps them so (a `GgufMatMul`) and any other file loads them
dense; the upsampler stays dense (its kernel dimension, 2, is the innermost, which Q8_0 cannot block). The audio GGUF's
converter records the classifier, a kernel-1 convolution, as a `[23863, 1280]` matrix so a recipe can quantize it; the
Q8_0 recipe adds `ctc_timestamp\.decoder\.(transformer\.layers\.\d+\.(ffn\.net\.\d+|attn\.w_qkv|attn\.out_proj)|decoder_layers\.0)\.weight$=q8_0`
(the Q8_0 audio GGUF 1.00 -> 0.88 GiB). The head's full-vocabulary logits are computed 256 frames at a time (a window's
were ~115 MB F32 and its BF16 copy). The parity script now transcribes and compares in two steps, the server stopped in
between: the first version left it up while transformers loaded the checkpoint (~85 GB).

**Finding (first try, every load quantizing the head in memory):** BF16 audio GGUF with the head at Q8_0, against
transformers: mix2 88 of 90 exact, the boundary of `work` / `changed` 80 ms early (11.994 -> 11.914 s); clip00 and
clip05 exact. A dense classifier with the rest at Q8_0 moved the same boundary, so the transformer alone flips it:
a near-tie between two frames rather than a drift. The user chose to run the head as the file stores it, so the BF16
path stays exact; they keep the Q8_0 GGUF on this machine.

**Result:**

| audio GGUF | head | 3090 paged pool | mix2 (90) | clip00 (112) | clip05 (79) |
|---|---|---|---|---|---|
| BF16 | dense | 96,384 tokens | 90 exact | 112 exact | 79 exact |
| Q8_0 | Q8_0 as stored | 194,688 tokens | 90 exact | 112 exact | 77 exact, `two` / `compilation` 80 ms early |

The Q8_0 head's flip lands on a different boundary than the first try's in-memory quantization of the same weights:
each is one near-tie boundary in 281 words. The dense head's ~183K tokens with the Q8_0 GGUF (Run 1) become ~194K.
