# Nemotron 3.5 Transcribe investigation

Porting nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B: a Transformer ASR encoder and a Nemotron-3-Diarization
speaker encoder (a "parallel expert encoder") in front of the Nemotron 3.5 Lightning 30B-A3B hybrid Mamba-2 /
attention / MoE decoder, with an MTP head and a CTC word-timestamp head. The decoder is Nemotron-H (#565), which the
tree does not have yet.

## Run 1 - 2026-11-01 16:40

**Question:** what does the checkpoint hold, what reference exists, and what has to be built?

**Commands:** the Hub's metadata and file list; `config.json`, `processor_config.json`, `generation_config.json`,
`tokenizer_config.json`, `chat_template.jinja` and the model card; the safetensors header by HTTP range; the Hub
search for quantized copies; transformers main's `models/nemotron3_5_transcribe/` and `models/nemotron_h/` listings;
llama.cpp's log since 2026-09; issue #565.

**Finding:**
- One 67.5 GB `model.safetensors` (BF16, plus a few F32/I64/U8 tensors), NeMo names: `llm.model.layers.*` (474
  tensors, 30.9 B params), `llm.mtp.layers.*` (1.3 B), `llm.lm_head`, `perception.encoder.asr_encoder.*` (631 M),
  `perception.encoder.diarization_model.*` (99 M), `perception.encoder.{asr,diar}_norm`, `perception.proj`
  (1280 -> 2688), `perception.preprocessor.featurizer` (mel filterbank and window), `ctc_timestamp.decoder.*` (2-layer
  Transformer CTC head over 23862 word pieces) and `ctc_timestamp.tokenizer.model_proto` (a SentencePiece model
  stored as a U8 tensor).
- Decoder (`text_config`, `nemotron_h`): 52 layers in the pattern `layers_block_type` (Mamba-2 `linear_attention`, `moe`,
  and a few attention layers), hidden 2688, 128 routed experts top 6 plus one shared (3712), expert width 1856,
  squared-ReLU, `routed_scaling_factor` 2.5, Mamba-2 with 64 heads of 64, state 128, `n_groups` 8, conv 4; attention 32
  heads, 2 KV heads, head dim 128; vocab 131072; MTP block `['full_attention', 'moe']`, two trained depths through a
  repeated layer.
- ASR encoder (`audio_config.asr_config`): NeMo `TransformerEncoder`, 32 pre-norm layers, d_model 1280, 10 heads, q/k
  norm, no qkv bias, partial rotary (factor 0.5, theta 10000), FFN 5120 GELU, 128 mel bins, subsampling 8 by feature
  stacking; encoder chunks of 30 s; one audio token per 80 ms.
- Speaker branch: Nemotron-3-Diarization (Sortformer-style, 8 speakers, 512-wide FastConformer plus a 192-wide
  Transformer head, streaming config), its speaker activity combined with the ASR features (`speaker_feature_mode:
  thresholded`, threshold 0.5) before `perception.proj`. #409 ported Nemotron-3 Diarization for standalone use.
- Reference: transformers main has `nemotron3_5_transcribe` (configuration, modeling, processing, generation, and a
  NeMo-to-HF converter) and `nemotron_h`; the card loads the repo directly with `AutoModelForMultimodalLM`, so the
  NeMo names map on load. A scratch venv has transformers 5.23.0.dev0 with torch 2.15 CPU. 125 GB RAM holds BF16
  (67 GB), not F32.
- Published GGUFs are for other runtimes: sakasegawa's for speech.cpp ("its own layout, so llama.cpp ... can't read
  it"), AtomicChat's for NVIDIA's NeMo-Speech.cpp (a text GGUF plus `Encoder-F16.gguf`). llama.cpp mainline has
  Nemotron-H and Nemotron 3.5 VL/Bolt, not Transcribe. Also published: NVIDIA's FP8, NVFP4 and a DFlash2 drafter.
- Not in the tree: Nemotron-H (Mamba-2 per-layer pattern, grouped gated RMSNorm, sigmoid-routed MoE with
  correction bias and squared ReLU). Granite 4.0-H has the Mamba-2 mixer, SSM kernels and hybrid cache to build on
  (#565's notes).

**Next:** download the checkpoint (the reference needs BF16); read transformers' two models; then Nemotron-H text
first (the Transcribe decoder, text-only prompts against transformers), then the ASR encoder and the chat/transcribe
path, then the speaker branch and word timestamps.

## Run 2 - 2026-11-01 17:00

**Question:** does Nemotron-H, built on Granite's hybrid blocks, compute what transformers' `NemotronHForCausalLM`
does?

**Commands:** a scratch script in the transformers-main venv building a random-weight `NemotronHConfig` (vocab 131072,
hidden 64, `layers_block_type` mamba/moe/mamba/attention/moe/mamba/mlp, 4 Mamba heads of 16 with `n_groups` 2 and state
16, 4 attention heads / 2 KV of 16, 4 experts top 2 of width 32 plus a shared 48, `routed_scaling_factor` 2.5,
`chunk_size` 8; random correction biases, `A_log`, `D`, `dt_bias` and norms), saved with `save_pretrained`, the
Transcribe tokenizer beside it; greedy 12 tokens from a 12-token prompt, F32 CPU. `inference serve --cpu --dtype f32`
on the saved directory, `/v1/completions` with logprobs.

**Finding:**
- An agent's comparison against Granite 4.0-H (the starting point #565 names): each Nemotron-H layer is one norm, one
  mixer, one residual; Granite's Mamba-2 mixer, SSM kernels and hybrid cache cover the shapes (state 128 dispatches
  through the scan kernel's `ceil(d_state/32)`). What differs: Mamba width is heads x head_dim (not expand x hidden), the
  gated RMSNorm is per group of `width / n_groups`, the SSM state is f32 (`mamba_ssm_cache_dtype`), prefill clamps dt
  at `time_step_min` (transformers' mixer sets `time_step_limit = (time_step_min, inf)`; decode does not clamp), NoPE
  attention scaled by `head_dim^-0.5`, sigmoid routing with the correction bias for selection only and the unbiased
  scores renormalized and scaled, ungated squared-ReLU experts and shared expert.
- Built as Granite's blocks: a mixer layer and the feed-forward layer after it make one block (norm, mixer, residual,
  norm, FFN, residual), a mixer followed by a mixer a block with no FFN; the Transcribe pattern makes 29 blocks of its 52
  layers, so the hybrid cache, paged attention and device mapping see only stateful layers. The experts are a new
  ungated backend of `MoEExperts` (gather path: up, act, down) so the shared checkpoint layouts and ISQ apply.
- First serve: `kv-cache: requested capacity (524) above max seq len (512)`: Granite's KV cache grows by 512 past a
  512-position config. Raised the tiny config's `max_position_embeddings` to 4096 (Nemotron-H has no position
  encoding); real configs are far larger.
- Same 12 tokens; prompt 12 tokens both; max |delta logprob| 6.7e-5 over the 12 steps (logprobs near -9, random
  weights).

**Next:** the llama.cpp `nemotron_h_moe` GGUF bindings, then the Transcribe text model from the real checkpoint
against transformers.

## Run 3 - 2026-11-01 17:10

**Question:** do llama.cpp's Nemotron-H GGUFs load onto the same model?

**Commands:** llama.cpp 970b51ce7 `convert_hf_to_gguf.py --outtype f32` on Run 2's tiny checkpoint (a copy whose tokenizer
drops `<|audio|>`, id 131072: the converter asserts every token id is below the padded vocabulary); `inference serve
--cpu --dtype f32 --format gguf` on it; Run 2's comparison.

**Finding:**
- AtomicChat's Transcribe text GGUF (header by HTTP range) is plain llama.cpp `nemotron_h_moe`: `blk.N.attn_norm`,
  `ssm_in/conv1d/a/d/dt/norm/out`, `attn_q/k/v/output`, `ffn_gate_inp`, `exp_probs_b`, `ffn_{up,down}_exps`,
  `ffn_{up,down}_shexp`, `ssm_norm` as `[512, 8]`, 53 blocks with the last `nextn.*` (MTP).
- The bindings map those onto transformers' names (`model.layers.N.norm`, `.mixer.*`, `model.embeddings`,
  `model.norm_f`), `ssm_a` back to `A_log` and the conv and norm reshapes as Granite's, the router and its bias in f32;
  the config comes from the metadata, each block's type from its tensors.
- First conversion failed (`assert max(tokenizer.vocab.values()) < vocab_size`), then twice more while the token was
  still in `model.vocab` and in `tokenizer_config.json`'s `audio_token` / special-token lists.
- Same 12 tokens, max |delta logprob| 6.7e-5, as from safetensors.
- Granite's tiny suites (dense, MoE, SWA, Speech, Vision) pass on the restructured blocks.

**Next:** the Transcribe model itself: the ASR encoder, the speaker branch, the projector, its prompt.

## Run 4 - 2026-11-01 17:45

**Question:** does the real Transcribe decoder, loaded as Nemotron-H from the checkpoint's NeMo names, answer text
prompts as transformers does?

**Commands:** transformers main, `AutoModelForMultimodalLM` BF16 on the CPU, `apply_chat_template(...,
enable_thinking=False)`, greedy 24 tokens with scores, on "Name three rivers in Europe." and "Translate to French: The
weather is nice today."; ours on a scratch directory holding the `text_config` as a `NemotronHForCausalLM` config beside
links to the checkpoint and its tokenizer and template, `inference serve --cpu --dtype bf16`, chat with
`enable_thinking: false` and logprobs.

**Finding:**
- First load: `cannot find tensor model.embeddings.weight`. The checkpoint is NeMo Automodel's layout:
  `llm.model.embed_tokens`, `llm.model.norm`, `llm.lm_head`; Mamba's `A_log`, `D`, `dt_bias` under
  `mixer._fp32_params` (F32); the experts stacked and transposed, `experts.gate_and_up_projs` `[128, 2688, 1856]`
  (ungated, so only up) and `experts.down_projs` `[128, 1856, 2688]`; `_extra_state` placeholders; MTP under
  `llm.mtp.layers.{0,1}`. transformers' conversion mapping renames these and transposes the stacks. Now read as they
  are: the loader finds the decoder under `llm.model` with its head at `llm.lm_head` and takes the Llama names for the
  ends, the Mamba layer reads `_fp32_params` when present, the ungated experts read the transposed stacks.
- Prompt tokens equal (22, 26). French: the same 7 tokens to `<|im_end|>`. Rivers: the same first 12 tokens, then
  "Rhône" for "Danube" as the first river. Per-step logprobs differ by up to 0.16 (step 0 of the French: -0.096 against
  -0.247). Our top logprobs sit on a coarse grid (-1.028, -2.028, -2.028; -1.17, -1.17): the BF16 `lm_head` output rounds
  the logits before the softmax, so near ties (step 3 of the rivers) are rounding. transformers also runs `lm_head` in
  BF16 and casts the F32 Mamba parameters to BF16; only `e_score_correction_bias` stays F32 there. F32 on both sides
  does not fit in 125 GB; the tiny F32 match (Run 2) is the arithmetic check.

**Next:** commit the NeMo layout; then the audio side.

## Run 5 - 2026-11-01 19:30

**Question:** does our Nemotron 3.5 Transcribe (the full model from the NeMo checkpoint) transcribe as transformers does?

**Commands:** transformers main BF16 CPU: the processor's default transcription prompt as a user turn with the audio,
`apply_chat_template(..., enable_thinking=False)`, greedy 96 tokens with scores, over the synthetic 3 s clip, the
28.7 s two-voice dialogue and the 74 s clip (two encoder chunks: 40 s chunks with 4 s of context either side); the
projected audio features saved. Ours: `inference serve --cpu --dtype bf16 -m <checkpoint>`, the same prompt with
`input_audio`, `enable_thinking: false`.

**Finding:**
- Built: the speech encoder as a NeMo `TransformerEncoder` shared with Nemotron-3 Diarization (`NemoTransformer`,
  now with per-head q/k LayerNorm and rotary over the first 64 of each head's 128 dims), run in 500-step chunks with 50
  steps of context; the diarization model on the unnormalized features, each step's 8 frame sigmoids averaged,
  thresholded at 0.5, LayerNorm over the 8 speakers, times a fixed sinusoid table, added to the normalized encoder
  states; the F32 projector; the placeholder (131072, past the vocabulary) looked up as token 0 and replaced. The
  audio weights are read from NeMo's names into the speech crate's (transformers') names, in F32 as transformers keeps
  them. The features are NeMo's with the filter bank and window rounded to bfloat16 (`mel_filters_dtype`), new as a
  `NemoMel` option.
- First run: the server was OOM-killed at 114 GB resident on the 28.7 s clip (375 audio tokens). The ungated experts'
  gather path copies a weight matrix per routed token (375 x 6 x 10 MB in one layer). Unquantized ungated experts now
  run grouped by expert (one matmul per expert over its rows); quantized ones keep the gather path.
- Prompt tokens equal (94, 426, 982; audio 43, 375, 931). Every greedy token equal: 11 of 11 (the pangram), 95 of 95
  and 96 of 96 (the dialogue and the 74 s text verbatim as far as decoded).

**Next:** GPU with ISQ (the 3090), then AtomicChat's GGUF pair (the `pee` encoder GGUF), the word-timestamp head.

## Run 6 - 2026-11-01 19:30

**Question:** does the model run on the 3090 quantized in place, and how does it fit?

**Commands:** `inference serve --isq q4k` (auto device map), then `--isq q4k -n 0:24`, then `--isq q3k -n 0:29`, on the
NeMo checkpoint; the LAN server stopped for the runs; Run 5's clips and comparison; `nvidia-smi` peaks.

**Finding:**
- First Q4K load: GPU OOM in seconds. The experts' immediate-ISQ patterns ended at the bare stack names
  (`experts.gate_and_up_projs$`) while the check appends `.weight` to the prefix, so the stacks loaded unquantized
  onto the GPU in BF16 (2.5 GB per MoE layer). Patterns fixed. Granite's transformers-v5 expert patterns
  (`experts.(gate_up_proj|down_proj)$`) look to have the same mismatch; not checked.
- Q4K: the auto map placed 27 of 29 blocks on the GPU, then `Failed to create hybrid cache: ... out of memory`; it
  underestimates the blocks. With 24 on the GPU: 21.6 GB peak, loads in 8 minutes (ISQ on the CPU); the first request
  failed `dtype mismatch in cat, lhs: BF16, rhs: F32`: CPU blocks beside a GPU compute in F32 over a Mamba conv state
  allocated in BF16. The conv state now takes the activation dtype around each Mamba call. Then decode ran at 0.2-0.4
  token/s, GPU at 1 %, one CPU core busy: the five CPU blocks' quantized experts dequantize a whole expert per routed
  token on the gather path.
- Q3K, all 29 blocks on the GPU: 22.4 GB at load, 23.8 GB serving; `Recurrent state for every sequence slot does not
  fit ... fitted_seqs=1`. 3 s clip: all 11 tokens equal. Dialogue: the same words, commas for some of the reference's
  full stops (same first 2 tokens). 74 s clip refused: `CUDA memory pressure prevented prompt admission requiring
  317567188 bytes of transient workspace`.

**Next:** a sizing plan (as for the Qwen3.8 27B recipe): a quantized artifact written once, sized to leave headroom,
measured by KLD against BF16 references, then speed.

## Plan - 2026-11-01 20:00 (sizing a GGUF for the 3090)

**Budget:** the card's usable 24,145 MiB (23.58 GiB), under the 23.5 GiB ceiling used for the Qwen3.8 27B recipe. Fixed
costs: CUDA context and kernels about 0.6 GiB; KV only on the 6 attention layers (2 KV heads x 128, BF16), 6 KiB per
token, so 64K tokens (the training context; one hour of audio is 45K audio tokens) is 0.38 GiB; Mamba state about
47 MiB per sequence; activations (the MoE prefill workspace, the encoder's 600-step window) about 1 GiB.

**Weights (from the checkpoint's shapes):** routed experts 29.4 B (55 GiB BF16; 14.5 GiB at 4.25 bpw, 15.4 at 4.5,
12.0 at 3.5); Mamba projections 0.89 B; shared experts 0.46 B; attention 0.14 B; embeddings and `lm_head` 0.35 B each;
audio tower 0.73 B (2.73 GiB F32 as transformers keeps it, 1.37 GiB BF16); MTP 1.3 B (unused; left out). Everything
but the experts at Q8_0 or Q6_K is about 2.1 GiB.

**Target:** about 21 GiB serving at 64K: experts at an IQ4_KS-class mix (14.5-15.5 GiB), the rest at Q8_0/Q6_K, the
audio tower BF16 if transcripts hold (else F32, 1.4 GiB more). Measured Q3K-everything with an F32 tower: 23.8 GiB and
no room to admit a 74 s prompt (Run 6).

**Steps:**
1. BF16 text GGUF for llama.cpp tools: the decoder renamed from NeMo's layout to transformers' (experts transposed,
   `_fp32_params` flattened, MTP dropped), the tokenizer without the out-of-vocabulary `<|audio|>`, then llama.cpp's
   `convert_hf_to_gguf.py` (Run 3 showed the converter's `nemotron_h_moe` output loads in our engine).
2. References: BF16 KLD base logits with ik_llama.cpp's `llama-perplexity --kl-divergence-base` (wiki, c512, 40 chunks,
   as for the 27B); an audio-conditioned reference from our BF16 CPU run (Run 5 matched transformers exactly):
   teacher-forced logprobs over the synthetic clips' transcripts.
3. imatrix: ik_llama.cpp's `llama-imatrix` on the BF16 GGUF (or AtomicChat's published imatrix to start).
4. Recipes with `llama-quantize --custom-q`: A experts IQ4_KS, the rest Q8_0; B experts down_proj one step up; C a 3.5
   bpw expert mix for headroom. KLD of each against step 2.
5. Audio weights beside the GGUF: load the tower from the original checkpoint (only its `perception.*` tensors are read)
   or from AtomicChat's `pee` encoder GGUF (needs its bindings and discovery through `nemo_speech.speech_model`).
6. Then speed: prefill and decode on the GPU, the ungated experts' CUDA path (fused kernels for `act(up)`), CUDA graphs.

**Disk:** 131 GB free on /mnt/data. Step 1 as transformers-layout safetensors then GGUF needs 63 + 59 GB at once;
converting the safetensors to the GGUF directly, or deleting the intermediate first, keeps it under.

## Run 7 - 2026-11-01 20:45

**Question:** what quantization tooling does inference.rs have for this model, and can the decoder become a llama.cpp
BF16 GGUF without a 63 GB intermediate?

**Commands:** a scratch wrapper around llama.cpp's `convert_hf_to_gguf.py` that replaces `ModelBase.index_tensors`
with the NeMo checkpoint's tensors under the Hub's Nemotron-H names (stacked experts sliced per expert and transposed,
`_fp32_params` flattened, MTP, audio and CTC left out), on a source directory holding the `text_config` and the
tokenizer without `<|audio|>`; `--outtype bf16`. Survey of `inference quantize`, `inference-quant` and
`inference-tensor`'s quantized types. ik_llama.cpp checked for `nemotron_h_moe` (local c446f1e0 and origin main
b25c3fd2: absent).

**Finding:**
- `Nemotron-3.5-Transcribe-text-BF16.gguf`, 63.2 GB, 401 tensors, 52 blocks: the same layout and metadata as
  AtomicChat's llama.cpp file. ik_llama.cpp cannot quantize it (no `nemotron_h_moe`), so the IQ-type recipes have to
  come from our own tooling.
- inference.rs has: imatrix collection at load (`--calibration-file`, 1024-token chunks) and online (calibration
  start/apply on a running ISQ model, saved as `.cimatrix`, then requantized from the source weights), routed per-expert
  statistics, imatrix-weighted encoding for the k-quants, per-tensor types through topology overrides, UQFF write and
  reload. Missing: encoders for the IQ types (we load IQ4_XS / IQ4_NL / IQ4_KS but cannot write them), and a KLD tool
  (Recipe A's numbers came from llama.cpp's `llama-perplexity`).
- The grouped path for unquantized ungated experts recorded no routed statistics, and calibration of pending
  (still unquantized) layers took the gather path, which for this model cannot fit in RAM. The grouped path now runs
  for any unquantized stack (pending ones included) and records both projections' statistics. A tiny test calibrates an
  ISQ load online and finds non-zero statistics for every MoE layer's up and down stacks.

**Next:** IQ encoders (IQ4_NL, IQ4_XS, then IQ4_KS) with round trips against our dequantizers; a KLD tool (BF16
reference logprobs over text and audio, then mean KLD, top-1 agreement, perplexity of a quantized load).
