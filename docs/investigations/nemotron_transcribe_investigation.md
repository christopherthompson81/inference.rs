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
