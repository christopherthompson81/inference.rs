# GGUF FLUX and T5 investigation

Question: can image generation load FLUX from GGUF, with the text encoders from local files, so a real-checkpoint
test (issue #74) runs on a local ComfyUI-style set: `flux1-dev-Q8_0.gguf`, `t5-v1_1-xxl-encoder-Q8_0.gguf`,
`clip_l.safetensors` and `ae.safetensors`? Today the FLUX loader wants `flux1-*.safetensors` plus diffusers
`transformer/config.json` / `vae/config.json`, and the stepper downloads T5-XXL safetensors
(`EricB/t5-v1_1-xxl-enc-only`) and CLIP-L (`openai/clip-vit-large-patch14`) from the hub. Nothing in the diffusion
path reads GGUF.

## Run 1 - 2026-09-28

Command: llama.cpp's `gguf-py` `GGUFReader` over both GGUF files; the safetensors header of `clip_l.safetensors`.

- `flux1-dev-Q8_0.gguf` (12.7 GB): `general.architecture = flux`, no model config in the metadata. 780 tensors,
  304 Q8_0 (the linear weights) and 476 F16 (biases, norms). Names are Black Forest Labs' own, the same as
  `flux1-dev.safetensors`: `double_blocks.N.{img,txt}_{attn,mlp,mod}.*` (19 blocks), `single_blocks.N.*` (38),
  `img_in`, `time_in`, `guidance_in` (so dev, not schnell), `final_layer.*`. Hidden size 3072, head dim 128.
- `t5-v1_1-xxl-encoder-Q8_0.gguf` (5.1 GB): `general.architecture = t5encoder`, hyperparameters in the metadata
  (`embedding_length` 4096, `feed_forward_length` 10240, `block_count` 24, `head_count` 64, `key_length` 64,
  `relative_buckets_count` 32, rms eps 1e-6) and a SentencePiece vocabulary (`tokenizer.ggml.model = t5`, 32,128
  tokens with scores). 219 tensors in llama.cpp naming, each mapping one to one onto HF T5 encoder names:
  `token_embd` -> `shared`, `enc.blk.N.attn_{q,k,v,o}` -> `encoder.block.N.layer.0.SelfAttention.{q,k,v,o}`,
  `enc.blk.0.attn_rel_b` -> `...relative_attention_bias`, `attn_norm` -> `layer.0.layer_norm`,
  `ffn_{gate,up,down}` -> `layer.1.DenseReluDense.{wi_0,wi_1,wo}`, `ffn_norm` -> `layer.1.layer_norm`,
  `enc.output_norm` -> `encoder.final_layer_norm`. Linear weights Q8_0, norms and the relative bias F32.
- `clip_l.safetensors` (246 MB): 196 tensors, all `text_model.*` in HF naming, so the existing
  `ClipTextTransformer::new(vb.pp("text_model"), ..)` reads it as is. It carries no config or tokenizer.
- `ae.safetensors` is the file the loader already takes.

Implication: the files need (1) FLUX linears fed from GGUF Q8_0 tensors, with the dev config inferred from the tensor
shapes, (2) T5 fed from GGUF under a name map, with its tokenizer built from the GGUF vocabulary (or the hub
tokenizer), and (3) the CLIP-L text config and tokenizer, which are standard and small. Next: how the text-model GGUF
path serves quantized tensors to layers, and whether the FLUX and T5 constructors build their linears through
inference-quant (where a GGUF weight source could reach them) or through plain candle `Linear`.

## Run 2 - 2026-09-28

Question: can the text-model GGUF machinery feed the FLUX and T5 constructors?

- Reading: `inference_quant::GgufWeightSource` (crates/inference-quant/src/gguf/weight_source.rs) takes a
  `GgufBindingMap` from native tensor names to GGUF tensors (with optional slice/concat/transpose/cast transforms),
  and `sharded_var_builder()` turns it into a `ShardedVarBuilder`. Linears built through inference-quant's
  `linear` / `linear_no_bias` ask that weight source for the layer by prefix and get a quantized `GgufMatMul`; every
  other tensor (norm scales, biases, embeddings) is materialized dense. The GGUF text and multimodal pipelines load
  this way (pipeline/gguf.rs, gguf/qwen_multimodal_bindings.rs).
- Blocker: the FLUX model (inference-models-diffusion/src/flux/model.rs) and the T5 encoder (t5/mod.rs) build every
  linear as a dense `candle_nn::Linear`, so they can only consume dense tensors. Loading the Q8_0 GGUF through them
  would dequantize FLUX dev to ~24 GB in BF16, which defeats the point of the 12.7 GB file.

Implication, as three PRs:
1. FLUX and T5 build their linears through inference-quant (`Arc<dyn QuantMethod>`), unquantized for safetensors
   as today; behavior-preserving, checked by the existing diffusion code paths and a small forward test.
2. GGUF loading: the FLUX transformer from a GGUF file (identity bindings; dev/schnell config inferred from the
   tensors, since the file carries none), T5 from a GGUF file (llama.cpp -> HF name map; tokenizer from the GGUF
   SentencePiece vocabulary), and CLIP-L from a local safetensors file with the standard CLIP-L text config; the
   diffusion selection gains local paths for the two text encoders.
3. The #74 test in the `--models` tier: generate a small image through `Engine::image_generation`, the ABI and
   `/v1/images/generations`, for both response formats, on the local files named by `INFERENCE_TEST_*` variables.

## Run 3 - 2026-09-28

Change: FLUX and T5 linears are `MaybeQuantLinear` (dense `candle_nn::Linear` unless the var builder carries a
quantized weight source, then inference-quant's layer; offloading moves dense layers only, and a GGUF load refuses
`flux-offloaded`). `inference_models_diffusion::gguf::var_builder` builds a `GgufWeightSource` var builder under a
tensor-name map (identity for FLUX, `t5_native_name` for llama.cpp's `t5encoder`). `FluxLoader` recognizes a local
single-file layout (`flux1-{dev,schnell}*.gguf`, `ae.safetensors`, optional `t5*.gguf` and `clip_l.safetensors`),
infers the transformer config from the weights (`Config::from_weights`) and uses the published FLUX autoencoder
config (`autoencoder::Config::flux`). The stepper takes the local T5 and CLIP weights; their configs and tokenizers
still come from the hub (a few MB). Side fix: `DiffusionLoaderType::matches_flux` used `r"^flux\\d+-..."`, which in a
raw string matches a literal backslash, so FLUX repos were never auto-detected.

Command: `INFERENCE_TEST_FLUX_DIR=/mnt/data/models cargo nextest run --profile cuda --features cuda --workspace --lib
--bins --tests -E 'binary(flux)'` (the `--cuda` mode's build), RTX 3090, BF16.

Result: `flux_gguf_generates_an_image_of_the_requested_size` passes in 17 s (load from page cache plus two 256x256
generations, one per response format). The saved PNG for "A red apple on a wooden table" is a clean, recognizable
red apple on a wooden surface, so the T5 name map, CLIP-L, the inferred config and the AE constants are all right.

Finding on the way: `b64_json` carries a `data:image/png;base64,...` string rather than bare base64, and
`response_format` takes `Url` / `B64Json` rather than OpenAI's `url` / `b64_json`. Both are documented in the image
generation guide, so this is a deliberate deviation, but an OpenAI client that base64-decodes `b64_json` fails on it.

Next: where the real-weight test runs (it needs the 18 GB local set and a GPU), the ABI and HTTP coverage #74 asks
for, and unit tests for layout detection and config inference.

## Run 4 - 2026-09-28

Question: once `b64_json` matches OpenAI (bare base64; `response_format` of `url` / `b64_json`, with the old `Url` /
`B64Json` accepted as aliases), does real-weight generation still decode with no prefix stripping, and does it work
through the surfaces #74 names rather than only the SDK builder?

Change: the test moved from the SDK (`DiffusionModelBuilder`) to `crates/inference-server-core/tests/flux.rs`. It loads a
`DiffusionPlain` EngineSpec with `inference_api::Engine::load`, then:

- asks `Engine::image_generation_json` (what `inference_image_generation` calls) for `b64_json`;
- sends `/v1/images/generations` through the router for `url`.

`url` writes into the working directory, so the test chdirs into a tempdir; nextest's process-per-test keeps that
contained. `INFERENCE_TEST_FLUX_DIR` is now in `~/.cargo/config.toml`, and the cuda nextest profile runs the test alone
(`threads-required = num-test-threads`), since ~18 GB plus six 2.5 GB gpu-model tests would overflow a 24 GB card.

Command: `cargo nextest run --no-fail-fast --profile cuda --features cuda --workspace --lib --bins --tests -E 'binary(flux) | test(image_response_formats)'`

Finding: both pass. FLUX takes 16.9 s for load plus two 256x256 generations. Bare base64 decodes to a 256x256 PNG, and
the HTTP `url` file opens at 256x256.

## Run 5 - 2026-09-28

Review follow-ups, rerun with the same filter plus the new unit tests.

Changes:
- The config is inferred from the GGUF shape map (`Config::from_weights(&shapes)`), so no weight is dequantized to read
  a dim. A unit test covers it.
- The layout is detected once (`DiffusionModelLoader::local_paths`) instead of three `read_dir`s.
- A local T5 is GGUF only; the dead safetensors branch and the unreachable T5 offload check are gone.
- ComfyUI's `model.diffusion_model.` prefix is stripped from FLUX GGUF names.
- The real-weight test now checks the centre of the image is red-dominant and the luma std dev is above 20. A size-only
  check would pass on a wrong text encoder or config.

Finding: 7 of 7 pass, and FLUX takes 17.5 s. The centre-red and detail checks hold on the Q8_0 set.
