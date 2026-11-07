# Nemotron 3 Nano Omni 30B-A3B: sound and vision on the Nemotron-H decoder

Part of #565 (Nemotron-H family). `NemotronH_Nano_Omni_Reasoning_V3`: the Nemotron 3 Nano 30B-A3B MoE decoder (runs
here since #766/#767) with a C-RADIO vision tower (Bolt's, #791) and a Parakeet sound encoder. Target: images and audio
in, from the public GGUF pair `lmstudio-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF` (Q4_K_M text, BF16
mmproj; base `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`), on one RTX 3090.

## Run 1 - 2026-11-05

**Question:** what is already here, and what does the sound path need?

**Command:** reading llama.cpp 970b51ce7 (`tools/mtmd/models/{nemotron-omni,parakeet}.cpp`, `tools/mtmd/clip.cpp`,
`clip-impl.h`) against `inference-models-other/src/{nemotron_omni,nemotron_transcribe}` and
`inference-models-speech/src/parakeet`.

**Finding:**
- Vision: `nemotron_omni` (projector `nemotron_omni`, C-RADIO, GELU-erf MLP, `n_merge`, min/max pixels) is Bolt's
  path; the module has no audio at all.
- Sound in llama.cpp is projector `parakeet`, its encoder a FastConformer:
  - dw_striding subsampling, conv1d layers 0, 2, 3, 5, 6 (`a.conv1d.%d.*`), factor 8 asserted;
  - relative-position attention (`linear_pos`, `pos_bias_u/v`), with a 128-frame local window past 8192 frames;
  - the conv module (`conv_pw1`, `conv_dw`, `conv_norm`, `conv_pw2`) and half-step FFNs;
  - log-mel at 16 kHz, n_fft 512, window 400, hop 160, its filters and window stored in the GGUF.
- The sound projector after the encoder: RMSNorm (eps 1e-6, `mm.a.norm_pre`), then `mm.a.mlp.1` -> squared ReLU ->
  `mm.a.mlp.2`, into the decoder's hidden size.
- Our Parakeet port (#407) has this encoder (`parakeet/encoder.rs`); its GGUF loader reads NVIDIA's `asr` GGUF with
  NeMo names, not llama.cpp's `a.*` mmproj names.
- Nemotron 3.5 Transcribe's audio tower is a different encoder (a NeMo Transformer), not reusable here.

**Implication:** four pieces:
1. A binding from the mmproj's `a.*` tensors and keys onto our FastConformer `Encoder`.
2. The sound projector.
3. The inputs processor: log-mel, and as many sound placeholders as encoder frames.
4. Splicing sound embeddings into the decoder's inputs beside images.

Next: the mmproj's tensor inventory and metadata, then the encoder against llama.cpp on one clip, bottom-up.

## Run 2 - 2026-11-06 09:20

**Question:** does the published GGUF pair carry the sound encoder, and how far is Nano Omni's vision from Bolt's?

**Command:** gguf-py `GGUFReader` over the lmstudio mmproj; HF API listings of every published Nano Omni mmproj;
the base checkpoint's `config.json`, `preprocessor_config.json`, `model.safetensors.index.json`, `audio_model.py`,
`modeling.py`, `processing.py`, `image_processing.py`; llama.cpp 970b51ce7 `conversion/nemotron.py`, `clip.cpp`,
`mtmd.cpp`, `models/nemotron-v2-vl.cpp`.

**Finding:**
- The lmstudio mmproj is vision only: `clip.projector_type nemotron_v2_vl`, 390 tensors, no `a.*`, no audio keys.
  `v.position_embd` is 32 x 32 (the converter resizes RADIO's 128 x 128 CPE table to a fixed 512 input), mean/std are
  ImageNet's (the converter's default: the checkpoint names them `norm_mean`/`norm_std`, CLIP's values).
- Every other published mmproj (unsloth BF16/F16/F32, DevQuasar) is the same 1.59 GB vision-only file.
- llama.cpp runs `nemotron_v2_vl` as one fixed-size 512 image (`mtmd_image_preprocessor_fixed_size`), 256 tokens.
  The checkpoint's processor is dynamic resolution, 1024-13312 patches before the 2x pixel shuffle: the vLLM
  `DynamicResolutionImageTiler` algorithm, the same one Bolt's processor (#791) uses.
- Today's converter writes the Parakeet encoder into the mmproj when `sound_config` is present (`clip.audio.*`,
  projector `parakeet`, `mm.a.*`), but it registers `NemotronH_Nano_VL_V2` and `NemotronH_Omni_Reasoning_V3` only;
  the checkpoint now reports `NemotronH_Nano_Omni_Reasoning_V3`.
- All vision and sound weights are in shard 1 of 17 (4.0 GB): `vision_model` 390, `mlp1` 3, `sound_encoder` 710,
  `sound_projection` 3 tensors.
- Vision names are timm's `vision_model.radio_model.model.*` and `mlp1.*`, the names our C-RADIO (`radio.rs`)
  already reads. The config differs: Nano Omni ships RADIO's training config (`args.model vit_huge_patch16_224`,
  `cpe_max_size 2048`, `register_multiple 10`, 4 teachers with `cls_token_per_teacher`), where Bolt's has
  `hidden_size`, `num_cls_tokens 3`, `num_registers 7` etc. 4 cls tokens + 6 registers = the mmproj's 10 class rows.
- Sound in the reference is transformers' `ParakeetEncoder` + `ParakeetFeatureExtractor` (names match our port's),
  then `sound_projection`: RMSNorm (eps 1e-5 in `audio_model.py`), `linear1` 1024 -> 4096, squared ReLU, `linear2`
  -> 2688, no biases. Each clip becomes `<so_start>` + one `<so_embedding>` (id 27) per encoder frame + `<so_end>`;
  the template writes `<so_embedding>\n` per audio part.

**Implication:** two loading routes, each faithful to its reference:
1. Safetensors (HF semantics): register `NemotronH_Nano_Omni_Reasoning_V3` on the Omni loader, derive the vision config
   from RADIO's args, add the sound encoder and projector. Dynamic resolution as Bolt.
2. GGUF (llama.cpp semantics): read the `nemotron_v2_vl` mmproj as llama.cpp does (one 512 image), and its `a.*`
   sound tensors when present. No published file has them, so the check builds one with llama.cpp's converter from
   shard 1.

Next: shard 1, then the sound encoder against the torch reference on one clip, bottom-up.

## Run 3 - 2026-11-06 09:45

**Question:** does our Parakeet encoder, with the reference's projector, reproduce Nano Omni's sound path?

**Command:** `sound_ref.py` (scratch): transformers 5.19 `ParakeetFeatureExtractor(sampling_rate=16000,
feature_size=128)` and `ParakeetEncoder` loaded from shard 1's `sound_encoder.encoder.*`, then `sound_projection` by
hand, all f32 on the CPU, over `examples/assets/sample_speech.wav` (9.08 s, resampled to 16 kHz). Ours:
`nemotron_omni::sound` (`SoundTower`: `parakeet::Encoder` from the same tensors, RMSNorm 1e-5, linear1, squared ReLU,
linear2; mel through `NemoMel` with transformers' defaults, normalized), in a scratch test reading the reference.

**Finding:**
- The reference's features are 908 x 128 with an attention mask of 907: the last frame (STFT centre padding) is zero.
  `NemoMel` gives the 907 valid frames; on them the features differ by at most 7.6e-5.
- Encoder, from the reference's features: 114 x 1024, max abs diff 2.6e-6. From our own 907 frames, with or without a
  zero 908th: 114 frames either way (the three stride-2 convs round up), max abs diff 2.5e-6.
- Projected: 114 x 2688, max abs diff 6.7e-6.
- One sound token per encoder frame: ceil(ceil(ceil(907 / 2) / 2) / 2) = 114.

**Implication:** the tower is right as written; what remains is getting its weights in (no published GGUF has them)
and the prompt side (`<so_start>`, 114 x `<so_embedding>`, `<so_end>`; splicing beside images).

## Run 4 - 2026-11-06 10:40

**Question:** with the towers read from shard 1 beside the published text GGUF, does Nano Omni answer images and
audio end to end?

**Changes:**
1. `GgufArchive::open_safetensors`: a `.safetensors` path among the projector files is a component of its own, its
   float tensors under their own names (integer buffers, a batch norm's `num_batches_tracked`, left out), reporting
   projector type `checkpoint`.
2. Family `NemotronOmniCheckpoint` (`nemotron_h_moe` + `checkpoint`): the decoder bound as for Bolt, every
   `vision_model.` / `mlp1.` / `sound_encoder.` / `sound_projection.` tensor bound to itself. The config is the
   repository's (`--tok-model-id`), since the shard has no metadata.
3. `Config::parse` fills Nano Omni's RADIO training config in as Bolt's shape (`vit_huge_patch16_224` -> 1280 / 16
   heads / 32 blocks / 4.0; 4 teachers -> 4 cls + 6 registers; `cpe_max_size` 2048) and `vision_final_layernorm`
   false, which its reference has none of. `NemotronH_Nano_Omni_Reasoning_V3` is an alias of the Omni loader.
4. Sound: `sound_config` + `sound_context_token_id` build the tower; the processor expands images and clips in one
   pass (one prompt rewrite, one prefix-cache layout with both kinds).

**Command:** `inference serve --format gguf -m nemotron-nano-omni-gguf -f ...Q4_K_M.gguf --mmproj
nemotron-nano-omni-bf16/model-00001-of-00017.safetensors --tok-model-id nemotron-nano-omni-bf16` (CUDA, RTX 3090,
defaults: the 24.5 GB text GGUF spills experts to pageable host memory), then chat requests (greedy, thinking off).

**Finding:**
- Audio (`examples/assets/sample_speech.wav`, "Transcribe this audio."): word for word what the clip says, 138 prompt
  tokens (114 of them sound).
- Image (the PaddleOCR-VL table fixture, 470 x 152): the three headers and both rows read back correctly, 311 prompt
  tokens.
- Image and audio in one message: with "What text is in the image, and what is said in the audio? Answer both." both
  come back right. With "First read the text ..., then transcribe the audio." it answered only the image; the clip's
  tokens are there (416 prompt tokens), the model chose one.
- An image-only request first failed: the processor asked for the active clip range of a sequence with no clips.
  Fixed (only sequences with clips ask).
- Speed at these defaults: first request prompt 13 tok/s (cold), then 80-216 tok/s; decode 29 tok/s cold, ~80 warm.
  Not tuned here.

**Implication:** the route works. Left for #565 on Nano Omni: video (`video_embedder`, EVS pruning), and a speed pass
for the 24.5 GB GGUF on one 24 GB GPU.

## Run 5 - 2026-11-06 23:10

Question: what does the reference do with video, and what does it need from this engine?

Read: the checkpoint's `modeling.py` (`extract_video_feature`, `generate`), `processing.py` (the `<video>`
expansion), `image_processing.py` (`_compute_target_patches_video`), `evs.py`, and `video_processing.py`, which the
first download lacked (fetched alone from nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16). Config:
`video_temporal_patch_size` 2, `video_target_num_patches` 1024, `video_maintain_aspect_ratio` true,
`video_pruning_rate` 0.7, `separate_video_embedder` true; the tower shard holds
`vision_model.radio_model.model.patch_generator.video_embedder.weight`.

- Frames: each resized (torch bicubic, antialias, align_corners false) to ~1024 16-px patches with the aspect ratio
  kept, each side snapped to the pixel shuffle's 2 (round up if the product still fits 1024, else down), then
  `(x/255 - mean) / std` with the image path's norm.
- Tubelets: frames in pairs (the last repeated to make the count even), stacked on the channel axis
  `(N/2, 2*3, H, W)`; RADIO runs with `video_embedder` (Linear 2*3*16*16 -> width) in place of the patch `embedder`,
  everything else as for an image. Then the image path's pixel shuffle and `mlp1`: (wp*hp)/4 tokens a tubelet.
- Prompt: per tubelet `"Frame {i} sampled at {t:.2f} seconds and frame {i+1} sampled at {t':.2f} seconds: "`
  (t = source frame index * int(1000/fps) / 1000; plain "Frame i" without fps), then `<img>`, the tokens as the image
  token (id 18; `video_context_token_id` 131081 decodes to nothing), `</img>`; tubelets joined by "\n".
- EVS (rate 0.7): over the projected tokens, keep all of the first tubelet and, of the rest, those least like the
  same position one tubelet earlier (1 - cosine), `max(one tubelet, int(T * tokens * 0.3))` in all, stable order; the
  pruned positions leave `inputs_embeds` and `input_ids` before the language model runs.

Implication: everything but EVS fits the image route (a second patch embedder, a frame resize rule, a prompt
layout). EVS does not: the prompt's length depends on the tower's output, and here the input processor fixes the
tokens before any forward (the encoder cache is the model's, inside the forward). With rate 0 the reference keeps
every token, a mode it supports. Plan: video with no pruning first, against torch at rate 0; EVS needs a step that
encodes a request's video before its prompt is final.

## Run 6 - 2026-11-06 23:40

Question: does our video path (no EVS) match transformers', and does the model see a video end to end?

Built: `video_embedder` in the tower (a tubelet's stacked frames in place of the patch embedder); the processor's
frame grid (`video_grid`), tubelet patches (each patch's frames as consecutive 768-wide rows of the cached pixel
tensor, the last frame repeated), labels (`tubelet_labels`), and one ordered pass over `<image>` and `<video>`
(`expand_visual`) that records each `<img>` block's kind; tubelets are their own multimodal items (kind Video, hash of
the video's and the index), so a prefill chunk can split a video between tubelets, and a step's images and tubelets
reach the tower in prompt order. Video sampling 2 fps, at most 128 frames. EVS is logged as not applied.

Reference: `scripts/nemotron_omni_video_parity.sh` (transformers 5.19's native `NemotronH_Omni_Reasoning_V3`
video processor and `replace_video_token`, `RadioModel` with `video_patch_projection` from shard 1, the projector by
hand; transformers 5.19 renamed RADIO's attention to `attention.{q,k,v,o}_proj`), f32 on CUDA, five synthetic 352 x
240 frames at fps 30, stride 15:

```
reference: 5 frames, 3 tubelets of 26 x 38 patches
`<video>` text: identical (5393 bytes; "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.49 seconds: ", ...,
                "Frame 5 sampled at 1.98 seconds: ")
patches: max diff 5.7e-5, cosine 1.0000000 (the fifth frame repeated into tubelet 2)
tubelets 0-2: features cosine 1.0000000, embeds 1.0000000 (max diff 2.3e-3 to 3.9e-3 of ~9), from our patches the same
```

End to end (Q4_K_M GGUF + shard 1, `--tok-model-id` the checkpoint), a synthetic 4 s clip at 8 fps (a yellow square
crossing a blue field left to right), as a `video_url` data URI: 1,152 prompt tokens (8 frames sampled, 4 tubelets of
256), 22 s. The reasoning: "Background: Solid blue color. Object: A yellow square ... The yellow shape starts on the
left side. It moves horizontally to the right." Found on the way: a chunked prefill with no images asked for the
images' active range and failed ("chunk is missing its active Image range"); a kind with no items is an empty range.

Next: EVS, which needs the video encoded before the prompt is final.

GGUF route: llama.cpp's published projector (`mmproj-...-BF16.gguf`) has no video patch embedder (only
`v.patch_embd.weight`), so a load from it offers no video. Ours (`inference quantize mmproj`, #833) writes it as
`v.patch_embd.weight.1`; the config synthesized from a projector now declares the video layout when that tensor is
there (frames per tubelet from its shape, the reference's 1024-patch frame target, which projectors do not state).
Q4_K_M + our projector, the same clip: video offered, 1,152 prompt tokens, "A blue background. A yellow square is
partially visible on the left edge ... moved towards the center ... moved further right".

## Run 7 - 2026-11-07 00:15

Question: EVS. Can a video's tokens be pruned as the reference does, when the pruning depends on the tower's output
and the prompt is otherwise fixed before any forward?

Built: `MultimodalModel::encode_prompt_media`, which admission calls before it expands a request's prompt (the
pipeline is held there already); its result rides on the sequence (`MultimodalData::encoded_media`). Nano Omni, with
a pruning rate above 0, runs each video's tubelets through the tower there, computes the retention (`evs.rs`: the
first tubelet kept at dissimilarity 255, `1 - cosine` to the same position one tubelet earlier, a stable descending
sort, `max(N, int(T * N * (1 - q)))` kept, positions in order) and keeps each tubelet's kept embeddings. The prompt
then gives each tubelet's block its kept count; a tubelet that kept nothing keeps its label and an empty `<img></img>`
and is no multimodal item; a step hands the kept embeddings to the decoder in place of the tower's. Video sampling now
2 fps up to 256 frames (the model card's launch).

The reference reshapes each tubelet to a square of `sqrt(N)` a side before comparing; that holds only for square
tubelets (a 16:9 frame's 252 tokens fail it), and the comparison is per position, so ours runs on flat positions.

Parity (`nemotron_omni_video_parity.sh`, now also transformers' `compute_retention_mask` at the checkpoint's 0.7 on a
`(T, 1, N)` grid):

```
5 synthetic frames, 3 tubelets of 26 x 38: reference keeps 247, ours 247 on its embeddings and on ours (all agree)
16 frames of the moving-square clip (8 fps, stride 2), 8 tubelets of 24 x 42: reference keeps 604, ours 604 on its
    embeddings and on ours (1.0000 agree); `<video>` text identical (14,766 bytes); lowest cosine 0.9999998
```

Tiny (`a_pruned_video_keeps_its_first_tubelet_and_decodes_the_same_again`, F32 CPU): rate 0.5 on two 2-token
tubelets keeps the first tubelet's 2 (the second block empty), 2 prompt tokens fewer than without pruning, and a
cached repeat decodes the same.

End to end (Q4_K_M + shard 1, the 4 s clip): 446 prompt tokens against 1,152 without EVS, 14.0 s against 22.3 s, the
reasoning still "a yellow square ... on the left ... has moved". A repeat hit 384 cached tokens in 4.0 s; its
reasoning text differed from the first's (BF16, the cached and uncached passes split the prompt differently; the F32
tiny repeat is exact).
