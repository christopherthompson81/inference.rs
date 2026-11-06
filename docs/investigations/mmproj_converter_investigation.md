# Projector converter: checkpoints to llama.cpp-style mmproj GGUFs

A general writer for multimodal projectors (`inference quantize mmproj`): one mmproj per checkpoint with every tower it
has (vision, audio), in llama.cpp's tensor names and keys, so the same file loads here and in llama.cpp. First family:
Nemotron 3 Nano Omni (#565), whose published mmprojs are vision-only and cut RADIO's position table to 512 pixels.

## Run 1 - 2026-11-06 10:05

**Question:** what does llama.cpp itself write for Nano Omni with audio, and will llama.cpp load a file with Bolt's
full-resolution `nemotron_omni` vision beside `parakeet` audio?

**Command:**
1. llama.cpp 970b51ce7 `convert_hf_to_gguf.py --mmproj --outtype bf16` on a scratch copy of shard 1: `architectures`
   rewritten to `NemotronH_Nano_VL_V2` (the converter does not register `NemotronH_Nano_Omni_Reasoning_V3`), the index
   cut to shard 1's tensors.
2. `llama-mtmd-cli` (build_cuda, Q4_K_M text GGUF, `-ngl 99 --n-cpu-moe 24`, greedy) on it, then on a probe rewritten
   from it with gguf-py: `clip.vision.projector_type nemotron_omni`, `clip.vision.nemotron_omni`, image min/max pixels
   1024 / 13312 patches of 256, CLIP mean/std, and `v.position_embd.weight` replaced by shard 1's full 16384 x 1280
   `pos_embed`.

**Finding:**
- llama.cpp's file: 1079 tensors, 3.0 GB. `clip.has_vision_encoder` and `clip.has_audio_encoder`, per-encoder
  `clip.vision.projector_type nemotron_v2_vl` / `clip.audio.projector_type parakeet`; audio keys `num_mel_bins 128`,
  `attention.layer_norm_epsilon 1e-5`, `subsampling_factor 8`, `conv_kernel_size 9`, `projection_dim 2688`,
  `embedding_length 1024`, `feed_forward_length 4096`, `block_count 24`, `head_count 8`.
- Its audio tensors (GGUF order, ne0 first):
  - subsampling: `a.conv1d.{0,2,3,5,6}.weight` F32 [3,3,1,256] / [1,1,256,256], biases [1,1,256,1];
    `a.pre_encode.out` BF16 [4096,1024] + F32 bias;
  - per block: `attn_{q,k,v,out}`, `linear_pos` BF16; `pos_bias_{u,v}` F32 [128,8]; `ffn_up`/`ffn_down` (first
    half-step FFN) and `ffn_up_1`/`ffn_down_1` (second) BF16; norms `ffn_norm`, `ln1`, `norm_conv`, `ffn_norm_1`,
    `ln2`; conv `conv_pw1` F32 [1024,2048], `conv_dw` F32 [9,1024,1,1], `conv_pw2` F32 [1024,1024], `conv_norm`
    weight/bias with `conv_norm_mean` / `conv_norm_var` (the batch norm unfolded);
  - `a.mel_filters` F32 [257,128,1], `a.window` F32 [400]; projector `mm.a.norm_pre`, `mm.a.mlp.1`, `mm.a.mlp.2`.
- Its vision is the published files': `nemotron_v2_vl`, `v.position_embd` [1280,1024,1] (32 x 32), ImageNet mean/std.
- llama.cpp on its own file: CUDA OOM at `--n-cpu-moe 12` (the 3 GB projector beside the experts), fine at 24. Audio:
  the clip word for word, as ours. On the published vision-only mmproj the table image reads correctly.
- llama.cpp on the probe: loads; the table image reads correctly (now at dynamic resolution, `nemotron_omni`'s erf
  GELU), the clip transcribes word for word. A first probe failed (`tensor 'mm.model.mlp.3.weight' has offset ...`):
  my rewrite gave BF16 tensors twice their bytes; writing them as gguf-py's `gguf_new_metadata.py` does fixed it.

**Implication:** the target file is llama.cpp's audio half plus Bolt's vision half, and llama.cpp runs it. The
converter writes that; our loader needs a binding for `parakeet` audio (`a.*`) beside Bolt's `nemotron_omni` one.

## Run 2 - 2026-11-06 11:15

**Question:** does our writer's file match llama.cpp's conversion where the two layouts agree, and do both engines run
it?

**Changes:**
1. `mmproj_writer`: a family spec is the file's keys and, per tensor, a checkpoint tensor (copied, reshaped, or rows
   of a fused one) with a role (matrices at `--matrix-type` or the checkpoint's float type; norms, biases, convs,
   embeddings in f32, as llama.cpp's converter keeps them). Shards an index lists but that are absent are skipped.
2. `parakeet_audio_bindings`: one table of llama.cpp names against transformers' Parakeet names, with each reshaped
   kind's shape rule both ways, drives the writer's spec and the binding.
3. Nano Omni's spec beside Bolt's `bind_vision`, from the same `BLOCK_ROLES`; Bolt's config synthesis adds
   `sound_config` and the `<so_embedding>` id when the projector has `parakeet` audio.
4. The checkpoint's chat template goes in the file (`clip.chat_template`): the published text GGUF's template has no
   media parts (10504 bytes against the checkpoint's 14277; no `<image>`, no audio branch), so with it alone every
   image or clip request failed the placeholder count. The GGUF loader takes a projector's template in place of the
   text GGUF's.
5. A projector naming its type per encoder (`clip.vision.projector_type`) now counts as describing its own config;
   only `clip.projector_type` did.

**Command:** `inference quantize mmproj -m nemotron-nano-omni-bf16 -o ours-mmproj-omni.gguf` (shard 1 of 17 on disk),
gguf-py diff against Run 1's probe, then `llama-mtmd-cli` and `inference serve` on the text GGUF + our file.

**Finding:**
- 1208 tensors, 2934 MiB, 8.5 s. Against the probe (1079): every shared tensor byte-identical in type, shape and data,
  audio included. Ours alone: q/k/v split per vision block (Bolt's layout) and `v.patch_embd.weight.1` (the video
  embedder); the probe alone: `attn_qkv` (`nemotron_v2_vl`'s fused layout). Keys: the probe's extra ones are
  `general.*` labels only.
- llama.cpp on our file: the table image and the clip, both right.
- Ours with no `--tok-model-id`: audio word for word (138 prompt tokens), the table (311), image + audio answered
  both (420): the shard route's counts exactly.
- Greedy traces of the image + audio request, 40 tokens: the same tokens on both routes; logprobs differ by up to 0.48
  (one early low-confidence token), but the same route differs by 0.24 between server restarts and 0.22 between
  repeats, so the end-to-end trace cannot tell the files apart. The tiny round trip below does.
- `the_towers_written_as_an_mmproj_bind_back_unchanged`: the tiny Omni checkpoint's towers written, bound back
  through `build_nemotron_omni_tower_bindings`, every tower tensor equal to the checkpoint's (diff 0.0); the template
  carried.

**Implication:** one file now serves both engines at the reference's fidelity. Next families for the writer:
Transcribe's audio GGUF (today a one-off writer) and the checkpoints our other bindings read.
