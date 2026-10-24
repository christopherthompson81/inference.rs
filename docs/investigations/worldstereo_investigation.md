# WorldStereo-2 native port

Goal: run Tencent HY-World 2.0's WorldStereo-2 (panorama -> generated keyframes along camera trajectories -> 3D world)
natively in inference.rs on one 24 GB RTX 3090, with quantization chosen and its quality cost measured. The user chose a
native port over running the reference Python with offload (2026-10-24): fitting, speed and quality control all need
the engine. Reference: the Tencent repo's `hyworld2/worldgen` (df9988e) in a local venv; WorldMirror 2.0 is already
native (`hyworld_investigation.md`).

## Run 1 - 2026-10-24 14:30

Question: what does the pipeline consist of, and which parts are model inference?

Read `video_gen.py`, `models/{worldstereo,controlnet,attention,dmd_scheduler,worldstereo_wrapper}.py`,
`models/pipelines/{pipeline_dmd_keyframe,_pipeline_common}.py`, `src/vae_utils.py`, plus the checkpoint's safetensors
header (fetched by range request) and the HF file listings.
```
worldstereo-memory-dmd/model.safetensors  34.86 GB, 17.43B params, all BF16:
  blocks 16.154B (40 blocks)  controlnet 0.998B  condition_embedder 0.240B  camera_embedding 0.039B
  patch_embedding, proj_out, scale_shift_table
Wan2.1-I2V-14B-480P-Diffusers: transformer 65.58 GB (fp32), text_encoder 22.72 GB (umT5-XXL fp32),
  image_encoder 1.26 GB (CLIP ViT-H), vae 0.51 GB
facebook/sam3 6.90 GB (gated; model.safetensors and sam3.pt are the same weights twice)
moge-2-vitl-normal 1.32 GB; Qwen3-VL-8B-Instruct 17.55 GB (trajectory planning and captions)
```
Findings:
- The WorldStereo checkpoint holds the complete transformer. The reference builds the module from the base model's
  65.6 GB fp32 transformer and then overwrites every weight; a native loader never needs it.
- Stages: WorldNav plans trajectories (VLM + navmesh), point-cloud renders along them, WorldStereo generates 21
  keyframes per trajectory conditioned on renders, cameras, caption and memory frames; MoGe-2 depth and a SAM3 "sky"
  mask fold each clip back into the scene point cloud; finally WorldMirror on all frames, then 3DGS training.
- SAM3 is only ever prompted with "sky": a small sky segmenter can replace it.
- Keyframe mode: every frame is VAE-encoded and decoded alone as a 1-frame video, so the Wan 3D-causal VAE only runs
  single frames (a 2D VAE in effect).
- DMD sampling: 4 transformer forwards, no CFG. Per forward ~32.8K video tokens (21 x 30 x 52 at 480x832) plus up to
  8 memory frames (~12.5K) in joint attention over 40 blocks of width 5120: roughly 4 min per clip on a 3090.
- Architecture notes in scratchpad `worldstereo_spec.md`.

Plan, bottom-up with per-component parity against the reference modules: (1) Wan VAE single-frame encode/decode,
(2) CLIP ViT-H penultimate hidden states, (3) umT5-XXL encoder, (4) the transformer: Wan block with joint reference
attention, ControlNet, camera and mask embeddings, block by block, (5) quantization of the blocks with measured cost,
(6) DMD sampler end to end on dumped reference inputs, (7) the orchestration around it (cameras, point-cloud renders,
memory bank, MoGe-2, sky mask, trajectories).

## Run 2 - 2026-10-24 14:40

Question: does a single-frame port of the Wan VAE reproduce the reference keyframe encode/decode?

`wan/vae.rs` (inference-models-diffusion): every causal 3D conv as the 2D conv of its last temporal kernel slice
(a lone frame sees zero padding in the other two taps), temporal resamplers dropped (the first chunk skips them),
channel-L2 norm x sqrt(C) x gamma, nearest 2x upsampling, single-head mid-block attention, posterior mode = the mean
half. Reference dump (`dump_wan_vae.py`, scratch): AutoencoderKLWan encoding a Park photo at 832x480 and the zero
image one frame at a time, normalized, then decoding, in f32 and in bf16 under autocast as the pipeline runs it.
```
wan::parity::vae_matches_the_reference
f32  latents: max 1.9e-3 rel rms 6.6e-4    decoded (from the reference latents): max 2.9e-3 rel rms 3.6e-4
bf16 latents: max 2.1e-2 rel rms 5.3e-3    decoded: max 3.7e-2 rel rms 4.7e-3
```
f32 differs at the level of the reference's TF32 cuDNN convs; bf16 at bf16 rounding. Next: CLIP ViT-H.
