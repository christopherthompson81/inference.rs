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

Orchestrator requirement (user, 2026-10-24): it loads and unloads models by available VRAM, prerequisite data and
timings. Each stage declares its model, measured footprint, inputs and typical duration; work that does not depend on
earlier clips is batched per model (every caption through umT5 up front, then freed), while the per-trajectory loop is
sequential (clip k's memory retrieval needs clip k-1's depth alignment), so the transformer stays resident through it
and the small helpers (VAE, MoGe-2, sky mask) sit beside it or swap in by measured headroom.

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

## Run 3 - 2026-10-24 14:47

Question: does the shared CLIP vision tower reproduce the Wan I2V image condition?

`wan/clip.rs`: inference-nn's `ClipVisionTransformer` (exact GELU added for OpenCLIP ViT-H), output of layer 30 of 32
(the pipeline's `hidden_states[-2]`). Preprocessing found by reading the wrapper: `CLIPImageProcessor` is built with
`do_rescale=False`, so the tower gets PIL-bicubic 224x224 pixels in 0..255 normalized as if they were 0..1 (values
2..944 on the test image). The model is tuned on that input, so the port reproduces it. Dump `dump_wan_clip.py`
(scratch): processor pixel values and the penultimate states in f32 and under bf16 autocast (the pipeline's mode).
```
wan::parity::image_encoder_matches_the_reference
pixel values (ours from the JPEG): max 10.9 (~3 levels) rel rms 4.0e-3; PIL in Python reproduces them exactly, so this
  is the image crate's JPEG decode against libjpeg-turbo, as in WorldMirror
f32 penultimate (on the reference pixels): max 8.1e-4 rel rms 1.9e-5
bf16 (whole tower in bf16) against the reference's autocast run: rel rms 9.4e-2
reference autocast against reference f32: rel rms 1.9e-1
```
The 0..255 inputs make the reference's own bf16 run 19% off its f32 run; ours in bf16 sits inside that noise. The
encoder runs in f32: one 632M-parameter forward per clip, and the precise values rather than another draw of rounding.
Next: umT5-XXL.

## Run 4 - 2026-10-24 15:14

Question: does the native transformer (Wan I2V blocks with joint memory attention, camera tokens, ControlNet,
embedders, output head) reproduce the reference forward?

`wan/{ops,transformer,controlnet,sampler}.rs`. Reference: `dump_ws_small.py` (scratch) builds WorldStereoRefSModel
with the checkpoint's first N blocks, the full ControlNet and embedders, on small random inputs (3 latent frames at
16 x 24, 2 memory frames at ref_index [0, 1], t = 937.5), under bf16 autocast with FlashAttention 2, as `video_gen.py`
runs it. (It needs RANK/WORLD_SIZE set for its import-time process group, and loguru/decord in the venv.)
```
wan::parity::small_transformer_matches_the_reference
2 blocks: prediction max 6.25e-2 rel rms 4.7e-3     (prediction rms 1.83)
6 blocks: prediction max 6.25e-2 rel rms 5.0e-3     (prediction rms 2.10)
```
bf16-level agreement that does not grow with depth. Also: the DMD schedule matches the reference's printed timesteps
(1000, 937.5, 833.3, 625), and the camera inputs (`worldstereo::cameras`, ray embedding and quaternion + translation)
match the reference functions on a synthetic 21-frame trajectory to 1.3e-6.

Fitting 24 GB: each forward is compute-bound (joint attention over ~45K tokens in 40 blocks), so the bf16 blocks can
stay in host memory and stream in one at a time (`WorldStereoTransformer::new_streamed`); quantization becomes a speed
option rather than a requirement. Next: time a real-size streamed step.
