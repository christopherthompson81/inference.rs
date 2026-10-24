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
earlier clips is batched per model (every caption through umT5 up front, then freed). Reading `retrieval_wm.py`
corrected an assumption: between clips the memory bank only appends the generated frames and their cameras
(`update_memory`); retrieval picks memory frames by frustum overlap (DINOv2 features break ties past 8), and depth
(WorldMirror, MoGe-2) and sky masks (SAM3) run once after every clip. So the stages are: panorama depth, sky and point
cloud, trajectories and renders, captions (all before generation); umT5 for every caption; the clip loop with VAE, CLIP
and the transformer resident; then, with the transformer unloaded, WorldMirror, alignment and the scene point cloud.

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

## Run 5 - 2026-10-24 15:36

Question: does the real-size forward fit in 24 GB, at what speed, and does umT5 match?

Real-size step (`full_size_streamed_step`, random inputs, 21 latent frames at 60 x 104, 8 memory frames: 32,760 video
+ 12,480 memory tokens), the 40 bf16 blocks streamed from host memory:
```
first try        CUDA out of memory: the ControlNet kept all 20 residuals ([1, 32760, 5120] bf16 each, 6.7 GB)
                 (5 frames: 6.3 GB in use after the ControlNet, 11 frames: 8.1 GB, growing with frames)
interleaved      ControlNet block i runs just before main block i, its residual used at once
                 21 frames + 8 memory: 55.5 s per step, ~5.5 GB in use between blocks
block upload     0.075 s per block (~11 GB/s from the page cache), ~3 s of each step
```
A 4-step clip is ~3.7 min of transformer time.

umT5-XXL (`text_encoder_matches_the_reference`; `dump_wan_text.py` on the CPU, the checkpoint stores only
`shared.weight`, which `embed_tokens` ties to as the wrapper patches): token ids identical (31 tokens with EOS);
our bf16 encoder against the reference's f32 states rel rms 2.0e-2, the reference's own bf16-autocast run 1.5e-2 from
its f32 run. The pipeline's padded 512-token prompt with a mask equals the real tokens encoded alone, then zeros.

## Run 6 - 2026-10-24 16:23

Question: does a whole clip (I2V condition, render and memory latents, 4 DMD steps, decode) match the reference
pipeline end to end?

`dump_ws_clip.py` (scratch) runs the reference's own `RefKFDMDGeneratorPipeline` at real size (21 frames at 480 x 832,
2 memory frames at ref_index [3, 12]) with its 40 blocks streamed by forward hooks (built on the meta device and
assigned the bf16 tensors: an f32 construction was OOM-killed at 111 GB of host memory). Inputs are synthetic but
structured: a sample-repo photo as the start frame, renders that shift it 6 px more each frame with masked holes, two
other photos as memory frames, a synthetic trajectory; fixed noise; prompt states from the umT5 dump. 219 s.
```
wan::parity::clip_matches_the_reference (ours: VAE and CLIP in f32, transformer bf16 streamed), denoise 167 s
after step 0: rel rms 3.2e-3   step 1: 7.6e-3   step 2: 2.4e-2   step 3 (final latents): 1.04e-1   frames: 7.7e-2
reference against itself, FlashAttention 2 vs PyTorch SDPA (NO_FLASH=1 rerun):
after step 0: rel rms 7.8e-4   step 1: 2.0e-3   step 2: 7.9e-3   step 3: 3.9e-2                     frames: 3.1e-2
CLIP in bf16: final 9.3e-2; VAE in bf16: final 1.01e-1 (neither is the source)
```
The 4-step distilled sampler amplifies any difference ~30-50x over the steps, for the reference's own attention backend
swap as for us; our starting difference is ~4x the backend swap's (f32 VAE and CLIP against the reference's bf16
autocast). Side by side (frames 0, 10, 20: render, reference, ours) both fill the render holes with the same content,
including a building and road that exist only in the generation. Parity at the level the sampler allows.

## Run 7 - 2026-10-24 16:30

Tier 1 chosen (user: "Proceed" on the core loop: panorama depth and sky, simple trajectories, renders, memory,
WorldStereo, then WorldMirror on the frames; no VLM, SAM3, navmesh or splat training yet).

Question: does a native MoGe-2 (ViT-L with normals) reproduce `MoGeModel.infer` at a known field of view?

`moge/mod.rs` (inference-recon): DINOv2 ViT-L/14 without registers, now shared with WorldMirror's (`dinov2.rs`, with
the hub's plain bicubic position interpolation at offset 0.1 beside WorldMirror's antialiased one); intermediate layers
5/11/17/23 projected and summed; the ConvStack neck and heads (stride-2 transposed convs and bilinear, replicate-padded
3x3 convs); exp remap; the metric-scale MLP on the class token; the z-shift fit at the given focal on the 64 x 64
nearest-downsampled map (Levenberg-Marquardt here, scipy's in the reference); depth back through the intrinsics.
Weights converted once from `model.pt` to safetensors beside it. Reference `dump_moge.py` (scratch): a 512 x 512 crop at
fov_x 90, as pred_pano_depth calls it (bf16 autocast outside, fp16 inside) and in plain f32.
```
moge::parity::moge_matches_the_reference
f32 trunk vs reference f32:   depth rel rms 4.1e-4, normal 2.6e-4, mask flips 0.0000%
f32 trunk vs reference amp:   depth rel rms 3.1e-3, normal 6.5e-4, mask flips 0.0004%
bf16 trunk vs reference f32:  depth rel rms 1.16e-2, normal 5.5e-3, mask flips 0.0019%
```
(depth max differences sit on the far sky). The trunk runs in f32: 300M parameters, ~20 views a panorama.

## Run 8 - 2026-10-24 16:37

Question: does a native `pred_pano_depth` (split, MoGe-2 per view, gradient-domain merge) reproduce the reference?

`worldstereo/panorama.rs`: views at the directions of a once-subdivided icosahedron (utils3d's), OpenCV look-at with
up +z; the two pole directions make that look-at degenerate, and the reference's cameras there are NaN (verified:
views 12 and 22 of 42), so they see nothing and are skipped: 40 views. Each view as split_panorama_image(INTER_AREA)
makes it (remap at half the panorama height with OpenCV's 1/32 sub-pixel steps, then INTER_AREA's enlarging
coefficients to 512); MoGe-2 at fov 90; the merge as merge_panorama_depth_gpu (warped log distance, masked x/y
gradients wrapping in x, 5-point Laplacian wrapping in x and edge-replicated in y, averaged over views) solved by LSMR
(SciPy's algorithm, atol = btol = 1e-5) over the stencils, coarse to fine from 240 x 120; hole fill at the 0.999
quantile; south-pole smoothing. Reference `dump_pano_depth.py` (scratch) on the sample repo's `case000` (1920 x 960),
needing cupy/open3d/trimesh in the venv.
```
worldstereo::panorama::parity::pano_depth_matches_the_reference
split views:        10.3% of values differ, max 5 levels (OpenCV's fixed-point remap and resize)
merge alone (reference views and masks): distance rel rms 2.4e-4, max 1.1e-2; mask flips 0     16.7 s (CPU)
full pano depth (our views, our MoGe):   distance rel rms 7.6e-4, max 6.8e-2                   41.2 s
reference: 32 s
```
The merge's LSMR on the CPU is the slow part; a later move to the GPU.

## Run 9 - 2026-10-24 16:49

Question: do the native scene preparation, regular trajectories and point renders reproduce the reference's?

`worldstereo/{scene,trajectory,render,memory}.rs`. Reference scripts (scratch) call the reference's own functions on the
pano-depth dump: `dump_ws_scene.py` (traj_generate Stage 1 inline: edge mask, 0.99 clip, outdoor contraction, point
cloud, mesh, start views, get_c2w with the mesh and the kd-tree) and `dump_ws_render.py` (point_rendering through
PyTorch3D, built from source into the venv). Sky is "not valid in any MoGe view" on both sides (the reference
segments it with GroundingDINO + ZIM), and each view gets fresh moves on both sides: the reference mutates its shared
aerial move in place when it collides, which would shrink later views' aerial moves.
```
scene_and_trajectories_match_the_reference
cleaned distance max diff 0; excluded flips 0; points 1,826,759 both; median 1.00920 both
start views: 0.012-0.016% of values differ (max 4 levels); view median depths identical (1.10703, 1.00569, 0.71590)
all 9 trajectories: 21 poses, max |c2w diff| <= 1.0e-6, with the reference's shortenings (0 to 3 per trajectory)
renders_match_the_reference (CPU, tiles of 16 px, top 20 per pixel)
view0.traj2: 0.95 s / 21 frames; hole-mask flips 0.0001%; rgb where both cover rms 1.5e-4 (max 0.23)
view1.traj0: 0.83 s / 21 frames; hole-mask flips 0.0008%; rgb rms 1.7e-4 (max 0.12)
```
Collision checks reproduced a reference detail: only the first check measures distance to the mesh; the checks after
a shortening use the mean distance to the 5 nearest cloud points.
