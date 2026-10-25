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

## Run 10 - 2026-10-24 16:54

Question: does memory retrieval pick the reference's frames?

`dump_ws_retrieval.py` (scratch, CPU) calls the reference's `PanoramaMemoryBank.retrieval` on a bank object holding the
27 panorama-bank cameras (no images: the selector only runs past 8 distinct frames), for view0's aerial clip, then adds
that clip's kept frames (`update_memory`: frames 3, 6, 9, 12, 14, 16, 18, 20) and retrieves for view0's right turn,
which takes the shortcut path (forced aerial frames at targets 1, 2, 3, 5, 7, 9, 13, 17).
```
retrieval_matches_the_reference
view0/traj2: per frame identical, ref_index [0, 2, 11] both
view0/traj0: per frame identical (shortcut frames 27-34 where forced), ref_index [0, 3, 5, 6, 14, 16] both
```
`inference worldgen` wires the stages (scene, trajectories, umT5 caption, clip loop, WorldMirror on the bank with
camera priors); the first full run is on the sample repo's indoor `case000`.

## Run 11 - 2026-10-24 17:39

Question: does `inference worldgen` run a whole panorama (the sample repo's indoor `case000`, 1920 x 960) to clips and a
reconstruction on the 3090, and where do time and memory go?

`ws_run.sh` (scratch): the CLI with a 0.5 s nvidia-smi sampler; prompt written by hand for the scene (the reference
captions each trajectory with Qwen3-VL), `--scene indoor`.
```
first run    7 of 9 clips, then CUDA out of memory (wall 1667 s, ~3.5 min a clip). Memory sat at ~22.1 GB from
             the caption stage on: umT5's 11.4 GB went back to the CUDA memory pool, not the driver, and the clip loop ran
             in a pool full of blocks cut for umT5 until one allocation found no room
fix          the pool trimmed (cuMemPoolTrimTo 0) after each stage drops its models and after every clip; finished
             clips are loaded from disk instead of generated again (the reference's skip_exist)
second run   the last 2 clips, then WorldMirror on the 99 bank frames at 832 (the reference's size): out of memory
             (~200K tokens; its four trunk feature maps alone ~6.5 GB)
third run    WorldMirror at 518 (99 frames, ~77K tokens): forward 23.2 s, peak 12.3 GB above idle
stages (third run; clips loaded from disk): scene 50.5 s, trajectories 14.2 s, umT5 33.3 s, clip models 37.9 s,
reconstruction 30.8 s
```
Reconstruction check (WorldMirror's predicted cameras against the trajectory cameras it was given as priors, similarity
fit over the 72 generated frames): position residual median 2.1 cm (max 8.0 cm) against a 69 cm mean camera spread;
rotation residual median 1.0 deg (max 4.1). Outputs: 2.0M points, 4.9M Gaussians (318 MB standard 3DGS PLY).
Next: write the reconstruction-to-scene transform into meta.json, per-clip timing on a clean run, the outdoor sample.

## Run 12 - 2026-10-24 18:19

Question: clean per-stage time and memory on a fresh run of the sample repo's second indoor panorama (`case001`, 1920 x
960), with the scene type decided automatically.

`ws_run.sh` with a hand-written prompt, no `--scene` (decided indoor by sky share).
```
scene (MoGe-2 panorama depth)     43.5 s    3037 MiB in use at its end
trajectories                      51.9 s    (case000: 14.2 s; more kNN rechecks on this layout)
caption (umT5-XXL)                38.4 s
clips (WorldStereo-2)           1861.6 s    9 clips, 189 frames, 3:15 to 3:35 a clip
reconstruction (WorldMirror 2.0)  30.5 s    at 518
wall 2027 s; nvidia-smi peak 21989 MiB above idle
nvidia-smi during the clip loop: 21.7 GB after clip 1, creeping to 23.5 GB by clip 7, dropping after the per-clip trim
reconstruction_to_scene: scale 1.49, position residual median 2.4 cm, rotation 0.90 deg
```
Clips look right: the carved panels, round window and table hold across views; one turn into the right-hand corridor
ends facing a dark door frame at close range (in the panorama, not a hole). The nvidia-smi figure is what the CUDA pool
has reserved, not what is live: a scheduler needs the allocated peak, so each stage now records the pool's used
high-water mark (reset at each stage start), and `--reconstruct-size` exposes WorldMirror's size. Next: a one-clip resume
of this run to read those peaks, with the reconstruction at 644.

## Run 13 - 2026-10-24 18:31

Question: the one-clip resume of Run 12 with the new per-stage allocated peaks and the reconstruction at 644; then, since
the clip loop is 92% of a run, where a real-size denoising step spends its 55 s.

Resume (8 clips from disk, 1 generated, `--reconstruct-size 644`):
```
scene            43.8 s    4314 MiB allocated at peak
trajectories     52.4 s    1262 MiB
caption          36.1 s   12113 MiB
clips           244.8 s   17639 MiB   (one clip with the model loads; the step itself sits at ~4.5 GB, so the VAE or
                                       ControlNet inputs hold a ~13 GB transient)
reconstruction   56.8 s   12092 MiB   at 644 x 378: forward 46.6 s (23.2 s at 518), camera fit 3.1 cm / 1.24 deg
                                       (2.4 cm / 0.90 deg at 518)
```
644 doubles the reconstruction for no better camera fit; 518 stays the default.

`large_batch_matmul_throughput` (inference-quant, new ignored bench), 45,056 tokens:
```
5120x5120    bf16 cuBLAS 33.7 ms  70.2 TFLOP/s   Q8_0 mmq 25.7 ms  92.0   Q4K mmq 26.8 ms  88.0
5120x13824   bf16 cuBLAS 89.6 ms  71.2           Q8_0 mmq 69.5 ms  91.7   Q4K mmq 73.6 ms  86.7
13824x5120   bf16 cuBLAS 84.9 ms  75.1           Q8_0 mmq 65.5 ms  97.4   Q4K mmq 69.2 ms  92.2
```
cuBLAS bf16 is at the 3090's fp32-accumulate peak (71); int8 mmq is only 1.3x faster here.

nsys on `full_size_streamed_step` (32,760 video + 12,480 memory tokens, 2 rounds of 55 s), per step:
```
flash_attn_ext_f16   27.75 s   (128-wide heads: joint self-attention ~0.6 s a block, ~60 TFLOP/s; 64-wide: ControlNet)
cuBLAS bf16 GEMM     14.0 s
elementwise           9.2 s    (bf16/f32 casts 3.2 s, the composed f32 LayerNorm/RMSNorm/modulate ops ~5 s, gelu, copies)
H2D block uploads     3.0 s    (33.6 GB, not overlapped with compute)
total kernel time    51.5 s    + 3.0 s copies = the measured 55 s: the GPU never idles
```
Attention is half the step. Levers, by size: an int8 QK^T / fp16-accumulate PV attention (SageAttention's scheme) for
~2x on that half; fused norm+modulate kernels in place of the composed f32 ops (~6 s); block uploads prefetched on a
second stream or kept resident (~3 s); Q8_0 linears (~3.5 s, at a quality cost). Next: the fused norms and the upload
overlap (exact), then the attention kernel.

## Run 14 - 2026-10-24 18:42

Question: what do fused kernels for the step's elementwise work buy (inference-nn `dit.cu`: LayerNorm + f32 affine +
adaLN modulation, gated residual, full-width QK RMSNorm + interleaved RoPE written head-major; bf16 GELU directly)?

`small_transformer_matches_the_reference`: 2 blocks rel rms 4.67e-3, 6 blocks 4.99e-3 (unchanged). nsys on
`full_size_streamed_step`, per step:
```
flash_attn_ext_f16   31.09 s   (was 27.75)
cuBLAS bf16 GEMM     15.78 s   (was 14.02)
elementwise           ~2.0 s   (was 9.2)
H2D uploads           3.63 s   (not overlapped)
step                 52.8 s    (was 55.0)
```
Attention and GEMM looked slower for the same work, which I first read as the power cap; it was other load on the
desktop during this run (see Run 15). fattn on Turing/Ampere accumulates KQ in f32 (half-rate MMA) and VKQ in fp16
(full rate); an int8 QK (SageAttention's per-block quantized Q and mean-smoothed K, int32 accumulate) is the remaining attention lever.

## Run 15 - 2026-10-24 18:53

Question: with the desktop's other GPU clients closed, are step timings consistent, and what did the fusion really buy?

`full_size_streamed_step`, quiet desktop, nvidia-smi sampling clocks, power and throttle reasons each second:
```
fused (HEAD)          47.0, 47.2, 47.3, 47.4 s   (two processes, two rounds each)
pre-fusion (HEAD~1)   55.2, 55.4 s               (the three Wan files checked out, rebuilt, then restored)
clocks                avg 1798 MHz, avg 320 W, 64 C; throttle reason mostly 0x4 (software power cap)
```
Run 14's 52.8 s ran beside other desktop load. The fusion saves 8.1 s a step (15%), about the 9.2 s of elementwise
work it removed; the power cap still bounds the big kernels, but it did not eat the gain. Timings repeat within 0.4 s
on a quiet desktop.

## Run 16 - 2026-10-24 19:00

Question: does uploading block i+1 on a forked stream (`CudaDevice::fork_stream`, ordered with stream joins; a block is
freed on the copy stream after the main stream has run it) hide the 3.6 s of copies?

`full_size_streamed_step`, quiet desktop: 45.7 / 45.9 s (47.2 before). nsys: the copies (2.85 s, 32 GB) run on their
own stream beside the kernels; per step flash_attn 28.1 s, GEMM 14.3 s, the rest ~1.5 s, ~1.8 s of gaps. Attention is
61% of the step.

## Run 17 - 2026-10-24 19:19

Question: does an int8 Q.K^T attention (inference-nn `int8_attention.cu`: K less its per-head token mean, Q and K int8
per token, int8 MMA for S, online softmax in f32, fp16 MMA for P.V accumulated in fp16 per 64-key tile into f32) speed
up the joint self-attention, and what does it cost in output?

Kernel against f32 attention on random inputs (std-2 Q and K, a large shared K offset): rel rms 1.8 to 2.1e-2 (bf16
flash: 1.7 to 1.8e-3). First bug: a head's K scales start at `bh * n_kv` floats, unaligned for 16-byte cp.async when
n_kv % 4 != 0 (head 0 right, later heads wrong); plain loads fixed it. `long_sequence_throughput` (45,240 tokens, 40
heads of 128):
```
bf16 flash (fattn)   665 ms   63 TFLOP/s
int8 QK              427 ms   98 TFLOP/s   (208 registers, 8 warps/SM; a warp-uniform skip of the O rescale was slower)
```
ncu: hmma active 35.6%, imma 17.8% of the time, which is each at its full rate; the other 47% is softmax and conversions
not overlapping the MMAs (2 warps per scheduler). A 3-stage pipeline computing the next tile's S during this softmax,
or fewer registers for more warps, is the next step for the kernel.

Model effect (`WORLDSTEREO_ATTENTION=int8` in the parity tests):
```
small transformer   2 blocks rel rms 5.05e-3 (bf16 4.67e-3), 6 blocks 5.37e-3 (bf16 4.99e-3)
real-size step      36.1 s (45.8 s)
whole clip          after step 0: 1.9e-3 (bf16 3.3e-3); final latents 14.4% (bf16 10.2%), frames 11.2% (bf16 7.6%);
                    denoise 111.7 s (136.9 s). The reference's own FA2 vs SDPA: 3.1% on frames
```
Side by side (frames 0, 10, 20 of the reference FA2, reference SDPA, ours bf16, ours int8): everything the render shows
matches; the differences sit in content the model invents at the right edge of frame 20 (pool in both references and
int8, road and building in bf16). int8 becomes worldgen's default (`--attention bf16` for the exact path). Later, after the kernel computed
`s * (sqa * kv)` instead of `s * sqa * kv`, the same clip ended at frames 9.2% (reproduced exactly in a second process):
one ulp in the logits moves the end of a clip by 2 points, the size of the bf16 / int8 gap.

## Run 18 - 2026-10-24 20:08

Question: the first full worldgen run with int8 attention and upload overlap died with CUDA out of memory in the first
clip (288 s; nvidia-smi 19.1 GB above idle). Where does the memory go?

Per-phase pool probes (temporary eprintln of the async pool's used/reserved high-water marks, reset each phase):
```
clip parity test   before clip 5.5 GB live; encodes peak 9.7; each step peak 12.2 (10.5 after chunking the FFN over
                   8192-token pieces: its 13824-wide hidden state is 1.25 GB twice at 45K tokens); reserved peak 12.9
worldgen           loop top / retrieval / renders 6.3 to 6.5 GB live; encodes peak 11.0; steps 11.8 to 12.2;
                   decode peak 13.1; but the pool held 20.4 to 21.9 GB reserved by each clip's end (trimmed back to
                   ~7 GB by the per-clip release); a second probed run survived 5 clips before my timeout
```
The live peak is ~13 GB; the rest is pool slack from phases of very different allocation shapes (the transformer's few
large tensors, then the VAE decoder's many full-resolution ones) caching beside each other. The earlier bf16 runs had
the same slack (nvidia-smi 21.7 to 23.5 GB with 17.6 GB live) and survived narrowly. Fix: the pipeline syncs and trims
the pool after the encodes, after denoising and after decoding.

## Run 19 - 2026-10-24 20:34

Question: end to end on case001 (same prompt as Run 12) with the fused kernels, overlapped uploads, int8 attention,
the chunked FFN, the pool trims and the trajectory BVH (mesh distance through a median-split BVH, exact: the scene test's
9 trajectories match the reference as before, and a unit test pins it to brute force).
```
scene            43.3 s   (43.5)
trajectories      8.1 s   (51.9)
caption          34.6 s   (38.4)
clips          1289.0 s   (1861.6; 143 s a clip, was 207)   allocated peak 13.2 GB (17.6)
reconstruction   32.3 s   (30.5)
wall           1408 s     (2027); nvidia-smi peak 16.9 GB above idle (22.0)
camera fit       2.4 cm / 0.85 deg (2.4 cm / 0.90)
```
Frames side by side with Run 12's: the same scenes; differences only in invented detail (a lamp, a side cabinet).
Left: the clip loop is still 92% of the run; per step attention ~16 s, GEMM ~14 s.

## Run 20 - 2026-10-24 20:46

Question: can the int8 attention overlap its softmax with its MMAs (Run 17: tensor pipes busy 53% of the time, the
8 warps in lockstep through one barrier a tile, so both warps of a scheduler sat in softmax at once)?

`long_sequence_throughput` (45,240 tokens, 40 heads of 128; bf16 flash 664 ms throughout):
```
v1                                      427 ms  98 TFLOP/s   208 regs
S one tile ahead, 3 stages (v2)         398 ms  105          246 regs; ncu: tensor 59% (53), barrier stall 0.48 (1.17)
v2 + ping-pong scores, fused rescale    399 ms               255 regs, 136 B spilled
  + Q fragments from shared memory      419 ms               255 regs (the compiler spent the freed registers)
v2 + O rescale fused into the add as one FMA + no bounds check on full tiles (v4)
                                        386 ms  108.5        240 regs
v4 + int-to-float by the magic-number add (off the XU pipe)
                                        391 ms               (no gain; reverted)
```
Kept v4: 1.72x the bf16 flash attention, 57% of the int8/fp16 MMA peak. More needs a redesign around the 2 warps per
scheduler that 240 registers allow. Model: small transformer 5.02e-3 / 5.41e-3; the real-size step 34.3 s (36.1); the
whole-clip parity reproduces Run 18's numbers exactly (frames 9.162e-2): the FMA's single rounding vanishes at bf16
output. Denoise of the reference clip 107.8 s.

## Run 21 - 2026-10-24 20:55

Question: does a ping-pong schedule (FlashAttention-3's, on Ampere) lift the int8 attention: two groups of four warps
alternating turns on the tensor cores through named barriers (bar.sync / bar.arrive), so one group's softmax runs
during the other's MMAs, with the stages tracked by mbarriers (full: cp.async.mbarrier.arrive.noinc from every thread;
empty: every thread done) instead of the per-tile block barrier that kept the warps in step?

`long_sequence_throughput`, same 45,240-token, 40-head case:
```
v4 (committed)                                   386 ms
ping-pong, load of t + 2 after the softmax       391 ms   errors identical; 246 regs. ncu: tensor 60.6% (59),
                                                          pipe throttle 0.94 (2.09), but "wait" 1.76, barrier 0.77,
                                                          branch (mbarrier spin) 0.57
  + K steps across all 8 key groups before the next (MMAs on one accumulator 8 apart), 4 partial maxima, 2 sums
                                                 389 ms
probe: ping-pong with no softmax math (P = scaled scores, no max, no exp)
                                                 379 ms
probe: no softmax math and no ldmatrix (register constants as fragments; output garbage)
                                                 327 ms   (ideal tensor time at the int8 / fp16-accumulate peaks: ~234)
```
The softmax costs ~10 ms of the 386, so overlapping it can't pay; Run 17's reading (the non-MMA 47% is softmax) was
wrong. The fragment loads cost ~52 ms (each 16-row warp loads the whole K and V tile, 8x over per block), and ~93 ms is
MMA issue with two warps per scheduler. Both shrink only with more rows per warp (each fragment feeding more MMAs, more
independent MMA chains), which needs the f32 output accumulator (64 registers at 16 rows) to fit at 32 rows. Reverted
to v4; the ping-pong source is kept in the session scratch only.

## Run 22 - 2026-10-24 21:30

Question: Q8_0 linears in WorldStereo's 40 main blocks (quantized once at load on the host, streamed at half the bytes,
run through the GGUF MMQ kernels): speed and quality?

The Wan `Linear` gained a quantized variant (upload through `QStorage::from_data` on the copy stream's device handle).
First streamed run: NaN. A unit test of one host-quantized, uploaded linear passed, so the layer was right; the cause
was `fast_mmq::shared_lhs` launching on the weight's device handle, i.e. the copy stream, unordered against the
activations. It now runs on the input's stream (normal GGUF inference has one stream for both, unchanged).
```
small transformer   2 blocks rel rms 6.30e-3, 6 blocks 6.39e-3 (int8 attention alone 5.02e-3 / 5.41e-3)
quantize at load    9 s for the 40 host blocks (rayon across blocks)
real-size step      30.4 s (34.3)
whole clip          after step 0: 1.6e-3 (int8 alone 1.9e-3, bf16 3.3e-3); final latents 15.1%, frames 11.4%
                    (int8 alone 9.2%, bf16 7.6%); denoise 94.6 s (107.8)
case001 end to end  scene 42.6 s, trajectories 8.1, caption 39.4, clips 1110.4 (1289.0; ~122 s a clip), reconstruction
                    31.3; wall 1233 s (1408); nvidia-smi peak 17.7 GB above idle; camera fit 2.3 cm / 0.85 deg
```
Side by side with Run 19's frames: the same rooms and detail. Q8_0 becomes worldgen's default (`--weights bf16` keeps
the blocks dense).

## Run 23 - 2026-10-24 21:57

Question: does worldgen hold up on an outdoor panorama? A CC0 Poly Haven park panorama (an autumn garden terrace:
paved plaza, stone steps, flower beds, trees, overcast sky), 1920 x 960, no `--scene` (decided by sky share), hand
prompt, Q8_0 blocks and int8 attention.
```
scene 47.0 s, trajectories 1.2, caption 41.6, clips 1044.8 (~116 s a clip), reconstruction 28.7; wall 1164 s;
nvidia-smi peak 17.7 GB above idle; decided outdoor (sky share); camera fit 1.8 cm / 1.05 deg; 2.0M points, 4.9M splats
```
Clips: the terraces, benches, beds and trees stay consistent across the 3 start views; the aerial move invents lawns and
paths beyond the wall plausibly. The light drifts sunnier and more saturated as the aerial move climbs, despite "soft
overcast" in the prompt; the fastest moves blur the nearest foliage. A top-down render of points.ply in the scene frame:
the plaza, round bed, steps and the red shrub in place, far trees squeezed into a ring by the outdoor contraction.
Scale: the ground sits 0.43 to 0.48 m below the camera (5th percentile of z in rings 0.3 to 3 m), so MoGe-2's metric
estimate is ~3.5x small for a standing-height panorama; the indoor case001 gives floor -0.47 m and ceiling +1.24 m (a
1.7 m room). Scene construction does not rescale (only clips at the 0.98 quantile and contracts far distances
outdoors), and the panorama depth matches the reference's, so the estimate is the model's. meta.json and the guide now
call the scale approximate and say to fit it to known geometry.

## Run 24 - 2026-10-24 22:20

Question: can the park run be walked in Godot (4.7.2), and is a mesh fused from the reconstruction better than its
point cloud for that?

A standalone Godot project beside the run (not in any repo) builds the scene at load from exported binaries: the
2M-point cloud (one draw call, a distance-scaled point-size shader), a ground heightfield (5th percentile of point
heights per 25 cm cell, holes from the nearest seen cell, 5x5 median; HeightMapShape3D scaled uniformly by the cell),
the source panorama as the sky, and a first-person walker kept inside the reconstruction's radius (17 m). The scene is
rescaled x3.35 so the eye sits 1.6 m above the estimated ground (Run 23's small scale). It runs; the walker stands on
the floor at every probe.

Fused mesh: each depth map masked to the pixels its frame's points confirm (the point cloud already drops sky and
edges; 83.4% of pixels kept), integrated into an open3d TSDF (4 cm voxels, 16 cm truncation) in the metric scene frame,
meshed (8.1M triangles), components under 2000 triangles dropped, decimated to 400K, coloured from the 4 nearest points;
in Godot it shows and collides as itself (M switches). open3d 0.18 segfaulted in integrate() under numpy 2.2; 0.19
fixed it. 1m40s.
```
mesh vertex to nearest point: median 8.5 cm, 90th 18.5 cm (aligned: no transform error)
point to nearest mesh surface: median 12.7 cm, 90th 42 cm, 99th 1.08 m
```
Side by side at the same 5 probes the mesh is blobby and wavy, its colours blurred, and the steps melt into a ramp,
while the point cloud reads as the place. WorldMirror's 99 depth maps are predicted per view and disagree by around a
decimetre; the TSDF averages the disagreement into blobs. The point cloud only looks crisper because the eye picks the
dominant layer. Better geometry needs views made consistent (optimized splats, tier 3) or the scene split by kind: a
terrain surface fitted to the ground, structures as planes, vegetation by an engine foliage system.

## Run 25 - 2026-10-25 00:39

Question: does the native port of the reference's depth alignment (retrieval_wm.py alignment(): WorldMirror depth at
832 with the planned cameras as priors, fitted per frame in inverse depth by a constrained RANSAC line to the panorama
cloud rendered into that view, anomalous fits borrowed from the clip's nearest inlier, per-clip statistical outlier
removal, voxel-first downsample to 2M points) make the per-view depths agree, and what does it cost?

Park run (clips reused), `inference worldgen ... --prompt <park caption>` into park_aligned:
```
alignment depth (WorldMirror 2.0)                  88.2 s   17598 MiB allocated at peak
alignment (MoGe-2 masks, fit to the panorama)    6112.8 s    7329 MiB allocated at peak
alignment: frames 99, fitted 89, borrowed 10, abandoned 0, points 2000000
```
Frame-pair reprojection agreement (scratchpad depth_agreement.py; per overlapping pair the median relative depth
difference, then quantiles over pairs):
```
raw WorldMirror (518): 7127 pairs, p25 0.0234 p50 0.0366 p75 0.0551 p90 0.0872, valid 100%
aligned (832):         7342 pairs, p25 0.0182 p50 0.0292 p75 0.0486 p90 0.0806, valid 77.6%
```
Alignment helps, but modestly (median 3.7% -> 2.9%); the views still disagree by a few percent of depth, about what
the reference's own 3DGS stage exists to fix. The 22.4% invalid pixels are the masked edges, sky and depths past
1.1x the panorama's maximum.

The 6113 s was CPU-only with the GPU idle. Timing outlier_mask alone on the exported 2M points: 408.8 s. PointGrid
sized its cells for points spread through the bounding volume, but these lie on surfaces, so cells held thousands of
points while isolated far points walked up to 1024 empty rings. Replaced by a balanced kd-tree (median splits on the
widest axis, 16-point leaves) behind the same API: 0.6 s, the same 1,898,409 points kept (the k nearest are exact
either way); brute-force test added. The trajectory obstacle checks use it too.

Rerun with the kd-tree (park_aligned2, then park_aligned with the scene dump below):
```
alignment depth (WorldMirror 2.0)                  91.1 s
alignment (MoGe-2 masks, fit to the panorama)      81.8 s   (was 6112.8 s)
alignment: frames 99, fitted 89, borrowed 10, abandoned 0, points 2000000
exit 0, wall 331 s
```
Aligned depths between the two runs: most frames bit-identical, the worst frame's median relative difference 1.1e-3
(GPU run-to-run noise in the depth model), no mask flips. worldgen now also writes `scene/` (the panorama's cleaned
distance, the sky mask, the scene cloud and the start views), what the Gaussian training data is cut from.

## Run 26 - 2026-10-25 01:16

Question: trained on our data, does the reference's 3DGS stage (gen_gs_data.py, then world_gs_trainer.py with its
gsplat fork) give geometry and novel views good enough to justify porting it, before writing any of it natively?

Setup, all outside the repo:
- gsplat_maskgaussian (the reference's fork of gsplat 1.5.3: MaskGaussian in the rasterizer, projection and
  wrapper) built in place in /mnt/data/build/gsplat_mg (glm headers from the stock package, CUDA 12.8, sm_86,
  2m46s) and put first on PYTHONPATH for the trainer only; the venv's stock gsplat is untouched. Installed into the
  venv: viser, nerfview, splines, tyro, torchmetrics, tensorboard, fused-ssim (git build), scikit-image, tensorly,
  pymeshlab.
- scratchpad gs_tools/make_gs_data.py: gen_gs_data.py's --save_normal --split_sky path over a native run, calling the
  reference's own helpers (split_panorama_image/depth, smooth_sky_depth_boundary, convert_rgbd2pcd_panorama,
  adaptive_voxel_downsample, MoGe-2 normals). 189 clip frames (frame 0 the start view; aligned depth on the 81 aligned
  frames), 54 panorama + 40 polar crops with the panorama's depth, 283 cameras; points.ply 3.11M (2M aligned + 1.11M
  panorama at voxel 0.003), sky_points.ply 300K. 1m43s.
- Training with the README's flags at the single-GPU length (--steps_scaler 5.3334: 8000 steps, refine 800 to 4000
  every 533; scale2d stop set to 4000 by hand; spz and the viewer off).

```
Step 3731: 2571655 GSs after the last refine; Step 7999 mem 2.67 GB, 928 s of steps, wall 16m28s
held-out views (every 32nd, 6): PSNR 24.191, SSIM 0.7549, LPIPS 0.203
MaskGaussian export: kept 2242322/2571655 by stochastic sampling; point_cloud_7999.ply 126 MB
mesh (the trainer's TSDF over its own rendered RGB+depth from all 283 cameras): 552742 triangles, simplified 10%
```
The trainer's mesh export failed in-run on the missing skimage import (caught, training finished); rerun from the
checkpoint with scratchpad gs_tools/ref_eval_mesh.py (the trainer's eval plus its mesh block verbatim), 27 s.

Novel views (gs_tools/novel_views.py: training cameras moved sideways 0.1 to 0.3 of the camera spread, about 0.5 to
1.3 m, some turned 15 to 20 degrees; splats left, the fused mesh right; gs_novel.jpg beside the run): the splats stay
sharp and coherent off the trajectories (steps, benches, flower beds, trees all hold). The splat-fused mesh is the
first usable geometry: clean steps, benches, a flat plaza, walls; shrubs and beds become smooth blobs and colours
soften (5 cm voxels in normalized units). Far larger moves (1x the spread) put the camera inside foliage and show
nothing useful; the reconstruction is a walkable neighbourhood of the trajectories, not the whole park.

Implication: the 3DGS stage is what turns WorldMirror's disagreeing per-view depths into one consistent scene (Run 24's
TSDF over raw depths was blobby at a 12.7 cm median error). Porting the trainer is worth it. Its pieces: the gsplat
forward/backward (projection, tile binning, rasterization with antialiasing, depth and normal outputs), the
MaskGaussian mask, SSIM/LPIPS/depth/normal losses, Adam with per-group rates, the default strategy's
densify/prune, anchor protection, sky depth from the sky cloud, then the TSDF mesh export.

## Run 27 - 2026-10-25 01:38

Question: do native Gaussian splatting kernels (inference-nn `kernels/cuda/gsplat.cu`: EWA projection with the
antialiasing compensation, tile intersection with a cub radix sort, RGB plus z rasterization with the MaskGaussian
gate, and the backward passes, one camera, raw parameters in and gradients back to them) reproduce the reference fork
on the trained park checkpoint?

The fork differs from stock gsplat 1.5.3 only by the gate: the rasterizer composites with gate * alpha (the threshold
test uses alpha alone), and the backward splits d/d(gate * alpha) into the opacity's and the gate's gradients. The
port is from stock gsplat with that term; glm is replaced by a 100-line column-major matrix type.

`scripts/gsplat_parity.sh <python> /mnt/data/build/gsplat_mg <ckpt_7999_rank0.pt>` (fixture by
scripts/gsplat_parity.py: a camera at the median centre looking along the widest spread, the forward outputs and the
gradients of a seeded random weighting of RGB+ED and alpha):
```
2571655 gaussians, 457926 visible
radii equal for 1.000000 of gaussians
rgb            max rel 4.73e-3  rms rel 3.00e-5
depth          max rel 1.46e-4  rms rel 7.55e-7
alpha          max rel 4.52e-4  rms rel 1.39e-6
v_means2d      max rel 2.86e-4  rms rel 2.13e-4
grad means     max rel 6.04e-4  rms rel 3.00e-4
grad quats     max rel 6.21e-5  rms rel 4.56e-5
grad scales    max rel 8.13e-6  rms rel 9.59e-6
grad opacity   max rel 1.45e-4  rms rel 5.79e-5
grad sh0       max rel 5.66e-5  rms rel 3.98e-5
grad gate      max rel 9.85e-6  rms rel 1.40e-5
```
The residue is float atomics' summation order. Dead ends on the way: the first fixture's view matrix came from
torch.linalg.inv, which numpy saved in Fortran order, and the test's reader took it as C order (every visible Gaussian
culled; the reader now refuses Fortran order and the dump writes contiguous arrays); gsplat's rasterization() defaults
to packed mode, while the trainer passes packed=False.

Self-contained check (`gradients_match_finite_differences`, 24 random Gaussians, 64x48): analytic against central
differences, cosine 0.998 (means) to 1.0 (sh0, gate); the geometry gradients' few percent come from what gsplat
leaves out by design (radii and the tile set are not differentiated).

Next: the trainer (losses through autograd with the render as the leaf, Adam, densification, MaskGaussian, the
anchor protection, sky depth), against the reference's PSNR on the same data.
