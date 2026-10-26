# World-generation checkpoints as a package: GGUFs and a Hugging Face repo

A world job reads about 64 GB from five repositories: WorldStereo-2 (33 GB bf16, quantized to Q8_0 at every load), the
umT5-XXL encoder of Wan2.1-I2V (22 GB fp32, run once per job in bf16), the rest of Wan (VAE, CLIP image encoder,
tokenizer), MoGe-2, LPIPS, and WorldMirror 2.0. The goal: one repo laid out as a `--world-models` root, with WorldStereo
and umT5 as Q8_0 GGUFs, and WorldMirror fetched from Tencent's own repository (its community license carries territory
and notice conditions we would otherwise take on by re-hosting).

## Run 1 - 2026-10-25 21:52

Question: does the writer produce the layout, and how big is it?

`inference quantize worldgen -m /mnt/data/models -o /mnt/data/build/worldgen-package/repo` (unchanged files hard-linked):

- `worldstereo-memory-dmd-Q8_0.gguf` 18,811 MiB: 480 Q8_0 (the 40 blocks' 12 linears), 1,319 bf16
- `umt5-xxl-encoder-Q8_0.gguf` 5,757 MiB: 193 Q8_0 (matrices and token embedding), 49 F32 (norms, relative biases)
- 20 files, 26.9 GiB in all (WorldMirror left out); 33 s

The WorldStereo GGUF quantizes the same bf16 values the loader quantized at load, so its blocks are the same bytes:
`a_worldstereo_gguf_block_linear_matches_quantizing_at_load` checks a block linear read from the GGUF gives the same
output, bit for bit, as one quantized at load. Only umT5 changes numerically.

## Run 2 - 2026-10-25 22:05

Question: is a Q8_0 umT5 close enough? Measure per-token cosine and relative RMS of the encoder states on four prompts.

`WAN_TEXT_ENCODER_DIR=... cargo nextest run -p inference-models-diffusion --features cuda --run-ignored only a_q8_0_umt5`

Against the bf16 encoder (today's runtime), Q8_0 through the GGUF linears (MMQ): first prompt cosine min 0.98364, mean
0.99606, relative RMS 0.0875. Worse than expected, so add an fp32 CPU reference and split the error into weights and
activations (Q8_0 weights dequantized to a bf16 safetensors and run dense):

| vs f32 | room | courtyard | alley | hall |
|---|---|---|---|---|
| bf16, cosine min / RMS | 0.99768 / 0.0383 | 0.99146 / 0.0523 | 0.95762 / 0.1003 | 0.99656 / 0.0275 |
| Q8_0 MMQ (CUDA) | 0.99034 / 0.0681 | 0.98409 / 0.0681 | **0.26229** / 0.1476 | 0.99433 / 0.0491 |
| Q8_0 weights, bf16 matmul | 0.99233 / 0.0575 | 0.99533 / 0.0438 | 0.98309 / 0.0572 | 0.99527 / 0.0448 |
| Q8_0 on CPU (q8 activations) | 0.99167 / 0.0763 | 0.97017 / 0.0939 | **0.20671** / 0.1636 | 0.99154 / 0.0574 |

Finding: the weights are fine; quantizing the activations to 8-bit blocks (MMQ's Q8_1, the CPU's Q8_0) loses a whole
token on the alley prompt. T5's activation outliers (the reason it overflows fp16) do not survive 8-bit blocks. bf16
itself is no closer to f32 than Q8_0 weights are. ComfyUI-GGUF, whose Q8_0 T5 encoders are widely used, dequantizes
weights per matmul and never quantizes activations, which is why it holds up there.

Implication: T5 linears from a GGUF dequantize their weight at each use (`MaybeQuantLinear::new_dequantizing`) instead of
going through the quantized matmul. T5 runs once per request, so the cost does not matter. This also changes the FLUX
T5 GGUF path, which had the same exposure.

## Run 3 - 2026-10-25 22:15

Same test with T5 dequantizing per matmul:

| vs f32 | room | courtyard | alley | hall |
|---|---|---|---|---|
| bf16 | 0.99768 / 0.0383 | 0.99146 / 0.0523 | 0.95762 / 0.1003 | 0.99656 / 0.0275 |
| Q8_0 (GGUF, dequantized per use) | 0.99233 / 0.0575 | 0.99533 / 0.0438 | 0.98309 / 0.0572 | 0.99527 / 0.0448 |
| Q8_0 on CPU in f32 | 0.99044 / 0.0590 | 0.99155 / 0.0537 | 0.98490 / 0.0537 | 0.99327 / 0.0384 |

Q8_0 now equals the dequantized-weights run exactly. Worst token: bf16 0.958, Q8_0 0.983; mean relative RMS bf16 0.0546,
Q8_0 0.0508. The committed test asserts Q8_0 lands no further from f32 than bf16 does (10% slack on both measures).

## Run 4 - 2026-10-25 22:10

Question: does a world run from the package match one from the original checkpoints? Same panorama (an indoor
reference panorama, a furnished room), same prompt, seed 1024; both output folders seeded with the same 8 clips from an
earlier run, so each generates only the first clip (`view0_traj2`, no memory frames) and the clip differs only by the
checkpoints.

`inference worldgen --panorama ... --out a_bf16_umt5 --worldstereo /mnt/data/models/... --wan ... --moge ...` against
the same with the package's folders.

| | original | package |
|---|---|---|
| caption (umT5) | 34.8 s, 12,113 MiB peak | 4.2 s, 8,202 MiB peak |
| clip | 138.1 s | 124.6 s (no quantize at load) |
| whole run | 3 m 30 s | 2 m 45 s |

Frames of the generated clip, package against original: PSNR per frame 49.6 (the start frame) falling to 31.4 at frame
9 and recovering to 33-38; mean 36.6 dB. Side by side, the worst frame shows no visible difference. The WorldStereo
blocks being identical, all of it comes from the Q8_0 text conditioning.

## Run 5 - 2026-10-25 22:19

Question: does a whole world job run from the package, fetching WorldMirror from `tencent/HY-World-2.0`?

`INFERENCE_TEST_WORLDGEN_MODELS=/mnt/data/build/worldgen-package/repo scripts/deep_checks.sh -E 'package(inference) &
test(/^world_job::/)'`, the hub cache holding no HY-World snapshot.

PASS in 1,922 s (the same test on the original checkpoints: 1,708 s), the difference mostly the 4.8 GB WorldMirror
download in the job's prepare step, which runs before the job takes the GPU. Every stage ran, splats included.
