# HiDream-O1-Image

Issue #393: HiDream-O1-Image text-to-image (and later editing and subject references) on our Qwen3-VL stack.

## Run 1 - 2026-10-10

Question: what is the checkpoint, and how much of it is our Qwen3-VL?

Commands:
- the HF tree API on `HiDream-ai/HiDream-O1-Image` and `-Dev`;
- `config.json` and `model.safetensors.index.json`;
- a clone of `github.com/HiDream-ai/HiDream-O1-Image` (main `2c2d29f`, dev `3237a63`) in `/mnt/data/Programming`.

Raw:
- **Checkpoint:** both repos hold 8 shards, 35.2 GB in F32 (8.80 B parameters). `config.json` is a stock `Qwen3VLForConditionalGeneration`, the Qwen3-VL-8B shape:
  - text: 36 layers, hidden 4096, 32 heads over 8 KV heads, head dim 128, interleaved M-RoPE `[24, 20, 20]`, theta 5e6;
  - vision: 27 layers with deepstack at 8/16/24.
- **Configs:** Full, Dev and Dev-2604 have byte-identical configs. Dev-2604 is a later Dev that the model card tailors to text-to-image.
- **Tensors beyond Qwen3-VL**, all under `model.`:
  - `x_embedder.proj1` (3072 -> 1024, no bias) and `proj2` (1024 -> 4096): a bottleneck patch embedding of 32x32x3 pixel patches;
  - `t_embedder1.mlp.{0,2}`: a 256-frequency sinusoidal timestep embedding, through Linear, SiLU, Linear;
  - `final_layer2.linear` (4096 -> 3072): back to pixel patches.
- **No VAE:** the model denoises 32 px RGB patches in [-1, 1] directly.
- **Generation forward** (`models/qwen3_vl_transformers.py::_forward_generation`):
  - The sequence is the chat-templated prompt, then `<|boi_token|>`, then one `<|tms_token|>` (id 151673), whose embedding is replaced by the timestep embedding. Then come the noisy target patches through `x_embedder`, then any reference-image patches.
  - Attention: the text tokens (type 0) are causal among themselves. The timestep token and every patch (type > 0) attend to the whole sequence bidirectionally. The reference runs this as two flash passes and keeps each row from the right one; equivalently, the gen rows of a causal mask are opened fully.
  - Gen tokens all follow the text, so the text's K/V depend on neither the step nor the noise. They are a prefix cache across all steps (and are shared by the CFG pair only if the prompts match, which they don't).
  - `final_layer2` runs on every row; the pipeline keeps the target rows as `x_pred`, a clean-image prediction.
- **Positions** (`get_rope_index_fix_point`):
  - text is 0..L-1 on all three axes;
  - target patches are `(4096, 4096 + row, 4096 + col)`, the "fix point";
  - reference patches continue from the previous maximum + 1;
  - conditioning images seen by the ViT use the usual Qwen3-VL positions.
- **Sampling** (`models/pipeline.py`):
  - initial `z = noise_scale * randn` (torch CPU generator, seed + 1), in patch layout;
  - per step, `t = 1 - timestep/1000` and `sigma = max(timestep/1000, 1e-3)`, `v = (x_pred - z) / sigma`, CFG on `v`, and the scheduler is fed `-v`.
- **Full model:** 50 steps, CFG 5.0 with the unconditional prompt `" "`, shift 3.0, `FlowUniPCMultistepScheduler` (Wan's flow UniPC, order 2, bh2, predict x0). `noise_scale` is 8.0 (pipeline default).
- **Dev model:** 28 fixed timesteps (`DEFAULT_TIMESTEPS`, 999..8), no CFG, `FlashFlowMatchEulerDiscreteScheduler`, `noise_scale` 7.5.
  - The step is `z' = sigma_next * s_noise * n + (1 - sigma_next) * x_pred`, with `s_noise` 7.5 and `n` a fresh randn clipped to 2.5 of its own std.
  - It re-noises the prediction at every step. The last `sigma_next` is 0, so the output is the last `x_pred`.
- **Resolutions:** the pipeline snaps any size to the nearest aspect among 11 presets of about 4.2 MP (2048x2048, 2560x1440, 3104x1312, ...). That is 4096-3977 patch tokens per forward, plus about 100-300 text tokens.
- **Precision:** reference inference runs BF16 under autocast. The README notes F32 gives "more detailed images" at more memory.

Implications:
- **Reuse:** the text stack is our `DecoderStack` (Qwen3-VL's `decoder_spec`) run without an lm_head, with an `AttentionMask::Custom` mask. On CUDA that takes fattn, per `custom_mask_flash_investigation.md`.
- **New code:** the three heads, the position rule, the two samplers, and the prompt and sequence assembly.
- **Memory:** the RTX 3090 has 24 GB. BF16 weights are 17.6 GB, so the dense path fits; 8-bit ISQ is about 9 GB.
- **Parity:** the RNGs differ (torch CPU MT19937 for the initial noise, CUDA Philox for the Dev re-noise). The reference script dumps each step's input `z`, the per-step noise and `x_pred`. Our side replays its `z` and noise, then compares `x_pred` by cosine and the final image.
- **Cost:** a 2048x2048 forward is about 2 * 8.8e9 * 4300 = 76 TFLOP. That is roughly 1.5-2 s on a 3090 at good utilisation, so Dev (28 forwards) takes about a minute and Full (100 forwards) about 3 min.
- **Plan:** text-to-image first, for both variants, in one PR. Reference images (editing, subject references, layout) and quantized loading follow.

## Run 2 - 2026-10-10

Question: does the port's sampler schedule match the reference exactly, and does the engine path work end to end on a tiny checkpoint?

Commands:
- `FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=3.0).set_timesteps(50)`, in a venv on `/mnt/data/venvs/hidream`. It sits over the user-site torch 2.11+cu130 and adds transformers 4.57.1 and diffusers 0.39, as the reference pins.
- `cargo nextest run -p inference-models-diffusion hidream`
- `cargo nextest run -p inference hidream_o1` (CPU, F32, tiny checkpoint: 2 layers, hidden 128)

Raw:
- **UniPC schedule:** the first port missed sigma 48 by one f32 ulp (`0x3de37c3b` vs `0x3de37c3a`). Two causes:
  - the scheduler's constructor shifts its sigmas in f32, not f64, and `sigma_max` comes from that f32 value;
  - numpy's linspace is `i * step + start`, not `start - start * i / n`.
  With both fixed, all 50 timesteps and the pinned sigma bits match.
- **Reference quirk:** `set_timesteps` shifts a range its constructor already shifted (shift 3 applied twice). Ported as-is.
- **Engine, full variant:** a 640x360 request snaps to 2560x1440. That is 3600 patches plus about 25 text tokens on the tiny model. Two steps with CFG take 8 s on CPU. The same seed repeats the image byte for byte; seed + 1 differs.
- **Engine, Dev variant:** the 28-step fixed schedule, twice, takes 31 s on CPU.
- **Dev refusals:** a `steps` other than 28, or any `guidance_scale`, is refused before queueing. A model error inside `forward` reaches the client only as the generic "The model failed to process the request."; `DiffusionModel::validate` (via `Pipeline::validate_image_params`, beside the speech hook) returns the real message as a validation error.
- **Detection:** the stock `Qwen3VLForConditionalGeneration` config and file names match a Qwen3-VL chat model, so auto-detection reads `model.safetensors.index.json` for that architecture only and looks for `model.x_embedder.proj1.weight`.
- **Variant choice:** Full and Dev are byte-identical in config and file layout, so Dev is chosen by a `-dev` in the model id's last component, or explicitly with `hidream-o1-dev`.

Next: the real Dev checkpoint (downloading, about 4-5 MiB/s; the Xet backend stalled after 1.9 GB and plain HTTP took over) with `scripts/hidream_o1_parity.sh --dev`.

## Run 3 - 2026-10-10

Question: before the real weights arrive, does the parity tooling run end to end, and does the port agree with the reference's own forward and samplers on a random-weight model?

Setup:
- A tiny reference checkpoint in `/mnt/data/_tmp/hidream_tiny`: `Qwen3VLForConditionalGeneration` from the reference's own module, with the tiny fixture's text config and the Qwen3-VL test vision config.
- All parameters are random: 0.05 * randn, and norms 1 + 0.1 * randn. The reference initialises `final_layer2` to zeros, which would make every prediction 0.
- Saved with `save_pretrained`. The script now sets `tms_token_id` from the tokenizer, since the reference hardcodes 151673.

Commands:
- `HIDREAM_REPO=... HIDREAM_PYTHON=/mnt/data/venvs/hidream/bin/python scripts/hidream_o1_parity.sh /mnt/data/_tmp/hidream_tiny --dev --size 256x256`
- the same with `--size 256x384 --steps 12`, the full variant.

Both sides run on CUDA in BF16, on the reference's 4D-mask path (flash_attn is not installed).

Raw:

| run | forwards | teacher-forced x_pred cosine (min / mean) | sampled image vs reference |
|---|---|---|---|
| Dev, 256x256 | 28 | 0.999975 / 0.999988 | mean abs diff 0.70, max 6, PSNR 48.0 dB |
| Full, 256x384, 12 UniPC steps, CFG 5 | 24 | 0.999985 / 0.999988 | mean abs diff 1.73, max 13, PSNR 40.8 dB |

- **Free running:** every step's prediction keeps cosine 1.0000 (0.9999 at the last step). The non-square size would have caught a row/column swap in the positions or the patch layout.
- **The full run's lower PSNR** fits guidance 5 amplifying the BF16 differences between the cond and uncond forwards.

Implication: the mask, positions, timestep token, heads, CFG and both samplers match the reference. Next: the same on the real Dev checkpoint at 1024x1024 and at a preset size, plus timing.

## Run 4 - 2026-10-10

Question: what did the review find, and do the fixes hold on the tiny checkpoint?

Findings and fixes:
- **Batching:** every image request in an engine step took the first request's size, steps and seed (`diffusion_processor.rs` read `input_seqs[0]`).
  - Parameters are now per sequence. Choice `i` of a request takes `seed + i`.
  - The pipeline batches only unseeded prompts with equal parameters; a seeded prompt runs alone, so batching cannot change its image.
- **`n` was ignored for images** (pre-existing): `generation_request` always sent `n_choices` 1, so `n = 2` returned one image. It is now passed through, limited to OpenAI's 1 to 10.
  - Test: `n = 2` with seed 7 returns exactly the seed-7 and seed-8 single-image results.
- **CLI:** `inference serve diffusion` hardcoded FLUX.
  - It now takes `--diffusion-arch` (`flux` default, `flux-offloaded`, `hidream-o1`, `hidream-o1-dev`), and the TOML config takes `diffusion_arch`.
  - `flux-offloaded` was SDK-only before.
- **Pixels:** the reference adds 1 to the BF16 latent in BF16 before converting to F32. Doing the same moved the tiny Dev replay from 48.01 to 48.31 dB PSNR, and the full replay from 40.79 to 40.84 dB.
- **Dev detection** now also accepts a hub cache snapshot path (`models--<org>--<name>-Dev/snapshots/<sha>`) and logs the variant it picked.
- **Remote weight-index fetch** now happens only when the repo listing has an index; a failed read falls through instead of aborting detection.
- **Dev guidance:** Dev accepts a `guidance_scale` of 1 or less, as the reference runs Dev at 0.
- **Docs:**
  - FLUX.1-dev's `guidance_scale` is distilled guidance, not CFG.
  - The seed guarantee holds only on the same device and dtype.
  - Dev's `steps` wording.
  - Two stale Rust SDK passages (an old `generate_image` signature; a claim that the SDK writes URL images to disk). The SDK example now saves the PNG from the file store.
- **Left for the quantization PR:** `tune` / `--quant auto` infer `TuneKind::Multimodal` from HiDream's Qwen3-VL config.

Next: the real Dev checkpoint.

## Run 5 - 2026-10-10

Question: does the port hold on the real Dev checkpoint, and what precision bar should parity use?

Command: `scripts/hidream_o1_parity.sh /mnt/data/models/HiDream-O1-Image-Dev --dev --size 1024x1024` (CUDA, BF16 both sides; the default fox prompt, seed 7).

Raw, first pass:
- **Teacher-forced x_pred cosine:** min 0.979646 (forward 0, the pure-noise step), mean 0.997339. That is below the 0.999 gate.
- **Free-running image:** mean abs diff 9.3, PSNR 23.0 dB. By eye, the reference and ours show the same subject, pose, composition and light; only fine fur and twig texture differ.
- **Speed:** 28 steps at 1024x1024 in 8.5 s on the RTX 3090, after a 26 s load.

Floor measurements (`hidream_floor.py`, scratchpad), on the reference's own inputs:

| forward | reference BF16, SDPA math vs efficient | reference BF16 vs F32 (CPU) | ours BF16 vs F32 |
|---|---|---|---|
| 0 | 0.991690 | 0.975157 | 0.993795 |
| 7 | 0.995988 | | |
| 14 | 0.997967 | 0.997890 | 0.999072 |
| 27 | 0.999986 | | |

- **Backend sensitivity:** the dumped reference is the efficient SDPA kernel (cosine 1.0 to it). Swapping only that one kernel moves its prediction by more than 0.008 at the noisy steps.
- **Accuracy:** our BF16 is closer to the F32 truth than the reference's BF16 at both forwards. The early high-noise steps are simply this sensitive in BF16.

Implication:
- Parity against a BF16 reference cannot demand 0.999 at the noisy steps. The example now takes `--against <f32 dumps>` and requires each covered forward to be at least as close to F32 as the reference (1e-3 slack); `--min-cosine` is then not required of the BF16 reference.
- On this run, forwards 0 and 14 pass.

End to end through the server:
- `inference serve -m /mnt/data/models/HiDream-O1-Image-Dev` auto-detects `hidream-o1-dev` from the weight index.
- Four 2048x2048 logo prompts (seeds 1-4) took 34-35 s each: 28 forwards over 4096 patches plus about 100 text tokens, about 1.2 s per forward.
- The wordmark text rendered correctly in all four. The prompt asked for a white background, and every seed gave mid-gray.

## Run 6 - 2026-10-11

Question: does the full (undistilled) model hold on its real weights: 50 UniPC steps, CFG 5, shift 3?

- **Tooling:** the F32 floor moved from a scratch script into `scripts/hidream_o1_parity.sh --f32 <forwards>`.
  - The reference pass records the exact forward inputs of the listed forwards, uncond ones included.
  - A second pass replays them in F32 on CPU and passes the result as `--against`.
- **Rebase:** the rebase onto master merged the branch's `n_choices` into master's new `run_unlogged`. Both now go through one `dispatch` with the choice count.
- **Command:** `HIDREAM_REPO=... HIDREAM_PYTHON=/mnt/data/venvs/hidream/bin/python TMPDIR=/mnt/data/_tmp scripts/hidream_o1_parity.sh /mnt/data/models/HiDream-O1-Image --size 1024x1024 --f32 0,1,50,98 --keep-dumps /mnt/data/_tmp/hidream_full_1024`
  - It ran on CUDA in BF16 on both sides, with the default fox prompt and seed 7.

Raw:
- **Reference:** 100 forwards (cond and uncond per step), 31.6 s for 50 steps.
- **F32 floor:** cosine to F32, ours against the reference:

| forward | ours vs F32 | reference vs F32 |
|---|---|---|
| 0, cond, pure noise | 0.996017 | 0.996440 |
| 1, uncond | 0.999804 | 0.999796 |
| 50, cond, step 25 | 0.998301 | 0.998302 |
| 98, cond, step 49 | 0.999936 | 0.999940 |

- **Teacher forced:** min 0.999014 (forward 2), mean 0.999902. That clears even the 0.999 gate the Dev model missed: undistilled, its noisy steps are less sensitive to BF16.
- **Free running:** sampled in 28.5 s.
  - Against the reference image: mean abs diff 8.6, max 206, PSNR 24.43 dB.
  - The x_pred cosine by step falls from 0.9992 at step 0 to 0.9754 at the last step (49), as the two trajectories drift under guidance 5.
  - Side by side, it is the same fox, pose, framing, light and background trees; only small snowy branches differ.

Implication: the full variant's UniPC sampler, CFG (uncond prompt and guidance combine) and shifted schedule match the reference on real weights. Next: review, CI, PR.

## Run 7 - 2026-10-11

Question: what did the branch review find, and do the fixes keep parity on both variants?

Fixes:
- **The F32 gate:** `--f32` waived `--min-cosine` for every forward. Now each forward with an `f32.<k>` entry must sit at least as close to F32 as the reference; every other forward keeps `--min-cosine`.
  - The run reports the worst gated forward.
  - The F32 pass refuses indices the reference never recorded.
  - The manifest includes the reference checkout's commit.
- **Prompt length:** a prompt longer than the patches' M-RoPE offset (4096) shared positions with them silently; `sequence` now refuses it.
- **Duplication:**
  - Qwen3-VL's `TextConfig` and `decoder_spec` moved to `inference_nn::decoder::qwen3_vl`. Qwen3-VL re-exports them and HiDream reuses them. Both had hardcoded `interleaved: true`.
  - The checkpoint file names and the shard filter are shared between the pipeline and the core loader: `hidream::{CONFIG_FILE, TOKENIZER_FILE, weight_shards}`.
- **Sampler settings:**
  - `Settings` has a `Sampler` enum (UniPC, or Flash with its clip std) instead of `unipc: bool`.
  - Dev's start-to-end noise interpolation, always equal, is gone.
- **Timestep embeddings:** they were built and uploaded on every forward, twice per step under CFG. `timestep_embeddings` now builds every step's token once per image.
- **Docs:** `serve diffusion` defaults to FLUX, so HiDream there needs `--diffusion-arch`. Auto-detection belongs to plain `serve -m`.

Parity after the fixes:
- **Full, cached dumps:**
  - The F32 numbers are identical, and teacher forced is unchanged: min 0.999014.
  - The sample moved from 24.43 to 24.29 dB PSNR, because the batched timestep MLP rounds differently in BF16.
- **Dev, `--f32 0,14` first:**
  - The F32 numbers reproduce Run 5 exactly (forward 0: ours 0.993795, reference 0.975157).
  - The new gate failed on the forwards without F32: min 0.99275.
- **Dev, `--f32 0,1,2,3,4,5,6,7,14`** (the noisy steps; 9 CPU forwards, ~30 min). Cosine to F32:

| forward | ours | reference |
|---|---|---|
| 0 | 0.993795 | 0.975157 |
| 1 | 0.996360 | 0.996369 |
| 2 | 0.997671 | 0.996632 |
| 3 | 0.997379 | 0.996520 |
| 4 | 0.997479 | 0.997542 |
| 5 | 0.997042 | 0.996959 |
| 6 | 0.997845 | 0.997373 |
| 7 | 0.996788 | 0.994310 |
| 14 | 0.999072 | 0.997890 |

- **Reading:**
  - Ours is closer on six forwards, and within 1e-4 on the other three.
  - The forwards without F32 reach min 0.996626 (forward 8). That is inside the spread Run 5 measured between two BF16 runs of the reference that differ only in the SDPA kernel: 0.9917 at forward 0, 0.9960 at forward 7.
- **Gate:** the script defaults Dev's `--min-cosine` to 0.99, the measured kernel spread. Full keeps 0.999.
- **Result:** Dev passes with the full F32 list.
