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

## Run 8 - 2026-10-11

Question: what HiDream-O1 GGUFs exist, and what would loading them take?

- **Hub search:** the only HiDream-O1 GGUFs are in `smthem/HiDream-O1-Image-Dev`: Full, Dev and Dev-2604, each at Q8_0 (10.3 GiB) and Q6_K (9.2 GiB).
  - The other quantized releases are safetensors: ComfyUI fp8 and mxfp8, nvfp4, SDNQ, int8, and MLX.
  - `calcuis/hidream-gguf` and `city96/HiDream-I1-*-gguf` are the older HiDream-I1/E1, a different model.
- **Header, read with a 32 MiB HTTP range** (`gguf_header.py`, scratchpad):
  - GGUF v3 with three keys only: `general.architecture = qwen`, `general.type = diffusion`, `general.file_type` (7 = Q8_0, 18 = Q6_K).
  - So it is the ComfyUI-GGUF (city96) convention: no config, no tokenizer.
- **Tensors:** 759 under the original Hugging Face names, including the vision tower (`model.visual.*`) and `lm_head`, which text-to-image never reads.
  - Q8_0 file: 306 Q8_0 (all text-stack matrices), 56 F16 (embedding, `lm_head`, part of the vision tower), 11 BF16 (`x_embedder`, vision merger), 386 F32 (norms, biases, `final_layer2`).
  - Q6_K file: 180 Q6_K plus 126 Q8_0 matrices; the rest as in Q8_0.
- **Config and tokenizer:** `HiDream-O1-Image` and `-Dev` have byte-identical `config.json` and `tokenizer.json`, so either repo serves any of the GGUFs.
- **Existing support:**
  - The diffusion crate's `gguf::var_builder` maps GGUF names onto a `ShardedVarBuilder` over a `GgufWeightSource`, as FLUX's GGUF path does.
  - The shared decoder's linears take quantized weights straight from such a source (`distributed/layers.rs` checks `weight_source()`), so the text stack runs Q8_0/Q6_K without dequantizing.

Plan:
- Bind every tensor except `lm_head` and `model.visual.*`.
- Detect a HiDream GGUF by `general.type = diffusion` plus `model.x_embedder.proj1.weight`.
- Take the variant from the filename (`-dev`, as for repo names), overridable with `--diffusion-arch`.
- Read the config and tokenizer from beside the file, else from the HiDream-ai repo.
- Next: download Dev Q8_0 and Full Q6_K to `/mnt/data/models/hidream-o1-gguf`, implement, and run parity against the reference dumps of Runs 6 and 7.

## Run 9 - 2026-10-11

Question: does a HiDream-O1 GGUF load, ours or the published one, and how much do the quantized types cost?

Implementation:
- **Loading:**
  - `gguf::hidream_native_name` binds every tensor but `lm_head` and `model.visual.*`.
  - `hidream::local_gguf` finds the file by contents, not name: `general.type = diffusion` plus the patch embedder.
  - Diffusion loaders now name their GGUF tensor mapping (`gguf_native_name`); FLUX's was hardcoded.
  - The auto loader probes a local path, and `HiDreamO1Loader::local_paths` takes the GGUF.
  - The config and tokenizer come from beside the file, else from `HiDream-ai/HiDream-O1-Image`.
- **Writing:** `inference quantize hidream-o1 -m <dir|repo> -o <file> --type <t> [--keep fragment=type]` (`hidream::gguf_writer`).
  - Types: text-stack matrices quantized; embedding F16; other rank-2 BF16; rank-1 and `final_layer2` F32.
  - `GgmlDType::{from_name, name}` over `WRITABLE_TYPES` moved to inference-tensor, shared with the `asr` writer.
  - `GgufWriteSummary` is shared too.
- **Conversion versus download:** the user asked whether converting would beat downloading.
  - The download ran at 3.5-6.7 MiB/s, ~45 min for the two files.
  - Converting Dev to Q8_0 from the local F32 shards, already in the page cache, took 9.2 s: 8.1 GiB, 252 Q8_0, 150 F32, 4 BF16, 1 F16.
  - Peak RSS 41 GB, mostly the mmapped shards.
- **Byte comparison** against `smthem`'s Dev Q8_0, with a custom header parser:
  - llama.cpp's gguf-py refuses their file: `model.visual.patch_embed.proj.weight` has 5 dimensions.
  - 405 of our 407 tensors are byte-identical, including all 252 Q8_0 matrices.
  - Two differ only in type: their `t_embedder1.mlp.0.weight` is F32 (ours BF16), and one more is F16 where ours is F32.
  - They carry 352 more tensors: the vision tower and `lm_head`.
- **The 5-dimension cap:** our GGUF reader also refused their file, with GGML's 4-dimension cap, even though that tensor is never bound. The cap is now 8 in both readers; the element-count overflow check remains.

Dev at 1024x1024, cosine to F32 at forwards 0-7 and 14, against the Run 7 dumps:

| file | size | F32 cos 1-7 | gated min | image PSNR | 28 steps | GPU peak (less 0.9 GiB idle) |
|---|---|---|---|---|---|---|
| BF16 safetensors (Run 7) | 16 GB | 0.9964-0.9978 | 0.9966 | 24.87 dB | 8.4 s | |
| ours Q8_0 | 8.1 GiB | 0.9823-0.9896 | 0.9891 | 22.92 dB | 6.6 s | 8.6 GiB |
| smthem Q8_0 | 10.3 GiB | identical to ours | 0.9891 | 22.30 dB | 6.6 s | |
| Q6_K + `down_proj` and `o_proj` Q8_0 (smthem's Q6_K mix) | 7.1 GiB | 0.9712-0.9760 | 0.9825 | 20.89 dB | 7.3 s | |
| Q6_K + `down_proj` Q8_0 | 7.0 GiB | 0.9714-0.9755 | 0.9820 | 20.54 dB | 7.4 s | 7.5 GiB |
| Q6_K + `o_proj` Q8_0 | 6.7 GiB | 0.9440-0.9649 | 0.9679 | 18.80 dB | 7.6 s | |
| Q6_K | 6.6 GiB | 0.9472-0.9618 | 0.9691 | 17.57 dB | 7.7 s | 7.1 GiB |
| Q4_K | 4.9 GiB | NaN | NaN | 4.5 dB | | 5.5 GiB |

- **Images:** Q8_0 Dev's is the same fox, pose and light as the reference, with the twigs shifted like BF16's.
- **Q4_K NaN:** a CUDA probe in inference-quant used 512x4096 BF16 activations and a 512x4096 weight.
  - Plain outliers up to 3e4: all of Q8_0/Q6_K/Q4_K stay at cos 0.9999+.
  - Three 3e4 outliers inside one 32-value block: Q4_K gives NaN.
  - The cause: MMQ quantizes activations to Q8_1, and the DS4 layout stores each block's sum as F16 (`make_half2(d, sum)`, as upstream llama.cpp does), so a sum past 65504 is inf.
  - Layouts: Q4_0, Q4_1, Q5_1, Q4_K and Q5_K read DS4; Q8_0, Q5_0, Q3_K and Q6_K store the scale only (D4); Q2_K keeps float sums (D2S6).
  - The diffusion var builder holds no packed-affine reservation, so every prompt-sized matmul here goes through MMQ.
  - The writer refuses DS4 types for HiDream. The kernel fix (float sums, or sums stored in units of d) is shared-kernel work and should get its own issue: any GGUF model with clustered outlier activations is exposed.
- **Q6_K:** its matmul matches the dequantized one (probe), so the loss is the weights. `down_proj`, whose input carries the outlier channels, is the sensitive matrix; `o_proj` adds nothing over it.
- **Bar for a quantized file** (script default with `--gguf`):
  - cosine to the BF16 reference at least 0.98 on forwards without F32;
  - at most 0.03 further from F32 than the reference.
  - Ours Q8_0, smthem's Q8_0 and our Q6_K + `down_proj` pass; plain Q6_K fails 8 forwards.

Full at 1024x1024, our Q8_0 against the Run 6 dumps:
- **F32 cos, ours vs reference BF16:** 0.997023 vs 0.996440 (forward 0), 0.999877 vs 0.999796 (1), 0.999455 vs 0.998302 (50), 0.999863 vs 0.999940 (98).
- **Teacher forced:** min 0.990039, mean 0.998056.
- **Free running:** 21.74 dB PSNR in 22.3 s (BF16: 28.5 s).
- **Image:** the same fox, pose and framing.
- **Reading:** the undistilled model is as robust to Q8_0 as to BF16; Dev's distilled noisy steps are where quantization shows.

Next:
- smthem's Full Q6_K, once downloaded;
- full CI;
- an issue for the DS4 F16-sum overflow.

## Run 10 - 2026-10-11

Questions:
- does the published Full Q6_K load and pass?
- does the F16 activation-sum overflow reproduce in upstream llama.cpp?
- what did the GGUF review change?

- **`smthem/HiDream-O1-Image-Q6_K.gguf`** (9.2 GiB; Q6_K with `down_proj` and `o_proj` at Q8_0) against the Run 6 dumps:
  - F32 cos, ours vs reference BF16: 0.993718 vs 0.996440 (forward 0), 0.999721 vs 0.999796 (1), 0.997743 vs 0.998302 (50), 0.999834 vs 0.999940 (98).
  - Teacher forced: min 0.992568, mean 0.999318. Free running: 21.61 dB, 24.8 s. It passes the quantized bar.
- **Upstream reproduction** on llama.cpp `5c8854fd8` (2026-10-11), built in the user's checkout (`build_cuda`, CUDA 12.8, RTX 3090).
  - A standalone ggml program (`q81_repro.cpp`, scratchpad) runs one `MUL_MAT`: K=4096, N=512, weights 0.02*randn quantized by `ggml_quantize_chunk`.
  - Activations are randn, with every other row carrying 3 adjacent channels (7, 8, 9) set to the outlier value. CUDA is compared with CPU.
  - M=512, outlier 3e4:
    - Q4_K and Q5_K are non-finite in 131072 of 262144 outputs, i.e. every outlier row; the clean rows match CPU.
    - Q8_0, Q6_K and Q4_0 match CPU (cos 1.000000).
  - M=1, outlier 3e4: Q4_0 is non-finite in all 512 outputs; Q8_0, Q6_K, Q4_K and Q5_K match.
  - M=512, outlier 1e4 (block sum 3e4, under F16's 65504): Q4_K matches CPU (0.999999).
  - Upstream still writes `make_half2(d, sum)` in `quantize.cu` (the MMVQ `block_q8_1` at line 100, the MMQ DS4 layout at line 548).
  - CPU is unaffected.
- **The report:** the user files llama.cpp reports themselves, since that project forbids AI-written issues. The facts above are for that report.
- **Review fixes:**
  - A checkpoint directory (`config.json` plus shards) always loads its own weights; a GGUF beside it is ignored.
  - A directory is probed only when it holds exactly one `.gguf`.
  - The variant rule also reads the directory's name.
  - Both GGUF writers flush explicitly, so a failed final write is reported.
  - `write_hidream_gguf` fetches through the loader's offline-aware `get_model_paths` and `hf::get_file`.
  - `GgmlDType::from_name` returns the error, `GgufWriteSummary` is re-exported at the `inference_api` root, and the help lists only allowed types.
  - The guard covers exactly the writable DS4 types (Q4_0, Q4_K, Q5_K); the upstream run confirms Q4_0 on the MMVQ path.
  - The engine test writes its Q8_0 and Dev files with the real writer.

## Run 11 - 2026-10-11

Question: what does the reference's image-conditioned path (editing with one reference image, subject references with two or more) add to text-to-image, and what do we lack?

Read: `models/pipeline.py::generate_image`, `models/utils.py` (`resize_pilimage`, `calculate_dimensions`, `get_rope_index_fix_point`), `models/qwen3_vl_transformers.py::_forward_generation`, `inference.py` (checkout `2c2d29f`, still the remote head).

Raw:
- **Each reference image enters twice.**
  - *Through the ViT.* It is LANCZOS-resized to `calculate_dimensions(384, aspect)`, multiples of 32 (384 becomes 288 for K of 5 to 8, and 192 above 8). Then the stock Qwen3-VL processor runs (mean/std 0.5, patch 16, merge 2) and it is placed in the user turn as `<|vision_start|><|image_pad|>*N<|vision_end|>`, one per image, before the caption. The ViT output replaces the pads, and the deepstack features are added at those rows after decoder layers 0-2, as in stock Qwen3-VL.
  - *As pixel patches.* `resize_pilimage(img, max_size, 32)`: BOX halving while the short side is at least 2 * max_size, then BICUBIC resize to the largest 32-aligned size whose area fits max_size^2, then a center crop. It is normalized to [-1, 1] and patchified like the target. It runs through `x_embedder` and is appended after the noisy target patches, at every step, unchanged.
  - `max_size` is `max(h, w)` for K = 1, then 48/64 for K = 2, 1/2 for K up to 4, 24/64 for K up to 8, and 1/4 above that.
- **Mask:** rows of type > 0 (the timestep token, the target and the reference patches) see the whole sequence; the prompt rows, image pads included, stay causal. This is our text-to-image mask with the prefix longer.
- **Positions** (`skip_vision_start_token = [0]*K + [1] + [1]*K`):
  - the prompt and its image pads use stock Qwen3-VL M-RoPE, on the merged grid;
  - the target sits at `(4096, 4096 + row, 4096 + col)` as before;
  - each reference continues at the previous maximum + 1, `(s, s + row, s + col)`.
  - The first "pad" of each grid is a `vision_start` id standing in for patch 0, which the skip flag accounts for. No extra token.
- **CFG** (Full): the uncond sample is `" "` with the same K images; the ViT runs once and its outputs are reused for both samples and every step.
- **Samplers:**
  - Full: UniPC as in text-to-image.
  - Dev with exactly one reference (editing) defaults to diffusers' `FlowMatchEulerDiscreteScheduler` over the same 28 timesteps, sigmas `t/1000` then 0: plain Euler, `z' = z + (sigma_next - sigma) * (-v)`, with no re-noising. `--editing_scheduler flash` keeps the flash sampler. Dev with two or more references uses the flash sampler.
- **Output size:** snapped to the 11 presets, unless `keep_original_aspect` with one reference: the reference is resized to max 2048 and the output takes its size.
- **Layout and skeleton** conditioning are extra reference images drawn by `create_layout_reference_images`; the model sees nothing new.

What we lack:
- the Qwen3-VL vision tower. It lives in `inference-models-qwen`, which the always-built diffusion crate cannot depend on. Our GGUF writer drops `model.visual.*` (`HIDREAM_UNREAD`); the published `smthem` Dev Q8_0 keeps all 351 ViT tensors (F16 weights, F32 norms and biases);
- PIL-style resizes for the two reference paths;
- the sequence and position builder with images;
- the Euler step;
- an edit request: there is no `/v1/images/edits`, no engine operation and no ABI entry.

Plan:
1. Move the Qwen3-VL vision encoder (`vision.rs`, `VisionConfig`, `deepstack_process`) to `inference-nn`, so one implementation serves Qwen3-VL, Qwen3.5 and HiDream.
2. HiDream reference images: model, samplers, writer keeps the ViT, an `edit_image` engine operation, `/v1/images/edits` (multipart and JSON), the ABI and bindings, and the parity script with `--ref-image`.
3. The demo's Images tab gains an edit mode.

## Run 12 - 2026-10-11

Question: does the reference's Dev edit run here, and at what size can parity check it?

Commands:
- `scripts/hidream_o1_parity.sh <Dev> --dev --prompt "remove the earphones" --size 1024x1024 --ref-image assets/edit/test.jpg --keep-dumps ...`. The prompt and image are the reference repo's own editing sample, a 1792x2304 product poster with a person.
- `inference.py --model_type dev --keep_original_aspect` on the same inputs, through a wrapper that forces the 4D-mask path (flash_attn is not installed here).
- `vit_check.py` (scratchpad): the reference ViT in BF16 on CUDA against F32 on CPU, on the dumped `pixel_values`.

Raw:
- **At 1024x1024 the reference itself fails.** The output is noise. x_pred is a flat grey-blue field for the first ~10 steps (std 0.10-0.11), then turns to noise (std 0.83 at the last step).
  - The sequence is 146 prompt tokens (130 image pads for a 26x20 ViT grid), then 1024 target patches and 1008 reference patches (28x36), 2178 rows in all.
- **The ViT is not the cause** (torch 2.11; the README warns against 2.9): main output cosine 0.99822, deepstack 0.99990 / 0.99915 / 0.99865.
- **At the trained size it works:** 1792x2304 with `--keep_original_aspect`, 28 steps in 85 s (3.07 s/step, BF16, 4D mask) on the RTX 3090. The earphones are gone; the rest of the poster (text, can, pose, background) is kept.

Implications:
- Editing parity must run at full size: 4032 target patches plus 4032 reference patches, about 8.2k rows. The dumps are about 2.8 GB for Dev.
- Small sizes are no smoke test for editing. The tiny random-weight engine test only checks plumbing and determinism.

## Run 13 - 2026-10-11

Question: does our reference-image path (ViT in the prompt, reference patches, the Euler edit sampler) match the reference's Dev edit?

Commands:
- `scripts/hidream_o1_parity.sh <Dev> --dev --prompt "remove the earphones" --ref-image assets/edit/test.jpg --keep-aspect --keep-dumps /mnt/data/_tmp/hidream_edit_dev_keep`;
- the same dumps with the reference image re-saved as a PNG of Pillow's own decode, replayed by the example alone.

Raw:
- **Layout:** output 1792x2304 and ViT grid [1, 26, 20], as the reference. Sample 0 has 146 tokens and 8210 rows; ids and every M-RoPE position match exactly.
- **Preprocessing from the JPEG:** the ViT pixels differ by up to 2.24 8-bit levels and the patches by up to 4.23, even after rounding ours to the BF16 the reference feeds.
  - From Pillow's decode as a PNG, the ViT pixels match bit for bit, so the port of Pillow's resampler (fixed point, horizontal then vertical, Lanczos) is exact. The JPEG gap is the decoder: `image` against libjpeg-turbo.
  - The patches differed by 3e-5 in 24154 values until the scale became `x * (1/255)`. torchvision multiplies (0 of 256 levels differ against `ToDtype(scale=True)`); dividing differs in 111. Now 0 of 12386304 differ.
- **Teacher forced** (28 forwards): x_pred cosine min 0.995955 (forward 0, the noisiest), mean 0.999778. From the JPEG directly: min 0.995716, mean 0.999781.
- **Free running:** 1792x2304 in 95.9 s (the reference: 87 s), PSNR 29.03 dB, mean |diff| 3.5. By eye they are the same edit, and both carry the same faint speckle.
- The per-step cosine stays at 0.9999-1.0000 through the middle, then drops to 0.995 at the last steps, as in text-to-image.

Implications: the model side is right. The JPEG decoder costs about 0.0002 of cosine, so it is not worth a libjpeg dependency. Next: Full with CFG (the uncond sample sees the images too), Dev with two references (the flash sampler), and a GGUF that keeps the ViT.

## Run 14 - 2026-10-12

Question: does the full model's edit (UniPC, CFG with the images in the unconditional sample too) match?

Command: `scripts/hidream_o1_parity.sh <Full> --prompt "remove the earphones" --ref-image <Pillow PNG of the sample> --keep-aspect --keep-dumps /mnt/data/_tmp/hidream_edit_full_keep`

Raw:
- **Reference:** 100 forwards (50 steps x cond/uncond) at 1792x2304, 5m37s (6.75 s/step).
- **Layout:**
  - Sample 0 has 146 tokens and 8210 rows; sample 1 (`" "` with the same image) has 143 tokens and 8207 rows. Ids and positions match for both.
  - ViT pixels and patches are bit-exact.
- **Teacher forced:** x_pred cosine min 0.996823 (forward 1, the uncond half of the noisiest step), mean 0.999916. This misses the default 0.999 bar the script holds Full to without F32 replays; Run 7 put the BF16 floor of those noisy forwards at about 0.9966.
- **Free running:** 346.5 s, PSNR 33.99 dB, mean |diff| 3.79, max 59. The per-step x_pred cosine is 0.9996-1.0000 throughout. By eye it is the same clean edit (no Dev speckle).

Implication: the full model's edit is right. Next: an F32 replay of forwards 0 and 1, so the noisy forwards are held to the F32 bar rather than a bare cosine.

## Run 15 - 2026-10-12

Question: do several references (the README's 10-image subject example, Dev, flash sampler) match? And our Q8_0 GGUF with the ViT kept?

Commands:
- `scripts/hidream_o1_parity.sh <Dev> --dev --prompt <the README's subject prompt> --size 2048x2048 --ref-image assets/IP/{1..10}.jpg --keep-dumps ...`;
- the same with PNGs of Pillow's decode;
- the example with `--their-inputs` (the reference's own ViT pixels and patches) and `--their-vision` (its ViT outputs);
- `vit_check.py` for F32 truth;
- `inference quantize hidream-o1 -m <Dev> -o .../vit/HiDream-O1-Image-Dev-Q8_0.gguf`, replayed on the Run 13 edit dumps.

Raw:
- **Q8_0 with the ViT:** written in 18 s, 9.88 GB (8.73 GB without the ViT). On the edit: teacher forced min 0.996242, mean 0.999753; free running 93.2 s, 31.86 dB. It passes the quantized bar.
- **Subject layout:**
  - 10 ViT grids match the reference's. With 10 references the ViT input is 192 px, below the processor's 65536-pixel minimum, so its own upscale runs; that path works.
  - Sample 0 has 850 tokens and 7339 rows; ids and positions match.
  - From PNGs, the patches are bit-exact (BOX halving and BICUBIC fit). The ViT pixels differ by up to 2 levels in 7514 of 4435968 values: our Pillow bicubic stands in for torchvision's in the processor's upscale.
- **Subject teacher forced, first:** min 0.967773 (forward 2), mean 0.992817, against the edit's 0.9957 / 0.9998. Free running 86.5 s at 19.27 dB: the same boy, pose, props and steps, with the details diverging.
- **Bisecting:**

  | step | ViT output cos vs reference | teacher forced (min / mean) |
  |---|---|---|
  | as written | 0.994814 | 0.966840 / 0.992935 |
  | `--their-inputs` | 0.995134 | 0.964228 / 0.993641 |
  | `--their-vision` (reference ViT outputs) | - | 0.988542 / 0.997204 |
  | our ViT in F32 vs reference F32 | 0.999999 | - |

  The implementations agree; ours in BF16 is noisier. Against F32: ours 0.995765, the reference's BF16 0.998230, and the reference's own BF16 rerun 0.9979.
- **Fixes found on the way:**
  - The merger's GELU is `nn.GELU()` (erf). Ours was the tanh approximation, in all three Qwen3-VL mergers; little effect here.
  - `Conv3dNoBias` (the Qwen2/2.5/3-VL patch embedding) ran as two conv2ds summed in BF16; it is now one matmul for a single-kernel input. Little effect.
  - **The CUDA tanh-GELU for bf16/f16 evaluated every step of the formula in half precision.** It now computes in float and rounds once, as torch does; the CPU path likewise, and `gelu_erf`, `elu`, f16 `silu`/`sigmoid` round once too. Ours vs F32: 0.995765 -> 0.997309, deepstack 0.9998 / 0.9995 / 0.9987. Teacher forced: 0.973951 / 0.994615.
  - Attention in F32 (an experiment, reverted) moved ours vs F32 from 0.99549 to 0.99606. Head dim 72 has no flash kernel here, so the cuBLASLt path rounds the logits to BF16.

Implications:
- Ten references amplify the conditioning's noise far more than one. The reference's own BF16 run is the noisy party too.
- The fair bar is the F32 one, as for text-to-image's noisy steps: replay forwards 1 and 2 in F32 and hold ours to the reference's distance from it.

## Run 16 - 2026-10-12

Question: held to F32 rather than to the reference's BF16, how far is ours on the 10-reference subject run?

Commands:
- `scripts/hidream_o1_parity.sh <Dev> --dev ... --ref-image <10 PNGs> --f32 1,2 --keep-dumps /mnt/data/_tmp/hidream_subject_f32`. The F32 pass now leaves out the reference's cached `precomputed_*` ViT outputs, so it encodes the images in F32 itself; before that, the recording failed on the deepstack list.
- `floor_refs.py` (scratchpad): the reference in BF16, replaying forwards 1 and 2 from their recorded inputs.

Raw:

| forward | ours vs F32 | reference (dumped run) vs F32 | reference replayed vs F32 | replayed vs dumped |
|---|---|---|---|---|
| 1 | 0.978767 | 0.986232 | 0.984344 | 0.992801 |
| 2 | 0.980801 | 0.988201 | - | - |

- MATH SDPA does not fit 24 GB at 7339 rows (6.4 GiB per allocation), so the floor comes from the replay alone. Two BF16 runs of the reference itself are 0.9928 apart.
- With vision attention in F32 (experiment, reverted): the ViT is closer to the reference (0.998095), but forwards 1 and 2 land at 0.974199 and 0.978902, further from F32. One precision change moves a forward by about 0.005 either way here.
- Teacher forced against the BF16 reference (min / mean): 0.977937 / 0.994979. Free running: 82.0 s, 19.09 dB; same subject, pose and props, with details apart.

Implications:
- Ten references put the denoiser in a regime where BF16 noise in the conditioning moves predictions by about 0.01. Ours is about 0.006 further from F32 than torch's BF16, the same order as the run-to-run spread of the reference itself.
- The rest is our cuBLASLt attention for the ViT's head dim 72 (BF16 logits), which F32 attention did not cure. It is kept in BF16: in F32 it costs the chat models with large images and buys nothing measurable here.
- One reference (editing) stays at the BF16 bar (Runs 13, 14), which is the case the reference recommends.

## Run 17 - 2026-10-12

Question: the full model's edit held to F32 at its two noisiest forwards (cond and uncond of step 0), after the GELU fix.

Commands:
- `scripts/hidream_o1_parity.sh <Full> ... --keep-aspect --f32 0,1 --keep-dumps /mnt/data/_tmp/hidream_edit_full_f32`;
- the example's new `--quick` (only the F32-covered forwards, no sampling), with and without F32 vision attention;
- `vit_check.py` for this image's F32 ViT.

Raw:

| forward | ours vs F32 | reference vs F32 | ours, F32 vision attention |
|---|---|---|---|
| 0 (cond) | 0.998657 (ok, slack 1e-3) | 0.999561 | 0.998450 |
| 1 (uncond, `" "` with the image) | 0.995033 (0.003 short) | 0.998115 | 0.998575 |

- **ViT vs F32 on this image:** ours (BF16 attention) 0.997640; the reference's dumped BF16 run 0.996296; the reference's two BF16 runs are 0.99797 apart. Our ViT is the closer of the two BF16 encoders.
- **Teacher forced:** every forward without F32 has min 0.999758. Free running: 339.7 s, 34.48 dB, max |diff| 62.

Implications:
- The uncond forward reads little but the image, so the ViT's BF16 noise moves it by a few thousandths. That is noise, not a defect: our encoder is nearer F32 than the reference's, and switching our attention to F32 only trades which forward comes out ahead.
- Vision attention stays BF16.
