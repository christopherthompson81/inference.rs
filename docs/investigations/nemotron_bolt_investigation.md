# Nemotron 3.5 Bolt 74B-A9B

Bolt is a Puzzle-pruned Nemotron-H (`nemotron_h_puzzle` inside the `NemotronH_Omni_Reasoning_V3` wrapper) with latent
MoE. Target: NVIDIA's own 24 GB GGUF (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`, text file 20.5 GiB) on one
RTX 3090, scored against llama.cpp on the same file.

## Run 1 - 2026-11-02 20:45 (local clock)

**Question:** what does Bolt's text GGUF need beyond Nemotron 3 Nano?

**Commands:** a header reader over the GGUF (metadata and tensor inventory), the BF16 checkpoint's `config.json`,
`llama-perplexity -ngl 99 -c 512 --chunks 40 --kl-divergence-base` on wikitext-2, our perplexity example.

**Findings:**
- 88 layers (40 Mamba, 8 attention, 40 MoE); hidden 4096, 32 query / 2 KV heads, Mamba 128 heads x 64.
- Latent MoE: `moe_latent_size` 1024, `ffn_latent_down` / `ffn_latent_up` (Q8_0) around the routed experts; the router
  and the shared expert (5376 wide) see the full width.
- Puzzle layout: per-layer `expert_feed_forward_length` (1536 - 2560) and `expert_used_count` (9) arrays;
  `expert_count` is the largest (496), each layer's own count (240 - 496) only in its router's shape. The checkpoint's
  `block_configs` gives all three per layer.
- Types: routed and shared experts IQ2_XXS (68.9G params), attention / latent / embeddings / output Q8_0, Mamba
  `ssm_in` and `ssm_out` NVFP4 (GGML type 40, 4.4G params) each with an F32 `.scale` that llama.cpp multiplies onto the
  matmul's output (`build_lora_mm(..., ssm_in_s)`).
- llama.cpp base: PPL 5.6384 (`llamacpp-wiki-c512-ch40.kld` beside the GGUF).

**Changes:** latent MoE in `NemotronMoe` (`fc1_latent_proj` / `fc2_latent_proj`), per-layer MoE specs from
`block_configs`, each layer's expert count from its router, `NemotronHPuzzleForCausalLM`, GGUF synthesis of
`block_configs` from the per-layer arrays, a Nemotron-H GGUF resolving to `nemotron_h`. Tiny case
`a_latent_moe_decode_scores_as_one_prefill`.

**Blocker:** our load stops at `blk.0.ssm_in.weight`: NVFP4 is not a GGUF type we load. Next: adopt llama.cpp's
NVFP4 (block of 64: four UE4M3 sub-block scales, 32 bytes of E2M1) for dequantization, MMVQ and MMQ, with the
per-tensor scale after the matmul.

## Run 2 - 2026-11-02 21:42 (local clock)

**Question:** with NVFP4 GGUF tensors loading (the `gguf-nvfp4` branch merged locally over this one), does Bolt score
as llama.cpp does on the same file, and how fast is it on the 3090?

**Commands:** our perplexity example `--kld-base llamacpp-wiki-c512-ch40.kld` (CUDA, BF16 and `--dtype f32`
activations); `llama-perplexity -dev none -ngl 0 -t 8 --kl-divergence` against the same base for llama.cpp's own
CPU-against-CUDA level; `inference serve` with scratch `bench/text_prompt.py` (539 tokens) and `bench/decode.py`
(128 tokens); `llama-bench -ngl 99 -p 512 -n 128 -r 3`.

**Findings on the way:**
- The first load gave non-finite logits on CUDA and the CPU alike. Each block's largest input (temporary prints)
  went 0.055 -> 22528 across block 0: the NVFP4 tensor-wide scales (1.3e-4 - 1.6e-3) were never applied, because
  the shared GGUF name parser takes only `weight` / `bias` suffixes and left `blk.N.ssm_in.scale` as an unknown role.
  The Nemotron-H bindings now split `.scale` off themselves; magnitudes after: 0.055, 0.28, 0.44, ...
- The engine hid the cause behind "Internal server error." (`scored()` in `engine_logits.rs` found the logits not
  finite and returned no detail); found with a temporary print.

**Result (wikitext-2, 40 chunks of 512, 10,200 positions):**

| run | PPL | mean KLD | median | p99 | same top-1 |
|---|---|---|---|---|---|
| llama.cpp CUDA (the base) | 5.6315 | | | | |
| ours CUDA, BF16 activations | 5.6365 | 0.00279 | 0.00096 | 0.0269 | 97.72 % |
| ours CUDA, F32 activations | 5.6395 | 0.00250 | 0.00074 | 0.0242 | 98.12 % |
| llama.cpp CPU against its CUDA base | 5.6424 | 0.00263 | 0.00083 | 0.0263 | 97.82 % |

Ours sits on either side of llama.cpp's own backend-to-backend level; at 2 bits per expert weight that level is 2.6x
Nemotron Nano 2's Q4_K_M (0.00102).

**Speed (RTX 3090, 8192-token server):**

| | prompt (539 / 512 tokens) | decode |
|---|---|---|
| ours | 1,898 - 1,905 tok/s (939 the first, cold) | 91.4 - 91.6 tok/s |
| llama.cpp `llama-bench` | 1,622 tok/s (`pp512`) | 89.5 tok/s (`tg128`) |

The server holds 23.1 GiB of the card (the desktop's share included). Chat: "17 * 23" -> 391; a time sum reasoned
correctly in 98 tokens; a coherent story.

**Open:** F16 activations could overflow before the NVFP4 scale (raw outputs reach ~1e5); BF16 (the GGUF default on
CUDA) and F32 do not. Bolt's vision and audio towers (C-RADIO, Parakeet) and its "lilicorr" draft layers (past
`block_count`, skipped by our bindings) are not loaded.

## Run 3 - 2026-11-03 08:30

**Question:** what does Bolt's image input need, and what is the reference?

NVIDIA's GGUF repo carries `mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` (1.7 GB, `clip.projector_type
nemotron_omni`); Bolt has no sound tower (`sound_config: null`). The tower is C-RADIO (ViT-H/16: 32 blocks, width 1280,
16 heads, 10 prefix tokens = 3 cls + 7 registers, a 128 x 128 CPE position grid, GELU-erf MLP, no layer scale weights),
then InternVL's pixel shuffle (`ps_version v2`, factor 2) and `mlp1`: RMSNorm(5120, eps 1e-5) -> 20480 -> ReLU^2 ->
4096, no biases. `vision_final_layernorm` is false.

**mmproj against the BF16 checkpoint** (shards 61-62 only, `cmp_mmproj.py`): every tensor is bit-identical under
llama.cpp's names. `v.patch_embd.weight` is the linear embedder reshaped to (1280, 3, 16, 16); `v.blk.N.attn_{q,k,v}`
concatenate to `attn.qkv`; `v.position_embd.weight` is `pos_embed` (16384, 1280); `v.class_embd` the 10 prefix tokens;
`v.patch_embd.weight.1` the video embedder.

**Where llama.cpp departs from transformers (so llama.cpp is not the reference here):**

- Position embeddings: transformers resizes the 128 x 128 grid bilinearly to a `max(h, w)` square and crops to (h, w);
  llama.cpp's `resize_position_embeddings` resizes straight to (h, w). Same only for square images.
- Image size: transformers picks the patch grid from a patch budget (1024 - 13312 patches, aspect kept, each side even,
  Python's `round(x + 0.5)`, so ties go to even), then antialiased bicubic in float; llama.cpp's `dyn_size` uses its own
  min/max-pixel rule.

**Reference** (`bolt_ref/make_ref.py`, f32 on CUDA, NVIDIA's `image_processing_nemotron_h_omni.py` and `modeling_radio.py`
with the checkpoint's weights) on three synthetic images:

| image | resized to | patches | image tokens |
|---|---|---|---|
| 500 x 375 | 608 x 448 (scaled up to the 1024 floor, width made even) | 1064 | 266 |
| 640 x 480 | unchanged | 1200 | 300 |
| 4000 x 3000 | 2112 x 1600 (budget-capped, antialiased down) | 13200 | 3300 |

Saved: pixels, RADIO features and projected embeddings per image. Next: the Rust tower against these.

## Run 4 - 2026-11-03 08:44

**Question:** does our C-RADIO tower and projector match transformers?

`nemotron_omni_parity` (example in `inference-models-other`) loads the BF16 checkpoint's vision shards, runs our
processor on each PNG and the tower on the reference's own patches and on ours, and compares in f64 (an f32 cosine over
13.5 M values drifted past 1 in the first pass).

| | patches max diff | features cosine | embeds cosine | embeds max diff (of max) |
|---|---|---|---|---|
| CUDA f32, 640 x 480 | 0 | 1.0000000 | 1.0000000 | 4.4e-3 (16.4) |
| CUDA f32, 4000 x 3000 | 2.3e-4 | 1.0000000 | 1.0000000 | 1.2e-2 (31.8) |
| CUDA f32, 500 x 375 | 9.9e-5 | 1.0000000 | 1.0000000 | 2.2e-3 (15.5) |
| CPU f32, 500 x 375 | 9.9e-5 | 1.0000000 | 1.0000000 | 5.1e-5 (15.5) |
| CUDA BF16, 640 x 480 | | 0.9999759 | 0.9998552 | 0.27 |
| CUDA BF16, 4000 x 3000 | | 0.9999372 | 0.9992064 | 1.11 |
| CUDA BF16, 500 x 375 | | 0.9999824 | 0.9997701 | 0.63 |

The resize (antialiased bicubic, width pass first) and patch order match NVIDIA's processor to 2e-4 on [-2, 2] inputs.

**BF16 floor** (`bf16_floor.py`: transformers' own tower in BF16, projector in f32, against its f32 run): features
0.999985 on all three, embeds 0.99985 - 0.99990. Ours is at that floor on the small images and 5x past it on the 13,210
token one. Probes on the large image:

| attention inputs | features cosine | embeds cosine | tower time |
|---|---|---|---|
| BF16 (fattn) | 0.9999372 | 0.9992 | 870 ms |
| F16 (fattn) | 0.9999345 | 0.9992 | |
| F32 (unfused path) | 0.9999846 | 0.9998 | 5,530 ms |

So the loss is fattn's own F16 arithmetic over 13 K keys, not its input type; transformers' BF16 attention keeps it.
At 1,200 patches the F32 path costs 89 vs 54 ms. Kept fattn: the gap is only at the largest images, where F32
attention costs 6x. Open choice for later: F32 attention below some patch count.

## Run 5 - 2026-11-03 09:09

**Question:** does the whole model answer about images, and does it agree with llama.cpp?

The mmproj through `nemotron_omni_parity --gguf <text> <mmproj>` gives the same numbers as the BF16 shards (f32,
cosine 1.0000000 on all three), so the bindings and the config synthesized from the two GGUFs are right.

**Fitting the card:** `serve --mmproj ...` ran out of memory creating the hybrid cache: the text GGUF alone held 23.1
GiB of the 3090 and the tower adds 1.7 GB. `--moe-expert-cache` did nothing: Bolt's experts are IQ2_XXS, which the CPU
expert path (`cpu_dot::supports`) does not run, so `offloadable_experts` is `None`. The auto map also predicted a fit
it did not have. `-n "0:43"` (5 of 48 blocks on the CPU) loads at 21.9 GiB.

**Answers** (thinking off, greedy, synthetic images: shapes and text drawn with PIL, and Run 3's 640 x 480 pattern),
against `llama-server` (970b51ce7, `-ngl 99 --n-cpu-moe 6`) on the same files:

| prompt | prompt tokens (ours / llama.cpp) | answer |
|---|---|---|
| 800 x 600 shapes, "what shapes, colors and text" | 508 / | "a red circle on the left, a blue square on the right, and the text 'HELLO 42' in black at the bottom" |
| 640 x 640 shapes, same | 433 / 433 | identical to llama.cpp's, word for word |
| 640 x 640 shapes, "every element in detail" | 439 / 439 | same first ~60 tokens, then each says something different and true |
| 640 x 480 pattern, "in detail" | 325 / 325 | same first ~45 tokens (grid of colored squares, two red circles), then differ |

The 640 x 640 image has no resize and no position-grid crop in either, so the two meet exactly there; on 640 x 480
llama.cpp resizes the position grid straight to 30 x 40 where the reference crops a 40 x 40 one.

**Speed with 5 blocks on the CPU:** prompt 27 - 32 tok/s, decode 0.9 - 1.6 tok/s; llama.cpp with 6 layers' experts on
the CPU: prompt 465 tok/s, decode 76 tok/s. The gap is whole blocks (Mamba included) on our CPU, against llama.cpp's
experts-only offload. A CPU IQ2_XXS expert kernel would let `--moe-expert-cache` do the same here.

## Run 6 - 2026-11-03 11:11

**Question:** with a CPU IQ2_XXS dot, does `--moe-expert-cache` let Bolt and its tower fit the 3090?

The kernel (`cpu_dot`, scalar plus AVX2 one-row, four-row and tile decode) matches dequantized rows. One thread, this
AVX2 box (`single_thread_rate`, `tile_rate`):

| type | one row | four rows | tile dot |
|---|---|---|---|
| IQ2_XXS | 1.51 GB/s (5.9 G values/s) | 4.51 GB/s | 61.9 GMAC/s |
| IQ3_S | 3.57 GB/s (8.3 G values/s) | 11.14 GB/s | 57.4 GMAC/s |

The table lookups (four grid and four sign words per 32 values) bound it, as in ggml.

**Serving still fails:** with `--moe-expert-cache 128` the planner logs "MoE experts leave the GPU" and quantizes
the KV cache to q3, then the load runs out of memory. `MoEExperts::new` takes the ungated path (Nemotron-H's
experts are up and down with ReLU^2) before it looks at the expert cache, and `OffloadedExperts` runs only gated
stacks: three projections, `indexed_mmvq_with` on the cached slots for decode and `grouped_parked_with` for staged
prefill. The CPU misses already take ungated experts (`CpuExpertGate::SquaredRelu`). With no flag the planner
predicts a resident fit that is not there (the same OOM).

**Next:** offload for ungated experts (two projections, the ungated GPU kernels over cached slots), and the resident fit
estimate.

## Run 7 - 2026-11-03 12:05

**Question:** with ungated experts offloadable, does Bolt with its tower serve from the 3090, and does the planner pick it?

`OffloadedExperts` now takes `[up, down]` with squared ReLU beside `[gate, up, down]` with a GLU: the cache runs
the ungated indexed matvec on its slots, staged prefill the ungated grouped mmq with parked routes, misses
`CpuExpertGate::SquaredRelu`. Test: IQ2_XXS ungated experts offloaded against resident, 12 decode steps and a 40-token
prompt (`offloaded_ungated_experts_match_resident_ones`).

**Explicit flags** (`--pa-context-len 8192 --moe-expert-cache N`, mmproj loaded, thinking off):

| slots per layer (of 240 - 496) | GPU | decode (warm) | prompt |
|---|---|---|---|
| 128 (KV left to fill the rest: 7.8 GB) | 23.1 GiB | 42 - 43 tok/s | 825 - 868 tok/s |
| 300 (KV 17 MB) | 22.5 GiB | 75 - 77 tok/s | 1,737 - 1,833 tok/s |

llama.cpp on the same files with `--n-cpu-moe 6`: decode 76 tok/s. Image answers as Run 5's (a 640 x 640 image: 433
prompt tokens, the same answer).

**Planner:** two sizing bugs, both from GGUF blocks (88 for Bolt) not being the loader's layers (48 hybrid blocks):
`with_source_layers` kept the first 48 blocks' bytes and dropped the other 40, and `layer_expert_bytes` lined expert
bytes up by GGUF block and gave one expert count (the first layer's 240) for all. Now a loader can say which mapped
layer each checkpoint layer folds into (`checkpoint_layer_groups`; Nemotron-H's from its layout), and the expert
counts are per layer. With them the auto plan picks 287 slots, but the post-load check then finds 23 - 30 MB short of
the 1.5 GB it holds back.

Estimate against what the card holds (desktop share taken off), `--pa-context-len 32768`:

| model | slots | planner | measured | short |
|---|---|---|---|---|
| Bolt text | 126 | 13,760 MB | ~14,094 MB | ~334 MB |
| Bolt text | 252 | 19,439 MB | ~19,822 MB | ~383 MB |
| Qwen3.6-35B-A3B IQ4_XS | 64 | 6,389 MB | ~6,566 MB | ~177 MB |

A constant per model, not per slot; the kernel libraries are ~30 MB in all, so not code. Open: where the rest goes
(the pool keeping load-time temporaries, library workspaces). Before the fold fix text-only Bolt planned resident only
because 40 layers were missing from the sum.

## Run 8 - 2026-11-03 12:32

**Question:** where do the 180 - 380 MB the offload plan misses go?

Free memory and the CUDA pool's used bytes logged through a Bolt load (`-f` picks up the adjacent mmproj, so these
"text" runs held the tower too), against the planner's estimate at 126 slots:

| stage | pool used | estimate |
|---|---|---|
| vision tower, before the decoder | 1,642 MB | ~1,720 MB (part of non-mapped) |
| embeddings / output head | +1,088 MB (BF16 head) | |
| 48 blocks | +9,514 MB | 9,511 MB |
| hybrid cache | +163 MB | |
| between the model and the KV sizing | **+1,142 MB** | **0** |

Trimming the pool frees nothing; it holds live tensors. The +1,142 MB is the recurrent state pool: 7 lanes x 40 Mamba
layers x 4.08 MB of f32 state (128 heads x 64 x 128). `NemotronHLoader::model_config` returned a plain
`ModelConfigMetadata`: no recurrent state bytes (the trait default, 0) and paged KV on all 48 blocks. The built
model's `paged_kv_config` had both right. The non-mapped estimate (~880 MB high: the output head at BF16) hid most of it.

Qwen3.6-35B's "177 MB short" was my comparison, not the planner: nvidia-smi took in its KV cache and the CUDA context.
Its loader reports its recurrent state; against the pool its estimate is ~190 MB high.

**Fix:** one `recurrent_state_bytes` / `hybrid_paged_kv_config` in granite.rs that the built model and the loader's
plan both use (the conv window counted in f32 at plan time).

**Result**, `serve -f NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` with no flags: 252 slots, 32,736-token context,
21.2 GiB, image answered as before, decode 70 - 72 tok/s warm, prompt ~1.5 K tok/s. local_ci `--lint --tests --cuda`
passes. Open: a failed load segfaults on exit after a run of CUDA_ERROR_DEINITIALIZED.
