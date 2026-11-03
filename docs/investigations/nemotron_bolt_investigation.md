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
