# MoE offload planner reservations (#857)

A 21.1 GiB MoE GGUF (bartowski's Kimi-Linear-48B-A3B Q3_K_M) runs resident in llama.cpp on a 3090 (24 GB), while our
planner keeps 228 of 256 experts a layer on the GPU and pages the rest (kimi_linear_investigation Run 4).

## Run 1 - 2026-11-07 18:20

Question: what fills the gap between the weights and the card?

Read `pipeline/loading.rs` (`resolve_map_setting`, `ExpertCachePlan`) and `loaders/auto_device_map.rs`. The load
reserves, beside the weights and the planned context's cache:

- `cuda_free_floor_bytes` (512 MB on this card), when the paged cache keeps its default memory;
- `OFFLOAD_RUNTIME_RESERVE_BYTES` (1 GiB), once the planner moves experts off the GPU and sizes the cache to the
  planned context; the two log as "1536 MB on the primary device for post-load model components";
- the prompt chunk's activations (`ExpertCachePlan::prompt_chunk`), the "1600 MB ... for runtime components and
  activations" line.

The weights alone are 21,627 MiB of the 22,592 MiB free after the CUDA context, leaving 965 MiB for the KV cache, the
recurrent pool, activations and workspaces; the reserves above ask for 3.1 GiB.

llama.cpp's footprint (nvidia-smi every 100 ms over `llama-bench -fa 1 -ngl 99 -p 4096 -n 128`): 23,551 MiB peak,
1,338 MiB before it, so 22,213 MiB for the weights, its cache and its compute buffers (pp4096 2,737, tg128 171).

The planner's terms, logged temporarily from `ExpertCachePlan::spare` on load (MiB):

```
slots=None    tokens=32768 cap=21982 model=21681 (layers 21141, non-mapped 540) activations=64 cache_total=1070
slots=None    tokens=32768 cap=21982 model=21681                                activations=64 cache_total=2094
slots=Some(252) ...        cap=21982 model=21648 (layers 20904, non-mapped 744)                 cache_total=2094
...  down to slots=228
```

- The model's own size is right (21,681 against the file's 21,627).
- `cap` is the free memory less `device_memory_cap`'s max(2%, 512 MB): 21,982, below what llama.cpp runs in.
- `cache_total` at 32K tokens: 252 KV + 512 free floor + 308 recurrent pool (7 rows of 44 MB) = 1,070; resident
  misses by 834 MiB before any reserve.
- Once the planner decides to offload it adds the 1 GiB `OFFLOAD_RUNTIME_RESERVE_BYTES` to the config, and the slot
  search's own resident candidate (`slots=None`, second line) is then charged it too, though it would offload nothing.
- `plan_expert_slots` tries resident at the planned context (32K) only, then slots at 32K and at the requested
  context; never resident at the requested context.

Dead end, margins cut to 128 MB (cap reserve and free floor, temporarily): cap 22,366, resident still misses at 32K
(21,681 + 64 + 746). With `--pa-context-len 4096` it chooses resident (cache_total 337), and the 3.9K-token prompt
fails with CUDA_ERROR_OUT_OF_MEMORY at 24,031 MiB used, the whole card: a 512-token chunk's prefill needs more than
the ~300 MiB left, where llama.cpp's ubatch-512 compute buffer fits beside the same weights. So the planner's
activation estimate (64 MB) is low for this model's prefill, and its margins are what keep prompts from failing.

Next: measure our prefill's peak per chunk (which allocation dominates: logits over the 163,840 vocabulary, MoE
dispatch buffers, MLA's expanded keys and values), then decide the policy question below.

Policy (user, 2026-11-07): when resident fits at the requested context but not at the planned 32K, keep every expert
resident at the requested context rather than offload to keep 32K.

## Run 2 - 2026-11-07 20:10

Question: what dominates the prefill peak above the planner's 64 MB activation estimate?

Per-layer pool high watermarks (`/metrics` `inference_cuda_memory_pool_bytes`, reset around each layer temporarily),
Kimi Q3_K_M with experts offloaded, 512-token prompt chunks, a 3,911-token prompt:

- at rest 19,847 MiB; the first MoE layer leaves 247 MiB of expert staging that persists
- KDA and MoE layers: ~75 MiB above that; MLA layers ~158 MiB on the first chunk
- MLA layers on later chunks: 20,970 MiB, +876 MiB at a 3.9K prefix, growing with the prefix

The growth is `mla_cache_forward`'s cached-prefix path: it gathers the latent prefix and expands it to every head
(32 heads x (512 + 64) keys and 512 values per token) for an unfused SDPA. With split k_b/v_b (GGUF), q is already
absorbed into the latent space, so the prefix can attend as one head instead.

Change: the absorbed prefix runs through fattn at head dim 576 with V a view of K's first 512 columns. Dead ends on
the way:

- the probe returned false: fattn takes 576 only on its GQA path, which needs a mask and K length a multiple of 256
  (`FATTN_KQ_STRIDE`); K is now padded to 256 rows behind `-inf` mask columns
- `k_pe` is `(b, 1, seq, rope)`; flattening (0, 1) left a rank-3 tensor (narrow error on the 2nd chunk)

Result (`inference serve`, same prompts):

```
prompt 3,911: rest 19,845 MiB, high 20,252 MiB (was 20,973)
prompt 7,811: rest 19,845 MiB, high 20,252 MiB
```

The prefill peak no longer grows with the prefix; the 8-token summary check is within the run-to-run noise seen
before (first logprob -0.788, earlier runs -0.73 to -0.82).

Tiny-checkpoint coverage: the tiny Kimi only had HF's fused kv_b, which never takes the absorbed path. A split kv_b
kind (heads flattened, as safetensors store k_b/v_b) failed: the dense split projection read 3-D (GGUF) weights only,
CUDA's head-indexed gather took 2-D dense weights, and an F32 strided query hit a non-contiguous matmul. All three
fixed; the family tests that pinned the rank error now pin snapshots. Split vs fused on the CPU agree within 1.3e-3
(the latent sums in a different order).

The chunked-vs-one-chunk BF16 CUDA test diverges at greedy step 1 on master too (near-ties at bf16 on a flat random
model); in F32 the two agree within 2e-6 over 8 steps, so the test compares the first token, which reads every
chunk's cache. Scaling the latent softmax by 2 fails it.

Next: the planner (resident at the requested context; no offload reserve on the resident candidate).

## Run 3 - 2026-11-07 21:40

Question: with prefill flat, does Kimi Q3_K_M run all-resident at a 4K context, and what does the planner charge it?

Margins cut to 128 MB again (temporarily), `--pa-context-len 4096`: the planner chose resident, but the map put
layer 26 on the CPU, which disables PagedAttention, and the prompt hit CUDA_ERROR_OUT_OF_MEMORY. The map's terms
(logged temporarily): KV 682 MiB for 4,096 tokens. `get_device_layers` charged every paged layer
`num_kv_heads * k_head_dim` per token (32 x (192 + 128)), where MLA's latent holds 512 + 64: about 32 MiB.
Fixed: the map takes each layer's own `layer_kv_cache_elements_per_token` (the planner already did).

Real margins, `--pa-context-len 4096`: every layer on the GPU, 130 blocks (4,128 tokens), idle 23,464 MiB used
(1,221 of it other processes). The 3.9K prompt was refused at admission: "CUDA memory pressure prevented prompt
admission requiring 554172416 bytes". The prompt preflight (`prompt_prefill_workspace`) still sized MLA layers as
the per-head gather + SDPA the latent path replaced; `--max-num-batched-tokens 512` gave the same 554 MB, so it is
the prefix estimate, not the chunk. Fixed: MLA layers on CUDA at f16/bf16 get the latent path's workspace (gathered
latents, padded keys and mask, absorbed q, fattn output). Fused kv_b (HF checkpoints) now takes the latent prefix
path too (q absorbed through the fused weight's K rows, values through its V rows), so the estimate holds for both;
the tiny chunked test runs both kinds, and doubling the fused absorbed q fails it.

Result, resident at 4K:

```
prompt 3,911: 5.14 s with 128 generated, prefill 1,292 tok/s (offloaded: ~470), decode 64.6 tok/s
peak 23,724 MiB of 24,576 (852 MiB free at the peak)
```

Repeating one greedy request gives the same opening text but 78 or 128 tokens: the run-to-run logprob noise seen in
Run 2 flips the stop token; a cached and an uncached run differ alike.

Planner on a default load (no flags), with resident at the requested context now tried before any slots
(`plan_expert_slots` tries resident at both contexts first; `fits_resident` also counts the requested one):

```
slots=None tokens=32768 cap=22099 model=21681 act=64 cache=1070   (misses by 716)
slots=None tokens=4096  cap=22099 model=21681 act=64 cache=849    (misses by 495)
```

cache at 4K = 32 KV + 512 MB free floor + 305 recurrent pool. `cap` already holds back `device_memory_cap`'s 512 MB,
and the default cache adds the 512 MB free floor on top; the run above left 852 MiB free at its peak, so resident
fits in practice where the planner's two margins together refuse it. Charging the larger of the two rather than
their sum would leave it short by ~7 MiB at this desktop's usage. Margins are the open decision.

## Run 4 - 2026-11-07 22:30

Decision (user): charge the cap reserve and the free floor as overlapping for the resident check. (The pool trim
offered with it saves nothing: a resident Kimi already holds 7 rows = 4 sequences + 1 graph pad + 2 snapshots.)

Changes:
- For a model whose experts can offload, the default cache's free floor reserves only what exceeds
  `device_memory_cap`'s reserve while resident is tried; an offload restores the full floor (its 1 GiB runtime
  reserve was measured beside it).
- Resident at the requested context (a default load's 4,096) is tried before any slots; when it wins only there, a
  best-effort cache keeps at least that context's KV as its minimum, so the recurrent pool's KV floor is what the
  planner counted. Without that minimum the pool budgeted 5% of the device for KV and served 1 sequence (`fitted_seqs=1`)
  while the cache took 403 MB.
- The map's best-effort KV budget subtracts the weights, as its utilization branch did: on the first default resident
  load it pushed layer 26 to the CPU to make room for a 1M-token context.

Default load, no flags (`inference serve --format gguf ... Q3_K_M.gguf`):

```
Layers 0-26: cuda[0]; recurrent pool capacity=7; KV 307 MB, 39,872 tokens
idle 23,720 MiB, peak 23,980 MiB of 24,576 (3.9K and 7.8K prompts, 4 concurrent requests in another run)
```

Offload plans are unchanged by construction (the floor is restored before the offload branch). Qwen3.8-Flash-Next
UD-IQ4_XS: slots 128 of 512, 32K q3 KV (380 MB), chunk 1,024 (base reservation 1,536 MiB = floor + 1 GiB, as on
master). Qwen3.6-35B UD-IQ4_XS: resident, 262K KV (5,120 MB), as before. The LAN copy of master (1807d0c92, Nov 1)
plans Flash-Next's chunk at 2,560: older planner code, not a comparison for this branch.

Planner tests: `resident_at_the_shorter_context_beats_slots_at_the_longer`; the preflight's MLA estimate is pinned
(`mla_prefix_workspace_is_the_latent_not_per_head_keys`, 115 MiB at 512 queries over 3,911 tokens; F32 still takes
the per-head estimate, over 4x that).
