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
