# Nemotron 3 Super 120B-A12B investigation

Issue #565 (Nemotron-H family): the 120B-A12B MoE (nemotron_h_moe, 88 blocks: 40 Mamba, 8 attention, 40 latent MoE
with 512 experts, top-22, latent 1024) on one 24 GB RTX 3090 with its experts in host memory, then its MTP head.

## Run 1 - 2026-11-06 18:05

Question: does the published Q4_K GGUF (ggml-org/Nemotron-3-Super-120B-GGUF, Nemotron-3-Super-120B-Q4_K.gguf,
69,910,021,216 bytes, sha256 5b624f6b... matching the hub's LFS oid) load and answer here unchanged, and how does its
speed compare with llama.cpp (970b51ce7, build_cuda)?

Ours (master 23a646f18, `scripts/build_cli.sh cuda`):

```
target/debug/inference serve -p 1235 --no-ui --pa-context-len 16384 --format gguf \
  -m /mnt/data/models/nemotron-super-gguf -f Nemotron-3-Super-120B-Q4_K.gguf
```

The planner kept the experts in host memory (read in place from the file mapping, pageable; registering the mapping
with the device returned CUDA_ERROR_NOT_SUPPORTED) with 112 of 512 per layer cached on the GPU, KV at q3. The 88
blocks load as 48 layers (each Mamba/attention block with the MoE block after it). Load to listening: 5 s. 22.3 GB
VRAM in use while decoding.

Answers at temperature 0 were right and coherent: "The capital of Australia is Canberra." after a short reasoning
block; a documented iterative Fibonacci function.

| | ours | llama.cpp `-ngl 99 -fa 1` |
|---|---|---|
| prefill, ~570-token prompt | 85-116 tok/s (warm; 69 cold) | pp512 118.8 (`-ncmoe 72`), 122.1 (68), 122.9 (70) |
| decode, prose ("explain hash map collisions", 512 tokens) | 12.5-12.7 tok/s | tg128 12.0 (72), 12.3 (68), 12.3 (70) |
| decode, after a word-list prompt (128 tokens) | 13-20 tok/s | |

llama.cpp at `-ncmoe 30` and 36 failed to allocate (46.5 GB and 41.9 GB CUDA0 buffers); 66 printed no result (output
filtered to result and allocation lines, not rechecked); 68 is the lowest that ran. Decode after the word-list prompt
varies with how often the chosen experts are already cached, so the prose figure is the comparable one; llama-bench's
tg128 decodes from random tokens.

Implication: the text model needs no code. Decode is level with llama.cpp at its best fit, prefill is a little behind
(up to ~30% on the first warm requests, ~5% on the best). Next: the MTP head, which no published GGUF carries.

## Run 2 - 2026-11-06 18:40

Question: why is our prefill behind? Run 1 compared our ~570-token prompts with llama-bench's pp512, which is one
ubatch; ours were two 512-token chunks, and prefill with offloaded experts costs one staging pass of the routed experts
per chunk (a 58-token tail of 22-of-512 routing still touches ~92% of them).

Matched lengths, llama.cpp `-ncmoe 70 -fa 1 -p 500,570,1024 -n 0 -r 2`:

```
pp500   117.63 +- 1.30
pp570    77.63 +- 0.61
pp1024  119.62 +- 3.89
```

Ours (same server as Run 1, `max_tokens` 1, fresh prompts so no prefix hits):

```
tokens  tok/s  seconds
482     110.2  4.38
481     108.4  4.44
483     107.8  4.48
487     110.6  4.40
491      91.4  5.37
498      83.9  5.94
504      79.4  6.35
506      78.7
510      76.5
516     108.2
1084     93.4
1099     98.6
```

The time steps up with the tokens past 480 (15 blocks of 32): `build_prompt_chunk_plan` (pipeline/prompt_chunks.rs)
aligns chunk ends to the block size and caps a chunk at the last block boundary (`recurrent_checkpoint_boundaries`),
since a hybrid model's recurrent state can only be checkpointed between forwards and the prefix cache needs the state
there. So 504 tokens run as 480 + 24, and the 24-token tail stages ~65% of the experts again (1 - (1 - 22/512)^24).
Prompts of 481-487 tokens have tails of 1-7 tokens, which touch few experts.

Per chunk of ~480 tokens we take ~4.4 s, llama.cpp ~4.35 s per 512 (110 vs 117.6 tok/s, ~6% behind). The rest of the
Run 1 gap is the extra pass. The chunk itself is 512 because the planner only sizes prompt chunks for the Qwen3.5
family (`prompt_activation_bytes_per_token`, #616); nemotron_h keeps the default, so long prompts take a staging pass
per 512 where a larger chunk would take one.

Implication, in order of payoff for offloaded MoE prefill: (1) take the recurrent checkpoint inside the forward
(the SSD scan already computes states at its chunk boundaries; the conv state is the last d_conv-1 inputs), which
removes the split for every hybrid model; (2) give nemotron_h `prompt_activation_bytes_per_token` so the planner
picks chunks up to 8192; (3) the remaining ~6% per pass.

## Run 3 - 2026-11-06 19:10

Question: what should the planner give a 24 GB card for Super: expert slots (decode) or prompt chunk (prefill)?

Built: nemotron_h reports `prompt_activation_bytes_per_token` (the widest MoE layer's), the Qwen3.5 fit moved to
`inference_nn::loaders::moe_prompt_activation_bytes_per_token` and shared. The fit takes a latent MoE's latent size as
the routed experts' input width: 22 x (6 x 1024 + 16 x 2688) + 40 x 4096 = 1.19 MiB a token.

Chunk sweep at 64 slots (`pf_chunk.sh`, scratch: `--moe-expert-cache 64 --max-prefill-chunk-tokens N
--max-num-batched-tokens max(N, 4096) --max-seqs 1`, three fresh 5.5K-token prompts, peak nvidia-smi):

```
chunk 512    131-136 tok/s   peak 17,088 MiB (base 15,899)
chunk 1024   199-210         17,696
chunk 2048   295-324         18,880
chunk 4096   438-449         21,248   (7.7K-token prompt)
chunk 8192   CUDA OOM        23,996
```

1.16 MiB a chunk token measured against 1.19 predicted (3%). With no flags the planner then chose 96 slots (88 at the
default 4 sequences) and chunk 2048 (its 2048-token floor, MIN_PLANNED_PROMPT_CHUNK); a 15K-token prompt peaked at
22,622 MiB, the same as 5.5K, so ~1.4 GB under the planner's cap is never touched (the 1 GiB
OFFLOAD_RUNTIME_RESERVE_BYTES was fitted to Flash-Next's long prompts).

First decode figures for the planned maps (10.2-11.5 tok/s) were taken on a freshly loaded server with the expert
cache cold; the 12.5 baseline came after warm-up requests. Rerun with one warm-up request, the same flags for all
(`super_ab.sh`, scratch: `--moe-expert-cache S --max-prefill-chunk-tokens C --max-num-batched-tokens 4096`, default 4
sequences, 16K context):

```
slots  chunk  decode (prose, 512 tokens)   prefill (5.5K prompt)   peak VRAM
112    512    12.36-12.59 tok/s            159-165 tok/s           23,177 MiB
96     1536   11.36-11.73                  301-326                 22,442
88     2048   11.09-11.32                  343-354                 22,130
```

A 1024-token floor makes the planner pick 96 slots and stretch the chunk to 1536 with what is left: -7% decode for
2x prefill. 2048 costs another 3% decode for another 10% prefill. Kept MIN_PLANNED_PROMPT_CHUNK at 1024.

Implication: 96/1536 still peaks 2.1 GB below the card's 24,576 MiB; reclaiming the unused runtime reserve would
return ~8 slots. htop and nvtop during these runs looked to the user like neither the CPU nor the GPU was busy (not
measured here); if so, offloaded decode serializes GPU work, CPU misses and slot fills, a question apart from the split.

## Run 4 - 2026-11-06 20:45

Question: with the recurrent checkpoint captured inside the forward, do prompts stop splitting at the last full block,
does a prefix hit resume from the captured state exactly, and does anything regress against master?

Built: a hybrid model's prompt chunks no longer end at a checkpoint boundary when the forward can capture there
(`captures_recurrent_state_mid_chunk`: Granite/Nemotron-H Mamba layers, CUDA scheduler-visible chunks, no drafter
state kept per prefix). The step reserves a snapshot slot for each checkpoint inside a row's chunk
(`HybridCache::reserve_recurrent_snapshot`, `set_recurrent_captures`); each Mamba layer runs its scan in segments
split at those offsets and writes the state after each into its slot; the step then caches the prefix with that slot
(`snapshot_paged_recurrent_prefix(.., Some(slot))`). Attention and the MoE layers still run the whole chunk at once,
so the experts stage once.

Tiny check (`nemotron_h_tiny::a_repeated_prompt_resumes_from_a_state_captured_mid_chunk`, CUDA F32): the prompt fits
one chunk, so the repeat's prefix hit at the last full block can only come from a captured state; the repeat decodes
within 5e-3 of the first. With the capture offset moved one token past the boundary (mutation, reverted) the repeat's
greedy tokens differed from the first token on. `local_ci.sh --lint --tests --cuda --vulkan` passed.

On Super a repeated 522-token prompt resumed at 480 (prefill 3.89 s against 5.18 s) and gave the same answer; its
reasoning text diverged after ~15 words. That is BF16 rounding of the different batch shapes: before, the first
request also ran 480 + 42, so a repeat took the identical path.

Noise found on the way: a Godot game the user was running held VRAM and GPU time during the Run 3 slot A/B and the
first runs here (the planner chose a 1024 chunk where it had chosen 1536), and KDE's baloo indexer took 110% CPU
(load average 10), which cost decode ~13% at a fixed config (10.6-11.1 against 12.4) through the CPU expert misses.
The 104-vs-96 slot run from Run 3 (11.0 / 9.2-10.5) overlapped a build and is discarded. The model file was 99.9%
resident in the page cache throughout (mincore).

Clean A/B, indexer suspended, load average under 1.5 before each server (`super_full.sh`, scratch: 16K context,
default 4 sequences, one warm-up request, decode = prose x3 at 512 tokens, short = fresh prompts, long = 5.5K):

```
                       slots/chunk   decode tok/s         short prompts (tokens, s)                 5.5K prefill
master, flags          112/512       10.8-11.1 *          -                                         162-163 tok/s
master, planner        112/512       12.43-12.71          489 4.79, 500 5.40, 512 5.62, 514 3.96    161-162
branch, flags          112/512       12.53-12.72          503 3.92, 506 3.90, 516 4.12, 520 4.21    169-170
branch, planner        104/1024      11.93-12.17          491 3.94, 496 3.90, 504 3.91, 524 4.00    255-257
```

(* started right after the master build, load average 10.5; the planner row is the same map, measured clean.)

The split is gone: master's 489-512-token prompts take 4.8-5.6 s, the branch's 3.9-4.2 s. Decode at the same map is
unchanged. The planner's 1024-token floor gives 104 slots here (no game holding VRAM): -4% decode for +57% prefill on
long prompts.

## Run 5 - 2026-11-06 21:15

Question: does Super run with its MTP head, which no published GGUF carries?

Reference head: llama.cpp (970b51ce7) `convert_hf_to_gguf.py --mtp` over a scratch directory holding links to the BF16
checkpoint's shards 1 (embeddings), 49 (norm_f, lm_head, most of `mtp.*`) and 50, its configs, and an index trimmed to
those shards (1,888 of 42,683 tensors). BF16: 21 tensors, 8.0 GB, `blk.88` (`nextn.eh_proj/enorm/hnorm`, attention,
`post_attention_norm`, `nextn.shared_head_norm`, 512 latent experts, shared expert, router with `exp_probs_b`) plus
`token_embd`, `output_norm` and `output`; `nemotron_h_moe.block_count` 89, `nextn_predict_layers` 1. Q8_0: 4.3 GB.

Our archive refused the head: it repeats the model's `token_embd`, `output_norm` and `output` (so it also runs as a
standalone draft), and `merge_mtp_head` rejected any tensor already in the model. Now a repeated tensor outside the
blocks keeps the model's copy; one inside a block still fails (test `an_mtp_head_file_extends_the_model_it_is_listed_with`).

```
serve -m /mnt/data/models -f "nemotron-super-gguf/Nemotron-3-Super-120B-Q4_K.gguf;nemotron-super-mtp/<head>" \
  --mtp --mtp-n-predict 2 --pa-context-len 16384
```

BF16 head: no expert offload at all; the map put layers 0-14 on the GPU and 15-48 on the CPU, with PagedAttention
off. `layer_expert_bytes` returns `None` for the whole model when any expert stack is not a type the CPU expert path
runs, and the head's experts were BF16. Resident, those 512 BF16 experts would take ~5.6 GB of VRAM, so the head
has to be quantized like the trunk.

Q8_0 head: experts offloaded (104 slots, chunk 1024, 49 repeating layers). The decode graph precapture failed ("moe
experts offloaded ... to_cpu_storage during CUDA graph capture"), and its teardown logged 252 "CUDA error with no
caller to return it to ... CUDA_ERROR_INVALID_VALUE" lines: the burst the user saw once at the end of a web UI code
demo on the Qwen3.8 27B + DFlash 2 slot, which did not reproduce in four reruns (likely a lazy capture failing
there). `supports_cuda_decode_graphs` checks the trunk's MoE layers (`decode_is_device_only`), not the MTP head's.

Decode (prose x3, 512 tokens, after a warm-up): 12.37, 11.94, 11.74 tok/s, MTP acceptance 54-74% (2.1-2.5 tokens a
step), against 11.93-12.17 without the head at the same map (Run 4): no gain. As on the other offloaded MoE models,
a 3-token verify routes to up to ~3x the distinct experts a layer and the CPU misses grow with them, and the step
here also runs without graphs.

Implication: the head loads and drafts correctly. Fix the graph check for the head's experts and the failed-capture
log spam; MTP itself will not pay on this card while experts are offloaded.

## Run 6 - 2026-11-06 22:00

Question: the Flash-Next work (qwen4exp investigation Runs 41-52) made MTP pay with offloaded experts through a
resident, requantized head, one draft, and graphed verify steps, built for the Qwen3.5 family alone. Do they carry
over to Nemotron-H once they are generic, so a new model gets them by naming its head's experts?

Built:
- Graphs: a model reports the most query tokens a captured decode step may hold (`cuda_decode_graph_max_tokens`);
  Granite/Nemotron-H with offloaded experts (trunk or head) report the offload's split limit, 8 tokens
  (`OFFLOADED_GRAPH_MAX_TOKENS`: a larger batch reads its routes on the host). Precapture skips wider buckets and a
  lazy capture runs them eager. With 2 drafts the 4-sequence bucket was 12 tokens, its capture failed and took the
  whole precapture with it (Run 5's 252 error lines); now 8 graphs capture, no error lines, ~1,000 replays a request.
- Resident head experts, generic: loaders name a built-in head's expert stacks (`mtp_head_expert_weights`; Qwen3.5's
  three, Nemotron-H's `mtp.layers.1.mixer.experts.{up,down}_proj`), and the requant type follows each stack's row
  width (Q2_K where 256 divides it, else the 32-wide Q4_0) in place of Qwen's fixed table (`MTP_HEAD_EXPERTS`).
  Granite's head MoE loads without the expert cache.
- One draft by default with offloaded experts: `DEFAULT_MTP_N_PREDICT_OFFLOADED` moved to `inference_nn::decoder`,
  used by Qwen3.5 and Granite.
- Nemotron-H's loader sizes its head (`non_mapped_layer_sizes_in_bytes`, and in `non_mapped_size_in_bytes` as the
  sizing expects: it swaps the head's estimate for the source's bytes and then its experts for their resident size).
  Before, the head went uncounted and the planner gave the MTP load 112 slots, 8 more than without it, and the load
  ran out of memory for its 37 MB KV cache.

Head footprint, measured: at 32 slots / chunk 1024 the head adds 1.55 GB after load (13,726 against 12,176 MiB),
peaking 2.4 GB higher while it requantizes.

Not fixed (found on the way): with `--max-prefill-chunk-tokens` given, the planner still reserves the activations of
the chunk it would have picked (8192 tokens, 9.7 GB at 32 slots; ~6 GB at 64), since the flag reaches only the
scheduler. Loads with explicit slots and chunk run out of memory for nothing; the flag needs to reach the planner.

No flags but `--mtp` (planner: 88 slots, chunk 1536, n_predict 1, 21.2 GB in use), against plain (104 slots, chunk
1024), same client, one warm-up, 512 tokens, three each:

```
                      prose (hash map)       code (LRU cache)       accept
MTP, resident head    12.07 12.58 12.23      14.28 13.55 13.80      70-100%, 1.70-2.00 tokens a step
plain                 12.51 12.29 12.23      13.40 13.65 13.62
```

Break-even on prose, +2% on code: where Flash-Next ended (Run 52: tie on prose, +3% on code). The head's resident
experts cost 16 slots, and a 2-token verify still misses about twice a plain step's experts on the CPU.
