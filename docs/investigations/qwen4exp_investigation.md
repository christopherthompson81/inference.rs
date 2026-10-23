# Qwen3.8-Flash-Next (qwen4exp) investigation

Issue #558. Qwen's experimental "Qwen4" architecture, `Qwen4ExpForConditionalGeneration` / `Qwen4ExpForCausalLM`
(`model_type` `qwen4_exp`), released as Qwen3.8-Flash-Next: 180B total, 512 experts with 10 active, 262K context, a
vision tower. Reference: transformers 5.19 `models/qwen4_exp/modular_qwen4_exp.py`, llama.cpp 970b51ce7
`src/models/qwen4exp.cpp` (1,486 lines) and `conversion/qwen4exp.py`.

## Run 1 - 2026-10-22 19:01

Question: what does the architecture add over our Qwen3.5/3.8 MoE text model, will it run on this machine, and how
should the work be staged?

Read: the reference's text model, decoder layer, gated residual, n-gram embedding, PLE layer, QSA indexer and
attention; the published `config.json`; the unsloth GGUF listing; our `qwen3_5/text.rs` (3,102 lines), the hybrid
cache and the GDN gated norm.

Config (text): hidden 2560, 48 layers (GDN, full attention every 4th), 24 q heads and 2 KV heads of 256 with partial
rotary 0.25 and interleaved M-RoPE [11, 11, 10], GDN 16 key heads / 48 value heads of 128, 512 experts of 640 with a
640 shared expert, top-10; `hc_count` 4, `hc_lowrank` 320; PLE at layer 2 (`ple_layer_ids` [2], 0-based layer 1),
trigrams, 8 hash heads per n-gram, base vocabulary 20M; QSA indexer 4 heads + 1 KV head of 128, 4-token blocks,
budget 2048; one MTP layer.

What is new against Qwen3.5 MoE (everything else, the GDN and gated attention layers, the MoE and the M-RoPE, is ours):
- Hyper-connections. The residual is 4 x 2560 wide: the embedding is repeated 4 times. Each block's input is
  `mean_s(sigmoid(up(silu(down(n) / 4)))_s * n_s)` where n is the hc input RMS-normed per stream group
  (`(1 + w)` Gemma-style weight over 10240), and it writes back `h + y * 2 sigmoid(inject(n) / 4)` per stream. Two per
  layer (attention, MLP) replace `input_layernorm` and `post_attention_layernorm`; a final mixer without the inject
  replaces the output norm.
- GDN output gate: `output_gate_type` sigmoid instead of SiLU. Our gated norm hard-codes SiLU in three CUDA kernels.
- PLE (layer 1's input): 16 hashed n-gram ids per token (bigram and trigram, 8 heads each; tokens shifted within EOS
  segments, a splitmix64-seeded odd multiplier per position, xor, mod a per-head prime near 20M, plus the head's
  offset) index one embedding table of ~320M rows x 160 (about 51B parameters, so it stays in mapped host memory and
  only 16 rows per token are read). The 2560 concatenation projects to a value (2560) and one key per stream
  (10240); `gate = signed_sqrt(<norm(key_s), norm(stream_s)> / sqrt(2560))`, `out_s = sigmoid(gate_s) * value`, plus a
  depthwise conv (kernel 4, dilation 3, SiLU) over the RMS-normed `out`, with 9 positions of state per sequence. The
  n-gram ids need only the token history, which the host has.
- QSA on the full-attention layers: the indexer scores RoPE'd, RMS-normed means of each complete 4-token key block
  and keeps the top `2048 / 4` blocks plus the incomplete tail. With at most 512 complete blocks every block is kept,
  so up to 2,051 tokens of context QSA is exactly dense attention.
- MTP: one QSA full-attention block over the hc-wide residual. Vision: a Qwen3-VL tower.

Fit: the unsloth UD-IQ4_XS GGUF is 87.2 GiB (UD-Q3_K_XL 83.8, UD-Q4_K_XL 103.7). Experts in host memory behind a GPU
cache (#524) and the rest on the 3090 fits 125 GB of RAM and 24 GB of VRAM; the n-gram table goes to mapped host
memory like an untied embedding (#515). 270 GB free on /mnt/data.

Infrastructure that fits: each recurrent layer of the hybrid cache gets its own `RecurrentStatePool`, so PLE's conv
state can be one more recurrent entry with its own sizes and ride along with slots, snapshots and the prefix store.
The QSA indexer's per-token keys (128 per full layer) could be auxiliary paged layers (#597).

Plan (one implementation: extend the Qwen3.5 text model, not a copy):
1. Text model on CUDA without speculative decoding: config, hyper-connections, the sigmoid gate flag in the GDN
   kernels, PLE with its state, GGUF bindings for `qwen4exp`, the loader rows. Context capped at 2,051 tokens, where
   attention is exact. Parity against llama.cpp on the same GGUF; a tiny random-weight test.
2. QSA: indexer keys in the paged cache, block selection, sparse attention; lift the cap.
3. MTP (hc input) for speculative decoding.
4. The vision tower.

## Run 2 - 2026-10-22 19:48

Question: do the hyper-connections, PLE and the sigmoid GDN gate match transformers, in prefill and across decode
steps, before the real GGUF arrives (87 GiB, still downloading)?

Built: the config fields (`FlashNextConfig`), `qwen3_5/hyper.rs` (gated residual, grouped RMSNorm, final mixer), the
sigmoid gate as a flag in the GDN norm kernels (the deferred and fused speculative kernels stay SiLU-only and are
skipped for a sigmoid layer), `qwen3_5/ple.rs` (host-side n-gram ids and table gather, gate, dilated conv), an
auxiliary recurrent entry in the hybrid cache for PLE's conv state, a token window the text inputs processor builds
when a model asks for it (each row's chunk led by its two previous tokens, so prefix restores and chunking need no
token state), the `qwen4exp` GGUF architecture and bindings, and the `Qwen4ExpForCausalLM` loader row. Context capped
at 2,051 tokens (QSA's dense limit); speculative decoding refused.

The PLE hash constants derived the reference's way match the GGUF's metadata exactly: multipliers
[23703573157769, 20109073645365, 8052911324071], head tables the 16 consecutive primes from 20,000,003.

Command: `INFERENCE_NATIVE_DIR=$PWD/target/debug PYTHONPATH=bindings/python python3 scripts/qwen4exp_tiny_parity.py
--device cpu|cuda:0`: transformers 5.19 builds a tiny `Qwen4ExpForCausalLM` (hidden 64, 4 layers, 4 experts, hc 4,
PLE at layer 2 over a 1000-base table, sigmoid gate) with perturbed norm weights; both score 18 ids with an EOS in the
middle, then decode 12 tokens greedily.

```
CPU, f32:        mean KL -1.95e-08, top-1 1.000, max logit diff 0.0000, 12 greedy tokens identical
CUDA, ours bf16: mean KL  9.74e-08, top-1 1.000, max logit diff 0.0021, 12 greedy tokens identical
negative controls (ours only, CPU): SiLU gate -> KL 2.99e-04, top-1 0.556; PLE off -> KL 2.40e-05, top-1 0.833
```

Failures on the way: transformers saves the n-gram table as `ngram_embedding.shard_N` even in one part (now loaded
whole or by shard); the tensor crate's `rms_norm` has no CUDA backend (the grouped norm now uses inference-nn's);
our GDN kernels on CUDA take half precision only, so the CUDA run loads ours in bf16. The first decode comparison
"failed" on text only: the fixture tokenizer decodes special tokens differently, the ids were equal.

Implication: the default threshold is 1e-6 (PLE off gives 2.4e-5, so 1e-4 would have passed it). Next: engine tests
on a tiny checkpoint for chunked prefill and prefix restore (PLE's state), then the GGUF against llama.cpp.

## Run 3 - 2026-10-22 19:55

Question: does PLE's per-sequence state (token history and dilated conv state) survive step decode and a prefix
restore? Engine tests on a tiny recorded checkpoint, `crates/inference/tests/integration/qwen4exp_text_tiny.rs`:

```
cargo nextest run -p inference --features cuda -E 'test(/^qwen4exp_text_tiny::/)'
flash_next_decode_matches_prefill_scoring     PASS  (CPU f32: 12 greedy steps' logprobs vs prompt_logits of prompt+reply, < 1e-3)
flash_next_prefix_hit_restores_the_ple_state  PASS  (CUDA bf16 paged, 100-token prompt twice: cached > 0, same trace)
```

Negative controls:

```
conv state never written back after a step  -> decode test fails: step 1 decode -0.924, prefill -1.060
snapshot copy skips the PLE pool (last aux) -> prefix test fails: diverges at step 2 (265 vs 151)
```

Failures on the way: the recorded checkpoint needs `hidden_act` (transformers' saved config always carries it), the
config literal passes `json!`'s recursion limit (split in two), and the fused expert layout probe reads
`gate_up_proj`/`down_proj` shapes before any tensor, so the recording takes them up front (new
`record_checkpoint_seeded_with_shapes`).

Implication: stage 1's state handling holds on the engine paths. Next: the GGUF (download about 25 of 87 GiB at
19:55) against llama.cpp.

## Run 4 - 2026-10-22 20:00

Question: how should QSA (stage 2) fit our paged engine?

Read: transformers' `Qwen4ExpTextQSAIndexer` and attention; llama.cpp's `build_qsa_sel`/`build_attn_qsa` and its
k-pool inputs; our paged attention's fattn paths, the auxiliary paged layers (#597) and the top-k kernels.

The reference, per query at position p: the visible tokens 0..=p split into blocks of 4 from position 0; a block's
key is `rope(k_norm(mean(raw keys)), 4j)` (the mean in f32, the norm and rotation after pooling); its score is
`sum_h relu(q_h . k_j) / sqrt(128)` over the 4 indexer heads (`q` normed and rotated at p); the top `2048 / 4` blocks
plus the incomplete tail form the mask that the gated attention adds to its causal mask. HF fuses `index_qk_proj`;
llama.cpp's GGUF has `blk.N.indexer.{q_proj,k_proj,q_norm,k_norm}`. llama.cpp keeps raw | pooled keys in an indexer
cache that tracks the attention cache cell for cell, scores with its lightning-indexer op, and builds the mask over
all cells (dense cost, sparse selection).

What we have: fattn's paged kernels take an additive `(b, seq_q, n_kv)` mask, decode and prefill; the auxiliary paged
layers give extra per-layer K/V shapes that follow the block tables, so prefix caching and the store carry them; the
batched top-k kernels stop at k = 128 (QSA needs 512 of up to 65,536 blocks).

Design:
- One auxiliary paged layer per QSA layer (1 KV head of 128, model dtype): K holds each token's raw key, V at a
  block's first token holds its pooled key once the block completes. 512 B per token per layer, a quarter of the
  layer's own KV (2 x 256 x 2 heads x 2 B). Every 4-block lies in one page (page sizes are multiples of 4), so a
  shared page never holds a half-pooled block.
- Every step writes the raw keys and pools the blocks it completes, dense or not.
- While every query sees at most 2,051 tokens the layer attends densely, as now. Past that, scores over the pooled
  keys go to a new CUDA kernel that radix-selects each row's top 512 (ties to the lowest index) and writes the f16
  mask row directly (causal, the chosen blocks, the tail), and the layer calls fattn's paged kernel with it.
- The non-paged path (CPU, and Metal, which has no masked paged kernel) keeps the 2,051 cap.

## Run 5 - 2026-10-22 20:39

Question: does the QSA implementation (Run 4's design) match transformers past the budget?

Built: `qsa_select_mask` (inference-nn `cuda/qsa.rs`, kernel `qsa_select.cu`: one CUDA block per query row, a 4-pass
8-bit radix select of the top k block scores with ties to the lowest block, then the f16 mask row) with a CPU
reference in `cpu_fwd`; `PagedAttention::write_rows` and `attend_selected` (fattn's paged kernel under that mask);
the model's `qwen3_5/qsa.rs` (indexer weights, fused `index_qk_proj` or split q/k, the step plan built once per
forward from the token window, raw-key writes, pooling of completed blocks, scores and attention 128 query rows at a
time); the indexer's auxiliary paged layers in both cache configs; the context cap now only without paged CUDA.
GGUF: `blk.N.indexer.{q_proj,k_proj,q_norm,k_norm}` (norms carry llama.cpp's +1).

```
cargo nextest run -p inference-nn --features cuda -E 'test(/qsa::/)'      2 passed (CUDA kernel == reference, ties incl.)
scripts/qwen4exp_tiny_parity.py --device cuda:0 --budget 8                 (QSA selects from the 12th token)
  18 tokens: mean KL 9.08e-05, top-1 1.000, max logit diff 0.0978
  greedy: transformers [209, 95, 188, 55, 16, ...], ours [209, 95, 188, 55, 14, 67, ...]
  our tokens below transformers' best logit: 0 0 0 0 0.116 0 0 0 0 0 0 0
same, dense budget:  KL 9.22e-05, 12 tokens identical; CPU f32: KL 1.61e-08, identical
30-token sparse prefill vs transformers (default init): per-position KL <= 5.3e-07
negative control, scores negated (QSA keeps the worst blocks): KL 3.80e-03, top-1 0.833, gaps up to 0.309 -> fails
negative control, pooled key not normed: KL unchanged -> too weak (2 to 4 candidate blocks rank the same)
```

The greedy split at step 4 is a near tie (transformers' top two 1.161 and 1.045 against our ~0.1 bf16 logit error);
after it every one of our tokens is transformers' argmax given our prefix, so the parity script now checks that
teacher-forced gap (0.25 logits) on every bf16 step. The script's tiny model uses head_dim 64 (fattn has no head-dim-32
kernel) and initializer_range 0.1 (at 0.02 every distribution is near uniform: logprobs about -5.35 of ln 266 = 5.58).

Dead end: a decode-vs-prefill logprob test on the recorded random checkpoint. CUDA bf16 decode and prefill differ there
by up to 2.2 nats at one position even with QSA dropped (bisect: plain Qwen3.5 MoE <= 0.02, hyper-connections ~0.2,
PLE the outlier); CPU f32 decode == prefill, CPU bf16 too (it accumulates in f32), and CPU f32 agrees with CUDA's
greedy tokens. The recorded weights give activations around +-25 by layer 1 and bf16 error compounds from there (layer
1 already moves one position's hidden state ~3%). So the engine's sparse test compares paths that must agree exactly:
a prefix hit past the budget (pooled keys restore with their pages) against the cold run, and that the selection
changes the trace from the dense budget's. `flash_next_sparse_prefix_hit_restores_the_indexer_keys` passes.

Open: UQFF writing of the hyper-connection, PLE and indexer projections (none is in the ISQ regexes or the residual
tensors); M-RoPE positions for the pooled keys of image tokens (vision stage); masked attention still costs dense
compute (a gather of the selected cells for decode would cut it).

## Run 6 - 2026-10-22 21:00

Question: upstream mistral.rs added Qwen3.8-Flash-Next after our fork (3f2515e9b, #2462, 2026-10-01, 96 files, +15K).
What does it do differently, and what is worth taking? Read only (three parallel readers: QSA, PLE/model/GGUF, MTP).

No UQFF format change since the fork point (4b95b0f81); #2462 covers UQFF per model by promoting the hc, PLE
key/value, indexer and MTP fc projections to the promoted ISQ type, and never serializes the n-gram table (it is read
raw from the source checkpoint).

QSA. Upstream keeps raw key | cos | sin per slot and one pooled key per block in its own per-layer CUDA buffers (not
paged), so block copies, swap and the host/disk prefix tiers never carry them, and they reset when the block count
changes; ours rides the paged blocks. Same radix select, but its ties fall in atomicAdd order (nondeterministic; ReLU
sums tie at 0 often); ours takes the lowest block. Its attention is a true gather of each query's <= 2051 cells (split-K
kernel), always on in decode and graph-capturable (device-side lengths); prefill under the budget runs dense paged, over
it the scalar gather kernel, with cuTile tensor-core scoring/attention on bf16 when the JIT is there. The non-paged path
masks densely on any device, so it has no 2051 cap. Bug in ours: pooled keys rotate with `compute_text_cos_sin` of the
block start, while the reference rotates with the step's M-RoPE cos/sin, so pooled keys inside and after an image are
wrong (upstream stores each token's cos/sin at write time).

PLE. Hashing matches ours. Upstream never loads the table as a weight: raw safetensors shards or raw GGUF bytes, made
device-resident when they fit (IQ4_NL as is, bf16 quantized to Q8/Q4 on the host), else pageable mmap or a host
gather, with page cache dropped as it loads. Its conv state and an f32 token history live in a recurrent pool updated
on the device, so decode graphs run with PLE; ours rebuilds history from the token window (nothing to roll back,
survives prefix restore) but gathers on the host, so graphs are off. Upstream does gate/conv math and the `1 + w`
scales in f32; ours in bf16. Upstream bails on padded batches; ours masks them. Fused kernels we lack: hc_norm, hc_mix,
hc_combine (with the next block's grouped norm), ple_gate, ple_conv.

GGUF and loading. Bindings agree (upstream rejoins indexer q/k into `index_qk_proj`; we load either). Upstream reads
layer types from `attention.compress_ratios`; ours from `full_attention_interval`. Missing in ours: the
`Qwen4ExpForConditionalGeneration` loader class (the HF checkpoint's), the vision projector, host-RAM budgeting of the
table. Sizing bug in ours: `layer_norm_elems` divides the hc matrices by `weight_pack_factor`, though they are never
quantized (about 1 GB under Q4-class ISQ).

MTP. Upstream's new shared `speculative/{builtin_mtp,gdn_transitions,hybrid_state}.rs` mostly moves Qwen3.5 code we
already have inline. Its Flash-Next head: `fc_embedding(enorm(e))` plus `fc_hidden(hnorm(h))` per stream (no concat),
one QSA block with both hyper-connections, its own mixer; the target capture is the pre-mixer hc-wide residual. Verify
uses the transition log (built-in MTP reserves `n_predict + 1` lanes; ours returns early for built-in MTP) and a
sigmoid gate in the fused speculative recurrence; PLE conv state is stashed and rolled back; QSA heals by overwrite, as
ours does. The depth autotuner (971 lines, generic: DFlash depths [3,5,7], MTP [2,3,4,6]) is self-contained.

Implication, for this PR: fix the pooled-key rotation before vision (text-only is unaffected), the hc sizing, and add
the ConditionalGeneration class and the promoted ISQ predicates; check `compress_ratios` against the interval on the real
GGUF. Next PRs: a QSA gather for decode (dense fattn over a 262K context loses QSA's point); PLE on the device (rows
staged into a graph input, then the device table if it fits) to restore decode graphs; MTP written on our types with
upstream's forward as the spec; the autotuner ported nearly as is, separately.

## Run 7 - 2026-10-22 21:10

Question: do Run 6's small fixes hold, and does a Flash-Next UQFF round-trip?

Fixes: the GGUF synthesis now writes `layer_types` from `attention.compress_ratios` (the real file: full attention
every 4th of 48, as `full_attention_interval` says, so no change for it) and rejects mixed ratios; the loader sizes the
hc, PLE and indexer projections unpacked; the hc, PLE key/value, indexer and output-mixer projections are ISQ layers
at the promoted type; each indexer V row holds its token's cos | sin until its block pools, so pooled keys rotate with
the step's own (M-RoPE) positions. Engine tests 3/3 and both parity runs unchanged (text positions equal the stored
ones): dense CUDA KL 9.22e-05, 12 tokens identical; sparse KL 9.08e-05, tie gaps 0 except the known 0.116.

Not done: `Qwen4ExpForConditionalGeneration` would route to the multimodal Qwen3.5 loader, whose inputs processor
supplies no token window, so PLE would fail on the first forward; it belongs to the vision stage.

UQFF round trip (tiny Flash-Next, CPU f32, Q8_0 ISQ with write_uqff, then UqffTextModelBuilder reload): the reload
decodes different tokens from the ISQ load, silently. The uqff report lists every hc projection and the n-gram table
as serialized layers (f32 fallbacks for widths ISQ cannot quantize), so nothing is missing from the file. Bisect:
dropping the indexer, then PLE, then hyper-connections, then the gate still fails, and the dense Qwen3.5 tiny
checkpoint fails too (same tokens, logprobs off by up to 0.14: -0.514 vs -0.435 at step 0). The llama tiny round trip
is exact in logprobs. So the Qwen3.5 UQFF round trip is lossy on its own, before anything Flash-Next adds; the
Flash-Next test is set aside until that is found. Ruled out: the live model running the layers unquantized (its trace
and the reload's both leave the unquantized one at the same step); packed q/k/v (ISQ loads skip packing).

## Run 8 - 2026-10-22 22:01

Question: does the real GGUF (unsloth UD-IQ4_XS, 3 shards, 88 GiB) load and match llama.cpp past QSA's budget?

Tensor names in the shards match our bindings: `blk.N.indexer.{q_proj,k_proj}` BF16, norms F32; hc down/up Q8_0,
inject F32; PLE key/value Q8_0; `per_layer_token_embd` IQ4_NL [160, 320001536] (about 29 GB); experts IQ3_S gate/up
and IQ4_NL down; `output` Q6_K; no nextn (MTP) tensors in this quant.

Reference, llama.cpp 970b51ce7 build_cuda:

```
llama-perplexity -m <shard 1> -f <40K-word docs text> -c 4096 --chunks 2 -ngl 99 --n-cpu-moe 48
[1]3.1312,[2]3.2677   Final estimate: PPL = 3.2677 +/- 0.10016
99.83 s per pass, max RSS 64.0 GB, wall 3:07
```

Ours: `examples perplexity --gguf <shard 1> --file <same> --llama-cpp-ctx 4096` scores the second half of the first
4096-token window, llama.cpp's [1].

Our first runs never reached scoring at 4096:
- The perplexity example took one GGUF path (a split file needs all three; now `--gguf` takes several) and its GGUF
  path never asked for paged attention (now `--pa-cache-type`); the SDK had no expert-cache setter (now
  `with_moe_expert_cache`, example `--moe-expert-cache`).
- Even with `--moe-expert-cache 96` the automatic map put layers 0-15 on the GPU and 16-47 on the CPU, which turns
  paged attention off, so QSA's cap (2051) refused the 4096-token prompt. Sizing print: every layer about 1.2 GB, the
  whole expert stack (512 x ~2.3 MB): `layer_expert_bytes` returned None. Cause: the CPU expert path only runs
  256-value super-block types with columns a multiple of 256 (`cpu_experts::supports`), and this file's down experts
  are IQ4_NL (32-value blocks) over 640 columns, so the planner refuses expert offload for the whole model.
- At 2048 the mixed map runs dense, but its 32 CPU layers ran at about one core with the GPU idle (0%): stopped after
  6.5 min, where llama.cpp took 25 s per pass. llama.cpp at 2048: [1]4.4016, [2]3.3383.
- Peak RSS of our loads: 111-114 GB against llama.cpp's 64 GB (to look at once offload works).

Implication: real-weight parity needs IQ4_NL (32-block) experts on the CPU expert path, so the model offloads its
experts and keeps every layer and paged attention on the GPU.

## Run 9 - 2026-10-22 23:12

Question: why is every load slower than the last (67 s, then 89, 150, 294 s of weight loading), and how much of it can go?

Measured on the running server (`/proc/PID/smaps`): 56 GiB of pinned shared memory (the offloaded experts' host
stacks), 27 GiB of heap (the n-gram table, IQ4_NL, copied out of the file), 19 GiB of the file's own mapped pages:
108 GB resident on a 125 GB machine with an 88 GB file, so each load also re-read pages the copies had evicted. The
expert stacks were copied twice: the weight source into an owned buffer (`cpu_gguf_blocks`), then into pinned memory.

Changes:
- `QuantizedWeightSource::mapped_blocks`: a GGUF tensor's blocks in place in its shard mapping (`MappedBlocks`,
  holding the archive), when stored as read (little-endian, not CPU-repacked).
- The n-gram table gathers its rows from the mapping (`MappedBlocks::gather_rows`, per-row dequantization).
- Offloaded experts build `HostExpertStack::from_mapped`: register the whole shard read-only and mapped into the
  device (one registration per shard, shared, since registrations may not overlap and tensors share pages), else one
  copy straight from the mapping.

Raw: registering fails here, `CUDA_ERROR_NOT_SUPPORTED` (read-only file-backed memory on this driver), so the
experts take the one-copy path. Weight loading 43 s (05:11:03.96 to 05:11:46.95); resident 0.8 GiB anon, 57 GiB pinned,
51 GiB file pages (clean, reclaimable); `available` 53 GB where it was 26.

Implication: loads are short enough to iterate. A zero-copy expert path would need writable registration (a private
writable mapping would copy on write anyway), so one copy is the floor here.

## Run 10 - 2026-10-22 23:34

Question: with experts offloaded, does the real GGUF match llama.cpp past QSA's budget?

Expert offload first needed the CPU expert path to run the file's 32-wide types: down experts are IQ4_NL in 43 layers
and Q8_0 in 5 (gate/up IQ3_S, one layer IQ4_XS), over 640 columns, which no Q8_K super-block fills. Added: Q8 blocks
of 32 for activations and IQ4_NL / Q8_0 dots against them (`cpu_dot::vec_dot_32`), row sizes from the block size;
tests against dense weights for both mixes, and a broken IQ4_NL dot fails them. Q8_0 serves offloaded experts only:
`runs_resident` keeps resident CPU Q8_0 MoE on Candle (taking it changed in-memory ISQ Q8_0 decoding, and the UQFF
round trip caught that).

Then every layer and paged attention sat on the GPU, and prompts went NaN (`invalid CUDA top-1 output`), deterministic
per prompt, from about 10 tokens, the same at 32 and 64 cached experts. Probes (after layers, then around attention and
MoE, then the routes): layer 22's routed MoE output had 2 NaN, column 2559 (the down projection's last row) of two
tokens sharing experts; everything after was NaN. Cause: prompt staging gathers the routed experts into a staging
buffer every offloaded layer shares and views per type; the kernels' whole-tile reads past the last staged expert hit
whatever a larger fill of another type left there, and those bytes read as IQ4_NL blocks can hold an Inf/NaN scale
(NaN times the zero-padded activations is NaN). Fix: `DeviceExpertStack::zero_padding_async` after each staging.
Test `offloaded_32_wide_experts_match_the_cpu_path` poisons the staging with 0xFF: fails without the fix, passes with.
Debug aid: `maybe_log_error` now logs every internal error (a client sees "Internal server error." only).

```
examples perplexity --gguf <3 shards> --file <docs text> --pa-cache-type auto --moe-expert-cache 64
--llama-cpp-ctx 4096 (QSA sparse past 2051):  ours 3.1195   llama.cpp [1] 3.1312   (-0.37%)
--llama-cpp-ctx 2048 (dense):                 ours 4.3848   llama.cpp [1] 4.4016   (-0.38%)
wall 1:16 for load plus the 4096 window; llama.cpp 99.8 s per 4096 window
server: prompts of 9, 467, 1460 and 3365 tokens all decode
```

Open: CPU and GPU experts without pinning, as llama.cpp maps them (pageable copies from the file mapping; read-only
registration is not supported here); the n-gram table stays host-side, so decode graphs stay off with PLE.

## Run 11 - 2026-10-23 07:00

Question: decode past the budget masks QSA's selection over every cached cell, so attention costs the whole context.
Does a gather of the kept cells pay, and does it decode the same tokens?

Baseline first, master 2a9e56a4b on the real GGUF (UD-IQ4_XS, 64 cached experts, bf16 paged cache, block size 32):

```
inference bench --format gguf -m <dir> -f <3 shards> --moe-expert-cache 64 --pa-context-len 70000
    --prompt-len 512 --gen-len 32 --depth 2048,16384,65536 --iterations 1 --warmup 0
TTFT (512)            3697 ms
decode @ d2048        82.6 ms/token (12.1 tok/s)
decode @ d16384       68.1 ms/token (14.7 tok/s)
decode @ d65536       77.2 ms/token (13.0 tok/s)
prefill at depth      ~250 tok/s
```

Decode does not grow with depth: it is bound by the offloaded experts (CPU misses). Dense attention over 64K in the
12 full-attention layers reads about 12 x 64K x 2 x 2 x 256 x 2 B = 0.8 GB a token, about 1 ms at this card's
bandwidth, inside the run-to-run noise above. So the gather is for scaling (262K, a resident or smaller model,
batched decode), not for this machine's single-stream number.

Built: `qsa_select_cells` (the mask kernel's selection, written as each row's kept positions in order: the chosen
blocks' cells, then the tail; slots past the last kept one repeat the first, so a gathered row is always finite),
`qsa_gather_cells` (a warp per (row, slot, head) copies raw cache rows, 16-byte, word or byte copies by row size, so
bf16, fp8 and the quantized u8 caches all gather), and `PagedAttention::attend_gathered` (fattn's paged kernel over
the compact cache, no mask, each row's length its kept count, computed on the host from the position). Steps with at
most 16 query rows take it; each query row is its own one-token sequence, so it also serves verification steps.
Longer prefill chunks keep the mask.

```
cargo nextest run --profile cuda --features cuda ... -E 'test(/qsa/) | test(/qwen4exp/) | test(/flash_next/)'
  10 passed (new: cells == the mask's kept cells in order; CUDA cells == CPU; gather == the paged rows;
  CUDA gather == CPU for f32 rows of 3 and 4, bf16 and u8)
scripts/qwen4exp_tiny_parity.py --device cuda:0 --budget 8
  mean KL 9.08e-05, top-1 1.000, max logit diff 0.0978; teacher-forced gaps 0 except the known 0.116 near tie
  (identical to Run 5's mask path)
negative control, kept count + 2 (two duplicated cells attended):
  greedy diverges from step 3 (gaps 0.055, 0.003) -> the decode steps do run the gather
```

Same bench with the gather (one run each; every depth's prompt is past the budget, 2048 + 512 > 2051):

```
                 mask (master)   gather
decode @ d2048   82.6 ms         71.0 ms   (-14%)
decode @ d16384  68.1 ms         52.9 ms   (-22%)
decode @ d65536  77.2 ms         69.1 ms   (-11%)
TTFT (512)       3697 ms         3601 ms   (prefill keeps the mask; noise)
```

More than the 1 ms the bandwidth estimate allows: the mask path also builds an n_kv-wide f16 mask per layer and
fattn reads it, and both paths repeat each depth's content, so routing and expert misses match pair by pair. Single
runs, and depth order is non-monotonic in both columns (content changes the experts' hit rate), so take the size of
the gain loosely; the direction holds at every depth.

Implication: keep the gather for decode and verification. Prefill past the budget still pays dense compute under the
mask; a gathered prefill would need per-row caches of up to 2051 cells each (rows x 4 MB at bf16), so it would want a
kernel that reads the cell list in place instead of copying.

## Run 12 - 2026-10-23 07:42

Question: PLE gathers on the host, so decode runs without CUDA graphs. How much would graphs recover?

Scope found first: PLE is not the only blocker. QSA's step plan also builds host tensors every step (pool cells,
positions, the gather inputs), and the pool count grows every 4 tokens, so the shapes change with the context. Graphs
would need both moved to the device with fixed shapes.

```
nsys profile -t cuda,osrt --cuda-graph-trace=node inference bench ... --moe-expert-cache 64 --pa-context-len 8192
    --prompt-len 3000 --gen-len 64        (decode 67.5 ms/token; last 3.5 s of the trace, about 52 tokens)
GPU busy 45% (kernels 25%; H2D 12.9 GB, about 250 MB a token of expert fills)
191K kernel launches (about 3,700 a token), 49K copies
idle gaps: < 200 us   226 ms total  (about 4 ms a token: what graphs could take)
           0.2-5 ms 1685 ms total  (about 40 a token, one per MoE layer: 32 ms a token)
long gaps: 1186 ms start after moe_weighted_reduce_flat_kernel, 1459 ms end with a 10,240 B H2D copy
           (2560 f32: one token's CPU expert output)
```

The long gaps are the offload miss round trip: the GPU finishes its cached experts, then waits for the CPU to run the
missed ones and upload the sum, once per MoE layer. Launch overhead is about 6% of a token.

```
--moe-expert-cache 64    decode 67.5 ms/token
--moe-expert-cache 128   decode 66.6 ms/token, TTFT 17.0 s for 3000 tokens
--moe-expert-cache 160   does not fit: the automatic map put layers 43-47 on the CPU and turned paged attention off,
                         silently (13 min load, single-threaded CPU decode, GPU at 0%); killed
```

A bigger cache barely moves decode, so the round trip's fixed cost (routing read back, host function, CPU thread
wake, the upload), not the miss count, is what to measure next.

Implication: graphs (PLE and QSA on the device) would save about 4 ms of 67; set aside. Next: time the miss round
trip's parts and cut its fixed cost. Separately, an explicit expert cache that does not fit should fail or shrink,
never spill layers to the CPU.

## Run 13 - 2026-10-23 13:04

Question: what speculative decoding can Qwen3.8-Flash-Next use here, built-in MTP or a DFlash drafter, and what do
the published weights expect?

The unsloth UD-IQ4_XS GGUF has blocks 0-47 and no MTP layer. Fetched (public repositories, no token):
- unsloth's `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2.6 GB, their recommended head): one block, `blk.48`,
  with the attention, MoE and hyper-connection tensors of a full-attention layer plus `nextn.eh_proj` (5120 to
  2560), `nextn.enorm` (2560), `nextn.hnorm` (10240, the four streams) and its own mixer `nextn.hc_head_{down,up,
  norm}`; `qwen4exp.nextn_shared_target_tensors` true (it borrows the target's token embedding and output head),
  `nextn_predict_layers` 1, `block_count` 49. Their measurement: 1.67x single stream greedy on a B200 at 2 drafts,
  66% acceptance, a net loss at concurrency 8.
- incoai's `Qwen3.8-Flash-Next-DFlash2` (bf16 safetensors, 1.2 GB): a 5-layer Qwen3 drafter (hidden 2560, sliding
  window 2048, block 8, selector rank 256 / top-k 16, mask token 248077) over target layers [5, 15, 25, 35, 45] of
  48. Its capture patch: completed zero-based layer outputs averaged over the four hyper-connection streams, 5 x
  2560 = 12800 features. Their table (GB300, block 4 "limited by the current QSA implementation"): acceptance
  length 2.83-3.26 against MTP's 2.50-3.08.

Ours: DFlash 2 runs for Qwen3.8 27B; the Qwen3.5 forward pushes `xs` at each tap layer, which for Flash-Next is the
10240-wide stream state, not the 2560 stream mean the drafter was trained on.

Plan: DFlash 2 first (the drafter type exists; the capture needs the stream mean), then MTP (a head with its own
hyper-connection mixer, the GGUF `nextn` bindings and borrowed embedding and head).

## Run 14 - 2026-10-23 13:36

Question: does incoai's DFlash 2 drafter run on Flash-Next, and what breaks when a verify step's drafts are rejected?

Attaching any drafter bailed ("not supported yet for Qwen3.8-Flash-Next's hyper-connection residual"). The guard now
covers built-in MTP only. The drafter taps take the mean of the four streams at each tap layer (incoai's capture).
Then, bottom-up on the tiny Flash-Next with a tiny DFlash drafter (`flash_next_dflash_drafts_and_keeps_greedy_output`,
plain greedy decode against speculative, BF16 on CUDA, up to near ties):

1. "token window 18..21 is outside a 19-token sequence": a verify step runs drafts staged past the sequence's tokens,
   which PLE's and QSA's token window did not include. The window now appends the staged tokens.
2. Dense budget, then step 2: same token, logprob 0.37 off. With PLE off the test passed (after fixing a QSA model
   without PLE, which never requested its token window: `token_window()` is now `Option`). Cause: PLE's conv state
   was written past every drafted row. The PLE forward now stashes its conv history in multi-token decode, and
   `speculative_commit` sets each row's state to the window ending at its kept rows, after the GDN commit.
3. Sparse budget 8, step 1. Scoring both continuations with a prefill showed the speculative one wrong. The selected
   cells and the attention output of every verify row 0 matched plain decode's. Cause: without a mask the layer reads
   the attention output tokens first (`reshape` without a transpose), as paged decode leaves it, and QSA returned
   `(b, heads, s, d)`; one-token decode reads the same either way. QSA's output is transposed on maskless steps. (Also:
   a verify step's metadata gives each token its own block table row, so QSA maps its sequence rows to their last
   token row and always takes the gather there.)
4. Sparse, step 7, by 1.7 nats. A pooled key overwrote the V row at its block's first cell, which holds the cos | sin
   the block rotates with. A rejected draft completing a block destroyed it, and redoing the pooling then rotated with
   the pooled key. Pooled keys now go to the block's last cell, whose V the token's own write restores first.

Both budgets pass; the sparse transformers parity is unchanged (KL 9.08e-05).

Real GGUF, `--moe-expert-cache 64`, two greedy chat prompts of 400 tokens (a coding task, an explanation):

```
base                 13.5 / 12.9 tok/s   (an earlier, colder run: 10.4 / 11.6)
DFlash 2, depth 2    14.7 / 12.2         accept 65-71%, length 2.3-2.4
DFlash 2, depth 3    15.8 / 11.6         accept 41-95%, length 2.2-3.9
DFlash 2, depth 5    14.0 /  9.5         accept 41-47%, length 3.1-3.3
```

A verify step routes several tokens, so more experts miss the cache and run on the CPU; with offloaded experts the
drafts barely pay (+17% at best on the code prompt, a loss on prose). The outputs match plain greedy until a late
divergence (bytes 488 to 1220) where both continuations read as near ties ("compute the index" against "compute
$h(k)$"), as the tiny model's BF16 ties.

Implication: correct, but little gain while experts are offloaded. The pooled-key move changes what a stored prefix
holds in the indexer layer, so prefix entries written before it are stale for this model.
