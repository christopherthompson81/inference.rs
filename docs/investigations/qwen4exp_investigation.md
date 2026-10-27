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

## Run 15 - 2026-10-23 14:27

Question: does Qwen3.8-Flash-Next's built-in MTP head run on our types, from unsloth's separate head GGUF?

Built, after upstream mistral.rs's head (3f2515e9b `qwen4_exp/mtp.rs`, vLLM's `Qwen4ExpMultiTokenPredictor`):
- `Qwen3_5MtpHead` takes either input: one `fc` over concat(norm(embedding), norm(hidden)) (Qwen3.5, and a GGUF's
  `eh_proj`, applied per stream) or Flash-Next's `fc_embedding` and `fc_hidden`. `pre_fc_norm_hidden` norms all
  four streams at once. The block is `DecoderLayer::load_hyper_qsa` (QSA attention with its own indexer in one more
  auxiliary paged layer, hyper-connections, the MoE); the head's own mixer collapses for `lm_head`; `MtpDraft`
  keeps the two outputs apart (`lm_head` reads the collapse, the next chained draft the streams).
- The target captures its pre-mixer streams for the head when it has hyper-connections.
- Draft rows run as a batch of one-row sequences on a QSA head, so QSA's per-sequence step plan applies; the prompt
  chunk catch-up plans its chunk ranges.
- GGUF: `GgufArchive::open` merges a file holding only an MTP head (`nextn` metadata, no split keys) into the model
  it is listed with. The head's block count and `nextn` keys win, any other key it repeats keeps the model's.
  Bindings: `blk.48.hc_attn/hc_ffn`, the head's indexer, and `nextn.hc_head_*` as the head's mixer, beside the
  generic `nextn` bindings (`eh_proj` as `mtp.fc`, `enorm`, `hnorm`).

```
cargo nextest run ... -E 'test(/flash_next_builtin_mtp/)'   pass, dense and sparse budget (greedy against plain,
                                                             up to near ties; the head's weights recorded at random)
cargo nextest run ... -E 'test(/an_mtp_head_file/)'         pass
real GGUF + MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf, --mtp --mtp-n-predict 2, 64 cached experts:
  coding prompt   13.0 tok/s (base 13.5)   accept 88-98%, length 2.8-3.0
  prose prompt    12.2 tok/s (base 12.9)   accept 70-82%, length 2.4-2.6
```

Acceptance is far above unsloth's 66%, but no speedup: a draft step runs the head's own 512-expert MoE as well as
the verify's wider routing, and both miss the expert cache. The outputs diverge from plain greedy at the same bytes,
with the same continuations, as both DFlash 2 runs (488 and 1220): a near tie in plain decode that verification
breaks the other way, not a fault in either drafter.

## Run 16 - 2026-10-23 15:19

Question: does Flash-Next read images, from the unsloth projector GGUF and from the HF checkpoint layout?

Probe: the UD-IQ4_XS GGUF with `--mmproj mmproj-BF16.gguf` (clip, `qwen3vl_merger`, 27 blocks of 1152, projection
2560, patch 16, no deepstack) and the published tokenizer, then an OCR crop and a small table image (fixtures).

1. The GGUF multimodal path did not know `qwen4exp`: added it to the Qwen family names (Qwen3-VL vision, Flash-Next
   text bindings, the MTP bindings). Loads.
2. Every request then failed "QSA reads the batch's token window": the multimodal inputs carried no window, and
   `MultimodalModel` had no `token_window()`. Added it (the Qwen3.5 VL model delegates to its text model, as it now
   does for packed prefill and decode graphs, both of which PLE rules out). The media inputs processor builds the
   window when the model asks for one, beside the preprocessor config.
3. Then decode failed "token window 113..51 is outside a 114-token sequence": a completion's `position_ids` carry
   the M-RoPE delta (51 = 113 less the image's position savings), not token indices. A completion's chunk runs to
   the sequence's end plus its staged drafts, so the window takes that end; prompt positions are plain.

```
ocr:   The quick brown fox jumps 42 times.
table: The table presents data on two types of fruit, apples and lemons, categorized by their respective colors
       (red for apples and yellow for lemons) and the count of each (12 apples and 7 lemons).
```

Image tokens and PLE: llama.cpp feeds an image as an embedding batch with no ids and stands in
`qwen4exp.ple.image_token_id` (248056) for each image position, which it says is what the reference hashes. Ours
hashes the sequence's own ids, and image positions there are `<|image_pad|>`, id 248056 in the tokenizer: the same.

HF safetensors: the published index has `model.language_model.*` (the Flash-Next text tensors) and `model.visual.*`
(Qwen3-VL blocks and merger), the Qwen3.5-MoE VL layout. Added a `Qwen4ExpForConditionalGeneration` multimodal row
on `Qwen3_5MoeLoader`; the GGUF family now reports it too.

```
cargo nextest run -p inference [--features cuda] -E 'test(/flash_next_text_and_image/)'   pass, CPU and CUDA
```

The tiny checkpoint is the Qwen3.5-MoE VL fixture with the Flash-Next text fields. Its random text decode has a
near tie at step 2 that BF16 on the GPU breaks the other way (logprob -0.95 against -0.98 pinned), so the GPU run
pins only the image decode.

## Run 17 - 2026-10-23 15:40

Question: what keeps Flash-Next decode out of CUDA graphs, and what fixed-shape form can each part take?

Read: `qwen3_5/{ple,qsa}.rs`, `pipeline/cuda_graph.rs`, `inference-nn/src/cuda/{graph_capture,qsa}.rs`,
`moe/experts/offload.rs`. A capture freezes every host-to-device copy into a constant
(`enable_cuda_graph_htod_cache`), and core refreshes only its own persistent buffers (ids, slot mappings, block
tables, context lens, rope positions, recurrent state slots) before a replay. Graphs are off when the model has PLE
(`SUPPORTS_CUDA_DECODE_GRAPHS && !has_ple()`); a QSA model without PLE would capture QSA's host plan (none exists).

- PLE: the n-gram table is `per_layer_token_embd` IQ4_NL `[160, 320001536]`, 27,466 MiB, so it stays on the host.
  Each decode token hashes its last 3 tokens into 16 rows. The step needs the token window, which only the host
  has.
- QSA, host-built per step: pooling only on steps that complete a 4-cell block, pool count growing with the
  context, dense attention until the 2048 budget, and the cells of each from the host rows and block table.

Fixed shapes:
- QSA decode plan on the device, one small kernel from the block table and context lens: pool the block holding
  each row's position on every step (a not-yet-complete block writes its future last cell, which that token's own
  write replaces before any read), pools sized from the table width (graph tables are bucketed, and the selection
  kernels only look at `(position + 1) / ratio` blocks), kept counts by the gather plan's formula. A bucket whose
  table capacity is within the dense length stays dense; past it, selection with the gather (which keeps every
  cell while `visible <= k`). Eager decode takes the same plan, so decode has one path.
- PLE decode: core stages a `[rows, ngram - 1 + 1]` device token window (missing tokens as the model's fill token)
  into a graph buffer like the rope positions; the layer copies it into mapped host memory, a host function on a
  side stream hashes and gathers the rows, and they come back to the device. This is #525's offload pattern, which
  records into a graph as it runs eagerly.

Prior estimate (Run 12): graphs save ~4 ms of 67 ms per token here; the expert misses (~32 ms) stay.

Plan: (1) QSA plan kernel with a CPU twin and a parity test against the host plan, (2) device token window + PLE
host-function decode with a parity test, (3) graphs on for plain decode, graph against eager on the tiny
checkpoint, (4) the real model.

## Run 18 - 2026-10-23 16:32

Question: with Run 17's plan built, do Flash-Next decode steps capture and replay, and do they decode as eager steps?

Built:
- `qsa_decode_plan` (kernel plus CPU twin, `inference-nn/src/cuda/qsa.rs`): one u32 row per decode row, the current
  block's cells, the position, the kept count and the pool cells. `QsaStep::decode` takes it whenever a step's rows are
  one token and the metadata has device context lens; pools are `table width * block size / 4`. The text model's
  decode step no longer needs the host window for QSA.
- Device token window: `TokenWindow::decode_rows` lays each one-token row out as `[real preceding, tokens...]`;
  `ModelForwardContext::decode_token_window` carries it. Core builds it for a graph step, pads it like the other inputs,
  stages it per replay through the graph's host staging, and gives the warmup and the captured forward the device copy.
  A window model graphs only one-token steps, and skips the one-token lookahead continuation (its token is on the
  device only, and the window comes from the host).
- PLE decode on CUDA: the window goes to mapped memory, a host function on a side stream hashes it and gathers the
  rows, they come back (`ple.rs` `host_gather`). The mapped-buffer helpers moved out of the MoE offload into
  `inference-nn/src/cuda/mapped.rs`, which both use.

Checks:
```
cuda decode plan matches the CPU twin (random tables, 5 rows)          pass
eager decode takes the device plan: 44 of 44 steps, table width 64      probe
PLE host gather on every CUDA decode step: 33 calls in the prefix test  probe
graph decode against prefill scoring, BF16, 24 steps
  graph and eager: identical per-step differences to 4 decimals at both budgets
  master: the same differences (up to 0.39 dense, 4.57 sparse at step 4)
```
Decode against prefill is the wrong check for this checkpoint: master shows the same BF16 gaps (the sparse one a top-k
pick that flips on near-tied block scores). The graph test compares against pinned eager traces instead: token prompts
of 18, 6, 2, 30 and 11 tokens, rounds of 1, 3 and 5 concurrent requests, 24 steps, sparse budget.

Failures on the way:
1. Every step skipped graphs as `model_unsupported`: QSA's auxiliary indexer layer (1 kv head) fails the fattn paged
   shape check in `cuda_decode_graph_supported_for_model`, though no attention kernel reads it. Fix:
   `AuxiliaryPagedLayers::attended` (false for QSA's) and `ModelConfigLike::layer_is_attended`, which the check
   filters on.
2. Precapture at load then hit `CUDA_ERROR_ILLEGAL_ADDRESS`. memcheck pointed at the indexer cache reads, but flags
   every access to that cache as "potentially before allocated", eager runs included (noise). A device-wide
   synchronize before precapture changed nothing. Bisecting: precapture of bucket 1 alone passes, bucket 2 (the first
   padded bucket) faults. Padded rows carry `GDN_PAD_SLOT` (`u32::MAX`) as their recurrent slot; the GDN kernels skip
   it, but PLE's conv state goes through the pool's plain `index_select` gather and the indexed row copy, which read
   and write row `u32::MAX`. (Inline PLE had looked like a fix only because it rejects the device window and ends the
   precapture early.) Fix: the pool's gathers clamp a pad row to the last slot, and the row copy skips pad rows.

```
cargo nextest run -p inference --features cuda -E 'test(/flash_next_decode_.*cuda_graphs/)'   2 pass
  (graphs: captures and replays, no eager fallback; both decode the pinned traces up to BF16 ties)
qwen4exp_text_tiny, qwen_vl_tiny, qwen3_5_text_tiny, qwen3_5_mtp on CUDA                     44 pass
inference-nn indexed_copy, hybrid_cache, qsa, offload, forward on CUDA                        54 pass
```

## Run 19 - 2026-10-23 16:51

Question: what do graphs give on real weights, the 4-layer real-width checkpoint (resident) and the UD-IQ4_XS model
(experts offloaded)?

First real-model run: graphs on 87 ms per token, graphs off a device-side assert in `index_select`
(`ids[id_i] < src_dim_size`), reproduced in seconds on the real-width checkpoint. A probe of the decode plan's inputs:
`context_lens [0]`, rows `[start 0, len 1]`: a one-token prompt is a decode-shaped step whose context lens leave out
its own token, so the plan's position wrapped to `u32::MAX` and its cells were garbage (2,728,887,231 in an
8,192-row cache). The plan now runs only on completion steps (`!is_first_prompt_chunk && num_cached_tokens.is_none()`,
the graph path's own prompt test). The graph test's 2-token prompt is now 1 token; without the gate both graph tests
fail, with it both pass.

That first graphs-on run had also never exited (it logged expert-fill `CUDA_ERROR_NOT_PERMITTED` errors at its end and
held the GPU), so every real-model number taken while it lived is void. Clean runs exit normally with no fill errors.

```
inference bench -m widegen (4 layers, 2560 wide, BF16, resident) --prompt-len 512 --gen-len 64, alternating
  graphs   3.26 / 3.25 ms per token
  eager    6.25 / 6.26 ms per token
inference bench --format gguf (UD-IQ4_XS) --moe-expert-cache 64 --pa-context-len 8192 --prompt-len 512 --gen-len 64
  --iterations 3, three alternating rounds (master is the LAN copy of aa567b860)
              round 1   round 2   round 3   mean
  master      81.90     73.10     73.26     76.1 ms
  eager       74.59     75.47     79.37     76.5 ms
  graphs      71.11     68.22     76.86     72.1 ms
  GPU 56-59 C, throttle reasons 0x400 / 0x600 throughout
```

Graphs halve a resident step and take about 4 ms (5%) off an offloaded one, as Run 12 estimated: the miss round trip
per MoE layer still sets the offloaded step. Eager decode, which now always takes the host-function gather and the
device plan, is within noise of master.

## Run 20 - 2026-10-23 17:33

Question: what does basalt (an MIT engine built only for this model, Blackwell, experts in RAM) do that we don't,
and what does it suggest for one 3090 with offloaded experts?

Read only, a survey of its docs and kernels. Its closest setup to ours, from its community table: one RTX 5080 16 GB
with UD-Q4_K_XL (72 GiB of experts, most of them in RAM) decodes at 77.5 tok/s with MTP and prefills 16K at
2835 tok/s; we decode at about 14 tok/s. Its quality runs put about 0.8% of routed experts on the CPU, so its
headline numbers (325 tok/s, 5090 plus a second card) are not ours to compare.

What it does in the miss path that we don't:
- No host function and no per-token uploads. Per layer, a route kernel sorts the routed experts into tiers from
  device-side residency tables, writes the CPU's expert ids and FFN input into mapped pinned memory and raises a
  doorbell word; a host thread spins on it (`_mm_pause`, a `cudaStreamQuery` every 2 ms to keep the driver
  submitting), the pool writes result rows into mapped memory and raises a flag, a GPU kernel spins on that flag
  while the GPU's own experts run, and the combine reads the CPU rows from mapped memory. It once streamed missed
  experts over PCIe for a share of them and dropped that for the CPU pool.
- Residency by a routing-frequency profile ranked over a trace (shipped with the model), never evicting on a miss;
  slow adaptive rounds swap up to 96 of the most-routed missing experts in for the least-routed resident ones (counts
  decayed 0.85 a round), seeded at request start from the prompt's routing. The RAM copy is pinned
  (`cudaHostRegister` on a buffer filled with O_DIRECT reads).
- CPU pool: one worker per physical core less one, spinning 20 ms before parking, lock-free claims; every expert
  split by rows over all workers (gate/up rows, quantize, down rows), so a few experts per layer still stream at all
  cores' bandwidth. Verify windows decode each 32-value weight chunk once for every token. Idle workers prefetch the
  likeliest CPU expert of the next layer into cache from an int8 copy of its router.
- Speculation sized to the CPU share: a window costs the union of its tokens' CPU experts; the draft stops when the
  product of acceptance estimates drops below a threshold that rises with the CPU's expert bytes. The MTP drafter's
  512 experts are requantized to Q2_0 and all resident (708 MB), so drafting never reaches the CPU.
- Prefill: dense projections always on the GPU, only experts split; chunks up to 16K; under 6144 tokens it stages
  the routed experts, above that it streams every non-resident expert through a pinned ring.
- PLE: rows read from the SSD with unbuffered 4 KiB reads behind a ~95 MB row cache, issued once the token id is
  known and gathered before layer 1 (a keep-alive read holds the SSD out of its low-power state).
- Fusions: a whole-layer kernel lost (288 against 325 tok/s with separate kernels); what stayed is the GDN middle
  (qkv+z, recurrence, out) in one persistent kernel, the hc read plus router, QSA's KV post-ops, PLE post-ops. Their
  fused kernels use TMA and PDL (sm_90+).

Against our measurements: Run 12 found the per-layer miss round trip at ~32 ms a token, ~250 MB a token of expert
fills over PCIe 3 (about 20 ms of DMA at ~12 GB/s), and 64 against 128 slots barely different. That fits basalt's
choice: the fills and the round trip's fixed cost, not the hit rate.

Plan, bottom-up: (1) measure what the fills cost by freezing the cache after a warm-up; (2) a frequency-ranked
placement with rare swaps; (3) the doorbell miss path and a row-split spinning CPU pool; (4) dense prefill on the
GPU. Fused hc/PLE kernels drop below these.

## Run 21 - 2026-10-23 18:09

Question: where does an offloaded Flash-Next decode step's time go, after Run 20's survey pointed at the miss path?

Bottom-up probes in the offload's host function and prompt staging (removed after):
```
freeze the cache after 64 steps (no fills): misses 26% -> 88% of picks, 61-74 -> 98-106 ms a token
CPU time per missed expert in the engine: ~310 us, 3.3-4.5 misses a layer, ~1.1 ms a layer, ~53 ms of a 64 ms token
```
So fills pay for themselves against a stale cache, and the CPU's misses are the step. A new ignored bench,
`flash_next_offload_misses`, reproduces it with the UD-IQ4_XS types (IQ3_S gate/up, IQ4_NL down, 640 x 2560):
```
3 misses: 251 us an expert, 9.3 GB/s, 8 cores busy
by shape (3 misses): 2048x512 IQ4_XS down 86 us (17.0 GB/s); 2560x512 103 us; 2560x768 128 us (21.3 GB/s);
                     2048x640 IQ4_NL down 203 us (9.2 GB/s)
```
A 640-wide intermediate is not a multiple of 256, so the down projection takes the 32-wide family, and
`vec_dot_32` was scalar for IQ4_NL and Q8_0. AVX2 kernels (`iq4_nl_32`, `q8_0_32`: a `pshufb` codebook lookup and
the signed maddubs, as the QK_K kernels do) with a test against the generic ones:
```
3 misses: 108.7 us an expert, 21.4 GB/s (was 251 us, 9.3 GB/s); 10 misses 105.5 us, 22.1 GB/s
real model, CPU time per miss ~150 us (was ~310)
```
The user saw one CPU thread busy for seconds per request: prompt staging. Each layer's routed experts (~410 MB at
512 tokens, 96 layers' worth in a run) were copied from the pageable mapping by the driver, one thread at a time,
8.6 GB/s. They now go through a pinned ring (4 x 16 MB) that the thread pool fills while the previous chunk is on the
bus: 10.0 GB/s, eight threads busy through prefill. The 3090 links at PCIe 3 x16, so that is near the link.

```
inference bench --format gguf (UD-IQ4_XS) --moe-expert-cache 64 --pa-context-len 8192 --prompt-len 512 --gen-len 64
  --iterations 3, alternating (master = LAN copy of 909132d8a)
           round 1   round 2   round 3   mean
  master   60.56     56.42     60.53     59.2 ms a token   TTFT 2.01-2.03 s
  branch   41.61     46.04     40.22     42.6 ms a token   TTFT 1.97-2.05 s
```
Decode is 28% faster. TTFT does not move: the staging copies are bound by the link, so prefill needs fewer bytes
(Run 20's basalt notes: more experts on the CPU share, or a streaming order that overlaps attention).

## Run 22 - 2026-10-23 19:00

Question: with the CPU experts twice as fast (Run 21), where does an offloaded decode step go now, and does a bigger
cache pay this time?

A probe in the offload's host function (removed after), per layer over a 128-token decode at 64 slots:
```
host function 540-860 us (the CPU's ~3.5 misses), gap to the next layer's 390-420 us
```
48 x (gap + host function) is the 46 ms step. With no misses the step would be about 19 ms, so the misses still cost
about 27 ms; the host function's dispatch sits inside the 400 us gap, so a doorbell can only win part of that.

```
--moe-expert-cache, --pa-context-len 8192 --prompt-len 512 --gen-len 128 --iterations 2, twice each
  64 slots    45.9 / 42.1 ms   host function 525-642 us a layer
  128 slots   29.9 / 30.2 ms   236-287 us
  160 slots   31.2 / 27.9 ms   (shrinks to fit beside the context)
```
Unlike Run 12 (66.6 against 67.5 ms), doubling the cache now takes a third off: the CPU's share was the slow part.

Without `--moe-expert-cache`, the planner chose 96 slots for a 32,768-token context although the paged cache was
sized for 8,192 tokens: it planned for the larger of the asked context and its 32K default. An asked context now
replaces the default (`planned_offload_context`, with a unit test):
```
no --moe-expert-cache, --pa-context-len 8192: before 96 slots, 34.4 ms; after 128 slots, 27.7 / 29.7 ms
```

The second 64-slot run above printed its results and then hung at exit after logging expert fills failing with
`CUDA_ERROR_NOT_PERMITTED` (like Run 19's first graphs run). A guess that fills copied from pageable memory are refused
while another stream captures was wrong: a test showed such a copy succeeds during a relaxed capture, and every
capture here is relaxed. Ten clean runs of the build without the probe neither failed nor hung; both earlier hangs
ran with something abnormal in flight (Run 19's garbage plan cells, this run's probe locking and printing inside a
host function). Kept: a fill that fails now frees its slot (four failed fills used to stop the cache admitting
experts at all), and fill errors name the step (the copy or the landing callback).

## Run 23 - 2026-10-23 19:43

Question: would basalt's frequency-ranked placement cut our misses, and does MTP pay now that the CPU share is
smaller?

Routing traces (a probe in the host function writing every decode step's routed experts per layer; removed after),
three real prompts at 128 slots (code 400 tokens, prose 400, a translation and summary 88), replayed offline
(`placement_sim.py`, scratch):
```
misses per routed pick            64 slots   128 slots
LRU, 4 fills a step (ours)        0.290      0.154
static profile, other prompts     0.932      0.822
static profile, own prompt        0.265      0.095    (an oracle bound)
decayed counts 0.85 / 0.95        0.286 / 0.277  0.152 / 0.150
decayed 0.95 seeded by a profile  0.275      0.142
```
A profile taken over other prompts is barely better than random placement (64 random slots of 512 miss 0.875):
which experts run hot is set by the content, and nothing carries across prompts. Our LRU is within 0.01 of basalt's
decayed counts. Dead end for placement.

MTP at 128 slots (`--mtp --mtp-n-predict 2|3`, the unsloth head GGUF), two prompts each:
```
base             34.4 / 37.1 tok/s
MTP 2 drafts     24.9 / 22.5 tok/s   accept ~76%, 2.5 tokens a step
MTP 3 drafts     26.7 / 20.8 tok/s   accept ~65%, 2.9 tokens a step
per call (trace):  base step 1.55 misses, 239 us of CPU;  3-token verify 9.5 misses (8.1 distinct), 1,214 us;
                   the head's own MoE layer misses 6.8 of 10 (1.7 ms a draft)
fill cap 4 -> 12 / 32: verify misses 9.15 / 9.03, decode slower (16-18 tok/s, the fills crowd the bus)
```
A verify step's tokens cannot use the fills the step itself triggers, and the head's experts are mostly cold. Basalt
gains from speculation with about 0.8% of its routes on the CPU; we are at about 15%. Dead end until the CPU share
drops.

(A first round of these numbers was void: a server from the trace run outlived its `kill`, because `$!` named the
subshell of `cd ... && serve &`, and every later script's server failed to start beside it while its requests went to
the old one. The script now refuses a port another server answers.)

Where the step goes now (Run 22): about 400 us of GPU work a layer (19 ms of a 28 ms step at 128 slots) and the
misses. The GPU side is the larger part: next, its kernel profile.

Seeding the cache from the request's own prompt at turn start (the user's suggestion; basalt seeds its counts from
the prompt): traces of prefill and decode routing for three long prompts (a code review of 1,655 tokens, a guide
rewritten as an essay 1,098, a plan from notes 823; 300 tokens each), `seed_sim.py`:
```
miss rate, 128 slots                       code    essay   plan    mean
LRU carried over (ours)                    0.177   0.128   0.166   0.157
prompt-seeded, then LRU                    0.161   0.124   0.169   0.151
prompt-seeded, then decayed counts         0.158   0.119   0.162   0.147
prompt-seeded, then frozen                 0.454   0.321   0.365   0.380
oracle: own decode profile, static         0.135   0.084   0.122   0.114
(64 slots: 0.315, 0.313, 0.305, 0.587, 0.316)
```
The prompt routes differently from the reply (a cache frozen at the prompt's top experts misses 38%), and LRU
catches up within a few tokens; the best practical policy saves about 6% of misses (about 0.5 ms a token). Not built.

CPU expert bandwidth: the machine reads 23 GB/s on one thread and peaks at 35.8 GB/s on four (a standalone AVX2 sum
over 2 GiB); the miss kernel holds about 22 GB/s on eight threads for every type pair tried (IQ3_S, IQ4_XS, Q4_K gate
and up with IQ4_NL or Q8_0 down), so it is not decode-bound. Larger work items (`TILE_ELEMS` 32K/64K/128K) do not lift
it and slow a single miss (130 to 155/285/542 us). Open: what holds it at 60% of the read peak.

What holds it, by phase (one step of 3 misses, a probe in the bench): setup and quantize 6.5 us, gate/up 182 us
(4.0 MB, 22 GB/s), quantize the gated rows 4.8 us, down 137 us (2.6 MB, 19 GB/s), weighted sum 1.4 us. The
projections themselves, and single-token dots are decode-bound per thread (`single_thread_rate`): IQ3_S 3.87 GB/s,
IQ4_XS 5.28 GB/s on cache-resident rows, while four tokens at once reach 11.6 GB/s per token. `iq3_s_one` looked up
its 8 grid codes with scalar loads and `_mm256_set_epi32`; it now builds the 9-bit indices with a variable shift and
gathers them (`_mm256_i32gather_epi32`), as llama.cpp's AVX2 kernel does:
```
single_thread_rate IQ3_S: 3.87 -> 4.44 GB/s
flash_next_offload_misses, 3 misses: 116.6 / 117.0 -> 112.5 / 109.8 us an expert (two runs each)
```
About 5% off the CPU's share; the kernel still holds near 21 GB/s against the 33.5 GB/s eight threads read.

## Run 24 - 2026-10-23 20:40

Question: where do the GPU's 15 ms of a 28 ms offloaded step go, and what is worth cutting?

```
nsys profile -t cuda,osrt --cuda-graph-trace=node inference bench ... --moe-expert-cache 128 --prompt-len 512
    --gen-len 64 (63 decode tokens, 28.7 ms each)
kernel busy 15.4 ms a token (54%); the rest is the wait on the CPU's misses
Q8_0 dense GEMVs (mmvq plain)   5.1 ms   by shape: GDN qkv 10240x2560, hc up 10240x320 (96 calls ~12 us),
                                         GDN out, shared-expert down 2560x640 (48 calls ~4.3 us), QSA o_proj,
                                         hc down 320x10240 (97 calls 5.7 us, ~616 GB/s)
expert GEMVs (IQ3_S, IQ4_NL)    2.4 ms
lm_head (Q6_K)                  0.69 ms  (~760 GB/s)
~30 small elementwise launches a layer (bmul, affine, sum, sigmoid, badd, silu, casts, copies)  ~3 ms
```
The hc up projection runs at ~250 GB/s: a 320-wide Q8_0 row keeps 40 of the single-column kernel's 256 threads busy
(8 warps over 2 rows, 4 threads a 32-value block). A short-row variant (one warp a block over 4 rows) for one token
over rows up to 512 wide, with a test against MMQ (`short_row_mmvq_matches_mmq`, a row count 4 does not divide):
```
engine, per call: hc up 12.0 -> 8.4 us; shared-expert down (640 wide) 4.3 -> 5.6 us at a 1024 cutoff, so the cutoff
is 512; kernel busy 15.40 -> 15.11 ms a token
hot loop, 10240x320: 6.21 / 5.67 / 8.33 us at 4 / 8 / 16 rows a block (4 kept: a k-quant shape prefers it)
bundle +0.09 MiB
```
About 0.3 ms of a step, a quarter of the estimate.

The hyper-connection elementwise work, about 30 launches a layer (each block's read: grouped norm then a scale
multiply, the low-rank activation's divide and silu, the up projection's sigmoid, multiply and mean over streams, the
write-back weights' divide, sigmoid and doubling; and two write-backs of multiply and add), now runs in five kernels
(`inference_nn::cuda::hyper`: grouped norm, scaled silu, mix, write-back weights, write-back), each tested against
the tensor ops in F32 (`fused_ops_match_the_tensor_ops`, within 1e-5). A layer's feed-forward write-back now lands in
its residual directly (`DecoderLayerOutput::branch` is `None`). CPU layers keep the tensor ops.

The fused kernels accumulate in F32 and round once where the tensor ops rounded to BF16 between steps, so BF16
decodes move: the tiny checkpoint's GPU score against the F32 CPU reference (`flash_next_split_across_gpu_and_cpu`'s
mean logprob error) goes from 0.240 to 0.140, and the graph tests' pinned traces were re-pinned.

The first grouped norm (a block of 256 threads per group, reading the slice twice) took 5.8 us a call, more than the
rms norm plus multiply it replaced (2.0 + 2.3 us): one token gives it only four blocks. It now keeps a thread's
values in registers with 1,024 threads a block.

```
nsys, 63 decode tokens: kernel busy 15.11 -> 13.73 ms a token, 3,544 -> 2,768 launches a token
bench, 128 slots, fused against unfused in one binary, seven alternating rounds (ms a token):
  unfused  36.32 32.23 36.26 30.05 37.20 30.90 32.35
  fused    48.13 33.56 35.73 32.51 32.16 31.32 31.59
```
The wall time spreads by several ms between runs (the CPU's misses), so the 1.4 ms of kernel time is not resolved
end to end; rounds 4-7 average 31.9 fused against 32.6 unfused.

## Run 25 - 2026-10-23 21:30

Question: does our Flash-Next (with today's fused kernels and the expert offload) agree with an independent
implementation on the same weights?

No Q8_0 base exists locally, so the reference is llama.cpp on the same UD-IQ4_XS GGUF. The perplexity example gained
`--kld-base`: it reads a `llama-perplexity --kl-divergence-base` file, scores the same chunks' second halves with full
logits, and computes KL(base || ours) as llama-perplexity does (base terms at or below -16 skipped), checking on the
first row that a logits row predicts the token after it.
```
llama-perplexity (970b51ce7) -c 512 --chunks 20 -ngl 99 --n-cpu-moe 48 --kl-divergence-base   PPL 2.3442
perplexity --gguf <3 shards> --moe-expert-cache 128 --pa-cache-type auto --kld-base <base>  (4.7 s a chunk)
  5100 positions over 20 chunks of 512
  PPL            2.3340 (base 2.3406, from the stored log-probs)
  KLD mean       0.03469
  KLD median     0.00292
  KLD p99        0.49934
  same top-1     94.65%
```
For scale (basalt's quality notes): two implementations of this model agree to KLD 0.023-0.024 once both read the
hyper-connections in F32, and a stock llama.cpp base adds 0.01-0.02 of its own (its Q8_0 hc read quantizes the
activations; basalt's Q8_0 against it, 0.0456 and 95.6%). Ours at 0.0347 and 94.65% sits in that range: no sign of a
numerics fault in today's kernels. Whether our own hc read (also q8_1 activations through MMVQ) carries the same
noise needs an F32 reference, which this does not give.

## Run 26 - 2026-10-23 22:33

Question: does MTP pay at 128 slots once the head's experts stop missing (basalt keeps its head's experts resident at
2 bits), and does the draft depth matter?

First, Run 23's MTP rows were not at 128 slots: with the head file loaded the planner logged `requested=128 slots=64`
for every MTP server, while the base ran at 128. Equal-slot runs at 64 (`spec_measure.sh`, two prompts, tok/s):
```
base                                  27.3 / 25.4
MTP 2, head experts offloaded (Q8_0)  26.4 / 23.2
MTP 2, head experts resident (Q8_0)   24.0 / 22.5     (a throwaway switch loading the head's MoE without the cache)
MTP 1, head experts resident          25.3 / 23.5     accept ~80%, 1.8 tokens a step
```
Resident head experts change nothing measurable: the head's misses (about 1 ms a draft) are small beside the verify.

Why the planner shrank the cache (probes in `ExpertCachePlan::fits`, removed after): the head counted 4,685 MiB more
than it costs. Two faults, both fixed:
- The loaders size the head inside `non_mapped_size_in_bytes` at the conservative pack factor with all 512 experts
  resident (4,960 MiB); the source's own bytes for `blk.48` were dropped (the resident-bytes step truncates layers
  past the loader's count) and its experts never left the budget. Loaders now report the head separately
  (`non_mapped_layer_sizes_in_bytes`); the planner swaps in the source's bytes and offloads its experts like a mapped
  layer's (slots, plus the staging max).
- `layer_expert_bytes` counted a stack once per binding; the head's stacks are bound twice, so the head's experts
  read as 5,100 MiB and the staging estimate followed. It now dedupes source tensors as `layer_resident_bytes` does.

After: the Q8_0 head needs its full 2,550 MiB of experts staged (the largest layer sets the staging buffer, against
1,537 MiB for the model's largest), so it gets 96 slots; a head with only its experts requantized (`llama-quantize
--allow-requantize --tensor-type ffn_gate_exps=q2_k --tensor-type ffn_up_exps=q2_k --tensor-type
ffn_down_exps=iq4_nl`, 1,072 MiB, 975 of it experts) fits 128 (22,065 of 22,374 MiB planned).
```
128 slots                              tok/s          accept
base                                   36.7 / 39.3
MTP 2, Q2_K/IQ4_NL head                36.9 / 36.4    76.4 / 74.7%  (Q8_0 head: ~75%, so the requant costs nothing)
MTP 1, Q2_K/IQ4_NL head                33.9 / 34.7    80-82%
MTP 2, Q8_0 head (planner: 96 slots)   31.5 / 29.6
```
MTP is break-even at best at equal slots. The verify is the cost, not the head: three tokens miss about 6x a single
step's experts (Run 23), each distinct miss costs the CPU what it costs a plain step, and at ~15% of routes on the CPU
the 2.5 tokens a step do not pay for it. A depth controller sized to the CPU share would pick no drafts here; not
built. Kept: the two planner fixes (MTP servers now get the cache they fit) and an offload-guide note on head types.

## Run 27 - 2026-10-23 23:06

Question: is QSA's masked prefill (dense compute past the 2,048 budget) what holds offloaded prefill at ~250 tok/s,
or is it the expert staging?

Run 11 already put prefill at ~250 tok/s at every depth to 64K, and Run 21 found each 512-token chunk staging about
410 MB of routed experts a layer over PCIe 3 (~2 s a chunk). If staging is the cost, a longer chunk pays it over
more tokens: a chunk of a few thousand tokens routes to nearly every expert, so its bytes stop growing with its
length. Recurrent prompt batches take `--max-prefill-chunk-tokens` (512) as their chunk even with nothing decoding.

`pf_chunk.sh` (scratch): `serve --max-prefill-chunk-tokens N --max-num-batched-tokens max(N, 4096)
--pa-context-len 8192 --max-seqs 1`, one prompt three times with a fresh nonce (no prefix hits), `max_tokens 1`:
```
128 slots, 5,243-token prompt          64 slots, 7,597-token prompt      peak VRAM (64 slots)
chunk 512    118-122 tok/s             126 tok/s                          16,159 MiB
chunk 1024   193-195                   -
chunk 2048   301-305                   333                                16,831
chunk 4096   CUDA OOM                  528-534                            17,533
chunk 8192   -                         779-800                            18,749
```
Staging it is: 6.3x from the chunk alone, and the slope says more is there (basalt runs chunks to 16K and streams
every non-resident expert above 6K). The attention mask is noise beside it, so QSA sparse prefill would not move
TTFT on this machine; dropped from the list.

Memory: about 0.34 MiB of activations a chunk token. At 128 slots the planner leaves ~0.65 GiB, which holds 2048
but not 4096; the activation reservation (48 MiB here) is the attention estimate and knows nothing of the MoE
intermediates. Next: an offloaded model's planner sizes the prefill chunk from what the slots leave, reserves its
activations, and the scheduler takes it for recurrent prompt batches unless the flag is given.

## Run 28 - 2026-10-23 23:28

Question: can the loader pick the prompt chunk for an offloaded model, from the memory its expert cache leaves,
without the OOM Run 27 hit at 4096 with 128 slots?

A second model to fit the activations against (Qwen3.6-35B-A3B UD-IQ4_XS, 64 slots, the same 7,597-token prompt):
```
chunk 512    393 tok/s    8,243 MiB peak
chunk 2048   1,104        8,689
chunk 4096   1,794        9,201
```
0.25-0.29 MiB a chunk token here, 0.30-0.44 on Flash-Next (the first step includes allocator growth). Both fit
`topk x (6 x hidden + 16 x moe_intermediate) + 40 x hidden` bytes a token within about 5% of the larger steps:
0.234 MiB (35B-A3B: hidden 2048, top 8, 512 wide) and 0.342 MiB (Flash-Next: 2560, top 10, 640).

Built: `prompt_activation_bytes_per_token` on the device-mapped loaders (the Qwen3.5-family MoE loaders implement it
with that fit); the expert-cache planner, once the slots are settled, takes the longest chunk in 512s up to 8192
whose activations fit beside them and the planned context, and reserves those bytes before the KV cache is sized;
the chunk reaches the scheduler through `GeneralMetadata::prompt_chunk_tokens`. `--max-prefill-chunk-tokens` and
`--max-num-batched-tokens` are now unset by default (512 and 4096, or the chunk when longer), so a given flag wins.

```
pf_chunk.sh auto (no chunk flags), 7,597-token prompt
--moe-expert-cache 128 --pa-context-len 8192   chunk=1536   282-287 tok/s   (512: 126)   decode 35.9 / 37.8 tok/s
--moe-expert-cache 64  --pa-context-len 8192   chunk=8192   774-787 tok/s   (512: 126)
no flags (planner: 96 slots at the 32K floor)  chunk=3584   KV 3,121 MB
Qwen3.6-35B-A3B, no flags                      resident, no chunk planned
Qwen3.6-35B-A3B, --max-seq-len 262144          208 slots, no room past the context: 512
```
The policy keeps the slots first and gives the chunk what is left; a slot step traded for a long chunk would be the
other way to spend it (64 slots and 8192 prefill 6x faster for 30% slower decode on Flash-Next), left to the flags.

## Run 29 - 2026-10-24 07:11

Question: what holds the CPU miss kernel near 21 GB/s against the 33.5 GB/s eight threads read (Run 23), and how
much of it can a decode step get back?

`flash_next_offload_misses` (UD-IQ4_XS types, 640 x 2560), and one-thread rates on cache-resident rows:
```
baseline, 3 misses          338 us a step, 20.7 GB/s, 8 cores
one thread, one token       IQ3_S 4.34 GB/s, IQ4_XS 5.13, IQ4_KSS 9.20, IQ4_NL (32-wide) 4.84, Q8_0 (32-wide) 9.36
CANDLE_NUM_THREADS 12 / 16  3 misses 23.6 / 22.8 GB/s, 10 misses 26.4 / 25.6
software prefetch, 4 rows ahead of the dotted row      19.0 GB/s (worse; reverted)
items per grab 16 / 8 / 4   within noise (22.4-23.4 GB/s at 3 misses)
```
Eight threads at about 4.4 GB/s each is the compute ceiling the IQ3_S gate and up projections (two thirds of a
miss's bytes) can reach, and compute and memory stalls do not overlap fully: the kernel is compute-bound per thread,
which is why hyper-threads lift it. They do not lift the engine: with the GPU's own host threads beside the pool,
16 threads decode at 12 tok/s (37 at 8) and 12 threads is within noise of 8.

Inside the IQ3_S dot (a probe replacing the grid gather with its indices: 4.34 -> 8.34 GB/s), the gather is half the
time. Broadcasting the qh byte from memory and loading the 2s + 1 scale from a table (four fewer port 5 ops a
sub-block) changed nothing (4.41 / 4.27): the gather's loads, not the shuffle port, set it. The grid is 512
arbitrary entries, so no shuffle can stand in for it; left as is.

The 32-wide kernels (IQ4_NL down) spent most of a block converting its f16 scale: `half` reaches F16C through a call
per value. `f16c_at` converts in-line (F16C now part of the AVX2 dispatch check; every AVX2 CPU has it), and two
accumulators let consecutive blocks' multiply-adds overlap (a `[acc; 2]` indexed by `i % 2` was slower, 6.2 GB/s):
```
one thread    IQ4_NL 4.84 -> 7.51 (F16C) -> 8.1 GB/s (pairs);  Q8_0 9.36 -> 14.7 -> 15.0
3 misses      338 -> 305-313 us a step (20.7 -> 22.4-23 GB/s);  10 misses 21.6 -> 23.9-24.6 GB/s
engine, 128 slots (spec_measure, two rounds)   34.0 / 37.5 and 36.7 / 39.4 tok/s   (Run 26 base 36.7 / 39.3)
```
About 8% off the CPU's share, which is below the engine's run-to-run spread. Q8_0 CPU layers (#602) take the same
kernel. The rest of the gap is IQ3_S's gather; the CPU miss item closes here.

## Run 30 - 2026-10-24 07:26

Question: does basalt's CPU expert path (fresh clone, 3d48edb) do anything on this machine's miss path that Run 29
left out?

Read: `src/kernels/cpu/{expert,iq_avx2,native_expert,pool}.cpp`, `src/core/verify.cpp`. Its AVX-512 path (VNNI and
VBMI, Q2_0 packs) does not apply to this CPU; its AVX2 path is the one to compare.
- IQ3_S: scalar grid loads into `set_epi32`, as llama.cpp. Run 23 measured that slower than our gather here.
- IQ4_NL down: F16C scales, one accumulator, the same decode as ours since Run 29.
- Pool: one worker per physical core, the host drains too, each expert's rows split over every worker (gate/up
  rows, the intermediate's quantization, down rows), as ours.
- Software prefetch: two lines 2 KB ahead of every QK_K block (one for a 32-wide block), inside the dot. Run 29's
  prefetch (a whole row four rows ahead, issued per row) was slower; this form was not tried.
- Next-layer warm-up: after layer l's doorbell the parked workers score layer l + 1's router (int8) on layer l's
  FFN input, rank the experts only the CPU holds, and read the likeliest one into cache until the next batch.
  No measurement of it in its docs.

What a warm-up could buy, bounded by misses whose experts are already in cache (the bench with fixed experts):
```
flash_next_offload_misses       3 misses       10 misses (23 MB, past the 16 MB L3)
cold                            306 us         960-1065 us
cache-hot                       264-266 us     893-950 us
```
14% for experts entirely in cache: the kernel is mostly compute-bound. Warming one of a layer's ~3.5 missed experts
bounds the gain near 4% of the CPU's share before mispredictions; not built.

The prefetch, A/B against the committed kernels in alternating rounds (3 misses / 10 misses, us a step):
```
none           307.3 319.2 308.7 / 971.7 1014.4 981.0
2048 ahead     283.0 291.6 282.2 / 906.0  925.3 890.6
1024 ahead     286.2 277.3 278.2 / 910.8  864.0 899.7
4096 ahead     284.2 294.9 310.2 / 906.7  958.6 960.4
```
Kept at 1024 bytes for the single-token IQ3_S, IQ4_XS, IQ4_KSS and 32-wide dots. The bench now also takes Qwen3.6-35B's
shapes (`qwen36_offload_misses`: IQ3_S gate/up, IQ4_XS down, 256 experts of 512 x 2048); three alternating rounds
against master:
```
                      3 misses              10 misses
Flash-Next   master   293 303 304           943 959 957
             branch   269 273 294           862 881 892    (-7% / -8%)
35B-A3B      master   275 260 249           685 727 699
             branch   229 228 220           682 717 654    (-14% / -3%)
```
That and Run 29's F16C are what basalt's AVX2 path has that ours lacked; the rest is either the same or slower here.

## Run 31 - 2026-10-24 07:40

Question: does batched decode work with offloaded experts, and what does it give?

`batch_measure.sh` (scratch): `serve --moe-expert-cache 128 --pa-context-len 8192 --max-seqs 4`, 1, 2 and 4
concurrent greedy requests of 300 tokens (four different prompts: code, an explanation, prose for a student, a
story):
```
streams   per stream    total
1         38.6 tok/s    38.6
2         21.6          43.2
4         10.6          42.4
Captured 4 CUDA decode graphs through batch bucket 4; no errors, fill failures or eager fallbacks in the log
```
Correct: the outputs are coherent, and a prompt's batched output matches its single-stream one for the first ~490
characters, then drifts (cache timing and the batched kernels' rounding, as the guide notes for single streams).
Throughput is flat because each stream brings its own misses: four tokens route to about four times the experts the
cache lacks, and the CPU's share grows with them, as a verify step's does (Run 23). Nothing to fix here; a larger
cache is the lever.

## Run 32 - 2026-10-24 07:53

Question: can a prefix stored on disk before #606 (QSA's pooled keys then sat in a block's first cell) still be
restored into today's build?

The store keeps entries apart by a namespace: the model identity (source, dtype, ISQ, drafter), the block size,
every paged layer's K/V shape and dtype, the recurrent state's shapes and layout, and the drafter's auxiliary
signature. #606 moved where the indexer layers keep their pooled keys without changing a shape, so a pre-#606
Flash-Next entry hashes to the same store and is restored with its pooled keys in the wrong cell.

Fixed: every namespace opens with a layout header (`layout 1`), bumped when a model changes what its cached state
means without changing its shapes; opening a store removes sibling stores under another header, which no store of
the build could reach again. Test `a_store_under_another_layout_is_removed_on_open`. The one-time cost is a cold
prefill per cached prefix, for every model, on the first start.

## Run 33 - 2026-10-24 08:51

Question: how much do the hyper-connection reads' 8-bit activations cost (basalt: 0.01-0.02 KLD), and does an F32
read pay for its cost?

The reads are three linears a block (`input_mix_weight_down` 10240 -> 320 and `_up` 320 -> 10240 in Q8_0, the inject
10240 -> 4 in F32); MMVQ and MMQ quantize their activations to q8_1. The perplexity example gained `--kld-base-out`
(write our own log-probs as a base over the same chunks). A temporary switch dequantized the Q8_0 weights to F32 at
load (2.5 GB more VRAM, so every run at 64 slots):
```
current read, against our own base (written by the same build)   KLD 0.00000   (prompt scoring is deterministic)
F32 read, against the current read's base                        KLD 0.04648   PPL 2.3340 -> 2.3500
F32 read, against llama.cpp (q8_1 hc read)                       KLD 0.04658   (current read: 0.03469)
```
Against llama.cpp's base the F32 read looked worse, but that base quantizes the same activations, so it cannot rank
the two. A copy of the GGUF with only the 194 hc down/up tensors dequantized to F32 (exact; llama.cpp then reads them
in float), a new base from it (`llama-perplexity` 970b51ce7, same 20 chunks of 512; PPL 2.3425 against 2.3442):
```
against llama.cpp with F32 hc reads       KLD mean   median    p99      same top-1
llama.cpp, q8_1 hc read                   0.04019    0.00340   0.5764   94.47%
ours, q8_1 hc read                        0.04297    0.00367   0.5813   93.71%
ours, F32 hc read                         0.03258    0.00274   0.4244   95.10%
```
The 8-bit activations cost about 0.04 KLD in either engine (more than basalt's figure), and the F32 read moves ours
0.010 closer to the float reference.

Built without the VRAM: `unquantized_activation_forward` (inference-quant) for a Q8_0 weight on CUDA runs a new matvec
(`launch_mmvq_gguf_q8_0_{bf16,f32}_exact`: the Q8_0 blocks against BF16/F32 activations, F32 sums) for up to 8 rows,
and a dequantized F32 copy for longer batches (0.02 ms to dequantize plus 0.2-0.4 ms of F32 GEMM a call at 512 rows,
about 60 ms of a 5 s chunk). Flash-Next's hc reads take it. First version (a block of threads a row, a byte a lane)
lost to the q8_1 path; a warp a row for rows up to 1024 wide and two values a lane (a `char2` weight load, a paired
activation load) win. nsys, kernel time a call:
```
                      q8_1 path (quantize + mmvq)    exact
320 x 10240, 1 row    1.41 + 4.78 us                 5.67 us
10240 x 320, 1 row    1.41 + 5.49 us                 5.14 us
3 rows                1.41 + ~9.3 us                 8.85 / 7.85 us
```
End to end: KLD 0.03292 against the F32-hc base (PPL 2.3421), decode at 128 slots 36.2 / 39.0 and 38.9 / 41.4 tok/s
(before 34-39). Test `exact_q8_0_matvec_matches_the_dequantized_weight`.

## Run 34 - 2026-10-24 09:22

Question: should Flash-Next's attention KV live in host memory (basalt's `kv-resident`), so long contexts stop
taking expert slots?

What a QSA decode step reads (code map): the indexer scores every block's pooled key (one pass over the indexer
layer, 64 B a context token a layer), then the main K/V of at most 2,051 selected cells a layer, copied into a compact
cache before fattn. The indexer layers are 20% of the KV bytes (12 x 512 of 12 x 2,560 B a token in BF16) and would
stay on the GPU; the main K/V is the candidate. Selection never leaves the device, and decode is graphed.

How stable is the selection? A probe wrote each layer's selected blocks every decode step (graphs off), two 26K-token
prompts (a code review, a guide rewritten as an essay), 200 greedy tokens, replayed through an LRU block cache
(`qsa_sim.py`, scratch):
```
                         code (6,600 blocks a layer)    essay
consecutive overlap      0.697 (min 0.216)               0.695 (min 0.265)
LRU 512 blocks a layer   miss 0.650, 333 a step          0.657, 337
LRU 768                  0.214, 110                      0.218, 112
LRU 1024                 0.134,  69                      0.146,  75
LRU 1536                 0.077,  40                      0.086,  44
```
Costs on this machine: zero-copy reads of every selected cell are 25-50 MB a token over PCIe 3 (q8 to BF16, about
2-4 ms) on a link that already carries the expert fills (Run 12: ~250 MB a token). A GPU block cache needs 15-23% of
the main KV resident to keep misses near 40-70 blocks a step, so it frees about 65% of the KV's VRAM (2.5 GB at a 128K
BF16 context, ~23 expert slots), for a device-side cache inside graphs plus host paths for prefill's masked attention
and the dense short-context path.

The quantized cache already gets most of that VRAM with no PCIe traffic, and it works with QSA (the cell gather copies
raw words of any cache dtype):
```
26K-token code prompt, 64 slots, --pa-context-len 40000
auto (BF16)   1,200 MB for 40K tokens (30 KB a token)
q8              733 MB (18.3 KB)    coherent, decode 26.7 tok/s
q4              518 MB (13 KB)      coherent, decode 26.4 tok/s; the indexer layers (BF16) are now ~46% of it
```
Not built. What remains of the idea: the offload planner could pick a quantized cache when its savings buy expert
slots (needs a decode-side quality check), the indexer layers could take a quantized cache (~0.8 GB at 262K with q4
main KV, about 7 slots), and a KV that grows into the expert cache at runtime (basalt's elastic KV) is still open.

## Run 35 - 2026-10-24 09:38

Question: does a quantized KV cache cost Flash-Next quality as decoding reads it? (Qwen3.8 27B takes q3 with no
measurable loss; basalt ships int8 and publishes no numbers for its q4_0 or k8v4.)

Prompt scoring never reads the cache, so the measure is decode-time teacher forcing (kv_cache_quant_investigation.md
Run 6): each scored position is a one-token `/v1/completions` of the passage so far, and prefix caching serves the
earlier tokens from the cache under test (hit rate 97.9-98.0% in every run). Six passages of a code-and-docs text, each
a 3,072-token prefix (past QSA's 2,051-token dense limit, so attention reads the selected quantized cells) and 48
scored positions; 64 slots, `--pa-context-len 16384`, one server per type (`kv_quality.sh`, scratch):
```
cache    ppl      top-1    vs auto   KV a token (main + indexer, 12 layers)
auto     2.3820   75.7%              30.0 KB
q8       2.3843   75.0%    +0.1%     ~18 KB
q4       2.4156   74.3%    +1.4%     ~13 KB
q3       2.3985   74.7%    +0.7%     ~11.5 KB (layer 3 at q8, the first cached layer)
```
q3 scoring better than q4 puts the noise of 288 positions near 1%; nothing here separates q8, q4 and q3 from each
other by more than that. The indexer layers stay BF16 whatever the main type, so QSA's block selection cannot move;
only attention over the selected cells reads the quantized values. The 27B's finding carries over within this
measure's resolution. At 262K tokens q3 takes about 3.0 GB of KV against 7.9 GB in BF16, roughly 45 expert slots.

## Run 36 - 2026-10-24 10:06

Question: with q3 costing nothing measurable (Run 35), does a q3 default for offloaded models buy cached experts, and
so decode speed?

Built: `--pa-cache-type auto` becomes q3 when MoE experts leave the GPU (an expert cache asked for, or the weights do
not fit resident beside an unquantized cache), decided before the expert plan so slots and the prompt chunk are sized
for it; `none` pins the model's dtype. Test `an_auto_cache_is_quantized_only_when_experts_leave_the_gpu`.

No flags (planned context 32K, the paged cache under `--pa-memory-fraction` 0.9), `plan_measure.sh` (scratch), two
prompts, two rounds:
```
none   96 slots   chunk 3072   KV pool 105K tokens   32.1 / 34.1, 33.9 / 34.3 tok/s
q3     96 slots   chunk 4608   KV pool 219K tokens   34.1 / 32.5, 33.4 / 33.8
```
No slots gained: the planner stepped in sixteenths of a layer's experts (32 slots, ~3.5 GB), and q3's 0.6 GB at 32K is less
than a step, so it went to the KV pool. In sixty-fourths (8 slots): 104 slots either way. A probe of the planner's estimate
(removed after) against nvidia-smi: the prompt staging buffer (~1.5 GB) is allocated at the first prompt, so the KV
pool, sized at load, takes its room and the staging then takes from the 10% the memory fraction holds back (21.7 GB in
use after load, 23.6 GB after a 1.5K-token prompt). That 10% (~2.2 GB) is what keeps slots off the GPU here: with
`--pa-context-len 32768` (a context-sized cache holds nothing back) and q3, 128 slots load and decode at 34.6 / 37.9
against 33.7 / 35.0 at 104.

An asked-for long context is where q3 pays:
```
--pa-context-len 131072    slots   decode tok/s (two prompts, two rounds)          mean
none (BF16, 3.9 GB)        96      26.4 / 33.9, 30.7 / 34.7                        31.4
q3 (1.5 GB)                120     36.9 / 35.9, 36.2 / 37.5                        36.6
```
Kept: the q3 default and the sixty-fourths. Open: the memory fraction's 10% (and the staging buffer it silently absorbs) is
the next 20-odd slots for a model with no context asked for.

## Run 37 - 2026-10-24 11:11

Question: Run 36 found the memory fraction's 10% keeping ~20 slots off the GPU with no context asked for. Can an
offloaded model's cache hold just the planned context and give the rest to cached experts?

First attempt: with no flags and the planner offloading, size the paged cache to the planned context
(`ContextSize`, so nothing is held back). 128 slots, decode 35.1 / 36.9, 37.0 / 39.7 tok/s, and a 24K-token prompt
ran out of memory (21.5 GB after load, OOM in prefill). The 10% had been covering what prompts allocate.

What prompts allocate, with the chunk fixed at 1024 (96 slots, 32K context-sized q3 cache), peak nvidia-smi above
the post-load level: short prompt +1.8 GB, 5.6K tokens +2.3, 10.7K +3.0, 22.5K +3.0. The planner counts the staging
(1.54 GB, the largest layer's experts) and the chunk's activations (0.35 GB). An nsys run with
`--cuda-memory-usage=true` hung the server ~70 s into the prompt (0% CPU, no log lines; the same prompt runs without
it), so the split came from the engine's own pool snapshot, logged at each prompt boundary (probe, removed after):
```
                          after load    long prompts
async pool reserved       16,480 MiB    18,656-18,720   (staging 850 + 425 + 425, activations)
async pool used           16,449        18,221 (per-step high 18,479)
graph pool reserved       0             32 -> 96
device available          6,220         3,975 -> 3,620
```
No fragmentation to trim: the pool's cached bytes stay ~0.5 GB. The gap is real allocations: the staging is sized
per projection at the largest type across layers (one layer's Q8_0 down: 850 MiB), so 1.7 GB against the planner's
1.54; a step's transients (~0.3-0.5 GB above steady use); and ~0.3-0.6 GB outside both pools.

Built: when the planner moves experts off the GPU itself and the memory setting is the server's default
(`PagedAttentionConfig::default_memory`, set only when no context, size or share was given), the cache holds the
planned context and 1 GiB (`OFFLOAD_RUNTIME_RESERVE_BYTES`) is reserved for the above. Test
`only_a_planned_offload_with_the_default_memory_sizes_the_cache_to_its_context`.
```
no flags: q3, 32K context, 120 slots, load 20,742 MiB
prompt peaks   62 tokens 22,537   5.6K 22,870   10.7K 23,043   22.5K 22,996   29.6K (+300 decoded) 23,110 MiB
A/B against --pa-memory-fraction 0.9 (the old default, 104 slots), three alternating rounds, two prompts:
  120 slots   33.9 / 35.5   33.3 / 34.5   36.3 / 38.9    mean 35.4 tok/s
  104 slots   32.1 / 29.9   32.6 / 33.6   35.2 / 34.9    mean 33.1
```
+7%, every pair in favor, with about 1 GB of headroom at a full-context prompt. A longer context needs
`--pa-context-len` or `--max-seq-len` now (the pool no longer stretches to 219K tokens by itself).

## Run 38 - 2026-10-26 11:06

Question: what does another engine decode at on the same weights and this machine? Basalt (Run 20) builds for sm_120
only and refuses other targets. Strata (MIT, github.com/Niko1221/Strata, 4b9e05a2), the engine basalt forked, builds
for sm_75 and up and lists UD-IQ4_XS among its Flash-Next packs; its docs say that file "has not been measured on
NVIDIA".

Setup: Strata built from source for sm_86 (CUDA 12.8, Release), its pack from our shard 1 (`tools/iq_pack.py
--compat-bf16`, 1.4 GB, experts read from the GGUF in place), its MTP draft layer from the BF16 checkpoint's `mtp.*`
tensors (`mtp_fetch`, `mtp_pack --experts q2_0`, `mtp_rt`: 512 Q2_0 experts, 708 MB, resident). Config as its setup
writes for this file with all experts in RAM: `--resident-budget-gib 64 --expert-cache auto --prefill auto --spec 4
--spec-min-p 0.5 --mtp mtp/rt --max-context 32768`, int8 KV, `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`. Ours:
`inference serve` built with debug-assertions and overflow-checks off, no flags (planner: q3 KV, 32K, 112 slots; 104
with `--mtp`, n_predict 2). Client: one streamed chat request at a time, greedy, thinking off, 512 new tokens, a prose
and a code prompt, three rounds after a warm-up; decode = tokens after the first over first-to-last token time. Round
0 of the first server loaded was cold (experts paging in) and is left out of its means.

```
                                     prose (tok/s)          code (tok/s)
ours, no draft (112 slots)           37.8 37.4              32.3 34.5
ours, MTP (104 slots)                27.7 26.0 31.1         28.7 28.3 27.4
Strata, MTP, tail skip off           48.4 48.6 47.9         42.5 43.8 43.8
Strata, MTP, tail skip on (default)  59.0 56.8 57.7         56.9 56.8 55.0
```

Strata takes no run without drafting on a native pack ("needs --spec T (T >= 2)"). Its exact configuration decodes
1.29x ours on prose and 1.31x on code. The route tail skip (a missed expert that every token of a verify window ranks
7th or lower is neither copied nor computed) adds another 1.2-1.3x; Strata measures it at KL 0.056 -> 0.086 against
Q8_0 on UD-Q4_K_XL with every expert in RAM, and its setup turns it off for these files when they fit in RAM.

What its logs show for the exact run:
- GPU cache 6,616 experts (14.9 GiB, ~138 a layer against our 112), filled by its shipped routing profile, with
  adaptive swaps; the other 44 GiB page-locked in RAM. Decode hit rate 74-84% per request.
- 3-6% of routed experts read by the GPU over PCIe (its `pcie_frac`, 0.35 from a 12.9 GB/s probe); the CPU computes
  the rest of the misses.
- MTP accepts 244-333 of 466-487 drafts per request (52-70%), `--spec 4` with an early stop at min-p 0.5; the head's
  experts never leave the GPU. Ours at 104 slots is slower than no drafting (Run 26: verify misses ~6x).
- sm_86 turns on nine "exact speed switches" by default (ADAPT_NOWAIT, ADAPT_LAG, CPU_IQ_ALL_EXACT1, SIDE_MIXER,
  ATTN_DAILY, GR_UP75, GU_SWIGLU, QSA_ROWS, B6_MMVQ_ROWS).

Implication: the gap is drafting that pays. Strata keeps the drafter's experts resident in Q2_0, so a draft costs no
miss, and its verify windows run more slots (one cache sized by its profile, no MoE KV share). Next, in order: an MTP
head with resident experts (Q2_0 or similar, ~0.7 GB) and a verify window sized to the CPU share, measured against
these rows; then more GPU slots (the 14.9 GiB against ours) and a GPU share of the misses over PCIe.

## Run 39 - 2026-10-26 11:20

Question: Run 38's 1.3x, from which part of Strata? First, Run 26 already answers the plan Run 38 ended on: the head's
experts resident (Q8_0, and a Q2_K/IQ4_NL head that frees slots) left MTP break-even; Run 23 put the verify at 9.5
misses a layer (8.1 distinct) and 1,214 us of CPU against a plain step's 1.55 and 239 us, about 150 us a distinct
expert either way.

Strata exact (tail skip off) with its GPU share of the misses off (`--pcie-frac 0`), same client:
```
                         prose              code
GPU share on (Run 38)    48.4 48.6 47.9     42.5 43.8 43.8
--pcie-frac 0            47.8 49.9 46.0     42.2 39.4 41.8
```
The PCIe share is worth 1-5%. Its log, per request: 71-82% decode hit rate; on the code prompt 92.5K missed routes
over ~187 verify steps (512 tokens, 325 drafts accepted), about 10.3 missed routes a layer a step, 65 ms a step,
1.37 ms a layer. Ours (Run 23 numbers, Run 38 throughput): 9.5 missed routes a layer, ~86 ms a verify step, 1.8 ms
a layer. The CPU's cost per miss is about the same in both; the gap is ~0.4 ms a layer outside the miss kernel
(GPU work not hidden behind the CPU, the host function round trip, our head's own misses at 6.8 of 10 a draft).

Next: an nsys profile of one of our MTP verify steps, to split those 0.4 ms before building anything.

Profile attempt (`nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`, one code request under `--mtp`): the
report's kernel records stop about 20 s in, before most of the decode (36.8K kernels, 0.3 s of kernel time over the
13 s recorded), and a run windowed onto the decode (`--delay 40 --duration 8`) recorded no kernels at all. Under the
profiler the request decoded at 21.6 tok/s (28 without). Not resolved; the split needs another way in (the engine's
own per-step timing, or nsys on a `bench` run without the server).

## Run 40 - 2026-10-26 11:40

Question: where does a verify step's time go? A probe in the offloaded experts' host function (env-gated, removed
after) logged each call's layer, tokens, misses, start, CPU miss time and end; one code request (512 tokens) after a
prose warm-up, built with debug-assertions and overflow-checks off, no flags (112 slots; 104 with `--mtp`).
"Between calls" is the end of one layer's host call to the start of the next: the rest of the layer on the GPU, the
next layer's attention and router, the copies and the host function's launch. Medians per step:
```
                       step      CPU on misses        between host calls        after the last layer
plain, 1 token         29.2 ms   11.9 ms (83 misses)  14.2 ms (296 us a layer)   1.5 ms
MTP verify, 3 tokens   79.4 ms   30.6 ms (293)        38.3 ms (800 us a layer)   9.7 ms (two drafts; head 11
                                                                                  misses, 2.0 ms of CPU)
```
The CPU side scales as Run 23 found (104 us a miss in a verify, 144 in a plain step). The GPU side is the gap: a
verify step's layers take 2.7x a plain step's between host calls, because verify steps (more than one query token)
run eager while plain steps replay CUDA graphs (#609). At a plain step's 14 ms the verify would be ~55 ms, Strata's
(Run 39: ~65 ms at its window), and 2.4 tokens a step would give ~43 tok/s against 33-35 without drafting.

Next: capture verify steps in CUDA graphs.

The server's `/metrics` after an MTP request: every verify step `inference_cuda_graph_dispatch_total{mode="skipped",
reason="incompatible_shape"}` (54 of 54), from three one-token assumptions:
- `TokenWindow::decode_rows` lays out one step token a row, and the graph attempt filters on `q_len == 1`; precapture
  skips every verify width for a window model ("a window model graphs only one-token steps").
- PLE's host gather (`ple.rs`, `host_gather`) serves `len == 1` rows only.
- QSA's device plan (`QsaStep::decode`, `qsa_decode_plan`) is the one-token path; a verify step builds its pooling and
  selection on the host from each row's start and length (per-token table rows), which a graph cannot replay.
The speculative graph state itself exists (Qwen3.5's `take_speculative_graph_state`, with the PLE commit); whether it
carries Flash-Next's MTP inputs (the pre-mixer hyper-connection streams) is to be checked.

Plan: (1) a q-token window layout and PLE host gather, and precapture of the verify width for window models; (2) a
device QSA plan for verify steps (each query token's table row and context, pooling ordered so the latest token's
write of a shared block wins); (3) the MTP graph state for the streams; then measure against the rows above.

## Run 41 - 2026-10-26 12:23

Question: with verify steps graphed, does MTP pay? Built: the token window's decode layout carries `query_len` step
tokens a row (`TokenWindow::decode_rows(preceding, query_len)`), PLE's host gather serves (batch, len) steps up to
eight tokens, precapture captures the verify widths of a window model, QSA's device plan (`qsa_decode_plan`) runs a
verify's per-token table rows as one-token rows (a sequence's pooled keys and length from its last token's row; tokens
of one block pool it alike from the step's written cells), and the PLE conv history a verify commits from moves into
the speculative graph state like the GDN stash. Bugs met on the way, from the tiny model's tests:
- with the window fixed but QSA still host-planned, capture failed safely ("QSA reads the batch's token window") and
  turned graphs off for every step;
- `cuMemFreeAsync` returned INVALID_VALUE inside the verify capture (an nsys API trace of the tiny test: the first
  non-zero return 0.6 ms into the capture): the forward replaced `ple_stash`, freeing the eager warm-up's history
  inside the capture. Taking the stash with the graph state also keeps a replay's commit fed: before, the first
  commit took it and later replays never refreshed it.
The MTP and DFlash greedy tests now also assert every non-prompt step replayed (`assert_only_prompts_skip`).

Real model, same client and server flags as Run 38:
```
                     prose                 code
MTP, verify graphs   28.3 28.8 31.7        30.9 32.3 31.6     1,403 replays, no eager fallback; accept 76.9%
no draft (same build) 41.4 42.6 42.1       35.9 36.5 37.3
```
The no-draft rows are above Run 38's 37.6 / 33.4 with that path unchanged (run-to-run spread). The host-call probe
(Run 40's, re-applied and removed after): a verify step's layers 55.3 ms (CPU misses 35.3, between host calls 18.0
against 38.3 eager) and 18.0 ms after the last layer (9.7 eager). A host timer around the speculative step (removed
after), means over 100 verify steps:
```
target forward (waiting on its logits)   41-56 ms
commit                                    2-4.5 ms
propose (two head drafts, eager)          8-14 ms
```
Graphs took ~20 ms out of the verify's layers; MTP still loses to no drafting here. Left: the drafts (eager, the head's
experts missing), the CPU's misses (~35 ms of the forward), and the real model's precapture still logging "CUDA graph
captured forward failed" before the lazy captures succeed. nsys with `--cuda-graph-trace=graph` lost its records
after 8 s again (5 graph launches recorded), so host-function graphs and CUPTI do not mix here.

## Run 42 - 2026-10-26 12:40

Question: Run 41 left the CPU's misses as the cap. Strata caches 6,616 experts (~138 a layer) against our 120; how
much VRAM does our offload hold idle during decode, and can it become slots?

At decode 21.4 GB of 24.5 was in use; a 14.6K-token prompt peaked about 0.5 GB above that. Prompt staging held a
whole layer's experts (1.7 GB) for the grouped kernels. Built: a prompt layer now runs its cached experts straight
from the layer's cache stack, then its uncached ones over the bus `OFFLOAD_STAGING_EXPERTS` (64) at a time, each pass
running every route with the routes of other passes parked: `grouped_parked_with` dispatches over one expert more than
its kernels run and the parked routes carry weight 0, which the weighted reduce now skips (an unwritten row could hold
a NaN). Staging and the planner's count take a group's room. `offloaded_experts_match_resident_ones` runs a cached
pass and two staged groups (relative error < 1e-5 against all experts resident).

```
no flags, no draft         slots   chunk   decode prose      decode code       14.6K-token prompt
before (Run 41 build)      120     1024    41.4 42.6 42.1    35.9 36.5 37.3    -
group staging              128     2048    42.2 44.6 46.5    36.2 38.3 38.5    405 tok/s
```
Run 27 had the 2,048 chunk at 301-305 tok/s; the cached pass reads its experts in place instead of copying them into
staging. Decode moved a few percent, within the spread. Slots move in steps of eight (`PLANNED_SLOT_STEPS`), and a
second load, with the desktop holding a little less, planned 136 slots and shrank the prompt chunk to 512 (Run 27: ~120
tok/s): slots first, chunk second turns one slot step into a 3x slower prefill.

The planner's terms at 136 slots (a probe in `ExpertCachePlan::spare`, removed after): device available 22,315 MiB,
weights 19,813, KV 380, the runtime reserve 1,024, the recurrent pool 998 (9 slots of ~111 MB: one sequence, the graph
pad, and 7 kept for prefix snapshots), 51 spare. Those 7 snapshot slots are ~7 cached experts a layer.

Against Strata (Run 39): our verify misses 306 routes of 1,440 (21%, 79% hits) at ~115 us a miss; Strata hit 74-84%
at ~133 us. The slot gap is mostly closed and the miss cost is now the same in both engines. What separates the verify
steps is the drafting (8-14 ms here; Strata's head is resident) and the commit.

Kept (user's call): an offloading plan reserves a 2,048-token chunk's activations while it picks slots
(`MIN_PLANNED_PROMPT_CHUNK`), and keeps 2 prefix snapshots in the recurrent pool (`OFFLOAD_PREFIX_SNAPSHOTS`; a host
prefix store already capped them at 2). No flags: 128 slots, chunk 2,560, the 14.6K-token prompt at 438 tok/s, decode
38.5 / 42.4 prose and 34.3 / 37.5 code.

## Run 43 - 2026-10-26 13:06

Question: what can the drafting work buy? A probe in `mtp_propose` (a device sync after each phase; removed after),
means per proposal of two drafts over each 100 verify steps:
```
refresh forward (anchor + accepted rows)   2.2-6.4 ms
chained forward (one row)                  2.0-4.5 ms
lm_head + sampling, both drafts            1.5 ms
```
A forward is ~2 ms at its fastest; the spread is the head's missed experts, ~3-4 ms a proposal. Keeping the head's
experts resident costs ~1 GB at Q2_K/IQ4_NL (Run 26), one slot step of the layers' cache (8 slots), which the head
already costs: the MTP servers plan 120 slots against 128 without it. Graphing the head would take the ~4 ms of eager
work down by part of it. With all of the drafting gone a verify step is ~61 ms for 2.5 tokens, ~41 tok/s: level with no
drafting.

Draft depth, verify graphs on, no flags otherwise:
```
                   slots   prose          code           accept
no draft           128     38.5-46.5      34.3-38.5      -       (Runs 42 and its planner follow-up)
MTP, 1 draft       120     38.6 40.2      35.8 35.9      84.7%, 1.85 tokens a step
MTP, 2 drafts      120     34.7 35.6      34.7 35.9      78.8%, 2.58
```
One draft is level with no drafting; two lose. On this machine (PCIe 3, an AVX2 8-core CPU computing ~20% of routes)
a verify's extra tokens cost the CPU nearly what they save, so the drafting work's ceiling is break-even. Not built.

## Run 44 - 2026-10-26 13:24

Question: what takes Strata to 58 tok/s? Its own profile (`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`, the
exact config of Run 38, tail skip off) and two ablations, same client, one round each.

```
Strata                          prose   code    per window (code unless noted)
--spec 4 (default, adaptive)    49.4    43.1    62.3 ms, T 3.47, 2.68 tokens: verify 53.3 (GPU-reach wait 11.1 + CPU
                                                40.7 for ~308 missed experts) + draft 2.2 + commit 0.04
--spec 2 (one draft, as ours)   51.9    42.0    prose 32.4 ms, T 2.00, 1.68 tokens: GPU-reach 10.4 + CPU 14.4 (~107
                                                misses) + draft 1.2 + commit 0.06 + ~5 other
STRATA_SM86_PERF=0              44.4    40.4    the nine sm86 switches: 6-11%
tail skip on (Run 38)           57.7    56.8
```
Ours at one draft (Run 43): 38.6-40.2 / 35.8-35.9, a 2-token verify at ~47 ms; a plain step 27-29 ms (CPU 11.9, GPU
between host calls 14.2). Per miss the CPUs cost the same (~135 us Strata, 115-143 here), and Strata's GPU time not
hidden behind the CPU (10.4 ms for 2 tokens) is below ours for one. What separates the windows:
- the draft: ~1 ms a draft (its Q2_0 head resident, a draft vocabulary subset) against ~4.5 ms here plus 1.5 ms of
  lm_head and sampling;
- the commit: 0.04-0.06 ms against 2-4.5 ms here;
- the GPU's part of a window: Strata hides more of it behind the CPU's misses and reads each weight once for the
  window's tokens;
- nine sm86 speed switches, 6-11%;
- the route tail skip (lossy, +20-30%).
Run 43's ceiling was MTP against our own plain decode with our drafting costs; with Strata's draft and commit costs a
one-draft window pays here too.

## Run 45 - 2026-10-26 13:46

Question: Run 44 put our draft at ~4.5 ms against Strata's ~1. Where does it go, and does a resident head pay now?

Probes (device syncs between phases; removed after), one draft (`--mtp-n-predict 1`):
- the commit takes the replay path every window (the GDN state re-run for rows with a rejected draft): 0.6-1.6 ms,
  PLE's 0.1 ms. The 2-4.5 ms of Run 41 also held the publish. Small; left.
- a head forward over ~1.7 rows: host setup 0.3 ms, then 3.8-4.2 ms in the forward call (uploads from pageable memory
  wait on the stream, so this is the GPU's time): input projections 0.23, the block 4.0-4.3 (attention 1.2, the MoE
  side 2.1-2.2, which holds ~11 missed experts on the CPU), the mixer 0.05.

The head's cache was sized like a layer's at Q8_0 (120 slots x 5.2 MB, ~0.63 GB) and its 5.2 MB experts set the
prompt staging's group size (64 x 5.2 MB against 150 MB for the layers). Resident at Q2_K gate/up and IQ4_NL down
(Run 26's file), 112 slots both, one draft (a temporary switch loading the head's MoE without the cache):
```
                         prose        code         accept
head offloaded           37.3 39.0    35.1 34.6    90.3%
head resident            40.9 43.5    38.7 37.5    85.7%
```
Built: with the model's experts offloaded the head's MoE loads resident, and when the weight source holds the head's
experts the loader adds immediate-ISQ overrides for them (gate/up Q2_K; the 640-wide down Q4_0, as Q2_K/Q3_K need
256-wide rows and IQ4_NL has no quantizer here). The stock Q8_0 head file, no flags but `--mtp`: load 24 s (~9 s of it
the requantization), 120 slots, chunk 2,560, 21,953 MiB after load (128 slots without MTP: 21,360; eight slots free
~0.9 GB, so the head holds ~1.5 GB with its dense weights, not the 2.6 GB of Q8_0 experts).
```
                                      prose        code         accept
MTP 1 draft, resident head (120)      42.1 44.3    37.8 38.5    88.9%, 1.89 tokens a window
MTP 2 drafts, resident head (120)     33.8 38.8    34.4 39.4    79.3%, 2.59    (round 0 cold)
no draft (128)                        38.5-46.5    34.3-38.5
```
One draft now matches or beats no drafting. The planner still counts the head as a cached layer (120 Q8_0 slots and
its staging, ~0.96 GB against ~1.0 GB resident); the 1 GiB reserve covers the difference.

## Run 46 - 2026-10-26 13:50

Question: what are Strata's nine sm86 switches (6-11% of its decode here, Run 44), and do any map onto our kernels?

Read: `src/core/nv_defaults.cpp` and each switch's site. Each picks between bit-identical variants of Strata's own code,
kept by removing one at a time on an RTX 3060 and an A4000 (drop-one cost in points, A4000 / 3060):
- `ADAPT_NOWAIT` + `ADAPT_LAG=2` (-6.3 / -0.8): its adaptive expert-swap rounds stop waiting for their swaps. We fill
  an LRU cache from the CPU's misses instead; no counterpart.
- `CPU_IQ_ALL_EXACT1` (-3.3 / -0.6): which of its AVX2 IQ dot kernels a one-token expert group takes
  (`native_expert.cpp`).
- `SIDE_MIXER=2` (-1.2 / -2.3): the hyper-connection mixer on a branch of the window's graph (`verify.cpp`).
- `ATTN_DAILY` (-1.2 / -1.3), `QSA_ROWS` (-0.5 / -1.2): variants of its QSA decode attention and block scores.
- `GR_UP75` (-0.9 / -1.8), `GU_SWIGLU` (-0.8 / -1.4): its fused gate/up kernels (a 75-row tile; SwiGLU in the epilogue).
- `B6_MMVQ_ROWS` (0 / -1.7): rows per block in its matvec.
None is a kernel we run. Transferable ideas: the mixer on a parallel graph branch and gate/up with SwiGLU fused, ~1%
each by Strata's numbers. Left for later.

## Run 47 - 2026-10-26 14:15

Question: Strata leaves ~10 ms of a 2-token window's GPU work exposed (Run 44); a plain step here leaves 14 ms between
host calls. What of ours could run while the CPU computes the misses?

Read: Qwen3.5's MoE block ran the router, then the experts (offloaded: the cached ones on the stream, the misses through
the host function, the stream waiting for its result), then the shared expert. The shared expert queued behind the
wait; Strata's stage list runs `shared+quant` before its CPU waits. Built: `MoEExperts::forward_overlapping` takes work
to overlap; offloaded experts run it after launching the cached experts' kernels and before the stream waits for the
CPU, other backends after the experts. The shared expert and its gate go through it.

No drafts, interleaved after / before / after / before, two rounds each (the first after round cold, left out):
```
          prose                    code
after     41.4 40.5 41.6 (41.2)    33.9 36.9 34.9 (35.2)
before    38.1 41.7 39.4 42.3      35.9 33.5 36.6 36.8
```
Within the spread. The host-call probe (Run 40's, applied and removed for each build), one code request each:
```
          step      CPU on misses   between host calls
before    27.9 ms   11.4            13.97 ms (220 us min gap)
after     25.4 ms   10.6            12.64 ms (201 us)
```
1.3 ms a step of the shared expert now runs beside the CPU (the CPU's share varies run to run). Kept.

## Run 48 - 2026-10-26 14:31

Question: which GPU work does a step leave exposed, and how does a one-draft window compare with Strata's?

nsys with graphs off (`INFERENCE_RS_CUDA_GRAPHS=0`, `--delay 35 --duration 4`, ~97 tokens of the code request; a run
without the window stops at exactly 40,000 kernel records, which is what truncated Runs 39-41's captures). Kernel time
12.2 ms a token: Q8_0 dense matvecs 3.7 (the plain, fused-QKV and fused-GLU ones; ~82% of the 3090's bandwidth bound
for those weights), cached experts 2.25 (IQ3_S 1.30, IQ4_NL 0.84, Q8_0 0.11), the hyper-connection reads and mixing
~1.8 (`q8_0_exact_rows` 1.07), lm_head 0.62, attention 0.32, router 0.30, GDN recurrence 0.22, and ~1.5 of kernels of
1-3 us (casts, adds, copies, gathers). Strata's GPU stages for a 2.8-token window sum to ~15 ms: per window it reads the
weights once for every token in it.

The host-call probe on `--mtp --mtp-n-predict 1`, the prose request (300 windows): median 36.5 ms, mean 39.4, p90 54.8,
p99 75.3. The slowest tenth are miss bursts (274 missed experts, 34.4 ms of CPU, against 65 and 8.4 ms in the faster
half). The faster half against Strata's `--spec 2` window (Run 44):
```
                         ours (faster half)   Strata
CPU on misses            8.4 ms               14.4
GPU not under the CPU    18.8                 10.4 (its GPU-reach wait)
after the last layer     3.7                  ~6.4 (draft, commit, emit, other)
window                   31.2                 32.4
```
Strata's exposed GPU (10.4-11 ms) is its GPU stages less what runs beside the CPU (cached experts, shared expert, the
PCIe share). Ours is several ms more than its kernels account for. Each layer here pays a host-function round trip:
four copies out (ids, weights, slots, activations), an event, the host function's launch on a side stream, a copy
back. Strata and basalt use a doorbell instead (routes written to mapped memory by the GPU, a spinning host thread, a
GPU kernel spinning on the result flag; Run 20). Next: measure that round trip's fixed cost per layer.

## Run 49 - 2026-10-26 14:48

Question: does the per-layer host round trip (Run 48's suspect) cost enough to replace it with a doorbell?

`split_round_trip_costs_a_fixed_latency` (offload.rs, ignored benchmark): an offloaded layer of 64 Q4_K experts
(2560 x 256, top 10) with every expert cached after a warm-up, so the host function computes nothing, against the same
cached-expert kernel resident; 500 steps, eager and replayed from a graph. Ten runs:
```
          offloaded      resident
eager     69-101 us      44-50 us
graph     59-66 us       40-46 us      (~19 us a layer: ~0.9 ms over 48 layers)
```
With the cached experts' kernel this long, most of the round trip runs beside it. When the CPU's misses outlast that
kernel (a 2-token window: ~175 us of CPU a layer against ~98 us of cached experts), the CPU cannot start before the
host function does, so the round trip's latency, up to the ~55-60 us an empty one takes here, lands on every layer:
up to ~2.6 ms a window. A doorbell (a GPU-written flag a spinning host thread watches) would start the CPU within a
few us: worth perhaps 1.5-2.5 ms a window, 5-8%. Not the whole of Run 48's gap.

Kernel time per 2-token verify window against a plain step (nsys, graphs off, `--mtp-n-predict 1`, 54 windows
against 87 steps): 19.1 ms against 13.6. Dense Q8_0 matvecs 3.44 ms for two columns against 3.0 for one; cached
experts +2.1 (two tokens route to ~twice the distinct experts); new in a window: cuBLAS GEMMs for the 2-row router
and gate projections (`gemv_kernel_rowtile`, `Kernel2`, `splitKreduce`, ~1 ms) and an f32 `index_select` 37 times
(`is_u32_f32`, 16 us each, 0.6 ms; GDN state copies for the rollback).

A first version of the benchmark failed intermittently with CUDA_ERROR_ILLEGAL_ADDRESS on replay: it captured without
the engine's preparation, and kernels whose parameter vectors are uploaded from the host replayed copies from freed
host buffers. Capturing as the engine does (`prepare_cuda_graph_memory_pool`, the h2d parameter cache warmed by an
eager call first) passes 10 of 10. Not an engine fault.

## Run 50 - 2026-10-26 15:21

Question: Run 49's two small projections that only a verify window pays for: why, and what does fixing them buy?

Launch grids in the two nsys captures. The one-output shared-expert gate and the 512-output router are plain
`Linear`s that call the tensor's matmul: one row runs cuBLAS's own `dot_kernel` (1.8 us), two rows a split-K GEMM
(`Kernel2` + `splitKreduce`, ~8.7 us, 49 a window). The 4-output bf16 projections the layers run twice took the
one-row-a-block gemv at one row (4 blocks, 3.1 us) and the row-tile gemv at two, 4 rows a block, so one block on one
SM (6.1 us, 96 a window).

Built: the MoE block's router and shared-expert gate take the small-batch matvec (`inference_quant::gemv`, now
re-exported) when it fits, the matmul otherwise; the row tile needs 64 output rows or more (`ROWTILE_MIN_M`). Kernel
time per 2-token window (nsys, graphs off, `--mtp-n-predict 1`): 19.14 -> 18.72 ms; the four kernels' 1.44 ms became
1.05 ms of `gemv_kernel_batched`. ~0.4 ms, about 1%.

(`gemv.cu` is not clang-format clean on master; only the touched lines follow it.)

## Run 51 - 2026-10-26 15:43

Question: does a doorbell in place of the per-layer host function (Run 49: worth 1.5-2.5 ms a CPU-bound window, if
the host function's start latency sat on the critical path) pay?

Built and measured, then dropped: a publish kernel wrote the picks and activations into the layer's mapped buffers
and rang a sequence number; one host thread polled every layer's doorbell (spinning while decoding, parked 50 ms after
the last ring or step start, which `forward_step` signalled), ran the same `split`, and answered; a wait kernel spun
on the answer (2 s timeout) and copied the CPU's rows back. Both offload tests passed. The round-trip benchmark: graph
61-68 us against 41-47 resident, the same ~20 us as the host function.

`--mtp --mtp-n-predict 1`, two rounds each, after / before / after (a fourth server, before, did not start: below):
```
           2-token window (median)   between host calls   CPU      prose          code
doorbell   40.2 ms / 38.2            16.6 / 14.9          16.6 / 16.2   37.2 42.8 / (23.0 cold) 43.7   39.0 37.8 / 38.8 39.7
host fn    41.1 ms                   15.9                 16.2     38.5 41.9      34.3 39.0
```
At most 1-2%, within the spread, and the spinning thread takes a core from the CPU's expert pool while decoding. The
host function's start latency is not the gap; dropped. The kernel source is kept outside the repo.

A planner fault the A/B exposed: the fourth server stopped with "PagedAttention KV cache requires 410 MB but only 311
MB is safely available after model loading". The planner counts a resident MTP head as a cached layer (Run 45:
~0.96 GB counted against ~1.0 GB resident); with a little more of the GPU held elsewhere the KV cache no longer fit.
Next: count the head at its resident, requantized size.

Fixed: the planner counts a resident MTP head's experts at their requantized size (`MTP_HEAD_EXPERTS`: the
projections, their GGUF types for sizing and ISQ types for the loader's overrides, one table) from the source's
blocks, and only the mapped layers' experts size the prompt staging. `--mtp --mtp-n-predict 1`, no other flags: 120
slots, chunk 2,048 (2,560 before, which the undercount had paid for), 22,031 MiB after load and 22,700 MiB at a
14.6K-token prompt's peak, prose 35.3 and code 36.0 tok/s (one round).

## Run 52 - 2026-10-26 16:04

Question: where does the branch stand, on numbers firm enough to quote? Interleaved, no-draft / MTP / no-draft /
MTP servers, three rounds each, the same client (no flags but `--mtp --mtp-n-predict 1`):
```
                      prose median (range)     code median (range)
no draft (128 slots)  45.1 (43.9-46.0)         38.5 (38.1-39.2)
MTP 1 draft (120)     45.1 (40.2-45.7)         39.6 (39.0-40.5)
```
Against the start of Run 38 (37.6 / 33.4) +20% / +15-19%; against Strata's exact one-draft window (51.9 / 42.0)
-13% / -6%. One draft ties on prose and wins 3% on code; two lose (Run 43). Kept: an offloaded model's MTP head
drafts one token unless `--mtp-n-predict` says otherwise (`DEFAULT_MTP_N_PREDICT_OFFLOADED`). With `--mtp` alone the
log reads `n_predict=1`; the precapture now takes all 8 graphs (Run 41's "captured forward failed" is gone since the
PLE history moved into the graph state), 70 replays and no failures over a request.

## Run 53 - 2026-10-26 16:34

Question: where do a verify window's exposed GPU milliseconds go, stage by stage, inside replayed graphs?

A probe (branch `probe/stage-marks`, local): a one-thread kernel writes `%globaltimer` to a mapped slot at named
points (layer start, PLE, the GDN read, the mixer, the MLP read, the router, the split's launch, the cached experts,
the shared expert, the CPU wait, the combine); slots are assigned in call order inside the target's `forward_step`,
so a lazily captured graph replays them (the startup precapture is skipped while the probe is on, and a capture
step's eager warm-up half is dropped in the analysis). `--mtp` (one draft), prose after a code warm-up, 435 2-token
windows, medians summed over layers:
```
window (first to last mark)                 34.0 ms
shared expert done -> the CPU's answer      13.3      (CPU-bound remainder)
cached experts' kernel                       4.9      (beside the CPU)
GDN mixer                                    3.9      Strata, 2.8-token window: 3.9
hyper-connection read into GDN layers        1.66     Strata hc-read0: 0.43
router -> split launched                     1.53     (casts, slot lookup, 4 copies; none in Strata)
attention layers (12)                        2.34     Strata QSA stages: ~1.5
PLE (host gather, inline)                    0.98     Strata: gathered before layer 1, off the critical path
shared expert                                0.96
hyper-connection write + MLP read            0.86
router                                       0.60
```
Run 49-50's "unexplained" exposed GPU time is these pieces: the hc reads (unquantized activations since #620, for
KLD 0.043 -> 0.033; `q8_0_exact_rows` at 6.2 us a call), PLE's inline gather, the per-layer preparation of the
split, and attention. The CPU-bound span (cached experts + shared + wait, ~19 ms) is longer than Strata's per-layer
host time (14.4 ms).

Next, by size: a faster exact hc-read kernel (~1.2 ms a window), PLE's gather before the step launches (~1 ms), one
kernel for the split's preparation (~1.5 ms), attention (~0.8 ms).

## Run 54 - 2026-10-26 16:56

Question: do the split's preparation in one kernel and a faster PLE gather take Run 53's milliseconds back?

Built: one kernel prepares an offloaded layer's split (each pick's slot from the mapped table, the slot ids for the
cached experts' kernel, and the picks, slots and f32 activations straight into the mapped exchange), in place of an
activation cast, an `index_select` and four copies. The stage marks: router -> split launched 1.53 -> 0.47 ms a
window. The window did not move in that run: PLE went 0.98 -> 3.95 ms.

PLE's host gather reads its rows from the GGUF's 27 GiB n-gram table (`per_layer_token_embd.weight`, IQ4_NL
[320,001,536 x 160], 90-byte rows) in place, one row after another; a row whose page the page cache no longer holds
faults alone. A probe in `MappedBlocks::gather_rows` (removed after): 36-43 rows a call, 2.0 ms mean, 10 ms worst.
Built: the gather first asks for every row's page (`madvise(MADV_WILLNEED)`), so the faults are in flight together:
0.22 ms mean, 2 ms worst.

Interleaved as Run 52 (two servers per setup, three rounds each):
```
                      prose median (range)     code median (range)
no draft              48.3 (46.3-51.0)         41.2 (40.0-41.8)
MTP 1 draft           48.4 (45.3-50.3)         41.1 (40.5-41.6)
```
+7% on both against Run 52 (45.1 / 38.5, 45.1 / 39.6). Against Strata's exact one-draft window (51.9 / 42.0): -7% /
-2%. One draft and none are level again here.

## Run 55 - 2026-10-26 17:49

Question: with PLE's noise gone (Run 54), does the doorbell Run 51 dismissed pay after all?

Run 51's A/B ran while PLE's gather faulted pages one at a time (0.98-3.95 ms a window), so its spread hid a
difference of a few percent; that dismissal was confounded. A probe on master timing each layer's host function
from inside (its CPU work) and from the stream's side (copy-out to copy-back) put ~132 us a layer of round trip
outside the function itself. The fills' landing callbacks run on the same driver callback thread, and ~45 us of that
is the host function queued behind them.

Built: the split's preparation kernel writes the picks to mapped buffers as before, then a one-thread kernel rings the
layer's doorbell (a sequence number and the token count in mapped memory); a `moe-doorbell` host thread polls every
offloaded layer's bell, runs the misses on the CPU, and writes the sequence back; a wait kernel spins on that reply
(2 s timeout, then the step adds nothing and the thread logs it) and copies the rows in. No host function and no
side stream. The thread spins one core while decode steps arrive and parks to a 1 ms poll after 50 ms without
activity (the decoder marks each step).

Interleaved A/B, MTP 1 draft, two servers per setup, three rounds each:
```
                      prose median             code median (range)
doorbell              50.3                     42.9 (41.3-43.2)
host function         49.0                     41.8 (38.8-43.0)
```
+2.7% / +2.6%. Against Strata's exact one-draft window (51.9 / 42.0): -3% / +2%. Shipped; the cost is one spinning
host core while a model with offloaded experts is decoding.

## Run 56 - 2026-10-26 18:15

Question: nvtop showed GPU headroom during the speed tests; where is the GPU idle in a decode step, and can it be used?

Measured: `nvidia-smi --query-gpu=utilization.gpu -lms 100` through four 512-token decodes (MTP 1 draft) reads 96%
median, but that counter (nvtop's too) is "a kernel is resident", and the doorbell's wait kernel is resident while it
spins. An nsys capture of 2.6 s of code decode with GPU metrics (`--gpu-metrics-set=ga10x`) and graph nodes traced:
```
                                   all     during doorbell spin    outside it
GPU Active (what nvtop shows) %    93.9    99.9                    90.0
SMs Active %                       30.8     3.8                    48.5
DRAM read bandwidth %              19.8     2.0                    31.5
PCIe RX %                          43.6    29.3                    52.8
```
On the main stream: 39% of the time is the doorbell spin (the GPU waiting for the CPU's misses, ~17 ms a step), 43%
kernels, 16% gaps. Of the gaps, 354 of 419 ms fall inside the eager (non-graph) run between a step's graph replays
(sampling, accept/rollback, GDN state commits, the MTP head): 6.7 ms a step of span for 1.2 ms of kernels, with ~70
us between consecutive eager kernels. The launching thread spent 3.5 ms of each run in `pthread_rwlock_wrlock`,
called from `cuModuleGetFunction` (cudarc `get_function`): `CudaDevice::get_or_load_func` looked every launch's
function up again, and that driver lock is held by the `moe-expert-fill` thread's `cuMemcpyHtoDAsync` from the
pageable GGUF mapping (60-150 us a call, three a fill; registering the mapping fails with CUDA_ERROR_NOT_SUPPORTED).

Built: each module keeps the functions looked up in it (`LoadedModule` in `cuda_backend/device.rs`). Same capture:
```
                                   before      after
eager run span (median)            6657 us     2840 us
  its kernels                      1167 us     1162 us
  launcher rwlock waits            3465 us       31 us
main-stream gaps                   16.2%       7.6%
steps in 2.6 s                     60          63
```
The spin is now 46% of the time. What is left idle: the spin (CPU misses, ~19 ms a step) and ~1.8 ms a step of host
gaps around sampling (top1 -> next step's embedding 0.69 ms, top1 -> draft 0.53 ms, hc_inject_add -> cast 0.33 ms,
row copy -> embedding 0.22 ms). Outside the spin only ~half the SMs are busy (small decode kernels).

Interleaved A/B against master (MTP 1 draft, two servers per build, three rounds each):
```
                      prose median (range)     code median (range)
master                50.3 (47.5-51.1)         42.3 (41.0-43.0)
functions cached      51.6 (48.3-52.8)         44.8 (43.5-45.8)
```
+2.6% / +5.9%. Against Strata exact one draft (51.9 / 42.0): level on prose, +7% on code. Next, by size: the spin
(CPU misses; Strata's GPU share over PCIe was worth 1-5% there, Run 39), the ~1.8 ms of host gaps around sampling,
and the small kernels outside the spin.

## Run 57 - 2026-10-26 19:21

Question: after Run 56, what is each remaining idle piece of a decode step, and which can be taken back?

A step (code decode, MTP 1 draft, nsys over two streams) now runs: one graph (embedding, the 48 layers, lm_head),
top-1 sampling, a host read of the token, the MTP head eager (refresh over the anchor and accepted rows, draft
lm_head, top-1), a host read of the draft, the next graph. Per step, of ~37 ms:
```
doorbell spin (CPU misses)                    ~17-19 ms
graph kernels                                 ~17 ms
MTP head, eager                               2.2 ms span: 1.15 ms kernels (0.70 of it the Q6_K draft lm_head), 1.24 ms between launches
draft top-1 -> next graph's first kernel      0.56 ms: ~80 us blocking-sync wake, ~340 us host work, ~250 us launch to first kernel
verify top-1 -> MTP head                      0.27 + 0.21 ms
PLE gather inside the graph                   0.23-0.37 ms (layer 0 -> 1)
```
PLE: the gather ran when its layer (1) reached it: the window's copy out, the host function on a side stream (on
the driver's callback thread), the rows back, and the stream waited for all of it. Built: the step starts the gather
before layer 0 (`PleLayer::gather_ahead`; `HostGather` keeps it pending and the layer's forward joins it). The layer
0 -> 1 gap goes (334 us a step on average); a ~112 us gap opens after the step's first 11 kernels where the graph
moves its main branch to another stream around the host node. Steps in 2.6 s: 63 -> 66. Interleaved A/B (as Run 56):
```
                      prose median     code median
master (#643)         51.1             45.1
PLE gathered ahead    51.7             45.1
```
Within noise, as ~0.2 ms of 37 would be.

The draft lm_head reads 521 MB (Q6_K, 248,320 x 2560) in 0.70 ms, ~750 GB/s: near the bandwidth, so only a smaller
draft vocabulary (FR-Spec style) would shrink it, at some acceptance. The 1.24 ms between the head's 115 launches is
eager overhead (~11 us a launch); a graph needs its paged metadata (block tables, slot mappings, the QSA plan, the
flashinfer views) in resident buffers as core's decode graphs keep the target's.

CPU pool size, now that misses no longer go through host functions (Run 29 had 16 threads at 12 tok/s, 12 level
with 8). Interleaved, same build:
```
                      prose median     code median
CANDLE_NUM_THREADS 8  50.6             45.4
12                    50.7             45.6
16                    23.8             18.5
```
And 9 and 10, interleaved with 8 again:
```
                      prose median (range)     code median (range)
8                     51.9 (48.4-54.5)         45.2 (44.0-45.9)
9                     51.1 (49.1-54.0)         45.1 (44.4-45.8)
10                    51.7 (49.5-54.5)         45.4 (44.4-46.0)
```
Still no gain from 9 to 12 threads, and 16 collapses: the doorbell thread spins one core, and sixteen pool threads leave
no core for it or the host. Eight (the physical cores) stays.

## Run 58 - 2026-10-26 20:38

Question: with `perf` now allowed (`kernel.perf_event_paranoid` 2), which threads hold the CPU during a decode, and
does the cache-fill thread cost the CPU misses anything?

`perf record -F 4999 -g --call-graph dwarf -p <server>` for 4 s of a code decode (MTP 1 draft): nine threads at a
full core each (the doorbell thread, the seven pool workers, `moe-expert-fill`), the launching thread at 0.6% of
the samples. The fill thread spends 83% of its samples in a libcuda SSE load / non-temporal store loop: the driver
staging each pageable `cuMemcpyHtoDAsync` through its own pinned buffer. The pool workers split about evenly between
the dot kernels and the barrier's spin.

`flash_next_offload_misses` beside a stand-in for that thread (one thread copying 3 MB chunks with SSE loads and
non-temporal stores; dev profile):
```
                          1 miss      3 misses    10 misses (us a step)
8 threads alone           137         329-331     1049-1074
8 + copy at 5 GB/s        147         414         1229
8 + copy unpaced (12.7)   157         414         1228
7 threads alone           132         390         1104
7 + copy at 5 GB/s        154         406         1342
6 threads alone           151         410         1277
6 + copy at 5 GB/s        157         414         1297
```
The copy costs the kernel 15-26% at eight threads, the same paced or not: a core more than bandwidth.

In the engine the fill thread copies only because the experts are pageable: registering the GGUF mapping fails
(CUDA_ERROR_NOT_SUPPORTED; the kernel refuses long-term pins of file-backed ext4 pages). A probe copied the 55.4
GiB of routed experts into pinned memory at load (`HostExpertStack::new`, removed after), so fills DMA with no CPU
copy. Interleaved A/B, a warm round before each measured three:
```
                      prose median     code median
pageable (master)     52.0             44.1
pinned                52.2             44.7
```
Level, and the pinned load took 1 min 42 s against 11 s (allocating, locking and copying 55 GiB on one thread),
holding 55 GiB of the machine's 125. The first, unwarmed run read 38.0 / 41.7 for pageable: the pinned server had
evicted the page cache the pageable one reads from. Not pursued.

## Run 59 - 2026-10-26 21:12

Question: replayed from a CUDA graph, how much of the MTP head's 2.2 ms eager span (Run 57) comes back?

Built: the head's drafter forward (embedding, hyper-connections, the QSA attention block, its resident MoE) captured
per row shape and replayed (`qwen3_5::mtp_graph`). Its paged metadata goes through decode rows
(`make_paged_rows_decode_rows`, block tables as wide as the paged context) into the decode-graph metadata buffers
core's target graphs use, now in `inference_nn::cuda::graph_metadata`; tokens, MRoPE positions and target hidden
states are staged into fixed buffers. A shape's first run warms eagerly over those buffers under the parameter
upload cache, its second captures, and a failed capture or replay leaves the shape eager. The draft lm_head and the
sampler stay eager. The QSA plan takes its device decode path from the buffers' context lengths.

nsys, code decode, MTP 1 draft:
```
                          eager (Run 57)                      graph
MTP head                  115 launches, 2.14 ms span          108-node graph 0.47 ms span (0.41 ms kernels)
draft lm_head + top-1     (in the above)                      7 launches, 0.72 ms
gap before the head       0.22 ms                             0.39 ms (0.11 ms staging, ~0.09 ms graph launch)
```
About 0.8 ms a step. Acceptance on the real model is unchanged (69-89% over the run against 63-91% eager), and
a unit test replays one shape with changing tokens, positions, hidden states and context lengths against the same
forward run eagerly; skipping the token staging fails it at the first replay. Interleaved A/B (as Run 56, a warm
round before each measured three):
```
                      prose median     code median
master (#644)         52.5             45.2
MTP graph             52.7             46.2
```
+0.5% / +2.2%.
