# Adapters from merged fine-tunes

Many fine-tunes are published as full weights with a LoRA merged into the base. This log measures how far such
checkpoints are from their bases, and checks `scripts/lora_from_merged.py`, which recovers the adapter.

## Run 1 - 2026-10-28 09:50

Question: how far are the decision checkpoints from a stock base, i.e. could each be served as a small adapter on a
shared base (as the OmniVoice IPA patch was: rank-16 LoRA factors plus 5416 changed embedding rows, 31 MB on a
2.45 GB base)?

Command: the Hub listing of each repo (adapter configs, file sizes); `wdiff/fetch.py` pulls single tensors by HTTP
range from a repo and its declared base; `wdiff/rank.py` reports `||delta|| / ||W||` and what a rank-r truncation of
the delta leaves, relative to the tuned weight.

Finding:
- Released as adapters: kev 0.8B / 4B / 9B (rank 16, on the Qwen3.5 *Base* models; 43 / 130 / 173 MB, plus a 2-8 MB
  `head.pt`), nimble 9B (rank 64 on Qwen3.5-9B, 693 MB). Every linear is targeted, the GDN projections included.
- Released as full weights: clef-flash (19.1 GB + a 244 MB joint head), h2o-lightning-4b (9.3 GB), d1-3B (6.3 GB),
  granite-switch-4.1-8b (17.6 GB), kev-27b (51 GB).
- clef-flash against Qwen3.5-9B, three tensors (an attention k_proj, a GDN out_proj, an MLP down_proj): the delta is
  5-7% of the weight, 97% of the values change. Left after rank r, relative to the weight: r16 3.5-3.9%, r64 2.4-2.6%,
  r128 1.5-1.7%, r256 0.14-0.16%, r512 0.09-0.15%. The fall at exactly 256 to the bf16 rounding floor (~0.15%) says a
  rank-256 LoRA merged into the base.
- h2o-lightning-4b against Qwen3.5-4B (k_proj, down_proj): delta 6-7%; r16 3.3-3.5%, r64 0.16%, then flat: a merged
  rank-64 LoRA.
- Not measured: d1-3B (base LFM2.5-VL-3B), granite-switch, kev-27b, and the embeddings / LM heads of any of them.

Implication: these are stock bases plus LoRAs, exactly recoverable (to bf16 rounding) by SVD of the delta at the
fall-off rank. In bf16 an adapter costs about 10.8 MB per rank on a 9B and 8.1 MB on a 4B (from nimble's and kev's
files), so clef as an adapter is ~2.8 GB beside its 9B base (15% of its full weights) and h2o ~0.5 GB (6%). Served
on a base already loaded for chat, each decision model would add only that (less in Q8), but only where the base is
the same checkpoint: kev's are the pretrain bases, not the instruct models.

## Run 2 - 2026-10-28 10:05

Question: does `scripts/lora_from_merged.py` find these ranks from the Hub without downloading, and does an extracted
adapter on its base reproduce the fine-tune in inference.rs?

Command: `lora_from_merged.py measure <base> <tuned> --tensors 4` for three pairs over the Hub (range reads only).
End to end on a small pair: kev-0.8b's published rank-16 adapter merged into Qwen3.5-0.8B-Base in bf16 (scratch
`merge_kev.py`) as a stand-in for a full-weight release; `measure` and `extract --rank 16` on it; then `inference
serve` of the merged model, and of the base with `--lora ext=<extracted>`, compared on three prompts by greedy
24-token completions with logprobs (scratch `lora_cmp.py`).

Finding:
- Hub measure, ~18 s a pair: clef-flash vs Qwen3.5-9B knee 256, at most 1.64 GB as an adapter (19.06 GB full);
  h2o-lightning-4b vs Qwen3.5-4B knee 64, at most 0.32 GB (9.32 GB); d1-3B vs LFM2.5-VL-3B no knee (delta 26-34% of
  the weight, decaying smoothly to 15-25% at r512): a full fine-tune. The vision towers of all three are unchanged.
  These sizes, from the real shapes, replace Run 1's estimates (2.8 / 0.5 GB).
- First measure of the kev stand-in said "no drop": its rank is the smallest tried, so no lower rank showed the fall.
  Rank 0 (the whole delta) is now the starting point; it then reports rank 16 (0.03 GB against 1.75 GB).
- Extract: 186 matrices, worst residual 0.0016 (the bf16 floor), 21.7 MB.
- Served: same greedy text as the merged model on all three prompts (the plain base differs on all three), logprobs
  within 0.069.
- kev's *published* adapter does not load on that base: `adapter contains tensors not consumed by registered LoRA
  sites`. It names modules from the text model (`layers.0.mlp.down_proj`), while the checkpoint loads under
  `model.language_model.`. `lora_from_merged.py rename --prefix model.language_model.` rewrites it; renamed, it gives
  the merged model's text on all three prompts, logprobs within 0.096.

Implication: measure, extract and rename work on Qwen3.5 (GDN projections included). An adapter trained on the
text model alone needs the rename for a full checkpoint; whether the runtime should map such names itself is open.
