# #253: built-in MTP for Qwen3.5-MoE GGUF

## Run 1 - 2026-10-08 15:17

Question: can a text-only `qwen35moe` GGUF load through the Qwen3.5 text model (which has the built-in MTP head)
instead of Qwen3Next (which has none), and does MTP drafting then work on a real checkpoint?

Before: `qwen35moe` resolved to `NormalLoaderType::Qwen3Next` (`build_qwen3_next`, generic bindings), which never set
`mtp_num_hidden_layers` and skipped the `blk.N` nextn blocks. Multimodal `qwen35moe` (with an mmproj) already used the
Qwen3.5 MoE model and `bind_mtp`. The Qwen3.5 text model handles MoE already (`qwen3_5_moe` re-exports it).

Change: a `Qwen3_5Moe` normal loader (`Qwen3_5MoeForCausalLM`, `qwen3_5_moe_text`, `Qwen3_5MoeTextLoader` generated
with the dense one by one macro, expert ISQ patterns as the multimodal loader has); `build_qwen35_moe` (Qwen3.5 text
config plus expert fields; the shared expert length stands in for the absent dense `feed_forward_length`, as for
Qwen3Next); `build_qwen35_text_bindings` takes `qwen35moe`; external HF `qwen3_5_moe` configs resolve to the new
loader too, which drops the Qwen3Next-only normalization.

Checkpoint: unsloth's `Qwen3.6-35B-A3B-MTP-GGUF` UD-IQ4_XS (18.2 GB, sha256 `df27a780...`, matches the hub's LFS
record), which keeps block 40 as its nextn layer (`nextn_predict_layers` 1, `blk.40.nextn.{eh_proj,enorm,hnorm,
shared_head_norm}` plus a full MoE block). It replaced the stripped UD-IQ4_XS from the same publisher (`block_count`
40, no nextn).

Raw, first on the stripped file through the new route (no MTP): PPL 4.2493 (mainline 4.2256, 0.56%; the Qwen3Next
route gave 4.2335); `inference bench --prompt-len 512 --gen-len 128`: prefill 3978 tok/s (same), decode 107.5 tok/s
(Qwen3Next route 75.4).

On the MTP file, `cargo nextest run --profile deep --features cuda -E 'test(moe_gguf_builtin_mtp)'`
(`INFERENCE_TEST_QWEN3_5_MOE_GGUF`): both prompts keep their greedy ids (40 of 40, 34 of 34), MTP accepted 45 of 48
drafts (0.94, per position 23 / 22), 27 s. Bench on the same file:

| | prefill 512 tok/s | decode tok/s |
|---|---|---|
| plain | 3877 | 104.4 |
| `--mtp --mtp-n-predict 2` | 3753 | 184.0 |

(The bench's own prompt; mainline's `llama-bench` 152 tok/s is plain decode with no server and a different prompt,
so it does not compare with the MTP row. Run 2 compares like for like.)

## Run 2 - 2026-10-08 15:27

Question: how does our built-in MTP compare with mainline llama.cpp's on the same file and requests?

Mainline at the pinned `4a8993735` has `--spec-type draft-mtp` (`common_speculative_impl_draft_mtp`: one trained head
for qwen35 / qwen35moe, read from the model's nextn layer). Commands: `llama-server -m <file> -ngl 99 -c 4096 -np 1
[--spec-type draft-mtp --spec-draft-n-max 2]` and `inference serve --format gguf ... --max-seqs 1 [--mtp
--mtp-n-predict 2]`, then the same three chat requests to each (greedy, 256 tokens max, thinking off), reading
`timings.predicted_per_second` and `usage.avg_compl_tok_per_sec`. RTX 3090.

| Prompt | mainline plain | ours plain | mainline MTP | ours MTP | mainline drafts accepted |
|---|---|---|---|---|---|
| count to fifty | 132.4 | 109.0 | 187.2 | 200.9 | 84 / 88 |
| lighthouse story | 135.9 | 111.4 | 168.4 | 153.4 | 141 / 227 |
| hash table | 131.6 | 111.1 | 183.5 | 196.2 | 160 / 190 |

With MTP the two are level (ours ahead on the predictable prompts, behind on the story). Plain decode trails mainline
by about 20%; that is the raw-MoE decode kernel gap (#373), which MTP's verify batches hide.

Review of the change, and what changed: the dead Qwen3Next branches for `qwen35moe` went (the tiled-layout insert,
the split beta/alpha bindings and norm arm, the adapter's layout entry), with the owner's go-ahead the pieces that kept
old artifacts persisted as `Qwen3NextForCausalLM` loading went too (the LoRA site prefix alias, only used for them, and
Qwen3Next's `gdn_v_head_layout` field); the text loaders share the multimodal loader's ISQ patterns, which cover the
MTP layers the text loader's own patterns had missed (dense too), and the MoE text loader joins the loader test tables.
`--arch qwen3next` on a `qwen35moe` file now fails as incompatible; `qwen3_5moe` is its name.
