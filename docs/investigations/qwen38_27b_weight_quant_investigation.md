# Qwen3.8 27B weight quantization investigation

Goal: a weight quantization of Qwen3.8 27B smaller than the IQ4_XS GGUF in use (Unsloth's, 4.25 bpw for most
tensors) with the same small quality loss, so a 24 GB card keeps more room for the KV cache. Candidates are the
types we already run, then types we could support with work.

## Run 1 - 2026-10-18 23:50

Question: what has been published for this model, and which candidate is likeliest?

Command: web search and the model cards and PRs below; the GGUF's own metadata.

The GGUF in use: `general.quantized_by Unsloth`, `quantize.imatrix.file Qwen3.8-27B-GGUF/imatrix_unsloth.gguf`
(496 entries, 45 chunks of `unsloth_calibration_Qwen3.8-27B.txt`), file type 30 (IQ4_XS). The imatrix is in
https://huggingface.co/unsloth/Qwen3.8-27B-GGUF and is downloaded (13 MB).

Published measurements (each on its own text, engine and reference, so only indicative):

```
EXL3 (yeasah/Qwen3.8-27B-exl3), excess KLD vs the unquantized model, Δ perplexity
  2.00 bpw   9.90 GiB   0.0888   +9.62%
  3.00 bpw  12.73 GiB   0.0178   +1.79%
  4.00 bpw  15.71 GiB   0.0046   +0.37%
  5.00 bpw  18.54 GiB   0.0011   +0.14%
malaiwah/Qwen3.8-27B-K4, KL(BF16 || candidate), one shared BF16 head
  GGUF Q8_0  27.05 GiB  0.001087      turboderp EXL3 6.00 bpw  0.001583      GGUF Q6_K  21.31 GiB  0.002035
  EXL3 5.00 bpw         0.004005      GGUF UD-Q5_K_XL 18.83 GiB  0.004444   official FP8  0.005197
  llama-perplexity --kl-divergence on WikiText-2 (n_ctx 512): Q8_0 0.000926, Q6_K 0.002286, UD-Q5_K_XL 0.004426
ubergarm/Qwen3.8-27B-GGUF, wiki.test.raw, 580 chunks of 512
  BF16 6.9540   Q8_0 6.9554   IQ4_KS mix (4.731 bpw, 15.749 GiB) 6.9938 (+0.55%)
```

No published IQ4_XS or trellis (KT) number for this model. ik's PR 113 (the trellis types): IQ2_KT 2.125,
IQ3_KT 3.125, IQ4_KT 4.0 bpw; his own summary is "a small improvement", about 0.1-0.2 bpw less for the same error
than the i/k/iqk quants. EXL3 (QTIP's trellis with Hadamard rotation and Viterbi search, plus per-tensor bit
allocation) reads as better per bit: its 4.00 bpw is +0.37% perplexity.

Our own 0.8B ladder (gguf_iq_trellis_quants_investigation.md, one 512-token window, mixed embedding types) is too
noisy and too small a model to rank these for a 27B.

Implication: in our lineup the likeliest is IQ4_KT (4.0 bpw vs 4.25, about 0.8 GB less on this model) and, for a
bigger cut, IQ3_K (3.44) or IQ3_KT (3.125). Outside it, EXL3 near 3.5 bpw. Plan: all GGUF candidates from the BF16
weights with Unsloth's imatrix and one tool (ik_llama.cpp: we decode the ik types but have no quantizers), an
IQ4_XS made the same way as the baseline, Unsloth's file as one more row; KL divergence against BF16 on the same
text, then the winner checked in our engine.

## Run 2 - 2026-10-19 09:45

Question: do the candidates share Unsloth's recipe, so that only the bulk type differs?

Command: BF16 weights converted with mainline llama.cpp's `convert_hf_to_gguf.py` (ik_llama.cpp's converter does not
register `Qwen3_5ForConditionalGeneration`); Unsloth's imatrix converted to the legacy `.dat` that ik's quantizer
reads (`llama-imatrix --in-file imatrix_unsloth.gguf --output-format dat`); per-tensor types compared with ik's
gguf-py.

ik's default IQ4_XS recipe (13.88 GiB) differs from Unsloth's (14.62 GiB) where it matters:

```
tensor (count)                      Unsloth          ik default
linear attention attn_qkv (48)      Q5_K 1650 MiB    IQ4_XS 1275 MiB
linear attention ssm_out (48)       Q5_K  990 MiB    IQ4_XS  765 MiB
ssm_alpha / ssm_beta (48 each)      F32               IQ4_XS
attn_v (17)                         Q5_K              IQ5_K (same size)
token_embd                          Q4_K              IQ4_XS
MTP layer (blk.64)                  Q4_K, attn_v Q5_K, nextn.eh_proj Q8_0     IQ4_XS
output                              Q6_K              Q6_K
```

The candidates now take Unsloth's recipe through `--custom-q` (first match wins; the head and embedding rules are
anchored, as an unanchored `output\.weight` also caught `attn_output.weight`) and change only the bulk type. Our
IQ4_XS so made is 14.63 GiB against Unsloth's 14.62. The default-recipe files are kept aside for comparison.

## Run 3 - 2026-10-19 10:30

Question: which bulk type, in Unsloth's recipe, keeps IQ4_XS's quality at a smaller size?

Command: `llama-perplexity` (ik_llama.cpp c446f1e0) on WikiText-2 test, `-c 512 --chunks 40`; BF16 reference logits
saved once (`--kl-divergence-base`, 24 layers offloaded, PPL 5.9281 +/- 0.1408), each candidate fully offloaded
with `--kl-divergence`. RTX 3090, the LAN server stopped.

```
                      size        mean KLD             same top token    99% KLD   PPL(Q)/PPL(base)
Unsloth IQ4_XS        14.62 GiB   0.01750 +/- 0.00047  94.1%             0.176     1.0032
ours IQ4_XS           14.63 GiB   0.01739 +/- 0.00043  94.10%            0.176     1.0028
ours IQ4_KS           14.64 GiB   0.01620              94.48%            0.164     1.0006
ours IQ4_KT           14.05 GiB   0.02082 +/- 0.00071  93.36%            0.200     1.0075
ours IQ4_KSS          14.05 GiB   0.02206              93.41%            0.223     1.0073
ours IQ3_K            12.72 GiB   0.04735              90.77%            0.491     1.0231
ours IQ3_KT           12.01 GiB   0.05700              89.89%            0.558     1.0378
```

Our IQ4_XS matches Unsloth's file, so the tool, imatrix and recipe are reproduced. No smaller bulk type keeps the
quality: the 4.0 bpw types (IQ4_KT, IQ4_KSS) cost 0.58 GiB less for 19-27% more KLD (paired, about 5 standard
errors; one token in ~140 more takes a different top token), the 3-bit types 2.7-3.3x the KLD. IQ4_KS beats
IQ4_XS at the same size (7% less KLD, perplexity within noise of BF16).

Implication: a smaller file at IQ4_XS's quality has to come from the recipe rather than one bulk type: IQ4_KS as
the base, buying back size per tensor where the KLD says it costs least (a per-tensor sensitivity sweep, with the
lower types on the least sensitive tensors). That is also the route to a recipe that leaves room for the full
256K context beside a DFlash drafter. Check IQ4_KS decode speed in our engine before adopting it.

## Run 4 - 2026-10-19 11:30

Question: which tensor groups give back size most cheaply (KLD per GiB) from the IQ4_KS base recipe, toward a
recipe that leaves room for 256K context beside a q4k DFlash 2 drafter under 23.5 GiB (about 1.1 GiB to find:
the 128K setup measures 22,582 MiB, 256K adds ~2,576 MiB of q4 cache)?

Command: `sweep_recipe.sh` (scratchpad): each probe prepends one `--custom-q` rule to the IQ4_KS recipe, quantizes,
and scores as in Run 3; layer quarters are blocks 0-15, 16-31, 32-47, 48-63.

```
probe (from the IQ4_KS recipe)          saves      KLD (base 0.01620)   dKLD/GiB   same top token
ssm_out Q5_K -> IQ4_KS                  0.220 GiB  0.01610              ~0         94.46%
GDN attn_qkv Q5_K -> IQ4_KS             0.365      0.01699              0.002      94.42%
token_embd Q4_K -> IQ3_K                0.158      0.01667              0.003      94.45%
FFN blocks 0-15 -> IQ3_K                0.408      0.01785              0.004      94.04%
output Q6_K -> Q5_K                     0.158      0.01708              0.006      94.21%
FFN blocks 16-31 -> IQ3_K               0.408      0.01982              0.009      93.97%
attn_gate -> IQ3_K                      0.145      0.01772              0.010      94.11%
full attention q/k/output -> IQ3_K      0.165      0.01849              0.014      93.96%
FFN blocks 32-47 -> IQ3_K               0.408      0.02244              0.015      93.22%
FFN blocks 48-63 -> IQ3_K               0.408      0.02889              0.031      92.98%
attn_v Q5_K -> IQ4_KS                   0.014      0.01639              0.014      94.44%
```

With an IQ4_KS bulk, Unsloth's Q5_K on the linear-attention projections (`attn_qkv`, `ssm_out`) buys almost
nothing; FFN sensitivity rises with depth (the last quarter costs 8x the first); the output head and embeddings
are cheap to trim.

Implication: two recipes to verify (additivity is assumed above): A, `ssm_out` and `attn_qkv` at IQ4_KS
(~14.05 GiB, ~0.0169, under IQ4_XS's 0.0174), which reaches 256K with a q4/q3 KV cache if V at 3 bits holds; B,
A plus `token_embd` and FFN blocks 0-15 at IQ3_K (~13.49 GiB, ~0.019), which reaches 256K with the cache at q4/q4.

## Run 5 - 2026-10-19 12:40

Question: do recipes A and B hold their predicted sizes and KLD, does a 3-bit KV cache hold on this model, and
does A fit 256K beside a q4k DFlash 2 drafter under 23.5 GiB?

Command: the sweep script's `recipe-A` / `recipe-B` probes; `ppl_decode.py` (decode-time teacher forcing, KV
investigation Run 6) on recipe A per cache type; model-manager loads with the drafter and Run 3's code workload
from the DFlash investigation; the small-batch matmul bench on the ik types.

```
recipe                                    size        mean KLD   same top token
A (ssm_out, attn_qkv at IQ4_KS)           14.056 GiB  0.01689    94.44%
B (A + token_embd, FFN 0-15 at IQ3_K)     13.491 GiB  0.01916    94.01%
Unsloth IQ4_XS                            14.62 GiB   0.01750    94.1%

recipe A, decode perplexity by KV type    bf16 2.6778   q4 2.6743   q4/q3 2.6705   q3 2.6706   q8/q3 2.6609
```

The sweep's costs add up (A predicted 0.0169, B 0.019). The KV cache at 3 bits for K and V costs nothing
measurable here (the 27B has 16 attention layers; KV investigation Run 11 found K at 3 bits +0.2%).

Recipe A at first decoded slowly in our engine (33-41 tok/s plain, 37-47 with the drafter): IQ4_KS's MMVQ costs
5.8x at 8 rows (IQ4_XS 2x), and `prefers_mmvq` kept unmeasured types on MMVQ to 8 rows. Benched crossovers:

```
ffn_up 17408x5120     1-row MMVQ    MMQ (flat)    MMVQ rows kept
IQ4_KS                60.8 us       90 us         2
IQ4_KSS               58.6          96            2
IQ4_KT                67.7          95            1
IQ3_K                 57.8          91            1
IQ3_KT                73.9          96            1
```

With those cutovers, recipe A decodes at 50.1 tok/s plain (IQ4_XS's rate), and with the q4k DFlash 2 drafter at 7
drafts and a q3 cache it loads the full 262K context in 23,227 MiB and decodes the code workload at 129.5 / 135.9
tok/s (q4/q3 cache: 23,829 MiB).

Implication: recipe A plus a q3 KV cache meets the goal: better than Unsloth's IQ4_XS (KLD 0.0169 vs 0.0175),
0.56 GiB smaller, and the full context with DFlash 2 under 23.5 GiB.
