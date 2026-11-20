# Residual-stream rotation (#1013 item 3)

QuaRot-style, residual only: every RMSNorm gain over the residual stream folded into the projections that read it
(and set to ones), then one orthogonal Q applied to the stream, h' = h Q. Same architecture and GGUF layout, so
llama.cpp, ik_llama.cpp and our engine load the result unchanged. The aim: spread the residual stream's outliers so
the groups writing it (embedding, GDN, attention) quantize with less error; on Qwen3.6-35B-A3B those groups do
74-95% of their harm through flipped expert routes (allocation investigation Run 24).

## Run 1 - 2026-11-20 13:30

Question: is the transform exact, and does the rotated model quantize better?

Setup: Qwen3.6-35B-A3B BF16 GGUF; Q a seeded random-sign Walsh-Hadamard matrix (D H / sqrt(2048)). The GGUF stores
the norms' effective gains (layer 0 `attn_norm` mean 1.03), so a folded gain set to 1 is what llama.cpp computes.

```
readers  W' = W diag(g) Q   attn_qkv, attn_gate, ssm_alpha/beta, attn_q/k/v (attn_norm); router, shared gate,
                            experts' and shared expert's gate/up (post_attention_norm); output (output_norm);
                            MTP eh_proj's [e_norm, h_norm] halves (enorm, hnorm)
writers  W' = Q^T W         ssm_out, attn_output, experts' and shared expert's down, MTP eh_proj's output
embedding E' = E Q
```

`ssm_norm` and the q/k norms act inside head space after the projections, so they stay. The MTP block's
`shared_head_norm` cannot fold exactly (its head is the main one, folded with `output_norm`'s gain); it becomes the
best scalar multiple, 1.114, residual 3.5% of the gain's norm. Hessians come from the original's by algebra: a
reader's input is x diag(1/g) Q, so H' = Q^T diag(1/g) H diag(1/g) Q (251 transformed, the writers' 10,320 linked).
Spike scripts, not kept (gguf-py, tensor by tensor on the GPU in f64, 10 min for 71 GB; the Hessians by the algebra above).

ik `llama-perplexity`, against the unrotated BF16's base:

```
file                                       KLD set A           KLD set B    p99 A / B       same top A
rotated BF16 (re-rounded to BF16)          0.00384                          0.0374          97.53%
v2 recipe, unrotated (allocation Run 29)   0.01399             0.01446      0.1712 / 0.1456 94.60%
v2 recipe, rotated                         0.01552             0.01480      0.1535 / 0.1322 94.63%
```

The rotated model's quantization is 11% / 2% worse in mean, better in the tail (p99 -10%), but the rotated BF16
already sits 0.0038 from the original before any quantization: either BF16 re-rounding of every rotated weight (its
8-bit mantissa; Q8_0 costs the 2B ~0.0015) amplified by the route flips, or a defect in the transform. Next: the
same rotation written as F16 (4x finer rounding), which separates the two.

## Run 2 - 2026-11-20 14:30

Question: is the rotated BF16's 0.0038 floor BF16 rounding, a defect, or this model's sensitivity?

```
file (ik llama-perplexity, experts on the host)          set A     set B
original BF16                                            0.00236   0.00013
gains folded, Q = I, F16 (re-rounded, no rotation)                 0.00341
rotated, F16                                             0.00382   0.00424
rotated, BF16                                            0.00384
```

Set A's base predates the ik build in use (Oct 21) and reads 0.0024 against the original itself, a constant offset
in every set-A number (comparisons within the set stand); set B's, written today, reads 0.0001: the true floor.

Folding the gains alone, re-rounded to F16 (finer than the BF16 they came from) and run on the F16 kernels, costs
0.0034: the route flips turn any whole-model perturbation of the weights into ~0.003-0.004, so the rotation's floor
is not a defect. The file was checked directly: embedding, `attn_qkv` and `ssm_out` equal E Q, W diag(g) Q and Q^T W
to F16 rounding (2e-4 relative), the gains are ones. (The twin stream cannot localize this: a twin starts from the
model's own embeddings.)

So a rotated recipe must be quantized from unrounded weights, the rotation applied inside the quantizer, not through
a BF16/F16 file (F32 would be 140 GB). Before building that: does the rotation lower each group's quantization
error at all? GPTQ's reconstruction report over the calibration (tr(dW H dW^T) / tr(W H W^T), comparable across the
two since both models compute the same function) for the v2 recipe, original against rotated.

## Run 3 - 2026-11-20 15:00

GPTQ reconstruction report (median over each group's tensors; output error relative to output energy over the
calibration, comparable across the two since both compute the same function), v2 recipe:

```
group (type)              original    rotated    rot/orig   Hessian off-diagonal share orig/rot
attn_v (Q8_0)             1.73e-05    1.04e-05   0.607      0.837/0.999
attn_k (Q8_0)             1.07e-05    7.34e-06   0.701      0.837/0.999
attn_q (Q8_0)             1.06e-06    8.66e-07   0.764      0.837/0.999
ffn_gate_shexp (Q8_0)     5.63e-06    4.43e-06   0.795      0.946/0.998
ffn_down_shexp (Q8_0)     1.56e-05    1.38e-05   0.872      0.303/0.303
ffn_up_shexp (Q8_0)       1.35e-05    1.25e-05   0.883      0.946/0.998
ssm_out (Q8_0)            1.70e-05    1.47e-05   0.895      0.882/0.882
attn_output (Q8_0)        1.20e-05    1.04e-05   0.895      0.918/0.918
attn_qkv (Q6_K)           4.67e-05    4.35e-05   0.936      0.820/0.998
output (Q6_K)             8.10e-05    7.86e-05   0.970      0.966/0.997
attn_gate (Q6_K)          1.41e-05    1.50e-05   1.028      0.820/0.998
ffn_down_exps (IQ4_KS)    3.92e-03    3.90e-03   0.993      0.736/0.736
ffn_gate_exps (IQ4_KS)    2.14e-03    2.75e-03   1.241      0.946/0.998
ffn_up_exps (IQ4_KS)      3.49e-03    4.41e-03   1.259      0.946/0.998
```

The rotation does what it is for on the dense groups (errors 3-40% lower; writers, whose inputs do not change, gain
from their outputs' spread too), and costs the routed experts' gate/up 25%, which are 16.8 of the file's 18.6 GiB and
most of its KLD. The residual's few large channels let the importance weights and GPTQ's feedback steer rounding away
from them; rotated, every input channel looks alike (the Hessians go fully dense, off-diagonal share 0.946 -> 0.998)
and that steering is lost. The experts read the same rotated stream as every other reader of their layer, so they
cannot be exempted without an online rotation (a change of architecture).

Residual-only rotation is a net loss on this MoE at these widths (the rotated v2 recipe 0.0155 / 0.0148 against
0.0140 / 0.0145 in Run 1, with ~0.001 of it possibly the re-rounding of Run 2). It may still pay where the dense
groups carry the bits (a dense model), or at 2-3-bit experts, where rotation is reported to help weight-only
quantization most.

## Run 4 - 2026-11-20 16:20

Question: does the experts' penalty shrink at 3 bits, where rotation is reported to help weight-only quantization
most? Run 3's reports again with the routed experts at IQ3_KS (the v2 rules with iq4_ks -> iq3_ks),
original and rotated.

```
group (type)              original    rotated    rot/orig   (IQ4_KS ratio, Run 3)
ffn_gate_exps (IQ3_KS)    9.40e-03    1.21e-02   1.283      1.285
ffn_up_exps (IQ3_KS)      1.53e-02    1.93e-02   1.263      1.262
ffn_down_exps (IQ3_KS)    1.72e-02    1.71e-02   0.995      0.994
```

(dense groups unchanged from Run 3, same types). The penalty is the same 26-28% at 3 bits as at 4: it is the loss
of the importance weights' and GPTQ's steering around the residual's large channels, which does not depend on the
width. No KLD run: the reports already rule the 35B out at both widths.

## Run 5 - 2026-11-20 16:45

Question: does the rotation pay on a dense model? Qwen3.5-2B (`qwen35`, tied embedding), the scripts extended to
dense FFNs (`ffn_gate/up` readers of `post_attention_norm`, `ffn_down` a writer) and to tied heads (an untied
`output.weight` = E diag(g_out) Q written after the embedding). The rotated model is written as F32, so the
quantizer reads unrounded weights; the control is the original with the embedding copied to `output.weight`
(bit-exact), so both files have the same tensors. Fresh KLD bases from the original BF16 with the
current ik build (the 2B's older bases read 0.00065 against the original itself):

```
file (ik llama-perplexity)   wiki      wiki_b
original BF16                0.000331  0.000333   (the stored logits' floor)
untied control               0.000331  0.000333
rotated F32                  0.000331  0.000336
```

Exact. Quantized with the same rules (embedding and head Q6_K, ssm_alpha/beta/conv1d F32, every other block weight
IQ4_KS or IQ3_KS), GPTQ with each file's Hessians (151 bound in both):

```
recipe          original           rotated            same top orig/rot (wiki)
IQ4_KS bulk     0.01960 / 0.02101  0.02053 / 0.02189  92.78 / 92.30
IQ3_KS bulk     0.09310 / 0.09810  0.08870 / 0.09566  85.37 / 85.36
```

Reconstruction (rot/orig, median per group; same at both widths within 0.03): attn_v 0.86, output 0.87, attn_k /
attn_output / ssm_out 0.93-0.95, ffn_down 0.97, attn_q / attn_gate 0.94-0.99; attn_qkv 1.09-1.11, ffn_gate 1.18,
ffn_up 1.22. So the dense FFN's gate/up lose to rotation just as the experts did: the penalty belongs to the
post-attention norm's input, not to MoE routing, and it is the largest group in both models. Net: rotated 4-5% worse
at 4 bits, 2.5-5% better at 3 bits, where the dense groups' own errors are large enough for their gains to outweigh
the FFN's loss.

Conclusion for item 3: residual-only rotation helps the attention-side readers and the writers, hurts the FFN /
expert gate+up (their importance weights and GPTQ feedback lean on the residual's large channels), and nets a loss
at the 4-bit widths the recipes use; a small gain only at a 3-bit bulk on the dense model. Not worth carrying in
the quantizer.
