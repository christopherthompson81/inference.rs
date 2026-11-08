# Qwen GGUFs on Vulkan

Goal (user): Qwen models run on the Vulkan backend (3090 here; the 780M iGPU is the other target). Local GGUFs:
Qwen3.5-0.8B Q8_0 (small check), Qwen3.6-35B-A3B UD-IQ4_XS (the MoE the CUDA recipes use).

## Run 1 - 2026-11-08 04:20

Question: what stops them?

- RoPE: `inference_quant::rotary` dispatches CUDA and Metal and otherwise the CPU kernel, which refuses device
  tensors ("RoPE positions must be CPU u32", "unsupported CPU RoPE dtype F32 F32 F32"). Added a composite in tensor
  ops (gather the cos/sin rows, two multiply-adds over the halves or interleaved pairs, the rest passed through) that
  Vulkan takes; CPU tests pin it to the CPU kernel for both layouts, partial rotation, positions and every cache row form.
- Next: "paged decode of 8 heads over 2 kv heads, head size 256" (0.8B) and "16 heads over 2 kv heads, head size 256"
  (35B). Every Vulkan attention kernel stops at head size 128 (paged decode and prefill `MAX_HEAD4 = 32`, the
  matrix-core prefill and flash attention 128).

`--paged-attn off` (tensor-op attention, any head size), `inference bench --vulkan --prompt-len 512 --gen-len 32`:

```
Qwen3.5-0.8B Q8_0: prefill 681 tok/s, decode 20.9 tok/s
```

Greedy chat against CUDA: the first five tokens agree (logprobs within 0.03); token six is a near tie that flips
(" sunlight" -1.31 on CUDA, " Ray" -1.37 here), so the text then differs.

Next: a head-256 paged decode. Its shared memory at MAX_HEAD4 64 is ~49 KB with the 16-head group limit, over the
3090's 48 KB; head 256 allows at most 8 heads a kv head anyway (8 value heads x 256 / 256 threads), so a variant
with MAX_GROUP 8 is 47.2 KB.

## Run 2 - 2026-11-08 04:35

A wide paged decode: `paged_decode_main.glsl` takes `DECODE_MAX_HEAD4` 64 and `DECODE_MAX_GROUP` 8 from
`paged_decode_h256{,_f16,_quant}.comp`; the op allows head size 256 there (128 for prefill, unchanged).
`vulkan_paged_attention_matches_the_cpu` runs heads 8 and 16 over 2 kv heads at 256 too.

`inference bench --vulkan --prompt-len 512,2048 --gen-len 64` (paged, the default):

```
Qwen3.5-0.8B Q8_0:          prefill 704 / 765 tok/s, decode 29.3 tok/s
Qwen3.6-35B-A3B UD-IQ4_XS:  prefill 65 / 84 tok/s,  decode 7.5 tok/s
```

Greedy chats now give CUDA's text (both models; 35B first-token logprob -0.23 vs CUDA -0.10, the rest within ~0.05).

Profile (`INFERENCE_VULKAN_PROFILE=1`, 5 s windows of decode): GPU busy 456 ms (35B) and 501 ms (0.8B) of 5,000.
The 0.8B does ~640 `map_f32` dispatches a token: GDN's recurrence and causal conv have CUDA and Metal kernels
and otherwise run as tensor-op composites (`gdn/backend.rs`), one step at a time, so the host's dispatch
recording bounds decode and prompts loop per token. The 35B's profile has no expert matvecs: its IQ4_XS
experts have no Vulkan kernel and run on the CPU.

Next: GDN on Vulkan (the recurrence and causal conv as kernels), then IQ4_XS expert kernels for the 35B.

## Run 3 - 2026-11-08 05:10

GDN on Vulkan:
- `gdn_recurrence.comp`: the CPU pass's recurrence from the convolved rows, a workgroup a (value head, sequence),
  a thread a value column holding S[:, j] in registers (head_k and head_v up to 128), reading the checked-out
  key-major state and writing the advanced one (the cache commits it back as before).
- The causal conv is Mamba's conv step (`ssm_conv_step`: the state the last `kernel` inputs, no bias, SiLU), in
  place on the checked-out rows, for a prompt as for a step.
- RoPE as a kernel (`rope.comp`, a thread a rotated pair), the composite kept for other dtypes.
Parity tests (each fails when its kernel is broken on purpose): `vulkan_gdn_recurrence_matches_the_cpu` (Qwen3.6's
16/32 heads at 128, both layouts, a step and a 37-token prompt over two sequences), `vulkan_gdn_conv_matches_the_cpu`,
`vulkan_rope_matches_the_cpu_kernel`.

```
Qwen3.5-0.8B Q8_0, prefill (2048) / decode:
  composite RoPE + GDN       765 / 29.3 tok/s
  + GDN recurrence kernel   1189 / 68.0
  + conv step + RoPE kernel 2186 / 89.6
Qwen3.6-35B-A3B UD-IQ4_XS:   97 / 14.7 (from 84 / 7.5)
```

Greedy text matches CUDA's (35B's sixth token is a near tie that flips: " gas" -1.19 here, " molecules" -1.18 on CUDA).

The 35B's experts (gguf-py over the file): IQ3_S 78 tensors 8.38 GiB, IQ4_XS 39 tensors 5.18 GiB, Q6_K 3, Q3_K 2,
Q4_K 1; attention, GDN and shared experts Q8_0. Neither i-quant has a Vulkan kernel, so the experts compute
through the fallback. Next (user): IQ4_XS and IQ3_S kernels.
