# #247: raw GGUF types (IQ, trellis, IQK) in MoE expert stacks

## Run 1 - 2026-10-08 12:30

Question: what does a raw-type expert stack need, and does the only local MoE checkpoint load once it has it?

Before: `RawGgufTensor::new` rejected any rank-3 tensor ("Iq3S weights must be rank 2, got [256, 512, 2048]"), and
`GgufRawMatMul` had no `gather_forward_raw`. Both MoE fast paths (decode `indexed_moe_fused_decode`, prefill
`fast_mmq::grouped*`) take a `QTensor` and only Candle's types; grouped mmq `_moe` launchers for every raw type are
compiled (`DEFINE_MMQ_MOE_LAUNCHER` in each instance) but not declared in `ffi.rs` or listed in `mmq_moe_launcher`.

Change (first step): rank-3 raw tensors; `RawGgufTensor::experts(ids, dtype)` dequantizes only the selected experts
(CUDA: gathers their row blocks and runs the existing dequantizer; CPU: the Rust dequantizers on their bytes);
`GgufRawMatMul::gather_forward_raw` runs one matmul per selected expert over the inputs routed to it and puts the
outputs back in slot order. `UnquantLinear::gather_forward` was not reusable: it copies one full expert per slot.

Checkpoint: `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (17.7 GB), 40 layers, 256 experts; gate/up experts IQ3_S everywhere, down
experts IQ4_XS in 37 layers and Q6_K in 3. Mainline `llama-perplexity -c 512 --chunks 1 -ngl 99` on README.md
(references from `scripts/build_gguf_references.sh`): PPL = 4.2256, 18 s.

Ours (`examples/rust/advanced/perplexity --gguf ... --llama-cpp-ctx 512`), raw findings:
- It loads now. The auto device map put layers 35-39 on the CPU ("Layers 0-34: cuda[0]", "Layers 35-39: cpu").
- Scoring failed: "moe experts forward: dtype mismatch in matmul, lhs: BF16, rhs: F32". The CPU layers' Q6_K down
  experts take `qtensor_indexed_moe_forward`, whose dequantize fallback built F32 weights for BF16 inputs (CPU QTensors
  report no quantized activation type, so the input stays BF16). Existing bug, fixed by dequantizing to the input dtype.
- The example's GGUF branch never enabled logging, so the first failure printed only "prompt scoring did not return
  logits"; it now calls `with_logging()` as its model-id branch does.
- Load took 7.5 min, mostly system time. `RawGgufTensor::to_device` cloned the CPU bytes before upload; it now uploads
  from them directly.
- Why the CPU offload: the GGUF weight source turns each layer into one integer pack factor and takes the minimum over
  layers (`pack_factor`, gguf/weight_source.rs). IQ4_XS at 4.25 bits floors to 3 instead of 3.76, and the minimum is
  set by the three Q6_K-down layers, so every layer is sized well above its bytes. Not specific to raw types; a
  separate fix (size GGUF layers from their resident bytes).

## Run 2 - 2026-10-08 12:48

Question: with the fixes, does the MoE checkpoint score like mainline?

Command: `target/debug/examples/perplexity --gguf Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --file README.md --llama-cpp-ctx 512`
(CUDA build, same auto map: layers 35-39 on the CPU).

Raw:
- A first attempt sat with one thread at 100% and the GPU at 11-15%: `dequantize_rows` ran serially, and a 512-token
  window selects nearly all 256 experts of each CPU layer (about 4G elements per window). It now runs over rayon.
- PPL = 4.2130 against mainline's 4.2256: 0.30% drift.
- Wall 11m15s (user 3m15s, sys 8m33s). Weights loaded in 10 s (18:37:14 to 18:37:24); the rest is scoring two
  windows. Every call dequantizes the selected experts afresh (on CUDA a gather plus the dequantizer, on the CPU new
  F32 buffers of up to 1 GB per projection), which is the system time. Mainline takes 18 s.

Review of the change: the gather reads the routing back to the host and uploads index tensors, so a CUDA decode graph
capture over it fails and the engine turns graphs off for the session (a warning, correct output). It now dequantizes
one expert at a time (prefill selected nearly all experts at once, up to several GB per projection) and builds the
routed index in one upload.

So the raw expert path is correct on a real checkpoint but slow. Next: grouped mmq for raw experts on CUDA (the
compiled `_moe` launchers), and GGUF layer sizing from resident bytes so the layers stay on the GPU.
