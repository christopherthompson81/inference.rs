# Third-party provenance

Pristine or trimmed drops live in subdirectories here with their own license files. Code that is adapted and
maintained in-tree lives in `kernels/`; this table records where it came from.

| in-tree path | upstream | revision | license | status |
|---|---|---|---|---|
| `third_party/deepgemm_sm90/` | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 2.6.1 (`include/official`) and the TensorRT-LLM DeepGEMM subset via FlashInfer (`include/deep_gemm`) | `559d79fb` / FlashInfer `4927c0e1` | MIT, Apache-2.0 | `official/` pristine; modified files are marked; see its README, NOTICE and LICENSE files |
| `kernels/cuda/gptq/` | [turboderp/exllamav2](https://github.com/turboderp/exllamav2); `q_gemm.cu` also credits GPTQ-for-LLaMa | not recorded | MIT (exllamav2), Apache-2.0 (GPTQ-for-LLaMa) | `qdq_*` copied (`qdq_3.cuh` assumed from its siblings), `q_gemm` adapted |
| `kernels/cuda/marlin/marlin_kernel.cuh` | [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) via vLLM `gptq_marlin` | not recorded | Apache-2.0 | adapted: per-type TUs, `extern "C"` |
| `kernels/cuda/marlin/marlin/`, `marlin_repack.cu`, `gguf_affine_packed/marlin_gguf_affine_repack.cu` | vLLM marlin | not recorded | Apache-2.0 | adapted |
| `kernels/cuda/mmq_gguf/`, `mmvq_gguf/`, `indexed_moe/` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) ggml-cuda mmq/mma/vecdotq/quantize/mmvq (`indexed_moe` also via candle-kernels) | not recorded | MIT | adapted: standalone, one TU per quant type |
| `kernels/cuda/moe/`, `cutlass_moe/moe_data.cu`, `rotary/`, `blockwise_fp8/blockwise_fp8_cutlass_sm90.cu` | vLLM | not recorded | Apache-2.0 | adapted |
| `kernels/cuda/hqq/hqq.cu` | [mobiusml/hqq](https://github.com/mobiusml/hqq) `hqq_aten_cuda_kernel.cu` | not recorded | Apache-2.0 | adapted |
| `kernels/cuda/bitsandbytes/dequant.cu` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) `kDequantizeBlockwise` / NF4 | not recorded | MIT | adapted |
| `kernels/metal/{bf16,utils,quantized,sort,sort_impl,scan,scan_impl,copy,copy_impl}.metal` | [ml-explore/mlx](https://github.com/ml-explore/mlx) | not recorded | MIT | adapted |
| `kernels/cuda/gemv/` | design follows llama.cpp `mmvf.cu` | not recorded | MIT | adapted |
| `kernels/metal/flash_attn.metal` | llama.cpp ggml-metal | not recorded | MIT | adapted |
| `kernels/metal/sdpa_with_sinks.metal`, `fused_glu.metal` | [huggingface/candle](https://github.com/huggingface/candle) metal kernels | not recorded | MIT OR Apache-2.0 | adapted |

Original to this project: `kernels/cuda/` `afq`, `scalar_fp8`, `blockwise_fp8` (non-CUTLASS), `lora`, `mxfp4`,
`nvfp4_cutlass`, `cutlass_moe/grouped_mm_2x.cu`, `hqq_bitpack`, `moe_grouped`, `ops` (inspired by PyTorch
`Nonzero.cu`), the `*_dummy.cu` stubs, and the remaining `kernels/metal/` files.

The MLX-derived `.metal` headers say "Apache License 2.0"; MLX is MIT-licensed, so those headers need checking.
