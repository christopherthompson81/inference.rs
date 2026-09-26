# Third-party provenance

Pristine or trimmed drops live in subdirectories here with their own license files. Code that is adapted and
maintained in-tree lives in `kernels/`; this table records where it came from.

| in-tree path | upstream | revision | license | status |
|---|---|---|---|---|
| `third_party/flash-attention/hopper/` | FlashAttention-3 from [vllm-project/flash-attention](https://github.com/vllm-project/flash-attention) | `f3e1a4f7` (CUTLASS `62750a2b`) | BSD-3-Clause | trimmed subset; see its README and LICENSE |
| `kernels/cuda/pagedattention*.cuh`, `kernels/cuda/*pagedattention*.cu`, `kernels/cuda/attention/` | [vllm-project/vllm](https://github.com/vllm-project/vllm) paged attention v1/v2 (FasterTransformer lineage) | not recorded | Apache-2.0 | adapted: per-dtype TUs, `extern "C"` launchers |
| `kernels/cuda/cuda_compat.h`, `reshape_and_cache*`, `gather_kv_cache*`, `copy_blocks*`, `attention/dtype_fp8.cuh`, `quantization/fp8/nvidia/` | vLLM `cache_kernels.cu` and fp8 `quant_utils` | not recorded | Apache-2.0 | adapted: `extern "C"` launchers |
| `kernels/cuda/concat_and_cache_mla_kernel.cu`, `gather_mla_cache_kernel.cu` | vLLM MLA cache kernels (unconfirmed: inferred from names) | not recorded | Apache-2.0 | adapted |
| `kernels/cuda/flashinfer/` | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) headers; `fastdiv.cuh` credits Milakov (Apache-2.0), `fp16.h` credits Marat Dukhan and AMD (MIT) | not recorded | Apache-2.0 (MIT for `fp16.h`) | unmodified as far as known |
| `kernels/metal/utils.metal` | [ml-explore/mlx](https://github.com/ml-explore/mlx) | not recorded | MIT | adapted |
| `kernels/metal/pagedattention.metal` | MLX and vLLM (per file header) | not recorded | MIT (MLX), Apache-2.0 (vLLM) | adapted |
| `kernels/cuda/update_kvscales.cu` | [guoqingbao/attention.rs](https://github.com/guoqingbao/attention.rs) (reference implementation; license unconfirmed) | not recorded | unconfirmed | adapted |
| `kernels/metal/{copy_blocks,reshape_and_cache,gather_kv_cache,kv_scale_update,float8}.metal` | Metal ports of vLLM cache kernels (unconfirmed) | not recorded | Apache-2.0 | adapted |

The MLX-derived `.metal` headers say "Apache License 2.0"; MLX is MIT-licensed, so those headers need checking.

Original to this project: `kernels/cuda/fa3/`, the FlashInfer decode wrappers (`flashinfer_decode*`,
`flashinfer_mla_decode.cu`), and `flash_attn_sinks.cu`.
