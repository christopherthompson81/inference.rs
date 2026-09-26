# Third-party provenance

Pristine or trimmed drops and per-upstream NOTICE files live in subdirectories here. Code that is adapted and
maintained in-tree lives in `kernels/`; this table records where it came from.

| in-tree path | upstream | revision | license | status |
|---|---|---|---|---|
| `kernels/cuda/gdn_decode.cu` | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) GDN decode | `4927c0e1` | Apache-2.0 | adapted; see `flashinfer_gdn/` |
| `kernels/cuda/gdn_spec_recurrence.cu` | [vllm-project/vllm](https://github.com/vllm-project/vllm) fused GDN | `c8438a3d` | Apache-2.0 | adapted; see `flashinfer_gdn/` |
| `kernels/cuda/radix_topk.cuh` | FlashInfer top-k | `a0a6b019` | Apache-2.0 | adapted; see `flashinfer_topk/` |
| `third_party/flashinfer_gdn_sm90/` | FlashInfer SM90 delta-rule prefill | `28406af5` | Apache-2.0 | project wrapper over FlashInfer headers fetched at build time; NOTICE in `flashinfer_gdn/` |
| `kernels/cuda/moe_gemm.cu`, `moe_gemm_wmma.cu` | [guoqingbao/attention.rs](https://github.com/guoqingbao/attention.rs) (license unconfirmed) | not recorded | unconfirmed | adapted |
| `kernels/cuda/moe_utils.h` | vLLM | not recorded | Apache-2.0 | adapted |
| `kernels/cuda/moe_gemv.cu`, `ssm.cu` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (ssm-scan, mmv approach) | not recorded | MIT | adapted |
| `kernels/cuda/sort.cu` (bitonic `asort`) | [huggingface/candle](https://github.com/huggingface/candle) `candle-kernels/sort.cu` (llama.cpp argsort lineage) | not recorded | MIT OR Apache-2.0 | adapted; the rmsnorm and top-k parts are original |
| `kernels/metal/` | ports of this crate's `gdn_*.cu` and `ssm.cu` | n/a | as above | adapted |

Everything else in `kernels/cuda/` (other `gdn_*`, `gdn_chunked/`, `dflash_*`, `dynamic_conv`, `graph`,
`indexed_copy`, `input_packing`, `attention_prep`, `speculative_rejection`) is original to this project.
