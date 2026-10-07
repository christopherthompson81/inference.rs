# Third-party provenance

The kernels in `src/` are candle-kernels' (see `../README.md` for what changed), which in turn carry code from:

| in-tree path | upstream | revision | license | status |
|---|---|---|---|---|
| `src/` | [huggingface/candle](https://github.com/huggingface/candle) `candle-kernels/src/` | `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` | MIT OR Apache-2.0 | adapted (see `../README.md`) |
| parts of the elementwise kernels | [coreylowman/dfdx](https://github.com/coreylowman/dfdx), via candle | not recorded | MIT OR Apache-2.0 | adapted |
| `src/quantized.cu`, the softmax in `src/reduce.cu` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml-cuda`, via candle | not recorded | MIT | adapted |
