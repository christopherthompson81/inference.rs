# Third-party provenance

| in-tree path | upstream | revision | license | status |
|---|---|---|---|---|
| `hkernel/` | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) `hopper/` (FlashAttention-3), by way of Michael Feil's port and [candle](https://github.com/huggingface/candle) `candle-flash-attn-v3` | candle `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` | BSD-3-Clause (`LICENSE-BSD-3-flash-attention`); the port's own changes MIT OR Apache-2.0 | copied unchanged |
| `build.rs`, `src/` | candle `candle-flash-attn-v3` | `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` | MIT OR Apache-2.0 | adapted: built on inference-tensor instead of candle-core |
