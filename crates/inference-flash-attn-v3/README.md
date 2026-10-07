> FlashAttention-3 for Hopper (sm90), derived from candle-flash-attn-v3 0.11.0 of
> [candle](https://github.com/huggingface/candle) at rev `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` (MIT OR Apache-2.0,
> see `LICENSE-MIT` and `LICENSE-APACHE`). Changes: renamed, built on inference-tensor instead of candle-core, tests
> not carried over. Outside the workspace's `--workspace` builds (see the root `Cargo.toml`); inference-nn's
> `flash-attn-v3` feature builds it.

# Candle Flash Attention v3 Layer

Flash Attention v3 Layer for Hopper (compatible nvidia `sm90a` arch) and the candle framework. 
