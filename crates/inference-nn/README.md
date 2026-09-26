# `inference-nn`

Model-facing building blocks of `inference.rs`: layers, attention and its metadata, KV and paged-attention caches,
GDN and MoE, and the CUDA/Metal kernels behind them. Models and the engine live in `inference-core`.
