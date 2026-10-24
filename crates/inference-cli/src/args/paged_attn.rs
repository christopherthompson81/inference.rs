//! PagedAttention configuration options
//!
//! Unified design replacing the confusing 5-flag system with clear semantics.

use clap::{Args, ValueEnum};
use inference_api::engine::{PagedCacheLayers, PagedCacheType};
use serde::Deserialize;

/// Cache and attention configuration
#[derive(Args, Clone, Default)]
pub struct CacheOptions {
    #[command(flatten)]
    pub paged_attn: PagedAttentionOptions,
}

/// PagedAttention configuration
#[derive(Args, Clone, Deserialize)]
pub struct PagedAttentionOptions {
    /// PagedAttention mode
    /// - auto: enabled on CUDA, disabled on Metal/CPU (default)
    /// - on: force enable (fails if unsupported)
    /// - off: force disable
    #[arg(long = "paged-attn", default_value = "auto", value_enum)]
    #[serde(default)]
    pub mode: PagedAttnMode,

    /// Allocate KV cache for this context length.
    /// If not specified, defaults to using 90% of available VRAM.
    #[arg(long = "pa-context-len")]
    pub context_len: Option<usize>,

    /// GPU memory to allocate in MBs (alternative to context-len)
    #[arg(long = "pa-memory-mb", conflicts_with = "context_len")]
    pub memory_mb: Option<usize>,

    /// GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb)
    #[arg(long = "pa-memory-fraction", conflicts_with_all = ["context_len", "memory_mb"])]
    pub memory_fraction: Option<f32>,

    /// Tokens per block (default: 32 on CUDA)
    #[arg(long = "pa-block-size")]
    pub block_size: Option<usize>,

    /// KV cache storage: auto (the model's dtype, or q3 when that buys GPU room for offloaded MoE experts), none (the
    /// model's dtype), f8e4m3, q2..q8 (K and V at those bits), or qK/qV such as q8/q4
    #[arg(long = "pa-cache-type", default_value = "auto", value_parser = parse_cache_type)]
    #[serde(default)]
    pub cache_type: PagedCacheType,

    /// Per-layer KV cache storage on top of --pa-cache-type: layers=type items such as 0=q8,20-27=q6, or uniform
    /// (the type everywhere, without the model's profile or the general rule)
    #[arg(long = "pa-cache-layers", value_parser = parse_cache_layers)]
    #[serde(default)]
    pub cache_layers: Option<PagedCacheLayers>,
}

impl Default for PagedAttentionOptions {
    fn default() -> Self {
        Self {
            mode: PagedAttnMode::Auto,
            context_len: None,
            memory_mb: None,
            memory_fraction: None,
            block_size: None,
            cache_type: PagedCacheType::Auto,
            cache_layers: None,
        }
    }
}

/// PagedAttention operation mode
#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PagedAttnMode {
    /// Automatic: enabled on CUDA, disabled on Metal/CPU
    #[default]
    Auto,
    /// Force enable (error if device doesn't support it)
    On,
    /// Force disable
    Off,
}

impl PagedAttentionOptions {
    /// The mode (None for auto), memory in MB, memory fraction, context length, block size and cache type.
    pub fn into_builder_flags(self) -> PagedAttnBuilderFlags {
        let enable = match self.mode {
            PagedAttnMode::Auto => None,
            PagedAttnMode::On => Some(true),
            PagedAttnMode::Off => Some(false),
        };

        (
            enable,
            self.memory_mb,
            self.memory_fraction,
            self.context_len,
            self.block_size,
            self.cache_type,
            self.cache_layers.unwrap_or_default(),
        )
    }
}

fn parse_cache_type(s: &str) -> Result<PagedCacheType, String> {
    s.parse()
}

fn parse_cache_layers(s: &str) -> Result<PagedCacheLayers, String> {
    s.parse()
}

/// PagedAttention builder flags type alias
pub type PagedAttnBuilderFlags = (
    Option<bool>,     // paged_attn enable flag
    Option<usize>,    // gpu_mem (MBs)
    Option<f32>,      // gpu_mem_usage (fraction)
    Option<usize>,    // context_len
    Option<usize>,    // block_size
    PagedCacheType,   // cache_type
    PagedCacheLayers, // cache_layers
);
