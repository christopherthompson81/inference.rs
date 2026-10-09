#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvCacheScales {
    pub k: f32,
    pub v: f32,
}

pub const DEFAULT_FP8_KV_CACHE_SCALES: KvCacheScales = KvCacheScales { k: 1.0, v: 1.0 };

/// Head sizes Metal's paged decode kernel is instantiated for.
pub const METAL_PAGED_HEAD_SIZES: [usize; 8] = [64, 80, 96, 112, 128, 192, 256, 512];

#[cfg(all(feature = "cuda", target_family = "unix"))]
mod cuda;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use cuda::*;

#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use metal::*;
