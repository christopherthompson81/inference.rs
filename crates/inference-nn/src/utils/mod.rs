pub mod debug;
pub mod memory_usage;
pub mod normal;
pub mod progress;
pub mod tokenizer;
pub mod unvarbuilder;
pub mod varbuilder_utils;

#[doc(hidden)]
#[macro_export]
macro_rules! get_mut_arcmutex {
    ($thing:expr) => {
        loop {
            if let Ok(inner) = $thing.try_lock() {
                break inner;
            }
            // Yield to allow other threads to make progress and release the lock.
            // This prevents deadlock when a spawned async task busy-loops while
            // another task holds the lock across an await point.
            std::thread::yield_now();
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default_fn {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}

/// `true` if built with CUDA (requires Unix) /Metal
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub const fn paged_attn_supported() -> bool {
    true
}

/// `true` if built with CUDA (requires Unix) /Metal
#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub const fn paged_attn_supported() -> bool {
    false
}

/// `true` if a CUDA FlashAttention backend runs here: fattn on Turing or newer, FA3 with its feature.
pub fn using_flash_attn() -> bool {
    #[cfg(feature = "cuda")]
    {
        cfg!(feature = "flash-attn-v3") || inference_fattn::mma_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
