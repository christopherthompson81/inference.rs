pub(crate) mod debug;
pub(crate) mod memory_usage;
pub(crate) mod normal;
pub(crate) mod progress;
pub(crate) mod unvarbuilder;
pub(crate) mod varbuilder_utils;

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

/// `true` if built with the `flash-attn` or `flash-attn-v3` features, false otherwise.
#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
pub const fn using_flash_attn() -> bool {
    false
}

/// `true` if built with the `flash-attn` or `flash-attn-v3` features, false otherwise.
#[cfg(any(feature = "flash-attn", feature = "flash-attn-v3"))]
pub const fn using_flash_attn() -> bool {
    true
}
