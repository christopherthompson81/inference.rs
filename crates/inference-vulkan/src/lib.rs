//! Vulkan compute: a device, its buffers (mapped host RAM on a unified-memory iGPU, which also reads host allocations
//! imported without a copy; the device's own memory on a discrete GPU), compute pipelines from SPIR-V compiled at
//! build time, and the kernels on them. Everything is behind the `vulkan` feature; without it the crate is empty.
#![cfg(feature = "vulkan")]

mod context;
mod kernels;
mod map;
mod memory;
mod ops;
mod pipeline;
mod stream;
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

pub use context::{Context, Recorder};
pub use kernels::{BlockType, MATVEC_MAX_ROWS, MatVec};
pub use map::{MAX_DIMS, Map, MapOp, Operand};
pub use memory::{Bound, Buffer, ImportedMemory, f32_bytes};
pub use ops::{
    At, CacheShape, ConvStep, FlashAttn, MatMul, MatOperand, MoeMatVec, PagedDecode, PagedPrefill,
    PagedWrite, RankedTopK, Router, RouterFn, RowOp, Rows, ScanStep, Select,
};
pub use stream::Retained;

pub use ash::vk;

#[derive(Debug)]
pub enum Error {
    /// The Vulkan loader would not open.
    Load(String),
    Vk(vk::Result),
    /// No device takes compute work with the extensions the crate needs.
    NoDevice,
    NoMemoryType,
    /// An argument's bounds or alignment the kernels cannot take.
    Invalid(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(err) => write!(f, "the Vulkan loader would not open: {err}"),
            Self::Vk(err) => write!(f, "Vulkan call failed: {err}"),
            Self::NoDevice => write!(
                f,
                "no Vulkan device with compute, host-memory import and push descriptors"
            ),
            Self::NoMemoryType => write!(f, "no Vulkan memory type fits the allocation"),
            Self::Invalid(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<vk::Result> for Error {
    fn from(err: vk::Result) -> Self {
        Self::Vk(err)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
