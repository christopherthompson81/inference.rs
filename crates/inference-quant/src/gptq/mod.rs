#[cfg(feature = "cuda")]
mod ffi;
#[cfg(not(feature = "cuda"))]
mod gptq_cpu;
#[cfg(feature = "cuda")]
mod gptq_cuda;
#[cfg(feature = "cuda")]
mod marlin_backend;
#[cfg(feature = "cuda")]
mod marlin_ffi;

#[cfg(not(feature = "cuda"))]
pub use gptq_cpu::{GptqLayer, gptq_linear};
#[cfg(feature = "cuda")]
pub use gptq_cuda::{GptqLayer, gptq_linear};
