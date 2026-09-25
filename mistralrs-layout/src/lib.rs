//! Document layout detection models built on candle.

mod cpu_conv;
mod cpu_direct;
#[cfg(feature = "cuda")]
mod cuda_kernels;
mod depthwise;
mod im2col;
mod layers;
mod mask_box;
mod msda;
pub mod pp_doclayout_v3;

pub use pp_doclayout_v3::{LayoutDetection, PPDocLayoutV3Detector};
