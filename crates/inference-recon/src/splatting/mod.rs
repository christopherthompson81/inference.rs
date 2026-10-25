//! Gaussian splatting from generated views: HY-World 2.0's 3DGS stage (world_gs_trainer) over its `gs_data` layout.

pub mod data;
pub mod lpips;
pub mod ssim;
#[cfg(feature = "cuda")]
pub mod train;
