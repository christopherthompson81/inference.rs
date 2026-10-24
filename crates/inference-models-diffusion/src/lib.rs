//! Image generation models: FLUX, with its T5 and CLIP text encoders, and HiDream-O1.

use inference_nn::{attention, layers, utils};

mod clip;
pub mod flux;
pub mod gguf;
pub mod hidream;
pub mod noise;
mod qlinear;
mod t5;
pub mod wan;

pub use inference_nn::model::DiffusionGenerationParams;

inference_nn::json_config!(
    flux::autoencoder::Config,
    flux::model::Config,
    hidream::model::Config
);
