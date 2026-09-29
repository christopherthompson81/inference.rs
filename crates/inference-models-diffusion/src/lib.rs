//! Image generation models: FLUX, with its T5 and CLIP text encoders.

use inference_nn::{attention, layers, utils};

mod clip;
pub mod flux;
pub mod gguf;
mod qlinear;
mod t5;

pub use inference_nn::model::DiffusionGenerationParams;

inference_nn::json_config!(flux::autoencoder::Config, flux::model::Config);
