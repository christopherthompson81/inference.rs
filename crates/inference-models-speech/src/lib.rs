//! Speech generation models: Dia, with its DAC decoder and BS.1770 loudness normalization.

use std::sync::Arc;

use inference_nn::{attention, layers, ops, utils as nn_utils};

mod bs1770;
mod dia;
pub mod utils;

pub use dia::{DiaConfig, DiaPipeline};

#[derive(Clone, Copy, Debug)]
pub enum SpeechGenerationConfig {
    Dia {
        max_tokens: Option<usize>,
        cfg_scale: f32,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    },
}

impl SpeechGenerationConfig {
    pub fn dia_default() -> Self {
        Self::Dia {
            max_tokens: None,
            cfg_scale: 3.,
            temperature: 1.3,
            top_p: 0.95,
            top_k: Some(35),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeechGenerationOutput {
    pub pcm: Arc<Vec<f32>>,
    pub rate: usize,
    pub channels: usize,
}

inference_nn::json_config!(dia::DiaConfig);
