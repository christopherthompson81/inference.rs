//! Speech models: Dia, with its DAC decoder and BS.1770 loudness normalization, and Kokoro to speak; Parakeet to
//! transcribe.

use std::sync::Arc;

use inference_nn::{attention, layers, ops, utils as nn_utils};

mod bs1770;
mod dia;
pub mod diarization;
pub mod kokoro;
pub mod nemo;
pub mod parakeet;
pub mod silero;
mod transcript;
pub mod utils;
mod weight_norm;

pub use dia::{DiaConfig, DiaPipeline};
pub use transcript::{TimedText, Transcription, TranscriptionOptions};

#[derive(Clone, Copy, Debug)]
pub enum SpeechGenerationConfig {
    Dia {
        max_tokens: Option<usize>,
        cfg_scale: f32,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    },
    Kokoro {
        speed: f32,
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

    pub fn kokoro_default() -> Self {
        Self::Kokoro { speed: 1. }
    }
}

/// What one speech request asks for beyond its input text; each model reads the fields it understands.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpeechOptions {
    /// A voice name, or several separated by commas to blend them.
    pub voice: Option<String>,
    pub speed: Option<f32>,
    /// The model's own phoneme string, spoken instead of the input text.
    pub phonemes: Option<String>,
    /// Seeds the model's sampling noise; unset draws a fresh seed.
    pub seed: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct SpeechGenerationOutput {
    pub pcm: Arc<Vec<f32>>,
    pub rate: usize,
    pub channels: usize,
}

inference_nn::json_config!(dia::DiaConfig);
