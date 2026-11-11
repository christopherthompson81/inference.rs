//! Speech models: Dia, with its DAC decoder and BS.1770 loudness normalization, Kokoro and Qwen3-TTS to speak; Parakeet and
//! Whisper to transcribe.

use std::sync::Arc;

use inference_nn::{attention, layers, ops, utils as nn_utils};

mod bs1770;
mod dia;
pub mod diarization;
pub mod ecapa;
pub mod kokoro;
pub mod nemo;
pub mod parakeet;
pub mod qwen3_tts;
pub mod silero;
mod transcript;
pub mod utils;
pub mod voxlingua;
mod weight_norm;
pub mod whisper;

pub use dia::{DiaConfig, DiaPipeline};
pub use transcript::{
    ProgressFn, TRANSCRIPTION_STOPPED, TimedText, Transcription, TranscriptionOptions,
    TranscriptionProgress,
};

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
    Qwen3Tts {
        sampling: qwen3_tts::Sampling,
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

    pub fn qwen3_tts_default() -> Self {
        Self::Qwen3Tts {
            sampling: qwen3_tts::Sampling::default(),
        }
    }
}

/// What one speech request asks for beyond its input text; each model reads the fields it understands.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, inference_json::FromJson)]
pub struct SpeechOptions {
    /// A voice name, or several separated by commas to blend them.
    pub voice: Option<String>,
    pub speed: Option<f32>,
    /// The model's own phoneme string, spoken instead of the input text.
    pub phonemes: Option<String>,
    /// Seeds the model's sampling noise; unset draws a fresh seed.
    pub seed: Option<u64>,
    /// How to speak, or for a voice design model the voice itself (Qwen3-TTS).
    pub instructions: Option<String>,
    /// The language to speak, by name (Qwen3-TTS); unset lets the model pick.
    pub language: Option<String>,
    /// A clip of the voice to speak in, for a model that clones voices (Qwen3-TTS Base).
    #[serde(skip)]
    pub reference: Option<Arc<SpeechReference>>,
}

/// A voice to clone: a clip of it, and optionally what the clip says.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechReference {
    /// Mono samples.
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    /// The clip's transcript; with it the model continues the clip, without it it takes the voice alone.
    pub text: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SpeechGenerationOutput {
    pub pcm: Arc<Vec<f32>>,
    pub rate: usize,
    pub channels: usize,
    /// The input's words in order with when each is spoken, for a model that reads text and can tell (Kokoro).
    pub words: Option<Vec<SpokenWord>>,
}

/// A word of the input and when it is spoken, in seconds from the start of the audio.
#[derive(Clone, Debug, PartialEq)]
pub struct SpokenWord {
    pub text: String,
    /// The word's canonical IPA, as the phonemizer read it; empty when it can't be placed on the word.
    pub ipa: String,
    pub start: f64,
    pub end: f64,
}

inference_nn::json_config!(dia::DiaConfig);
