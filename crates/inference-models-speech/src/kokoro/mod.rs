//! Kokoro-82M: a StyleTTS2-style text-to-speech model with an iSTFTNet vocoder, driven by phoneme strings.
//!
//! See: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and its reference [kokoro](https://github.com/hexgrad/kokoro).

mod albert;
mod config;
mod dsp;
mod istftnet;
mod lstm;
mod prosody;
mod tts;
mod weights;

use std::collections::HashMap;
use std::path::Path;

use inference_tensor::nn::{Linear, Module, VarBuilder, linear};
use inference_tensor::{Device, Result, Tensor};

pub use config::KokoroConfig;
pub use dsp::{HARMONICS, SAMPLE_RATE, SourceNoise};
pub use tts::KokoroTts;
pub use weights::{VoicePack, pth_var_builder};

use albert::Albert;
use istftnet::Decoder;
use prosody::{ProsodyPredictor, TextEncoder};

// the token that brackets every phoneme sequence
const PAD_ID: u32 = 0;

/// One synthesis: 24 kHz mono samples and the frames each input token was held for.
#[derive(Debug, Clone)]
pub struct KokoroOutput {
    pub audio: Vec<f32>,
    pub durations: Vec<u32>,
}

pub struct KokoroModel {
    bert: Albert,
    bert_encoder: Linear,
    predictor: ProsodyPredictor,
    text_encoder: TextEncoder,
    decoder: Decoder,
    vocab: HashMap<char, u32>,
    context_length: usize,
    style_dim: usize,
    device: Device,
}

impl KokoroModel {
    pub fn new(cfg: &KokoroConfig, vb: VarBuilder) -> Result<Self> {
        let vocab = cfg
            .vocab
            .iter()
            .filter_map(|(k, &v)| {
                let mut chars = k.chars();
                match (chars.next(), chars.next()) {
                    (Some(c), None) => Some((c, v)),
                    _ => None,
                }
            })
            .collect();
        Ok(Self {
            bert: Albert::new(&cfg.plbert, cfg.n_token, vb.pp("bert"))?,
            bert_encoder: linear(
                cfg.plbert.hidden_size,
                cfg.hidden_dim,
                vb.pp("bert_encoder"),
            )?,
            predictor: ProsodyPredictor::new(
                cfg.style_dim,
                cfg.hidden_dim,
                cfg.n_layer,
                cfg.max_dur,
                vb.pp("predictor"),
            )?,
            text_encoder: TextEncoder::new(
                cfg.hidden_dim,
                cfg.text_encoder_kernel_size,
                cfg.n_layer,
                cfg.n_token,
                vb.pp("text_encoder"),
            )?,
            decoder: Decoder::new(
                cfg.hidden_dim,
                cfg.style_dim,
                &cfg.istftnet,
                vb.pp("decoder"),
            )?,
            vocab,
            context_length: cfg.plbert.max_position_embeddings,
            style_dim: cfg.style_dim,
            device: vb.device().clone(),
        })
    }

    /// Loads the PyTorch release: `config.json` and the `.pth` checkpoint.
    pub fn from_pth(config: &Path, checkpoint: &Path, device: &Device) -> Result<Self> {
        let cfg: KokoroConfig = serde_json::from_str(&std::fs::read_to_string(config)?)
            .map_err(inference_tensor::Error::wrap)?;
        Self::new(&cfg, pth_var_builder(checkpoint, device)?)
    }

    /// The most phonemes one call takes: the context minus the two bracketing pads.
    pub fn max_phonemes(&self) -> usize {
        self.context_length - 2
    }

    /// Kokoro's ids for `phonemes`; characters outside its vocabulary are dropped, as the reference does.
    pub fn token_ids(&self, phonemes: &str) -> Vec<u32> {
        phonemes
            .chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .collect()
    }

    /// Speaks `phonemes` with `voice`; `speed` scales the predicted durations down.
    pub fn synthesize(
        &self,
        phonemes: &str,
        voice: &VoicePack,
        speed: f32,
        noise: &mut SourceNoise,
    ) -> Result<KokoroOutput> {
        let style = voice.style(phonemes.chars().count())?;
        self.synthesize_ids(&self.token_ids(phonemes), style, speed, noise)
    }

    /// The reference's `forward_with_tokens` on the unpadded ids and one 2 * style_dim style row.
    pub fn synthesize_ids(
        &self,
        ids: &[u32],
        style: &[f32],
        speed: f32,
        noise: &mut SourceNoise,
    ) -> Result<KokoroOutput> {
        if ids.len() > self.max_phonemes() {
            inference_tensor::bail!(
                "{} phonemes, at most {} fit",
                ids.len(),
                self.max_phonemes()
            )
        }
        let padded = [&[PAD_ID], ids, &[PAD_ID]].concat();
        let tokens = padded.len();
        let input = Tensor::new(padded.as_slice(), &self.device)?.unsqueeze(0)?;
        let style = Tensor::new(style, &self.device)?.unsqueeze(0)?;
        let decoder_style = style.narrow(1, 0, self.style_dim)?;
        let prosody_style = style.narrow(1, self.style_dim, self.style_dim)?;

        let d_en = self.bert_encoder.forward(&self.bert.forward(&input)?)?;
        let (d, raw) = self
            .predictor
            .durations(&d_en, &prosody_style, f64::from(speed))?;
        let durations = raw
            .iter()
            .map(|&x| x.round_ties_even().max(1.) as u32)
            .collect::<Vec<_>>();
        let frames = (0..tokens as u32)
            .zip(&durations)
            .flat_map(|(t, &n)| std::iter::repeat_n(t, n as usize))
            .collect::<Vec<_>>();
        let frames = Tensor::new(frames.as_slice(), &self.device)?;

        // the one-hot alignment matmul of the reference is a gather
        let en = d.contiguous()?.index_select(&frames, 1)?;
        let (f0, n) = self.predictor.f0_n(&en, &prosody_style)?;
        let asr = self
            .text_encoder
            .forward(&input)?
            .index_select(&frames, 2)?;
        let audio = self.decoder.forward(&asr, &f0, &n, &decoder_style, noise)?;
        Ok(KokoroOutput { audio, durations })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
