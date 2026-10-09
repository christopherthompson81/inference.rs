use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, bail};

use super::{KokoroConfig, KokoroModel, SAMPLE_RATE, SourceNoise, VoicePack, pth_var_builder};
use crate::{SpeechGenerationOutput, SpeechOptions};

// the reference pipeline's default voice, used when the request names none and the model ships it
const DEFAULT_VOICE: &str = "af_heart";
const VOICE_BLEND_SEPARATOR: char = ',';
const CHANNELS: usize = 1;

/// Kokoro with its voice packs, as the engine serves it.
pub struct KokoroTts {
    model: KokoroModel,
    voices: BTreeMap<String, VoicePack>,
    default_voice: String,
}

impl KokoroTts {
    /// `weights` is the release `.pth` or a safetensors file with the same names; `voices` are `.pt` or raw `.bin` packs.
    pub fn load(
        config: &Path,
        weights: &Path,
        voices: &[PathBuf],
        device: &Device,
    ) -> Result<Self> {
        let cfg: KokoroConfig = serde_json::from_str(&std::fs::read_to_string(config)?)
            .map_err(inference_tensor::Error::wrap)?;
        let vb = if weights.extension().is_some_and(|e| e == "safetensors") {
            // SAFETY: the weight file is mmapped read-only and not modified while the model is alive.
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? }
        } else {
            pth_var_builder(weights, device)?
        };
        let mut packs = BTreeMap::new();
        for path in voices {
            let name = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let pack = match path.extension().and_then(|e| e.to_str()) {
                Some("pt") => VoicePack::from_pt(path)?,
                _ => VoicePack::from_f32_le(&std::fs::read(path)?)?,
            };
            if packs.insert(name.clone(), pack).is_some() {
                bail!("two voice packs are named `{name}`")
            }
        }
        let voices = packs;
        let default_voice = if voices.contains_key(DEFAULT_VOICE) {
            DEFAULT_VOICE.to_string()
        } else {
            match voices.keys().next() {
                Some(first) => first.clone(),
                None => bail!("Kokoro needs at least one voice pack"),
            }
        };
        Ok(Self {
            model: KokoroModel::new(&cfg, vb)?,
            voices,
            default_voice,
        })
    }

    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// One voice, or the mean of a comma-separated list as the reference blends them.
    fn voice(&self, name: Option<&str>) -> Result<VoicePack> {
        let name = name.unwrap_or(&self.default_voice);
        let packs = name
            .split(VOICE_BLEND_SEPARATOR)
            .map(|v| match self.voices.get(v.trim()) {
                Some(pack) => Ok(pack.clone()),
                None => bail!(
                    "unknown voice `{}`; this model has {}",
                    v.trim(),
                    self.voices.keys().cloned().collect::<Vec<_>>().join(", ")
                ),
            })
            .collect::<Result<Vec<_>>>()?;
        VoicePack::mean(&packs)
    }

    /// The request errors `generate` would hit, found before any synthesis.
    pub fn validate(&self, options: &SpeechOptions) -> Result<()> {
        if options
            .phonemes
            .as_deref()
            .is_none_or(|p| p.trim().is_empty())
        {
            bail!("Kokoro reads phonemes, not text: pass `phonemes`")
        }
        if let Some(speed) = options.speed
            && !(speed.is_finite() && speed > 0.)
        {
            bail!("speed must be a positive number, got {speed}")
        }
        self.voice(options.voice.as_deref()).map(|_| ())
    }

    /// Speaks the request's phonemes; longer input is split at spaces into chunks the model's context holds.
    pub fn generate(
        &self,
        options: &SpeechOptions,
        default_speed: f32,
        seed: u64,
    ) -> Result<SpeechGenerationOutput> {
        self.validate(options)?;
        let phonemes = options.phonemes.as_deref().unwrap_or_default();
        let speed = options.speed.unwrap_or(default_speed);
        let voice = self.voice(options.voice.as_deref())?;
        let mut noise = SourceNoise::seeded(seed);
        let mut pcm = Vec::new();
        for chunk in chunks(phonemes, self.model.max_phonemes()) {
            pcm.extend(
                self.model
                    .synthesize(&chunk, &voice, speed, &mut noise)?
                    .audio,
            );
        }
        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: SAMPLE_RATE,
            channels: CHANNELS,
        })
    }
}

/// Greedy packing of whitespace-separated words into chunks of at most `max` characters; a longer word is cut.
fn chunks(text: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let mut word: Vec<char> = word.chars().collect();
        while word.len() > max {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.push(word.drain(..max).collect());
        }
        let word: String = word.into_iter().collect();
        let len = current.chars().count();
        if len > 0 && len + 1 + word.chars().count() > max {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(&word);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::chunks;

    #[test]
    fn chunks_pack_words_and_cut_overlong_ones() {
        assert_eq!(chunks("ab cd ef", 5), ["ab cd", "ef"]);
        assert_eq!(chunks("abcdefgh ij", 3), ["abc", "def", "gh", "ij"]);
        assert_eq!(chunks("  ", 4), Vec::<String>::new());
        assert!(
            chunks("ðə kwˈɪk bɹˈaʊn fˈɑːks", 8)
                .iter()
                .all(|c| c.chars().count() <= 8)
        );
    }
}
