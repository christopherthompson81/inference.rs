use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, bail};

use super::chunker::chunk_for_synthesis;
use super::g2p;
use super::gguf::{GGUF_EXTENSION, read_kokoro_gguf};
use super::{KokoroConfig, KokoroModel, SAMPLE_RATE, SourceNoise, VoicePack, pth_var_builder};
use crate::{SpeechGenerationOutput, SpeechOptions};

// the reference pipeline's default voice, used when the request names none and the model ships it
const DEFAULT_VOICE: &str = "af_heart";
const VOICE_BLEND_SEPARATOR: char = ',';
// where a chunk boundary sounds natural
const PAUSES: [char; 6] = ['.', '!', '?', ';', ':', ','];
const CHANNELS: usize = 1;

/// Kokoro with its voice packs, as the engine serves it.
pub struct KokoroTts {
    model: KokoroModel,
    voices: BTreeMap<String, VoicePack>,
    default_voice: String,
}

impl KokoroTts {
    /// From the `.pth` release, same-named safetensors, or a Kokoro GGUF (which brings its own config and voices).
    pub fn load(
        config: &Path,
        weights: &Path,
        voices: &[PathBuf],
        device: &Device,
    ) -> Result<Self> {
        if weights.extension().is_some_and(|e| e == GGUF_EXTENSION) {
            let gguf = read_kokoro_gguf(weights, device)?;
            return Self::new(&gguf.config, gguf.weights, gguf.voices);
        }
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
        Self::new(&cfg, vb, packs)
    }

    fn new(
        cfg: &KokoroConfig,
        vb: VarBuilder,
        voices: BTreeMap<String, VoicePack>,
    ) -> Result<Self> {
        let default_voice = if voices.contains_key(DEFAULT_VOICE) {
            DEFAULT_VOICE.to_string()
        } else {
            match voices.keys().next() {
                Some(first) => first.clone(),
                None => bail!("Kokoro needs at least one voice pack"),
            }
        };
        Ok(Self {
            model: KokoroModel::new(cfg, vb)?,
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

    /// `text` cut where Kokoro's context needs it and read as `lang`, one phoneme string per piece.
    fn text_phonemes(&self, text: &str, lang: &str) -> Result<Vec<String>> {
        let in_vocab = |c: char| self.model.in_vocab(c);
        let mut count = |s: &str| {
            Ok(g2p::phonemes(s, lang, in_vocab)?
                .chars()
                .filter(|&c| in_vocab(c))
                .count())
        };
        chunk_for_synthesis(text, &mut count)?
            .iter()
            .map(|piece| g2p::phonemes(piece, lang, in_vocab))
            .collect()
    }

    /// The request errors `generate` would hit, found before any synthesis.
    pub fn validate(&self, options: &SpeechOptions) -> Result<()> {
        if let Some(speed) = options.speed
            && !(speed.is_finite() && speed > 0.)
        {
            bail!("speed must be a positive number, got {speed}")
        }
        self.voice(options.voice.as_deref())?;
        if options.phonemes.is_none() {
            g2p::readable(g2p::voice_language(self.first_voice(options))?)?;
        }
        Ok(())
    }

    // a blend speaks the language of its first voice, as the reference pipeline's language code does
    fn first_voice<'a>(&'a self, options: &'a SpeechOptions) -> &'a str {
        let name = options.voice.as_deref().unwrap_or(&self.default_voice);
        name.split(VOICE_BLEND_SEPARATOR).next().unwrap_or(name)
    }

    /// Speaks the request's `phonemes`, or `text` phonemized for the voice's language, in chunks the context holds.
    pub fn generate(
        &self,
        text: &str,
        options: &SpeechOptions,
        default_speed: f32,
        seed: u64,
    ) -> Result<SpeechGenerationOutput> {
        self.validate(options)?;
        let speed = options.speed.unwrap_or(default_speed);
        let voice = self.voice(options.voice.as_deref())?;
        let pieces = match options.phonemes.as_deref() {
            Some(phonemes) => vec![phonemes.to_string()],
            None => self.text_phonemes(text, g2p::voice_language(self.first_voice(options))?)?,
        };
        if pieces.iter().all(|p| p.trim().is_empty()) {
            bail!("nothing to speak: the input has no phonemes")
        }
        let mut noise = SourceNoise::seeded(seed);
        let mut pcm = Vec::new();
        for piece in &pieces {
            // the text chunker already fits its pieces; a caller's phoneme string is cut here
            for chunk in chunks(piece, self.model.max_phonemes()) {
                pcm.extend(
                    self.model
                        .synthesize(&chunk, &voice, speed, &mut noise)?
                        .audio,
                );
            }
        }
        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: SAMPLE_RATE,
            channels: CHANNELS,
        })
    }
}

/// Words packed into chunks of at most `max` characters, ending after a pause where one fits; a longer word is cut.
fn chunks(text: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let len = |words: &[String]| {
        words
            .iter()
            .map(|w| w.chars().count() + 1)
            .sum::<usize>()
            .saturating_sub(1)
    };
    for word in text.split_whitespace() {
        let mut word: Vec<char> = word.chars().collect();
        while word.len() > max {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current).join(" "));
            }
            out.push(word.drain(..max).collect());
        }
        let word: String = word.into_iter().collect();
        if !current.is_empty() && len(&current) + 1 + word.chars().count() > max {
            let pause = current
                .iter()
                .rposition(|w| w.ends_with(PAUSES))
                .filter(|&i| i + 1 < current.len());
            let rest = pause.map_or_else(Vec::new, |i| current.split_off(i + 1));
            out.push(std::mem::replace(&mut current, rest).join(" "));
            if !current.is_empty() && len(&current) + 1 + word.chars().count() > max {
                out.push(std::mem::take(&mut current).join(" "));
            }
        }
        current.push(word);
    }
    if !current.is_empty() {
        out.push(current.join(" "));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::chunks;

    #[test]
    fn chunks_pack_words_and_cut_overlong_ones() {
        assert_eq!(chunks("ab cd ef", 5), ["ab cd", "ef"]);
        assert_eq!(chunks("ab, cd ef gh", 9), ["ab,", "cd ef gh"]);
        assert_eq!(chunks("a, bcdefg hijklmn", 9), ["a,", "bcdefg", "hijklmn"]);
        assert_eq!(chunks("abcdefgh ij", 3), ["abc", "def", "gh", "ij"]);
        assert_eq!(chunks("  ", 4), Vec::<String>::new());
        assert!(
            chunks("ðə kwˈɪk bɹˈaʊn fˈɑːks", 8)
                .iter()
                .all(|c| c.chars().count() <= 8)
        );
    }
}
