use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, bail};

use super::chunker::chunk_for_synthesis;
use super::g2p;
use super::gguf::{GGUF_EXTENSION, read_kokoro_gguf};
use super::{KokoroConfig, KokoroModel, SAMPLE_RATE, SourceNoise, VoicePack, pth_var_builder};
use crate::{SpeechGenerationOutput, SpeechOptions, SpokenWord};

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

    /// `text` cut where Kokoro's context needs it and read as `lang`, one reading per piece.
    fn text_phonemes(&self, text: &str, lang: &str) -> Result<Vec<g2p::Phonemized>> {
        let in_vocab = |c: char| self.model.in_vocab(c);
        let mut count = |s: &str| {
            Ok(g2p::phonemes(s, lang, in_vocab)?
                .chars()
                .filter(|&c| in_vocab(c))
                .count())
        };
        chunk_for_synthesis(text, &mut count)?
            .iter()
            .map(|piece| g2p::phonemize(piece, lang, in_vocab))
            .collect()
    }

    /// Loads the phonemizer's data for the default voice's language, so the first request doesn't wait on it.
    pub fn prepare_text_input(&self) -> Result<()> {
        Self::prepare_language(g2p::voice_language(&self.default_voice)?)
    }

    /// Loads a language's phonemizer data, which may download; slow the first time per process.
    pub fn prepare_language(lang: &str) -> Result<()> {
        g2p::readable(lang)
    }

    /// The language a request's text is read in, when it has text to phonemize in a language the phonemizer reads.
    pub fn text_language(&self, options: &SpeechOptions) -> Option<&'static str> {
        if options.phonemes.is_some() {
            return None;
        }
        g2p::voice_language(self.first_voice(options))
            .ok()
            .filter(|lang| g2p::supported(lang).is_ok())
    }

    /// The request errors `generate` would hit, found before any synthesis and without loading data.
    pub fn validate(&self, options: &SpeechOptions) -> Result<()> {
        if let Some(speed) = options.speed
            && !(speed.is_finite() && speed > 0.)
        {
            bail!("speed must be a positive number, got {speed}")
        }
        self.voice(options.voice.as_deref())?;
        if options.phonemes.is_none() {
            g2p::supported(g2p::voice_language(self.first_voice(options))?)?;
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
            // a caller's phonemes have no source words to time
            Some(phonemes) => vec![g2p::Phonemized {
                phonemes: phonemes.to_string(),
                words: Vec::new(),
                groups: None,
            }],
            None => self.text_phonemes(text, g2p::voice_language(self.first_voice(options))?)?,
        };
        if pieces.iter().all(|p| p.phonemes.trim().is_empty()) {
            bail!("nothing to speak: the input has no phonemes")
        }
        let space = self.model.token_ids(" ").first().copied();
        let mut noise = SourceNoise::seeded(seed);
        let mut pcm = Vec::new();
        let mut words = Vec::new();
        for piece in &pieces {
            let piece_start = pcm.len();
            let mut groups = Vec::new();
            // the text chunker already fits its pieces; a caller's phoneme string is cut here
            for chunk in chunks(&piece.phonemes, self.model.max_phonemes()) {
                let output = self.model.synthesize(&chunk, &voice, speed, &mut noise)?;
                let offset = seconds(pcm.len() - piece_start);
                let ids = self.model.token_ids(&chunk);
                groups.extend(
                    group_spans(&ids, &output.durations, output.audio.len(), space)
                        .into_iter()
                        .map(|(start, end)| (start + offset, end + offset)),
                );
                pcm.extend(output.audio);
            }
            let piece_offset = seconds(piece_start);
            let duration = seconds(pcm.len() - piece_start);
            words.extend(
                word_times(piece, &groups, duration)
                    .into_iter()
                    .map(|mut word| {
                        word.start += piece_offset;
                        word.end += piece_offset;
                        word
                    }),
            );
        }
        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: SAMPLE_RATE,
            channels: CHANNELS,
            words: options.phonemes.is_none().then_some(words),
        })
    }
}

fn seconds(samples: usize) -> f64 {
    samples as f64 / SAMPLE_RATE as f64
}

/// Each spoken group's span in seconds: a group is a run of tokens between spaces, and the predicted durations are
/// scaled to the audio they made. `durations` covers the padding token on each side of `ids`.
fn group_spans(
    ids: &[u32],
    durations: &[u32],
    samples: usize,
    space: Option<u32>,
) -> Vec<(f64, f64)> {
    let total: u32 = durations.iter().sum();
    if total == 0 || durations.len() != ids.len() + 2 {
        return Vec::new();
    }
    let per_unit = seconds(samples) / f64::from(total);
    let mut at = f64::from(durations[0]) * per_unit;
    let mut spans = Vec::new();
    let mut open: Option<f64> = None;
    for (&id, &units) in ids.iter().zip(&durations[1..]) {
        let end = at + f64::from(units) * per_unit;
        if Some(id) == space {
            if let Some(start) = open.take() {
                spans.push((start, at));
            }
        } else if open.is_none() {
            open = Some(at);
        }
        at = end;
    }
    if let Some(start) = open {
        spans.push((start, at));
    }
    spans
}

/// The piece's words timed by the groups mapped onto them, or spread evenly when the map doesn't fit the groups; a word
/// no group reads (punctuation alone) takes no time where it falls.
fn word_times(piece: &g2p::Phonemized, groups: &[(f64, f64)], duration: f64) -> Vec<SpokenWord> {
    let count = piece.words.len();
    let mut spans: Vec<Option<(f64, f64)>> = vec![None; count];
    match &piece.groups {
        Some(map) if map.len() == groups.len() => {
            for (&word, &(start, end)) in map.iter().zip(groups) {
                if let Some(span) = spans.get_mut(word) {
                    *span = Some(span.map_or((start, end), |(s, _)| (s, end)));
                }
            }
        }
        _ => {
            for (i, span) in spans.iter_mut().enumerate() {
                let share = duration / count as f64;
                *span = Some((share * i as f64, share * (i + 1) as f64));
            }
        }
    }
    let mut cursor = 0.0;
    piece
        .words
        .iter()
        .zip(spans)
        .map(|(word, span)| {
            let (start, end) = span.unwrap_or((cursor, cursor));
            cursor = end;
            SpokenWord {
                text: word.text.clone(),
                ipa: word.ipa.clone(),
                start,
                end,
            }
        })
        .collect()
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
    use super::{chunks, g2p, group_spans, word_times};

    const SPACE: u32 = 16;

    #[test]
    fn groups_run_between_spaces_scaled_to_the_audio() {
        // pad, a b, space, c, pad: 10 duration units over 1 s of audio
        let spans = group_spans(&[1, 2, SPACE, 3], &[1, 2, 2, 1, 3, 1], 24_000, Some(SPACE));
        let rounded: Vec<_> = spans
            .iter()
            .map(|(s, e)| ((s * 10.).round() / 10., (e * 10.).round() / 10.))
            .collect();
        assert_eq!(rounded, [(0.1, 0.5), (0.6, 0.9)]);
    }

    #[test]
    fn words_take_their_groups_or_an_even_share() {
        let word = |text: &str| g2p::PhonemizedWord {
            text: text.into(),
            ipa: String::new(),
        };
        let mut piece = g2p::Phonemized {
            phonemes: String::new(),
            words: vec![word("$3"), word("now")],
            groups: Some(vec![0, 0, 1]),
        };
        let groups = [(0.0, 0.2), (0.2, 0.5), (0.6, 0.9)];
        let times = word_times(&piece, &groups, 1.0);
        assert_eq!((times[0].start, times[0].end), (0.0, 0.5));
        assert_eq!((times[1].start, times[1].end), (0.6, 0.9));
        piece.groups = None;
        let even = word_times(&piece, &groups, 1.0);
        assert_eq!((even[1].start, even[1].end), (0.5, 1.0));
    }

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
