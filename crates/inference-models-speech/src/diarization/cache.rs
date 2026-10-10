//! The arrival-order speaker cache and FIFO of Streaming Sortformer, ported from NeMo's `SortformerModules` and
//! transformers' `Nemotron3DiarizationSpeakerCache` for one stream: the frames each chunk sees before its own.

use inference_tensor::{DType, Result, Tensor};

const LN_HALF: f32 = -std::f32::consts::LN_2;
const SPEECH_PROBABILITY: f32 = 0.5;

/// The cache's sizes and the score policy that chooses which frames it keeps once full.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub cache_length: usize,
    pub fifo_length: usize,
    pub update_period: usize,
    pub silence_frames: usize,
    pub score_threshold: f32,
    pub latest_boost: f32,
    pub num_speakers: usize,
    pub min_positive_scores: usize,
    pub strong_boosted: usize,
    pub weak_boosted: usize,
}

/// What fills the cache slots no frame earns.
pub enum Silence {
    /// A trained embedding (Nemotron-3).
    Learned(Tensor),
    /// The running mean of popped frames whose speaker probabilities sum below `threshold` (Sortformer v2).
    Popped {
        threshold: f32,
        mean: Tensor,
        frames: usize,
    },
}

impl Silence {
    fn embedding(&self) -> &Tensor {
        match self {
            Self::Learned(t) | Self::Popped { mean: t, .. } => t,
        }
    }

    // NeMo's `_get_silence_profile`, over the frames leaving the FIFO
    fn observe(&mut self, popped: &Tensor, probs: &[f32], speakers: usize) -> Result<()> {
        let Self::Popped {
            threshold,
            mean,
            frames,
        } = self
        else {
            return Ok(());
        };
        let is_silent: Vec<f32> = probs
            .chunks(speakers)
            .map(|row| f32::from(row.iter().sum::<f32>() < *threshold))
            .collect();
        let count = is_silent.iter().filter(|&&s| s > 0.).count();
        if count == 0 {
            return Ok(());
        }
        // in F32 whatever the model dtype, as NeMo's float32 zeros keep it: a long mean outruns BF16's mantissa
        let mask = Tensor::new(is_silent.as_slice(), popped.device())?.unsqueeze(0)?;
        let sum = mask.matmul(&popped.to_dtype(DType::F32)?)?.squeeze(0)?;
        let total = *frames + count;
        *mean = ((((&*mean * *frames as f64)? + sum)?) / total as f64)?;
        *frames = total;
        Ok(())
    }
}

pub struct SpeakerCache {
    config: CacheConfig,
    silence: Silence,
    // (frames, hidden) on the device, or none while empty
    embeds: Option<Tensor>,
    // (frames, speakers) row-major, the probabilities stored beside a compressed cache's frames
    probs: Vec<f32>,
    fifo: Option<Tensor>,
    compressed: bool,
}

fn cat(parts: &[&Option<Tensor>]) -> Result<Option<Tensor>> {
    let present: Vec<&Tensor> = parts.iter().filter_map(|p| p.as_ref()).collect();
    match present.as_slice() {
        [] => Ok(None),
        [only] => Ok(Some((*only).clone())),
        _ => Ok(Some(Tensor::cat(&present, 0)?)),
    }
}

fn rows(t: &Option<Tensor>) -> usize {
    t.as_ref().map_or(0, |t| t.dims()[0])
}

impl SpeakerCache {
    pub fn new(config: CacheConfig, silence: Silence) -> Self {
        Self {
            config,
            silence,
            embeds: None,
            probs: Vec::new(),
            fifo: None,
            compressed: false,
        }
    }

    /// The cached frames then the FIFO's, to prepend to the next chunk.
    pub fn embeds(&self) -> Result<Option<Tensor>> {
        cat(&[&self.embeds, &self.fifo])
    }

    fn popped(&self, fifo_frames: usize) -> usize {
        if fifo_frames <= self.config.fifo_length {
            return 0;
        }
        self.config
            .update_period
            .max(fifo_frames - self.config.fifo_length)
            .min(fifo_frames)
    }

    /// Files `chunk_frames` of the step's `input` (cache, FIFO, then context-wrapped chunk), scored by `probs`.
    pub fn update(
        &mut self,
        input: &Tensor,
        probs: &[f32],
        left: usize,
        chunk_frames: usize,
    ) -> Result<()> {
        let s = self.config.num_speakers;
        let (cache_frames, fifo_frames) = (rows(&self.embeds), rows(&self.fifo));
        let chunk_start = cache_frames + fifo_frames + left;
        let chunk = Some(input.narrow(0, chunk_start, chunk_frames)?);
        // the FIFO's frames re-scored by this step, then the chunk's
        let mut fifo_probs = probs[cache_frames * s..(cache_frames + fifo_frames) * s].to_vec();
        fifo_probs.extend_from_slice(&probs[chunk_start * s..(chunk_start + chunk_frames) * s]);
        let fifo = cat(&[&self.fifo, &chunk])?;
        let fifo_len = rows(&fifo);
        let popped = self.popped(fifo_len);
        let fifo = if popped > 0 {
            let fifo = fifo.expect("the FIFO holds the chunk");
            let popped_embeds = fifo.narrow(0, 0, popped)?;
            let popped_probs = &fifo_probs[..popped * s];
            self.silence.observe(&popped_embeds, popped_probs, s)?;
            // an uncompressed cache is plain chunk frames this step re-scored; a compressed one keeps its own
            let mut cache_probs = if self.compressed {
                self.probs[..cache_frames * s].to_vec()
            } else {
                probs[..cache_frames * s].to_vec()
            };
            cache_probs.extend_from_slice(popped_probs);
            let mut cache = cat(&[&self.embeds, &Some(popped_embeds)])?
                .expect("the cache gains the popped frames");
            if cache.dim(0)? > self.config.cache_length {
                (cache, cache_probs) = self.compress(&cache, &cache_probs)?;
                self.compressed = true;
            }
            self.embeds = Some(cache);
            self.probs = cache_probs;
            (popped < fifo_len)
                .then(|| fifo.narrow(0, popped, fifo_len - popped))
                .transpose()?
        } else {
            fifo
        };
        self.fifo = fifo;
        Ok(())
    }

    // log-odds of each frame for each speaker against the others; silence and, once a speaker has enough, its
    // non-positive frames are never kept
    fn frame_scores(&self, probs: &[f32], frames: usize) -> Vec<f32> {
        let (s, th) = (self.config.num_speakers, self.config.score_threshold);
        let mut scores = vec![0f32; frames * s];
        for f in 0..frames {
            let row = &probs[f * s..(f + 1) * s];
            let complements: Vec<f32> = row.iter().map(|p| (1. - p).max(th).ln()).collect();
            let total: f32 = complements.iter().sum();
            for k in 0..s {
                let score = row[k].max(th).ln() - complements[k] + total - LN_HALF;
                scores[f * s + k] = if row[k] > SPEECH_PROBABILITY {
                    score
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
        for k in 0..s {
            let positive = (0..frames).filter(|&f| scores[f * s + k] > 0.).count();
            if positive >= self.config.min_positive_scores {
                for f in 0..frames {
                    let score = &mut scores[f * s + k];
                    if *score <= 0. && probs[f * s + k] > SPEECH_PROBABILITY {
                        *score = f32::NEG_INFINITY;
                    }
                }
            }
        }
        scores
    }

    // the `count` best frames of each speaker gain `boost`
    fn boost(&self, scores: &mut [f32], frames: usize, count: usize, boost: f32) {
        let s = self.config.num_speakers;
        for k in 0..s {
            let mut order: Vec<usize> = (0..frames).collect();
            order.sort_by(|&a, &b| {
                scores[b * s + k]
                    .total_cmp(&scores[a * s + k])
                    .then(a.cmp(&b))
            });
            for &f in order.iter().take(count) {
                scores[f * s + k] += boost;
            }
        }
    }

    // keeps `cache_length` (speaker, frame) choices, speaker by speaker in frame order, the silence embedding
    // standing in for slots no frame earns
    fn compress(&self, embeds: &Tensor, probs: &[f32]) -> Result<(Tensor, Vec<f32>)> {
        let silence = self.silence.embedding().to_dtype(embeds.dtype())?;
        let c = &self.config;
        let s = c.num_speakers;
        let frames = embeds.dim(0)?;
        let mut scores = self.frame_scores(probs, frames);
        for score in &mut scores[c.cache_length.min(frames) * s..] {
            *score += c.latest_boost;
        }
        self.boost(&mut scores, frames, c.strong_boosted, -2. * LN_HALF);
        self.boost(&mut scores, frames, c.weak_boosted, -LN_HALF);
        let scored = frames + c.silence_frames;
        // speaker-major, as the reference flattens (speakers, frames); the silence rows always win a slot
        let mut flat: Vec<(f32, usize)> = (0..s)
            .flat_map(|k| (0..scored).map(move |f| (k, f)))
            .map(|(k, f)| {
                let score = if f < frames {
                    scores[f * s + k]
                } else {
                    f32::INFINITY
                };
                (score, k * scored + f)
            })
            .collect();
        flat.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let sentinel = scored * s;
        let mut chosen: Vec<usize> = flat
            .iter()
            .take(c.cache_length)
            .map(|&(score, i)| {
                if score == f32::NEG_INFINITY {
                    sentinel
                } else {
                    i
                }
            })
            .collect();
        chosen.sort_unstable();
        let frame_of: Vec<u32> = chosen
            .iter()
            .map(|&i| if i == sentinel { frames } else { (i % scored).min(frames) } as u32)
            .collect();
        let with_silence = Tensor::cat(&[embeds, &silence.unsqueeze(0)?], 0)?;
        let index = Tensor::new(frame_of.as_slice(), &embeds.device().clone())?;
        let kept = with_silence.index_select(&index, 0)?;
        let mut kept_probs = Vec::with_capacity(frame_of.len() * s);
        for &f in &frame_of {
            let f = f as usize;
            if f < frames {
                kept_probs.extend_from_slice(&probs[f * s..(f + 1) * s]);
            } else {
                kept_probs.extend(std::iter::repeat_n(0., s));
            }
        }
        Ok((kept, kept_probs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::Device;

    const SPEAKERS: usize = 2;

    fn config() -> CacheConfig {
        CacheConfig {
            cache_length: 8,
            fifo_length: 0,
            update_period: 1,
            silence_frames: 1,
            score_threshold: 0.25,
            latest_boost: 0.05,
            num_speakers: SPEAKERS,
            min_positive_scores: 1,
            strong_boosted: 1,
            weak_boosted: 2,
        }
    }

    fn frames(rows: &[f32]) -> Result<Tensor> {
        let pairs: Vec<f32> = rows.iter().flat_map(|&v| [v, v]).collect();
        Tensor::from_vec(pairs, (rows.len(), 2), &Device::Cpu)
    }

    #[test]
    fn popped_silence_averages_into_the_slots_compression_fills() -> Result<()> {
        let silence = Silence::Popped {
            threshold: 0.2,
            mean: Tensor::zeros(2, DType::F32, &Device::Cpu)?,
            frames: 0,
        };
        let mut cache = SpeakerCache::new(config(), silence);
        // frames 0 and 2 sum below the threshold, so the mean is theirs: 3
        cache.update(
            &frames(&[1., 7., 5.])?,
            &[0.05, 0.05, 0.9, 0.0, 0.0, 0.1],
            0,
            3,
        )?;
        assert_eq!(cache.silence.embedding().to_vec1::<f32>()?, [3., 3.]);

        // a silent frame of 6 then five speaking ones overflow the 8-frame cache; the mean takes the silent frame
        // (to 4) before compression fills each speaker's silence slot with it
        let chunk: Vec<f32> = (0..6).map(|i| 6. + 4. * i as f32).collect();
        let input = Tensor::cat(&[&cache.embeds()?.expect("cached"), &frames(&chunk)?], 0)?;
        let mut probs = vec![0.05, 0.05, 0.9, 0.0, 0.0, 0.1, 0.05, 0.05];
        for i in 1..6 {
            probs.extend_from_slice(if i % 2 == 0 {
                &[0.9, 0.05]
            } else {
                &[0.05, 0.9]
            });
        }
        cache.update(&input, &probs, 0, 6)?;
        assert_eq!(cache.silence.embedding().to_vec1::<f32>()?, [4., 4.]);
        let kept = cache.embeds()?.expect("cached").to_vec2::<f32>()?;
        assert_eq!(kept.len(), 8);
        assert_eq!(kept.iter().filter(|row| row[0] == 4.).count(), SPEAKERS);
        Ok(())
    }
}
