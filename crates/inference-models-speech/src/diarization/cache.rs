//! The arrival-order speaker cache and FIFO of Streaming Sortformer, ported from transformers'
//! `Nemotron3DiarizationSpeakerCache` for one stream: the frames each chunk sees before its own.

use inference_tensor::nn::ops::sigmoid;
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
    pub subsampling: usize,
    pub min_positive_scores: usize,
    pub strong_boosted: usize,
    pub weak_boosted: usize,
}

pub struct SpeakerCache {
    config: CacheConfig,
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
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
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

    /// Files a step's `chunk_frames` into the FIFO and its overflow into the cache, scored by the step's logits.
    pub fn update(
        &mut self,
        input: &Tensor,
        logits: &Tensor,
        silence: &Tensor,
        chunk_frames: usize,
    ) -> Result<()> {
        let s = self.config.num_speakers;
        let (cache_frames, fifo_frames) = (rows(&self.embeds), rows(&self.fifo));
        let steps = logits.dim(0)? / self.config.subsampling;
        let probs = sigmoid(&logits.to_dtype(DType::F32)?)?
            .reshape((steps, self.config.subsampling, s))?
            .mean(1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let chunk = Some(input.narrow(0, cache_frames + fifo_frames, chunk_frames)?);
        let fifo = cat(&[&self.fifo, &chunk])?;
        let fifo_len = rows(&fifo);
        let popped = self.popped(fifo_len);
        let fifo = if popped > 0 {
            let fifo = fifo.expect("the FIFO holds the chunk");
            let fifo_probs = &probs[cache_frames * s..(cache_frames + fifo_len) * s];
            // an uncompressed cache is plain chunk frames this step re-scored; a compressed one keeps its own
            let mut cache_probs = if self.compressed {
                self.probs[..cache_frames * s].to_vec()
            } else {
                probs[..cache_frames * s].to_vec()
            };
            cache_probs.extend_from_slice(&fifo_probs[..popped * s]);
            let mut cache = cat(&[&self.embeds, &Some(fifo.narrow(0, 0, popped)?)])?
                .expect("the cache gains the popped frames");
            if cache.dim(0)? > self.config.cache_length {
                (cache, cache_probs) = self.compress(&cache, &cache_probs, silence)?;
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
    fn compress(
        &self,
        embeds: &Tensor,
        probs: &[f32],
        silence: &Tensor,
    ) -> Result<(Tensor, Vec<f32>)> {
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
