//! VoxLingua107 spoken language identification: SpeechBrain's ECAPA-TDNN over a 60-mel fbank, then its x-vector
//! classifier over the 107 languages (`speechbrain/lang-id-voxlingua107-ecapa`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inference_audio::fft::plan_forward_f32;
use inference_quant::{ShardedSafeTensors, ShardedVarBuilder};
use inference_tensor::nn::ops::{leaky_relu, log_softmax};
use inference_tensor::nn::{Linear, Module};
use inference_tensor::pickle::PthTensors;
use inference_tensor::{DType, Device, Error, Result, Tensor};
use rustfft::num_complex::Complex;
use serde::Deserialize;

use crate::ecapa::{BatchNorm, Ecapa, EcapaConfig, EcapaLayout};

pub const HYPERPARAMS: &str = "hyperparams.yaml";
pub const LABELS: &str = "label_encoder.txt";
pub const EMBEDDING_MODEL: &str = "embedding_model.ckpt";
pub const CLASSIFIER: &str = "classifier.ckpt";
const EMBEDDING_PREFIX: &str = "embedding_model";
const CLASSIFIER_PREFIX: &str = "classifier";
// the hyperparams' marks of a SpeechBrain ECAPA-TDNN language classifier
const ECAPA_CLASS: &str = "ECAPA_TDNN.ECAPA_TDNN";
const XVECTOR_CLASSIFIER: &str = "Xvector.Classifier";
// the label encoder's lines end at this rule
const LABELS_END: &str = "====";

pub const SAMPLE_RATE: u32 = 16_000;
// SpeechBrain's Fbank defaults: 25 ms Hamming windows every 10 ms, centered with zeros, power spectrum, HTK mels
const N_FFT: usize = 400;
const HOP: usize = 160;
const AMIN: f32 = 1e-10;
const TOP_DB: f32 = 80.0;
const LEAKY_SLOPE: f64 = 0.01;
/// Longer clips are read for their first minute: accuracy has long leveled off, and the pooled layers grow with it.
pub const MAX_SECONDS: usize = 60;

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

#[derive(Deserialize)]
struct Hyperparams {
    compute_features: FbankParams,
    embedding_model: EcapaParams,
    classifier: ClassifierParams,
}

fn default_lin_blocks() -> usize {
    1
}

fn default_classifier_neurons() -> usize {
    512
}

// SpeechBrain's `Xvector.Classifier` arguments, with its defaults; its classes are the label encoder's
#[derive(Deserialize)]
struct ClassifierParams {
    #[serde(default = "default_lin_blocks")]
    lin_blocks: usize,
    #[serde(default = "default_classifier_neurons")]
    lin_neurons: usize,
}

#[derive(Deserialize)]
struct FbankParams {
    n_mels: usize,
}

fn default_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}

fn default_kernels() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}

fn default_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}

fn default_attention() -> usize {
    128
}

fn default_scale() -> usize {
    8
}

fn default_lin_neurons() -> usize {
    192
}

// SpeechBrain's `ECAPA_TDNN` arguments, with its defaults
#[derive(Deserialize)]
struct EcapaParams {
    input_size: usize,
    #[serde(default = "default_channels")]
    channels: Vec<usize>,
    #[serde(default = "default_kernels")]
    kernel_sizes: Vec<usize>,
    #[serde(default = "default_dilations")]
    dilations: Vec<usize>,
    #[serde(default = "default_attention")]
    attention_channels: usize,
    #[serde(default = "default_scale")]
    res2net_scale: usize,
    #[serde(default = "default_attention")]
    se_channels: usize,
    #[serde(default = "default_lin_neurons")]
    lin_neurons: usize,
}

/// The four files of a SpeechBrain language-ID repo.
#[derive(Clone, Debug)]
pub struct VoxLinguaFiles {
    pub hyperparams: PathBuf,
    pub labels: PathBuf,
    pub embedding_model: PathBuf,
    pub classifier: PathBuf,
}

impl VoxLinguaFiles {
    pub fn in_dir(dir: &Path) -> Self {
        Self {
            hyperparams: dir.join(HYPERPARAMS),
            labels: dir.join(LABELS),
            embedding_model: dir.join(EMBEDDING_MODEL),
            classifier: dir.join(CLASSIFIER),
        }
    }
}

/// Whether `hyperparams` (a `hyperparams.yaml`'s text) builds an ECAPA-TDNN with an x-vector classifier.
pub fn is_language_id_hyperparams(hyperparams: &str) -> bool {
    hyperparams.contains(ECAPA_CLASS) && hyperparams.contains(XVECTOR_CLASSIFIER)
}

/// Whether `dir` holds a SpeechBrain ECAPA-TDNN language identifier.
pub fn is_language_id_dir(dir: &Path) -> bool {
    let files = VoxLinguaFiles::in_dir(dir);
    files.labels.is_file()
        && std::fs::read_to_string(&files.hyperparams).is_ok_and(|h| is_language_id_hyperparams(&h))
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Language {
    /// ISO 639-1 (or 639-3, where the corpus has no two-letter code).
    pub code: String,
    pub name: String,
}

// `'ab: Abkhazian' => 0` per class, in index order
fn read_labels(text: &str) -> Result<Vec<Language>> {
    let mut labels: Vec<(usize, Language)> = Vec::new();
    for line in text.lines() {
        if line.starts_with(LABELS_END) {
            break;
        }
        let Some((label, index)) = line.rsplit_once("=>") else {
            continue;
        };
        let index: usize = index.trim().parse().map_err(msg)?;
        let label = label.trim().trim_matches('\'');
        let (code, name) = label.split_once(':').unwrap_or((label, label));
        labels.push((
            index,
            Language {
                code: code.trim().to_string(),
                name: name.trim().to_string(),
            },
        ));
    }
    labels.sort_by_key(|(index, _)| *index);
    if labels.iter().enumerate().any(|(i, (index, _))| i != *index) {
        inference_tensor::bail!("the label encoder's indices are not 0..n");
    }
    Ok(labels.into_iter().map(|(_, language)| language).collect())
}

/// SpeechBrain's `Fbank` then sentence mean normalization: `[frames][n_mels]` log-mels of 16 kHz `pcm`.
pub fn fbank(pcm: &[f32], n_mels: usize) -> Vec<Vec<f32>> {
    let pad = N_FFT / 2;
    let frames = pcm.len() / HOP + 1;
    #[allow(clippy::cast_precision_loss)]
    let window: Vec<f32> = (0..N_FFT)
        .map(|n| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos())
        .collect();
    let filters = htk_filters(n_mels);
    let fft = plan_forward_f32(N_FFT);
    let mut buffer = vec![Complex::new(0f32, 0f32); N_FFT];
    let mut mels: Vec<Vec<f32>> = (0..frames)
        .map(|t| {
            for (n, slot) in buffer.iter_mut().enumerate() {
                let i = (t * HOP + n).checked_sub(pad);
                let sample = i.and_then(|i| pcm.get(i)).copied().unwrap_or(0.0);
                *slot = Complex::new(sample * window[n], 0.0);
            }
            fft.process(&mut buffer);
            let power: Vec<f32> = buffer[..N_FFT / 2 + 1]
                .iter()
                .map(Complex::norm_sqr)
                .collect();
            filters
                .iter()
                .map(|f| {
                    let energy: f32 = f.iter().zip(&power).map(|(w, p)| w * p).sum();
                    10.0 * energy.max(AMIN).log10()
                })
                .collect()
        })
        .collect();
    let floor = mels.iter().flatten().fold(f32::MIN, |a, &b| a.max(b)) - TOP_DB;
    mels.iter_mut().flatten().for_each(|v| *v = v.max(floor));
    #[allow(clippy::cast_precision_loss)]
    let mean: Vec<f32> = (0..n_mels)
        .map(|m| mels.iter().map(|row| row[m]).sum::<f32>() / frames as f32)
        .collect();
    for row in &mut mels {
        row.iter_mut().zip(&mean).for_each(|(v, m)| *v -= m);
    }
    mels
}

// SpeechBrain's triangular filters: HTK mel centers, each triangle as wide on both sides as its left band
fn htk_filters(n_mels: usize) -> Vec<Vec<f32>> {
    let to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let to_hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    let top = f64::from(SAMPLE_RATE / 2);
    let max_mel = to_mel(top);
    #[allow(clippy::cast_precision_loss)]
    let hz: Vec<f64> = (0..n_mels + 2)
        .map(|i| to_hz(max_mel * i as f64 / (n_mels + 1) as f64))
        .collect();
    let bins = N_FFT / 2 + 1;
    (0..n_mels)
        .map(|m| {
            let (center, band) = (hz[m + 1], hz[m + 1] - hz[m]);
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            (0..bins)
                .map(|k| {
                    let freq = top * k as f64 / (bins - 1) as f64;
                    let slope = (freq - center) / band;
                    (slope + 1.0).min(1.0 - slope).max(0.0) as f32
                })
                .collect()
        })
        .collect()
}

// LeakyReLU, BatchNorm, then each block's linear, LeakyReLU and BatchNorm, then the output linear and log-softmax
struct Classifier {
    norm: BatchNorm,
    blocks: Vec<(Linear, BatchNorm)>,
    out: Linear,
}

impl Classifier {
    fn new(
        input: usize,
        params: &ClassifierParams,
        classes: usize,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let linear = |in_dim: usize, out_dim: usize, vb: &ShardedVarBuilder| -> Result<Linear> {
            let weight = vb.get((out_dim, in_dim), "weight")?;
            Ok(Linear::new(weight, Some(vb.get(out_dim, "bias")?)))
        };
        let mut width = input;
        let blocks = (0..params.lin_blocks)
            .map(|i| {
                let block = vb.pp("DNN").pp(format!("block_{i}"));
                let lin = linear(width, params.lin_neurons, &block.pp("linear").pp("w"))?;
                width = params.lin_neurons;
                Ok((lin, BatchNorm::new(width, &block.pp("norm").pp("norm"))?))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            norm: BatchNorm::new(input, &vb.pp("norm").pp("norm"))?,
            blocks,
            out: linear(width, classes, &vb.pp("out").pp("w"))?,
        })
    }

    // log-probabilities `(classes,)` of an embedding `(dim,)`
    fn forward(&self, embedding: &Tensor) -> Result<Tensor> {
        let mut xs = self
            .norm
            .forward_2d(&leaky_relu(&embedding.unsqueeze(0)?, LEAKY_SLOPE)?)?;
        for (linear, norm) in &self.blocks {
            xs = norm.forward_2d(&leaky_relu(&linear.forward(&xs)?, LEAKY_SLOPE)?)?;
        }
        log_softmax(&self.out.forward(&xs)?, 1)?.squeeze(0)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LanguageProbability {
    #[serde(flatten)]
    pub language: Language,
    pub probability: f32,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LanguageIdentification {
    /// Every language the model knows, likeliest first.
    pub languages: Vec<LanguageProbability>,
    /// The pooled utterance embedding the classifier reads.
    pub embedding: Vec<f32>,
    /// Seconds of audio read: the clip, or its first [`MAX_SECONDS`].
    pub duration: f64,
}

pub struct VoxLingua {
    ecapa: Ecapa,
    classifier: Classifier,
    labels: Vec<Language>,
    n_mels: usize,
    device: Device,
}

// a torch checkpoint's tensors under `prefix.`, as F32
fn read_ckpt(path: &Path, prefix: &str, out: &mut HashMap<String, Tensor>) -> Result<()> {
    let pth = PthTensors::new(path, None)?;
    for name in pth.tensor_infos().keys() {
        if let Some(t) = pth.get(name)? {
            out.insert(format!("{prefix}.{name}"), t.to_dtype(DType::F32)?);
        }
    }
    Ok(())
}

impl VoxLingua {
    pub fn load(files: &VoxLinguaFiles, device: &Device) -> Result<Self> {
        let hyperparams = std::fs::read_to_string(&files.hyperparams)?;
        if !is_language_id_hyperparams(&hyperparams) {
            inference_tensor::bail!(
                "`{}` is not an ECAPA-TDNN language classifier with an x-vector head",
                files.hyperparams.display()
            );
        }
        let labels = std::fs::read_to_string(&files.labels)?;
        let mut tensors = HashMap::new();
        read_ckpt(&files.embedding_model, EMBEDDING_PREFIX, &mut tensors)?;
        read_ckpt(&files.classifier, CLASSIFIER_PREFIX, &mut tensors)?;
        let vb = ShardedSafeTensors::wrap(tensors, DType::F32, device.clone());
        Self::new(&hyperparams, &labels, &vb, device)
    }

    /// `vb` holds the embedding model's tensors under `embedding_model` and the classifier's under `classifier`.
    pub fn new(
        hyperparams: &str,
        labels: &str,
        vb: &ShardedVarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let params: Hyperparams = serde_saphyr::from_str(hyperparams).map_err(msg)?;
        let labels = read_labels(labels)?;
        let p = params.embedding_model;
        let cfg = EcapaConfig {
            input_dim: p.input_size,
            channels: p.channels,
            kernel_sizes: p.kernel_sizes,
            dilations: p.dilations,
            attention_channels: p.attention_channels,
            res2net_scale: p.res2net_scale,
            se_channels: p.se_channels,
            out_dim: p.lin_neurons,
            layout: EcapaLayout::SpeechBrain,
        };
        Ok(Self {
            ecapa: Ecapa::new(&cfg, &vb.pp(EMBEDDING_PREFIX))?,
            classifier: Classifier::new(
                p.lin_neurons,
                &params.classifier,
                labels.len(),
                &vb.pp(CLASSIFIER_PREFIX),
            )?,
            labels,
            n_mels: params.compute_features.n_mels,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn languages(&self) -> &[Language] {
        &self.labels
    }

    /// The language posterior of mono `pcm` at `sample_rate`, resampled to 16 kHz first.
    pub fn identify(&self, pcm: &[f32], sample_rate: u32) -> Result<LanguageIdentification> {
        let pcm = inference_audio::mel::resample(pcm, sample_rate, SAMPLE_RATE).map_err(msg)?;
        let pcm = &pcm[..pcm.len().min(MAX_SECONDS * SAMPLE_RATE as usize)];
        let features = fbank(pcm, self.n_mels);
        let embedding = self.ecapa.embed(&features, &self.device, DType::F32)?;
        let probabilities: Vec<f32> = self.classifier.forward(&embedding)?.exp()?.to_vec1()?;
        let mut languages: Vec<LanguageProbability> = self
            .labels
            .iter()
            .zip(probabilities)
            .map(|(language, probability)| LanguageProbability {
                language: language.clone(),
                probability,
            })
            .collect();
        languages.sort_by(|a, b| b.probability.total_cmp(&a.probability));
        Ok(LanguageIdentification {
            languages,
            embedding: embedding.to_vec1()?,
            duration: pcm.len() as f64 / f64::from(SAMPLE_RATE),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_read_in_index_order() {
        let text =
            "'zh: Chinese' => 1\n'ab: Abkhazian' => 0\n================\n'starting_index' => 0\n";
        let labels = read_labels(text).unwrap();
        assert_eq!(labels[0].code, "ab");
        assert_eq!(labels[1].name, "Chinese");
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn tagged_hyperparams_read() {
        let text = "n_mels: 60\ncompute_features: !new:speechbrain.lobes.features.Fbank\n  n_mels: 60\n\
                    embedding_model: !new:speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN\n  input_size: 60\n  \
                    channels: [8, 8, 8, 8, 24]\n  lin_neurons: 16\n\
                    classifier: !new:speechbrain.lobes.models.Xvector.Classifier\n  out_neurons: !ref <n_mels>\n";
        assert!(is_language_id_hyperparams(text));
        let params: Hyperparams = serde_saphyr::from_str(text).unwrap();
        assert_eq!(params.embedding_model.channels, vec![8, 8, 8, 8, 24]);
        assert_eq!(params.embedding_model.dilations, vec![1, 2, 3, 4, 1]);
        assert_eq!(params.compute_features.n_mels, 60);
    }
}
