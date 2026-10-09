use std::collections::HashMap;
use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::nn::var_builder::SimpleBackend;
use inference_tensor::pickle::PthTensors;
use inference_tensor::{DType, Device, Result, Shape, Tensor, bail};

// the release's state dicts, each saved from a DataParallel wrapper
const PARTS: [&str; 5] = [
    "bert",
    "bert_encoder",
    "predictor",
    "decoder",
    "text_encoder",
];
const PART_PREFIX: &str = "module.";
const VOICE_ROWS: usize = 510;
const VOICE_DIM: usize = 256;

/// The `.pth` release: one state dict per top-level module, keys under `module.`.
struct KokoroPth(HashMap<&'static str, PthTensors>);

impl KokoroPth {
    fn tensor(&self, name: &str) -> Result<Tensor> {
        let (part, rest) = name.split_once('.').unwrap_or((name, ""));
        let key = if rest.is_empty() {
            PART_PREFIX.trim_end_matches('.').to_string()
        } else {
            format!("{PART_PREFIX}{rest}")
        };
        match self.0.get(part).map(|p| p.get(&key)).transpose()?.flatten() {
            Some(t) => Ok(t),
            None => bail!("no tensor {name} in the Kokoro checkpoint"),
        }
    }
}

impl SimpleBackend for KokoroPth {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: inference_tensor::nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let t = self.tensor(name)?;
        if t.shape() != &s {
            bail!(
                "shape mismatch for {name}: expected {s:?}, got {:?}",
                t.shape()
            )
        }
        t.to_device(dev)?.to_dtype(dtype)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.tensor(name)?.to_device(dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensor(name).is_ok()
    }
}

/// A VarBuilder over the release checkpoint, naming tensors `bert.embeddings...` as the GGUF does.
pub fn pth_var_builder(checkpoint: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let parts = PARTS
        .iter()
        .map(|&part| Ok((part, PthTensors::new(checkpoint, Some(part))?)))
        .collect::<Result<HashMap<_, _>>>()?;
    Ok(VarBuilder::from_backend(
        Box::new(KokoroPth(parts)),
        DType::F32,
        device.clone(),
    ))
}

/// A voice: one 2 * style_dim style row per phoneme count, 1 to 510.
#[derive(Debug, Clone)]
pub struct VoicePack {
    rows: Vec<f32>,
}

impl VoicePack {
    pub fn new(rows: Vec<f32>) -> Result<Self> {
        if rows.len() != VOICE_ROWS * VOICE_DIM {
            bail!(
                "a voice pack has {} values, expected {VOICE_ROWS} x {VOICE_DIM}",
                rows.len()
            )
        }
        Ok(Self { rows })
    }

    /// A release voice file (`voices/*.pt`, a bare (510, 1, 256) tensor).
    pub fn from_pt(path: &Path) -> Result<Self> {
        match PthTensors::new(path, None)?.get("")? {
            Some(t) => Self::new(t.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?),
            None => bail!("{} holds no tensor", path.display()),
        }
    }

    /// A raw little-endian f32 pack (audio.cpp's `voices/*.bin`, also embedded in its GGUF).
    pub fn from_f32_le(bytes: &[u8]) -> Result<Self> {
        let (values, rest) = bytes.as_chunks::<4>();
        if !rest.is_empty() {
            bail!(
                "a raw voice pack holds f32 values, got {} bytes",
                bytes.len()
            )
        }
        Self::new(values.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    /// The element-wise mean of several voices, as the reference blends a comma-separated voice list.
    pub fn mean(packs: &[VoicePack]) -> Result<Self> {
        let Some(first) = packs.first() else {
            bail!("no voices to blend")
        };
        let mut rows = first.rows.clone();
        for pack in &packs[1..] {
            rows.iter_mut().zip(&pack.rows).for_each(|(a, b)| *a += b);
        }
        let n = packs.len() as f32;
        rows.iter_mut().for_each(|a| *a /= n);
        Self::new(rows)
    }

    /// The style for a `phonemes`-character input.
    pub fn style(&self, phonemes: usize) -> Result<&[f32]> {
        if phonemes == 0 || phonemes > VOICE_ROWS {
            bail!("a voice covers 1 to {VOICE_ROWS} phonemes, got {phonemes}")
        }
        let row = phonemes - 1;
        Ok(&self.rows[row * VOICE_DIM..(row + 1) * VOICE_DIM])
    }
}
