//! Builds a tiny random-weight PaddleOCR-VL checkpoint at test time; shared by the SDK, server and ABI tests.
// Each test crate includes this file and uses a different subset of it.
#![allow(dead_code)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Result as CandleResult, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init};
use inference_models_other::paddleocr_vl::{config::Config, PaddleOcrVlModel};
use inference_nn::{
    device_map::DeviceMapSetting, model::NormalLoadingMetadata,
    paged_attention::AttentionImplementation,
};
use inference_quant::{ShardedSafeTensors, TensorShapes};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

// Resolves from any crate under crates/, so every test crate reads the same committed config and tokenizer.
pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/paddleocr_vl/tiny"
);
pub const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/paddleocr_vl"
);
// Large enough that the image moves the logits, small enough to stay finite through two layers.
const WEIGHT_STD: f32 = 0.5;
// Fixed so a failure reproduces with the same weights; the constructor requests tensors in a fixed order.
const WEIGHT_SEED: u64 = 0x0CE1_2024;

/// Hands out random tensors for whatever the model constructor asks for, and keeps them to write a checkpoint.
#[derive(Clone)]
struct RecordingWeights(Arc<Mutex<(StdRng, HashMap<String, Tensor>)>>);

impl RecordingWeights {
    fn new() -> Self {
        Self(Arc::new(Mutex::new((
            StdRng::seed_from_u64(WEIGHT_SEED),
            HashMap::new(),
        ))))
    }
}

impl SimpleBackend for RecordingWeights {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        dev: &Device,
    ) -> CandleResult<Tensor> {
        let mut guard = self.0.lock().unwrap();
        let (rng, seen) = &mut *guard;
        if let Some(t) = seen.get(name) {
            return Ok(t.clone());
        }
        let normal = Normal::new(0f32, WEIGHT_STD).map_err(candle_core::Error::wrap)?;
        let data = (0..s.elem_count())
            .map(|_| normal.sample(rng))
            .collect::<Vec<_>>();
        let t = Tensor::from_vec(data, s, &Device::Cpu)?.to_dtype(dtype)?;
        seen.insert(name.to_string(), t.clone());
        t.to_device(dev)
    }

    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> CandleResult<Tensor> {
        candle_core::bail!("no shape for {name}")
    }

    fn contains_tensor(&self, _: &str) -> bool {
        true
    }
}

impl TensorShapes for RecordingWeights {
    fn tensor_shapes(&self) -> HashMap<String, Vec<usize>> {
        HashMap::new()
    }
}

/// The committed tiny config, tokenizer and templates plus random weights for exactly the tensors the model loads.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for entry in std::fs::read_dir(TINY)? {
        let path = entry?.path();
        std::fs::copy(&path, dir.path().join(path.file_name().unwrap()))?;
    }
    let cfg: Config =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json"))?)?;
    let weights = RecordingWeights::new();
    let num_layers = cfg.text_config().num_hidden_layers;
    let metadata = NormalLoadingMetadata {
        mapper: DeviceMapSetting::dummy().into_mapper(
            num_layers,
            &Device::Cpu,
            None,
            &[Device::Cpu],
        )?,
        loading_isq: false,
        real_device: Device::Cpu,
        multi_progress: Arc::new(indicatif::MultiProgress::new()),
        matformer_slicing_config: None,
        rope_pairing: None,
    };
    let vb = ShardedSafeTensors::wrap(weights.clone(), DType::F32, Device::Cpu);
    PaddleOcrVlModel::new(&cfg, vb, metadata, AttentionImplementation::Eager)?;
    let tensors = std::mem::take(&mut weights.0.lock().unwrap().1);
    candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors"))?;
    Ok(dir)
}
