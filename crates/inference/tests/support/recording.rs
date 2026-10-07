//! Random weights for exactly the tensors a model constructor loads, written out as a checkpoint.
#![allow(dead_code)]

use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use inference_nn::{device_map::DeviceMapSetting, model::NormalLoadingMetadata};
use inference_quant::{ShardedSafeTensors, ShardedVarBuilder, TensorShapes};
use inference_tensor::nn::{Init, var_builder::SimpleBackend};
use inference_tensor::{DType, Device, Result as CandleResult, Shape, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

// Large enough that the input moves the logits, small enough to stay finite through a couple of layers.
const WEIGHT_STD: f32 = 0.5;
// Fixed so a failure reproduces with the same weights; the constructor requests tensors in a fixed order.
const WEIGHT_SEED: u64 = 0x0CE1_2024;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Hands out random tensors for whatever the model constructor asks for, and keeps them to write a checkpoint.
#[derive(Clone)]
// The second field names optional tensors to report absent, since a probed-then-loaded tensor carries no shape.
struct RecordingWeights(
    Arc<Mutex<(StdRng, HashMap<String, Tensor>)>>,
    Arc<Vec<String>>,
    // shapes a loader reads before any tensor (stacked or per-expert MoE layouts)
    Arc<HashMap<String, Vec<usize>>>,
    // seed each tensor from its name, so the weights do not depend on the order the constructor asks for them
    bool,
);

impl RecordingWeights {
    fn new(absent: &[&str], shapes: HashMap<String, Vec<usize>>, seed_by_name: bool) -> Self {
        Self(
            Arc::new(Mutex::new((
                StdRng::seed_from_u64(WEIGHT_SEED),
                HashMap::new(),
            ))),
            Arc::new(absent.iter().map(|name| name.to_string()).collect()),
            Arc::new(shapes),
            seed_by_name,
        )
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
        let normal = Normal::new(0f32, WEIGHT_STD).map_err(inference_tensor::Error::wrap)?;
        let mut named = self
            .3
            .then(|| StdRng::seed_from_u64(WEIGHT_SEED ^ fnv1a(name)));
        let rng = named.as_mut().unwrap_or(rng);
        let data = (0..s.elem_count())
            .map(|_| normal.sample(rng))
            .collect::<Vec<_>>();
        let t = Tensor::from_vec(data, s, &Device::Cpu)?.to_dtype(dtype)?;
        seen.insert(name.to_string(), t.clone());
        t.to_device(dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> CandleResult<Tensor> {
        let Some(shape) = self.2.get(name) else {
            inference_tensor::bail!("no shape for {name}")
        };
        self.get(shape.as_slice().into(), name, Init::Const(0.), dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        !self.1.contains(&name.to_string())
    }
}

impl TensorShapes for RecordingWeights {
    fn tensor_shapes(&self) -> HashMap<String, Vec<usize>> {
        self.2.as_ref().clone()
    }
}

// A stable string hash (std's is randomized per process)
fn fnv1a(name: &str) -> u64 {
    name.bytes().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    })
}

/// Copies `files` into a new directory and adds random weights for every tensor `build` asks the var builder for.
pub fn record_checkpoint(
    files: &[&Path],
    num_layers: usize,
    absent: &[&str],
    build: impl FnOnce(ShardedVarBuilder, NormalLoadingMetadata) -> CandleResult<()>,
) -> anyhow::Result<tempfile::TempDir> {
    record_checkpoint_with_shapes(files, num_layers, absent, HashMap::new(), build)
}

/// As [`record_checkpoint`], with each tensor's values seeded by its name, so a refactor that loads the same
/// tensors in another order sees the same weights.
pub fn record_checkpoint_seeded_by_name(
    files: &[&Path],
    num_layers: usize,
    absent: &[&str],
    build: impl FnOnce(ShardedVarBuilder, NormalLoadingMetadata) -> CandleResult<()>,
) -> anyhow::Result<tempfile::TempDir> {
    record(files, num_layers, absent, HashMap::new(), true, build)
}

/// As [`record_checkpoint`], with the shapes of tensors the loader inspects before reading them.
pub fn record_checkpoint_with_shapes(
    files: &[&Path],
    num_layers: usize,
    absent: &[&str],
    shapes: HashMap<String, Vec<usize>>,
    build: impl FnOnce(ShardedVarBuilder, NormalLoadingMetadata) -> CandleResult<()>,
) -> anyhow::Result<tempfile::TempDir> {
    record(files, num_layers, absent, shapes, false, build)
}

fn record(
    files: &[&Path],
    num_layers: usize,
    absent: &[&str],
    shapes: HashMap<String, Vec<usize>>,
    seed_by_name: bool,
    build: impl FnOnce(ShardedVarBuilder, NormalLoadingMetadata) -> CandleResult<()>,
) -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for file in files {
        std::fs::copy(file, dir.path().join(file.file_name().unwrap()))?;
    }
    let weights = RecordingWeights::new(absent, shapes, seed_by_name);
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
    build(
        ShardedSafeTensors::wrap(weights.clone(), DType::F32, Device::Cpu),
        metadata,
    )?;
    let tensors = std::mem::take(&mut weights.0.lock().unwrap().1);
    inference_tensor::safetensors::save(&tensors, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

/// Every file in a committed fixture directory.
pub fn fixture_files(dir: &str) -> anyhow::Result<Vec<std::path::PathBuf>> {
    Ok(std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?)
}
