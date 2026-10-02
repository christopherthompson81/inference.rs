//! Fixtures for the tiny random-weight model tests in the model family crates.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use inference_quant::{ShardedSafeTensors, ShardedVarBuilder, TensorShapes};
use serde_json::Value;

use crate::device_map::DummyDeviceMapper;
use crate::model::{ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel};

pub const PROMPT: [u32; 5] = [3, 17, 42, 8, 59];
// (position, vocab id) pairs whose logits each snapshot records.
pub const PROBES: [(usize, usize); 4] = [(0, 0), (1, 13), (3, 40), (4, 63)];

const WEIGHT_SCALE: f32 = 0.3;
const NORM_JITTER: f32 = 0.1;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const SNAPSHOT_TOL: f32 = 1e-4;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// Seeded by the tensor name alone, so adding or reordering tensors never changes the others.
pub fn fill(name: &str, shape: &[usize]) -> Result<Tensor> {
    let mut state = name.bytes().fold(FNV_OFFSET, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
    });
    let is_norm = name.ends_with("norm.weight");
    let count = shape.iter().product::<usize>();
    let data = (0..count)
        .map(|_| {
            let bits = u16::try_from(splitmix(&mut state) >> 48).expect("16 bits");
            let unit = f32::from(bits) / 32768.0 - 1.0;
            if is_norm {
                1.0 + NORM_JITTER * unit
            } else {
                WEIGHT_SCALE * unit
            }
        })
        .collect::<Vec<_>>();
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?)
}

/// Tensor names and shapes of a test checkpoint, as HF names them.
#[derive(Default, Clone)]
pub struct Checkpoint(pub BTreeMap<String, Vec<usize>>);

impl Checkpoint {
    pub fn add(&mut self, name: impl Into<String>, shape: &[usize]) {
        let name = name.into();
        assert!(
            self.0.insert(name.clone(), shape.to_vec()).is_none(),
            "duplicate {name}"
        );
    }

    pub fn linear(&mut self, prefix: &str, out_dim: usize, in_dim: usize) {
        self.add(format!("{prefix}.weight"), &[out_dim, in_dim]);
    }

    pub fn tensors(&self) -> Result<HashMap<String, Tensor>> {
        self.0
            .iter()
            .map(|(name, shape)| Ok((name.clone(), fill(name, shape)?)))
            .collect()
    }
}

/// An in-memory backend that remembers every tensor name the model reads.
#[derive(Clone)]
struct Recording {
    tensors: Arc<HashMap<String, Tensor>>,
    seen: Arc<Mutex<BTreeSet<String>>>,
}

impl Recording {
    fn mark(&self, name: &str) {
        self.seen.lock().unwrap().insert(name.to_string());
    }
}

impl SimpleBackend for Recording {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.mark(name);
        SimpleBackend::get(self.tensors.as_ref(), s, name, h, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.mark(name);
        SimpleBackend::get_unchecked(self.tensors.as_ref(), name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

impl TensorShapes for Recording {
    fn tensor_shapes(&self) -> HashMap<String, Vec<usize>> {
        self.tensors.tensor_shapes()
    }
}

/// Runs `load` over `checkpoint` and asserts it read every tensor the checkpoint has and no other.
pub fn load_checked<T>(
    checkpoint: &Checkpoint,
    load: impl FnOnce(ShardedVarBuilder) -> Result<T>,
) -> Result<T> {
    let backend = Recording {
        tensors: Arc::new(checkpoint.tensors()?),
        seen: Arc::new(Mutex::new(BTreeSet::new())),
    };
    let model = load(ShardedSafeTensors::wrap(
        backend.clone(),
        DType::F32,
        Device::Cpu,
    ))?;
    let provided = checkpoint.0.keys().cloned().collect::<BTreeSet<_>>();
    let seen = backend.seen.lock().unwrap().clone();
    let unused = provided.difference(&seen).collect::<Vec<_>>();
    let missing = seen.difference(&provided).collect::<Vec<_>>();
    assert!(
        unused.is_empty() && missing.is_empty(),
        "unused tensors {unused:?}, requested but absent {missing:?}"
    );
    Ok(model)
}

pub fn metadata() -> NormalLoadingMetadata {
    NormalLoadingMetadata {
        mapper: Box::new(DummyDeviceMapper {
            nm_device: Device::Cpu,
        }),
        loading_isq: false,
        real_device: Device::Cpu,
        multi_progress: Arc::new(indicatif::MultiProgress::new()),
        matformer_slicing_config: None,
        rope_pairing: None,
    }
}

pub fn wrap(tensors: HashMap<String, Tensor>) -> ShardedVarBuilder {
    ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu)
}

/// Overlays `patch` onto `base` key by key.
pub fn patched(mut base: Value, patch: Value) -> Value {
    let (Value::Object(base_map), Value::Object(patch_map)) = (&mut base, patch) else {
        panic!("configs are JSON objects");
    };
    base_map.extend(patch_map);
    base
}

fn prompt_forward(
    run: impl FnOnce(&Tensor, &mut ModelForwardContext<'_>) -> candle_core::Result<Tensor>,
) -> Result<Tensor> {
    let input = Tensor::new(&PROMPT, &Device::Cpu)?.unsqueeze(0)?;
    let offsets = [0];
    let context_lens = [(0, PROMPT.len())];
    let position_ids = [PROMPT.len()];
    let flash_params = crate::attention::FlashParams::empty(true);
    let mut ctx =
        ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash_params);
    Ok(run(&input, &mut ctx)?.to_dtype(DType::F32)?)
}

/// Logits of `PROMPT` as one prefill.
pub fn forward_normal(model: &(dyn NormalModel + Send + Sync)) -> Result<Tensor> {
    prompt_forward(|input, ctx| model.forward(input, ctx))
}

/// Logits of `PROMPT` as one text-only prefill.
pub fn forward_multimodal(model: &(dyn MultimodalModel + Send + Sync)) -> Result<Tensor> {
    prompt_forward(|input, ctx| {
        model.forward(input, None, model.default_model_specific_args(input), ctx)
    })
}

/// Logits recorded at `PROBES`, plus the sum and L2 norm over every logit.
pub struct Snapshot {
    pub probes: [f32; PROBES.len()],
    pub sum: f32,
    pub l2: f32,
}

fn close(actual: f32, expected: f32, tol: f32) -> bool {
    (actual - expected).abs() <= tol * expected.abs().max(1.0)
}

pub fn assert_snapshot(logits: &Tensor, vocab: usize, expected: &Snapshot) -> Result<()> {
    assert_eq!(logits.dims(), [1, PROMPT.len(), vocab]);
    let rows = logits.squeeze(0)?.to_vec2::<f32>()?;
    let probes = PROBES.map(|(pos, id)| rows[pos][id]);
    let sum = logits.sum_all()?.to_scalar::<f32>()?;
    let l2 = logits.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let ok = probes
        .iter()
        .zip(&expected.probes)
        .all(|(a, e)| close(*a, *e, SNAPSHOT_TOL))
        && close(sum, expected.sum, SNAPSHOT_TOL)
        && close(l2, expected.l2, SNAPSHOT_TOL);
    assert!(
        ok,
        "logits moved: Snapshot {{ probes: {probes:?}, sum: {sum:?}, l2: {l2:?} }}"
    );
    Ok(())
}

/// Asserts `result` failed with an error whose message names `needle`, so a pinned failure cannot change cause silently.
pub fn assert_err_contains<T, E: std::fmt::Display>(
    result: std::result::Result<T, E>,
    needle: &str,
) {
    match result {
        Ok(_) => panic!("expected an error naming {needle:?}"),
        Err(err) => {
            let msg = format!("{err:#}");
            assert!(msg.contains(needle), "error moved: {msg}");
        }
    }
}
