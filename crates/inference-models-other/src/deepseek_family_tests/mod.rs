//! Shared fixtures for the DeepSeek-V2/V3 and GLM4-MoE(-Lite) behaviour-locking tests beside each model.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use inference_nn::device_map::DummyDeviceMapper;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::model::{ModelForwardContext, NormalLoadingMetadata, NormalModel};
use inference_nn::paged_attention::AttentionImplementation;
use inference_quant::{ShardedSafeTensors, ShardedVarBuilder, TensorShapes};
use serde_json::Value;

pub const VOCAB: usize = 64;
pub const HIDDEN: usize = 32;
pub const INTERMEDIATE: usize = 48;
pub const MOE_INTERMEDIATE: usize = 16;
pub const LAYERS: usize = 2;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 8;
pub const Q_LORA: usize = 16;
pub const KV_LORA: usize = 16;
pub const ROPE_DIM: usize = 8;
pub const NOPE_DIM: usize = 8;
pub const V_DIM: usize = 16;
pub const NARROW_V_DIM: usize = 8;
pub const EXPERTS: usize = 8;
pub const TOP_K: usize = 2;
pub const N_SHARED: usize = 2;
pub const MAX_POS: usize = 64;
pub const SCALE: f64 = 2.5;

pub const PROMPT: [u32; 5] = [3, 17, 42, 8, 59];
// (position, vocab id) pairs whose logits each snapshot records.
pub const PROBES: [(usize, usize); 4] = [(0, 0), (1, 13), (3, 40), (4, 63)];

// Router logits per token: the gate weight is the identity, so these are the gate logits directly.
pub const ROUTER_LOGITS: [[f32; EXPERTS]; 2] = [
    [0.1, 2.0, -1.0, 0.5, 1.5, -0.5, 0.0, 1.0],
    [1.2, -0.3, 0.8, 2.5, -1.5, 0.3, 1.9, -0.7],
];
pub const ROUTER_BIAS: [f32; EXPERTS] = [0.0, -0.5, 0.3, 0.0, 0.2, 0.0, 0.4, 0.1];

const WEIGHT_SCALE: f32 = 0.3;
const NORM_JITTER: f32 = 0.1;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const SNAPSHOT_TOL: f32 = 1e-4;
const ROUTE_TOL: f32 = 1e-5;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// Seeded by the tensor name alone, so adding or reordering tensors never changes the others.
fn fill(name: &str, shape: &[usize]) -> Result<Tensor> {
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

    pub fn mlp(&mut self, prefix: &str, intermediate: usize) {
        self.linear(&format!("{prefix}.gate_proj"), intermediate, HIDDEN);
        self.linear(&format!("{prefix}.up_proj"), intermediate, HIDDEN);
        self.linear(&format!("{prefix}.down_proj"), HIDDEN, intermediate);
    }

    /// Embeddings, final norm, lm_head and per-layer norms plus the dense layer 0 MLP and the MoE layer 1 MLP.
    pub fn skeleton(&mut self, shared_intermediate: usize, gate_bias: bool) {
        self.add("model.embed_tokens.weight", &[VOCAB, HIDDEN]);
        self.add("model.norm.weight", &[HIDDEN]);
        self.add("lm_head.weight", &[VOCAB, HIDDEN]);
        for layer in 0..LAYERS {
            let p = format!("model.layers.{layer}");
            self.add(format!("{p}.input_layernorm.weight"), &[HIDDEN]);
            self.add(format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]);
        }
        self.mlp("model.layers.0.mlp", INTERMEDIATE);
        let p = "model.layers.1.mlp";
        self.add(format!("{p}.gate.weight"), &[EXPERTS, HIDDEN]);
        if gate_bias {
            self.add(format!("{p}.gate.e_score_correction_bias"), &[EXPERTS]);
        }
        for expert in 0..EXPERTS {
            self.mlp(&format!("{p}.experts.{expert}"), MOE_INTERMEDIATE);
        }
        self.mlp(&format!("{p}.shared_experts"), shared_intermediate);
    }

    /// The MLA attention tensors of every layer (DeepSeek-V2/V3, GLM4-MoE-Lite).
    pub fn mla_attention(&mut self, q_lora: Option<usize>, split_kv_b: bool, v_dim: usize) {
        let q_head = NOPE_DIM + ROPE_DIM;
        for layer in 0..LAYERS {
            let p = format!("model.layers.{layer}.self_attn");
            match q_lora {
                Some(rank) => {
                    self.linear(&format!("{p}.q_a_proj"), rank, HIDDEN);
                    self.add(format!("{p}.q_a_layernorm.weight"), &[rank]);
                    self.linear(&format!("{p}.q_b_proj"), HEADS * q_head, rank);
                }
                None => self.linear(&format!("{p}.q_proj"), HEADS * q_head, HIDDEN),
            }
            self.linear(
                &format!("{p}.kv_a_proj_with_mqa"),
                KV_LORA + ROPE_DIM,
                HIDDEN,
            );
            self.add(format!("{p}.kv_a_layernorm.weight"), &[KV_LORA]);
            if split_kv_b {
                self.linear(&format!("{p}.k_b_proj"), HEADS * KV_LORA, NOPE_DIM);
                self.linear(&format!("{p}.v_b_proj"), HEADS * v_dim, KV_LORA);
            } else {
                self.linear(
                    &format!("{p}.kv_b_proj"),
                    HEADS * (NOPE_DIM + v_dim),
                    KV_LORA,
                );
            }
            self.linear(&format!("{p}.o_proj"), HIDDEN, HEADS * v_dim);
        }
    }

    fn tensors(&self) -> Result<HashMap<String, Tensor>> {
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

/// Loads the model through its loader, asserts every provided tensor was read, and returns the logits.
pub fn load_and_forward(
    loader: &dyn NormalModelLoader,
    config: &Value,
    checkpoint: &Checkpoint,
) -> Result<Tensor> {
    let backend = Recording {
        tensors: Arc::new(checkpoint.tensors()?),
        seen: Arc::new(Mutex::new(BTreeSet::new())),
    };
    let vb = ShardedSafeTensors::wrap(backend.clone(), DType::F32, Device::Cpu);
    let model = loader.load(
        &config.to_string(),
        vb,
        metadata(),
        AttentionImplementation::Eager,
    )?;
    let provided = checkpoint.0.keys().cloned().collect::<BTreeSet<_>>();
    let seen = backend.seen.lock().unwrap().clone();
    let unused = provided.difference(&seen).collect::<Vec<_>>();
    let missing = seen.difference(&provided).collect::<Vec<_>>();
    assert!(
        unused.is_empty() && missing.is_empty(),
        "unused tensors {unused:?}, requested but absent {missing:?}"
    );
    forward(model.as_ref())
}

fn forward(model: &(dyn NormalModel + Send + Sync)) -> Result<Tensor> {
    let input = Tensor::new(&PROMPT, &Device::Cpu)?.unsqueeze(0)?;
    let offsets = [0];
    let context_lens = [(0, PROMPT.len())];
    let position_ids = [PROMPT.len()];
    let flash_params = inference_nn::attention::FlashParams::empty(true);
    let mut ctx =
        ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash_params);
    Ok(model.forward(&input, &mut ctx)?.to_dtype(DType::F32)?)
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

pub fn assert_snapshot(logits: &Tensor, expected: &Snapshot) -> Result<()> {
    assert_eq!(logits.dims(), [1, PROMPT.len(), VOCAB]);
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

/// A gate checkpoint whose weight is the identity, so the router logits are the hidden states.
pub fn router_vb(bias: bool) -> Result<ShardedVarBuilder> {
    let mut tensors = HashMap::from([(
        "weight".to_string(),
        Tensor::eye(EXPERTS, DType::F32, &Device::Cpu)?,
    )]);
    if bias {
        tensors.insert(
            "e_score_correction_bias".to_string(),
            Tensor::new(&ROUTER_BIAS, &Device::Cpu)?,
        );
    }
    Ok(wrap(tensors))
}

pub fn router_input() -> Result<Tensor> {
    Ok(Tensor::new(&ROUTER_LOGITS, &Device::Cpu)?.unsqueeze(0)?)
}

/// Compares routes per token as (expert, weight) pairs sorted by expert, so output order is free.
pub fn assert_routes(
    routes: (Tensor, Tensor),
    expected: [[(u32, f32); TOP_K]; ROUTER_LOGITS.len()],
) -> Result<()> {
    let (indices, weights) = routes;
    let indices = indices.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let weights = weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let actual = indices
        .iter()
        .zip(&weights)
        .map(|(idx, w)| {
            let mut pairs = idx
                .iter()
                .copied()
                .zip(w.iter().copied())
                .collect::<Vec<_>>();
            pairs.sort_by_key(|(i, _)| *i);
            pairs
        })
        .collect::<Vec<_>>();
    let ok = actual.iter().zip(&expected).all(|(a, e)| {
        a.len() == e.len()
            && a.iter()
                .zip(e)
                .all(|((ai, aw), (ei, ew))| ai == ei && (aw - ew).abs() <= ROUTE_TOL)
    });
    assert!(ok, "routes moved: {actual:?}");
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
