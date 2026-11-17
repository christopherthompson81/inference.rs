//! EAGLE-3 drafters (arXiv 2503.01840): `fc` fuses three target layers' hidden states into one, a single Llama layer
//! reads `concat(norm(embed(next token)), norm(fused))` with the fused state as its residual, and a reduced-vocab
//! `lm_head` maps back to the target's ids through `d2t`. Each chained draft feeds the layer's pre-norm output back in
//! place of the fused state. Checkpoints: SpecForge `LlamaForCausalLMEagle3`, vLLM/AngelSlim `Eagle3LlamaForCausalLM`,
//! the speculators library's `Eagle3Speculator`, and yuhuili's plain one-layer `LlamaForCausalLM`.

use std::{
    fs,
    path::Path,
    sync::{Arc, atomic::AtomicUsize},
};

use inference_quant::{IsqType, QuantMethod, QuantizeOntoGuard, ShardedVarBuilder};
use inference_tensor::quantized::gguf_file::Value as GgufValue;
use inference_tensor::{D, DType, Device, Result, Tensor};
use serde_json::Value;

use crate::{
    attention::AttentionMask,
    device_map::DeviceMapper,
    kv_cache::KvCache,
    layers::{Activation, Llama3RopeConfig},
    model::ModelForwardContext,
    paged_attention::{AttentionImplementation, AuxiliaryPagedLayers, PagedAttention},
    utils::{
        unvarbuilder::UnVarBuilder,
        varbuilder_utils::{DeviceForLoadTensor, from_mmaped_safetensors},
    },
};

use super::{
    AttentionBlock, DecoderSpec, LayerAttention, LayerLoad, MlpKind, ModuleNames, MtpBlock,
    MtpHead, MtpSpec, Norm, NormKind, NormNames, QkvProj, RopeKind,
    mtp::{DraftVocab, MtpConcatParts},
};

const CONFIG_FILE: &str = "config.json";
const WEIGHT_EXTENSIONS: &[&str] = &["safetensors", "bin", "pt", "pth"];
/// Target layers an EAGLE-3 drafter reads.
pub const EAGLE3_TAPS: usize = 3;
// SpecForge names its one layer `midlayer`; the speculators library and vLLM `layers.0`
const MIDLAYER: &str = "midlayer";
const LAYER0: &str = "layers.0";
const MIDLAYER_PROBE: &str = "midlayer.input_layernorm.weight";
const FC: &str = "fc";
// EAGLE 3.1's per-tap norms ahead of `fc`, vLLM's names
const FC_NORMS: &str = "fcs";
// the speculators library's one norm over the concatenated taps (`norm_before_fc`)
const FC_INPUT_NORM: &str = "input_norm";
const FINAL_NORM: &str = "norm";
const LM_HEAD: &str = "lm_head";
const D2T: &str = "d2t";
// tensors the drafter ships but never reads: its copy of the target's embedding, and the target-to-draft id mask
const UNREAD: &[&str] = &["embed_tokens.weight", "t2d"];
// the speculators library's proposal length, when the drafter names none
const DEFAULT_DRAFTS: usize = 3;
const DEFAULT_ROPE_THETA: f64 = 10_000.0;
// llama.cpp's EAGLE-3 GGUF (`convert_hf_to_gguf.py` with `--target-model-dir`)
const GGUF_EXTENSION: &str = "gguf";
const GGUF_ARCHITECTURE: &str = "eagle3";
const GGUF_DRAFT_HEAD: &str = "output.weight";
// our name for each tensor, llama.cpp's for it; its token embedding (the target's copy) stays unread
const GGUF_TENSORS: &[(&str, &str)] = &[
    ("fc.weight", "fc.weight"),
    ("input_norm.weight", "enc.output_norm.weight"),
    ("norm.weight", "output_norm.weight"),
    ("lm_head.weight", GGUF_DRAFT_HEAD),
    ("layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"),
    ("layers.0.hidden_norm.weight", "blk.0.attn_norm_2.weight"),
    (
        "layers.0.post_attention_layernorm.weight",
        "blk.0.ffn_norm.weight",
    ),
    ("layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
    ("layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight"),
    ("layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight"),
    (
        "layers.0.self_attn.o_proj.weight",
        "blk.0.attn_output.weight",
    ),
    ("layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
    ("layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
    ("layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
];

fn msg(e: impl std::fmt::Display) -> inference_tensor::Error {
    inference_tensor::Error::msg(e.to_string())
}

fn usize_field(v: &Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(|n| n as usize)
}

fn gguf_uint(value: &GgufValue) -> Option<usize> {
    match *value {
        GgufValue::I32(v) => usize::try_from(v).ok(),
        GgufValue::I64(v) => usize::try_from(v).ok(),
        ref v => v.to_u64().ok().and_then(|v| usize::try_from(v).ok()),
    }
}

fn flag(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// An EAGLE-3 drafter's config, read from any of its checkpoint formats.
#[derive(Clone, Debug)]
pub struct Eagle3Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope: RopeKind,
    max_position_embeddings: usize,
    vocab_size: Option<usize>,
    draft_vocab_size: Option<usize>,
    layer_ids: Option<Vec<usize>>,
    target_hidden_size: Option<usize>,
    norm_before_residual: bool,
    fc_norm: bool,
    norm_before_fc: bool,
    norm_output: bool,
    n_predict: Option<usize>,
    // a llama.cpp GGUF drafter: its file, its q/k rows in llama.cpp's adjacent-pair RoPE order
    gguf: Option<std::path::PathBuf>,
}

impl Eagle3Config {
    /// The config of the drafter in `dir`, or `None` when it is not an EAGLE-3 drafter.
    pub fn peek(dir: &Path) -> Result<Option<Self>> {
        if dir.extension().and_then(|e| e.to_str()) == Some(GGUF_EXTENSION) && dir.is_file() {
            return Self::from_gguf(dir);
        }
        let Ok(text) = fs::read_to_string(dir.join(CONFIG_FILE)) else {
            return Ok(None);
        };
        let raw: Value = serde_json::from_str(&text).map_err(msg)?;
        Self::parse(&raw)
    }

    fn parse(raw: &Value) -> Result<Option<Self>> {
        let archs = raw
            .get("architectures")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        // the speculators library nests the layer's config
        let layer = raw.get("transformer_layer_config").unwrap_or(raw);
        let one_layer = usize_field(layer, "num_hidden_layers").is_none_or(|n| n == 1);
        let named = archs.iter().any(|a| a.contains("Eagle3"));
        // yuhuili's checkpoints are a plain one-layer Llama with a draft vocabulary
        let plain = archs.contains(&"LlamaForCausalLM")
            && usize_field(layer, "num_hidden_layers") == Some(1)
            && raw.get("draft_vocab_size").is_some();
        if !named && !plain {
            return Ok(None);
        }
        if !one_layer {
            inference_tensor::bail!("an EAGLE-3 drafter has one layer");
        }
        let need = |key: &str| {
            usize_field(layer, key).ok_or_else(|| msg(format!("the EAGLE-3 config has no `{key}`")))
        };
        let hidden_size = need("hidden_size")?;
        let num_heads = need("num_attention_heads")?;
        let rope_params = layer.get("rope_parameters");
        if rope_params
            .and_then(|p| p.get("partial_rotary_factor"))
            .is_some()
            || rope_params.and_then(|p| p.get("mrope_section")).is_some()
        {
            inference_tensor::bail!("EAGLE-3 drafters with partial or M-RoPE are not supported");
        }
        let theta = layer
            .get("rope_theta")
            .or_else(|| rope_params.and_then(|p| p.get("rope_theta")))
            .and_then(Value::as_f64)
            .unwrap_or(DEFAULT_ROPE_THETA);
        let scaling = match layer.get("rope_scaling") {
            None | Some(Value::Null) => None,
            Some(scaling) => Some(
                serde_json::from_value::<Llama3RopeConfig>(scaling.clone()).map_err(|e| {
                    msg(format!(
                        "an EAGLE-3 drafter's `rope_scaling` is not Llama 3's: {e}"
                    ))
                })?,
            ),
        };
        let layer_ids = raw
            .get("eagle_aux_hidden_state_layer_ids")
            .or_else(|| {
                raw.get("eagle_config")
                    .and_then(|c| c.get("eagle_aux_hidden_state_layer_ids"))
            })
            .and_then(Value::as_array)
            .map(|ids| {
                ids.iter()
                    .map(|id| id.as_u64().map(|id| id as usize))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| msg("EAGLE-3 layer ids are not integers"))
            })
            .transpose()?;
        let n_predict = raw
            .get("speculators_config")
            .and_then(|c| c.get("proposal_methods"))
            .and_then(Value::as_array)
            .and_then(|methods| methods.first())
            .and_then(|method| usize_field(method, "speculative_tokens"));
        Ok(Some(Self {
            hidden_size,
            intermediate_size: need("intermediate_size")?,
            num_heads,
            num_kv_heads: usize_field(layer, "num_key_value_heads").unwrap_or(num_heads),
            head_dim: usize_field(layer, "head_dim").unwrap_or(hidden_size / num_heads),
            rms_norm_eps: layer
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6),
            rope: RopeKind::Llama3 {
                theta: theta as f32,
                scaling,
            },
            max_position_embeddings: need("max_position_embeddings")?,
            vocab_size: usize_field(layer, "vocab_size"),
            draft_vocab_size: usize_field(raw, "draft_vocab_size"),
            layer_ids,
            target_hidden_size: usize_field(raw, "target_hidden_size"),
            norm_before_residual: flag(raw, "norm_before_residual"),
            fc_norm: flag(raw, "fc_norm"),
            norm_before_fc: flag(raw, "norm_before_fc"),
            norm_output: flag(raw, "norm_output"),
            n_predict,
            gguf: None,
        }))
    }

    /// A llama.cpp EAGLE-3 GGUF's config, from its `eagle3.*` metadata; `None` for another architecture.
    fn from_gguf(path: &Path) -> Result<Option<Self>> {
        let archive = inference_quant::GgufArchive::open([path])?;
        let meta = |key: &str| archive.metadata_value(&format!("{GGUF_ARCHITECTURE}.{key}"));
        if !matches!(archive.metadata_value("general.architecture"), Some(GgufValue::String(arch)) if arch == GGUF_ARCHITECTURE)
        {
            return Ok(None);
        }
        let uint = |key: &str| -> Result<usize> {
            meta(key).and_then(gguf_uint).ok_or_else(|| {
                msg(format!(
                    "the EAGLE-3 GGUF has no `{GGUF_ARCHITECTURE}.{key}`"
                ))
            })
        };
        let float = |key: &str| {
            meta(key).and_then(|v| v.to_f64().or_else(|_| v.to_f32().map(f64::from)).ok())
        };
        let flag = |key: &str| meta(key).and_then(|v| v.to_bool().ok()).unwrap_or(false);
        let layer_ids = meta("target_layers")
            .map(|ids| {
                ids.to_vec()
                    .map_err(msg)?
                    .iter()
                    .map(|id| {
                        gguf_uint(id).ok_or_else(|| msg("EAGLE-3 GGUF layer ids are not integers"))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        // the draft head's rows
        let draft_vocab_size = archive
            .tensor_info(GGUF_DRAFT_HEAD)
            .ok()
            .and_then(|info| info.shape().first().copied());
        let hidden_size = uint("embedding_length")?;
        let num_heads = uint("attention.head_count")?;
        Ok(Some(Self {
            hidden_size,
            intermediate_size: uint("feed_forward_length")?,
            num_heads,
            num_kv_heads: uint("attention.head_count_kv").unwrap_or(num_heads),
            head_dim: uint("attention.key_length").unwrap_or(hidden_size / num_heads),
            rms_norm_eps: float("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            rope: RopeKind::Llama3 {
                theta: float("rope.freq_base").unwrap_or(DEFAULT_ROPE_THETA) as f32,
                scaling: None,
            },
            max_position_embeddings: uint("context_length")?,
            vocab_size: uint("vocab_size").ok(),
            draft_vocab_size,
            layer_ids,
            target_hidden_size: uint("target_hidden_size").ok(),
            norm_before_residual: flag("norm_before_residual"),
            fc_norm: false,
            norm_before_fc: flag("norm_before_fc"),
            norm_output: false,
            n_predict: None,
            gguf: Some(path.to_path_buf()),
        }))
    }

    /// The paged layer the drafter's attention keeps in the target's cache.
    pub fn auxiliary_layers(&self) -> AuxiliaryPagedLayers {
        AuxiliaryPagedLayers {
            count: 1,
            num_attn_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            attended: true,
        }
    }

    /// The draft depth the checkpoint was tuned for, when it records one.
    pub fn n_predict(&self) -> usize {
        self.n_predict.unwrap_or(DEFAULT_DRAFTS)
    }

    /// The target layers whose outputs the drafter reads, low to high. Checkpoints name the layer each state is the
    /// input of (vLLM's and llama.cpp's convention), defaulting to vLLM's `(2, L / 2, L - 3)`.
    pub fn tap_layers(&self, target_layers: usize) -> Result<[usize; EAGLE3_TAPS]> {
        let ids = self
            .layer_ids
            .clone()
            .unwrap_or_else(|| vec![2, target_layers / 2, target_layers.saturating_sub(3)]);
        let ids: [usize; EAGLE3_TAPS] = ids
            .try_into()
            .map_err(|ids: Vec<usize>| msg(format!("EAGLE-3 reads three layers, not {ids:?}")))?;
        if ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2] {
            inference_tensor::bail!("EAGLE-3 reads three distinct layers, not {ids:?}");
        }
        if ids.iter().any(|&id| id == 0 || id > target_layers) {
            inference_tensor::bail!(
                "EAGLE-3 layer ids {ids:?} are outside the target's 1..={target_layers}"
            );
        }
        Ok(ids.map(|id| id - 1))
    }
}

/// The drafter's one layer over `concat(embedding, fused)`: both halves normed, the attention reading them together
/// and the fused half (normed, with `norm_before_residual`) as the residual.
pub struct Eagle3Block {
    input_norm: Norm,
    hidden_norm: Norm,
    attention: AttentionBlock,
    post_attention_norm: Norm,
    mlp: Eagle3Mlp,
    norm_before_residual: bool,
    hidden_size: usize,
}

impl MtpBlock for Eagle3Block {
    fn forward(
        &self,
        xs: &Tensor,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        kv_layer_idx: usize,
    ) -> Result<Tensor> {
        let embeds = xs.narrow(D::Minus1, 0, self.hidden_size)?;
        let fused = xs.narrow(D::Minus1, self.hidden_size, self.hidden_size)?;
        let normed = self.hidden_norm.forward(&fused)?;
        let residual = if self.norm_before_residual {
            normed.clone()
        } else {
            fused
        };
        let attention_input = Tensor::cat(&[self.input_norm.forward(&embeds)?, normed], D::Minus1)?;
        let attended = self.attention.forward(
            &attention_input,
            &AttentionMask::None,
            Some(kv_cache),
            ctx,
            kv_layer_idx,
            None,
        )?;
        let xs = (attended + residual)?;
        let mlp_out = self.mlp.forward(&self.post_attention_norm.forward(&xs)?)?;
        mlp_out + xs
    }

    // the drafter is loaded on its own, outside the target's ISQ and UQFF
    fn add_residual(&self, _uvb_m: &UnVarBuilder) {}
}

/// The layer's gated SiLU MLP, its projections at the drafter's own weight type.
struct Eagle3Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl Eagle3Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = (self.gate.forward(xs)?.silu()? * self.up.forward(xs)?)?;
        self.down.forward(&gated)
    }
}

// a projection at `isq`, or as the checkpoint stores it
fn at_isq(
    layer: Arc<dyn QuantMethod>,
    isq: Option<IsqType>,
    device: &Device,
) -> Result<Arc<dyn QuantMethod>> {
    match isq {
        None => Ok(layer),
        Some(isq) => layer.apply_isq(
            Some(isq),
            device.clone(),
            &AtomicUsize::new(0),
            None,
            QuantizeOntoGuard::new(),
        ),
    }
}

/// What a target gives its EAGLE-3 drafter at attach.
pub struct Eagle3Target<'a> {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    /// The paged layer after the target's own (and any built-in head's).
    pub kv_layer_idx: usize,
    pub mapper: &'a dyn DeviceMapper,
    pub device: &'a Device,
    pub dtype: DType,
    /// The drafter's own weight type (`--mtp-draft-isq`); unset, the checkpoint's.
    pub isq: Option<IsqType>,
}

// what `fc` reads: the taps as they are, each normed (EAGLE 3.1), or normed together
enum FcInput {
    Raw,
    PerTap(Vec<Norm>),
    Concat(Norm),
}

/// A loaded EAGLE-3 drafter: the chained head over its layer, and the fusion of the target's taps it reads.
pub struct Eagle3Drafter {
    pub head: MtpHead<Eagle3Block>,
    fc: Arc<dyn QuantMethod>,
    fc_input: FcInput,
    taps: [usize; EAGLE3_TAPS],
}

impl Eagle3Drafter {
    /// The target layers whose outputs [`Eagle3Drafter::fuse`] takes, in order.
    pub fn taps(&self) -> &[usize; EAGLE3_TAPS] {
        &self.taps
    }

    /// `fc(concat(taps))`, `[.., 3 * target hidden]` to the drafter's hidden.
    pub fn fuse(&self, taps: &[Tensor]) -> Result<Tensor> {
        let fused = match &self.fc_input {
            FcInput::Raw => Tensor::cat(taps, D::Minus1)?,
            FcInput::PerTap(norms) => {
                let normed = taps
                    .iter()
                    .zip(norms)
                    .map(|(tap, norm)| norm.forward(tap))
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&normed, D::Minus1)?
            }
            FcInput::Concat(norm) => norm.forward(&Tensor::cat(taps, D::Minus1)?)?,
        };
        self.fc.forward(&fused)
    }

    /// As [`Eagle3Drafter::fuse`], over taps recorded in layer order.
    pub fn fuse_layer_ordered(&self, taps: &[Tensor]) -> Result<Tensor> {
        let mut layers = self.taps;
        layers.sort_unstable();
        let ordered = self
            .taps
            .iter()
            .map(|tap| taps[layers.partition_point(|layer| layer < tap)].clone())
            .collect::<Vec<_>>();
        self.fuse(&ordered)
    }

    /// The EAGLE-3 drafter `config` names (`--mtp-model`), drafting `config.n_predict` or its own proposal length;
    /// `None` when the directory holds another kind of drafter.
    pub fn attach(
        config: &crate::speculative::MtpConfig,
        target: Eagle3Target<'_>,
    ) -> Result<Option<(Self, crate::speculative::SpeculativeAttachInfo)>> {
        let dir = config.resolve_path()?;
        let Some(cfg) = Eagle3Config::peek(&dir)? else {
            return Ok(None);
        };
        let n_predict = config.n_predict.unwrap_or(cfg.n_predict());
        if n_predict == 0 {
            inference_tensor::bail!("EAGLE-3 needs at least one draft token");
        }
        let mut drafter = Self::load(&dir, &cfg, target)?;
        drafter.head.set_n_predict(n_predict);
        let info = crate::speculative::SpeculativeAttachInfo::mtp(
            format!("EAGLE-3 {}", dir.display()),
            n_predict,
        );
        Ok(Some((drafter, info)))
    }

    /// Records `taps` (layer order) fused as the head's target states: every row, and the rows the logits read.
    pub fn record_layer_ordered(
        &self,
        taps: &[Tensor],
        logits_rows: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<()> {
        let fused = self.fuse_layer_ordered(taps)?;
        let selected = logits_rows(&fused)?;
        self.head.record(&fused, &selected);
        Ok(())
    }

    pub fn load(dir: &Path, cfg: &Eagle3Config, target: Eagle3Target<'_>) -> Result<Self> {
        if cfg.hidden_size != target.hidden_size {
            inference_tensor::bail!(
                "the EAGLE-3 drafter's hidden size {} is not the target's {}: it would need its own embedding",
                cfg.hidden_size,
                target.hidden_size
            );
        }
        if let Some(width) = cfg.target_hidden_size
            && width != target.hidden_size
        {
            inference_tensor::bail!(
                "the EAGLE-3 drafter was trained on a {width}-wide target, this one is {}",
                target.hidden_size
            );
        }
        if let Some(vocab) = cfg.vocab_size
            && vocab != target.vocab_size
        {
            inference_tensor::bail!(
                "the EAGLE-3 drafter's vocabulary ({vocab}) is not the target's ({})",
                target.vocab_size
            );
        }
        let taps = cfg.tap_layers(target.num_layers)?;
        let vb = match &cfg.gguf {
            Some(path) => gguf_var_builder(path, target.device, target.dtype)?,
            None => checkpoint_var_builder(dir, target.device, target.dtype)?,
        };
        let kv_layer_idx = target.kv_layer_idx;
        let spec = DecoderSpec {
            vocab_size: target.vocab_size,
            hidden_size: cfg.hidden_size,
            attention_input_size: Some(2 * cfg.hidden_size),
            intermediate_size: cfg.intermediate_size,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            hidden_act: Activation::Silu,
            rms_norm_eps: cfg.rms_norm_eps,
            rope: cfg.rope.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            layer_windows: vec![None; kv_layer_idx + 1],
            norm: NormKind::Rms,
            norm_names: NormNames::PRE,
            module_names: ModuleNames::HF,
            mlp: MlpKind::Gated,
            ..Default::default()
        };
        // a GGUF's q/k rows are in llama.cpp's adjacent-pair order
        let half_split = cfg.gguf.is_none();
        let rope = Arc::new(spec.rope(&spec.rope, &vb, target.device, half_split, target.dtype)?);
        let comm = target.mapper.get_comm_for(kv_layer_idx)?;
        let load = LayerLoad {
            mapper: target.mapper,
            layer_idx: kv_layer_idx,
            loading_isq: false,
            comm: &comm,
            device: target.device,
            attention: &AttentionImplementation::PagedAttention,
        };
        let prefix = if vb.contains_tensor(MIDLAYER_PROBE) {
            MIDLAYER
        } else {
            LAYER0
        };
        let vb_layer = vb.pp(prefix);
        let paged = PagedAttention::new(cfg.head_dim, target.device, None)?;
        let mut attention =
            AttentionBlock::new(&spec, &load, vb_layer.pp("self_attn"), rope, Some(paged))?;
        let isq = |layer| at_isq(layer, target.isq, target.device);
        attention.qkv = match attention.qkv {
            QkvProj::Split { q, k, v } => QkvProj::Split {
                q: isq(q)?,
                k: isq(k)?,
                v: isq(v)?,
            },
            QkvProj::Fused(qkv) => QkvProj::Fused(isq(qkv)?),
        };
        attention.o_proj = isq(attention.o_proj)?;
        let vb_mlp = vb_layer.pp("mlp");
        let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let linear = |i, o, name| isq(spec.replicated_linear(i, o, false, vb_mlp.pp(name))?);
        let mlp = Eagle3Mlp {
            gate: linear(hidden, inter, "gate_proj")?,
            up: linear(hidden, inter, "up_proj")?,
            down: linear(inter, hidden, "down_proj")?,
        };
        let norm = |vb: ShardedVarBuilder, width: usize| {
            Norm::new(NormKind::Rms, width, cfg.rms_norm_eps, vb)
        };
        let block = Eagle3Block {
            input_norm: norm(vb_layer.pp("input_layernorm"), cfg.hidden_size)?,
            hidden_norm: norm(vb_layer.pp("hidden_norm"), cfg.hidden_size)?,
            attention,
            post_attention_norm: norm(vb_layer.pp("post_attention_layernorm"), cfg.hidden_size)?,
            mlp,
            norm_before_residual: cfg.norm_before_residual,
            hidden_size: cfg.hidden_size,
        };
        let fc = isq(spec.replicated_linear(
            EAGLE3_TAPS * target.hidden_size,
            cfg.hidden_size,
            false,
            vb.pp(FC),
        )?)?;
        let fc_input = if cfg.fc_norm {
            FcInput::PerTap(
                (0..EAGLE3_TAPS)
                    .map(|tap| norm(vb.pp(FC_NORMS).pp(tap), target.hidden_size))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else if cfg.norm_before_fc {
            FcInput::Concat(norm(
                vb.pp(FC_INPUT_NORM),
                EAGLE3_TAPS * target.hidden_size,
            )?)
        } else {
            FcInput::Raw
        };
        let draft_vocab = cfg.draft_vocab_size.unwrap_or(target.vocab_size);
        let lm_head =
            isq(spec.replicated_linear(cfg.hidden_size, draft_vocab, false, vb.pp(LM_HEAD))?)?;
        let vocab = (draft_vocab != target.vocab_size)
            .then(|| {
                draft_vocab_map(
                    draft_offsets(&vb, cfg.gguf.as_deref())?,
                    draft_vocab,
                    target.vocab_size,
                    target.device,
                    target.dtype,
                )
            })
            .transpose()?;
        let head = MtpHead::concat(MtpConcatParts {
            block,
            final_norm: norm(vb.pp(FINAL_NORM), cfg.hidden_size)?,
            lm_head,
            vocab,
            names: MtpSpec {
                embedding_norm: "input_layernorm",
                hidden_norm: "hidden_norm",
                projection: FC,
                block: prefix,
                final_norm: Some(FINAL_NORM),
                dense_block: true,
                mask_first_position: false,
                at_root: false,
            },
            chain_normed: cfg.norm_output,
            kv_layer_idx,
            device: target.device.clone(),
            dtype: target.dtype,
        });
        Ok(Self {
            head,
            fc,
            fc_input,
            taps,
        })
    }
}

// a checkpoint directory's safetensors or pickled weights
fn checkpoint_var_builder(dir: &Path, device: &Device, dtype: DType) -> Result<ShardedVarBuilder> {
    let mut paths = fs::read_dir(dir)
        .map_err(msg)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| WEIGHT_EXTENSIONS.contains(&e))
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        inference_tensor::bail!("no EAGLE-3 weights in {}", dir.display());
    }
    from_mmaped_safetensors(
        paths,
        Some(dtype),
        device,
        Vec::new(),
        true,
        None,
        |name| !UNREAD.contains(&name.as_str()),
        Arc::new(|_| DeviceForLoadTensor::Base),
    )
}

// a llama.cpp GGUF's tensors under our names, its linears kept in their quantized types
fn gguf_var_builder(path: &Path, device: &Device, dtype: DType) -> Result<ShardedVarBuilder> {
    let archive = Arc::new(inference_quant::GgufArchive::open([path])?);
    let bindings = GGUF_TENSORS
        .iter()
        .filter(|(_, gguf)| archive.tensor_info(gguf).is_ok())
        .map(|(ours, gguf)| {
            (
                (*ours).to_string(),
                inference_quant::GgufTensorBinding::tensor(*gguf),
            )
        })
        .collect::<inference_quant::GgufBindingMap>();
    let source = Arc::new(inference_quant::GgufWeightSource::new(
        archive, &bindings, dtype,
    )?);
    Ok(source.sharded_var_builder(device.clone()))
}

// each target id's draft id (`target = draft + d2t[draft]`), the floor's column past the draft vocabulary otherwise
// the draft-to-target offsets: a checkpoint's `d2t` tensor, or a GGUF's, which llama.cpp stores as the target ids
fn draft_offsets(vb: &ShardedVarBuilder, gguf: Option<&Path>) -> Result<Vec<i64>> {
    let Some(path) = gguf else {
        return vb
            .get_unchecked_dtype(D2T, DType::I64)?
            .to_device(&Device::Cpu)?
            .to_vec1::<i64>();
    };
    let archive = inference_quant::GgufArchive::open([path])?;
    let data = archive.tensor_data(D2T)?;
    Ok(data
        .bytes()
        .as_chunks::<{ std::mem::size_of::<i64>() }>()
        .0
        .iter()
        .map(|bytes| i64::from_le_bytes(*bytes))
        .zip(0i64..)
        .map(|(target, draft)| target - draft)
        .collect())
}

fn draft_vocab_map(
    d2t: Vec<i64>,
    draft_vocab: usize,
    target_vocab: usize,
    device: &Device,
    dtype: DType,
) -> Result<DraftVocab> {
    if d2t.len() != draft_vocab {
        inference_tensor::bail!(
            "the EAGLE-3 `d2t` has {} entries for a {draft_vocab}-token draft vocabulary",
            d2t.len()
        );
    }
    let ids = d2t
        .iter()
        .enumerate()
        .map(|(id, offset)| {
            u32::try_from(id as i64 + offset)
                .ok()
                .filter(|&t| (t as usize) < target_vocab)
                .ok_or_else(|| {
                    msg(format!(
                        "EAGLE-3 draft id {id} maps outside the target vocabulary"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    #[allow(clippy::cast_possible_truncation)]
    let mut gather = vec![draft_vocab as u32; target_vocab];
    for (draft, &target) in ids.iter().enumerate() {
        gather[target as usize] = draft as u32;
    }
    Ok(DraftVocab {
        gather: Tensor::from_vec(gather, target_vocab, device)?,
        floor: Tensor::from_vec(vec![f32::NEG_INFINITY], (1, 1), device)?.to_dtype(dtype)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_checkpoint_format_reads_as_eagle3() {
        let specforge = serde_json::json!({
            "architectures": ["LlamaForCausalLMEagle3"], "hidden_size": 4096, "intermediate_size": 12288,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 1,
            "max_position_embeddings": 40960, "rope_theta": 1000000.0, "vocab_size": 151936,
            "draft_vocab_size": 32000
        });
        let speculators = serde_json::json!({
            "architectures": ["Eagle3Speculator"], "draft_vocab_size": 32000, "norm_before_residual": true,
            "norm_before_fc": true,
            "speculators_config": {"proposal_methods": [{"speculative_tokens": 5}]},
            "transformer_layer_config": {
                "hidden_size": 4096, "intermediate_size": 12288, "num_attention_heads": 32,
                "num_key_value_heads": 8, "num_hidden_layers": 1, "max_position_embeddings": 40960,
                "rope_theta": 1000000, "vocab_size": 151936
            }
        });
        let plain = serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "intermediate_size": 14336,
            "num_attention_heads": 32, "num_hidden_layers": 1, "max_position_embeddings": 2048,
            "draft_vocab_size": 32000
        });
        let eagle31 = serde_json::json!({
            "architectures": ["LlamaForCausalLMEagle3"], "hidden_size": 2048, "intermediate_size": 16384,
            "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "num_hidden_layers": 1,
            "max_position_embeddings": 262144, "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "draft_vocab_size": 32000, "fc_norm": true, "norm_output": true
        });
        let base = serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "intermediate_size": 14336,
            "num_attention_heads": 32, "num_hidden_layers": 32, "max_position_embeddings": 2048
        });
        let a = Eagle3Config::parse(&specforge).unwrap().unwrap();
        assert_eq!(
            (a.head_dim, a.num_kv_heads, a.n_predict()),
            (128, 8, DEFAULT_DRAFTS)
        );
        let b = Eagle3Config::parse(&speculators).unwrap().unwrap();
        assert!(b.norm_before_residual);
        assert_eq!(b.n_predict(), 5);
        let c = Eagle3Config::parse(&plain).unwrap().unwrap();
        assert_eq!((c.head_dim, c.num_kv_heads), (128, 32));
        let d = Eagle3Config::parse(&eagle31).unwrap().unwrap();
        assert!(d.fc_norm && d.norm_output && !a.fc_norm && !a.norm_output);
        assert!(b.norm_before_fc && !a.norm_before_fc && !d.norm_before_fc);
        assert!(matches!(d.rope, RopeKind::Llama3 { theta, scaling: None } if theta == 1e7));
        assert!(Eagle3Config::parse(&base).unwrap().is_none());
    }

    #[test]
    fn layer_ids_name_the_layer_each_state_feeds() {
        let mut cfg = serde_json::json!({
            "architectures": ["Eagle3LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "max_position_embeddings": 128
        });
        let parsed = Eagle3Config::parse(&cfg).unwrap().unwrap();
        assert_eq!(parsed.tap_layers(36).unwrap(), [1, 17, 32]);
        cfg["eagle_config"] = serde_json::json!({"eagle_aux_hidden_state_layer_ids": [3, 19, 35]});
        let parsed = Eagle3Config::parse(&cfg).unwrap().unwrap();
        assert_eq!(parsed.tap_layers(40).unwrap(), [2, 18, 34]);
        assert!(parsed.tap_layers(30).is_err());
        cfg["eagle_config"] = serde_json::json!({"eagle_aux_hidden_state_layer_ids": [3, 3, 35]});
        let parsed = Eagle3Config::parse(&cfg).unwrap().unwrap();
        assert!(parsed.tap_layers(40).is_err());
    }
}
