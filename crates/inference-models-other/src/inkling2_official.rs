//! Inkling 2's own checkpoints (`model.llm.*`, thinkingmachines' NVFP4 and MXFP8 releases) bound to the names
//! `crate::inkling2` reads, as llama.cpp's `conversion/inkling2.py` lays them out: the routed experts as GGUF NVFP4
//! blocks (the same codes and scale bytes, each expert's second-level scale kept apart) or blockwise FP8 with 1x32
//! power-of-two scales, every other tensor renamed and, where its layout differs, rearranged.

use std::collections::HashMap;
use std::sync::Arc;

use inference_quant::nvfp4::gguf_blocks::ModeloptNvfp4Weight;
use inference_quant::{
    GgufRawMatMul, GgufType, QuantMethod, QuantizedWeightSource, RawGgufTensor, Shard,
    ShardedSafeTensors, ShardedVarBuilder, TensorShapes,
};
use inference_tensor::nn::{Init, var_builder::SimpleBackend};
use inference_tensor::{DType, Device, Result, Shape, Tensor};

use crate::inkling2::Config;

const MARKER: &str = "model.llm.embed.weight";
const LLM: &str = "model.llm";
// a looped layer's message queries, from the cross-pass queries' rows `[q 2i, kv 2i, kv 2i+1, q 2i+1]`
const CROSS_Q: &str = "model.llm.middle_cycle_cross_pass_queries";
const CROSS_KV: &str = "model.llm.middle_cycle_cross_pass_kv_queries";
// MXFP8's block of values sharing a power-of-two scale; E8M0's 0 is 2^-127, an f32 denormal
const MXFP8_BLOCK: usize = 32;
const DENORMAL_E8M0: u32 = 1 << 22;

/// How a tensor the model reads comes from the checkpoint's.
#[derive(Clone, Debug)]
enum Source {
    Plain(String),
    // `[c, 1, w]` depthwise conv weights as `[c, w]`
    Squeeze(String),
    // the dynamic convs' `[w, c]` static taps, the current input's first, as `[c, w]` with it last
    StaticTaps(String),
    // depth-mix queries as rows
    Rows(String),
    // a looped layer's four message queries
    Messages(usize),
    // stacked experts' second-level NVFP4 scales: `amax / outer_scale_denominator`, `[experts]` F32
    ExpertScales(String),
}

struct Inner {
    base: ShardedVarBuilder,
    sources: HashMap<String, (Source, Vec<usize>)>,
}

impl Inner {
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.base
            .tensor_shape(name)
            .map(<[usize]>::to_vec)
            .ok_or_else(|| inference_tensor::Error::Msg(format!("Inkling 2: no tensor `{name}`")))
    }

    fn load(&self, source: &Source, dtype: DType, dev: &Device) -> Result<Tensor> {
        let base = self.base.clone().set_device(dev.clone());
        let get = |name: &str| base.get_unchecked_dtype(name, dtype);
        match source {
            Source::Plain(name) => get(name),
            Source::Squeeze(name) => get(name)?.squeeze(1),
            Source::StaticTaps(name) => {
                let taps = get(name)?;
                let width = taps.dim(0)?;
                let reversed = (0..width)
                    .rev()
                    .map(|lag| taps.narrow(0, lag, 1))
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&reversed, 0)?.t()?.contiguous()
            }
            Source::Rows(name) => {
                let rows = get(name)?;
                let width = *rows.dims().last().expect("a query has a width");
                rows.reshape(((), width))
            }
            Source::Messages(i) => {
                let (q, kv) = (get(CROSS_Q)?, get(CROSS_KV)?);
                Tensor::cat(
                    &[
                        &q.narrow(0, 2 * i, 1)?,
                        &kv.narrow(0, 2 * i, 2)?,
                        &q.narrow(0, 2 * i + 1, 1)?,
                    ],
                    0,
                )
            }
            Source::ExpertScales(experts) => {
                let amax = base.get_unchecked_dtype(&format!("{experts}_amax"), DType::F32)?;
                let denominator = base.get_unchecked_dtype(
                    &format!("{experts}_outer_scale_denominator"),
                    DType::F32,
                )?;
                amax.broadcast_div(&denominator)?.to_dtype(dtype)
            }
        }
    }

    // the layer's stored expert projection: `w13` for gate (even rows) and up (odd rows), `w2` for down
    fn experts(&self, key: &str, device: &Device) -> Result<Option<Arc<dyn QuantMethod>>> {
        let Some((layer, proj)) = key
            .strip_prefix("model.layers.")
            .and_then(|rest| rest.split_once(".mlp.experts."))
        else {
            return Ok(None);
        };
        let (stored, half) = match proj {
            "gate_proj" => ("w13", Some(0)),
            "up_proj" => ("w13", Some(1)),
            "down_proj" => ("w2", None),
            _ => return Ok(None),
        };
        let name = format!("{LLM}.layers.{layer}.mlp.experts.{stored}_weight");
        let host = self.base.clone().set_device(Device::Cpu);
        let dims = self.shape(&name)?;
        let (experts, rows, packed_cols) = (dims[0], dims[1], dims[2]);
        let rows_of = |tensor: Tensor| -> Result<Tensor> {
            match half {
                Some(half) => {
                    let cols = tensor.dim(2)?;
                    tensor
                        .reshape((experts, rows / 2, 2, cols))?
                        .narrow(2, half, 1)?
                        .squeeze(2)?
                        .contiguous()
                }
                None => Ok(tensor),
            }
        };
        let out_rows = half.map_or(rows, |_| rows / 2);
        let scale_name = format!("{name}_scale");
        if self.base.contains_tensor(&format!("{name}_amax")) {
            let flat = |t: Tensor| -> Result<Tensor> {
                let cols = t.dim(2)?;
                t.reshape((experts * out_rows, cols))
            };
            let cols = packed_cols * 2;
            let blocks = ModeloptNvfp4Weight {
                weight: flat(rows_of(host.get_unchecked_dtype(&name, DType::U8)?)?)?,
                scales: flat(rows_of(
                    host.get_unchecked_dtype(&scale_name, DType::F8E4M3)?,
                )?)?,
                global_scale: 1.0,
            }
            .block_bytes()?;
            let raw =
                RawGgufTensor::new(GgufType::Nvfp4, &[experts, out_rows, cols], blocks, device)?;
            let expert_scales =
                self.load(&Source::ExpertScales(name.clone()), DType::F32, device)?;
            return Ok(Some(Arc::new(
                GgufRawMatMul::new(raw, None).with_expert_scales(Some(expert_scales)),
            )));
        }
        let weight = rows_of(host.get_unchecked_dtype(&name, DType::F8E4M3)?)?;
        let exponents = rows_of(host.get_unchecked_dtype(&scale_name, DType::U8)?)?;
        let scale_dims = exponents.dims().to_vec();
        let scales = exponents
            .flatten_all()?
            .to_vec1::<u8>()?
            .into_iter()
            .map(|e| match e {
                0 => f32::from_bits(DENORMAL_E8M0),
                e => f32::from_bits(u32::from(e) << 23),
            })
            .collect::<Vec<_>>();
        let scales = Tensor::from_vec(scales, scale_dims, device)?;
        inference_quant::blockwise_fp8_moe(
            weight.to_device(device)?,
            scales,
            vec![1, MXFP8_BLOCK],
            None,
            DType::BF16,
        )
        .map(Some)
    }
}

/// The tensors the model reads by its names, from the checkpoint's.
struct Tensors(Arc<Inner>);

impl SimpleBackend for Tensors {
    fn get(&self, s: Shape, name: &str, _: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, dev)?;
        if tensor.shape() != &s {
            inference_tensor::bail!(
                "Inkling 2: `{name}` is {:?} in the checkpoint, the model reads {s:?}",
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        let (source, _) =
            self.0
                .sources
                .get(name)
                .ok_or_else(|| inference_tensor::Error::CannotFindTensor {
                    path: name.to_string(),
                })?;
        self.0.load(source, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.sources.contains_key(name)
    }
}

impl TensorShapes for Tensors {
    fn tensor_shapes(&self) -> HashMap<String, Vec<usize>> {
        self.0
            .sources
            .iter()
            .map(|(name, (_, shape))| (name.clone(), shape.clone()))
            .collect()
    }
}

/// The routed experts, loaded as the fast expert kernels read them.
struct Experts(Arc<Inner>);

impl QuantizedWeightSource for Experts {
    fn contains(&self, name: &str) -> bool {
        name.strip_suffix(".weight")
            .is_some_and(|key| key.contains(".mlp.experts.") && !key.ends_with("_scales"))
    }

    fn load_linear(
        &self,
        key: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Option<Arc<dyn QuantMethod>>> {
        if shard != Shard::default() {
            inference_tensor::bail!("Inkling 2's own checkpoints load on one device");
        }
        self.0.experts(key, device)
    }

    fn load_optional_tensor(&self, _name: &str, _device: &Device) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn shard_alignment(&self, _key: &str) -> Result<usize> {
        Ok(1)
    }

    fn pack_factor(&self, _dtype: DType) -> Result<usize> {
        Ok(1)
    }

    fn pack_factor_for(&self, _key: &str, _dtype: DType) -> Result<Option<usize>> {
        Ok(None)
    }
}

/// `vb` rebound to the model's names when it holds one of Inkling 2's own checkpoints.
pub fn bind(cfg: &Config, vb: ShardedVarBuilder) -> Result<ShardedVarBuilder> {
    if !vb.contains_tensor(MARKER) {
        return Ok(vb);
    }
    let inner = Arc::new(sources(cfg, &vb)?);
    Ok(
        ShardedSafeTensors::wrap(Tensors(inner.clone()), vb.dtype(), vb.device().clone())
            .with_embeddings_on_host(vb.embeddings_on_host())
            .with_moe_expert_cache(vb.moe_expert_cache())
            .with_weight_source(Arc::new(Experts(inner))),
    )
}

/// One of Inkling 2's own checkpoints as the tensors the model reads by its names, other than the routed experts:
/// each one's shape, and its values as the checkpoint load reads them.
pub struct OfficialTensors(Inner);

impl OfficialTensors {
    /// `vb`'s checkpoint, if it is one of Inkling 2's own.
    pub fn open(cfg: &Config, vb: ShardedVarBuilder) -> Result<Option<Self>> {
        if !vb.contains_tensor(MARKER) {
            return Ok(None);
        }
        Ok(Some(Self(sources(cfg, &vb)?)))
    }

    /// The model's names for the tensors, sorted, each with its shape.
    pub fn tensors(&self) -> Vec<(&str, &[usize])> {
        let mut tensors = self
            .0
            .sources
            .iter()
            .map(|(name, (_, shape))| (name.as_str(), shape.as_slice()))
            .collect::<Vec<_>>();
        tensors.sort_unstable();
        tensors
    }

    /// `name`'s values at F32 on the host.
    pub fn values(&self, name: &str) -> Result<Tensor> {
        let (source, _) =
            self.0
                .sources
                .get(name)
                .ok_or_else(|| inference_tensor::Error::CannotFindTensor {
                    path: name.to_string(),
                })?;
        self.0.load(source, DType::F32, &Device::Cpu)
    }
}

// the model's names for a checkpoint's tensors, each with where it comes from and its shape
fn sources(cfg: &Config, vb: &ShardedVarBuilder) -> Result<Inner> {
    let names = vb
        .tensor_names()
        .ok_or_else(|| inference_tensor::Error::Msg("Inkling 2: no tensor listing".to_string()))?;
    let mut inner = Inner {
        base: vb.clone(),
        sources: HashMap::new(),
    };
    for name in &names {
        if let Some((target, source)) = target_of(name) {
            inner.sources.insert(target, (source, Vec::new()));
        }
    }
    let start = cfg.loop_config.core_start;
    for i in 0..cfg.loop_config.core_end - start {
        inner.sources.insert(
            format!("model.layers.{}.loop.msg_query", start + i),
            (Source::Messages(i), Vec::new()),
        );
    }
    let shapes = inner
        .sources
        .iter()
        .map(|(target, (source, _))| Ok((target.clone(), shape_of(&inner, source, cfg)?)))
        .collect::<Result<Vec<_>>>()?;
    for (target, shape) in shapes {
        inner.sources.get_mut(&target).expect("listed above").1 = shape;
    }
    Ok(inner)
}

fn shape_of(inner: &Inner, source: &Source, cfg: &Config) -> Result<Vec<usize>> {
    Ok(match source {
        Source::Plain(name) => inner.shape(name)?,
        Source::Squeeze(name) => {
            let dims = inner.shape(name)?;
            vec![dims[0], dims[2]]
        }
        Source::StaticTaps(name) => {
            let dims = inner.shape(name)?;
            vec![dims[1], dims[0]]
        }
        Source::Rows(name) => {
            let dims = inner.shape(name)?;
            let width = *dims.last().expect("a query has a width");
            vec![dims.iter().product::<usize>() / width, width]
        }
        Source::Messages(_) => vec![CONTROLLER_QUERIES, cfg.hidden_size],
        Source::ExpertScales(_) => vec![cfg.n_routed_experts],
    })
}

// a looped layer's message queries: q, k, v, then the FFN's
const CONTROLLER_QUERIES: usize = 4;

// the model's name for a checkpoint tensor, and how it reads; `None` for what the text model skips
fn target_of(name: &str) -> Option<(String, Source)> {
    let plain = |target: String| Some((target, Source::Plain(name.to_string())));
    let rest = name.strip_prefix(LLM)?.strip_prefix('.')?;
    match rest {
        "embed.weight" => return plain("model.embed_tokens.weight".to_string()),
        "embed_norm.weight" => return plain("model.embed_norm.weight".to_string()),
        "norm.weight" => return plain("model.norm.weight".to_string()),
        "unembed.weight" => return plain("lm_head.weight".to_string()),
        "attn_res.output_query" => {
            return Some((
                "model.final_input.query".to_string(),
                Source::Rows(name.to_string()),
            ));
        }
        _ => {}
    }
    if let Some(i) = rest.strip_prefix("attn_res.attn_queries.") {
        // block i's query feeds layer i + 1
        let layer = i.parse::<usize>().ok()? + 1;
        return Some((
            format!("model.layers.{layer}.attention_input.query"),
            Source::Rows(name.to_string()),
        ));
    }
    if let Some(i) = rest.strip_prefix("attn_res.mlp_queries.") {
        return Some((
            format!("model.layers.{i}.mlp_input.query"),
            Source::Rows(name.to_string()),
        ));
    }
    let (layer, tail) = rest.strip_prefix("layers.")?.split_once('.')?;
    let p = format!("model.layers.{layer}");
    let a = format!("{p}.self_attn");
    let target = match tail {
        "attn.wq_du.weight" => format!("{a}.q_proj.weight"),
        "attn.wk_dv.weight" => format!("{a}.k_proj.weight"),
        "attn.wv_dv.weight" => format!("{a}.v_proj.weight"),
        "attn.wo_ud.weight" => format!("{a}.o_proj.weight"),
        "attn.gate_attn.weight" => format!("{a}.gate_proj.weight"),
        "attn.wr_du.weight" => format!("{a}.r_proj.weight"),
        "attn.attn_sink_proj.weight" => format!("{a}.sink_proj.weight"),
        "attn.q_l2_clip.scale" => format!("{a}.q_l2_clip.scale"),
        "attn.rel_l2_clip.scale" => format!("{a}.rel_l2_clip.scale"),
        "attn.rel_logits_proj.proj" => format!("{a}.rel_logits_proj.proj"),
        "attn.k_sconv.weight_proj" => format!("{a}.k_sconv.weight_proj.weight"),
        "attn.v_sconv.weight_proj" => format!("{a}.v_sconv.weight_proj.weight"),
        "attn_norm.weight" => format!("{p}.input_layernorm.weight"),
        "q_input_norm.weight" => format!("{p}.q_input_layernorm.weight"),
        "k_input_norm.weight" => format!("{p}.k_input_layernorm.weight"),
        "v_input_norm.weight" => format!("{p}.v_input_layernorm.weight"),
        "mlp_norm.weight" => format!("{p}.post_attention_layernorm.weight"),
        "mlp.gate.weight" => format!("{p}.mlp.gate.weight"),
        "mlp.gate.bias" => format!("{p}.mlp.gate.e_score_correction_bias"),
        "attn.k_sconv.weight" => return squeeze(format!("{a}.k_sconv.weight"), name),
        "attn.v_sconv.weight" => return squeeze(format!("{a}.v_sconv.weight"), name),
        "attn_sconv.weight" => {
            return squeeze(format!("{p}.post_attention_convolution.weight"), name);
        }
        "mlp_sconv.weight" => return squeeze(format!("{p}.post_mlp_convolution.weight"), name),
        "attn.k_sconv.static_w" => {
            return Some((
                format!("{a}.k_sconv.weight"),
                Source::StaticTaps(name.to_string()),
            ));
        }
        "attn.v_sconv.static_w" => {
            return Some((
                format!("{a}.v_sconv.weight"),
                Source::StaticTaps(name.to_string()),
            ));
        }
        _ => {
            let (controller, part) = tail.strip_prefix("controllers.")?.split_once('.')?;
            let controller = match controller {
                "attn" => "q",
                "mlp" => "ffn",
                other => other,
            };
            let part = part.strip_suffix("_weight")?;
            format!("{p}.loop.{controller}.{part}.weight")
        }
    };
    plain(target)
}

fn squeeze(target: String, name: &str) -> Option<(String, Source)> {
    Some((target, Source::Squeeze(name.to_string())))
}
