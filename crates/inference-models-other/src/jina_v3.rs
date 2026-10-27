//! Jina Embeddings v3: XLM-R with rotary positions in Nomic BERT's layout (fused `mixer.Wqkv`, GELU `mlp.fc1` /
//! `mlp.fc2`, post-norm `norm1` / `norm2`), and five task LoRAs on every linear layer and embedding table, kept as
//! `parametrizations.weight` (torch's parametrize) beside each base weight.
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use inference_quant::{QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear};
use inference_tensor::nn::{Embedding, LayerNorm, Linear, Module};
use inference_tensor::{Device, Result, Tensor};
use serde::Deserialize;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams},
    device_map::DeviceMapper,
    embedding::{EmbeddingExtras, EmbeddingTasks, TaskEmbeddingLora, TaskLora, TaskRows},
    layers::layer_norm,
    model::{EmbeddingModel, IsqModel, NormalLoadingMetadata},
    paged_attention::AttentionImplementation,
    rope_encoder::{QkvLayout, RopeAttentionShape, RopeSelfAttention, RopeStyle, rotary},
    utils::unvarbuilder::UnVarBuilder,
};

/// The `position_embedding_type` a Jina v3 config carries, which sets it apart from plain XLM-R.
pub const ROTARY_POSITIONS: &str = "rotary";
const BASE_PREFIX: &str = "roberta";
const EMB_LN: &str = "emb_ln.weight";
const PARAMETRIZED: &str = "parametrizations";
const WEIGHT: &str = "weight";
const ORIGINAL: &str = "original";
// torch's parametrize keeps the first parametrization of `weight` at index 0
const FIRST: usize = 0;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub type_vocab_size: usize,
    pub pad_token_id: usize,
    pub rotary_emb_base: f32,
    pub lora_adaptations: Vec<String>,
    #[serde(default)]
    pub task_instructions: HashMap<String, String>,
    pub lora_rank: usize,
    pub lora_alpha: f64,
}

impl Config {
    /// XLM-R numbers its positions from `pad_token_id + 1`, and Jina keeps that limit.
    pub fn max_seq_len(&self) -> usize {
        self.max_position_embeddings - self.pad_token_id - 1
    }

    pub fn tasks(&self) -> EmbeddingTasks {
        EmbeddingTasks {
            names: self.lora_adaptations.clone(),
            instructions: self
                .lora_adaptations
                .iter()
                .map(|task| {
                    self.task_instructions
                        .get(task)
                        .cloned()
                        .unwrap_or_default()
                })
                .collect(),
        }
    }

    fn scaling(&self) -> f64 {
        self.lora_alpha / self.lora_rank as f64
    }
}

/// A linear layer stored as `parametrizations.weight.original` with its task adapters.
struct AdaptedLinear {
    base: Arc<dyn QuantMethod>,
    weight: Tensor,
    bias: Tensor,
    lora: TaskLora,
}

impl AdaptedLinear {
    fn new(cfg: &Config, in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let (tasks, rank) = (cfg.lora_adaptations.len(), cfg.lora_rank);
        let param = vb.pp(PARAMETRIZED).pp(WEIGHT);
        let weight = param.get((out_dim, in_dim), ORIGINAL)?;
        let bias = vb.get(out_dim, "bias")?;
        let adapters = param.pp(FIRST);
        let lora = TaskLora::new(
            adapters.get((tasks, rank, in_dim), "lora_A")?,
            adapters.get((tasks, out_dim, rank), "lora_B")?,
            cfg.scaling(),
        );
        let base = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(weight.clone(), Some(bias.clone())),
        ))?);
        Ok(Self {
            base,
            weight,
            bias,
            lora,
        })
    }

    fn forward(&self, xs: &Tensor, tasks: Option<&TaskRows>) -> Result<Tensor> {
        let ys = self.base.forward(xs)?;
        match tasks {
            Some(rows) => ys + self.lora.delta(xs, rows)?,
            None => Ok(ys),
        }
    }

    fn add_residual(&self, uvb: &UnVarBuilder) {
        add_parametrized(uvb, &self.weight, self.lora.a(), self.lora.b());
        uvb.add_tensor("bias", self.bias.clone());
    }
}

fn add_parametrized(uvb: &UnVarBuilder, weight: &Tensor, a: &Tensor, b: &Tensor) {
    let param = uvb.pp(PARAMETRIZED).pp(WEIGHT);
    param.add_tensor(ORIGINAL, weight.clone());
    param.pp(FIRST).add_tensor("lora_A", a.clone());
    param.pp(FIRST).add_tensor("lora_B", b.clone());
}

/// An embedding table stored as `parametrizations.weight.original` with its task adapters.
struct AdaptedEmbedding {
    table: Embedding,
    lora: TaskEmbeddingLora,
}

impl AdaptedEmbedding {
    fn new(cfg: &Config, rows: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let (tasks, rank, hidden) = (cfg.lora_adaptations.len(), cfg.lora_rank, cfg.hidden_size);
        let param = vb.pp(PARAMETRIZED).pp(WEIGHT);
        let adapters = param.pp(FIRST);
        Ok(Self {
            table: Embedding::new(param.get((rows, hidden), ORIGINAL)?, hidden),
            lora: TaskEmbeddingLora::new(
                adapters.get((tasks, rows, rank), "lora_A")?,
                adapters.get((tasks, rank, hidden), "lora_B")?,
                cfg.scaling(),
            ),
        })
    }

    fn forward(&self, ids: &Tensor, tasks: Option<&TaskRows>) -> Result<Tensor> {
        let xs = self.table.forward(ids)?;
        match tasks {
            Some(rows) => xs.clone() + self.lora.delta(ids, rows, xs.dtype())?,
            None => Ok(xs),
        }
    }

    fn add_residual(&self, uvb: &UnVarBuilder) {
        add_parametrized(uvb, self.table.embeddings(), self.lora.a(), self.lora.b());
    }
}

struct Layer {
    attn: RopeSelfAttention,
    // the attention's projections, kept to write them back under their parametrized names
    wqkv: AdaptedLinear,
    out_proj: AdaptedLinear,
    norm1: LayerNorm,
    fc1: AdaptedLinear,
    fc2: AdaptedLinear,
    norm2: LayerNorm,
}

impl Layer {
    fn forward(&self, xs: &Tensor, tasks: Option<&TaskRows>) -> Result<Tensor> {
        let attn = self.attn.forward_tasks(xs, &AttentionMask::None, tasks)?;
        let xs = self.norm1.forward(&(attn + xs)?)?;
        let ffn = self
            .fc2
            .forward(&self.fc1.forward(&xs, tasks)?.gelu_erf()?, tasks)?;
        self.norm2.forward(&(ffn + xs)?)
    }
}

pub struct Model {
    word: AdaptedEmbedding,
    token_type: AdaptedEmbedding,
    emb_ln: LayerNorm,
    layers: Vec<Layer>,
    prefix: Option<&'static str>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    device: Device,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            inference_tensor::bail!("Expected AttentionImplementation::Eager");
        }
        let prefix = (!vb.contains_tensor(EMB_LN)).then_some(BASE_PREFIX);
        let vb = match prefix {
            Some(prefix) => vb.pp(prefix),
            None => vb,
        };
        let NormalLoadingMetadata {
            mapper,
            real_device,
            ..
        } = normal_loading_metadata;
        let (hidden, eps) = (cfg.hidden_size, cfg.layer_norm_eps);
        let head_dim = hidden / cfg.num_attention_heads;
        let base_rotary = rotary(
            cfg.rotary_emb_base,
            head_dim,
            cfg.max_position_embeddings,
            &real_device,
            vb.dtype(),
        )?;
        let shape = RopeAttentionShape {
            hidden,
            num_heads: cfg.num_attention_heads,
            bias: true,
            layout: QkvLayout::Stacked,
            style: RopeStyle::NeoX,
        };
        let vb_l = vb.pp("encoder").pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb = mapper.set_device(i, vb_l.pp(i), false);
                let layer_rotary = match mapper.device_for(i, false) {
                    Some(device) if !device.same_device(&real_device) => rotary(
                        cfg.rotary_emb_base,
                        head_dim,
                        cfg.max_position_embeddings,
                        device,
                        vb.dtype(),
                    )?,
                    _ => base_rotary.clone(),
                };
                let wqkv = AdaptedLinear::new(cfg, hidden, 3 * hidden, vb.pp("mixer").pp("Wqkv"))?;
                let out_proj =
                    AdaptedLinear::new(cfg, hidden, hidden, vb.pp("mixer").pp("out_proj"))?;
                let attn = RopeSelfAttention::from_projections(
                    &shape,
                    layer_rotary,
                    wqkv.base.clone(),
                    out_proj.base.clone(),
                    Some((
                        TaskLora::new(wqkv.lora.a().clone(), wqkv.lora.b().clone(), cfg.scaling()),
                        TaskLora::new(
                            out_proj.lora.a().clone(),
                            out_proj.lora.b().clone(),
                            cfg.scaling(),
                        ),
                    )),
                );
                Ok(Layer {
                    attn,
                    wqkv,
                    out_proj,
                    norm1: layer_norm(hidden, eps, vb.pp("norm1"))?,
                    fc1: AdaptedLinear::new(
                        cfg,
                        hidden,
                        cfg.intermediate_size,
                        vb.pp("mlp").pp("fc1"),
                    )?,
                    fc2: AdaptedLinear::new(
                        cfg,
                        cfg.intermediate_size,
                        hidden,
                        vb.pp("mlp").pp("fc2"),
                    )?,
                    norm2: layer_norm(hidden, eps, vb.pp("norm2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let vb_e = mapper.set_nm_device(vb.pp("embeddings"), false);
        Ok(Self {
            word: AdaptedEmbedding::new(cfg, cfg.vocab_size, vb_e.pp("word_embeddings"))?,
            token_type: AdaptedEmbedding::new(
                cfg,
                cfg.type_vocab_size,
                vb_e.pp("token_type_embeddings"),
            )?,
            emb_ln: layer_norm(hidden, eps, mapper.set_nm_device(vb.pp("emb_ln"), false))?,
            layers,
            prefix,
            mapper,
            device: real_device,
        })
    }
}

impl IsqModel for Model {
    // nothing here is quantized in place, so every tensor goes back under its own name
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let root = UnVarBuilder::new();
        let prefixed = self.prefix.map(|prefix| root.pp(prefix));
        let uvb = prefixed.as_ref().unwrap_or(&root);
        let uvb_e = uvb.pp("embeddings");
        self.word.add_residual(&uvb_e.pp("word_embeddings"));
        self.token_type
            .add_residual(&uvb_e.pp("token_type_embeddings"));
        uvb.pp("emb_ln").add(&self.emb_ln);
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("encoder").pp("layers").pp(i);
            layer.wqkv.add_residual(&uvb_l.pp("mixer").pp("Wqkv"));
            layer
                .out_proj
                .add_residual(&uvb_l.pp("mixer").pp("out_proj"));
            layer.fc1.add_residual(&uvb_l.pp("mlp").pp("fc1"));
            layer.fc2.add_residual(&uvb_l.pp("mlp").pp("fc2"));
            uvb_l.pp("norm1").add(&layer.norm1);
            uvb_l.pp("norm2").add(&layer.norm2);
        }
        root.to_safetensors()
    }
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        extras: &EmbeddingExtras,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let tasks = extras.tasks.as_ref();
        let xs = self.word.forward(input_ids, tasks)?;
        // one token type: a pair's segment ids all read row 0
        let types = input_ids.zeros_like()?;
        let xs = (xs + self.token_type.forward(&types, tasks)?)?;
        let mut xs = self.emb_ln.forward(&xs)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let rows = tasks
                .map(|rows| -> Result<TaskRows> {
                    Ok(TaskRows {
                        ids: rows.ids.to_device(xs.device())?,
                        active: rows.active.to_device(xs.device())?,
                    })
                })
                .transpose()?;
            xs = layer.forward(&xs, rows.as_ref())?;
        }
        xs.to_device(&self.device)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for Model {}
