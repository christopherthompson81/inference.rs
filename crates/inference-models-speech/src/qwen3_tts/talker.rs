//! The talker (a Qwen3 decoder predicting each frame's first code) and the code predictor (a small Qwen3 decoder
//! filling the frame's other codebooks), both over the shared decoder stack with caches of their own.

use std::sync::Arc;

use indicatif::MultiProgress;
use inference_nn::attention::FlashParams;
use inference_nn::decoder::{DecoderSpec, DecoderStack, ModuleNames, QkNorm, RopeKind};
use inference_nn::device_map::DummyDeviceMapper;
use inference_nn::kv_cache::KvCache;
use inference_nn::model::{ModelForwardContext, NormalLoadingMetadata};
use inference_nn::paged_attention::AttentionImplementation;
use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Embedding, Linear, Module};
use inference_tensor::{DType, Device, IndexOp, Result, Tensor};

use super::config::{CodePredictorConfig, TalkerConfig};
use crate::layers::Activation;

// the cache's first allocation, in positions; it grows past this on demand
const CACHE_CAPACITY: usize = 512;
const KV_DIM: usize = 2;

// one Qwen3 stack's shape, read from the talker's or the code predictor's config
struct Qwen3Stack {
    layers: usize,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    act: Activation,
    eps: f64,
    theta: f64,
    max_positions: usize,
    vocab: usize,
}

impl Qwen3Stack {
    // the stack's token embedding is `embed` under its prefix, `embed_dim` wide when not the stack's width
    fn spec(&self, embed: &'static str, embed_dim: Option<usize>) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab,
            hidden_size: self.hidden,
            embed_dim,
            intermediate_size: self.intermediate,
            num_heads: self.heads,
            num_kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            hidden_act: self.act,
            rms_norm_eps: self.eps,
            #[allow(clippy::cast_possible_truncation)]
            rope: RopeKind::Default {
                theta: self.theta as f32,
            },
            max_position_embeddings: self.max_positions,
            qk_norm: Some(QkNorm::BEFORE_ROPE),
            layer_windows: vec![None; self.layers],
            module_names: ModuleNames {
                embed_tokens: embed,
                ..ModuleNames::HF
            },
            ..Default::default()
        }
    }
}

fn metadata(device: &Device) -> NormalLoadingMetadata {
    NormalLoadingMetadata {
        mapper: Box::new(DummyDeviceMapper {
            nm_device: device.clone(),
        }),
        loading_isq: false,
        real_device: device.clone(),
        multi_progress: Arc::new(MultiProgress::new()),
        matformer_slicing_config: None,
        rope_pairing: None,
    }
}

fn linear(in_dim: usize, out_dim: usize, bias: bool, vb: &ShardedVarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = bias.then(|| vb.get(out_dim, "bias")).transpose()?;
    Ok(Linear::new(weight, bias))
}

fn embedding(rows: usize, dim: usize, vb: &ShardedVarBuilder) -> Result<Embedding> {
    Ok(Embedding::new(vb.get((rows, dim), "weight")?, dim))
}

/// One stack's KV cache, a layer each.
pub struct StackCache(Vec<KvCache>);

impl StackCache {
    fn new(layers: usize, max_positions: usize) -> Self {
        Self(
            (0..layers)
                .map(|_| KvCache::new_normal(KV_DIM, max_positions, CACHE_CAPACITY))
                .collect(),
        )
    }

    pub fn reset(&mut self) {
        self.0.iter_mut().for_each(KvCache::reset);
    }
}

// the normed hidden states of `xs` (`(1, tokens, hidden)`) after the `offset` positions already in `cache`
fn run_stack(
    stack: &DecoderStack,
    xs: Tensor,
    cache: &mut StackCache,
    offset: usize,
) -> Result<Tensor> {
    let tokens = xs.dim(1)?;
    let shape = Tensor::zeros((1, tokens), DType::U32, xs.device())?;
    let masks = stack.masks(&shape, xs.dtype(), &cache.0, offset == 0)?;
    let offsets = [offset];
    let context_lens = [(0, tokens)];
    let position_ids = [offset + tokens];
    let flash = FlashParams::empty(true);
    let mut ctx = ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash);
    stack.forward(xs, &masks, Some(&mut cache.0), &mut ctx)
}

pub struct CodePredictor {
    stack: DecoderStack,
    // codebook g's input table is `tables[g - 1]`, in the talker's width; the stack's own embedding is table 0
    tables: Vec<Embedding>,
    heads: Vec<Linear>,
    project: Option<Linear>,
    layers: usize,
    max_positions: usize,
}

impl CodePredictor {
    fn new(
        cfg: &CodePredictorConfig,
        talker_hidden: usize,
        groups: usize,
        vb: &ShardedVarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let spec = Qwen3Stack {
            layers: cfg.num_hidden_layers,
            hidden: cfg.hidden_size,
            intermediate: cfg.intermediate_size,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            act: cfg.hidden_act,
            eps: cfg.rms_norm_eps,
            theta: cfg.rope_theta,
            max_positions: cfg.max_position_embeddings,
            vocab: cfg.vocab_size,
        }
        .spec("codec_embedding.0", Some(talker_hidden));
        let vb_m = vb.pp("model");
        let stack = DecoderStack::new(
            &spec,
            vb_m.clone(),
            None,
            true,
            metadata(device),
            &AttentionImplementation::Eager,
        )?;
        let tables = (1..groups - 1)
            .map(|i| {
                embedding(
                    cfg.vocab_size,
                    talker_hidden,
                    &vb_m.pp("codec_embedding").pp(i),
                )
            })
            .collect::<Result<_>>()?;
        let heads = (0..groups - 1)
            .map(|i| {
                linear(
                    cfg.hidden_size,
                    cfg.vocab_size,
                    false,
                    &vb.pp("lm_head").pp(i),
                )
            })
            .collect::<Result<_>>()?;
        let project = (cfg.hidden_size != talker_hidden)
            .then(|| {
                linear(
                    talker_hidden,
                    cfg.hidden_size,
                    true,
                    &vb.pp("small_to_mtp_projection"),
                )
            })
            .transpose()?;
        Ok(Self {
            stack,
            tables,
            heads,
            project,
            layers: cfg.num_hidden_layers,
            max_positions: cfg.max_position_embeddings,
        })
    }

    pub fn new_cache(&self) -> StackCache {
        StackCache::new(self.layers, self.max_positions)
    }

    pub fn groups(&self) -> usize {
        self.heads.len() + 1
    }

    /// Codebook `group`'s (1-based) embedding of `codes`, in the talker's width.
    pub fn embed(&self, group: usize, codes: &Tensor) -> Result<Tensor> {
        match group {
            1 => self.stack.embed(codes),
            g => self.tables[g - 2].forward(codes),
        }
    }

    fn project(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.project {
            Some(p) => p.forward(xs),
            None => Ok(xs.clone()),
        }
    }

    /// Group `group`'s logits `(1, vocab)` after feeding `xs` (talker width) at `offset`.
    pub fn step(
        &self,
        xs: &Tensor,
        group: usize,
        cache: &mut StackCache,
        offset: usize,
    ) -> Result<Tensor> {
        let hidden = run_stack(&self.stack, self.project(xs)?, cache, offset)?;
        let last = hidden.i((.., hidden.dim(1)? - 1, ..))?;
        self.heads[group - 1].forward(&last)
    }
}

pub struct Talker {
    stack: DecoderStack,
    head: Linear,
    text_embedding: Embedding,
    // the text projection's two linears; none when the table holds projected rows (llama.cpp's GGUF folds them)
    text_projection: Option<(Linear, Linear)>,
    text_act: Activation,
    pub predictor: CodePredictor,
    layers: usize,
    max_positions: usize,
}

impl Talker {
    /// `vb` is the checkpoint's `talker` prefix.
    pub fn new(cfg: &TalkerConfig, vb: &ShardedVarBuilder, device: &Device) -> Result<Self> {
        let spec = Qwen3Stack {
            layers: cfg.num_hidden_layers,
            hidden: cfg.hidden_size,
            intermediate: cfg.intermediate_size,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            act: cfg.hidden_act,
            eps: cfg.rms_norm_eps,
            theta: cfg.rope_theta,
            max_positions: cfg.max_position_embeddings,
            vocab: cfg.vocab_size,
        }
        .spec("codec_embedding", None);
        let vb_m = vb.pp("model");
        let stack = DecoderStack::new(
            &spec,
            vb_m.clone(),
            None,
            true,
            metadata(device),
            &AttentionImplementation::Eager,
        )?;
        let projection = vb.pp("text_projection");
        let text_projection = projection
            .contains_tensor("linear_fc1.weight")
            .then(|| -> Result<(Linear, Linear)> {
                Ok((
                    linear(
                        cfg.text_hidden_size,
                        cfg.text_hidden_size,
                        true,
                        &projection.pp("linear_fc1"),
                    )?,
                    linear(
                        cfg.text_hidden_size,
                        cfg.hidden_size,
                        true,
                        &projection.pp("linear_fc2"),
                    )?,
                ))
            })
            .transpose()?;
        let text_width = if text_projection.is_some() {
            cfg.text_hidden_size
        } else {
            cfg.hidden_size
        };
        Ok(Self {
            stack,
            head: linear(cfg.hidden_size, cfg.vocab_size, false, &vb.pp("codec_head"))?,
            text_embedding: embedding(cfg.text_vocab_size, text_width, &vb_m.pp("text_embedding"))?,
            text_projection,
            text_act: cfg.hidden_act,
            predictor: CodePredictor::new(
                &cfg.code_predictor_config,
                cfg.hidden_size,
                cfg.num_code_groups,
                &vb.pp("code_predictor"),
                device,
            )?,
            layers: cfg.num_hidden_layers,
            max_positions: cfg.max_position_embeddings,
        })
    }

    pub fn new_cache(&self) -> StackCache {
        StackCache::new(self.layers, self.max_positions)
    }

    pub fn dtype(&self) -> DType {
        self.head.weight().dtype()
    }

    /// Text ids `(1, n)` through the text embedding and its projection, `(1, n, hidden)`.
    pub fn text(&self, ids: &Tensor) -> Result<Tensor> {
        let xs = self.text_embedding.forward(ids)?;
        match &self.text_projection {
            Some((fc1, fc2)) => fc2.forward(&self.text_act.forward(&fc1.forward(&xs)?)?),
            None => Ok(xs),
        }
    }

    /// Codec ids `(1, n)` through the talker's codec embedding.
    pub fn codec(&self, ids: &Tensor) -> Result<Tensor> {
        self.stack.embed(ids)
    }

    /// The last position's normed hidden state `(1, 1, hidden)` and codec logits `(1, vocab)` after `xs` at `offset`.
    pub fn step(
        &self,
        xs: Tensor,
        cache: &mut StackCache,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = run_stack(&self.stack, xs, cache, offset)?;
        let last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?;
        let logits = self.head.forward(&last.squeeze(1)?)?;
        Ok((last, logits))
    }
}
