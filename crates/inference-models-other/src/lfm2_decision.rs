//! LiquidAI's d1-omni (llama.cpp's `lfm2-d1-omni` decision type): an LFM2 trunk read bidirectionally (centred
//! three-tap short convolutions, non-causal attention), then laya's decision head (a question-type embedding, a few
//! pre-norm blocks with biases, and a scorer over each option's `[MASK]` marker). It loads from llama.cpp's GGUF under
//! its tensor names. The logits are left uncalibrated: the decision profile applies the temperatures.
#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear};
use inference_tensor::nn::{Embedding, LayerNorm, LayerNormConfig, Module};
use inference_tensor::{D, DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams, SdpaParams},
    layers::{
        RmsNorm, RotaryEmbedding, Sdpa, apply_rotary_preselected_qk, dense_embedding, layer_norm,
    },
    model::{EmbeddingModel, IsqModel, NormalLoadingMetadata},
    modern_bert_decision::{self, Block},
    utils::unvarbuilder::UnVarBuilder,
};

const BLOCKS: &str = "blk";
const QUESTION_TYPES: usize = 3;
// the short convolutions' taps: the previous token, the token, the next one
const CONV_TAPS: usize = 3;

/// The trunk and head sizes, from the GGUF's `lfm2.*` keys.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    /// per block, the trunk's then the head's; 0 marks a short-convolution block
    pub num_key_value_heads: Vec<usize>,
    pub intermediate_sizes: Vec<usize>,
    pub head_layers: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub layer_norm_eps: f64,
}

impl Config {
    pub fn trunk_layers(&self) -> usize {
        self.num_key_value_heads.len() - self.head_layers
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// `out(c * conv(b * x))` over `[b | c | x] = in_proj(xs)`, the convolution centred on each token.
struct ShortConv {
    in_proj: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,
    taps: [Tensor; CONV_TAPS],
}

impl ShortConv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, width) = xs.dims3()?;
        let bcx = self.in_proj.forward(xs)?;
        let part = |i: usize| bcx.narrow(D::Minus1, i * width, width);
        let bx = (part(0)? * part(2)?)?;
        let mut y = bx.broadcast_mul(&self.taps[1])?;
        if seq_len > 1 {
            // a tap outside the sequence reads zero
            let zero = Tensor::zeros((b_sz, 1, width), bx.dtype(), bx.device())?;
            let prev = Tensor::cat(&[&zero, &bx.narrow(1, 0, seq_len - 1)?], 1)?;
            let next = Tensor::cat(&[&bx.narrow(1, 1, seq_len - 1)?, &zero], 1)?;
            y = (y + prev.broadcast_mul(&self.taps[0])?)?;
            y = (y + next.broadcast_mul(&self.taps[2])?)?;
        }
        self.out_proj.forward(&(part(1)? * y)?)
    }
}

/// Grouped-query attention over the whole prompt, with RMS-normed q and k before NeoX rotary positions.
struct Attention {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    v: Arc<dyn QuantMethod>,
    o: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary: Arc<RotaryEmbedding>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, width) = xs.dims3()?;
        let heads = |proj: &Arc<dyn QuantMethod>, n: usize, norm: Option<&RmsNorm>| {
            let x = proj
                .forward(xs)?
                .reshape((b_sz, seq_len, n, self.head_dim))?;
            let x = match norm {
                Some(norm) => norm.forward(&x)?,
                None => x,
            };
            x.transpose(1, 2)?.contiguous()
        };
        let q = heads(&self.q, self.heads, Some(&self.q_norm))?;
        let k = heads(&self.k, self.kv_heads, Some(&self.k_norm))?;
        let v = heads(&self.v, self.kv_heads, None)?;
        let (cos, sin) = self.rotary.get_cos_sin()?;
        let (q, k) = apply_rotary_preselected_qk(
            &q,
            &k,
            &cos.narrow(0, 0, seq_len)?,
            &sin.narrow(0, 0, seq_len)?,
            true,
        )?;
        let params = SdpaParams {
            n_kv_groups: self.heads / self.kv_heads,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
            sinks: None,
            chunk: None,
        };
        let attn = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::None,
            Some(&FlashParams::empty(false)),
            &params,
        )?;
        self.o
            .forward(&attn.transpose(1, 2)?.reshape((b_sz, seq_len, width))?)
    }
}

enum Mixer {
    Conv(ShortConv),
    Attention(Attention),
}

struct TrunkLayer {
    norm: RmsNorm,
    mixer: Mixer,
    ffn_norm: RmsNorm,
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl TrunkLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(xs)?;
        let mixed = match &self.mixer {
            Mixer::Conv(conv) => conv.forward(&normed)?,
            Mixer::Attention(attn) => attn.forward(&normed)?,
        };
        let xs = (xs + mixed)?;
        let normed = self.ffn_norm.forward(&xs)?;
        let gated = (self.gate.forward(&normed)?.silu()? * self.up.forward(&normed)?)?;
        xs + self.down.forward(&gated)?
    }
}

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<TrunkLayer>,
    norm: RmsNorm,
    type_emb: Embedding,
    blocks: Vec<Block>,
    scorer_norm: LayerNorm,
    scorer_hidden: Arc<dyn QuantMethod>,
    scorer_out: Arc<dyn QuantMethod>,
    device: Device,
}

fn linear(i: usize, o: usize, bias: bool, vb: ShardedVarBuilder) -> Result<Arc<dyn QuantMethod>> {
    inference_quant::linear_b(i, o, bias, &None, vb)
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let mapper = &normal_loading_metadata.mapper;
        let device = normal_loading_metadata.real_device.clone();
        let width = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let layers = (0..cfg.trunk_layers())
            .map(|i| {
                let vb = mapper.set_device(i, vb.pp(BLOCKS).pp(i), false);
                let kv_heads = cfg.num_key_value_heads[i];
                let mixer = match kv_heads {
                    0 => {
                        let conv = vb.pp("shortconv");
                        let weight = conv.pp("conv").get((width, CONV_TAPS), "weight")?;
                        let tap = |k: usize| weight.narrow(1, k, 1)?.squeeze(1);
                        Mixer::Conv(ShortConv {
                            in_proj: linear(width, 3 * width, false, conv.pp("in_proj"))?,
                            out_proj: linear(width, width, false, conv.pp("out_proj"))?,
                            taps: [tap(0)?, tap(1)?, tap(2)?],
                        })
                    }
                    kv_heads => Mixer::Attention(Attention {
                        q: linear(
                            width,
                            cfg.num_attention_heads * head_dim,
                            false,
                            vb.pp("attn_q"),
                        )?,
                        k: linear(width, kv_heads * head_dim, false, vb.pp("attn_k"))?,
                        v: linear(width, kv_heads * head_dim, false, vb.pp("attn_v"))?,
                        o: linear(
                            cfg.num_attention_heads * head_dim,
                            width,
                            false,
                            vb.pp("attn_output"),
                        )?,
                        q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("attn_q_norm"))?,
                        k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("attn_k_norm"))?,
                        rotary: Arc::new(RotaryEmbedding::new(
                            cfg.rope_theta,
                            head_dim,
                            cfg.max_position_embeddings,
                            vb.device(),
                            true,
                            vb.dtype(),
                        )?),
                        heads: cfg.num_attention_heads,
                        kv_heads,
                        head_dim,
                    }),
                };
                let ff = cfg.intermediate_sizes[i];
                Ok(TrunkLayer {
                    norm: RmsNorm::new(width, cfg.rms_norm_eps, vb.pp("attn_norm"))?,
                    mixer,
                    ffn_norm: RmsNorm::new(width, cfg.rms_norm_eps, vb.pp("ffn_norm"))?,
                    gate: linear(width, ff, false, vb.pp("ffn_gate"))?,
                    up: linear(width, ff, false, vb.pp("ffn_up"))?,
                    down: linear(ff, width, false, vb.pp("ffn_down"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let head = mapper.set_nm_device(vb.clone(), false);
        let norm = |vb: ShardedVarBuilder| {
            layer_norm(
                width,
                LayerNormConfig {
                    eps: cfg.layer_norm_eps,
                    remove_mean: true,
                    affine: true,
                },
                vb,
            )
        };
        let blocks = (cfg.trunk_layers()..cfg.num_key_value_heads.len())
            .map(|i| {
                let vb = head.pp(BLOCKS).pp(i);
                let ff = cfg.intermediate_sizes[i];
                Ok(Block::from_parts(
                    (norm(vb.pp("attn_norm"))?, norm(vb.pp("ffn_norm"))?),
                    (
                        linear(width, 3 * width, true, vb.pp("attn_qkv"))?,
                        linear(width, width, true, vb.pp("attn_output"))?,
                    ),
                    (
                        linear(width, ff, true, vb.pp("ffn_up"))?,
                        linear(ff, width, true, vb.pp("ffn_down"))?,
                    ),
                    width,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let cls = head.pp("cls");
        // a 1-D GGUF tensor, the scorer's last projection to one logit
        let scorer_out = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            inference_tensor::nn::Linear::new(
                cls.pp("output").get(width, "weight")?.unsqueeze(0)?,
                Some(cls.pp("output").get(1, "bias")?),
            ),
        ))?);
        Ok(Self {
            embed_tokens: dense_embedding(cfg.vocab_size, width, vb.pp("token_embd"), &None)?,
            layers,
            norm: RmsNorm::new(width, cfg.rms_norm_eps, head.pp("token_embd_norm"))?,
            type_emb: Embedding::new(
                head.pp("token_types")
                    .get((QUESTION_TYPES, width), "weight")?,
                width,
            ),
            blocks,
            scorer_norm: norm(cls.pp("norm"))?,
            scorer_hidden: linear(width, width, true, cls.clone())?,
            scorer_out,
            device,
        })
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let root = UnVarBuilder::new();
        root.pp("token_embd").add(&self.embed_tokens);
        root.pp("token_embd_norm").add(&self.norm);
        root.pp("token_types").add(&self.type_emb);
        root.pp("cls").pp("norm").add(&self.scorer_norm);
        root.to_safetensors()
    }
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        extras: &crate::embedding::EmbeddingExtras,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let Some(rows) = &extras.decision else {
            inference_tensor::bail!("a decision model answers decision questions only");
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        let xs = self.norm.forward(&xs)?;
        let kinds = Tensor::new(rows.kinds.as_slice(), xs.device())?;
        let mut xs = xs.broadcast_add(&self.type_emb.forward(&kinds)?.unsqueeze(1)?)?;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        let picked = modern_bert_decision::Model::markers(&xs, rows)?;
        let hidden = self
            .scorer_hidden
            .forward(&self.scorer_norm.forward(&picked)?)?
            .gelu_erf()?;
        self.scorer_out
            .forward(&hidden)?
            .squeeze(D::Minus1)?
            .to_dtype(DType::F32)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::decision::DecisionRows;

    const WIDTH: usize = 8;
    const HEADS: usize = 2;
    const VOCAB: usize = 20;

    // a short-convolution block, an attention block (one KV head), then one head block
    fn config() -> Config {
        Config {
            vocab_size: VOCAB,
            hidden_size: WIDTH,
            num_attention_heads: HEADS,
            num_key_value_heads: vec![0, 1, HEADS],
            intermediate_sizes: vec![16, 16, 4 * WIDTH],
            head_layers: 1,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            layer_norm_eps: 1e-5,
        }
    }

    fn weights(cfg: &Config) -> HashMap<String, Tensor> {
        let head_dim = WIDTH / HEADS;
        let mut shapes: Vec<(String, Vec<usize>)> = vec![
            ("token_embd.weight".into(), vec![VOCAB, WIDTH]),
            ("token_embd_norm.weight".into(), vec![WIDTH]),
            ("token_types.weight".into(), vec![3, WIDTH]),
            ("cls.norm.weight".into(), vec![WIDTH]),
            ("cls.norm.bias".into(), vec![WIDTH]),
            ("cls.weight".into(), vec![WIDTH, WIDTH]),
            ("cls.bias".into(), vec![WIDTH]),
            ("cls.output.weight".into(), vec![WIDTH]),
            ("cls.output.bias".into(), vec![1]),
        ];
        let mut add = |name: String, shape: Vec<usize>| shapes.push((name, shape));
        for (i, &kv) in cfg.num_key_value_heads.iter().enumerate() {
            let b = format!("blk.{i}");
            let ff = cfg.intermediate_sizes[i];
            if i >= cfg.trunk_layers() {
                for norm in ["attn_norm", "ffn_norm"] {
                    add(format!("{b}.{norm}.weight"), vec![WIDTH]);
                    add(format!("{b}.{norm}.bias"), vec![WIDTH]);
                }
                for (name, shape) in [
                    ("attn_qkv", [3 * WIDTH, WIDTH]),
                    ("attn_output", [WIDTH, WIDTH]),
                    ("ffn_up", [ff, WIDTH]),
                    ("ffn_down", [WIDTH, ff]),
                ] {
                    add(format!("{b}.{name}.weight"), shape.to_vec());
                    add(format!("{b}.{name}.bias"), vec![shape[0]]);
                }
                continue;
            }
            add(format!("{b}.attn_norm.weight"), vec![WIDTH]);
            add(format!("{b}.ffn_norm.weight"), vec![WIDTH]);
            add(format!("{b}.ffn_gate.weight"), vec![ff, WIDTH]);
            add(format!("{b}.ffn_up.weight"), vec![ff, WIDTH]);
            add(format!("{b}.ffn_down.weight"), vec![WIDTH, ff]);
            match kv {
                0 => {
                    add(format!("{b}.shortconv.conv.weight"), vec![WIDTH, CONV_TAPS]);
                    add(
                        format!("{b}.shortconv.in_proj.weight"),
                        vec![3 * WIDTH, WIDTH],
                    );
                    add(format!("{b}.shortconv.out_proj.weight"), vec![WIDTH, WIDTH]);
                }
                kv => {
                    add(format!("{b}.attn_q.weight"), vec![WIDTH, WIDTH]);
                    add(format!("{b}.attn_k.weight"), vec![kv * head_dim, WIDTH]);
                    add(format!("{b}.attn_v.weight"), vec![kv * head_dim, WIDTH]);
                    add(format!("{b}.attn_output.weight"), vec![WIDTH, WIDTH]);
                    add(format!("{b}.attn_q_norm.weight"), vec![head_dim]);
                    add(format!("{b}.attn_k_norm.weight"), vec![head_dim]);
                }
            }
        }
        shapes
            .into_iter()
            .enumerate()
            .map(|(seed, (name, shape))| {
                let n = shape.iter().product::<usize>();
                let values = (0..n)
                    .map(|i| ((i * 7 + seed * 13) as f32 * 0.37).sin() * 0.5)
                    .collect::<Vec<_>>();
                (name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap())
            })
            .collect()
    }

    fn model() -> Model {
        let cfg = config();
        let vb = inference_quant::ShardedSafeTensors::wrap(weights(&cfg), DType::F32, Device::Cpu);
        let metadata = NormalLoadingMetadata {
            mapper: Box::new(crate::device_map::DummyDeviceMapper {
                nm_device: Device::Cpu,
            }),
            loading_isq: false,
            real_device: Device::Cpu,
            multi_progress: Arc::new(indicatif::MultiProgress::new()),
            matformer_slicing_config: None,
            rope_pairing: None,
        };
        Model::new(&cfg, vb, metadata).unwrap()
    }

    fn score(model: &Model, tokens: &[u32]) -> f32 {
        let extras = crate::embedding::EmbeddingExtras {
            decision: Some(DecisionRows {
                kinds: vec![0],
                markers: vec![vec![1]],
            }),
            ..Default::default()
        };
        let ids = Tensor::new(tokens, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        model
            .forward(&ids, &extras, &FlashParams::empty(false))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0]
    }

    // the trunk reads both ways: a marker's score moves with a token after it, which a causal trunk never sees
    #[test]
    fn a_marker_reads_the_tokens_after_it() {
        let model = model();
        let base = score(&model, &[3, 4, 5, 6, 7]);
        assert!(base.is_finite());
        assert_ne!(score(&model, &[3, 4, 5, 6, 9]), base);
        // and the same prompt scores the same
        assert_eq!(score(&model, &[3, 4, 5, 6, 7]), base);
    }
}
