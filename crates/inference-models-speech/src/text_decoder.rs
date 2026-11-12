//! An encoder-decoder's text decoder (Whisper, Cohere Transcribe): a position table added to the token embedding,
//! pre-norm layers of causal self-attention, cross-attention over the encoder output and a two-layer FFN, a final
//! LayerNorm, and an LM head of its own or tied to the token embedding.

use inference_nn::layers::{Activation, layer_norm, linear_b};
use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{DType, Result, Tensor};

const LAYER_NORM_EPS: f64 = 1e-5;

/// Where a checkpoint keeps each part, under the decoder's prefix.
pub struct DecoderNames {
    pub layers: &'static str,
    pub self_attn: &'static str,
    pub self_attn_norm: &'static str,
    pub cross_attn: &'static str,
    pub cross_attn_norm: &'static str,
    pub ffn_in: &'static str,
    pub ffn_out: &'static str,
    pub ffn_norm: &'static str,
    pub final_norm: &'static str,
    pub embed_tokens: &'static str,
    pub embed_positions: &'static str,
    /// A LayerNorm over the embedding plus positions, before the layers.
    pub embed_norm: Option<&'static str>,
    /// Query, key, value and output projections.
    pub projections: [&'static str; 4],
}

pub const WHISPER_NAMES: DecoderNames = DecoderNames {
    layers: "layers",
    self_attn: "self_attn",
    self_attn_norm: "self_attn_layer_norm",
    cross_attn: "encoder_attn",
    cross_attn_norm: "encoder_attn_layer_norm",
    ffn_in: "fc1",
    ffn_out: "fc2",
    ffn_norm: "final_layer_norm",
    final_norm: "layer_norm",
    embed_tokens: "embed_tokens.weight",
    embed_positions: "embed_positions.weight",
    embed_norm: None,
    projections: ["q_proj", "k_proj", "v_proj", "out_proj"],
};

/// NeMo's `TransformerDecoder`, as Cohere Transcribe keeps it.
pub const NEMO_NAMES: DecoderNames = DecoderNames {
    layers: "_decoder.layers",
    self_attn: "first_sub_layer",
    self_attn_norm: "layer_norm_1",
    cross_attn: "second_sub_layer",
    cross_attn_norm: "layer_norm_2",
    ffn_in: "third_sub_layer.dense_in",
    ffn_out: "third_sub_layer.dense_out",
    ffn_norm: "layer_norm_3",
    final_norm: "_decoder.final_layer_norm",
    embed_tokens: "_embedding.token_embedding.weight",
    embed_positions: "_embedding.position_embedding.pos_enc",
    embed_norm: Some("_embedding.layer_norm"),
    projections: ["query_net", "key_net", "value_net", "out_projection"],
};

pub struct DecoderConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub activation: Activation,
    /// Whisper's key projections have no bias.
    pub key_bias: bool,
    pub names: &'static DecoderNames,
    /// What the model is called in errors.
    pub model: &'static str,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &DecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let [q, k, v, out] = cfg.names.projections;
        Ok(Self {
            q_proj: linear_b(d, d, true, vb.pp(q))?,
            k_proj: linear_b(d, d, cfg.key_bias, vb.pp(k))?,
            v_proj: linear_b(d, d, true, vb.pp(v))?,
            out_proj: linear_b(d, d, true, vb.pp(out))?,
            num_heads: cfg.num_heads,
            head_dim: d / cfg.num_heads,
        })
    }

    // (batch, tokens, d) to (batch, heads, tokens, head_dim)
    fn heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        xs.reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn key_value(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            self.heads(&self.k_proj.forward(xs)?)?,
            self.heads(&self.v_proj.forward(xs)?)?,
        ))
    }

    // the scaled scores, before any mask or softmax, and the attended output
    fn attend(
        &self,
        xs: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, t, d) = xs.dims3()?;
        let q = self.heads(&self.q_proj.forward(xs)?)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = (q.matmul(&k.t()?)?.to_dtype(DType::F32)? * scale)?;
        let masked = match mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores.clone(),
        };
        let probs = inference_tensor::nn::ops::softmax_last_dim(&masked)?.to_dtype(v.dtype())?;
        let out = probs.matmul(v)?.transpose(1, 2)?.reshape((b, t, d))?;
        Ok((scores, self.out_proj.forward(&out)?))
    }
}

struct DecoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: Attention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

/// One window's decoder state: each layer's self-attention keys and values so far, and its cross-attention ones.
#[derive(Clone)]
pub struct DecoderCache {
    self_kv: Vec<Option<(Tensor, Tensor)>>,
    cross_kv: Vec<(Tensor, Tensor)>,
    len: usize,
}

impl DecoderCache {
    pub fn len(&self) -> usize {
        self.len
    }
}

pub struct Decoder {
    embed_tokens: Tensor,
    positions: Tensor,
    embed_norm: Option<LayerNorm>,
    layers: Vec<DecoderLayer>,
    layer_norm: LayerNorm,
    // none ties the head to the token embedding
    head: Option<Linear>,
    activation: Activation,
    max_positions: usize,
    model: &'static str,
}

/// A step's logits for each token, and, when asked, each layer's cross-attention scores `(heads, tokens, frames)`.
pub struct DecoderOutput {
    pub logits: Tensor,
    pub cross_scores: Vec<Tensor>,
}

impl Decoder {
    /// `head` is the LM head; none ties it to the token embedding.
    pub fn new(cfg: &DecoderConfig, vb: ShardedVarBuilder, head: Option<Linear>) -> Result<Self> {
        let d = cfg.d_model;
        let names = cfg.names;
        let layers = (0..cfg.num_layers)
            .map(|i| {
                let vb = vb.pp(names.layers).pp(i);
                Ok(DecoderLayer {
                    self_attn: Attention::new(cfg, vb.pp(names.self_attn))?,
                    self_attn_layer_norm: layer_norm(
                        d,
                        LAYER_NORM_EPS,
                        vb.pp(names.self_attn_norm),
                    )?,
                    encoder_attn: Attention::new(cfg, vb.pp(names.cross_attn))?,
                    encoder_attn_layer_norm: layer_norm(
                        d,
                        LAYER_NORM_EPS,
                        vb.pp(names.cross_attn_norm),
                    )?,
                    fc1: linear_b(d, cfg.ffn_dim, true, vb.pp(names.ffn_in))?,
                    fc2: linear_b(cfg.ffn_dim, d, true, vb.pp(names.ffn_out))?,
                    final_layer_norm: layer_norm(d, LAYER_NORM_EPS, vb.pp(names.ffn_norm))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: vb.get((cfg.vocab_size, d), names.embed_tokens)?,
            positions: vb.get((cfg.max_positions, d), names.embed_positions)?,
            embed_norm: names
                .embed_norm
                .map(|name| layer_norm(d, LAYER_NORM_EPS, vb.pp(name)))
                .transpose()?,
            layers,
            layer_norm: layer_norm(d, LAYER_NORM_EPS, vb.pp(names.final_norm))?,
            head,
            activation: cfg.activation,
            max_positions: cfg.max_positions,
            model: cfg.model,
        })
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// The cross-attention keys and values of an encoder output `(1, frames, d)`, computed once per window.
    pub fn cache(&self, encoded: &Tensor) -> Result<DecoderCache> {
        Ok(DecoderCache {
            self_kv: vec![None; self.layers.len()],
            cross_kv: self
                .layers
                .iter()
                .map(|layer| layer.encoder_attn.key_value(encoded))
                .collect::<Result<Vec<_>>>()?,
            len: 0,
        })
    }

    /// Runs `tokens` after the cache's, appending them to it. `cross_scores` keeps each layer's cross-attention
    /// scores, for alignment.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut DecoderCache,
        cross_scores: bool,
    ) -> Result<DecoderOutput> {
        let device = self.embed_tokens.device();
        let start = cache.len;
        let t = tokens.len();
        if start + t > self.max_positions {
            inference_tensor::bail!(
                "{}'s decoder takes {} tokens; this window reached {}",
                self.model,
                self.max_positions,
                start + t
            );
        }
        let ids = Tensor::new(tokens, device)?;
        let mut xs = self
            .embed_tokens
            .index_select(&ids, 0)?
            .broadcast_add(&self.positions.narrow(0, start, t)?)?
            .unsqueeze(0)?;
        if let Some(norm) = &self.embed_norm {
            xs = norm.forward(&xs)?;
        }
        // each new token sees the cache and the new tokens up to itself
        let mask = (t > 1)
            .then(|| {
                let total = start + t;
                let mask: Vec<f32> = (0..t)
                    .flat_map(|i| {
                        (0..total).map(move |j| {
                            if j <= start + i {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect();
                Tensor::from_vec(mask, (1, 1, t, total), device)
            })
            .transpose()?;
        let mut scores = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = layer.self_attn_layer_norm.forward(&xs)?;
            let (k, v) = layer.self_attn.key_value(&normed)?;
            let (k, v) = match &cache.self_kv[i] {
                Some((past_k, past_v)) => (
                    Tensor::cat(&[past_k, &k], 2)?,
                    Tensor::cat(&[past_v, &v], 2)?,
                ),
                None => (k, v),
            };
            let (_, attended) = layer.self_attn.attend(&normed, &k, &v, mask.as_ref())?;
            cache.self_kv[i] = Some((k, v));
            xs = (xs + attended)?;
            let normed = layer.encoder_attn_layer_norm.forward(&xs)?;
            let (cross_k, cross_v) = &cache.cross_kv[i];
            let (layer_scores, attended) =
                layer.encoder_attn.attend(&normed, cross_k, cross_v, None)?;
            if cross_scores {
                scores.push(layer_scores.squeeze(0)?);
            }
            xs = (xs + attended)?;
            let normed = layer.final_layer_norm.forward(&xs)?;
            let ffn = layer
                .fc2
                .forward(&self.activation.forward(&layer.fc1.forward(&normed)?)?)?;
            xs = (xs + ffn)?;
        }
        cache.len += t;
        let hidden = self.layer_norm.forward(&xs)?;
        let hidden = hidden.squeeze(0)?;
        let logits = match &self.head {
            Some(head) => head.forward(&hidden)?,
            None => hidden.matmul(&self.embed_tokens.t()?)?,
        }
        .to_dtype(DType::F32)?;
        Ok(DecoderOutput {
            logits,
            cross_scores: scores,
        })
    }

    pub fn last_logits(output: &DecoderOutput) -> Result<Vec<f32>> {
        output
            .logits
            .get(output.logits.dim(0)? - 1)?
            .to_vec1::<f32>()
    }
}
