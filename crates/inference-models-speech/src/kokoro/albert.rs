use inference_tensor::nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm, linear, ops,
};
use inference_tensor::{D, DType, Result, Tensor};

use super::config::PlBertConfig;

// transformers' AlbertConfig defaults, which the release keeps
const EMBEDDING_SIZE: usize = 128;
const LAYER_NORM_EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
struct AlbertLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    attention_norm: LayerNorm,
    ffn: Linear,
    ffn_output: Linear,
    full_layer_norm: LayerNorm,
    heads: usize,
}

impl AlbertLayer {
    fn new(cfg: &PlBertConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let att = vb.pp("attention");
        Ok(Self {
            query: linear(h, h, att.pp("query"))?,
            key: linear(h, h, att.pp("key"))?,
            value: linear(h, h, att.pp("value"))?,
            dense: linear(h, h, att.pp("dense"))?,
            attention_norm: layer_norm(h, LAYER_NORM_EPS, att.pp("LayerNorm"))?,
            ffn: linear(h, cfg.intermediate_size, vb.pp("ffn"))?,
            ffn_output: linear(cfg.intermediate_size, h, vb.pp("ffn_output"))?,
            full_layer_norm: layer_norm(h, LAYER_NORM_EPS, vb.pp("full_layer_layer_norm"))?,
            heads: cfg.num_attention_heads,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, h) = xs.dims3()?;
        let head_dim = h / self.heads;
        let split = |l: &Linear| -> Result<Tensor> {
            l.forward(xs)?
                .reshape((b, t, self.heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (split(&self.query)?, split(&self.key)?, split(&self.value)?);
        let scores = (q.matmul(&k.t()?)? / (head_dim as f64).sqrt())?;
        let ctx = ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, t, h))?;
        let attended = self
            .attention_norm
            .forward(&(xs + self.dense.forward(&ctx)?)?)?;
        let ffn = self
            .ffn_output
            .forward(&self.ffn.forward(&attended)?.gelu()?)?;
        self.full_layer_norm.forward(&(ffn + attended)?)
    }
}

/// PL-BERT: an ALBERT whose layers all share one set of weights.
#[derive(Debug, Clone)]
pub struct Albert {
    word: Embedding,
    position: Embedding,
    token_type: Tensor,
    norm: LayerNorm,
    mapping_in: Linear,
    layer: AlbertLayer,
    layers: usize,
}

impl Albert {
    pub fn new(cfg: &PlBertConfig, n_token: usize, vb: VarBuilder) -> Result<Self> {
        let emb = vb.pp("embeddings");
        let enc = vb.pp("encoder");
        let token_type = emb.get((2, EMBEDDING_SIZE), "token_type_embeddings.weight")?;
        Ok(Self {
            word: embedding(n_token, EMBEDDING_SIZE, emb.pp("word_embeddings"))?,
            position: embedding(
                cfg.max_position_embeddings,
                EMBEDDING_SIZE,
                emb.pp("position_embeddings"),
            )?,
            token_type: token_type.narrow(0, 0, 1)?,
            norm: layer_norm(EMBEDDING_SIZE, LAYER_NORM_EPS, emb.pp("LayerNorm"))?,
            mapping_in: linear(
                EMBEDDING_SIZE,
                cfg.hidden_size,
                enc.pp("embedding_hidden_mapping_in"),
            )?,
            layer: AlbertLayer::new(cfg, enc.pp("albert_layer_groups.0.albert_layers.0"))?,
            layers: cfg.num_hidden_layers,
        })
    }

    /// `ids` is (1, T) u32; returns the last hidden state (1, T, hidden).
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let t = ids.dim(D::Minus1)?;
        let positions = Tensor::arange(0u32, t as u32, ids.device())?.unsqueeze(0)?;
        let emb = (self.word.forward(ids)? + self.position.forward(&positions)?)?
            .broadcast_add(&self.token_type.to_dtype(DType::F32)?)?;
        let mut xs = self.mapping_in.forward(&self.norm.forward(&emb)?)?;
        for _ in 0..self.layers {
            xs = self.layer.forward(&xs)?;
        }
        Ok(xs)
    }
}
