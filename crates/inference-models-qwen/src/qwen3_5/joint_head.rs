//! clef's joint decision head (`JointSchemaHead` of the checkpoint's `joint_schema_model.py`, llama.cpp's `clef`
//! architecture): over the normed hidden states of one prompt that lists every question, it scores each option of each
//! question at once. The options read the prompt (routing blocks), each question summarizes its options and reads the
//! other questions and the prompt (joint blocks), and an option's score adds a lexical prior to a gated joint score.

use std::sync::Arc;

use inference_nn::decision::{JointLayout, QuestionKind};
use inference_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::nn::LayerNorm;
use inference_tensor::nn::ops::softmax_last_dim;
use inference_tensor::{DType, Module, Result, Tensor};
use serde::Deserialize;

// F.normalize's floor on a norm
const L2_EPS: f64 = 1e-12;
const N_QUESTION_TYPES: usize = 3;
// the learned scalars, stored as their inference values: exp of the prior and joint logit scales, sigmoid of the gate
const PRIOR_SCALE: usize = 0;
const JOINT_SCALE: usize = 1;
const RESIDUAL_GATE: usize = 2;

/// The joint head's sizes, from the GGUF's `decision.*` keys and tensor shapes.
#[derive(Clone, Debug, Deserialize)]
pub struct JointHeadConfig {
    pub width: usize,
    pub feedforward: usize,
    pub heads: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub norm_eps: f64,
}

struct Attention {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    v: Arc<dyn QuantMethod>,
    o: Arc<dyn QuantMethod>,
}

struct FeedForward {
    norm: LayerNorm,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

/// A routing block (options cross-attend to the normed prompt memory) or a joint block (questions self-attend,
/// then cross-attend to the raw memory), each then a feed-forward.
struct HeadBlock {
    self_attn: Option<(LayerNorm, Attention)>,
    cross_norm: LayerNorm,
    cross_norm_kv: Option<LayerNorm>,
    cross_attn: Attention,
    ffn: FeedForward,
}

pub struct JointHead {
    cfg: JointHeadConfig,
    hidden_norm: LayerNorm,
    proj_memory: Arc<dyn QuantMethod>,
    proj_question: Arc<dyn QuantMethod>,
    proj_option_question: Arc<dyn QuantMethod>,
    proj_global: Arc<dyn QuantMethod>,
    proj_option_context: Arc<dyn QuantMethod>,
    proj_option_lexical: Arc<dyn QuantMethod>,
    type_embd: Tensor,
    blocks: Vec<HeadBlock>,
    option_summary_norm: LayerNorm,
    field_norm: LayerNorm,
    option_norm: LayerNorm,
    scorer: Arc<dyn QuantMethod>,
    scorer_out: Tensor,
    scorer_out_b: Tensor,
    scales: [f32; 3],
}

fn norm(size: usize, eps: f64, vb: &ShardedVarBuilder) -> Result<LayerNorm> {
    let vb = vb.clone().set_dtype(DType::F32);
    Ok(LayerNorm::new(
        vb.get(size, "weight")?,
        vb.get(size, "bias")?,
        eps,
    ))
}

fn linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    ReplicatedLayer::new(in_dim, out_dim, &None, bias, vb)
}

/// The layer applied to an f32 input, in f32: a quantized layer sums in f32 from any input, a dense one takes its
/// own dtype.
fn apply(layer: &Arc<dyn QuantMethod>, xs: &Tensor) -> Result<Tensor> {
    let input = match layer.get_qtensor() {
        Some(_) => xs.clone(),
        None => xs.to_dtype(layer.dtype_and_device().0)?,
    };
    layer.forward(&input)?.to_dtype(DType::F32)
}

fn l2_normalize(xs: &Tensor) -> Result<Tensor> {
    let norm = xs.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(L2_EPS, f64::MAX)?;
    xs.broadcast_div(&norm)
}

/// The mean over each span's rows of `xs`.
fn span_means(xs: &Tensor, spans: impl Iterator<Item = std::ops::Range<usize>>) -> Result<Tensor> {
    let rows = spans
        .map(|span| xs.narrow(0, span.start, span.len())?.mean_keepdim(0))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&rows, 0)
}

/// llama.cpp's type order for the type embedding: noul, choice, score.
fn type_row(kind: QuestionKind) -> u32 {
    match kind {
        QuestionKind::Noul => 0,
        QuestionKind::Choice => 1,
        QuestionKind::Score => 2,
    }
}

impl Attention {
    fn new(width: usize, prefix: &str, vb: &ShardedVarBuilder) -> Result<Self> {
        let proj = |name: &str| linear(width, width, true, vb.pp(format!("{prefix}_{name}")));
        Ok(Self {
            q: proj("q")?,
            k: proj("k")?,
            v: proj("v")?,
            o: proj("o")?,
        })
    }

    /// Multi-head attention of `queries` over `keys_values`, without a mask.
    fn forward(&self, queries: &Tensor, keys_values: &Tensor, heads: usize) -> Result<Tensor> {
        let (n_q, width) = queries.dims2()?;
        let n_kv = keys_values.dim(0)?;
        let head_dim = width / heads;
        let split = |xs: Tensor, n: usize| {
            xs.reshape((n, heads, head_dim))?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = split(apply(&self.q, queries)?, n_q)?;
        let k = split(apply(&self.k, keys_values)?, n_kv)?;
        let v = split(apply(&self.v, keys_values)?, n_kv)?;
        let scores = (q.matmul(&k.t()?.contiguous()?)? / (head_dim as f64).sqrt())?;
        let out = softmax_last_dim(&scores)?
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n_q, width))?;
        apply(&self.o, &out)
    }
}

impl FeedForward {
    fn new(cfg: &JointHeadConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            norm: norm(cfg.width, cfg.norm_eps, &vb.pp("ffn_norm"))?,
            up: linear(cfg.width, cfg.feedforward, true, vb.pp("ffn_up"))?,
            down: linear(cfg.feedforward, cfg.width, true, vb.pp("ffn_down"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        apply(
            &self.down,
            &apply(&self.up, &self.norm.forward(xs)?)?.gelu_erf()?,
        )
    }
}

impl JointHead {
    /// The head under llama.cpp's names: `decision.*`, `dec.blk.N.*` (routing blocks first) and `token_types`.
    pub fn new(cfg: &JointHeadConfig, hidden_size: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let decision = vb.pp("decision");
        let (w, eps) = (cfg.width, cfg.norm_eps);
        let proj = |name: &str| linear(hidden_size, w, false, decision.pp(name));
        let blocks = (0..cfg.routing_layers + cfg.layers)
            .map(|i| {
                let block = vb.pp("dec").pp("blk").pp(i);
                let routing = i < cfg.routing_layers;
                Ok(HeadBlock {
                    self_attn: match routing {
                        true => None,
                        false => Some((
                            norm(w, eps, &block.pp("attn_norm"))?,
                            Attention::new(w, "attn", &block)?,
                        )),
                    },
                    cross_norm: norm(w, eps, &block.pp("cross_attn_norm"))?,
                    cross_norm_kv: match routing {
                        true => Some(norm(w, eps, &block.pp("cross_attn_norm_kv"))?),
                        false => None,
                    },
                    cross_attn: Attention::new(w, "cross_attn", &block)?,
                    ffn: FeedForward::new(cfg, &block)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let dense = decision.clone().set_dtype(DType::F32);
        let scales = dense.get(3, "scales")?.to_vec1::<f32>()?;
        Ok(Self {
            cfg: cfg.clone(),
            hidden_norm: norm(hidden_size, eps, &decision.pp("hidden_norm"))?,
            proj_memory: proj("proj_memory")?,
            proj_question: proj("proj_question")?,
            proj_option_question: proj("proj_option_question")?,
            proj_global: proj("proj_global")?,
            proj_option_context: proj("proj_option_context")?,
            proj_option_lexical: proj("proj_option_lexical")?,
            type_embd: vb
                .pp("token_types")
                .set_dtype(DType::F32)
                .get((N_QUESTION_TYPES, w), "weight")?,
            blocks,
            option_summary_norm: norm(w, eps, &decision.pp("option_summary_norm"))?,
            field_norm: norm(w, eps, &decision.pp("field_norm"))?,
            option_norm: norm(w, eps, &decision.pp("option_norm"))?,
            scorer: linear(4 * w, w, true, decision.pp("scorer"))?,
            scorer_out: dense.pp("scorer_out").get(w, "weight")?.unsqueeze(1)?,
            scorer_out_b: dense.pp("scorer_out").get(1, "bias")?,
            scales: [
                scales[PRIOR_SCALE],
                scales[JOINT_SCALE],
                scales[RESIDUAL_GATE],
            ],
        })
    }

    /// One score per option of `layout`, in its order, from the prompt's normed hidden states `[tokens, hidden]` and
    /// the output embeddings of each option's tokens `[option tokens, hidden]` (the option spans' tokens, in order).
    pub fn forward(
        &self,
        hidden: &Tensor,
        lexical_rows: &Tensor,
        layout: &JointLayout,
    ) -> Result<Vec<f32>> {
        let device = hidden.device();
        let heads = self.cfg.heads;
        let hidden = self.hidden_norm.forward(&hidden.to_dtype(DType::F32)?)?;
        let n_tokens = hidden.dim(0)?;
        let memory = apply(&self.proj_memory, &hidden)?;
        let global = hidden.narrow(0, n_tokens - 1, 1)?;
        let question = span_means(
            &hidden,
            layout.questions.iter().map(|(_, span)| span.clone()),
        )?;
        let option_context =
            span_means(&hidden, layout.options.iter().map(|(_, span)| span.clone()))?;
        let mut offset = 0;
        let lexical_spans = layout.options.iter().map(|(_, span)| {
            let range = offset..offset + span.len();
            offset = range.end;
            range
        });
        let lexical = span_means(&lexical_rows.to_dtype(DType::F32)?, lexical_spans)?;
        let owners = Tensor::new(
            layout
                .options
                .iter()
                .map(|(q, _)| *q as u32)
                .collect::<Vec<_>>(),
            device,
        )?;

        let mut options = (apply(&self.proj_option_context, &option_context)?
            + apply(&self.proj_option_lexical, &lexical)?)?
        .add(&apply(&self.proj_option_question, &question)?.index_select(&owners, 0)?)?;
        let (routing, joint) = self.blocks.split_at(self.cfg.routing_layers);
        for block in routing {
            let kv = block
                .cross_norm_kv
                .as_ref()
                .expect("a routing block normalizes the memory")
                .forward(&memory)?;
            options = (&options
                + block
                    .cross_attn
                    .forward(&block.cross_norm.forward(&options)?, &kv, heads)?)?;
            options = (&options + block.ffn.forward(&options)?)?;
        }

        // each question weights its own options
        let mut fields = apply(&self.proj_question, &question)?;
        let mut mask = vec![f32::NEG_INFINITY; layout.questions.len() * layout.options.len()];
        for (i, (q, _)) in layout.options.iter().enumerate() {
            mask[q * layout.options.len() + i] = 0.0;
        }
        let mask = Tensor::from_vec(mask, (layout.questions.len(), layout.options.len()), device)?;
        let weights = (fields.matmul(&options.t()?)? / (self.cfg.width as f64).sqrt())?;
        let weights = softmax_last_dim(&(weights + mask)?)?;
        let summary = weights.matmul(&options)?;
        let types = Tensor::new(
            layout
                .questions
                .iter()
                .map(|(kind, _)| type_row(*kind))
                .collect::<Vec<_>>(),
            device,
        )?;
        fields = (fields + self.option_summary_norm.forward(&summary)?)?
            .broadcast_add(&apply(&self.proj_global, &global)?)?
            .add(&self.type_embd.index_select(&types, 0)?)?;
        for block in joint {
            let (self_norm, self_attn) = block
                .self_attn
                .as_ref()
                .expect("a joint block self-attends");
            let normed = self_norm.forward(&fields)?;
            fields = (&fields + self_attn.forward(&normed, &normed, heads)?)?;
            let normed = block.cross_norm.forward(&fields)?;
            fields = (&fields + block.cross_attn.forward(&normed, &memory, heads)?)?;
            fields = (&fields + block.ffn.forward(&fields)?)?;
        }
        let fields = self.field_norm.forward(&fields)?;

        // the prior: the option's output embeddings against its question and the end of the prompt
        let anchor = l2_normalize(&question.broadcast_add(&global)?)?.index_select(&owners, 0)?;
        let prior = (l2_normalize(&lexical)? * anchor)?.sum(1)?;
        let options = self.option_norm.forward(&options)?;
        let field = fields.index_select(&owners, 0)?;
        let cosine = (l2_normalize(&field)? * l2_normalize(&options)?)?.sum(1)?;
        let features = Tensor::cat(
            &[
                &field,
                &options,
                &(&field * &options)?,
                &(&field - &options)?.abs()?,
            ],
            1,
        )?;
        let residual = apply(&self.scorer, &features)?
            .gelu_erf()?
            .matmul(&self.scorer_out)?
            .broadcast_add(&self.scorer_out_b)?
            .squeeze(1)?;
        let [prior_scale, joint_scale, gate] = self.scales.map(f64::from);
        let joint = ((cosine * joint_scale)? + residual)?;
        ((prior * prior_scale)? + (joint * gate)?)?.to_vec1::<f32>()
    }
}
