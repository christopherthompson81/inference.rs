//! DSpine (Tencent, <https://arxiv.org/abs/2609.36173>): a DFlash block drafter whose every layer ends in a code
//! write passing each row's predicted predecessor features to the next row, and whose drafts come from a serial
//! refinement of the last write over the target head's top-k candidates. Spec: `sglang_dspine/models/dspine.py`,
//! `sglang_dspine/kernels/dspine.py` and the MLX port's `bridge.py` / `propose.py` in Tencent/DSpine (Apache-2.0).

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::{D, DType, Device, IndexOp, Result, Tensor};

use inference_nn::layers::RmsNorm;

// `torch.finfo(torch.float32).tiny`, the floor of the bridge's read RMS
const F32_TINY: f64 = 1.175_494_350_822_287_5e-38;
const INDEPENDENT_WRITES: &str = "independent";
const SERIAL_DECODE: &str = "serial";
const EXACT_SENDER: &str = "exact_sender";
const SERIAL_TABLE: &str = "serial_table";
const SUPPORTED_VERSION: i64 = 1;
const BRIDGE: &str = "selected_prefix_bridge";
/// The checkpoint prefix of the code bridge, which stays at the model dtype under a drafter weight type.
pub const BRIDGE_PREFIX: &str = "selected_prefix_bridge.";
/// The bridge's token codebook `[vocab, C]`, the one bridge tensor a drafter weight type quantizes.
pub const CODEBOOK: &str = "selected_prefix_bridge.codebook";

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct CodeBridgeConfig {
    pub code_size: usize,
    pub rank: usize,
    pub eps: f64,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct LastWriteRefine {
    pub kind: String,
    pub version: i64,
    pub head_topk: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct SecondPass {
    pub mode: String,
    pub first_exact: usize,
    pub head_topk: usize,
}

/// The `dflash_config` keys a DSpine checkpoint adds.
#[derive(Debug, Clone, Default, inference_json::FromJson)]
pub struct DSpineKeys {
    pub selected_prefix_code_bridge: Option<CodeBridgeConfig>,
    pub selected_prefix_feature_layers: Option<Vec<usize>>,
    pub selected_prefix_terminal: Option<bool>,
    pub selected_prefix_terminal_residual_rank: Option<usize>,
    pub selected_prefix_feature_write_mode: Option<String>,
    pub selected_prefix_feature_writes: Option<bool>,
    pub selected_prefix_feature_next_token_rows: Option<bool>,
    pub selected_prefix_decode_mode: Option<String>,
    pub selected_prefix_last_write_refine: Option<LastWriteRefine>,
    pub draft_block_causal: Option<bool>,
}

/// What the drafter needs of a checkpoint the reference serves: its bridge shape, the layers that write, the
/// refinement's candidate count.
#[derive(Debug, Clone)]
pub struct DSpineSpec {
    pub bridge: CodeBridgeConfig,
    pub active_layers: Vec<bool>,
    pub head_topk: usize,
}

impl DSpineSpec {
    /// The reference's checks (`DSpineCodeBridge.__init__`, `resolve_dspine_head_topk`): terminal-disabled,
    /// independent writes, causal blocks, anchor-row semantics, serial exact-sender refinement of the last write.
    pub fn from_config(
        keys: &DSpineKeys,
        second_pass: Option<&SecondPass>,
        num_layers: usize,
    ) -> Result<Self> {
        let require = |ok: bool, what: &str| {
            if ok {
                Ok(())
            } else {
                Err(inference_tensor::Error::msg(format!(
                    "DSpine drafter: {what}"
                )))
            }
        };
        require(
            keys.selected_prefix_terminal == Some(false),
            "only terminal-disabled code bridges are supported",
        )?;
        require(
            keys.selected_prefix_terminal_residual_rank.unwrap_or(0) == 0,
            "the terminal token residual is unsupported",
        )?;
        require(
            keys.selected_prefix_feature_write_mode.as_deref() == Some(INDEPENDENT_WRITES),
            "the code bridge must write independently",
        )?;
        require(
            keys.selected_prefix_feature_writes == Some(true),
            "the code bridge must write features",
        )?;
        require(
            keys.draft_block_causal == Some(true),
            "draft blocks must be causal",
        )?;
        require(
            !keys
                .selected_prefix_feature_next_token_rows
                .unwrap_or(false),
            "next-token rows are a DSpark layout",
        )?;
        require(
            keys.selected_prefix_decode_mode.as_deref() == Some(SERIAL_DECODE),
            "only serial decoding is supported",
        )?;
        let bridge = keys
            .selected_prefix_code_bridge
            .clone()
            .ok_or_else(|| inference_tensor::Error::msg("DSpine config has no code bridge"))?;
        let mut active_layers = vec![false; num_layers];
        for &layer in keys
            .selected_prefix_feature_layers
            .as_deref()
            .unwrap_or(&[])
        {
            // 1-based in the config
            let index = layer
                .checked_sub(1)
                .filter(|index| *index < num_layers)
                .ok_or_else(|| {
                    inference_tensor::Error::msg(format!(
                        "DSpine feature layer {layer} is outside 1..={num_layers}"
                    ))
                })?;
            active_layers[index] = true;
        }
        require(
            active_layers.last() == Some(&true),
            "the last layer must write, its write is the one refined",
        )?;
        let mut widths = Vec::new();
        if let Some(refine) = &keys.selected_prefix_last_write_refine {
            require(
                refine.kind == EXACT_SENDER && refine.version == SUPPORTED_VERSION,
                "the last-write refinement must be exact_sender version 1",
            )?;
            widths.push(refine.head_topk);
        }
        if let Some(second) = second_pass {
            require(
                second.mode == SERIAL_TABLE && second.first_exact + 1 == num_layers,
                "the second pass must be a serial table over the last layer",
            )?;
            widths.push(second.head_topk);
        }
        let head_topk = *widths.first().ok_or_else(|| {
            inference_tensor::Error::msg("DSpine config names no refinement width")
        })?;
        require(
            widths.iter().all(|width| *width == head_topk) && head_topk >= 2,
            "the refinement widths disagree",
        )?;
        Ok(Self {
            bridge,
            active_layers,
            head_topk,
        })
    }
}

/// The last write's pre-write state for the draft rows: `raw` `[b, n, H]`, `rms` `[b, n]` f32 and the receiver
/// query `[b, n, R]` the refinement re-gates.
pub struct BridgeStash {
    pub raw: Tensor,
    pub rms: Tensor,
    pub query: Tensor,
}

impl BridgeStash {
    pub fn narrow_rows(&self, start: usize, len: usize) -> Result<Self> {
        Ok(Self {
            raw: self.raw.narrow(1, start, len)?,
            rms: self.rms.narrow(1, start, len)?,
            query: self.query.narrow(1, start, len)?,
        })
    }
}

/// What a refinement tail reads: the drafting rows' normed hidden `[b, n, H]`, the last write's stash, the blocks'
/// anchors `[b]`, the drafter's final norm and the target's head.
#[derive(Clone, Copy)]
pub struct TailInputs<'a> {
    pub rows: &'a Tensor,
    pub stash: &'a BridgeStash,
    pub anchors: &'a Tensor,
    pub norm: &'a RmsNorm,
    pub lm_head: &'a dyn QuantMethod,
}

/// The code bridge's weights (`selected_prefix_bridge.*`) and what the refinement reads of the drafter.
pub struct DSpineHead {
    spec: DSpineSpec,
    // [vocab, C], gathered by row
    codebook: Arc<dyn QuantMethod>,
    // per layer [C, H]: the early reads, then the last layer's
    reads: Vec<Tensor>,
    // per layer [H, R]
    writes: Vec<Tensor>,
    // [3R, C]: gate, predecessor gate, value (`prev`)
    packed: Tensor,
    gate_bias: Tensor,
    // [K, K] upper-triangular ones: the sampled chain's CDF, built once so a captured graph copies nothing
    triu: Tensor,
}

fn version(vb: &ShardedVarBuilder, name: &str) -> Result<()> {
    let value = vb
        .get_unchecked_dtype(name, DType::I64)?
        .to_dtype(DType::F64)?
        .to_scalar::<f64>()?;
    #[allow(clippy::cast_possible_truncation)]
    if value as i64 != SUPPORTED_VERSION {
        inference_tensor::bail!(
            "DSpine checkpoint `{name}` is {value}, only version 1 is supported"
        );
    }
    Ok(())
}

impl DSpineHead {
    /// `vb` is the drafter's root and `hidden` its width; the codebook loads with the drafter's projections.
    pub fn load(
        vb: &ShardedVarBuilder,
        spec: DSpineSpec,
        codebook: Arc<dyn QuantMethod>,
        hidden: usize,
        num_layers: usize,
        device: &Device,
    ) -> Result<Self> {
        for name in ["draft_block_causal_version", "last_write_refine_version"] {
            version(vb, name)?;
        }
        let bridge = vb.pp(BRIDGE);
        for name in ["independent_write_version", "terminal_disabled_version"] {
            version(&bridge, name)?;
        }
        let (code, rank) = (spec.bridge.code_size, spec.bridge.rank);
        let out = bridge.pp("out").get((code, rank), "weight")?;
        if out
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?
            != 0.0
        {
            inference_tensor::bail!(
                "a terminal-disabled DSpine checkpoint carries a nonzero terminal projection"
            );
        }
        let early = bridge.get((num_layers - 1, code, hidden), "early_read_weights")?;
        let mut reads = (0..num_layers - 1)
            .map(|layer| early.i(layer))
            .collect::<Result<Vec<_>>>()?;
        reads.push(bridge.get((code, hidden), "read_weight")?);
        let writes = bridge.get((num_layers, hidden, rank), "feature_write_weights")?;
        let writes = (0..num_layers)
            .map(|layer| writes.i(layer))
            .collect::<Result<Vec<_>>>()?;
        let packed = Tensor::cat(
            &[
                bridge.pp("gate").get((rank, code), "weight")?,
                bridge.pp("predecessor_gate").get((rank, code), "weight")?,
                bridge.pp("prev").get((rank, code), "weight")?,
            ],
            0,
        )?;
        let k = spec.head_topk;
        let triu = Tensor::triu2(k, DType::F32, device)?;
        Ok(Self {
            codebook,
            reads,
            writes,
            packed,
            gate_bias: bridge.pp("gate").get(rank, "bias")?.to_dtype(DType::F32)?,
            triu,
            spec,
        })
    }

    pub fn head_topk(&self) -> usize {
        self.spec.head_topk
    }

    #[allow(clippy::cast_precision_loss)]
    fn sqrt_code(&self) -> f64 {
        (self.spec.bridge.code_size as f64).sqrt()
    }

    // `sqrt(C) * codebook[ids]`, rounded to the activation type, `[.., C]`
    fn exact_codes(&self, ids: &Tensor, dtype: DType) -> Result<Tensor> {
        let dims = ids.dims().to_vec();
        let rows = self
            .codebook
            .embedding_forward(&ids.flatten_all()?, DType::F32)?;
        let mut shape = dims;
        shape.push(self.spec.bridge.code_size);
        (rows.to_dtype(DType::F32)? * self.sqrt_code())?
            .to_dtype(dtype)?
            .reshape(shape)
    }

    /// One write after `layer` over `hs` `[b, T, H]` (the layer's output, residual included): each row past the
    /// anchor gains its predecessor's gated message. Returns the updated stream, and for the last layer the stash.
    pub fn write(
        &self,
        hs: &Tensor,
        anchors: &Tensor,
        layer: usize,
    ) -> Result<(Tensor, Option<BridgeStash>)> {
        if !self.spec.active_layers[layer] {
            return Ok((hs.clone(), None));
        }
        let (b, t, _) = hs.dims3()?;
        let dtype = hs.dtype();
        let rank = self.spec.bridge.rank;
        let eps = self.spec.bridge.eps;
        let f = hs.to_dtype(DType::F32)?;
        let rms = f
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .maximum(F32_TINY)?
            .sqrt()?;
        let normalized = f.broadcast_div(&(&rms + eps)?)?.to_dtype(dtype)?;
        let projected = normalized
            .broadcast_matmul(&self.reads[layer].t()?)?
            .to_dtype(DType::F32)?;
        let norm = projected.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let codes =
            (projected.broadcast_div(&(norm + eps)?)? * self.sqrt_code())?.to_dtype(dtype)?;
        let anchor_codes = self.exact_codes(&anchors.reshape((b, 1))?, dtype)?;
        let codes = Tensor::cat(&[&anchor_codes, &codes.narrow(1, 1, t - 1)?], 1)?;
        let packed = codes.broadcast_matmul(&self.packed.t()?)?;
        let parts = packed.to_dtype(DType::F32)?;
        let (q, s, v) = (
            parts.narrow(D::Minus1, 0, rank)?,
            parts.narrow(D::Minus1, rank, rank)?,
            parts.narrow(D::Minus1, 2 * rank, rank)?,
        );
        let gate = inference_tensor::nn::ops::sigmoid(
            &(q.narrow(1, 1, t - 1)? + s.narrow(1, 0, t - 1)?)?.broadcast_add(&self.gate_bias)?,
        )?;
        let message = Tensor::cat(
            &[
                &Tensor::zeros((b, 1, rank), DType::F32, hs.device())?,
                &(gate * v.narrow(1, 0, t - 1)?)?,
            ],
            1,
        )?
        .to_dtype(dtype)?;
        let delta = message
            .broadcast_matmul(&self.writes[layer].t()?)?
            .to_dtype(DType::F32)?;
        // the anchor row's message is zero, so its delta is too
        let out = (f + delta.broadcast_mul(&rms)?)?.to_dtype(dtype)?;
        let last = layer + 1 == self.spec.active_layers.len();
        let stash = last
            .then(|| -> Result<BridgeStash> {
                Ok(BridgeStash {
                    raw: hs.clone(),
                    rms: rms.squeeze(D::Minus1)?,
                    query: packed.narrow(D::Minus1, 0, rank)?,
                })
            })
            .transpose()?;
        Ok((out, stash))
    }

    /// Each draft row's top-k candidate ids `[b, n, K]` U32, by score (the reference orders them by id, which only
    /// changes which of two exactly tied scores wins).
    fn candidates(&self, logits: &Tensor) -> Result<Tensor> {
        let (b, n, vocab) = logits.dims3()?;
        let k = self.spec.head_topk;
        let rows = logits.reshape((b * n, vocab))?.contiguous()?;
        #[cfg(feature = "cuda")]
        let ids = if rows.device().is_cuda() {
            inference_nn::ops::cuda_topk_ranked_packed_batched(&rows, k)?
                .packed
                .narrow(1, k, k)?
                .contiguous()?
        } else {
            top_ids_host(&rows, k)?
        };
        #[cfg(not(feature = "cuda"))]
        let ids = top_ids_host(&rows, k)?;
        ids.to_dtype(DType::U32)?.reshape((b, n, k))
    }

    // the candidates' head rows `[b, n, K, H]` f32, and each row's sender ids `[b, n, K]` (the anchor, then the
    // previous row's candidates)
    fn candidate_rows(
        &self,
        cand: &Tensor,
        anchors: &Tensor,
        lm_head: &dyn QuantMethod,
    ) -> Result<(Tensor, Tensor)> {
        let (b, n, k) = cand.dims3()?;
        let heads = lm_head
            .embedding_forward(&cand.flatten_all()?, DType::F32)?
            .reshape((b, n, k, ()))?;
        let first = anchors.reshape((b, 1, 1))?.broadcast_as((b, 1, k))?;
        let senders = Tensor::cat(&[&first.contiguous()?, &cand.narrow(1, 0, n - 1)?], 1)?;
        Ok((heads, senders))
    }

    /// Greedy drafts `[b, n]` U32: the reference's serial table (`_serial_table`, `_serial_chain`), the last write
    /// re-scored for every candidate given each candidate the previous row could have taken.
    pub fn greedy(&self, tail: &TailInputs<'_>) -> Result<Tensor> {
        let (cand, idx) = self.greedy_table(tail)?;
        chain(&idx, &cand)
    }

    // the candidates and, per row and sender, the best candidate's index `[b, n, K]`
    fn greedy_table(&self, tail: &TailInputs<'_>) -> Result<(Tensor, Tensor)> {
        let TailInputs {
            rows,
            stash,
            anchors,
            norm,
            lm_head,
        } = *tail;
        let dtype = rows.dtype();
        let rank = self.spec.bridge.rank;
        let cand = self.candidates(&lm_head.forward(rows)?)?;
        let (heads, senders) = self.candidate_rows(&cand, anchors, lm_head)?;
        let norm_w = norm.weight().to_dtype(DType::F32)?;
        let raw = stash.raw.to_dtype(DType::F32)?.broadcast_mul(&norm_w)?;
        // [b, n, K]
        let base = heads
            .matmul(&raw.unsqueeze(D::Minus1)?)?
            .squeeze(D::Minus1)?;
        let tsv = self
            .exact_codes(&senders, dtype)?
            .broadcast_matmul(&self.packed.narrow(0, rank, 2 * rank)?.t()?)?
            .to_dtype(DType::F32)?;
        let message = inference_tensor::nn::ops::sigmoid(
            &tsv.narrow(D::Minus1, 0, rank)?
                .broadcast_add(&stash.query.to_dtype(DType::F32)?.unsqueeze(2)?)?
                .broadcast_add(&self.gate_bias)?,
        )?
        .mul(&tsv.narrow(D::Minus1, rank, rank)?)?;
        let last = self.writes.last().expect("a DSpine drafter has layers");
        let tg = heads
            .broadcast_mul(&norm_w)?
            .to_dtype(dtype)?
            .broadcast_matmul(last)?
            .to_dtype(DType::F32)?;
        // [b, n, sender, candidate]
        let coupling = message.matmul(&tg.transpose(2, 3)?)?;
        let score = coupling
            .broadcast_mul(&stash.rms.unsqueeze(2)?.unsqueeze(3)?)?
            .broadcast_add(&base.unsqueeze(2)?)?;
        Ok((cand, score.argmax(D::Minus1)?))
    }

    /// Sampled drafts with their candidate distributions (`serial_rs_table`, `_serial_sample_chain` without the
    /// request filters): the last write recomputed exactly for each sender, each row's pick drawn from the
    /// temperature softmax over its candidates given the previous pick. Rows with inverse temperature 0 take the
    /// greedy chain, their distribution one-hot. Returns tokens `[b, n]`, candidate ids and probs `[b, n, K]`.
    pub fn sample(
        &self,
        tail: &TailInputs<'_>,
        inverse_temperatures: &Tensor,
        uniforms: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (cand, greedy_idx) = self.greedy_table(tail)?;
        let TailInputs {
            rows,
            stash,
            anchors,
            norm,
            lm_head,
        } = *tail;
        let greedy = chain(&greedy_idx, &cand)?;
        let (b, n, k) = cand.dims3()?;
        let rank = self.spec.bridge.rank;
        let (heads, senders) = self.candidate_rows(&cand, anchors, lm_head)?;
        let codes = self.exact_codes(&senders, DType::F32)?;
        let weights = self.packed.to_dtype(DType::F32)?;
        let gate = inference_tensor::nn::ops::sigmoid(
            &codes
                .broadcast_matmul(&weights.narrow(0, rank, rank)?.t()?)?
                .broadcast_add(&stash.query.to_dtype(DType::F32)?.unsqueeze(2)?)?
                .broadcast_add(&self.gate_bias)?,
        )?;
        let value = codes.broadcast_matmul(&weights.narrow(0, 2 * rank, rank)?.t()?)?;
        let last = self
            .writes
            .last()
            .expect("a DSpine drafter has layers")
            .to_dtype(DType::F32)?;
        let refined = (gate * value)?
            .broadcast_matmul(&last.t()?)?
            .broadcast_mul(&stash.rms.unsqueeze(2)?.unsqueeze(3)?)?
            .broadcast_add(&stash.raw.to_dtype(DType::F32)?.unsqueeze(2)?)?;
        let scale = (refined.sqr()?.mean_keepdim(D::Minus1)? + norm.eps())?.sqrt()?;
        let normalized = refined
            .broadcast_div(&scale)?
            .broadcast_mul(&norm.weight().to_dtype(DType::F32)?)?;
        // [b, n, sender, candidate]
        let table = normalized.matmul(&heads.transpose(2, 3)?)?;

        let inv_t = inverse_temperatures.to_dtype(DType::F32)?.reshape((b, 1))?;
        let uniforms = uniforms.to_dtype(DType::F32)?.reshape((b, n))?;
        let mut sender = Tensor::zeros((b, 1), DType::U32, rows.device())?;
        let (mut tokens, mut probs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for row in 0..n {
            let scores = table
                .i((.., row))?
                .contiguous()?
                .gather(
                    &sender.unsqueeze(2)?.broadcast_as((b, 1, k))?.contiguous()?,
                    1,
                )?
                .squeeze(1)?
                .broadcast_mul(&inv_t)?;
            let p = inference_tensor::nn::ops::softmax_last_dim(&scores)?;
            let cdf = p.matmul(&self.triu)?;
            let threshold =
                uniforms
                    .narrow(1, row, 1)?
                    .broadcast_mul(&cdf.narrow(1, k - 1, 1)?)?;
            #[allow(clippy::cast_precision_loss)]
            let choice = threshold
                .broadcast_ge(&cdf)?
                .to_dtype(DType::F32)?
                .sum_keepdim(1)?
                .minimum((k - 1) as f64)?
                .to_dtype(DType::U32)?;
            tokens.push(cand.i((.., row))?.contiguous()?.gather(&choice, 1)?);
            probs.push(p);
            sender = choice;
        }
        let sampled = Tensor::cat(&tokens, 1)?;
        let probs = Tensor::stack(&probs, 1)?;
        let greedy_rows = inverse_temperatures
            .to_dtype(DType::F32)?
            .reshape((b, 1))?
            .le(0.0)?;
        let tokens = greedy_rows
            .broadcast_as((b, n))?
            .where_cond(&greedy, &sampled)?;
        let one_hot = cand
            .broadcast_eq(&greedy.unsqueeze(2)?)?
            .to_dtype(DType::F32)?;
        let probs = greedy_rows
            .unsqueeze(2)?
            .broadcast_as((b, n, k))?
            .where_cond(&one_hot, &probs)?;
        Ok((tokens, cand, probs))
    }
}

// rows' top-k ids `[rows, k]` f32 off the GPU, as the CUDA top-k packs them
fn top_ids_host(rows: &Tensor, k: usize) -> Result<Tensor> {
    rows.to_dtype(DType::F32)?
        .arg_sort_last_dim(false)?
        .narrow(1, 0, k)?
        .to_dtype(DType::F32)
}

// walk the serial table: row 0's sender is the anchor (every column alike), each later row's the previous pick
fn chain(idx: &Tensor, cand: &Tensor) -> Result<Tensor> {
    let (b, n, _) = cand.dims3()?;
    let mut j = Tensor::zeros((b, 1), DType::U32, cand.device())?;
    let mut picks = Vec::with_capacity(n);
    for row in 0..n {
        let next = idx
            .i((.., row))?
            .to_dtype(DType::U32)?
            .contiguous()?
            .gather(&j, 1)?;
        picks.push(cand.i((.., row))?.contiguous()?.gather(&next, 1)?);
        j = next;
    }
    Tensor::cat(&picks, 1)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use inference_quant::{QuantMethodConfig, UnquantLinear};
    use inference_tensor::nn::Linear;

    const H: usize = 16;
    const C: usize = 8;
    const R: usize = 4;
    const V: usize = 40;
    const K: usize = 4;
    const T: usize = 5;
    const LAYERS: usize = 2;
    const EPS: f64 = 1e-6;

    fn ramp(len: usize, salt: f32) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        (0..len).map(|i| ((i as f32) * 0.71 + salt).sin()).collect()
    }

    fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn linear(weight: Tensor) -> Arc<dyn QuantMethod> {
        Arc::new(
            UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weight, None))).unwrap(),
        )
    }

    struct Weights {
        codebook: Vec<f32>,
        reads: Vec<Vec<f32>>,
        writes: Vec<Vec<f32>>,
        gate: Vec<f32>,
        pred: Vec<f32>,
        prev: Vec<f32>,
        bias: Vec<f32>,
        head: Vec<f32>,
        norm: Vec<f32>,
    }

    fn weights() -> Weights {
        Weights {
            codebook: ramp(V * C, 1.0),
            reads: (0..LAYERS).map(|l| ramp(C * H, 2.0 + l as f32)).collect(),
            writes: (0..LAYERS)
                .map(|l| {
                    ramp(H * R, 4.0 + l as f32)
                        .iter()
                        .map(|w| w * 0.1)
                        .collect()
                })
                .collect(),
            gate: ramp(R * C, 6.0),
            pred: ramp(R * C, 7.0),
            prev: ramp(R * C, 8.0),
            bias: ramp(R, 9.0),
            head: ramp(V * H, 10.0),
            norm: ramp(H, 11.0).iter().map(|w| 1.0 + 0.2 * w).collect(),
        }
    }

    fn head(w: &Weights) -> DSpineHead {
        DSpineHead {
            spec: DSpineSpec {
                bridge: CodeBridgeConfig {
                    code_size: C,
                    rank: R,
                    eps: EPS,
                },
                active_layers: vec![true; LAYERS],
                head_topk: K,
            },
            codebook: linear(tensor(w.codebook.clone(), &[V, C])),
            reads: w.reads.iter().map(|r| tensor(r.clone(), &[C, H])).collect(),
            writes: w
                .writes
                .iter()
                .map(|r| tensor(r.clone(), &[H, R]))
                .collect(),
            packed: tensor(
                [w.gate.clone(), w.pred.clone(), w.prev.clone()].concat(),
                &[3 * R, C],
            ),
            gate_bias: tensor(w.bias.clone(), &[R]),
            triu: Tensor::triu2(K, DType::F32, &Device::Cpu).unwrap(),
        }
    }

    // `rows x cols` row-major times a vector
    fn matvec(m: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|r| (0..cols).map(|c| m[r * cols + c] * x[c]).sum())
            .collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // the reference's write (`prepare_read`, `normalize_codes`, `feature_message`, `feature_finish`) for one block
    fn reference_write(
        w: &Weights,
        hs: &[Vec<f32>],
        anchor: usize,
        layer: usize,
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>) {
        #[allow(clippy::cast_precision_loss)]
        let sqrt_c = (C as f32).sqrt();
        let eps = EPS as f32;
        let mut rms = Vec::new();
        let mut packed = Vec::new();
        for (t, x) in hs.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let r = (x.iter().map(|v| v * v).sum::<f32>() / H as f32).sqrt();
            rms.push(r);
            let code: Vec<f32> = if t == 0 {
                w.codebook[anchor * C..(anchor + 1) * C]
                    .iter()
                    .map(|v| v * sqrt_c)
                    .collect()
            } else {
                let normalized: Vec<f32> = x.iter().map(|v| v / (r + eps)).collect();
                let projected = matvec(&w.reads[layer], &normalized, C, H);
                let norm = projected.iter().map(|v| v * v).sum::<f32>().sqrt();
                projected
                    .iter()
                    .map(|v| sqrt_c * v / (norm + eps))
                    .collect()
            };
            packed.push(
                [
                    matvec(&w.gate, &code, R, C),
                    matvec(&w.pred, &code, R, C),
                    matvec(&w.prev, &code, R, C),
                ]
                .concat(),
            );
        }
        let out = hs
            .iter()
            .enumerate()
            .map(|(t, x)| {
                if t == 0 {
                    return x.clone();
                }
                let message: Vec<f32> = (0..R)
                    .map(|k| {
                        sigmoid(packed[t][k] + packed[t - 1][R + k] + w.bias[k])
                            * packed[t - 1][2 * R + k]
                    })
                    .collect();
                let delta = matvec(&w.writes[layer], &message, H, R);
                x.iter().zip(&delta).map(|(v, d)| v + rms[t] * d).collect()
            })
            .collect();
        (out, rms, packed)
    }

    #[test]
    fn writes_and_the_greedy_tail_match_the_reference_transcription() {
        let w = weights();
        let dspine = head(&w);
        let anchor = 7usize;
        let anchors = Tensor::new(&[u32::try_from(anchor).unwrap()], &Device::Cpu).unwrap();
        let hs: Vec<Vec<f32>> = (0..T).map(|t| ramp(H, 20.0 + t as f32)).collect();
        let input = tensor(hs.concat(), &[1, T, H]);

        let (out, stash) = dspine.write(&input, &anchors, LAYERS - 1).unwrap();
        let (expected, rms, packed) = reference_write(&w, &hs, anchor, LAYERS - 1);
        let ours = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in ours.iter().zip(expected.concat()) {
            assert!((a - b).abs() < 1e-4, "write {a} vs {b}");
        }
        let stash = stash.expect("the last layer stashes");

        // greedy tail over rows 1.. with the stash, against the reference's serial table and chain
        let n = T - 1;
        let norm = RmsNorm::from_w(tensor(w.norm.clone(), &[H]), EPS).unwrap();
        let rows = tensor(ramp(n * H, 30.0), &[1, n, H]);
        let lm_head = linear(tensor(w.head.clone(), &[V, H]));
        let stash = stash.narrow_rows(1, n).unwrap();
        let tokens = dspine
            .greedy(&TailInputs {
                rows: &rows,
                stash: &stash,
                anchors: &anchors,
                norm: &norm,
                lm_head: &*lm_head,
            })
            .unwrap()
            .to_vec2::<u32>()
            .unwrap();

        #[allow(clippy::cast_precision_loss)]
        let sqrt_c = (C as f32).sqrt();
        let rows_host = rows.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cand: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let logits = matvec(&w.head, &rows_host[i * H..(i + 1) * H], V, H);
                let mut order: Vec<usize> = (0..V).collect();
                order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
                order[..K].to_vec()
            })
            .collect();
        let head_row = |v: usize| &w.head[v * H..(v + 1) * H];
        let last = &w.writes[LAYERS - 1];
        let mut j = 0usize;
        let mut expected = Vec::new();
        let mut idx = vec![vec![0usize; K]; n];
        for i in 0..n {
            let raw = &hs[i + 1];
            for s in 0..K {
                let sender = if i == 0 { anchor } else { cand[i - 1][s] };
                let e: Vec<f32> = w.codebook[sender * C..(sender + 1) * C]
                    .iter()
                    .map(|v| v * sqrt_c)
                    .collect();
                let (tvg, tvv) = (matvec(&w.pred, &e, R, C), matvec(&w.prev, &e, R, C));
                let m: Vec<f32> = (0..R)
                    .map(|k| sigmoid(packed[i + 1][k] + tvg[k] + w.bias[k]) * tvv[k])
                    .collect();
                let score = |c: usize| {
                    let h = head_row(cand[i][c]);
                    let base: f32 = (0..H).map(|x| raw[x] * w.norm[x] * h[x]).sum();
                    let scaled: Vec<f32> = (0..H).map(|x| h[x] * w.norm[x]).collect();
                    let tg: Vec<f32> = (0..R)
                        .map(|r| (0..H).map(|x| scaled[x] * last[x * R + r]).sum())
                        .collect();
                    base + rms[i + 1] * (0..R).map(|r| m[r] * tg[r]).sum::<f32>()
                };
                idx[i][s] = (0..K)
                    .max_by(|a, b| score(*a).total_cmp(&score(*b)))
                    .unwrap();
            }
        }
        for i in 0..n {
            j = idx[i][j];
            expected.push(u32::try_from(cand[i][j]).unwrap());
        }
        assert_eq!(tokens[0], expected);
    }
}
