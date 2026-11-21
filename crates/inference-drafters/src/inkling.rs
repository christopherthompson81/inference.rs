//! Inkling 2's own drafter (`mtp.safetensors`, vLLM's `Inkling2DSparkModel`): a DSpark block drafter whose context
//! keys and values come from DFlare features (each layer's softmax mix of the target's taps, normalized, through its
//! own `k_proj_ctx` / `v_proj_ctx`), and whose Markov head is recurrent (`rnn_hidden`): a rank-`r` state gated by the
//! previous pick, its latent read through `hidden_out` folded into the target's head.

use std::sync::{Arc, Mutex};

use inference_nn::layers::RmsNorm;
use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::{D, DType, Device, IndexOp, Module, Result, Tensor};

pub(crate) const ARCHITECTURE: &str = "Inkling2DSpark";
pub(crate) const WEIGHTS: &str = "mtp.safetensors";
const RNN_HEAD: &str = "rnn_hidden";
const LOCAL: &str = "sliding_attention";
const GLOBAL: &str = "full_attention";
// `hidden_out`'s columns folded through the head at a time
const FOLD_COLUMNS: usize = 32;
// a projection built from its weight (in the drafter's weight type)
pub(crate) type MakeLinear<'a> = dyn FnMut(&str, Tensor) -> Result<Arc<dyn QuantMethod>> + 'a;

/// The drafter config of an Inkling 2 checkpoint's `config.json` (its `mtp_config` beside the text model's), or
/// `None` for another checkpoint.
pub(crate) fn drafter_config(value: &serde_json::Value) -> Option<serde_json::Value> {
    let mtp = value.get("mtp_config")?;
    let text = value.get("text_config")?;
    if !text.get("model_type")?.as_str()?.starts_with("inkling2") {
        return None;
    }
    let layers = mtp.get("num_hidden_layers")?.as_u64()?;
    let local = mtp.get("local_layer_ids")?.as_array()?;
    let layer_types = (0..layers)
        .map(|i| {
            if local.iter().any(|l| l.as_u64() == Some(i)) {
                LOCAL
            } else {
                GLOBAL
            }
        })
        .collect::<Vec<_>>();
    let mut config = serde_json::json!({
        "architectures": [ARCHITECTURE],
        "hidden_size": mtp["hidden_size"],
        "intermediate_size": mtp["intermediate_size"],
        "num_hidden_layers": layers,
        "num_attention_heads": mtp["num_attention_heads"],
        "num_key_value_heads": mtp["num_key_value_heads"],
        "head_dim": mtp["head_dim"],
        "max_position_embeddings": mtp["max_position_embeddings"],
        "rms_norm_eps": mtp["rms_norm_eps"],
        "vocab_size": text["vocab_size"],
        "rope_theta": mtp["rope_theta"],
        "sliding_window": mtp["sliding_window_size"],
        "layer_types": layer_types,
        // the block attends to itself both ways, its local layers to the window before it
        "is_causal": false,
        "block_size": mtp["block_size"],
        "target_layer_ids": mtp["target_layer_ids"],
        "markov_rank": mtp["markov_rank"],
        "markov_head_type": mtp["markov_head_type"],
        "sample_from_anchor": false,
        // the masks embed as `mask_embed`; the id is never looked up
        "mask_token_id": 0,
    });
    config["final_logit_softcapping"] = text["final_logit_softcapping"].clone();
    config["logit_vocab_size"] = text["unpadded_vocab_size"].clone();
    Some(config)
}

/// The recurrent head's fold of `hidden_out` into the head, `[padded vocab, rank]` at `dtype` and `[padded vocab]`
/// F32, in bytes: made on the first draft, after the paged cache is sized.
pub(crate) fn fold_bytes(padded_vocab: usize, rank: usize, dtype: DType) -> usize {
    padded_vocab * (rank * dtype.size_in_bytes() + DType::F32.size_in_bytes())
}

/// Whether a `markov_head_type` is the recurrent head this module runs.
pub(crate) fn is_rnn_head(kind: Option<&str>) -> bool {
    kind == Some(RNN_HEAD)
}

/// Each draft layer's context from the target's taps: `norm_l(sum_j alpha[l, j] tap_j)`, then its context K and V.
pub(crate) struct DFlareContext {
    // `[layers + 1, taps]` F32, the softmax of `fuse_weight`; the last row mixes the Markov head's context
    alpha: Tensor,
    norms: Vec<RmsNorm>,
    // per layer `[k | v]` rows over the hidden size
    kv_proj: Vec<Arc<dyn QuantMethod>>,
    pub(crate) k_norms: Vec<RmsNorm>,
    head_norm: RmsNorm,
}

pub(crate) struct DFlareShape {
    pub(crate) layers: usize,
    pub(crate) taps: usize,
    pub(crate) hidden: usize,
    pub(crate) kv_size: usize,
    pub(crate) head_dim: usize,
    pub(crate) eps: f64,
}

impl DFlareContext {
    pub(crate) fn load(
        vb: &ShardedVarBuilder,
        shape: DFlareShape,
        linear: &mut MakeLinear<'_>,
    ) -> Result<Self> {
        let DFlareShape {
            layers,
            taps,
            hidden,
            kv_size,
            head_dim,
            eps,
        } = shape;
        let alpha = inference_tensor::nn::ops::softmax_last_dim(
            &vb.get((layers + 1, taps), "fuse_weight")?
                .to_dtype(DType::F32)?,
        )?;
        let mut norms = Vec::with_capacity(layers);
        let mut kv_proj = Vec::with_capacity(layers);
        let mut k_norms = Vec::with_capacity(layers);
        for i in 0..layers {
            let layer = vb.pp("layers").pp(i);
            let attn = layer.pp("self_attn");
            // the mixes normalize in f32, as the reference's
            norms.push(RmsNorm::new(
                hidden,
                eps,
                layer.pp("dflare_norm").set_dtype(DType::F32),
            )?);
            k_norms.push(RmsNorm::new(head_dim, eps, attn.pp("k_norm_ctx"))?);
            let k = attn.pp("k_proj_ctx").get((kv_size, hidden), "weight")?;
            let v = attn.pp("v_proj_ctx").get((kv_size, hidden), "weight")?;
            kv_proj.push(linear(
                &format!("layers.{i}.kv_proj_ctx"),
                Tensor::cat(&[&k, &v], 0)?,
            )?);
        }
        Ok(Self {
            alpha,
            norms,
            kv_proj,
            k_norms,
            head_norm: RmsNorm::new(
                hidden,
                eps,
                vb.pp("markov_head").pp("dflare_norm").set_dtype(DType::F32),
            )?,
        })
    }

    // `packed` `[rows, taps * hidden]` (the taps side by side): row `row` of `alpha`'s mix, `[rows, hidden]` F32
    fn mix(&self, packed: &Tensor, row: usize) -> Result<Tensor> {
        let (rows, width) = packed.dims2()?;
        let taps = self.alpha.dim(1)?;
        let taps_rows = packed
            .to_dtype(DType::F32)?
            .reshape((rows, taps, width / taps))?;
        let alpha = self.alpha.i(row)?.reshape((1, taps, 1))?;
        taps_rows.broadcast_mul(&alpha)?.sum(1)
    }

    /// Raw context keys and values `[layers, kv_heads, rows, head_dim]` each (keys before their norm and RoPE).
    pub(crate) fn project(
        &self,
        packed: &Tensor,
        kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let rows = packed.dim(0)?;
        let (mut ks, mut vs) = (Vec::new(), Vec::new());
        for (layer, (norm, proj)) in self.norms.iter().zip(&self.kv_proj).enumerate() {
            let context = norm.forward(&self.mix(packed, layer)?)?.to_dtype(dtype)?;
            let kv = proj
                .forward(&context)?
                .reshape((rows, 2, kv_heads, head_dim))?
                .permute((1, 2, 0, 3))?;
            ks.push(kv.i(0)?);
            vs.push(kv.i(1)?);
        }
        Ok((
            Tensor::stack(&ks, 0)?.contiguous()?,
            Tensor::stack(&vs, 0)?.contiguous()?,
        ))
    }

    /// The Markov head's context of each row, `[rows, hidden]` F32: the last committed row's is the next block's.
    pub(crate) fn head_context(&self, packed: &Tensor) -> Result<Tensor> {
        self.head_norm
            .forward(&self.mix(packed, self.norms.len())?)?
            .to_dtype(DType::F32)
    }
}

/// The recurrent Markov head: draft `s`'s logits gain `fold(tanh(out))`, `out` gated from the state that the previous
/// pick advanced, then pass the target's final softcap; the picks chain greedily.
pub(crate) struct RnnMarkovHead {
    rank: usize,
    // `[vocab, rank]`, at the model dtype as the vocabulary-wide fold
    w1: Tensor,
    // `rnn_proj` transposed, F32: its `[state, w1[prev]]` rows `[2 rank, 3 rank]`, its `[hidden, f_rnn]` rows
    // `[2 hidden, 3 rank]`, and its bias `[3 rank]`
    step_w: Tensor,
    context_w: Tensor,
    rnn_b: Tensor,
    hidden_out_w: Tensor,
    hidden_out_b: Tensor,
    softcap: f64,
    vocab: usize,
    // the head's input dtype
    dtype: DType,
    // `hidden_out` folded into the head: `[padded vocab, rank]` at the model dtype and `[padded vocab]` F32, made on
    // first use
    fold: Mutex<Option<(Tensor, Tensor)>>,
}

pub(crate) struct RnnShape {
    pub(crate) dtype: DType,
    pub(crate) padded_vocab: usize,
    pub(crate) vocab: usize,
    pub(crate) rank: usize,
    pub(crate) hidden: usize,
    pub(crate) softcap: f64,
}

impl RnnMarkovHead {
    pub(crate) fn load(vb: ShardedVarBuilder, shape: RnnShape) -> Result<Self> {
        let RnnShape {
            dtype,
            padded_vocab,
            vocab,
            rank,
            hidden,
            softcap,
        } = shape;
        let f32 = |t: Tensor| t.to_dtype(DType::F32);
        let rnn_w = f32(vb
            .pp("rnn_proj")
            .get((3 * rank, 2 * rank + 2 * hidden), "weight")?)?;
        let columns = |start: usize, len: usize| rnn_w.narrow(1, start, len)?.t()?.contiguous();
        Ok(Self {
            rank,
            w1: vb.pp("markov_w1").get((padded_vocab, rank), "weight")?,
            step_w: columns(0, 2 * rank)?,
            context_w: columns(2 * rank, 2 * hidden)?,
            rnn_b: f32(vb.pp("rnn_proj").get(3 * rank, "bias")?)?,
            hidden_out_w: f32(vb.pp("hidden_out").get((hidden, rank), "weight")?)?,
            hidden_out_b: f32(vb.pp("hidden_out").get(hidden, "bias")?)?,
            softcap,
            vocab,
            dtype,
            fold: Mutex::new(None),
        })
    }

    fn folded(&self, lm_head: &dyn QuantMethod) -> Result<(Tensor, Tensor)> {
        let mut fold = self.fold.lock().expect("markov fold poisoned");
        if let Some(fold) = fold.as_ref() {
            return Ok(fold.clone());
        }
        // `head . hidden_out` is the head over `hidden_out`'s columns, so the head never leaves its weight type; a few
        // columns at a time keep its f32 output small beside the paged cache
        let dtype = self.dtype;
        let columns = self.hidden_out_w.t()?.contiguous()?.to_dtype(dtype)?;
        let rank = columns.dim(0)?;
        let parts = (0..rank)
            .step_by(FOLD_COLUMNS)
            .map(|start| {
                lm_head
                    .forward(&columns.narrow(0, start, FOLD_COLUMNS.min(rank - start))?)?
                    .to_dtype(dtype)
            })
            .collect::<Result<Vec<_>>>()?;
        let weight = Tensor::cat(&parts, 0)?.t()?.contiguous()?;
        let bias = lm_head
            .forward(&self.hidden_out_b.unsqueeze(0)?.to_dtype(dtype)?)?
            .to_dtype(DType::F32)?
            .squeeze(0)?;
        let made = (weight, bias);
        *fold = Some(made.clone());
        Ok(made)
    }

    /// Greedy picks `[batch, drafts]` from the draft rows `[batch, drafts, hidden]`, their base logits
    /// `[batch, drafts, padded vocab]`, each sequence's head context `[batch, hidden]` and anchor `[batch]`.
    pub(crate) fn chain_greedy(
        &self,
        rows: &Tensor,
        logits: &Tensor,
        contexts: &Tensor,
        anchors: &Tensor,
        lm_head: &dyn QuantMethod,
    ) -> Result<Tensor> {
        let (batch, drafts, hidden) = rows.dims3()?;
        let r = self.rank;
        let (fold_w, fold_b) = self.folded(lm_head)?;
        let inputs = Tensor::cat(
            &[
                &rows.to_dtype(DType::F32)?,
                &contexts
                    .to_dtype(DType::F32)?
                    .unsqueeze(1)?
                    .broadcast_as((batch, drafts, hidden))?
                    .contiguous()?,
            ],
            D::Minus1,
        )?;
        let context = inputs
            .broadcast_matmul(&self.context_w)?
            .broadcast_add(&self.rnn_b)?;
        let mut state = Tensor::zeros((batch, r), DType::F32, rows.device())?;
        let mut prev = anchors.clone();
        let mut picks = Vec::with_capacity(drafts);
        for s in 0..drafts {
            let prev_rows = self.w1.index_select(&prev, 0)?.to_dtype(DType::F32)?;
            let raw = Tensor::cat(&[&state, &prev_rows], 1)?
                .matmul(&self.step_w)?
                .add(&context.i((.., s))?)?;
            let gate = inference_tensor::nn::ops::sigmoid(&raw.narrow(1, 0, r)?)?;
            let cand = raw.narrow(1, r, r)?.tanh()?;
            let keep = (&gate * &state)?;
            state = (keep + ((1.0 - &gate)? * cand)?)?;
            let latent = raw.narrow(1, 2 * r, r)?.tanh()?;
            let bias = latent
                .to_dtype(self.dtype)?
                .matmul(&fold_w.t()?)?
                .to_dtype(DType::F32)?
                .broadcast_add(&fold_b)?;
            let biased = (logits.i((.., s))?.to_dtype(DType::F32)? + bias)?;
            let capped = ((biased / self.softcap)?.tanh()? * self.softcap)?;
            prev = capped.narrow(1, 0, self.vocab)?.argmax(D::Minus1)?;
            picks.push(prev.clone());
        }
        Tensor::stack(&picks, 1)
    }
}

/// A `[hidden]` mask row from the drafter's `mask_embed`, if it carries one.
pub(crate) fn mask_embedding(
    vb: &ShardedVarBuilder,
    hidden: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    if !vb.contains_tensor("mask_embed") {
        return Ok(None);
    }
    Ok(Some(
        vb.get(hidden, "mask_embed")?
            .to_dtype(dtype)?
            .to_device(device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_inkling_checkpoint_describes_its_drafter_and_others_do_not() {
        let checkpoint = serde_json::json!({
            "text_config": {
                "model_type": "inkling2_loop_text",
                "vocab_size": 272,
                "unpadded_vocab_size": 266,
                "final_logit_softcapping": 30.0,
            },
            "mtp_config": {
                "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 3,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "rope_theta": 1e6,
                "local_layer_ids": [0, 2], "sliding_window_size": 1024, "block_size": 8,
                "markov_rank": 8, "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48],
            },
        });
        let config = drafter_config(&checkpoint).expect("an Inkling 2 checkpoint");
        assert_eq!(config["architectures"][0], ARCHITECTURE);
        assert_eq!(
            config["layer_types"],
            serde_json::json!([LOCAL, GLOBAL, LOCAL])
        );
        assert_eq!(config["sample_from_anchor"], false);
        assert_eq!(config["logit_vocab_size"], 266);
        assert_eq!(config["final_logit_softcapping"], 30.0);
        assert!(is_rnn_head(config["markov_head_type"].as_str()));
        let mut other = checkpoint.clone();
        other["text_config"]["model_type"] = "llama".into();
        assert!(drafter_config(&other).is_none());
    }
}
