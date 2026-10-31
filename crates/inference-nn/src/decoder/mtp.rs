//! A multi-token prediction head beside the stack (ERNIE 4.5's `mtp_*`, the DeepSeek-V3 layout):
//! `proj(concat(norm(embed(next token)), norm(target hidden)))` through one decoder block, then the target's `lm_head`
//! (after the head's own norm, when it has one). Its attention writes the paged pool's layer after the stack's, through
//! the sequence's block table, so a rejected draft's K/V is trimmed with the target's.

use std::{collections::HashMap, sync::Arc, sync::Mutex};

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::{D, Device, Result, Tensor};

use crate::{
    attention::{AttentionMask, FlashParams},
    kv_cache::KvCache,
    model::ModelForwardContext,
    paged_attention::{AttentionImplementation, PagedAttentionMeta},
    speculative::{
        SpeculativeGraphState, SpeculativeKvCache, SpeculativePrefillCtx, SpeculativeProposal,
        SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
        mtp::{MtpDraft, MtpForwardInputs},
        paged_rows::make_paged_rows_metadata,
        proposer::sample_draft_rows,
    },
    utils::unvarbuilder::UnVarBuilder,
};

use super::{
    AttentionBlock, DecoderLayer, DecoderSpec, DecoderStack, Ffn, LayerBuilder, LayerLoad,
    LayerNorms, Norm, StandardLayers,
};

// the placeholder cache a paged layer is handed but never appends to
const PLACEHOLDER_KV_DIM: usize = 2;

/// Where a stack's MTP head keeps its weights, under the stack's prefix.
#[derive(Clone, Copy, Debug)]
pub struct MtpSpec {
    pub embedding_norm: &'static str,
    pub hidden_norm: &'static str,
    /// `[2 * hidden, hidden]` over `concat(normed embedding, normed hidden)`.
    pub projection: &'static str,
    /// One decoder layer, named as the stack's layers are inside.
    pub block: &'static str,
    /// The head's norm before `lm_head` (DeepSeek's `shared_head.norm`); ERNIE's feeds the block's output straight in.
    pub final_norm: Option<&'static str>,
    /// The block keeps a dense MLP though the stack's layers route over experts.
    pub dense_block: bool,
    /// Position 0 has no previous token, so its embedding is zeroed (vLLM's ERNIE and DeepSeek heads).
    pub mask_first_position: bool,
}

struct DraftRow {
    seq_id: usize,
    position: usize,
}

/// The token each row reads: host values, or a `[rows]` U32 tensor a device sampler left on the GPU.
enum RowTokens<'a> {
    Host(Vec<u32>),
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Device(&'a Tensor),
}

/// The target's last forward, post final norm: every row, and the rows its logits came from.
struct Capture {
    full: Tensor,
    selected: Tensor,
}

/// What a replayed decode graph hands the head: the captured forward's two hidden-state outputs.
struct MtpGraphState {
    full: Tensor,
    selected: Tensor,
}

impl SpeculativeGraphState for MtpGraphState {
    fn tensors(&self) -> Vec<Tensor> {
        vec![self.full.clone(), self.selected.clone()]
    }

    fn with_tensors(&self, tensors: Vec<Tensor>) -> Result<Box<dyn SpeculativeGraphState>> {
        let [full, selected]: [Tensor; 2] =
            tensors.try_into().map_err(|tensors: Vec<Tensor>| {
                inference_tensor::Error::msg(format!(
                    "the MTP graph state holds two hidden tensors, got {}",
                    tensors.len()
                ))
            })?;
        Ok(Box::new(Self { full, selected }))
    }

    fn for_real_batch(&self, real_batch: usize) -> Result<Box<dyn SpeculativeGraphState>> {
        let rows = |hidden: &Tensor| -> Result<Tensor> {
            let captured = hidden.dim(0)?;
            if real_batch > captured {
                inference_tensor::bail!(
                    "MTP batch {real_batch} exceeds the captured batch {captured}"
                );
            }
            hidden.narrow(0, 0, real_batch)
        };
        Ok(Box::new(Self {
            full: rows(&self.full)?,
            selected: rows(&self.selected)?,
        }))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct MtpHead {
    embedding_norm: Norm,
    hidden_norm: Norm,
    projection: Arc<dyn QuantMethod>,
    block: DecoderLayer<AttentionBlock, Ffn>,
    final_norm: Option<Norm>,
    names: MtpSpec,
    layer_norm_names: super::NormNames,
    self_attn_name: &'static str,
    mask_first_position: bool,
    kv_layer_idx: usize,
    device: Device,
    dtype: inference_tensor::DType,
    placeholder_kv: Mutex<KvCache>,
    n_predict: usize,
    capture: Mutex<Option<Capture>>,
    // the last prompt row's hidden state, whose next token is only known once it is sampled
    pending_tails: Mutex<HashMap<usize, (usize, Tensor)>>,
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    graphs: crate::speculative::mtp_graph::MtpGraphs,
}

impl MtpHead {
    /// The head on the device the stack's head and final norm use, as the paged layer after the stack's.
    pub(super) fn load<F>(
        spec: &DecoderSpec,
        mtp: &MtpSpec,
        vb_m: &ShardedVarBuilder,
        stack: &DecoderStack<AttentionBlock, F>,
        is_gptx: bool,
        loading_isq: bool,
        attention: &AttentionImplementation,
    ) -> Result<Self> {
        let mapper = &*stack.mapper;
        let device = stack.device.clone();
        let kv_layer_idx = spec.num_layers();
        let block_spec = DecoderSpec {
            moe: spec.moe.clone().filter(|_| !mtp.dense_block),
            layer_windows: vec![None; kv_layer_idx + 1],
            ..spec.clone()
        };
        let dtype = vb_m.dtype();
        let rope = spec.rope(&spec.rope, vb_m, &device, is_gptx, dtype)?;
        let builder = StandardLayers {
            spec: &block_spec,
            ropes: HashMap::from([(device.location(), Arc::new(rope))]),
            local_ropes: None,
        };
        let comm = mapper.get_comm_for(kv_layer_idx)?;
        let load = LayerLoad {
            mapper,
            layer_idx: kv_layer_idx,
            loading_isq,
            comm: &comm,
            device: &device,
            attention,
        };
        let vb_block = vb_m.pp(mtp.block);
        let (self_attn, mlp) = builder.build(&load, vb_block.clone())?;
        let names = spec.norm_names;
        if names.post_attn.is_some() || names.post_ffn.is_some() {
            inference_tensor::bail!(
                "an MTP head over post-norm or sandwich layers is not supported"
            );
        }
        let norm = |vb: ShardedVarBuilder| {
            Norm::new(
                spec.norm,
                spec.hidden_size,
                spec.rms_norm_eps,
                mapper.set_nm_device(vb, false),
            )
        };
        let block = DecoderLayer {
            self_attn,
            mlp,
            norms: LayerNorms {
                input: names
                    .input
                    .map(|name| norm(vb_block.pp(name)))
                    .transpose()?,
                pre_ffn: names
                    .pre_ffn
                    .map(|name| norm(vb_block.pp(name)))
                    .transpose()?,
                post: None,
            },
        };
        Ok(Self {
            embedding_norm: norm(vb_m.pp(mtp.embedding_norm))?,
            hidden_norm: norm(vb_m.pp(mtp.hidden_norm))?,
            projection: spec.replicated_linear(
                2 * spec.hidden_size,
                spec.hidden_size,
                false,
                mapper.set_nm_device(vb_m.pp(mtp.projection), loading_isq),
            )?,
            block,
            final_norm: mtp.final_norm.map(|name| norm(vb_m.pp(name))).transpose()?,
            names: *mtp,
            layer_norm_names: names,
            self_attn_name: spec.module_names.self_attn,
            mask_first_position: mtp.mask_first_position,
            kv_layer_idx,
            device,
            dtype,
            placeholder_kv: Mutex::new(KvCache::new_normal(PLACEHOLDER_KV_DIM, 1, 1)),
            n_predict: 0,
            capture: Mutex::new(None),
            pending_tails: Mutex::new(HashMap::new()),
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            graphs: Default::default(),
        })
    }

    pub fn n_predict(&self) -> usize {
        self.n_predict
    }

    pub(super) fn set_n_predict(&mut self, n_predict: usize) {
        self.n_predict = n_predict;
        *self.capture.lock().expect("mtp capture poisoned") = None;
        self.pending_tails
            .lock()
            .expect("mtp tails poisoned")
            .clear();
    }

    /// Keeps the target's post-norm hidden states while drafting is on.
    pub(super) fn record(&self, full: &Tensor, selected: &Tensor) {
        if self.n_predict > 0 {
            *self.capture.lock().expect("mtp capture poisoned") = Some(Capture {
                full: full.clone(),
                selected: selected.clone(),
            });
        }
    }

    /// The capture a decode graph's forward just made, which its replays hand back through [`MtpHead::install`].
    pub(super) fn take_graph_state(&self) -> Option<Box<dyn SpeculativeGraphState>> {
        let capture = self.capture.lock().expect("mtp capture poisoned");
        capture.as_ref().map(|capture| {
            Box::new(MtpGraphState {
                full: capture.full.clone(),
                selected: capture.selected.clone(),
            }) as Box<dyn SpeculativeGraphState>
        })
    }

    pub(super) fn install(&self, state: &dyn SpeculativeGraphState) -> Result<()> {
        let state = state
            .as_any()
            .downcast_ref::<MtpGraphState>()
            .ok_or_else(|| {
                inference_tensor::Error::msg("foreign speculative graph state for an MTP head")
            })?;
        *self.capture.lock().expect("mtp capture poisoned") = Some(Capture {
            full: state.full.clone(),
            selected: state.selected.clone(),
        });
        Ok(())
    }

    /// The last forward's selected hidden rows, `(batch_idx, row)` each, as `[rows, hidden]`.
    pub(super) fn target_hiddens(&self, rows: &[(usize, usize)]) -> Result<Option<Tensor>> {
        let capture = self.capture.lock().expect("mtp capture poisoned");
        let Some(capture) = capture.as_ref() else {
            return Ok(None);
        };
        let rows = rows
            .iter()
            .map(|&(batch_idx, row)| capture.selected.get(batch_idx)?.narrow(0, row, 1))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Tensor::cat(&rows, 0)?))
    }

    /// One query per row, each its own batch entry at its position; `hidden` is `[rows, hidden]`. The draft's
    /// `logits_hidden` feeds `lm_head` and its `chain_hidden` (the block's output) the next chained draft, `[rows, hidden]`
    /// each. A drafting step replays it from a CUDA graph per row count.
    fn forward_rows(
        &self,
        embed: &dyn Fn(&Tensor) -> Result<Tensor>,
        rows: &[DraftRow],
        tokens: RowTokens<'_>,
        hidden: &Tensor,
        kv_cache: &[(Tensor, Tensor)],
        paged_meta: &PagedAttentionMeta,
    ) -> Result<MtpDraft> {
        let n = rows.len();
        let seq_ids = rows.iter().map(|row| row.seq_id).collect::<Vec<_>>();
        let starts = rows.iter().map(|row| row.position).collect::<Vec<_>>();
        let context_lens = starts.iter().map(|p| p + 1).collect::<Vec<_>>();
        let hidden = hidden
            .to_device(&self.device)?
            .to_dtype(self.dtype)?
            .reshape((n, 1, ()))?
            .contiguous()?;
        let forward = |inputs: MtpForwardInputs<'_>| self.forward_inputs(embed, inputs, kv_cache);
        // a position-0 row's zeroed embedding is masked on the host, which a graph would bake in
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        if paged_meta.sliding_window.is_none() && !(self.mask_first_position && starts.contains(&0))
        {
            let (k, v) = &kv_cache[self.kv_layer_idx];
            let run = crate::speculative::mtp_graph::MtpGraphRun {
                shape: (n, 1),
                tokens: match &tokens {
                    RowTokens::Host(tokens) => {
                        crate::speculative::mtp_graph::MtpGraphTokens::Host(tokens)
                    }
                    RowTokens::Device(tokens) => {
                        crate::speculative::mtp_graph::MtpGraphTokens::Device(tokens)
                    }
                },
                head_positions: &[],
                positions: &context_lens,
                starts: &starts,
                target_hidden: &hidden,
                decode_rows: crate::speculative::paged_rows::make_paged_rows_decode_rows(
                    &seq_ids,
                    &context_lens,
                    paged_meta,
                    &self.device,
                )?,
                kv: vec![k.id(), v.id()],
            };
            if let Some(draft) = self.graphs.forward(&run, &forward)? {
                return Self::flatten(draft, n);
            }
        }
        let metadata = make_paged_rows_metadata(&seq_ids, &context_lens, paged_meta, &self.device)?;
        let tokens = match tokens {
            RowTokens::Host(tokens) => Tensor::from_vec(tokens, (n, 1), &self.device)?,
            RowTokens::Device(tokens) => tokens.reshape((n, 1))?,
        };
        let draft = forward(MtpForwardInputs {
            tokens: &tokens,
            positions: None,
            target_hidden: &hidden,
            metadata: &metadata,
            starts: &starts,
        })?;
        Self::flatten(draft, n)
    }

    fn flatten(draft: MtpDraft, rows: usize) -> Result<MtpDraft> {
        Ok(MtpDraft {
            logits_hidden: draft.logits_hidden.reshape((rows, ()))?,
            chain_hidden: draft.chain_hidden.reshape((rows, ()))?,
        })
    }

    /// The drafter forward over device inputs, eager or under a graph capture: `[n, 1]` tokens and `[n, 1, hidden]`
    /// target hidden states, roped from `inputs.metadata`'s positions or else `inputs.starts`.
    fn forward_inputs(
        &self,
        embed: &dyn Fn(&Tensor) -> Result<Tensor>,
        inputs: MtpForwardInputs<'_>,
        kv_cache: &[(Tensor, Tensor)],
    ) -> Result<MtpDraft> {
        let n = inputs.starts.len();
        let mut embeds = embed(inputs.tokens)?;
        if self.mask_first_position && inputs.starts.contains(&0) {
            let keep = inputs
                .starts
                .iter()
                .map(|&position| f32::from(u8::from(position != 0)))
                .collect::<Vec<_>>();
            let keep = Tensor::from_vec(keep, (n, 1, 1), &self.device)?.to_dtype(embeds.dtype())?;
            embeds = embeds.broadcast_mul(&keep)?;
        }
        let xs = self.projection.forward(&Tensor::cat(
            &[
                self.embedding_norm.forward(&embeds)?,
                self.hidden_norm.forward(inputs.target_hidden)?,
            ],
            D::Minus1,
        )?)?;
        let logit_rows = vec![(0, 1); n];
        let flash = FlashParams::empty(false);
        let mut ctx = ModelForwardContext::new(
            inputs.starts,
            &logit_rows,
            inputs.starts,
            Some((kv_cache, inputs.metadata)),
            &flash,
        );
        let mut placeholder = self.placeholder_kv.lock().expect("mtp cache poisoned");
        let chain_hidden = self.block.forward(
            &xs,
            &AttentionMask::None,
            Some(&mut placeholder),
            &mut ctx,
            self.kv_layer_idx,
            None,
        )?;
        let logits_hidden = match &self.final_norm {
            Some(norm) => norm.forward(&chain_hidden)?,
            None => chain_hidden.clone(),
        };
        Ok(MtpDraft {
            logits_hidden,
            chain_hidden,
        })
    }

    /// Catches the head up over a prompt chunk: row p takes token p+1 and the target's hidden at p, except the last
    /// prompt row, kept until its next token is sampled.
    pub(super) fn prefill(
        &self,
        embed: &dyn Fn(&Tensor) -> Result<Tensor>,
        ctx: SpeculativePrefillCtx<'_>,
    ) -> Result<()> {
        let SpeculativeKvCache::Paged {
            metadata: paged_meta,
            kv_cache,
        } = ctx.cache;
        let full = match self.capture.lock().expect("mtp capture poisoned").as_ref() {
            Some(capture) => capture.full.clone(),
            None => return Ok(()),
        };
        let mut rows = Vec::new();
        let mut tokens = Vec::new();
        let mut hidden_rows = Vec::new();
        let mut tails = self.pending_tails.lock().expect("mtp tails poisoned");
        for (i, &seq_id) in ctx.seq_ids.iter().enumerate() {
            let batch_idx = ctx.batch_indices[i];
            let toks = ctx.tokens[i];
            let (start, end) = ctx.chunk_ranges[i];
            if end <= start || full.dim(1)? < end - start {
                inference_tensor::bail!(
                    "MTP prefill rows out of range: chunk=({start}, {end}), hidden rows={}",
                    full.dim(1)?
                );
            }
            let seq_hidden = full.get(batch_idx)?;
            // a chunk after the first resumes from the tail its predecessor left
            if let Some((position, hidden)) = tails.remove(&seq_id)
                && position + 1 == start
            {
                rows.push(DraftRow { seq_id, position });
                tokens.push(toks[start]);
                hidden_rows.push(hidden);
            }
            let last = end - 1;
            tails.insert(seq_id, (last, seq_hidden.narrow(0, last - start, 1)?));
            for position in start..last {
                rows.push(DraftRow { seq_id, position });
                tokens.push(toks[position + 1]);
            }
            if last > start {
                hidden_rows.push(seq_hidden.narrow(0, 0, last - start)?);
            }
        }
        drop(tails);
        if rows.is_empty() {
            return Ok(());
        }
        let hidden = Tensor::cat(&hidden_rows, 0)?;
        self.forward_rows(
            embed,
            &rows,
            RowTokens::Host(tokens),
            &hidden,
            kv_cache,
            paged_meta,
        )?;
        Ok(())
    }

    /// Drafts `ctx.proposal_len` tokens per sequence: a refresh over the rows the last verify kept (and any prompt
    /// tail), then a chain of single-row steps, each fed the previous draft.
    pub(super) fn propose(
        &self,
        embed: &dyn Fn(&Tensor) -> Result<Tensor>,
        lm_head: &dyn QuantMethod,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        let n_predict = ctx.proposal_len;
        let batch = ctx.sequences.len();
        if batch == 0 || n_predict == 0 {
            return Ok(None);
        }
        if n_predict > self.n_predict {
            inference_tensor::bail!(
                "MTP proposal length {n_predict} is outside the configured range 1..={}",
                self.n_predict
            );
        }
        let SpeculativeKvCache::Paged {
            metadata: paged_meta,
            kv_cache,
        } = ctx.cache;
        if kv_cache.len() <= self.kv_layer_idx {
            inference_tensor::bail!("the paged cache has no MTP layer");
        }
        let selected = match self.capture.lock().expect("mtp capture poisoned").as_ref() {
            Some(capture) => capture.selected.clone(),
            None => return Ok(None),
        };
        {
            let mut kv_mgr = crate::get_mut_arcmutex!(paged_meta.kv_cache_manager);
            for (seq_id, base_len) in ctx.seq_ids.iter().zip(ctx.base_lens) {
                if kv_mgr
                    .allocate_slots(*seq_id, base_len + n_predict, &[])
                    .is_none()
                {
                    return Ok(None);
                }
            }
        }

        let mut rows = Vec::new();
        let mut row_tokens = Vec::new();
        let mut hidden_rows = Vec::with_capacity(batch);
        let mut last_rows = Vec::with_capacity(batch);
        let mut tails = self.pending_tails.lock().expect("mtp tails poisoned");
        for (i, seq) in ctx.sequences.iter().enumerate() {
            let (batch_idx, count) = ctx.target_rows[i];
            let base_len = ctx.base_lens[i];
            let toks = seq.get_toks();
            if count == 0 || base_len < count || toks.len() <= base_len {
                inference_tensor::bail!(
                    "MTP refresh rows out of range: base_len={base_len}, count={count}, toks={}",
                    toks.len()
                );
            }
            let seq_id = ctx.seq_ids[i];
            if let Some((position, hidden)) = tails.remove(&seq.id())
                && position + 1 < toks.len()
            {
                rows.push(DraftRow { seq_id, position });
                row_tokens.push(toks[position + 1]);
                hidden_rows.push(hidden);
            }
            for r in 0..count {
                let position = base_len - count + r;
                rows.push(DraftRow { seq_id, position });
                row_tokens.push(toks[position + 1]);
            }
            hidden_rows.push(selected.get(batch_idx)?.narrow(0, 0, count)?);
            last_rows.push(rows.len() - 1);
        }
        tails.retain(|seq_id, _| ctx.seq_ids.contains(seq_id));
        drop(tails);
        let hidden = Tensor::cat(&hidden_rows, 0)?;
        let out = self.forward_rows(
            embed,
            &rows,
            RowTokens::Host(row_tokens),
            &hidden,
            kv_cache,
            paged_meta,
        )?;
        #[allow(clippy::cast_possible_truncation)]
        let last = Tensor::from_vec(
            last_rows.iter().map(|&row| row as u32).collect::<Vec<_>>(),
            (batch,),
            out.chain_hidden.device(),
        )?;
        let mut draft = MtpDraft {
            logits_hidden: out.logits_hidden.index_select(&last, 0)?,
            chain_hidden: out.chain_hidden.index_select(&last, 0)?,
        };
        let mut cursor = last_rows
            .iter()
            .map(|&row| rows[row].position)
            .collect::<Vec<_>>();

        // drafts left on the device reach the verifier without a read-back of their own
        #[cfg(feature = "cuda")]
        {
            let device = draft.chain_hidden.device().clone();
            let sampler = crate::speculative::proposer::CudaDraftSampler::new(
                ctx.sequences,
                n_predict,
                &device,
                &ctx.rng,
            )?;
            let greedy =
                device.is_cuda() && ctx.sequences.iter().all(|seq| seq.sampler().is_argmax());
            if sampler.is_some() || greedy {
                let mut steps = Vec::with_capacity(n_predict);
                let mut samples = Vec::with_capacity(n_predict);
                let mut logits = Vec::with_capacity(n_predict);
                for step in 0..n_predict {
                    let step_logits = lm_head.forward(&draft.logits_hidden)?;
                    let vocab = step_logits.dim(D::Minus1)?;
                    let drafts = match &sampler {
                        Some(sampler) => {
                            let sample = sampler.sample_on_device(step, &step_logits)?;
                            let drafts = sample.tokens.clone();
                            samples.push(sample);
                            drafts
                        }
                        None => {
                            let drafts = greedy_device_drafts(&step_logits)?;
                            logits.push(step_logits);
                            drafts
                        }
                    };
                    steps.push(drafts.clone());
                    if step + 1 == n_predict {
                        break;
                    }
                    let chained = ctx
                        .seq_ids
                        .iter()
                        .zip(cursor.iter_mut())
                        .map(|(&seq_id, position)| {
                            *position += 1;
                            DraftRow {
                                seq_id,
                                position: *position,
                            }
                        })
                        .collect::<Vec<_>>();
                    // a row with no finite logits drafts an invalid id, which the embedding must not read
                    #[allow(clippy::cast_possible_truncation)]
                    let drafts = drafts.clamp(0u32, (vocab - 1) as u32)?;
                    draft = self.forward_rows(
                        embed,
                        &chained,
                        RowTokens::Device(&drafts),
                        &draft.chain_hidden,
                        kv_cache,
                        paged_meta,
                    )?;
                }
                // [batch, n_predict]
                let tokens = Tensor::stack(&steps, 1)?;
                let stack = |parts: Vec<&Tensor>| Tensor::stack(&parts, 1);
                let sparse = (!samples.is_empty())
                    .then(|| {
                        Ok::<_, inference_tensor::Error>((
                            stack(samples.iter().map(|sample| &sample.ids).collect())?,
                            stack(samples.iter().map(|sample| &sample.probs).collect())?,
                        ))
                    })
                    .transpose()?;
                let logits = (!logits.is_empty())
                    .then(|| stack(logits.iter().collect()))
                    .transpose()?;
                let proposals = (0..batch)
                    .map(|row| {
                        let tokens =
                            crate::speculative::SpeculativeTokens::from_device(tokens.get(row)?)?;
                        let distribution = match (&sparse, &logits) {
                            (Some((ids, probs)), _) => {
                                crate::speculative::SpeculativeProposalDistribution::SparseProbs(
                                    crate::speculative::SparseSpeculativeProbs::new(
                                        ids.get(row)?,
                                        probs.get(row)?,
                                    )?,
                                )
                            }
                            (None, Some(logits)) => {
                                crate::speculative::SpeculativeProposalDistribution::Logits(
                                    logits.get(row)?,
                                )
                            }
                            (None, None) => {
                                unreachable!("a device draft step keeps its distribution")
                            }
                        };
                        Ok(SpeculativeProposal {
                            tokens,
                            distribution: Some(distribution),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                return Ok(Some(SpeculativeProposalBatch::new(proposals)));
            }
        }

        let mut contexts = ctx
            .sequences
            .iter()
            .map(|seq| seq.get_toks().to_vec())
            .collect::<Vec<_>>();
        let mut tokens = vec![Vec::with_capacity(n_predict); batch];
        let mut logits = Vec::with_capacity(n_predict);
        for step in 0..n_predict {
            let step_logits = lm_head.forward(&draft.logits_hidden)?;
            let drafts = sample_draft_rows(&step_logits, ctx.sequences, &mut contexts, &ctx.rng)?;
            logits.push(step_logits);
            for (i, draft) in drafts.iter().enumerate() {
                tokens[i].push(*draft);
            }
            if step + 1 == n_predict {
                break;
            }
            let chained = ctx
                .seq_ids
                .iter()
                .zip(cursor.iter_mut())
                .map(|(&seq_id, position)| {
                    *position += 1;
                    DraftRow {
                        seq_id,
                        position: *position,
                    }
                })
                .collect::<Vec<_>>();
            draft = self.forward_rows(
                embed,
                &chained,
                RowTokens::Host(drafts),
                &draft.chain_hidden,
                kv_cache,
                paged_meta,
            )?;
        }
        // [n_predict, batch, vocab] -> per sequence [n_predict, vocab]
        let logits = Tensor::stack(&logits, 1)?;
        let proposals = tokens
            .into_iter()
            .enumerate()
            .map(|(row, tokens)| Ok(SpeculativeProposal::with_logits(tokens, logits.get(row)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(SpeculativeProposalBatch::new(proposals)))
    }

    /// The tensors ISQ leaves alone, under the stack's prefix.
    pub(super) fn add_residual(&self, uvb_m: &UnVarBuilder) {
        let norm_names = self.layer_norm_names;
        self.embedding_norm
            .add_residual(&uvb_m.pp(self.names.embedding_norm));
        self.hidden_norm
            .add_residual(&uvb_m.pp(self.names.hidden_norm));
        if let (Some(norm), Some(name)) = (&self.final_norm, self.names.final_norm) {
            norm.add_residual(&uvb_m.pp(name));
        }
        let uvb_b = uvb_m.pp(self.names.block);
        if let (Some(norm), Some(name)) = (&self.block.norms.input, norm_names.input) {
            norm.add_residual(&uvb_b.pp(name));
        }
        if let (Some(norm), Some(name)) = (&self.block.norms.pre_ffn, norm_names.pre_ffn) {
            norm.add_residual(&uvb_b.pp(name));
        }
        super::LayerAttention::add_residual(&self.block.self_attn, &uvb_b.pp(self.self_attn_name));
    }
}

/// Each row's argmax as a `[rows]` U32 tensor, left on the device.
#[cfg(feature = "cuda")]
fn greedy_device_drafts(logits: &Tensor) -> Result<Tensor> {
    let logits = logits
        .to_dtype(inference_tensor::DType::F32)?
        .contiguous()?;
    let top1 = crate::ops::cuda_top1_logits_f32_packed_batched(&logits)?;
    // packed rows are (value, id), the id exact in F32 below 2^24
    top1.packed
        .narrow(1, 1, 1)?
        .squeeze(1)?
        .to_dtype(inference_tensor::DType::U32)?
        .contiguous()
}
