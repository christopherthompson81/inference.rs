//! Inkling 2 as the target of an external drafter (its own `mtp.safetensors` among them): the drafter's taps by layer
//! visit (the looped core's per pass, the visit count itself the normed output), and a verify step's convolution
//! states cut back to the rows it kept.

use std::sync::{Arc, Mutex};

use inference_drafters::dflash::DFlashLoadTarget;
use inference_drafters::host::{DrafterAttach, DrafterTarget, ExternalDrafter};
use inference_quant::QuantMethod;
use inference_tensor::{Module, Result, Tensor};

use crate::kv_cache::HybridLayerCache;
use crate::speculative::{
    MtpConfig, MtpRuntimeConfig, SpeculativeAttachInfo, SpeculativeCommitRow, SpeculativeConfig,
};

use super::Model;

/// A verify step's convolution state of one cache slot before the step, and the inputs its convs took over the
/// step's rows: a row that keeps `k` of them keeps the last `width - 1` of `[state, inputs[..k]]`.
#[derive(Clone)]
pub(super) struct ConvReplay {
    // the hybrid cache layer of the slot's states
    pub(super) slot: usize,
    pub(super) indices: Tensor,
    pub(super) host_indices: Option<Vec<u32>>,
    // `[b, channels, width - 1]` and `[b, t, channels]`, F32
    pub(super) state: Tensor,
    pub(super) inputs: Tensor,
}

#[derive(Default)]
pub(super) struct SpecState {
    drafter: Mutex<Option<Arc<ExternalDrafter>>>,
    draft_lm_head: Option<Arc<dyn QuantMethod>>,
    // the visits whose inputs are tapped, then whether the normed output is
    visits: Vec<usize>,
    output: bool,
    spec_taps: Mutex<Option<Vec<Tensor>>>,
    prompt_taps: Mutex<Option<Vec<Tensor>>>,
    replays: Mutex<Vec<ConvReplay>>,
}

impl SpecState {
    pub(super) fn drafter(&self) -> Option<Arc<ExternalDrafter>> {
        self.drafter.lock().expect("drafter poisoned").clone()
    }

    pub(super) fn tap_visits(&self) -> &[usize] {
        &self.visits
    }

    pub(super) fn taps_output(&self) -> bool {
        self.output
    }

    /// A forward's taps in the drafter's order (`logits` selects the rows its logits come from) and its verify replays.
    pub(super) fn store(
        &self,
        taps: Vec<Tensor>,
        replays: Vec<ConvReplay>,
        prompt: bool,
        logits: impl Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<()> {
        if self.visits.is_empty() && !self.output {
            return Ok(());
        }
        let selected = taps.iter().map(logits).collect::<Result<Vec<_>>>()?;
        *self.spec_taps.lock().expect("spec taps poisoned") = Some(selected);
        *self.prompt_taps.lock().expect("prompt taps poisoned") = prompt.then_some(taps);
        *self.replays.lock().expect("conv replays poisoned") = replays;
        Ok(())
    }

    fn clear(&mut self) {
        *self.drafter.lock().expect("drafter poisoned") = None;
        self.draft_lm_head = None;
        self.visits.clear();
        self.output = false;
        *self.spec_taps.lock().expect("spec taps poisoned") = None;
        *self.prompt_taps.lock().expect("prompt taps poisoned") = None;
        self.replays.lock().expect("conv replays poisoned").clear();
    }
}

impl Model {
    // every layer visit, then the normed output
    fn tap_points(&self) -> usize {
        self.cfg.cache_slots() + 1
    }

    fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        let x = self.embed_tokens.embedding_forward(ids, self.dtype)?;
        self.embed_norm.forward(&x)
    }

    fn attach_external_drafter(
        &mut self,
        config: &MtpConfig,
        runtime: MtpRuntimeConfig,
    ) -> Result<SpeculativeAttachInfo> {
        let (recurrent_capacity, serving_capacity, checkpoint_lanes) = {
            let cache = self.cache.hybrid();
            let recurrent = cache.recurrent_capacity();
            let serving = cache.fitted_serving_capacity().unwrap_or_else(|| {
                recurrent.saturating_sub(crate::kv_cache::RECURRENT_GRAPH_PAD_SLOTS)
            });
            (recurrent, serving, cache.checkpoint_lanes())
        };
        let (drafter, info) = ExternalDrafter::attach(
            config,
            DrafterAttach {
                load: DFlashLoadTarget {
                    num_layers: self.tap_points(),
                    hidden_size: self.cfg.hidden_size,
                    // the drafter brings its own RoPE; the target has none
                    yarn_rope_config: None,
                    device: &self.device,
                    dtype: self.dtype,
                },
                checkpoint_lanes,
                recurrent_capacity,
                serving_capacity,
                paged_first_layer: self.meta.num_layers,
                runtime,
                recurrent_pad_slots: crate::kv_cache::RECURRENT_GRAPH_PAD_SLOTS,
            },
        )?;
        let output = self.tap_points() - 1;
        let ids = &drafter.target_layer_ids;
        if ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            inference_tensor::bail!("Inkling 2 taps layer visits in order, got {ids:?}");
        }
        self.spec.visits = ids.iter().copied().filter(|&id| id < output).collect();
        self.spec.output = ids.contains(&output);
        self.spec.draft_lm_head = config
            .draft_lm_head_isq
            .map(|ty| {
                self.lm_head.clone().apply_isq(
                    Some(ty),
                    self.device.clone(),
                    &std::sync::atomic::AtomicUsize::new(0),
                    None,
                    inference_quant::QuantizeOntoGuard::new(),
                )
            })
            .transpose()?;
        *self.spec.drafter.lock().expect("drafter poisoned") = Some(Arc::new(drafter));
        Ok(info)
    }

    // each kept row's states: the window of `[state, inputs]` ending at its last kept row
    fn roll_back_conv_states(&self, rows: &[SpeculativeCommitRow]) -> Result<()> {
        let replays =
            std::mem::take(&mut *self.spec.replays.lock().expect("conv replays poisoned"));
        let mut groups: std::collections::BTreeMap<usize, Vec<u32>> = Default::default();
        for row in rows.iter().filter(|row| !row.accepted_all) {
            groups
                .entry(row.keep_rows)
                .or_default()
                .push(u32::try_from(row.batch_idx).map_err(inference_tensor::Error::wrap)?);
        }
        if replays.is_empty() || groups.is_empty() {
            return Ok(());
        }
        let history = self.cfg.sconv_kernel_size - 1;
        let device = replays[0].indices.device().clone();
        let group_rows = groups
            .values()
            .map(|rows| Tensor::new(rows.as_slice(), &device))
            .collect::<Result<Vec<_>>>()?;
        let mut cache = self.cache.hybrid();
        for replay in &replays {
            for ((&keep, batch_rows), on_device) in groups.iter().zip(&group_rows) {
                let pick = |t: &Tensor| t.contiguous()?.index_select(on_device, 0);
                let window = Tensor::cat(
                    &[
                        &pick(&replay.state)?.transpose(1, 2)?,
                        &pick(&replay.inputs)?,
                    ],
                    1,
                )?
                .narrow(1, keep, history)?
                .transpose(1, 2)?
                .contiguous()?;
                let slots = pick(&replay.indices)?;
                let host: Option<Vec<u32>> = replay
                    .host_indices
                    .as_ref()
                    .map(|host| batch_rows.iter().map(|&row| host[row as usize]).collect());
                let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(replay.slot) else {
                    inference_tensor::bail!("Inkling 2: cache layer {} has no states", replay.slot);
                };
                pool.scatter_conv_state_with_host_indices(&slots, host.as_deref(), &window)?;
            }
        }
        Ok(())
    }
}

impl DrafterTarget for Model {
    fn spec_taps(&self) -> Option<Vec<Tensor>> {
        self.spec
            .spec_taps
            .lock()
            .expect("spec taps poisoned")
            .clone()
    }

    fn prompt_taps(&self) -> Option<Vec<Tensor>> {
        self.spec
            .prompt_taps
            .lock()
            .expect("prompt taps poisoned")
            .clone()
    }

    fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed(ids)
    }

    fn token_embedding(&self) -> &Arc<dyn QuantMethod> {
        &self.embed_tokens
    }

    fn draft_lm_head(&self) -> Arc<dyn QuantMethod> {
        self.spec
            .draft_lm_head
            .clone()
            .unwrap_or_else(|| self.lm_head.clone())
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        self.attach_speculative_with_runtime(config, Default::default())
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: SpeculativeConfig,
        runtime: MtpRuntimeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        self.spec.clear();
        let SpeculativeConfig::Mtp(config) = config else {
            return Ok(None);
        };
        if config.is_builtin() {
            inference_tensor::bail!(
                "Inkling 2 drafts with its checkpoint's drafter: pass the checkpoint directory as `--mtp-model`"
            );
        }
        self.attach_external_drafter(&config, runtime).map(Some)
    }

    fn has_speculative_proposer(&self) -> bool {
        self.spec.drafter().is_some()
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        self.spec.drafter().is_some()
    }

    fn speculative_plan(
        &self,
        batch_size: usize,
    ) -> Option<crate::speculative::SpeculativeBatchPlan> {
        let drafter = self.spec.drafter()?;
        Some(drafter.plan(drafter.n_predict(), batch_size))
    }

    fn speculative_graph_plans(&self) -> Vec<crate::speculative::SpeculativeGraphPlan> {
        self.spec
            .drafter()
            .map(|drafter| drafter.graph_plans(drafter.n_predict()))
            .unwrap_or_default()
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn precapture_speculative_cuda_graphs(&self) -> Result<()> {
        match self.spec.drafter() {
            Some(drafter) => drafter.precapture(self, drafter.n_predict()),
            None => Ok(()),
        }
    }

    fn evict_speculative_cuda_graphs(&self, max_entries: usize) -> usize {
        self.spec
            .drafter()
            .map_or(0, |drafter| drafter.evict_cuda_graphs_lru(max_entries))
    }

    fn speculative_bypass(&mut self, seq_ids: &[usize]) -> Result<()> {
        if let Some(drafter) = self.spec.drafter() {
            drafter.mark_seqs_dormant(seq_ids);
        }
        Ok(())
    }

    fn release_speculative_sequences(&mut self, seq_ids: &[usize]) -> Result<()> {
        if let Some(drafter) = self.spec.drafter() {
            drafter.release_seqs(seq_ids);
        }
        Ok(())
    }

    inference_drafters::external_drafter_prefix_hooks!(|this| this.spec.drafter());

    fn speculative_prefill(
        &mut self,
        ctx: crate::speculative::SpeculativePrefillCtx<'_>,
    ) -> Result<()> {
        match self.spec.drafter() {
            Some(drafter) => drafter.prefill(self, ctx),
            None => Ok(()),
        }
    }

    fn speculative_prepare_propose(
        &mut self,
        ctx: crate::speculative::SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn crate::speculative::SpeculativeProposePreparation>>> {
        match self.spec.drafter() {
            Some(drafter) => drafter.prepare_propose(self, ctx),
            None => Ok(None),
        }
    }

    fn speculative_propose(
        &mut self,
        ctx: crate::speculative::SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<crate::speculative::SpeculativeProposalBatch>> {
        match self.spec.drafter() {
            Some(drafter) => drafter.propose(self, ctx, drafter.n_predict(), &self.device),
            None => Ok(None),
        }
    }

    fn speculative_commit(&mut self, rows: &[SpeculativeCommitRow]) -> Result<()> {
        self.roll_back_conv_states(rows)?;
        match self.spec.drafter() {
            Some(drafter) => drafter.commit_terminal(self, rows),
            None => Ok(()),
        }
    }
}
