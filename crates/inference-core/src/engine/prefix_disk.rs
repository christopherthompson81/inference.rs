use inference_tensor::{Result, Tensor};

use super::Engine;
use crate::{
    get_mut_arcmutex, paged_attention::block_hash::compute_block_hashes, pipeline::Pipeline,
    prefix_store::LoadedPrefix, sequence::Sequence,
};

impl Engine {
    /// Rebuilds a stored prefix in the GPU cache when the disk holds a longer one, so the scheduler's ordinary prefix
    /// match then finds it. A failure only costs the hit.
    pub(super) fn promote_stored_prefix(&self, seq: &Sequence) {
        if let Err(error) = self.try_promote_stored_prefix(seq) {
            tracing::warn!("Failed restoring a prefix from the disk store: {error}");
        }
    }

    fn try_promote_stored_prefix(&self, seq: &Sequence) -> Result<()> {
        // only plain text prompts are written through
        if seq.return_raw_logits
            || !seq.mm_features().is_empty()
            || seq.adapter_generation().is_some()
        {
            return Ok(());
        }
        let Some(disk) = get_mut_arcmutex!(self.prefix_cacher).disk_tier().cloned() else {
            return Ok(());
        };
        let Some(block_size) = get_mut_arcmutex!(self.scheduler).block_size() else {
            return Ok(());
        };
        let tokens = seq.get_toks();
        let hashes = compute_block_hashes(tokens, block_size, &[], &[]);
        let Some(&owner) = hashes.last() else {
            return Ok(());
        };
        // a hit leaves the last token to compute, as the scheduler's match does
        let usable = (tokens.len().saturating_sub(1) / block_size).min(hashes.len());
        let on_gpu = get_mut_arcmutex!(self.prefix_cacher)
            .peek_longest_paged_recurrent_prefix(&hashes, usable)
            .map_or(0, |(n_blocks, _)| n_blocks);
        let values = hashes[..usable]
            .iter()
            .map(|hash| hash.value())
            .collect::<Vec<_>>();
        let Some(n_blocks) = disk.store.longest_entry(&values, on_gpu) else {
            return Ok(());
        };
        let started = std::time::Instant::now();
        let Some(loaded) = disk.store.load(&values[..n_blocks], tokens, block_size)? else {
            return Ok(());
        };
        let key = &hashes[..n_blocks];
        let Some(imported) = get_mut_arcmutex!(disk.kv_cache_manager).allocate_imported_blocks(key)
        else {
            return Ok(());
        };
        let restored =
            self.fill_imported_prefix(crate::scheduler::modality_signature(seq), &imported, loaded);
        if let Ok(Some((snapshot, auxiliary))) = &restored {
            get_mut_arcmutex!(disk.kv_cache_manager).cache_imported_blocks(&imported, key);
            get_mut_arcmutex!(self.prefix_cacher).add_paged_recurrent_prefix(
                owner,
                key.to_vec(),
                *snapshot,
                auxiliary.clone(),
            );
        }
        get_mut_arcmutex!(disk.kv_cache_manager).release_imported_blocks(&imported);
        if restored?.is_some() {
            tracing::info!(
                blocks = n_blocks,
                uploaded = imported.len(),
                seconds = started.elapsed().as_secs_f64(),
                "Restored a prefix from the disk store"
            );
            metrics::counter!("inference_prefix_store_restored_blocks_total")
                .increment(n_blocks as u64);
        }
        Ok(())
    }

    // Writes the imported blocks' KV and rebuilds the recurrent snapshot and drafter checkpoint; `None` when the
    // recurrent pool has no slot to spare or the drafter cannot take its checkpoint
    #[allow(clippy::type_complexity)]
    fn fill_imported_prefix(
        &self,
        modality: u8,
        imported: &[(usize, usize)],
        loaded: LoadedPrefix,
    ) -> Result<
        Option<(
            crate::kv_cache::RecurrentSnapshotSlot,
            Option<std::sync::Arc<dyn crate::kv_cache::PagedAuxiliaryPrefixState>>,
        )>,
    > {
        let mut pipeline = get_mut_arcmutex!(self.pipeline);
        write_imported_blocks(&*pipeline, imported, &loaded)?;
        let auxiliary = match &loaded.auxiliary {
            Some(data) => match pipeline.import_paged_auxiliary_prefix_state(data)? {
                Some(state) => Some(state),
                None => return Ok(None),
            },
            None if pipeline
                .speculative_prefix_checkpoint_policy()
                .uses_auxiliary_state(modality) =>
            {
                return Ok(None);
            }
            None => None,
        };
        let Some(snapshot) = pipeline
            .cache()
            .hybrid()
            .import_recurrent_snapshot(&loaded.recurrent)?
        else {
            return Ok(None);
        };
        Ok(Some((snapshot, auxiliary)))
    }
}

// Each attention layer's rows for the imported blocks, uploaded once and written in runs of consecutive block ids
fn write_imported_blocks(
    pipeline: &dyn Pipeline,
    imported: &[(usize, usize)],
    loaded: &LoadedPrefix,
) -> Result<()> {
    if imported.is_empty() {
        return Ok(());
    }
    let metadata = pipeline.get_metadata();
    let cache_engine = metadata
        .cache_engine
        .as_ref()
        .ok_or_else(|| inference_tensor::Error::msg("prefix import without a cache engine"))?;
    let kv = cache_engine.get_kv_cache().clone();
    let mut order = imported.to_vec();
    order.sort_by_key(|&(_, id)| id);
    let paged_layers = kv
        .iter()
        .enumerate()
        .filter(|(_, (k, _))| k.dim(0).is_ok_and(|blocks| blocks > 0))
        .map(|(layer, _)| layer)
        .collect::<Vec<_>>();
    for (slot, &layer) in paged_layers.iter().enumerate() {
        let (k_cache, v_cache) = &kv[layer];
        let rows = |pick: fn(&(usize, Tensor, Tensor)) -> &Tensor| -> Result<Tensor> {
            let parts = order
                .iter()
                .map(|&(i, _)| {
                    let stored = loaded.blocks[i]
                        .layers
                        .get(slot)
                        .filter(|(l, _, _)| *l == layer);
                    stored.map(pick).cloned().ok_or_else(|| {
                        inference_tensor::Error::msg(format!(
                            "stored block lacks attention layer {layer}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&parts, 0)?.to_device(k_cache.device())
        };
        let (k_rows, v_rows) = (rows(|(_, k, _)| k)?, rows(|(_, _, v)| v)?);
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && order[end].1 == order[end - 1].1 + 1 {
                end += 1;
            }
            let first = order[start].1;
            k_cache.slice_set(&k_rows.narrow(0, start, end - start)?, 0, first)?;
            v_cache.slice_set(&v_rows.narrow(0, start, end - start)?, 0, first)?;
            start = end;
        }
    }
    Ok(())
}
