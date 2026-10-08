use std::collections::HashMap;

use inference_quant::QuantMethod;
use inference_tensor::{Device, DeviceLocation, Tensor};

use crate::{
    attention::FlashParams,
    gdn::RecurrentBatchKind,
    kv_cache::KvCache,
    layers::masker::PastKvLenCache,
    paged_attention::{PagedAttentionInputMetadata, block_hash::MultimodalAttentionPolicy},
};

pub type DeviceTensorMap = HashMap<DeviceLocation, Tensor>;

pub fn metadata_rope_positions<'a>(
    metadata: &'a PagedAttentionInputMetadata,
    device: &Device,
) -> Option<&'a Tensor> {
    metadata
        .rope_positions
        .as_ref()
        .and_then(|positions| positions.get(&device.location()))
}

pub enum ForwardCache<'a> {
    Normal(&'a mut [KvCache]),
    Paged {
        kv_cache: &'a [(Tensor, Tensor)],
        metadata: &'a PagedAttentionInputMetadata,
    },
    None,
}

impl<'a> ForwardCache<'a> {
    pub fn from_paged(
        metadata: Option<(&'a [(Tensor, Tensor)], &'a PagedAttentionInputMetadata)>,
    ) -> Self {
        match metadata {
            Some((kv_cache, metadata)) => Self::Paged { kv_cache, metadata },
            None => Self::None,
        }
    }

    pub fn paged_metadata(&self) -> Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)> {
        match self {
            Self::Paged { kv_cache, metadata } => Some((kv_cache.to_vec(), metadata)),
            Self::Normal(_) | Self::None => None,
        }
    }

    pub fn paged_layer(
        &self,
        layer_idx: usize,
    ) -> Option<((Tensor, Tensor), &PagedAttentionInputMetadata)> {
        match self {
            Self::Paged { kv_cache, metadata } => Some((kv_cache[layer_idx].clone(), metadata)),
            Self::Normal(_) | Self::None => None,
        }
    }

    pub fn rope_positions(&self, device: &Device) -> Option<&Tensor> {
        match self {
            Self::Paged { metadata, .. } => metadata_rope_positions(metadata, device),
            Self::Normal(_) | Self::None => None,
        }
    }

    pub fn is_final_prompt_chunk(&self) -> bool {
        match self {
            Self::Paged { metadata, .. } => metadata.is_final_prompt_chunk,
            Self::Normal(_) | Self::None => true,
        }
    }

    pub fn is_first_prompt_chunk(&self) -> bool {
        match self {
            Self::Paged { metadata, .. } => metadata.is_first_prompt_chunk,
            Self::Normal(_) | Self::None => true,
        }
    }

    pub fn needs_logits(&self) -> bool {
        match self {
            Self::Paged { metadata, .. } => metadata.needs_logits,
            Self::Normal(_) | Self::None => true,
        }
    }
}

pub enum ForwardPositions<'a> {
    Text { seqlen_offsets: &'a [usize] },
    Mrope { position_ids: &'a Tensor },
    None,
}

struct RopeTables {
    computed: (Tensor, Tensor),
    on_device: HashMap<DeviceLocation, (Tensor, Tensor)>,
}

pub enum ForwardMaskCache<'a> {
    Normal(&'a [KvCache]),
    Paged(&'a [usize]),
}

pub fn recurrent_batch_kind_for_input(
    is_prompt: bool,
    has_staged_speculative_batch: bool,
) -> RecurrentBatchKind {
    if is_prompt {
        RecurrentBatchKind::Prefill
    } else if has_staged_speculative_batch {
        RecurrentBatchKind::SpeculativeDecode
    } else {
        RecurrentBatchKind::Decode
    }
}

#[derive(Clone, Debug)]
pub struct RecurrentMetadata {
    batch_kind: RecurrentBatchKind,
    state_indices: Tensor,
    state_indices_host: Option<Vec<u32>>,
}

impl RecurrentMetadata {
    pub fn new(
        batch_kind: RecurrentBatchKind,
        state_indices: Tensor,
        state_indices_host: Option<Vec<u32>>,
    ) -> Self {
        Self {
            batch_kind,
            state_indices,
            state_indices_host,
        }
    }

    pub fn batch_kind(&self) -> RecurrentBatchKind {
        self.batch_kind
    }

    pub fn state_indices(&self) -> &Tensor {
        &self.state_indices
    }

    pub fn state_indices_host(&self) -> Option<&[u32]> {
        self.state_indices_host.as_deref()
    }
}

impl PastKvLenCache for ForwardMaskCache<'_> {
    fn get_past_kv_len(&self) -> inference_tensor::Result<usize> {
        match self {
            Self::Normal(cache) => Ok(cache
                .iter()
                .map(KvCache::current_seq_len)
                .max()
                .unwrap_or(0)),
            Self::Paged(offsets) => offsets.get_past_kv_len(),
        }
    }
}

pub struct ModelForwardContext<'a> {
    cache: ForwardCache<'a>,
    positions: ForwardPositions<'a>,
    rope_positions: HashMap<(DeviceLocation, usize), Tensor>,
    rope_tables: Option<RopeTables>,
    context_lens: &'a [(usize, usize)],
    position_ids: &'a [usize],
    flash_params: &'a FlashParams,
    recurrent_metadata: Option<RecurrentMetadata>,
    recurrent_batch_kind: Option<RecurrentBatchKind>,
    requires_full_prefill_queries: bool,
}

impl<'a> ModelForwardContext<'a> {
    pub fn new(
        seqlen_offsets: &'a [usize],
        context_lens: &'a [(usize, usize)],
        position_ids: &'a [usize],
        metadata: Option<(&'a [(Tensor, Tensor)], &'a PagedAttentionInputMetadata)>,
        flash_params: &'a FlashParams,
    ) -> Self {
        Self {
            cache: ForwardCache::from_paged(metadata),
            positions: ForwardPositions::Text { seqlen_offsets },
            rope_positions: HashMap::new(),
            rope_tables: None,
            context_lens,
            position_ids,
            flash_params,
            recurrent_metadata: None,
            recurrent_batch_kind: None,
            requires_full_prefill_queries: false,
        }
    }

    pub fn with_cache(
        cache: ForwardCache<'a>,
        seqlen_offsets: &'a [usize],
        context_lens: &'a [(usize, usize)],
        position_ids: &'a [usize],
        flash_params: &'a FlashParams,
    ) -> Self {
        Self {
            cache,
            positions: ForwardPositions::Text { seqlen_offsets },
            rope_positions: HashMap::new(),
            rope_tables: None,
            context_lens,
            position_ids,
            flash_params,
            recurrent_metadata: None,
            recurrent_batch_kind: None,
            requires_full_prefill_queries: false,
        }
    }

    pub fn with_recurrent_batch_kind(mut self, recurrent_batch_kind: RecurrentBatchKind) -> Self {
        self.recurrent_batch_kind = Some(recurrent_batch_kind);
        self
    }

    /// The batch kind and, for a hybrid cache, its state indices for this forward.
    pub fn with_recurrent_cache(
        self,
        cache: &crate::kv_cache::EitherCache,
        batch_kind: RecurrentBatchKind,
    ) -> Self {
        self.with_recurrent_batch_kind(batch_kind)
            .with_recurrent_metadata(cache.recurrent_metadata(batch_kind))
    }

    pub fn with_recurrent_metadata(
        mut self,
        recurrent_metadata: Option<RecurrentMetadata>,
    ) -> Self {
        if let Some(metadata) = recurrent_metadata.as_ref() {
            self.recurrent_batch_kind = Some(metadata.batch_kind());
        }
        self.recurrent_metadata = recurrent_metadata;
        self
    }

    /// The (cos, sin) a model computed from its own positions, as M-RoPE does, for its layers to share.
    pub fn set_rope_tables(&mut self, cos: Tensor, sin: Tensor) {
        self.rope_tables = Some(RopeTables {
            computed: (cos, sin),
            on_device: HashMap::new(),
        });
    }

    /// The tables [`Self::set_rope_tables`] stored, moved to `device` once.
    pub fn rope_tables(&mut self, device: &Device) -> inference_tensor::Result<&(Tensor, Tensor)> {
        let Some(tables) = self.rope_tables.as_mut() else {
            inference_tensor::bail!("this forward set no RoPE tables");
        };
        if let std::collections::hash_map::Entry::Vacant(entry) =
            tables.on_device.entry(device.location())
        {
            let (cos, sin) = &tables.computed;
            entry.insert((cos.to_device(device)?, sin.to_device(device)?));
        }
        Ok(&tables.on_device[&device.location()])
    }

    pub fn require_full_prefill_queries(&mut self) {
        self.requires_full_prefill_queries = true;
    }

    pub fn requires_full_prefill_queries(&self) -> bool {
        self.requires_full_prefill_queries
    }

    pub fn cache(&self) -> &ForwardCache<'a> {
        &self.cache
    }

    pub fn is_paged(&self) -> bool {
        matches!(self.cache, ForwardCache::Paged { .. })
    }

    pub fn seqlen_offsets(&self) -> &[usize] {
        match self.positions {
            ForwardPositions::Text { seqlen_offsets } => seqlen_offsets,
            ForwardPositions::Mrope { .. } | ForwardPositions::None => &[],
        }
    }

    pub fn context_lens(&self) -> &[(usize, usize)] {
        self.context_lens
    }

    pub fn position_ids(&self) -> &[usize] {
        self.position_ids
    }

    pub fn position_ids_vec(&self) -> Vec<usize> {
        self.position_ids.to_vec()
    }

    pub fn flash_params(&self) -> &FlashParams {
        self.flash_params
    }

    pub fn recurrent_metadata(&self) -> Option<&RecurrentMetadata> {
        self.recurrent_metadata.as_ref()
    }

    pub fn recurrent_batch_kind(&self) -> Option<RecurrentBatchKind> {
        self.recurrent_batch_kind
    }

    pub fn prompt_chunk_attention_policy(&self) -> MultimodalAttentionPolicy {
        self.paged_input_metadata()
            .map_or(MultimodalAttentionPolicy::Causal, |metadata| {
                metadata.prompt_chunk_attention_policy
            })
    }

    pub fn paged_metadata(&self) -> Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)> {
        self.cache.paged_metadata()
    }

    pub fn paged_input_metadata(&self) -> Option<&PagedAttentionInputMetadata> {
        match &self.cache {
            ForwardCache::Paged { metadata, .. } => Some(*metadata),
            ForwardCache::Normal(_) | ForwardCache::None => None,
        }
    }

    pub fn paged_layer(
        &self,
        layer_idx: usize,
    ) -> Option<((Tensor, Tensor), &PagedAttentionInputMetadata)> {
        self.cache.paged_layer(layer_idx)
    }

    pub fn text_positions(
        &mut self,
        device: &Device,
        seq_len: usize,
    ) -> inference_tensor::Result<Option<&Tensor>> {
        if self.flash_params.packed {
            let positions = self.cache.rope_positions(device).ok_or_else(|| {
                inference_tensor::Error::msg("packed prefill is missing RoPE positions")
            })?;
            return Ok(Some(positions));
        }
        if self.cache.rope_positions(device).is_some() {
            return Ok(self.cache.rope_positions(device));
        }
        let ForwardPositions::Text { seqlen_offsets } = self.positions else {
            return Ok(None);
        };
        let location = device.location();
        let key = (location, seq_len);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.rope_positions.entry(key) {
            entry.insert(text_positions_tensor(seqlen_offsets, seq_len, device)?);
        }
        Ok(self.rope_positions.get(&key))
    }

    pub fn text_positions_from_offsets(
        &self,
        seqlen_offsets: &[usize],
        seq_len: usize,
        device: &Device,
    ) -> inference_tensor::Result<Tensor> {
        text_positions_tensor(seqlen_offsets, seq_len, device)
    }

    pub fn is_first_prompt_chunk(&self) -> bool {
        self.cache.is_first_prompt_chunk()
    }

    pub fn is_final_prompt_chunk(&self) -> bool {
        self.cache.is_final_prompt_chunk()
    }

    pub fn needs_logits(&self) -> bool {
        self.cache.needs_logits()
    }

    /// Runs `head` on the selected rows, or returns a rank-preserving placeholder when the pipeline
    /// discards this forward's logits.
    pub fn lm_head(&self, head: &dyn QuantMethod, xs: &Tensor) -> inference_tensor::Result<Tensor> {
        if self.needs_logits() {
            return head.forward(xs);
        }
        let mut dims = xs.dims().to_vec();
        *dims
            .last_mut()
            .expect("selected hidden states have a feature axis") = 1;
        Tensor::zeros(dims, xs.dtype(), xs.device())
    }

    pub fn mask_cache<'b>(&'b self, normal_cache: &'b [KvCache]) -> ForwardMaskCache<'b> {
        match self.cache {
            ForwardCache::Paged { .. } => ForwardMaskCache::Paged(self.seqlen_offsets()),
            ForwardCache::Normal(_) | ForwardCache::None => ForwardMaskCache::Normal(normal_cache),
        }
    }

    pub fn logits(&self, logits: &Tensor) -> inference_tensor::Result<Tensor> {
        let devices = [logits.device().clone()];
        let selection = if self.flash_params.packed {
            let query_lens = self
                .paged_input_metadata()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    inference_tensor::Error::msg("packed prefill requires logical query lengths")
                })?;
            LogitsSelection::from_packed_context_lens(
                logits,
                self.context_lens,
                query_lens,
                &devices,
            )?
        } else {
            LogitsSelection::from_context_lens(logits, self.context_lens, &devices)?
        };
        selection.select(logits)
    }
}

pub fn text_positions_tensor(
    seqlen_offsets: &[usize],
    seq_len: usize,
    device: &Device,
) -> inference_tensor::Result<Tensor> {
    let mut positions = Vec::with_capacity(seqlen_offsets.len() * seq_len);
    for offset in seqlen_offsets {
        for seq_idx in 0..seq_len {
            positions.push(u32::try_from(offset + seq_idx).map_err(inference_tensor::Error::wrap)?);
        }
    }
    Tensor::from_vec(positions, (seqlen_offsets.len() * seq_len,), device)
}

pub fn decode_positions_tensor(
    position_ids: &[usize],
    seq_len: usize,
    device: &Device,
) -> inference_tensor::Result<Tensor> {
    let mut positions = Vec::with_capacity(position_ids.len() * seq_len);
    for end in position_ids {
        let start = end.checked_sub(seq_len).ok_or_else(|| {
            inference_tensor::Error::msg(format!(
                "decode position end {end} is smaller than query length {seq_len}"
            ))
        })?;
        for position in start..*end {
            positions.push(u32::try_from(position).map_err(inference_tensor::Error::wrap)?);
        }
    }
    Tensor::from_vec(positions, (position_ids.len() * seq_len,), device)
}

#[derive(Clone, Debug)]
pub enum LogitsSelection {
    Decode {
        start: usize,
        len: usize,
    },
    Indices {
        indices: DeviceTensorMap,
        batch: usize,
        len: usize,
    },
    PackedIndices {
        indices: DeviceTensorMap,
        batch: usize,
        len: usize,
    },
    All,
}

impl LogitsSelection {
    pub fn from_context_lens(
        source: &Tensor,
        context_lens: &[(usize, usize)],
        devices: &[Device],
    ) -> inference_tensor::Result<Self> {
        let dims = source.dims();
        if dims.len() < 2 {
            inference_tensor::bail!("logits selection source must have rank >= 2");
        }
        let batch = dims[0];
        let seq_len = dims[1];
        if context_lens.len() != batch {
            inference_tensor::bail!(
                "logits selection batch mismatch: {} spans for batch {batch}",
                context_lens.len()
            );
        }
        let Some((first_start, first_len)) = context_lens.first().copied() else {
            inference_tensor::bail!("logits selection requires at least one span");
        };
        for (start, len) in context_lens.iter().copied() {
            let end = start
                .checked_add(len)
                .ok_or_else(|| inference_tensor::Error::msg("logits selection span overflow"))?;
            if end > seq_len {
                inference_tensor::bail!(
                    "logits selection span ({start}, {len}) exceeds sequence length {seq_len}"
                );
            }
        }
        if context_lens.iter().all(|span| *span == (0, seq_len)) {
            return Ok(Self::All);
        }
        if context_lens
            .iter()
            .all(|span| *span == (first_start, first_len))
        {
            return Ok(Self::Decode {
                start: first_start,
                len: first_len,
            });
        }

        if context_lens.iter().any(|(_, len)| *len != first_len) {
            inference_tensor::bail!("ragged logits selection spans are not supported");
        }

        let mut flat_indices = Vec::with_capacity(batch * first_len);
        for (batch_idx, (start, len)) in context_lens.iter().copied().enumerate() {
            let end = start + len;
            for pos in start..end {
                let idx = batch_idx
                    .checked_mul(seq_len)
                    .and_then(|idx| idx.checked_add(pos))
                    .ok_or_else(|| {
                        inference_tensor::Error::msg("logits selection index overflow")
                    })?;
                flat_indices.push(u32::try_from(idx).map_err(inference_tensor::Error::wrap)?);
            }
        }

        let cpu_indices = Tensor::from_vec(flat_indices, (batch * first_len,), &Device::Cpu)?;
        let mut indices = HashMap::new();
        for device in devices {
            indices.insert(device.location(), cpu_indices.to_device(device)?);
        }
        Ok(Self::Indices {
            indices,
            batch,
            len: first_len,
        })
    }

    pub fn from_packed_context_lens(
        source: &Tensor,
        context_lens: &[(usize, usize)],
        query_lens: &[usize],
        devices: &[Device],
    ) -> inference_tensor::Result<Self> {
        let (physical_batch, physical_seq_len, _) = source.dims3()?;
        if context_lens.len() != query_lens.len() {
            inference_tensor::bail!(
                "packed logits selection length mismatch: {} spans for {} queries",
                context_lens.len(),
                query_lens.len()
            );
        }
        let total_tokens = query_lens.iter().sum::<usize>();
        if physical_batch * physical_seq_len != total_tokens {
            inference_tensor::bail!(
                "packed logits selection token mismatch: source has {} rows, queries have {total_tokens}",
                physical_batch * physical_seq_len
            );
        }
        let Some((_, output_len)) = context_lens.first().copied() else {
            inference_tensor::bail!("packed logits selection requires at least one span");
        };
        if context_lens.iter().any(|(_, len)| *len != output_len) {
            inference_tensor::bail!("ragged packed logits selection spans are not supported");
        }

        let mut indices = Vec::with_capacity(context_lens.len() * output_len);
        let mut base = 0usize;
        for ((start, len), query_len) in context_lens.iter().copied().zip(query_lens) {
            let end = start.checked_add(len).ok_or_else(|| {
                inference_tensor::Error::msg("packed logits selection span overflow")
            })?;
            if end > *query_len {
                inference_tensor::bail!(
                    "packed logits selection span ({start}, {len}) exceeds query length {query_len}"
                );
            }
            for position in start..end {
                indices
                    .push(u32::try_from(base + position).map_err(inference_tensor::Error::wrap)?);
            }
            base += query_len;
        }

        let batch = query_lens.len();
        let cpu_indices = Tensor::from_vec(indices, (batch * output_len,), &Device::Cpu)?;
        let mut device_indices = HashMap::new();
        for device in devices {
            device_indices.insert(device.location(), cpu_indices.to_device(device)?);
        }
        Ok(Self::PackedIndices {
            indices: device_indices,
            batch,
            len: output_len,
        })
    }

    pub fn select(&self, logits: &Tensor) -> inference_tensor::Result<Tensor> {
        match self {
            Self::All => Ok(logits.clone()),
            Self::Decode { start, len } => {
                let seq_len = logits.dim(1)?;
                if *start == 0 && *len == seq_len {
                    Ok(logits.clone())
                } else {
                    logits.narrow(1, *start, *len)
                }
            }
            Self::Indices {
                indices,
                batch,
                len,
            } => {
                let (logits_batch, seq_len, hidden) = logits.dims3()?;
                if logits_batch != *batch {
                    inference_tensor::bail!(
                        "logits selection batch mismatch: logits batch {logits_batch}, selection batch {batch}"
                    );
                }
                let indices = indices.get(&logits.device().location()).ok_or_else(|| {
                    inference_tensor::Error::msg("missing logits selection indices")
                })?;
                let flat = logits.reshape((logits_batch * seq_len, hidden))?;
                flat.index_select(indices, 0)?
                    .reshape((*batch, *len, hidden))
            }
            Self::PackedIndices {
                indices,
                batch,
                len,
            } => {
                let (physical_batch, physical_seq_len, hidden) = logits.dims3()?;
                let indices = indices.get(&logits.device().location()).ok_or_else(|| {
                    inference_tensor::Error::msg("missing logits selection indices")
                })?;
                logits
                    .reshape((physical_batch * physical_seq_len, hidden))?
                    .index_select(indices, 0)?
                    .reshape((*batch, *len, hidden))
            }
        }
    }
}

pub fn extract_logits(
    logits: &Tensor,
    context_lens: Vec<(usize, usize)>,
) -> inference_tensor::Result<Tensor> {
    LogitsSelection::from_context_lens(logits, &context_lens, &[logits.device().clone()])?
        .select(logits)
}

#[cfg(test)]
mod tests {
    use inference_tensor::{Device, Result, Tensor};

    use super::*;

    #[test]
    fn ordinary_batch_expands_each_sequence_offset() -> Result<()> {
        let offsets = [4, 12];
        let context_lens = [(0, 1), (0, 1)];
        let position_ids = [3, 3];
        let flash_params = FlashParams::empty(true);
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash_params);

        assert_eq!(
            ctx.text_positions(&Device::Cpu, 3)?
                .unwrap()
                .to_vec1::<u32>()?,
            vec![4, 5, 6, 12, 13, 14]
        );
        Ok(())
    }

    #[test]
    fn packed_batch_uses_ragged_token_positions() -> Result<()> {
        let offsets = [4, 12];
        let context_lens = [(2, 1), (1, 1)];
        let position_ids = [3, 2];
        let packed_positions = Tensor::new(&[4u32, 5, 6, 12, 13], &Device::Cpu)?;
        let mut metadata = PagedAttentionInputMetadata::dummy(&Device::Cpu)?;
        metadata.rope_positions = Some(HashMap::from([(Device::Cpu.location(), packed_positions)]));
        let mut flash_params = FlashParams::empty(true);
        flash_params.packed = true;
        let kv_cache = Vec::new();
        let mut ctx = ModelForwardContext::new(
            &offsets,
            &context_lens,
            &position_ids,
            Some((kv_cache.as_slice(), &metadata)),
            &flash_params,
        );

        assert_eq!(
            ctx.text_positions(&Device::Cpu, 5)?
                .unwrap()
                .to_vec1::<u32>()?,
            vec![4, 5, 6, 12, 13]
        );
        Ok(())
    }
}
