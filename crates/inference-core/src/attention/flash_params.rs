#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Device, DeviceLocation, Tensor};

use crate::device_map::DeviceMapper;

/// Flash attention sequence length metadata.
///
/// `cumulative_seqlens_q/k` describe the physical Q/K layout. They use padded
/// lengths for normal batches and logical lengths when `packed` is true.
///
/// `logical_k` describes full logical K lengths. `sliding_k`, when present,
/// describes the physical K lengths returned by a rotating/sliding KV cache.
///
/// For the **prefix cache path**, K/V are gathered from the paged cache into a
/// packed (non-padded) layout via `gather_kv_cache`. The packed K/V lengths are
/// given by `PagedAttentionInputMetadata::cu_seqlens_kv`, NOT by the normal
/// `logical_k/sliding_k` metadata here. The prefix cache attention call must
/// build a local `FlashParams` matching the gathered KV layout.
#[derive(Clone, Debug)]
pub struct FlashKMeta {
    pub max: u32,
    pub cumulative_seqlens: HashMap<DeviceLocation, Tensor>,
}

impl FlashKMeta {
    pub fn empty() -> Self {
        Self {
            max: 0,
            cumulative_seqlens: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlashParams {
    pub max_q: u32,
    pub cumulative_seqlens_q: HashMap<DeviceLocation, Tensor>,
    pub logical_k: FlashKMeta,
    pub sliding_k: Option<FlashKMeta>,
    pub causal: bool,
    pub(crate) packed: bool,
    #[cfg_attr(
        not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")),
        allow(dead_code)
    )]
    pub(crate) varlen_segment_lens: Option<Vec<usize>>,
}

impl FlashParams {
    pub fn empty(causal: bool) -> Self {
        Self {
            max_q: 0,
            cumulative_seqlens_q: HashMap::new(),
            logical_k: FlashKMeta::empty(),
            sliding_k: None,
            causal,
            packed: false,
            varlen_segment_lens: None,
        }
    }

    pub fn k_meta(&self, sliding_window: Option<usize>) -> &FlashKMeta {
        if sliding_window.is_some() {
            self.sliding_k.as_ref().unwrap_or(&self.logical_k)
        } else {
            &self.logical_k
        }
    }
}

pub(crate) fn flash_param_devices(
    device: &Device,
    mapper: Option<&dyn DeviceMapper>,
) -> Vec<Device> {
    mapper
        .map(|mapper| mapper.get_unique_devices())
        .unwrap_or_else(|| vec![device.clone()])
}

pub(crate) fn cumulative_seqlens_map(
    lengths: &[u32],
    devices: &[Device],
) -> Result<(u32, HashMap<DeviceLocation, Tensor>)> {
    let max = *lengths.iter().max().unwrap_or(&0);
    if devices.is_empty() {
        return Ok((max, HashMap::new()));
    }

    // Create tensors on CPU first to avoid CUDA context issues when copying
    // between different GPU devices. Each GPU has its own CUDA context, and
    // candle/cudarc doesn't properly switch contexts when doing GPU-to-GPU
    // transfers (which go through CPU). By creating on CPU first, we avoid
    // the cross-context memory access that causes CUDA_ERROR_INVALID_VALUE.
    let cumulative_seqlens = Tensor::new(lengths, &Device::Cpu)?
        .to_dtype(DType::F32)?
        .cumsum(0)?
        .to_dtype(DType::U32)?;

    let mut cumulative_seqlens_map = HashMap::new();
    for device in devices {
        cumulative_seqlens_map.insert(device.location(), cumulative_seqlens.to_device(device)?);
    }

    Ok((max, cumulative_seqlens_map))
}

pub(crate) fn packed_rope_positions(
    seqlen_offsets: &[usize],
    query_lens: &[usize],
) -> Result<Vec<u32>> {
    if seqlen_offsets.len() != query_lens.len() {
        anyhow::bail!(
            "packed RoPE position length mismatch: {} offsets for {} queries",
            seqlen_offsets.len(),
            query_lens.len()
        );
    }
    let mut positions = Vec::with_capacity(query_lens.iter().sum());
    for (&offset, &query_len) in seqlen_offsets.iter().zip(query_lens) {
        for position in offset..offset + query_len {
            positions.push(u32::try_from(position)?);
        }
    }
    Ok(positions)
}

pub(crate) fn sliding_k_lengths(
    seqlens_q: &[u32],
    seqlens_k: &[u32],
    sliding_window: usize,
) -> Result<Vec<u32>> {
    if seqlens_q.len() != seqlens_k.len() {
        anyhow::bail!(
            "sliding FlashAttention metadata length mismatch: q={} k={}",
            seqlens_q.len(),
            seqlens_k.len()
        );
    }
    let window = u32::try_from(sliding_window)?;
    seqlens_q
        .iter()
        .zip(seqlens_k)
        .map(|(&query_len, &logical_k_len)| {
            let past_len = logical_k_len.checked_sub(query_len).ok_or_else(|| {
                anyhow::anyhow!(
                    "sliding FlashAttention query length {query_len} exceeds K length {logical_k_len}"
                )
            })?;
            if query_len > 1 {
                past_len
                    .min(window)
                    .checked_add(query_len)
                    .ok_or_else(|| anyhow::anyhow!("sliding FlashAttention K length overflow"))
            } else {
                Ok(logical_k_len.min(window))
            }
        })
        .collect()
}

pub(crate) fn make_flash_params(
    device: &Device,
    mapper: Option<&dyn DeviceMapper>,
    seqlens_q: &[u32],
    seqlens_k: &[u32],
    sliding_window: Option<usize>,
    causal: bool,
    packed: bool,
) -> Result<FlashParams> {
    let devices = flash_param_devices(device, mapper);
    let (max_q, cumulative_seqlens_q) = cumulative_seqlens_map(seqlens_q, &devices)?;
    let (logical_max_k, logical_cumulative_seqlens_k) =
        cumulative_seqlens_map(seqlens_k, &devices)?;
    let logical_k = FlashKMeta {
        max: logical_max_k,
        cumulative_seqlens: logical_cumulative_seqlens_k,
    };
    let sliding_k = sliding_window
        .map(|window| -> Result<FlashKMeta> {
            let sliding_seqlens_k = sliding_k_lengths(seqlens_q, seqlens_k, window)?;
            let (sliding_max_k, sliding_cumulative_seqlens_k) =
                cumulative_seqlens_map(&sliding_seqlens_k, &devices)?;
            Ok(FlashKMeta {
                max: sliding_max_k,
                cumulative_seqlens: sliding_cumulative_seqlens_k,
            })
        })
        .transpose()?;

    Ok(FlashParams {
        max_q,
        cumulative_seqlens_q,
        logical_k,
        sliding_k,
        causal,
        packed,
        varlen_segment_lens: None,
    })
}
