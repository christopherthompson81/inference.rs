//! Paged decode through the layer (fattn) against FlashInfer decode on the same plan and cache.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use inference_paged_attn::{KvCacheScales, flashinfer_decode};

use super::PagedAttention;
use crate::attention::{AttentionMask, SdpaParams, sliding_window_left};
use crate::paged_attention::{
    _PAD_SLOT_ID, Fp8AttentionScales, block_aligned_sliding_window_start,
    block_table_rows::BlockTableSnapshot, input_metadata::DecodePagedRows,
};

const NUM_BLOCKS: usize = 96;
const N_HEAD: usize = 8;
const N_HEAD_KV: usize = 2;
// Multiplier and offset of a full-period LCG over block ids, so each sequence gets scattered, distinct blocks.
const BLOCK_SHUFFLE: usize = 29;
const BLOCK_OFFSET: usize = 7;
// Max abs difference from FlashInfer: about one ulp of an O(1) output, more for bf16 than f16
const F16_TOLERANCE: f32 = 2e-3;
const BF16_TOLERANCE: f32 = 1.6e-2;
const FP8_SCALES: KvCacheScales = KvCacheScales { k: 0.5, v: 0.25 };

struct Case {
    head_dim: usize,
    block_size: usize,
    cache_dtype: DType,
    full_lens: &'static [usize],
    // query rows per sequence: above 1, a multi-token decode (MTP verify) with one table row per query row
    query_len: usize,
    // the window the metadata is built for; a full layer of a windowed model attends with None and the full tables
    model_window: Option<usize>,
    layer_window: Option<usize>,
    softcap: Option<f32>,
    // whether fattn serves the call; otherwise the layer falls back to FlashInfer
    fattn: bool,
}

fn check(c: Case) -> Result<()> {
    crate::skip_without_cuda!();
    let dev = Device::new_cuda(0)?;
    let (d, bs) = (c.head_dim, c.block_size);
    let b = c.full_lens.len();
    let mut next = 0;
    let tables: Vec<Vec<usize>> = c
        .full_lens
        .iter()
        .map(|len| {
            (0..len.div_ceil(bs))
                .map(|_| {
                    next += 1;
                    (next * BLOCK_SHUFFLE + BLOCK_OFFSET) % NUM_BLOCKS
                })
                .collect()
        })
        .collect();
    assert!(
        next <= NUM_BLOCKS,
        "the case needs more than {NUM_BLOCKS} blocks"
    );
    let q = c.query_len;
    // each query row sees the rows up to its own position
    let full_context_lens: Vec<usize> = c
        .full_lens
        .iter()
        .flat_map(|&len| (len + 1 - q..=len).collect::<Vec<_>>())
        .collect();
    let context_lens: Vec<usize> = full_context_lens
        .iter()
        .map(|&len| match c.model_window {
            Some(w) => len - block_aligned_sliding_window_start(len, 1, w, bs),
            None => len,
        })
        .collect();
    let rows = Arc::new(DecodePagedRows {
        slot_mappings: vec![vec![_PAD_SLOT_ID; q]; b],
        block_tables: BlockTableSnapshot::from_owned_sequence_tables(tables, q),
        context_lens,
        full_context_lens,
        query_len: q,
        block_size: bs,
        use_standard_metadata: false,
        max_paged_context_len: NUM_BLOCKS * bs,
        sliding_window: c.model_window,
        decode_window: 1,
        devices: vec![dev.clone()],
        num_kv_heads: N_HEAD_KV,
    });
    let mut metadata = rows.build_materialized().map_err(candle_core::Error::msg)?;
    metadata.flashinfer = metadata.flashinfer.map(|m| m.track_decode_tile_plan());

    let fp8 = c.cache_dtype == DType::F8E4M3;
    let q_dtype = if fp8 { DType::BF16 } else { c.cache_dtype };
    let cache = || {
        Tensor::randn(0f32, 1., (NUM_BLOCKS, N_HEAD_KV, bs, d), &Device::Cpu)?
            .to_dtype(c.cache_dtype)?
            .to_device(&dev)
    };
    let (k_cache, v_cache) = (cache()?, cache()?);
    let query = Tensor::randn(0f32, 1., (b, N_HEAD, q, d), &dev)?.to_dtype(q_dtype)?;
    let sdpa = SdpaParams {
        n_kv_groups: N_HEAD / N_HEAD_KV,
        softcap: c.softcap,
        softmax_scale: 1. / f32::from(u16::try_from(d).unwrap()).sqrt(),
        sliding_window: c.layer_window,
        sinks: None,
    };
    let scales = fp8.then_some(Fp8AttentionScales {
        q: 1.,
        k: FP8_SCALES.k,
        v: FP8_SCALES.v,
    });
    let layer = PagedAttention::new_with_fp8_attention_scales(d, &dev, None, scales)?;
    let out = layer.forward_donor_cache(
        &query,
        &k_cache,
        &v_cache,
        &AttentionMask::None,
        &metadata,
        &sdpa,
        None,
    )?;
    let plan = metadata.flashinfer.as_ref().unwrap();
    assert_eq!(
        !plan.decode_tile_plan_was_used(),
        c.fattn,
        "which kernel served decode"
    );

    let fi = plan.decode_metadata(&dev.location(), c.layer_window)?;
    let expected = flashinfer_decode(
        &query.transpose(1, 2)?.reshape((b * q, N_HEAD, d))?,
        &k_cache,
        &v_cache,
        if fp8 {
            FP8_SCALES
        } else {
            KvCacheScales { k: 1., v: 1. }
        },
        fi.paged_kv_indptr,
        fi.paged_kv_indices,
        fi.paged_kv_last_page_len,
        fi.request_indices,
        fi.kv_tile_indices,
        fi.o_indptr,
        fi.kv_chunk_size,
        fi.block_valid_mask,
        sdpa.softmax_scale,
        sliding_window_left(c.layer_window),
        c.softcap,
        None,
    )?;
    let diff = (out.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let tolerance = if q_dtype == DType::BF16 {
        BF16_TOLERANCE
    } else {
        F16_TOLERANCE
    };
    assert!(diff <= tolerance, "max abs diff {diff} over {tolerance}");
    Ok(())
}

#[test]
fn decode_matches_flashinfer() -> Result<()> {
    for head_dim in [64, 128, 256, 512] {
        for cache_dtype in [DType::BF16, DType::F16, DType::F8E4M3] {
            check(Case {
                head_dim,
                block_size: 32,
                cache_dtype,
                full_lens: &[1, 37, 300, 64],
                query_len: 1,
                model_window: None,
                layer_window: None,
                softcap: None,
                fattn: true,
            })?;
        }
    }
    Ok(())
}

#[test]
fn sliding_window_decode_matches_flashinfer() -> Result<()> {
    for block_size in [16, 32] {
        for (layer_window, softcap) in [(Some(100), None), (None, None), (Some(100), Some(30.))] {
            check(Case {
                head_dim: 128,
                block_size,
                cache_dtype: DType::BF16,
                full_lens: &[5, 99, 101, 333],
                query_len: 1,
                model_window: Some(100),
                layer_window,
                softcap,
                fattn: true,
            })?;
        }
    }
    Ok(())
}

#[test]
fn softcap_without_a_kernel_falls_back_to_flashinfer() -> Result<()> {
    check(Case {
        head_dim: 64,
        block_size: 32,
        cache_dtype: DType::BF16,
        full_lens: &[17, 90],
        query_len: 1,
        model_window: None,
        layer_window: None,
        softcap: Some(30.),
        fattn: false,
    })
}

#[test]
fn multi_token_decode_matches_flashinfer() -> Result<()> {
    check(Case {
        head_dim: 256,
        block_size: 32,
        cache_dtype: DType::BF16,
        full_lens: &[40, 333, 64],
        query_len: 3,
        model_window: None,
        layer_window: None,
        softcap: None,
        fattn: true,
    })
}

#[test]
fn f32_caches_fall_back_to_flashinfer() -> Result<()> {
    check(Case {
        head_dim: 128,
        block_size: 32,
        cache_dtype: DType::F32,
        full_lens: &[17, 90],
        query_len: 1,
        model_window: None,
        layer_window: None,
        softcap: None,
        fattn: false,
    })
}
