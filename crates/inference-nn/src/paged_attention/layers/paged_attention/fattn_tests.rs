//! The layer's fattn paths, decode and prefix prefill, and decode's gather fallback against a reference by hand.

use std::sync::Arc;

use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::{DType, Device, Result, Tensor};
use inference_paged_attn::KvCacheScales;

use super::{FattnPrefillCall, PagedAttention, PagedForwardCtx, PagedForwardDims};
use crate::attention::{AttentionMask, SdpaParams};
use crate::paged_attention::{
    _PAD_SLOT_ID, Fp8AttentionScales, PagedAttentionInputMetadata,
    block_aligned_sliding_window_start, block_table_rows::BlockTableSnapshot,
    input_metadata::DecodePagedRows,
};

const NUM_BLOCKS: usize = 96;
const N_HEAD: usize = 8;
const N_HEAD_KV: usize = 2;
// Multiplier and offset of a full-period LCG over block ids, so each sequence gets scattered, distinct blocks.
const BLOCK_SHUFFLE: usize = 29;
const BLOCK_OFFSET: usize = 7;
// Max abs difference from the references: one or two ulps of an O(1) output, more for bf16 than f16
const F16_TOLERANCE: f32 = 4e-3;
const BF16_TOLERANCE: f32 = 1.6e-2;
// the gather's eager attention runs softcap's tanh and the softmax in bf16
const BF16_GATHER_TOLERANCE: f32 = 4e-2;
const FP8_SCALES: KvCacheScales = KvCacheScales { k: 0.5, v: 0.25 };
const PREFILL_BLOCK_SIZE: usize = 16;

struct Case {
    head_dim: usize,
    // (q heads, kv heads)
    heads: (usize, usize),
    block_size: usize,
    cache_dtype: DType,
    full_lens: &'static [usize],
    // query rows per sequence: above 1, a multi-token decode (MTP verify) with one table row per query row
    query_len: usize,
    // the window the metadata is built for; a full layer of a windowed model attends with None and the full tables
    model_window: Option<usize>,
    layer_window: Option<usize>,
    softcap: Option<f32>,
    // fattn refuses the call, so the layer gathers (its output keeps a query axis: rank 4 against fattn's 3)
    gather: bool,
}

fn check(c: Case) -> Result<()> {
    check_with(c, false)
}

fn check_with(c: Case, capture: bool) -> Result<()> {
    crate::skip_without_cuda!();
    let dev = Device::new_cuda(0)?;
    let (n_head, n_head_kv) = c.heads;
    let (d, bs) = (c.head_dim, c.block_size);
    let b = c.full_lens.len();
    let mut next = 0;
    let full_tables: Vec<Vec<usize>> = c
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
        block_tables: BlockTableSnapshot::from_owned_sequence_tables(full_tables.clone(), q),
        context_lens,
        full_context_lens,
        query_len: q,
        block_size: bs,
        use_standard_metadata: false,
        max_paged_context_len: NUM_BLOCKS * bs,
        sliding_window: c.model_window,
        decode_window: 1,
        devices: vec![dev.clone()],
        num_kv_heads: n_head_kv,
    });
    let mut metadata = rows.build_materialized().map_err(candle_core::Error::msg)?;
    metadata.flashinfer = metadata.flashinfer.map(|m| m.track_decode_tile_plan());

    let fp8 = c.cache_dtype == DType::F8E4M3;
    let q_dtype = if fp8 { DType::BF16 } else { c.cache_dtype };
    let cache = || {
        Tensor::randn(0f32, 1., (NUM_BLOCKS, n_head_kv, bs, d), &Device::Cpu)?
            .to_dtype(c.cache_dtype)?
            .to_device(&dev)
    };
    let (k_cache, v_cache) = (cache()?, cache()?);
    let query = Tensor::randn(0f32, 1., (b, n_head, q, d), &dev)?.to_dtype(q_dtype)?;
    let sdpa = SdpaParams {
        n_kv_groups: n_head / n_head_kv,
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
    if capture {
        // a gather inside a capture would replay these lengths forever after
        let stream = dev.as_cuda_device()?.cuda_stream();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let captured = layer.forward_donor_cache(
            &query,
            &k_cache,
            &v_cache,
            &AttentionMask::None,
            &metadata,
            &sdpa,
            None,
        );
        crate::cuda::graph_capture::end_cuda_capture_discard(&stream);
        assert!(captured.is_err(), "the gather ran inside a graph capture");
        return Ok(());
    }
    let out = layer.forward_donor_cache(
        &query,
        &k_cache,
        &v_cache,
        &AttentionMask::None,
        &metadata,
        &sdpa,
        None,
    )?;
    assert_eq!(out.rank() == 4, c.gather, "which path served decode");
    // every query row against its sequence's rows from the full tables, the window applied by position
    let packed = query
        .permute((1, 0, 2, 3))?
        .reshape((1, n_head, b * q, d))?;
    let expected = attention_reference(
        &Reference {
            head_dim: d,
            block_size: bs,
            kv_lens: c.full_lens,
            query_lens: &vec![q; b],
            causal: true,
            window: c.layer_window,
            softcap: c.softcap,
            heads: c.heads,
        },
        &packed,
        (&k_cache, &v_cache),
        &full_tables,
        if fp8 {
            FP8_SCALES
        } else {
            KvCacheScales { k: 1., v: 1. }
        },
    )?
    .transpose(0, 1)?
    .to_device(&dev)?;
    // the gather returns (rows, heads, 1, d) where fattn returns (rows, heads, d); the order is the same
    let diff = (out.to_dtype(DType::F32)?.flatten_all()?
        - expected.to_dtype(DType::F32)?.flatten_all()?)?
    .abs()?
    .max(0)?
    .to_scalar::<f32>()?;
    let tolerance = match (q_dtype, c.gather) {
        (DType::BF16, true) => BF16_GATHER_TOLERANCE,
        (DType::BF16, false) => BF16_TOLERANCE,
        _ => F16_TOLERANCE,
    };
    assert!(diff <= tolerance, "max abs diff {diff} over {tolerance}");
    Ok(())
}

#[test]
fn decode_matches_a_reference() -> Result<()> {
    // every head dim the HND layout admits
    for head_dim in [64, 80, 96, 112, 128, 256, 512] {
        for cache_dtype in [DType::BF16, DType::F16, DType::F8E4M3] {
            check(Case {
                head_dim,
                heads: (N_HEAD, N_HEAD_KV),
                block_size: 32,
                cache_dtype,
                full_lens: &[1, 37, 300, 64],
                query_len: 1,
                model_window: None,
                layer_window: None,
                softcap: None,
                gather: false,
            })?;
        }
    }
    Ok(())
}

#[test]
fn sliding_window_decode_matches_a_reference() -> Result<()> {
    for block_size in [16, 32] {
        for (layer_window, softcap) in [(Some(100), None), (None, None), (Some(100), Some(30.))] {
            check(Case {
                head_dim: 128,
                heads: (N_HEAD, N_HEAD_KV),
                block_size,
                cache_dtype: DType::BF16,
                full_lens: &[5, 99, 101, 333],
                query_len: 1,
                model_window: Some(100),
                layer_window,
                softcap,
                gather: false,
            })?;
        }
    }
    Ok(())
}

#[test]
fn softcap_without_a_fattn_kernel_falls_back_to_the_gather() -> Result<()> {
    check(Case {
        head_dim: 64,
        heads: (N_HEAD, N_HEAD_KV),
        block_size: 32,
        cache_dtype: DType::BF16,
        full_lens: &[17, 90],
        query_len: 1,
        model_window: None,
        layer_window: None,
        softcap: Some(30.),
        gather: true,
    })
}

#[test]
fn decode_with_any_gqa_group() -> Result<()> {
    // groups that are not a multiple of the kernel's head tile, and MQA's one KV head
    for heads in [(6, 2), (18, 2), (71, 1)] {
        check(Case {
            head_dim: 128,
            heads,
            block_size: 32,
            cache_dtype: DType::BF16,
            full_lens: &[1, 37, 300],
            query_len: 1,
            model_window: None,
            layer_window: None,
            softcap: None,
            gather: false,
        })?;
    }
    Ok(())
}

#[test]
fn gather_decode_refuses_graph_capture() -> Result<()> {
    check_with(
        Case {
            head_dim: 64,
            heads: (N_HEAD, N_HEAD_KV),
            block_size: 32,
            cache_dtype: DType::BF16,
            full_lens: &[17, 90],
            query_len: 1,
            model_window: None,
            layer_window: None,
            softcap: Some(30.),
            gather: true,
        },
        true,
    )
}

#[test]
fn multi_token_decode_matches_a_reference() -> Result<()> {
    check(Case {
        head_dim: 256,
        heads: (N_HEAD, N_HEAD_KV),
        block_size: 32,
        cache_dtype: DType::BF16,
        full_lens: &[40, 333, 64],
        query_len: 3,
        model_window: None,
        layer_window: None,
        softcap: None,
        gather: false,
    })
}

#[test]
fn f32_caches_fall_back_to_the_gather() -> Result<()> {
    check(Case {
        head_dim: 128,
        heads: (N_HEAD, N_HEAD_KV),
        block_size: 32,
        cache_dtype: DType::F32,
        full_lens: &[17, 90],
        query_len: 1,
        model_window: None,
        layer_window: None,
        softcap: None,
        gather: true,
    })
}

struct PrefillCase {
    head_dim: usize,
    cache_dtype: DType,
    // each sequence's rows in the cache, cached prefix and new queries alike
    kv_lens: &'static [usize],
    query_lens: &'static [usize],
    causal: bool,
    window: Option<usize>,
}

// The shape of a reference attention: each sequence's queries are the last of its rows.
struct Reference<'a> {
    head_dim: usize,
    // (q heads, kv heads)
    heads: (usize, usize),
    block_size: usize,
    kv_lens: &'a [usize],
    query_lens: &'a [usize],
    causal: bool,
    window: Option<usize>,
    softcap: Option<f32>,
}

// Each sequence's queries against its rows gathered from the cache by hand, one head at a time in f32; the packed
// `(1, heads, total, d)` query gives `(heads, total, d)`.
fn attention_reference(
    c: &Reference<'_>,
    query: &Tensor,
    caches: (&Tensor, &Tensor),
    tables: &[Vec<usize>],
    scales: KvCacheScales,
) -> Result<Tensor> {
    let bs = c.block_size;
    let rows = |cache: &Tensor, s: usize, scale: f32| -> Result<Tensor> {
        let cache = (cache.to_device(&Device::Cpu)?.to_dtype(DType::F32)? * f64::from(scale))?;
        let picked = (0..c.kv_lens[s])
            .map(|p| cache.get(tables[s][p / bs])?.narrow(1, p % bs, 1))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&picked, 1)
    };
    let scale = 1. / f64::from(u16::try_from(c.head_dim).unwrap()).sqrt();
    let query = query.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let mut q_start = 0;
    let mut outs = Vec::new();
    for (s, (&kv_len, &q_len)) in c.kv_lens.iter().zip(c.query_lens).enumerate() {
        let (k, v) = (rows(caches.0, s, scales.k)?, rows(caches.1, s, scales.v)?);
        let mask: Vec<f32> = (0..q_len)
            .flat_map(|j| {
                let qp = kv_len - q_len + j;
                (0..kv_len).map(move |kp| {
                    let too_old = kp <= qp && c.window.is_some_and(|w| qp - kp >= w);
                    let hidden = c.causal && kp > qp || too_old;
                    if hidden { f32::NEG_INFINITY } else { 0. }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (q_len, kv_len), &Device::Cpu)?;
        let (n_head, n_head_kv) = c.heads;
        let heads = (0..n_head)
            .map(|h| {
                let q = query.get(0)?.get(h)?.narrow(0, q_start, q_len)?;
                let kh = h / (n_head / n_head_kv);
                let (k, v) = (k.get(kh)?, v.get(kh)?);
                let mut att = (q.matmul(&k.t()?)? * scale)?;
                if let Some(cap) = c.softcap {
                    att = ((att / f64::from(cap))?.tanh()? * f64::from(cap))?;
                }
                candle_nn::ops::softmax_last_dim(&(att + &mask)?)?.matmul(&v)
            })
            .collect::<Result<Vec<_>>>()?;
        outs.push(Tensor::stack(&heads, 0)?);
        q_start += q_len;
    }
    Tensor::cat(&outs, 1)
}

fn check_prefill(c: PrefillCase) -> Result<()> {
    crate::skip_without_cuda!();
    let dev = Device::new_cuda(0)?;
    let (d, bs, nseq) = (c.head_dim, PREFILL_BLOCK_SIZE, c.kv_lens.len());
    let mut next = 0;
    let tables: Vec<Vec<usize>> = c
        .kv_lens
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
    let max_blocks = tables.iter().map(Vec::len).max().unwrap();
    let flat: Vec<u32> = tables
        .iter()
        .flat_map(|t| (0..max_blocks).map(|j| t.get(j).map_or(0, |&b| u32::try_from(b).unwrap())))
        .collect();
    let block_tables = Tensor::from_vec(flat, (nseq, max_blocks), &dev)?;
    let mut cu = vec![0u32];
    for &len in c.kv_lens {
        cu.push(cu.last().unwrap() + u32::try_from(len).unwrap());
    }
    let cu_kv = Tensor::from_vec(cu, nseq + 1, &dev)?;

    let fp8 = c.cache_dtype == DType::F8E4M3;
    let q_dtype = if fp8 { DType::BF16 } else { c.cache_dtype };
    let cache = || {
        Tensor::randn(0f32, 1., (NUM_BLOCKS, N_HEAD_KV, bs, d), &Device::Cpu)?
            .to_dtype(c.cache_dtype)?
            .to_device(&dev)
    };
    let (k_cache, v_cache) = (cache()?, cache()?);
    // dense when every sequence has as many queries, else packed into one row
    let dense = c.query_lens.iter().all(|&l| l == c.query_lens[0]);
    let total_q: usize = c.query_lens.iter().sum();
    let (b, s) = if dense {
        (nseq, c.query_lens[0])
    } else {
        (1, total_q)
    };
    let packed = Tensor::randn(0f32, 1., (1, N_HEAD, total_q, d), &dev)?.to_dtype(q_dtype)?;
    let query = if dense {
        packed
            .reshape((N_HEAD, nseq, s, d))?
            .transpose(0, 1)?
            .contiguous()?
    } else {
        packed.clone()
    };
    let sdpa = SdpaParams {
        n_kv_groups: N_HEAD / N_HEAD_KV,
        softcap: None,
        softmax_scale: 1. / f32::from(u16::try_from(d).unwrap()).sqrt(),
        sliding_window: c.window,
        sinks: None,
    };
    let scales = if fp8 {
        FP8_SCALES
    } else {
        KvCacheScales { k: 1., v: 1. }
    };
    let layer = PagedAttention::new_with_fp8_attention_scales(
        d,
        &dev,
        None,
        fp8.then_some(Fp8AttentionScales {
            q: 1.,
            k: scales.k,
            v: scales.v,
        }),
    )?;
    let metadata = PagedAttentionInputMetadata::dummy(&dev)?;
    let slot_mapping = Tensor::zeros(total_q, DType::I64, &dev)?;
    let ctx = PagedForwardCtx {
        input_metadata: &metadata,
        sdpa_params: &sdpa,
        flash_params: None,
        slot_mapping_full: &slot_mapping,
        slot_mapping: slot_mapping.clone(),
        dims: PagedForwardDims {
            batch_size: b,
            attention_heads: N_HEAD,
            seq_len: s,
            head_size: d,
            key_value_heads: N_HEAD_KV,
        },
        use_full: true,
        alibi_slopes: None,
    };
    let out = layer
        .try_run_fattn_paged_prefill(FattnPrefillCall {
            ctx: &ctx,
            query: &query,
            key_cache: &k_cache,
            value_cache: &v_cache,
            block_tables: &block_tables,
            query_lens: c.query_lens,
            cu_kv: &cu_kv,
            causal: c.causal,
        })?
        .expect("fattn takes the call");
    // back to (1, heads, total, d), the reference's row order
    let out = if dense {
        out.transpose(0, 1)?.reshape((1, N_HEAD, total_q, d))?
    } else {
        out
    };
    let expected = attention_reference(
        &Reference {
            head_dim: d,
            block_size: bs,
            kv_lens: c.kv_lens,
            query_lens: c.query_lens,
            causal: c.causal,
            window: c.window,
            softcap: None,
            heads: (N_HEAD, N_HEAD_KV),
        },
        &packed,
        (&k_cache, &v_cache),
        &tables,
        scales,
    )?;
    let diff = (out
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .squeeze(0)?
        - expected)?
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
fn prefix_prefill_over_a_cached_prefix() -> Result<()> {
    for cache_dtype in [DType::BF16, DType::F16, DType::F8E4M3] {
        // dense: equal query lengths per sequence
        check_prefill(PrefillCase {
            head_dim: 128,
            cache_dtype,
            kv_lens: &[45, 20, 8],
            query_lens: &[8, 8, 8],
            causal: true,
            window: None,
        })?;
        // packed: sequences of their own lengths in one row
        check_prefill(PrefillCase {
            head_dim: 256,
            cache_dtype,
            kv_lens: &[45, 20, 70],
            query_lens: &[5, 17, 1],
            causal: true,
            window: None,
        })?;
    }
    Ok(())
}

#[test]
fn prefix_prefill_with_a_window_and_without_causality() -> Result<()> {
    check_prefill(PrefillCase {
        head_dim: 64,
        cache_dtype: DType::BF16,
        kv_lens: &[90, 33],
        query_lens: &[20, 9],
        causal: true,
        window: Some(24),
    })?;
    // a bidirectional prompt chunk sees every row of its sequence, and a window bounds only its left
    check_prefill(PrefillCase {
        head_dim: 128,
        cache_dtype: DType::BF16,
        kv_lens: &[40, 12],
        query_lens: &[16, 12],
        causal: false,
        window: None,
    })?;
    check_prefill(PrefillCase {
        head_dim: 128,
        cache_dtype: DType::BF16,
        kv_lens: &[60, 30],
        query_lens: &[16, 12],
        causal: false,
        window: Some(20),
    })
}
