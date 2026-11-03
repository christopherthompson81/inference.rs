//! CPU paged attention over the vLLM cache layout the cache engine allocates: keys
//! `[blocks, kv_heads, head_size / x, block_size, x]`, values `[blocks, kv_heads, head_size, block_size]`.

use std::collections::HashMap;

use half::{bf16, f16};
use inference_tensor::{CpuStorage, DType, Device, InplaceOp1, Layout, Result, Storage, Tensor};
use rayon::prelude::*;

// matches the Metal and CUDA kernels' softmax denominator guard
const SOFTMAX_EPS: f32 = 1e-6;
// `softcapping` of exactly this value means no capping
const NO_SOFTCAP: f32 = 1.0;

trait CacheElem: Copy + Send + Sync + 'static {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
    fn slice(s: &CpuStorage) -> Option<&[Self]>;
    fn slice_mut(s: &mut CpuStorage) -> Option<&mut [Self]>;
}

macro_rules! cache_elem {
    ($t:ty, $variant:ident, $to:expr, $from:expr) => {
        impl CacheElem for $t {
            #[inline(always)]
            fn to_f32(self) -> f32 {
                $to(self)
            }
            #[inline(always)]
            fn from_f32(v: f32) -> Self {
                $from(v)
            }
            fn slice(s: &CpuStorage) -> Option<&[Self]> {
                match s {
                    CpuStorage::$variant(v) => Some(v),
                    _ => None,
                }
            }
            fn slice_mut(s: &mut CpuStorage) -> Option<&mut [Self]> {
                match s {
                    CpuStorage::$variant(v) => Some(v),
                    _ => None,
                }
            }
        }
    };
}
cache_elem!(f32, F32, |v: f32| v, |v: f32| v);
cache_elem!(f16, F16, f16::to_f32, f16::from_f32);
cache_elem!(bf16, BF16, bf16::to_f32, bf16::from_f32);

fn check_vulkan_cache(
    op: &str,
    cache: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
) -> Result<()> {
    if cache.dtype() != DType::F32 || k_scale.is_some() || v_scale.is_some() {
        inference_tensor::bail!(
            "{op}: Vulkan caches are f32 without K/V scales, got {:?}",
            cache.dtype()
        );
    }
    Ok(())
}

fn check_cpu_cache(
    op: &str,
    cache: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
) -> Result<()> {
    if !cache.device().is_cpu() {
        inference_tensor::bail!("{op}: the cache is not on the CPU");
    }
    if !matches!(cache.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        inference_tensor::bail!(
            "{op}: CPU caches are f32, f16 or bf16, got {:?}",
            cache.dtype()
        );
    }
    if k_scale.is_some() || v_scale.is_some() {
        inference_tensor::bail!("{op}: CPU caches take no K/V scales");
    }
    Ok(())
}

fn host_vec<T: inference_tensor::WithDType>(t: &Tensor, dtype: DType) -> Result<Vec<T>> {
    t.to_dtype(dtype)?.flatten_all()?.to_vec1::<T>()
}

fn f32_rows(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

struct CacheGeom {
    kv_heads: usize,
    head_size: usize,
    block_size: usize,
    x: usize,
}

impl CacheGeom {
    fn of(key_cache: &Tensor, value_cache: &Tensor) -> Result<Self> {
        let (blocks, kv_heads, chunks, block_size, x) = key_cache.dims5()?;
        let head_size = chunks * x;
        if value_cache.dims4()? != (blocks, kv_heads, head_size, block_size) {
            inference_tensor::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                key_cache.shape(),
                value_cache.shape()
            );
        }
        Ok(Self {
            kv_heads,
            head_size,
            block_size,
            x,
        })
    }
    fn block_elems(&self) -> usize {
        self.kv_heads * self.head_size * self.block_size
    }
    fn head_elems(&self) -> usize {
        self.head_size * self.block_size
    }
    #[inline(always)]
    fn key_idx(&self, block: usize, head: usize, d: usize, t: usize) -> usize {
        block * self.block_elems()
            + head * self.head_elems()
            + (d / self.x) * self.block_size * self.x
            + t * self.x
            + d % self.x
    }
    #[inline(always)]
    fn value_idx(&self, block: usize, head: usize, d: usize, t: usize) -> usize {
        block * self.block_elems() + head * self.head_elems() + d * self.block_size + t
    }
}

fn contiguous_offset(op: &str, l: &Layout) -> Result<usize> {
    if !l.is_contiguous() {
        inference_tensor::bail!("{op}: the cache must be contiguous");
    }
    Ok(l.start_offset())
}

enum CacheSide {
    Key,
    Value,
}

struct WriteRows<'a> {
    side: CacheSide,
    geom: &'a CacheGeom,
    rows: &'a [f32],
    slots: &'a [i64],
}

impl WriteRows<'_> {
    fn write<T: CacheElem>(&self, dst: &mut [T]) {
        let g = self.geom;
        let row = g.kv_heads * g.head_size;
        for (token, &slot) in self.slots.iter().enumerate() {
            if slot < 0 {
                continue;
            }
            let (block, t) = (slot as usize / g.block_size, slot as usize % g.block_size);
            let src = &self.rows[token * row..(token + 1) * row];
            for h in 0..g.kv_heads {
                for d in 0..g.head_size {
                    let idx = match self.side {
                        CacheSide::Key => g.key_idx(block, h, d, t),
                        CacheSide::Value => g.value_idx(block, h, d, t),
                    };
                    dst[idx] = T::from_f32(src[h * g.head_size + d]);
                }
            }
        }
    }
}

impl InplaceOp1 for WriteRows<'_> {
    fn name(&self) -> &'static str {
        "cpu-reshape-and-cache"
    }
    fn cpu_fwd(&self, dst: &mut CpuStorage, dst_l: &Layout) -> Result<()> {
        let offset = contiguous_offset(self.name(), dst_l)?;
        if let Some(s) = f32::slice_mut(dst) {
            self.write(&mut s[offset..]);
        } else if let Some(s) = bf16::slice_mut(dst) {
            self.write(&mut s[offset..]);
        } else if let Some(s) = f16::slice_mut(dst) {
            self.write(&mut s[offset..]);
        } else {
            inference_tensor::bail!("{}: unsupported cache dtype", self.name());
        }
        Ok(())
    }
}

/// Writes `key`/`value` rows (`[tokens, kv_heads, head_size]`) to the cache slots `slot_mapping` (`[tokens]` i64)
/// names; a negative slot is a padding row and is skipped.
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    if key_cache.device().is_vulkan() {
        check_vulkan_cache("reshape_and_cache", key_cache, k_scale, v_scale)?;
        return inference_tensor::vulkan::paged_cache_write(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
        );
    }
    check_cpu_cache("reshape_and_cache", key_cache, k_scale, v_scale)?;
    let geom = CacheGeom::of(key_cache, value_cache)?;
    let (tokens, kv_heads, head_size) = key.dims3()?;
    if (kv_heads, head_size) != (geom.kv_heads, geom.head_size) || value.dims3()? != key.dims3()? {
        inference_tensor::bail!(
            "reshape_and_cache: rows {:?} / {:?} do not fit the cache {:?}",
            key.shape(),
            value.shape(),
            key_cache.shape()
        );
    }
    let slots = host_vec::<i64>(slot_mapping, DType::I64)?;
    if slots.len() < tokens {
        inference_tensor::bail!("reshape_and_cache: {} slots for {tokens} rows", slots.len());
    }
    let slots = &slots[..tokens];
    let (keys, values) = (f32_rows(key)?, f32_rows(value)?);
    key_cache.inplace_op1(&WriteRows {
        side: CacheSide::Key,
        geom: &geom,
        rows: &keys,
        slots,
    })?;
    value_cache.inplace_op1(&WriteRows {
        side: CacheSide::Value,
        geom: &geom,
        rows: &values,
        slots,
    })
}

struct DecodeCall<'a> {
    geom: CacheGeom,
    q: Vec<f32>,
    num_heads: usize,
    block_tables: Vec<u32>,
    max_blocks: usize,
    context_lens: Vec<u32>,
    alibi_slopes: Option<Vec<f32>>,
    sinks: Option<Vec<f32>>,
    softmax_scale: f32,
    softcapping: f32,
    key_cache: &'a Tensor,
    value_cache: &'a Tensor,
}

impl DecodeCall<'_> {
    fn run<T: CacheElem>(&self, kc: &[T], vc: &[T]) -> Vec<f32> {
        let g = &self.geom;
        let hs = g.head_size;
        let queries_per_kv = self.num_heads / g.kv_heads;
        let mut out = vec![0f32; self.q.len()];
        out.par_chunks_mut(hs)
            .enumerate()
            .for_each_init(Vec::new, |logits, (row, out)| {
                let (seq, head) = (row / self.num_heads, row % self.num_heads);
                let kv_head = head / queries_per_kv;
                let ctx = self.context_lens[seq] as usize;
                let q = &self.q[row * hs..(row + 1) * hs];
                let table = &self.block_tables[seq * self.max_blocks..(seq + 1) * self.max_blocks];
                let slope = self.alibi_slopes.as_ref().map_or(0.0, |s| s[head]);
                logits.clear();
                logits.resize(ctx, 0f32);
                for (b, chunk) in logits.chunks_mut(g.block_size).enumerate() {
                    let block = table[b] as usize;
                    for c in 0..hs / g.x {
                        let base = g.key_idx(block, kv_head, c * g.x, 0);
                        let qc = &q[c * g.x..(c + 1) * g.x];
                        for (t, acc) in chunk.iter_mut().enumerate() {
                            let k = &kc[base + t * g.x..base + (t + 1) * g.x];
                            *acc += qc.iter().zip(k).map(|(&a, &b)| a * b.to_f32()).sum::<f32>();
                        }
                    }
                }
                let mut max = self.sinks.as_ref().map_or(f32::NEG_INFINITY, |s| s[head]);
                for (t, l) in logits.iter_mut().enumerate() {
                    let mut s = *l * self.softmax_scale;
                    if self.softcapping != NO_SOFTCAP {
                        s = (s / self.softcapping).tanh() * self.softcapping;
                    }
                    s += slope * (t as f32 - ctx as f32 + 1.0);
                    *l = s;
                    max = max.max(s);
                }
                let mut sum = self.sinks.as_ref().map_or(0.0, |s| (s[head] - max).exp());
                for l in logits.iter_mut() {
                    *l = (*l - max).exp();
                    sum += *l;
                }
                let inv = 1.0 / (sum + SOFTMAX_EPS);
                for (b, chunk) in logits.chunks(g.block_size).enumerate() {
                    let block = table[b] as usize;
                    for (d, o) in out.iter_mut().enumerate() {
                        let base = g.value_idx(block, kv_head, d, 0);
                        let v = &vc[base..base + chunk.len()];
                        *o += chunk
                            .iter()
                            .zip(v)
                            .map(|(&p, &v)| p * v.to_f32())
                            .sum::<f32>();
                    }
                }
                out.iter_mut().for_each(|o| *o *= inv);
            });
        out
    }

    fn dispatch(&self) -> Result<Vec<f32>> {
        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let (Storage::Cpu(kc), Storage::Cpu(vc)) = (&*kc, &*vc) else {
            inference_tensor::bail!("paged_attention: the caches are not on the CPU");
        };
        let (ko, vo) = (
            contiguous_offset("paged_attention", kc_l)?,
            contiguous_offset("paged_attention", vc_l)?,
        );
        macro_rules! go {
            ($t:ty) => {
                if let (Some(k), Some(v)) = (<$t>::slice(kc), <$t>::slice(vc)) {
                    return Ok(self.run(&k[ko..], &v[vo..]));
                }
            };
        }
        go!(bf16);
        go!(f32);
        go!(f16);
        inference_tensor::bail!("paged_attention: unsupported cache dtype")
    }
}

/// One query row per sequence (`q` `[seqs, heads, head_size]`) attending over its `context_lens[seq]` cached rows
/// through `block_tables` (`[seqs, max_blocks]` u32); returns `q`'s shape and dtype.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    if key_cache.device().is_vulkan() {
        check_vulkan_cache("paged_attention", key_cache, k_scale, v_scale)?;
        if alibi_slopes.is_some() || sinks.is_some() || softcapping != NO_SOFTCAP {
            inference_tensor::bail!(
                "paged_attention: Vulkan's kernel takes no ALiBi, sinks or softcap"
            );
        }
        let decoded =
            inference_tensor::vulkan::paged_decode(&inference_tensor::vulkan::PagedDecode {
                q: &q.to_dtype(DType::F32)?,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                max_context: max_context_len,
                scale: softmax_scale,
            })?;
        return match decoded {
            Some(out) => out.to_dtype(q.dtype()),
            None => inference_tensor::bail!(
                "paged_attention: a context of {max_context_len} is past Vulkan's kernel"
            ),
        };
    }
    check_cpu_cache("paged_attention", key_cache, k_scale, v_scale)?;
    let geom = CacheGeom::of(key_cache, value_cache)?;
    let (seqs, num_heads, head_size) = q.dims3()?;
    if head_size != geom.head_size || !num_heads.is_multiple_of(geom.kv_heads) {
        inference_tensor::bail!(
            "paged_attention: query {:?} does not fit the cache {:?}",
            q.shape(),
            key_cache.shape()
        );
    }
    let (table_rows, max_blocks) = block_tables.dims2()?;
    if table_rows != seqs || context_lens.dims1()? != seqs {
        inference_tensor::bail!(
            "paged_attention: {seqs} queries, block tables {:?}, context lens {:?}",
            block_tables.shape(),
            context_lens.shape()
        );
    }
    let call = DecodeCall {
        geom,
        q: f32_rows(q)?,
        num_heads,
        block_tables: host_vec::<u32>(block_tables, DType::U32)?,
        max_blocks,
        context_lens: host_vec::<u32>(context_lens, DType::U32)?,
        alibi_slopes: alibi_slopes.map(f32_rows).transpose()?,
        sinks: sinks.map(f32_rows).transpose()?,
        softmax_scale,
        softcapping,
        key_cache,
        value_cache,
    };
    let out = call.dispatch()?;
    Tensor::from_vec(out, (seqs, num_heads, head_size), &Device::Cpu)?.to_dtype(q.dtype())
}

/// Every sequence's cached rows, laid end to end: `[num_tokens, kv_heads, head_size]` keys and values in `out_dtype`,
/// sequence `b` holding rows `cu_seq_lens[b]..cu_seq_lens[b + 1]` read through `block_table`'s row `b`.
#[allow(clippy::too_many_arguments)]
pub fn gather_kv_cache(
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    num_tokens: usize,
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    // a prompt's gather runs once a chunk: Vulkan's runs the CPU's on host copies of the caches
    if key_cache.device().is_vulkan() {
        check_vulkan_cache("gather_kv_cache", key_cache, k_scale, v_scale)?;
        let host = |t: &Tensor| t.to_device(&Device::Cpu);
        let (k, v) = gather_kv_cache(
            &host(key_cache)?,
            &host(value_cache)?,
            None,
            None,
            &host(block_table)?,
            &host(cu_seq_lens)?,
            num_tokens,
            out_dtype,
        )?;
        return Ok((
            k.to_device(key_cache.device())?,
            v.to_device(key_cache.device())?,
        ));
    }
    check_cpu_cache("gather_kv_cache", key_cache, k_scale, v_scale)?;
    let geom = CacheGeom::of(key_cache, value_cache)?;
    let (_, stride) = block_table.dims2()?;
    let table = host_vec::<u32>(block_table, DType::U32)?;
    let cu = host_vec::<u32>(cu_seq_lens, DType::U32)?;
    let row = geom.kv_heads * geom.head_size;
    let (kc, kc_l) = key_cache.storage_and_layout();
    let (vc, vc_l) = value_cache.storage_and_layout();
    let (Storage::Cpu(kc), Storage::Cpu(vc)) = (&*kc, &*vc) else {
        inference_tensor::bail!("gather_kv_cache: the caches are not on the CPU");
    };
    let (ko, vo) = (
        contiguous_offset("gather_kv_cache", kc_l)?,
        contiguous_offset("gather_kv_cache", vc_l)?,
    );
    let mut keys = vec![0f32; num_tokens * row];
    let mut values = vec![0f32; num_tokens * row];
    macro_rules! go {
        ($t:ty) => {
            if let (Some(k), Some(v)) = (<$t>::slice(kc), <$t>::slice(vc)) {
                let (k, v): (&[$t], &[$t]) = (&k[ko..], &v[vo..]);
                keys.par_chunks_mut(row)
                    .zip(values.par_chunks_mut(row))
                    .enumerate()
                    .for_each(|(token, (kr, vr))| {
                        let at = GatherAt {
                            geom: &geom,
                            table: &table,
                            stride,
                            cu: &cu,
                        };
                        at.row(k, v, kr, vr, token)
                    });
                let shape = (num_tokens, geom.kv_heads, geom.head_size);
                return Ok((
                    Tensor::from_vec(keys, shape, &Device::Cpu)?.to_dtype(out_dtype)?,
                    Tensor::from_vec(values, shape, &Device::Cpu)?.to_dtype(out_dtype)?,
                ));
            }
        };
    }
    go!(bf16);
    go!(f32);
    go!(f16);
    inference_tensor::bail!("gather_kv_cache: unsupported cache dtype")
}

struct GatherAt<'a> {
    geom: &'a CacheGeom,
    table: &'a [u32],
    stride: usize,
    cu: &'a [u32],
}

impl GatherAt<'_> {
    fn row<T: CacheElem>(
        &self,
        kc: &[T],
        vc: &[T],
        keys: &mut [f32],
        values: &mut [f32],
        token: usize,
    ) {
        let g = self.geom;
        let batch = self.cu.partition_point(|&start| start as usize <= token) - 1;
        let offset = token - self.cu[batch] as usize;
        let block = self.table[batch * self.stride + offset / g.block_size] as usize;
        let t = offset % g.block_size;
        for h in 0..g.kv_heads {
            for d in 0..g.head_size {
                keys[h * g.head_size + d] = kc[g.key_idx(block, h, d, t)].to_f32();
                values[h * g.head_size + d] = vc[g.value_idx(block, h, d, t)].to_f32();
            }
        }
    }
}

struct CopyBlocks<'a> {
    pairs: &'a [(usize, usize)],
    block_elems: usize,
}

impl InplaceOp1 for CopyBlocks<'_> {
    fn name(&self) -> &'static str {
        "cpu-copy-blocks"
    }
    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> Result<()> {
        let offset = contiguous_offset(self.name(), layout)?;
        macro_rules! go {
            ($t:ty) => {
                if let Some(s) = <$t>::slice_mut(storage) {
                    let s = &mut s[offset..];
                    for &(src, dst) in self.pairs {
                        s.copy_within(
                            src * self.block_elems..(src + 1) * self.block_elems,
                            dst * self.block_elems,
                        );
                    }
                    return Ok(());
                }
            };
        }
        go!(bf16);
        go!(f32);
        go!(f16);
        inference_tensor::bail!("{}: unsupported cache dtype", self.name())
    }
}

/// Copies each block `src` to every block in `block_mapping[src]`, in every layer's caches.
pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: &HashMap<usize, Vec<usize>>,
) -> Result<()> {
    let pairs = block_mapping
        .iter()
        .flat_map(|(&src, dsts)| dsts.iter().map(move |&dst| (src, dst)))
        .collect::<Vec<_>>();
    for cache in key_caches.into_iter().chain(value_caches) {
        let blocks = cache.dim(0)?;
        let op = CopyBlocks {
            pairs: &pairs,
            block_elems: cache.elem_count() / blocks.max(1),
        };
        cache.inplace_op1(&op)?;
    }
    Ok(())
}

/// FP8 caches, whose scales this updates, are CUDA and Metal only.
pub fn kv_scale_update(
    _key: &Tensor,
    _value: &Tensor,
    _k_scales: &Tensor,
    _v_scales: &Tensor,
) -> Result<()> {
    inference_tensor::bail!("kv_scale_update: FP8 KV caches are not supported on the CPU")
}

#[cfg(test)]
mod tests;
