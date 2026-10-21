use inference_tensor::{D, DType, Result, Storage, Tensor, TensorId};

use super::cache::GdnLayerCache;
use super::config::{GdnDims, GdnVHeadLayout};
#[cfg(feature = "cuda")]
use crate::cuda::gdn::{
    FusedDecodeRecurrence, FusedPrefillOutput, FusedPrefillRecurrence, GdnStateSlots,
    RecurrenceInputs,
};
use crate::gdn::RecurrentBatchKind;
use crate::kv_cache::RecurrentStateLayout;

#[cfg(any(feature = "cuda", feature = "metal"))]
const RECURRENCE_CHUNK_THRESHOLD: usize = 64;
const QK_NORM_EPS: f64 = 1e-6;
const QK_NORM_EPS_F32: f32 = 1e-6;
const SOFTPLUS_LINEAR_THRESHOLD: f32 = 20.0;
// Channels one pool item of the decode conv steps
const CONV_DECODE_CHANNELS: usize = 512;

#[cfg(any(feature = "cuda", feature = "metal"))]
enum RecurrenceOutput {
    BatchHeadMajor(Tensor),
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    TokenMajor(Tensor),
}

#[cfg(feature = "cuda")]
impl From<FusedPrefillOutput> for RecurrenceOutput {
    fn from(output: FusedPrefillOutput) -> Self {
        match output {
            FusedPrefillOutput::TokenMajor(output) => Self::TokenMajor(output),
        }
    }
}

#[cfg(feature = "cuda")]
fn use_warp_prefill_recurrence(dims: &GdnDims) -> bool {
    matches!(dims.head_k_dim, 64 | 128)
}

pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .broadcast_add(&Tensor::new(eps as f32, x.device())?.to_dtype(x.dtype())?)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let dtype = q.dtype();
    let k_head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (k_head_dim as f64).sqrt();

    let q = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?;
    let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let g = g.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let beta = beta.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

    let seq_len = q.dim(2)?;
    let (b_sz, n_heads) = (q.dim(0)?, q.dim(1)?);
    let k_dim = q.dim(3)?;
    let v_dim = v.dim(3)?;

    // Direct time scan per (batch, head): the per-timestep tensor-op chain costs more in
    // dispatch and allocation than the math itself at prefill lengths.
    let qf = q.flatten_all()?.to_vec1::<f32>()?;
    let kf = k.flatten_all()?.to_vec1::<f32>()?;
    let vf = v.flatten_all()?.to_vec1::<f32>()?;
    let gf = g.flatten_all()?.to_vec1::<f32>()?;
    let betaf = beta.flatten_all()?.to_vec1::<f32>()?;
    let mut sf = state
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut out = vec![0f32; b_sz * n_heads * seq_len * v_dim];
    let out_ptr = out.as_mut_ptr() as usize;
    let s_ptr = sf.as_mut_ptr() as usize;
    let (qf, kf, vf, gf, betaf) = (
        qf.as_slice(),
        kf.as_slice(),
        vf.as_slice(),
        gf.as_slice(),
        betaf.as_slice(),
    );

    inference_tensor::utils::barrier_pool().execute_chunked(b_sz * n_heads, |range| {
        let out_ptr = out_ptr as *mut f32;
        let s_ptr = s_ptr as *mut f32;
        let mut kv_mem = vec![0f32; v_dim];
        let mut delta = vec![0f32; v_dim];
        for bh in range {
            let s_head = unsafe {
                std::slice::from_raw_parts_mut(s_ptr.add(bh * k_dim * v_dim), k_dim * v_dim)
            };
            for t in 0..seq_len {
                let qk_base = (bh * seq_len + t) * k_dim;
                let v_base = (bh * seq_len + t) * v_dim;
                let q_t = &qf[qk_base..qk_base + k_dim];
                let k_t = &kf[qk_base..qk_base + k_dim];
                let v_t = &vf[v_base..v_base + v_dim];
                let decay = crate::attention::fast_exp(gf[bh * seq_len + t]);
                let beta_t = betaf[bh * seq_len + t];

                // pass 1: decay the state and read k^T S
                kv_mem.iter_mut().for_each(|x| *x = 0.0);
                for d in 0..k_dim {
                    let row = &mut s_head[d * v_dim..(d + 1) * v_dim];
                    let kd = k_t[d];
                    for (rv, mem) in row.iter_mut().zip(kv_mem.iter_mut()) {
                        *rv *= decay;
                        *mem += *rv * kd;
                    }
                }
                for ((dl, &vv), &mem) in delta.iter_mut().zip(v_t).zip(kv_mem.iter()) {
                    *dl = (vv - mem) * beta_t;
                }

                // pass 2: rank-1 update and the q readout
                let y = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(v_base), v_dim) };
                y.iter_mut().for_each(|x| *x = 0.0);
                for d in 0..k_dim {
                    let row = &mut s_head[d * v_dim..(d + 1) * v_dim];
                    let kd = k_t[d];
                    let qd = q_t[d];
                    for ((rv, &dl), yv) in row.iter_mut().zip(delta.iter()).zip(y.iter_mut()) {
                        *rv += kd * dl;
                        *yv += *rv * qd;
                    }
                }
            }
        }
    });

    *state = Tensor::from_vec(sf, state.shape(), state.device())?.to_dtype(state.dtype())?;

    Tensor::from_vec(out, (b_sz, n_heads, seq_len, v_dim), q.device())?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(dtype)
}

pub fn compute_beta_g(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "metal")]
    if b.device().is_metal() {
        let b_flat = b.contiguous()?.flatten_all()?;
        let a_flat = a.contiguous()?.flatten_all()?;
        let a_log_f32 = a_log.to_dtype(DType::F32)?.contiguous()?;
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?.contiguous()?;
        let (beta_flat, g_flat) =
            crate::metal::gdn::fused_gdn_gating_metal(&b_flat, &a_flat, &a_log_f32, &dt_bias_f32)?;
        let shape = b.shape();
        return Ok((beta_flat.reshape(shape)?, g_flat.reshape(shape)?));
    }

    compute_beta_g_cpu(b, a, a_log, dt_bias, dtype)
}

// The gate and decay stay f32 like HF's fla path; the recurrence upcasts anyway
fn compute_beta_g_cpu(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    _dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let beta = inference_tensor::nn::ops::sigmoid(&b.to_dtype(DType::F32)?)?;
    let a_f = a.to_dtype(DType::F32)?;
    let dt_bias_expanded = dt_bias.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;
    let g = a_log
        .to_dtype(DType::F32)?
        .exp()?
        .neg()?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .broadcast_mul(&softplus(&a_f.broadcast_add(&dt_bias_expanded)?)?)?;
    Ok((beta, g))
}

fn expand_k_heads(x: &Tensor, dims: &GdnDims, batch_size: usize, seq_len: usize) -> Result<Tensor> {
    if dims.v_per_group == 1 {
        return Ok(x.clone());
    }
    let expanded = match dims.v_head_layout {
        GdnVHeadLayout::Grouped => x.unsqueeze(3)?.repeat((1, 1, 1, dims.v_per_group, 1))?,
        GdnVHeadLayout::Tiled => x.unsqueeze(2)?.repeat((1, 1, dims.v_per_group, 1, 1))?,
    };
    expanded.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_k_dim))
}

#[allow(clippy::too_many_arguments)]
pub fn apply_recurrence_from_convolved(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if mixed_qkv.device().is_cuda() {
        return recurrence_cuda_from_convolved(
            mixed_qkv, b, a, a_log, dt_bias, dims, batch_size, seq_len, cache, dtype,
        );
    }

    if mixed_qkv.device().is_cpu() {
        return recurrence_cpu_from_convolved(
            mixed_qkv, b, a, a_log, dt_bias, dims, batch_size, seq_len, cache, dtype,
        );
    }

    let q = mixed_qkv.narrow(D::Minus1, 0, dims.key_dim)?;
    let k = mixed_qkv.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
    let v = mixed_qkv.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;
    let q = q.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let k = k.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let v = v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?;
    let q = expand_k_heads(&q, dims, batch_size, seq_len)?;
    let k = expand_k_heads(&k, dims, batch_size, seq_len)?;
    let (beta, g) = compute_beta_g(b, a, a_log, dt_bias, dtype)?;
    let q = l2_norm(&q, QK_NORM_EPS)?;
    let k = l2_norm(&k, QK_NORM_EPS)?;
    apply_recurrence(
        &q, &k, &v, &g, &beta, dims, batch_size, seq_len, cache, dtype,
    )
}

#[allow(clippy::too_many_arguments)]
fn recurrence_cpu_from_convolved(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    if cache.state_layout != RecurrentStateLayout::GdnKeyMajor {
        inference_tensor::bail!(
            "CPU GDN recurrence requires key-major state, got {:?}",
            cache.state_layout
        );
    }
    let dev = mixed_qkv.device();
    let mixed_f32 = mixed_qkv.to_dtype(DType::F32)?.contiguous()?;
    let b_f32 = b.to_dtype(DType::F32)?.contiguous()?;
    let a_f32 = a.to_dtype(DType::F32)?.contiguous()?;
    let a_log_f32 = a_log.to_dtype(DType::F32)?.contiguous()?;
    let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?.contiguous()?;

    let (mixed_storage, mixed_layout) = mixed_f32.storage_and_layout();
    let mixed = cpu_f32_slice(&mixed_storage, mixed_layout.start_offset(), "mixed_qkv")?;
    let (b_storage, b_layout) = b_f32.storage_and_layout();
    let b = cpu_f32_slice(&b_storage, b_layout.start_offset(), "b")?;
    let (a_storage, a_layout) = a_f32.storage_and_layout();
    let a = cpu_f32_slice(&a_storage, a_layout.start_offset(), "a")?;
    let (a_log_storage, a_log_layout) = a_log_f32.storage_and_layout();
    let a_log = cpu_f32_slice(&a_log_storage, a_log_layout.start_offset(), "a_log")?;
    let (dt_bias_storage, dt_bias_layout) = dt_bias_f32.storage_and_layout();
    let dt_bias = cpu_f32_slice(&dt_bias_storage, dt_bias_layout.start_offset(), "dt_bias")?;

    let mut output = vec![0.0f32; batch_size * seq_len * dims.num_v_heads * dims.head_v_dim];
    let q_scale = 1.0f32 / (dims.head_k_dim as f32).sqrt();
    let state_head_len = dims.head_k_dim * dims.head_v_dim;

    let out_ptr = output.as_mut_ptr() as usize;
    let run = |state: &mut [f32]| {
        let state_ptr = state.as_mut_ptr() as usize;
        // row-wise passes over each head's state keep the inner loops on contiguous value columns
        inference_tensor::utils::barrier_pool().execute_chunked(
            batch_size * dims.num_v_heads,
            |heads| {
                let mut kv_mem = vec![0f32; dims.head_v_dim];
                let mut delta = vec![0f32; dims.head_v_dim];
                let mut q_buf = vec![0f32; dims.head_k_dim];
                let mut k_buf = vec![0f32; dims.head_k_dim];
                for head in heads {
                    // SAFETY: each head's state and output rows are touched by this item alone, and both outlive the call
                    let state_head = unsafe {
                        std::slice::from_raw_parts_mut(
                            (state_ptr as *mut f32).add(head * state_head_len),
                            state_head_len,
                        )
                    };
                    let bidx = head / dims.num_v_heads;
                    let hv = head % dims.num_v_heads;
                    let hk = dims.k_head_for_v_head(hv);
                    for t in 0..seq_len {
                        let token = bidx * seq_len + t;
                        let gate_idx = token * dims.num_v_heads + hv;
                        // SAFETY: as above
                        let out_head = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(gate_idx * dims.head_v_dim),
                                dims.head_v_dim,
                            )
                        };
                        let row = token * dims.conv_dim;
                        let q_in = &mixed[row + hk * dims.head_k_dim..][..dims.head_k_dim];
                        let k_in =
                            &mixed[row + dims.key_dim + hk * dims.head_k_dim..][..dims.head_k_dim];
                        let v_in = &mixed[row + 2 * dims.key_dim + hv * dims.head_v_dim..]
                            [..dims.head_v_dim];

                        let q_sum = q_in.iter().map(|q| q * q).sum::<f32>();
                        let k_sum = k_in.iter().map(|k| k * k).sum::<f32>();
                        let q_mul = q_scale / (q_sum + QK_NORM_EPS_F32).sqrt();
                        let k_mul = 1.0f32 / (k_sum + QK_NORM_EPS_F32).sqrt();
                        for (buf, x) in q_buf.iter_mut().zip(q_in) {
                            *buf = x * q_mul;
                        }
                        for (buf, x) in k_buf.iter_mut().zip(k_in) {
                            *buf = x * k_mul;
                        }
                        let beta = sigmoid_f32(b[gate_idx]);
                        let decay =
                            (-a_log[hv].exp() * softplus_f32(a[gate_idx] + dt_bias[hv])).exp();

                        // pass 1: decay the state and read k^T S
                        kv_mem.fill(0.0);
                        for (state_row, &kd) in
                            state_head.chunks_exact_mut(dims.head_v_dim).zip(&k_buf)
                        {
                            for (s, mem) in state_row.iter_mut().zip(kv_mem.iter_mut()) {
                                *s *= decay;
                                *mem += *s * kd;
                            }
                        }
                        for ((d, &v), &mem) in delta.iter_mut().zip(v_in).zip(&kv_mem) {
                            *d = (v - mem) * beta;
                        }
                        // pass 2: rank-1 update and the q readout
                        out_head.fill(0.0);
                        for ((state_row, &kd), &qd) in state_head
                            .chunks_exact_mut(dims.head_v_dim)
                            .zip(&k_buf)
                            .zip(&q_buf)
                        {
                            for ((s, &d), y) in
                                state_row.iter_mut().zip(&delta).zip(out_head.iter_mut())
                            {
                                *s += kd * d;
                                *y += *s * qd;
                            }
                        }
                    }
                }
            },
        );
    };
    if !cpu_f32_in_place(
        &cache.recurrent_state,
        cache.pool_views.map(|[_, state]| state),
        run,
    )? {
        let mut state = cache
            .recurrent_state
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        run(&mut state);
        cache.recurrent_state = Tensor::from_vec(
            state,
            (
                batch_size,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            ),
            dev,
        )?
        .to_dtype(cache.recurrent_state.dtype())?;
    }

    Tensor::from_vec(
        output,
        (batch_size, seq_len, dims.num_v_heads, dims.head_v_dim),
        dev,
    )?
    .to_dtype(dtype)
}

fn cpu_f32_slice<'a>(
    storage: &'a Storage,
    start_offset: usize,
    name: &'static str,
) -> Result<&'a [f32]> {
    let Storage::Cpu(cpu) = storage else {
        inference_tensor::bail!("Expected CPU storage for {name}");
    };
    let data = cpu.as_slice::<f32>()?;
    Ok(&data[start_offset..])
}

struct CpuF32InPlace<F>(std::cell::Cell<Option<F>>);

impl<F: FnOnce(&mut [f32])> inference_tensor::InplaceOp1 for CpuF32InPlace<F> {
    fn name(&self) -> &'static str {
        "gdn-cpu-f32-in-place"
    }

    fn cpu_fwd(
        &self,
        storage: &mut inference_tensor::CpuStorage,
        layout: &inference_tensor::Layout,
    ) -> Result<()> {
        let inference_tensor::CpuStorage::F32(data) = storage else {
            inference_tensor::bail!("in-place GDN state is not F32");
        };
        let f = self.0.take().expect("an in-place op runs once");
        f(&mut data[layout.start_offset()..][..layout.shape().elem_count()]);
        Ok(())
    }
}

// Runs `f` on `t`'s values in place when `t` is the pool row view `view` the cache checked out (other tensors may
// share their storage) and is contiguous F32 on the CPU; reports whether it did
fn cpu_f32_in_place(
    t: &Tensor,
    view: Option<TensorId>,
    f: impl FnOnce(&mut [f32]),
) -> Result<bool> {
    if view != Some(t.id()) || !t.device().is_cpu() || t.dtype() != DType::F32 || !t.is_contiguous()
    {
        return Ok(false);
    }
    t.inplace_op1(&CpuF32InPlace(std::cell::Cell::new(Some(f))))?;
    Ok(true)
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus_f32(x: f32) -> f32 {
    if x > SOFTPLUS_LINEAR_THRESHOLD {
        x
    } else if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn recurrence_cuda_from_convolved(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    let mixed_qkv = mixed_qkv.contiguous()?;
    let (b, a) = if seq_len == 1 {
        (b.clone(), a.clone())
    } else {
        (b.contiguous()?, a.contiguous()?)
    };
    let a_log = a_log.to_dtype(DType::F32)?.contiguous()?;
    let dt_bias = dt_bias.to_dtype(DType::F32)?.contiguous()?;
    let mut state_flat = prepare_state_for_backend(cache, dims, batch_size)?;
    let slots = GdnStateSlots::from_option(cache.slots.as_ref());

    let output = if seq_len == 1 {
        RecurrenceOutput::BatchHeadMajor(crate::cuda::gdn::fused_decode_recurrence_cuda(
            FusedDecodeRecurrence {
                mixed_qkv: &mixed_qkv,
                b: &b,
                a: &a,
                a_log: &a_log,
                dt_bias: &dt_bias,
                state: &mut state_flat,
                batch_size,
                num_k_heads: dims.num_k_heads,
                num_v_heads: dims.num_v_heads,
                head_k_dim: dims.head_k_dim,
                head_v_dim: dims.head_v_dim,
                tiled_v_heads: dims.v_head_layout == GdnVHeadLayout::Tiled,
                state_layout: cache.state_layout,
                slots,
            },
        )?)
    } else {
        let fused_prefill = if cache.state_layout == RecurrentStateLayout::GdnValueMajor {
            crate::cuda::gdn::try_fused_vmajor_prefill_recurrence_cuda(FusedPrefillRecurrence {
                mixed_qkv: &mixed_qkv,
                b: &b,
                a: &a,
                a_log: &a_log,
                dt_bias: &dt_bias,
                state: &mut state_flat,
                batch_size,
                num_k_heads: dims.num_k_heads,
                num_v_heads: dims.num_v_heads,
                head_k_dim: dims.head_k_dim,
                head_v_dim: dims.head_v_dim,
                tiled_v_heads: dims.v_head_layout == GdnVHeadLayout::Tiled,
                state_layout: cache.state_layout,
                slots,
            })?
        } else {
            None
        };
        if let Some(output) = fused_prefill {
            output.into()
        } else {
            let (q_bh, k_bh, v_bh, g_bh, beta_bh) =
                crate::cuda::gdn::prepare_recurrence_inputs_cuda(
                    &mixed_qkv,
                    &b,
                    &a,
                    &a_log,
                    &dt_bias,
                    batch_size,
                    seq_len,
                    dims.num_k_heads,
                    dims.num_v_heads,
                    dims.head_k_dim,
                    dims.head_v_dim,
                    dims.v_head_layout == GdnVHeadLayout::Tiled,
                )?;
            let inputs = RecurrenceInputs {
                q: &q_bh,
                k: &k_bh,
                v: &v_bh,
                g: &g_bh,
                beta: &beta_bh,
            };
            if cache.state_layout == RecurrentStateLayout::GdnValueMajor {
                RecurrenceOutput::BatchHeadMajor(
                    crate::cuda::gdn::vmajor_prefill_gated_delta_rule_recurrence_cuda(
                        inputs,
                        &mut state_flat,
                        slots,
                        dtype,
                    )?,
                )
            } else if seq_len >= RECURRENCE_CHUNK_THRESHOLD && use_warp_prefill_recurrence(dims) {
                RecurrenceOutput::BatchHeadMajor(
                    crate::cuda::gdn::warp_gated_delta_rule_recurrence_cuda(
                        inputs,
                        &mut state_flat,
                        slots,
                    )?,
                )
            } else {
                RecurrenceOutput::BatchHeadMajor(
                    crate::cuda::gdn::gated_delta_rule_recurrence_cuda(
                        inputs,
                        &mut state_flat,
                        slots,
                    )?,
                )
            }
        }
    };

    finish_recurrence(output, state_flat, dims, batch_size, seq_len, cache, dtype)
}

#[cfg_attr(not(feature = "metal"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub fn apply_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return recurrence_metal(q, k, v, g, beta, dims, batch_size, seq_len, cache, dtype);
    }

    if cache.state_layout != RecurrentStateLayout::GdnKeyMajor {
        inference_tensor::bail!(
            "CPU GDN recurrence requires key-major state, got {:?}",
            cache.state_layout
        );
    }
    gated_delta_rule_recurrence(q, k, v, g, beta, &mut cache.recurrent_state)
}

#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
fn recurrence_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    if cache.state_layout != RecurrentStateLayout::GdnKeyMajor {
        inference_tensor::bail!(
            "Metal GDN recurrence requires key-major state, got {:?}",
            cache.state_layout
        );
    }
    let q_bh = prepare_q_for_backend(q, dims, batch_size, seq_len)?;
    let k_bh = prepare_kv_for_backend(k, dims, batch_size, seq_len, dims.head_k_dim)?;
    let v_bh = prepare_kv_for_backend(v, dims, batch_size, seq_len, dims.head_v_dim)?;
    let g_bh = prepare_gate_for_backend(g, dims, batch_size, seq_len)?;
    let beta_bh = prepare_gate_for_backend(beta, dims, batch_size, seq_len)?;
    let mut state_flat = prepare_state_for_backend(cache, dims, batch_size)?;

    let out_bh = if seq_len >= RECURRENCE_CHUNK_THRESHOLD {
        crate::metal::gdn::chunked_gated_delta_rule_recurrence_metal(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            &mut state_flat,
        )?
    } else {
        crate::metal::gdn::gated_delta_rule_recurrence_metal(
            &q_bh,
            &k_bh,
            &v_bh,
            &g_bh,
            &beta_bh,
            &mut state_flat,
        )?
    };

    finish_recurrence(
        RecurrenceOutput::BatchHeadMajor(out_bh),
        state_flat,
        dims,
        batch_size,
        seq_len,
        cache,
        dtype,
    )
}

#[cfg(feature = "metal")]
fn prepare_q_for_backend(
    q: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let scale = 1.0 / (dims.head_k_dim as f64).sqrt();
    (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?
        .reshape((batch_size * dims.num_v_heads, seq_len, dims.head_k_dim))?
        .contiguous()
}

#[cfg(feature = "metal")]
fn prepare_kv_for_backend(
    x: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor> {
    x.transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .reshape((batch_size * dims.num_v_heads, seq_len, head_dim))?
        .contiguous()
}

#[cfg(feature = "metal")]
fn prepare_gate_for_backend(
    x: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
) -> Result<Tensor> {
    x.to_dtype(DType::F32)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch_size * dims.num_v_heads, seq_len))?
        .contiguous()
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn recurrent_physical_dims(layout: RecurrentStateLayout, dims: &GdnDims) -> Result<(usize, usize)> {
    match layout {
        RecurrentStateLayout::GdnKeyMajor => Ok((dims.head_k_dim, dims.head_v_dim)),
        RecurrentStateLayout::GdnValueMajor => Ok((dims.head_v_dim, dims.head_k_dim)),
        RecurrentStateLayout::Opaque => {
            inference_tensor::bail!("GDN cache has opaque state layout")
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn prepare_state_for_backend(
    cache: &GdnLayerCache,
    dims: &GdnDims,
    batch_size: usize,
) -> Result<Tensor> {
    let (physical_dim_2, physical_dim_3) = recurrent_physical_dims(cache.state_layout, dims)?;
    let state_dims = cache.recurrent_state.dims4()?;
    if state_dims.1 != dims.num_v_heads
        || state_dims.2 != physical_dim_2
        || state_dims.3 != physical_dim_3
    {
        inference_tensor::bail!(
            "GDN {:?} state shape mismatch: got {:?}, expected [B, {}, {}, {}]",
            cache.state_layout,
            cache.recurrent_state.dims(),
            dims.num_v_heads,
            physical_dim_2,
            physical_dim_3
        );
    }
    if cache.slots.is_some() {
        if !crate::cuda::gdn::recurrent_state_dtype_supported(cache.recurrent_state.dtype())
            || !cache.recurrent_state.is_contiguous()
        {
            inference_tensor::bail!("pooled GDN recurrent state must be contiguous f16/bf16/f32");
        }
        return Ok(cache.recurrent_state.clone());
    }
    cache
        .recurrent_state
        .to_dtype(DType::F32)?
        .reshape((
            batch_size * dims.num_v_heads,
            physical_dim_2,
            physical_dim_3,
        ))?
        .contiguous()
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn finish_recurrence(
    output: RecurrenceOutput,
    state_flat: Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    if cache.slots.is_none() {
        let (physical_dim_2, physical_dim_3) = recurrent_physical_dims(cache.state_layout, dims)?;
        cache.recurrent_state = state_flat
            .reshape((batch_size, dims.num_v_heads, physical_dim_2, physical_dim_3))?
            .to_dtype(cache.recurrent_state.dtype())?;
    }

    match output {
        RecurrenceOutput::BatchHeadMajor(output) => output
            .reshape((batch_size, dims.num_v_heads, seq_len, dims.head_v_dim))?
            .transpose(1, 2)?
            .to_dtype(dtype),
        RecurrenceOutput::TokenMajor(output) => {
            if output.dims4()? != (batch_size, seq_len, dims.num_v_heads, dims.head_v_dim) {
                inference_tensor::bail!("GDN token-major output has an incompatible shape")
            }
            output.to_dtype(dtype)
        }
    }
}

pub fn causal_conv1d(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
    batch_kind: RecurrentBatchKind,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    if matches!(batch_kind, RecurrentBatchKind::Decode) {
        if seq_len != 1 {
            inference_tensor::bail!("GDN decode expects a single-token query.");
        }
        causal_conv1d_update(x, conv1d_weight, dims, cache)
    } else {
        causal_conv1d_full(x, conv1d_weight, dims, cache)
    }
}

fn causal_conv1d_update(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;

    if x.device().is_cpu() {
        return causal_conv1d_update_cpu(x, conv1d_weight, dims, cache);
    }

    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        let weight = conv1d_weight
            .squeeze(1)?
            .to_dtype(x.dtype())?
            .contiguous()?;
        let conv_state = cache.conv_state.contiguous()?;
        let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
            x,
            &weight,
            &conv_state,
            dims.conv_kernel_size,
            true,
            GdnStateSlots::from_option(cache.slots.as_ref()),
        )?;
        cache.conv_state = new_conv_state;
        return Ok(output);
    }

    let x_t = x.transpose(1, 2)?.contiguous()?;

    #[cfg(feature = "metal")]
    if x_t.device().is_metal() {
        let weight = conv1d_weight
            .squeeze(1)?
            .to_dtype(x_t.dtype())?
            .contiguous()?;
        let conv_state = cache.conv_state.contiguous()?;
        let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
            &x_t,
            &weight,
            &conv_state,
            true,
            dims.conv_kernel_size,
        )?;
        cache.conv_state = new_conv_state;
        return output.transpose(1, 2);
    }

    let state_len = cache.conv_state.dim(2)?;
    let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
    let new_len = hidden_new.dim(2)?;
    cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(hidden_new.dtype())?;
    let mut conv_outputs = Vec::with_capacity(seq_len);
    let total_len = hidden_new.dim(2)?;
    for i in (total_len - seq_len)..total_len {
        let window = hidden_new.narrow(2, i + 1 - dims.conv_kernel_size, dims.conv_kernel_size)?;
        let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
        conv_outputs.push(out);
    }
    inference_tensor::nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
}

fn causal_conv1d_update_cpu(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (batch_size, seq_len, conv_dim) = x.dims3()?;
    if seq_len != 1 {
        inference_tensor::bail!("GDN CPU conv decode expects a single-token query.");
    }

    let dev = x.device();
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?.contiguous()?;
    let weight_f32 = conv1d_weight.to_dtype(DType::F32)?.contiguous()?;
    let (x_storage, x_layout) = x_f32.storage_and_layout();
    let x = cpu_f32_slice(&x_storage, x_layout.start_offset(), "x")?;
    let (weight_storage, weight_layout) = weight_f32.storage_and_layout();
    let weight = cpu_f32_slice(
        &weight_storage,
        weight_layout.start_offset(),
        "conv1d_weight",
    )?;
    let mut output = vec![0.0f32; batch_size * conv_dim];
    let out_ptr = output.as_mut_ptr() as usize;
    let k = dims.conv_kernel_size;
    let step = |state: &mut [f32]| {
        let state_ptr = state.as_mut_ptr() as usize;
        let channels = batch_size * conv_dim;
        inference_tensor::utils::barrier_pool().execute_chunked(
            channels.div_ceil(CONV_DECODE_CHANNELS),
            |chunks| {
                let range = chunks.start * CONV_DECODE_CHANNELS
                    ..channels.min(chunks.end * CONV_DECODE_CHANNELS);
                // SAFETY: each chunk's channels (state windows and outputs) belong to it alone
                let (state, output) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(
                            (state_ptr as *mut f32).add(range.start * k),
                            range.len() * k,
                        ),
                        std::slice::from_raw_parts_mut(
                            (out_ptr as *mut f32).add(range.start),
                            range.len(),
                        ),
                    )
                };
                for ((idx, state_channel), output) in
                    range.zip(state.chunks_exact_mut(k)).zip(output)
                {
                    state_channel.copy_within(1.., 0);
                    state_channel[k - 1] = x[idx];
                    let weight = &weight[idx % conv_dim * k..][..k];
                    let sum = state_channel
                        .iter()
                        .zip(weight)
                        .map(|(s, w)| s * w)
                        .sum::<f32>();
                    *output = silu_f32(sum);
                }
            },
        );
    };
    if !cpu_f32_in_place(
        &cache.conv_state,
        cache.pool_views.map(|[conv, _]| conv),
        step,
    )? {
        let mut state = cache
            .conv_state
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        step(&mut state);
        cache.conv_state =
            Tensor::from_vec(state, (batch_size, conv_dim, dims.conv_kernel_size), dev)?
                .to_dtype(cache.conv_state.dtype())?;
    }

    Tensor::from_vec(output, (batch_size, 1, conv_dim), dev)?.to_dtype(dtype)
}

fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

fn causal_conv1d_full(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let seq_len = x.dim(1)?;
    if x.device().is_cpu() {
        return causal_conv1d_full_cpu(x, conv1d_weight, dims, cache);
    }

    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        let weight = conv1d_weight
            .squeeze(1)?
            .to_dtype(x.dtype())?
            .contiguous()?;
        let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
            x,
            &weight,
            &cache.conv_state,
            dims.conv_kernel_size,
            false,
            GdnStateSlots::from_option(cache.slots.as_ref()),
        )?;
        cache.conv_state = new_conv_state;
        return Ok(output);
    }

    let x_t = x.transpose(1, 2)?.contiguous()?;

    #[cfg(feature = "metal")]
    if x_t.device().is_metal() {
        let weight = conv1d_weight
            .squeeze(1)?
            .to_dtype(x_t.dtype())?
            .contiguous()?;
        let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
            &x_t,
            &weight,
            &cache.conv_state,
            false,
            dims.conv_kernel_size,
        )?;
        cache.conv_state = new_conv_state;
        return output.transpose(1, 2);
    }

    let state_len = cache.conv_state.dim(2)?;
    if state_len != dims.conv_kernel_size {
        inference_tensor::bail!(
            "GDN convolution state width is {state_len}, expected {}",
            dims.conv_kernel_size
        );
    }
    let prior_state = cache.conv_state.clone();
    let state_and_input = Tensor::cat(&[prior_state.clone(), x_t.clone()], 2)?;
    cache.conv_state = state_and_input.narrow(
        2,
        state_and_input.dim(2)? - dims.conv_kernel_size,
        dims.conv_kernel_size,
    )?;
    let padded_t = Tensor::cat(
        &[prior_state.narrow(2, 1, dims.conv_kernel_size - 1)?, x_t],
        2,
    )?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;

    let mut conv_outputs = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let window = padded_t.narrow(2, i, dims.conv_kernel_size)?;
        let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
        conv_outputs.push(out);
    }
    inference_tensor::nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
}

// Depthwise causal conv + silu over the `[b, seq, conv]` input as is; transposing it cost more than the conv
fn causal_conv1d_full_cpu(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (batch_size, seq_len, conv_dim) = x.dims3()?;
    let ksize = dims.conv_kernel_size;
    if cache.conv_state.dim(2)? != ksize {
        inference_tensor::bail!(
            "GDN convolution state width is {}, expected {ksize}",
            cache.conv_state.dim(2)?
        );
    }
    let (dev, dtype) = (x.device().clone(), x.dtype());
    let x = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    // taps innermost-channel: [k, conv]
    let w = conv1d_weight
        .to_dtype(DType::F32)?
        .reshape((conv_dim, ksize))?
        .t()?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let state = cache
        .conv_state
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    // input at time j of batch b, reaching back into the state ([b, conv, k]) for j < 0
    let input = |b: usize, j: isize, c: usize| -> f32 {
        if j >= 0 {
            x[(b * seq_len + j as usize) * conv_dim + c]
        } else {
            state[(b * conv_dim + c) * ksize + (ksize as isize + j) as usize]
        }
    };

    let mut out = vec![0f32; batch_size * seq_len * conv_dim];
    let out_ptr = out.as_mut_ptr() as usize;
    inference_tensor::utils::barrier_pool().execute_chunked(batch_size * seq_len, |rows| {
        for bt in rows {
            let (b, t) = (bt / seq_len, bt % seq_len);
            // SAFETY: each (b, t) row of `out` is written by this item alone, and `out` outlives the call
            let dst = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(bt * conv_dim), conv_dim)
            };
            for k in 0..ksize {
                let j = t as isize + k as isize - (ksize as isize - 1);
                let wk = &w[k * conv_dim..][..conv_dim];
                if j >= 0 {
                    let src = &x[(b * seq_len + j as usize) * conv_dim..][..conv_dim];
                    for ((d, &wc), &xc) in dst.iter_mut().zip(wk).zip(src) {
                        *d += wc * xc;
                    }
                } else {
                    for (c, (d, &wc)) in dst.iter_mut().zip(wk).enumerate() {
                        *d += wc * input(b, j, c);
                    }
                }
            }
            for d in dst.iter_mut() {
                *d /= 1.0 + crate::attention::fast_exp(-*d);
            }
        }
    });

    // the last k inputs of state ++ x, per channel
    let mut new_state = vec![0f32; batch_size * conv_dim * ksize];
    for b in 0..batch_size {
        for c in 0..conv_dim {
            for k in 0..ksize {
                let j = seq_len as isize - ksize as isize + k as isize;
                new_state[(b * conv_dim + c) * ksize + k] = input(b, j, c);
            }
        }
    }
    cache.conv_state = Tensor::from_vec(new_state, (batch_size, conv_dim, ksize), &dev)?
        .to_dtype(cache.conv_state.dtype())?;
    Tensor::from_vec(out, (batch_size, seq_len, conv_dim), &dev)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::{Device, Result as CandleResult};

    const ASSERT_EPS: f32 = 5e-5;
    const TEST_RMS_NORM_EPS: f64 = 1e-6;

    fn patterned(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as f32;
                ((x / 128.0) - 1.0) * scale + offset
            })
            .collect()
    }

    fn dims_with_layout(
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        v_head_layout: GdnVHeadLayout,
    ) -> GdnDims {
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        GdnDims {
            hidden_size: value_dim,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size: 4,
            key_dim,
            value_dim,
            conv_dim: key_dim * 2 + value_dim,
            v_per_group: num_v_heads / num_k_heads,
            v_head_layout,
        }
    }

    fn dims(
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> GdnDims {
        dims_with_layout(
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            GdnVHeadLayout::Grouped,
        )
    }

    fn assert_close(lhs: &Tensor, rhs: &Tensor) -> CandleResult<()> {
        assert_eq!(lhs.shape(), rhs.shape());
        let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&lhs, &rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= ASSERT_EPS,
                "idx={idx} lhs={lhs} rhs={rhs} diff={diff}"
            );
        }
        Ok(())
    }

    fn converter_tiled_heads(dims: &GdnDims) -> Vec<usize> {
        (0..dims.v_per_group)
            .flat_map(|within_group| {
                (0..dims.num_k_heads)
                    .map(move |key_head| key_head * dims.v_per_group + within_group)
            })
            .collect()
    }

    fn expand_head_map(heads: &[usize], head_dim: usize) -> Vec<usize> {
        heads
            .iter()
            .flat_map(|head| (0..head_dim).map(move |feature| head * head_dim + feature))
            .collect()
    }

    fn runtime_from_grouped(
        tensor: &Tensor,
        dim: usize,
        runtime_to_grouped: &[usize],
    ) -> CandleResult<Tensor> {
        let indices = runtime_to_grouped
            .iter()
            .map(|&index| index as u32)
            .collect::<Vec<_>>();
        tensor
            .contiguous()?
            .index_select(
                &Tensor::from_vec(indices, (runtime_to_grouped.len(),), tensor.device())?,
                dim,
            )?
            .contiguous()
    }

    fn grouped_from_runtime(
        tensor: &Tensor,
        dim: usize,
        runtime_to_grouped: &[usize],
    ) -> CandleResult<Tensor> {
        let mut grouped_to_runtime = vec![0; runtime_to_grouped.len()];
        for (runtime, &grouped) in runtime_to_grouped.iter().enumerate() {
            grouped_to_runtime[grouped] = runtime;
        }
        runtime_from_grouped(tensor, dim, &grouped_to_runtime)
    }

    struct BackendStep<'a> {
        mixed_qkv: &'a Tensor,
        b: &'a Tensor,
        a: &'a Tensor,
        a_log: &'a Tensor,
        dt_bias: &'a Tensor,
        conv_weight: &'a Tensor,
        dims: &'a GdnDims,
    }

    fn run_backend_step(
        step: BackendStep<'_>,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (batch_size, seq_len, _) = step.mixed_qkv.dims3()?;
        let convolved = causal_conv1d(
            step.mixed_qkv,
            step.conv_weight,
            step.dims,
            cache,
            batch_kind,
        )?;
        let recurrent = apply_recurrence_from_convolved(
            &convolved,
            step.b,
            step.a,
            step.a_log,
            step.dt_bias,
            step.dims,
            batch_size,
            seq_len,
            cache,
            DType::F32,
        )?;
        Ok((convolved, recurrent))
    }

    fn finish_backend_step(
        recurrent: &Tensor,
        z: &Tensor,
        norm_weight: &Tensor,
        out_weight: &Tensor,
        dims: &GdnDims,
    ) -> CandleResult<Tensor> {
        let (batch_size, seq_len, num_v_heads, head_v_dim) = z.dims4()?;
        assert_eq!(num_v_heads, dims.num_v_heads);
        assert_eq!(head_v_dim, dims.head_v_dim);
        let recurrent = recurrent.reshape(((), dims.head_v_dim))?;
        let gate = inference_tensor::nn::ops::silu(&z.reshape(((), dims.head_v_dim))?)?;
        let variance = recurrent.sqr()?.mean_keepdim(D::Minus1)?;
        let normalized = recurrent.broadcast_div(&(variance + TEST_RMS_NORM_EPS)?.sqrt()?)?;
        let normalized = normalized
            .broadcast_mul(norm_weight)?
            .broadcast_mul(&gate)?
            .reshape((batch_size * seq_len, dims.value_dim))?;
        normalized
            .matmul(&out_weight.t()?)?
            .reshape((batch_size, seq_len, dims.hidden_size))
    }

    fn assert_layout_equivalent(
        grouped: &Tensor,
        tiled: &Tensor,
        dim: usize,
        runtime_to_grouped: &[usize],
    ) -> CandleResult<()> {
        assert_close(
            grouped,
            &grouped_from_runtime(tiled, dim, runtime_to_grouped)?,
        )
    }

    fn assert_cache_layout_equivalent(
        grouped: &GdnLayerCache,
        tiled: &GdnLayerCache,
        conv_runtime_to_grouped: &[usize],
        head_runtime_to_grouped: &[usize],
    ) -> CandleResult<()> {
        assert_layout_equivalent(
            &grouped.conv_state,
            &tiled.conv_state,
            1,
            conv_runtime_to_grouped,
        )?;
        assert_layout_equivalent(
            &grouped.recurrent_state,
            &tiled.recurrent_state,
            1,
            head_runtime_to_grouped,
        )
    }

    fn run_recurrence_case(dims: GdnDims, batch_size: usize, seq_len: usize) -> CandleResult<()> {
        let dev = Device::Cpu;
        let mixed = Tensor::from_vec(
            patterned(batch_size * seq_len * dims.conv_dim, 1, 0.08, 0.01),
            (batch_size, seq_len, dims.conv_dim),
            &dev,
        )?;
        let b = Tensor::from_vec(
            patterned(batch_size * seq_len * dims.num_v_heads, 2, 0.2, 0.1),
            (batch_size, seq_len, dims.num_v_heads),
            &dev,
        )?;
        let a = Tensor::from_vec(
            patterned(batch_size * seq_len * dims.num_v_heads, 3, 0.2, -0.05),
            (batch_size, seq_len, dims.num_v_heads),
            &dev,
        )?;
        let a_log = Tensor::from_vec(
            patterned(dims.num_v_heads, 4, 0.05, -0.2),
            (dims.num_v_heads,),
            &dev,
        )?;
        let dt_bias = Tensor::from_vec(
            patterned(dims.num_v_heads, 5, 0.1, 0.3),
            (dims.num_v_heads,),
            &dev,
        )?;
        let initial_state = Tensor::from_vec(
            patterned(
                batch_size * dims.num_v_heads * dims.head_k_dim * dims.head_v_dim,
                6,
                0.02,
                0.0,
            ),
            (
                batch_size,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            ),
            &dev,
        )?;
        let conv_state = Tensor::zeros(
            (batch_size, dims.conv_dim, dims.conv_kernel_size),
            DType::F32,
            &dev,
        )?;
        let mut fast_cache = GdnLayerCache {
            conv_state: conv_state.clone(),
            recurrent_state: initial_state.clone(),
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        };
        let fast = recurrence_cpu_from_convolved(
            &mixed,
            &b,
            &a,
            &a_log,
            &dt_bias,
            &dims,
            batch_size,
            seq_len,
            &mut fast_cache,
            DType::F32,
        )?;

        let q = mixed.narrow(D::Minus1, 0, dims.key_dim)?;
        let k = mixed.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
        let v = mixed.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;
        let q = q.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?;
        let q = expand_k_heads(&q, &dims, batch_size, seq_len)?;
        let k = expand_k_heads(&k, &dims, batch_size, seq_len)?;
        let (beta, g) = compute_beta_g(&b, &a, &a_log, &dt_bias, DType::F32)?;
        let q = l2_norm(&q, QK_NORM_EPS)?;
        let k = l2_norm(&k, QK_NORM_EPS)?;
        let mut reference_cache = GdnLayerCache {
            conv_state,
            recurrent_state: initial_state,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        };
        let reference = gated_delta_rule_recurrence(
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut reference_cache.recurrent_state,
        )?;

        assert_close(&fast, &reference)?;
        assert_close(
            &fast_cache.recurrent_state,
            &reference_cache.recurrent_state,
        )
    }

    fn causal_conv1d_update_reference(
        x: &Tensor,
        conv1d_weight: &Tensor,
        dims: &GdnDims,
        cache: &mut GdnLayerCache,
    ) -> CandleResult<Tensor> {
        let (_, seq_len, _) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.contiguous()?;
        let state_len = cache.conv_state.dim(2)?;
        let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
        let new_len = hidden_new.dim(2)?;
        cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

        let weight = conv1d_weight.squeeze(1)?.to_dtype(hidden_new.dtype())?;
        let mut conv_outputs = Vec::with_capacity(seq_len);
        let total_len = hidden_new.dim(2)?;
        for i in (total_len - seq_len)..total_len {
            let window =
                hidden_new.narrow(2, i + 1 - dims.conv_kernel_size, dims.conv_kernel_size)?;
            let out = window
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        inference_tensor::nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
    }

    #[test]
    fn key_heads_expand_in_grouped_and_tiled_order() -> CandleResult<()> {
        let dev = Device::Cpu;
        let keys = Tensor::from_vec(vec![10.0f32, 20.0], (1, 1, 2, 1), &dev)?;
        let grouped = dims_with_layout(2, 4, 1, 1, GdnVHeadLayout::Grouped);
        let tiled = dims_with_layout(2, 4, 1, 1, GdnVHeadLayout::Tiled);

        assert_eq!(
            expand_k_heads(&keys, &grouped, 1, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            [10.0, 10.0, 20.0, 20.0]
        );
        assert_eq!(
            expand_k_heads(&keys, &tiled, 1, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            [10.0, 20.0, 10.0, 20.0]
        );
        Ok(())
    }

    #[test]
    fn converter_tiled_backend_matches_grouped_prefill_and_decode() -> CandleResult<()> {
        let dev = Device::Cpu;
        let batch_size = 2;
        let prefill_len = 5;
        let mut grouped_dims = dims_with_layout(2, 6, 3, 2, GdnVHeadLayout::Grouped);
        grouped_dims.hidden_size = 7;
        let mut tiled_dims = grouped_dims;
        tiled_dims.v_head_layout = GdnVHeadLayout::Tiled;

        let head_runtime_to_grouped = converter_tiled_heads(&grouped_dims);
        let value_runtime_to_grouped =
            expand_head_map(&head_runtime_to_grouped, grouped_dims.head_v_dim);
        let mut conv_runtime_to_grouped = (0..2 * grouped_dims.key_dim).collect::<Vec<_>>();
        conv_runtime_to_grouped.extend(
            value_runtime_to_grouped
                .iter()
                .map(|&index| 2 * grouped_dims.key_dim + index),
        );
        assert_eq!(head_runtime_to_grouped, [0, 3, 1, 4, 2, 5]);
        assert_eq!(
            value_runtime_to_grouped,
            [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]
        );
        assert_eq!(conv_runtime_to_grouped.len(), grouped_dims.conv_dim);

        let conv_weight_grouped = Tensor::from_vec(
            patterned(
                grouped_dims.conv_dim * grouped_dims.conv_kernel_size,
                31,
                0.09,
                -0.01,
            ),
            (grouped_dims.conv_dim, 1, grouped_dims.conv_kernel_size),
            &dev,
        )?;
        let conv_weight_tiled =
            runtime_from_grouped(&conv_weight_grouped, 0, &conv_runtime_to_grouped)?;
        let a_log_grouped = Tensor::from_vec(
            patterned(grouped_dims.num_v_heads, 32, 0.08, -0.2),
            (grouped_dims.num_v_heads,),
            &dev,
        )?;
        let a_log_tiled = runtime_from_grouped(&a_log_grouped, 0, &head_runtime_to_grouped)?;
        let dt_bias_grouped = Tensor::from_vec(
            patterned(grouped_dims.num_v_heads, 33, 0.12, 0.3),
            (grouped_dims.num_v_heads,),
            &dev,
        )?;
        let dt_bias_tiled = runtime_from_grouped(&dt_bias_grouped, 0, &head_runtime_to_grouped)?;
        let norm_weight = Tensor::from_vec(
            patterned(grouped_dims.head_v_dim, 34, 0.15, 1.0),
            (grouped_dims.head_v_dim,),
            &dev,
        )?;
        let out_weight_grouped = Tensor::from_vec(
            patterned(
                grouped_dims.hidden_size * grouped_dims.value_dim,
                35,
                0.1,
                0.01,
            ),
            (grouped_dims.hidden_size, grouped_dims.value_dim),
            &dev,
        )?;
        let out_weight_tiled =
            runtime_from_grouped(&out_weight_grouped, 1, &value_runtime_to_grouped)?;

        let conv_state_grouped = Tensor::from_vec(
            patterned(
                batch_size * grouped_dims.conv_dim * grouped_dims.conv_kernel_size,
                36,
                0.03,
                0.0,
            ),
            (
                batch_size,
                grouped_dims.conv_dim,
                grouped_dims.conv_kernel_size,
            ),
            &dev,
        )?;
        let recurrent_state_grouped = Tensor::from_vec(
            patterned(
                batch_size
                    * grouped_dims.num_v_heads
                    * grouped_dims.head_k_dim
                    * grouped_dims.head_v_dim,
                37,
                0.025,
                0.0,
            ),
            (
                batch_size,
                grouped_dims.num_v_heads,
                grouped_dims.head_k_dim,
                grouped_dims.head_v_dim,
            ),
            &dev,
        )?;
        let mut grouped_cache = GdnLayerCache {
            conv_state: conv_state_grouped.clone(),
            recurrent_state: recurrent_state_grouped.clone(),
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        };
        let mut tiled_cache = GdnLayerCache {
            conv_state: runtime_from_grouped(&conv_state_grouped, 1, &conv_runtime_to_grouped)?,
            recurrent_state: runtime_from_grouped(
                &recurrent_state_grouped,
                1,
                &head_runtime_to_grouped,
            )?,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        };

        let prefill_mixed_grouped = Tensor::from_vec(
            patterned(
                batch_size * prefill_len * grouped_dims.conv_dim,
                38,
                0.08,
                0.01,
            ),
            (batch_size, prefill_len, grouped_dims.conv_dim),
            &dev,
        )?;
        let prefill_mixed_tiled =
            runtime_from_grouped(&prefill_mixed_grouped, 2, &conv_runtime_to_grouped)?;
        let prefill_b_grouped = Tensor::from_vec(
            patterned(
                batch_size * prefill_len * grouped_dims.num_v_heads,
                39,
                0.2,
                0.1,
            ),
            (batch_size, prefill_len, grouped_dims.num_v_heads),
            &dev,
        )?;
        let prefill_b_tiled =
            runtime_from_grouped(&prefill_b_grouped, 2, &head_runtime_to_grouped)?;
        let prefill_a_grouped = Tensor::from_vec(
            patterned(
                batch_size * prefill_len * grouped_dims.num_v_heads,
                40,
                0.18,
                -0.04,
            ),
            (batch_size, prefill_len, grouped_dims.num_v_heads),
            &dev,
        )?;
        let prefill_a_tiled =
            runtime_from_grouped(&prefill_a_grouped, 2, &head_runtime_to_grouped)?;
        let prefill_z_grouped = Tensor::from_vec(
            patterned(
                batch_size * prefill_len * grouped_dims.value_dim,
                41,
                0.14,
                0.02,
            ),
            (
                batch_size,
                prefill_len,
                grouped_dims.num_v_heads,
                grouped_dims.head_v_dim,
            ),
            &dev,
        )?;
        let prefill_z_tiled =
            runtime_from_grouped(&prefill_z_grouped, 2, &head_runtime_to_grouped)?;

        let (grouped_beta, grouped_g) = compute_beta_g(
            &prefill_b_grouped,
            &prefill_a_grouped,
            &a_log_grouped,
            &dt_bias_grouped,
            DType::F32,
        )?;
        let (tiled_beta, tiled_g) = compute_beta_g(
            &prefill_b_tiled,
            &prefill_a_tiled,
            &a_log_tiled,
            &dt_bias_tiled,
            DType::F32,
        )?;
        assert_layout_equivalent(&grouped_beta, &tiled_beta, 2, &head_runtime_to_grouped)?;
        assert_layout_equivalent(&grouped_g, &tiled_g, 2, &head_runtime_to_grouped)?;

        let (grouped_conv, grouped_recurrent) = run_backend_step(
            BackendStep {
                mixed_qkv: &prefill_mixed_grouped,
                b: &prefill_b_grouped,
                a: &prefill_a_grouped,
                a_log: &a_log_grouped,
                dt_bias: &dt_bias_grouped,
                conv_weight: &conv_weight_grouped,
                dims: &grouped_dims,
            },
            &mut grouped_cache,
            RecurrentBatchKind::Prefill,
        )?;
        let (tiled_conv, tiled_recurrent) = run_backend_step(
            BackendStep {
                mixed_qkv: &prefill_mixed_tiled,
                b: &prefill_b_tiled,
                a: &prefill_a_tiled,
                a_log: &a_log_tiled,
                dt_bias: &dt_bias_tiled,
                conv_weight: &conv_weight_tiled,
                dims: &tiled_dims,
            },
            &mut tiled_cache,
            RecurrentBatchKind::Prefill,
        )?;
        assert_layout_equivalent(&grouped_conv, &tiled_conv, 2, &conv_runtime_to_grouped)?;
        for offset in [0, grouped_dims.key_dim] {
            let grouped_keys = grouped_conv
                .narrow(2, offset, grouped_dims.key_dim)?
                .reshape((
                    batch_size,
                    prefill_len,
                    grouped_dims.num_k_heads,
                    grouped_dims.head_k_dim,
                ))?;
            let tiled_keys = tiled_conv.narrow(2, offset, tiled_dims.key_dim)?.reshape((
                batch_size,
                prefill_len,
                tiled_dims.num_k_heads,
                tiled_dims.head_k_dim,
            ))?;
            let grouped_expanded =
                expand_k_heads(&grouped_keys, &grouped_dims, batch_size, prefill_len)?;
            let tiled_expanded = expand_k_heads(&tiled_keys, &tiled_dims, batch_size, prefill_len)?;
            assert_layout_equivalent(
                &grouped_expanded,
                &tiled_expanded,
                2,
                &head_runtime_to_grouped,
            )?;
        }
        assert_layout_equivalent(
            &grouped_recurrent,
            &tiled_recurrent,
            2,
            &head_runtime_to_grouped,
        )?;
        let grouped_prefill_output = finish_backend_step(
            &grouped_recurrent,
            &prefill_z_grouped,
            &norm_weight,
            &out_weight_grouped,
            &grouped_dims,
        )?;
        let tiled_prefill_output = finish_backend_step(
            &tiled_recurrent,
            &prefill_z_tiled,
            &norm_weight,
            &out_weight_tiled,
            &tiled_dims,
        )?;
        assert_close(&grouped_prefill_output, &tiled_prefill_output)?;
        assert_cache_layout_equivalent(
            &grouped_cache,
            &tiled_cache,
            &conv_runtime_to_grouped,
            &head_runtime_to_grouped,
        )?;

        let decode_mixed_grouped = Tensor::from_vec(
            patterned(batch_size * grouped_dims.conv_dim, 42, 0.08, 0.01),
            (batch_size, 1, grouped_dims.conv_dim),
            &dev,
        )?;
        let decode_mixed_tiled =
            runtime_from_grouped(&decode_mixed_grouped, 2, &conv_runtime_to_grouped)?;
        let decode_b_grouped = Tensor::from_vec(
            patterned(batch_size * grouped_dims.num_v_heads, 43, 0.2, 0.1),
            (batch_size, 1, grouped_dims.num_v_heads),
            &dev,
        )?;
        let decode_b_tiled = runtime_from_grouped(&decode_b_grouped, 2, &head_runtime_to_grouped)?;
        let decode_a_grouped = Tensor::from_vec(
            patterned(batch_size * grouped_dims.num_v_heads, 44, 0.18, -0.04),
            (batch_size, 1, grouped_dims.num_v_heads),
            &dev,
        )?;
        let decode_a_tiled = runtime_from_grouped(&decode_a_grouped, 2, &head_runtime_to_grouped)?;
        let decode_z_grouped = Tensor::from_vec(
            patterned(batch_size * grouped_dims.value_dim, 45, 0.14, 0.02),
            (
                batch_size,
                1,
                grouped_dims.num_v_heads,
                grouped_dims.head_v_dim,
            ),
            &dev,
        )?;
        let decode_z_tiled = runtime_from_grouped(&decode_z_grouped, 2, &head_runtime_to_grouped)?;

        let (grouped_conv, grouped_recurrent) = run_backend_step(
            BackendStep {
                mixed_qkv: &decode_mixed_grouped,
                b: &decode_b_grouped,
                a: &decode_a_grouped,
                a_log: &a_log_grouped,
                dt_bias: &dt_bias_grouped,
                conv_weight: &conv_weight_grouped,
                dims: &grouped_dims,
            },
            &mut grouped_cache,
            RecurrentBatchKind::Decode,
        )?;
        let (tiled_conv, tiled_recurrent) = run_backend_step(
            BackendStep {
                mixed_qkv: &decode_mixed_tiled,
                b: &decode_b_tiled,
                a: &decode_a_tiled,
                a_log: &a_log_tiled,
                dt_bias: &dt_bias_tiled,
                conv_weight: &conv_weight_tiled,
                dims: &tiled_dims,
            },
            &mut tiled_cache,
            RecurrentBatchKind::Decode,
        )?;
        assert_layout_equivalent(&grouped_conv, &tiled_conv, 2, &conv_runtime_to_grouped)?;
        assert_layout_equivalent(
            &grouped_recurrent,
            &tiled_recurrent,
            2,
            &head_runtime_to_grouped,
        )?;
        let grouped_decode_output = finish_backend_step(
            &grouped_recurrent,
            &decode_z_grouped,
            &norm_weight,
            &out_weight_grouped,
            &grouped_dims,
        )?;
        let tiled_decode_output = finish_backend_step(
            &tiled_recurrent,
            &decode_z_tiled,
            &norm_weight,
            &out_weight_tiled,
            &tiled_dims,
        )?;
        assert_close(&grouped_decode_output, &tiled_decode_output)?;
        assert_cache_layout_equivalent(
            &grouped_cache,
            &tiled_cache,
            &conv_runtime_to_grouped,
            &head_runtime_to_grouped,
        )
    }

    // an F32 model on a GPU reaches the conv in f32, which the CUDA kernel does not instantiate
    #[cfg(feature = "cuda")]
    #[test]
    fn causal_conv1d_on_cuda_in_f32_matches_the_cpu() -> CandleResult<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let dims = dims(2, 4, 5, 3);
        let batch_size = 2;
        let weight = Tensor::from_vec(
            patterned(dims.conv_dim * dims.conv_kernel_size, 8, 0.05, -0.01),
            (dims.conv_dim, 1, dims.conv_kernel_size),
            &Device::Cpu,
        )?;
        let initial_state = Tensor::from_vec(
            patterned(
                batch_size * dims.conv_dim * dims.conv_kernel_size,
                9,
                0.03,
                0.0,
            ),
            (batch_size, dims.conv_dim, dims.conv_kernel_size),
            &Device::Cpu,
        )?;
        let cache_on = |device: &Device| -> CandleResult<GdnLayerCache> {
            Ok(GdnLayerCache {
                conv_state: initial_state.to_device(device)?,
                recurrent_state: Tensor::zeros(
                    (
                        batch_size,
                        dims.num_v_heads,
                        dims.head_k_dim,
                        dims.head_v_dim,
                    ),
                    DType::F32,
                    device,
                )?,
                state_layout: RecurrentStateLayout::GdnKeyMajor,
                slots: None,
                pending_transitions: None,
                deferred_state: None,
                pool_views: None,
            })
        };
        // a prefill, then a decode step from the state it leaves
        for (seq_len, kind) in [
            (6, RecurrentBatchKind::Prefill),
            (1, RecurrentBatchKind::Decode),
        ] {
            let x = Tensor::from_vec(
                patterned(batch_size * seq_len * dims.conv_dim, 7, 0.08, 0.01),
                (batch_size, seq_len, dims.conv_dim),
                &Device::Cpu,
            )?;
            let mut cpu_cache = cache_on(&Device::Cpu)?;
            let mut cuda_cache = cache_on(&cuda)?;
            let expected = causal_conv1d(&x, &weight, &dims, &mut cpu_cache, kind)?;
            let actual = causal_conv1d(
                &x.to_device(&cuda)?,
                &weight.to_device(&cuda)?,
                &dims,
                &mut cuda_cache,
                kind,
            )?;
            assert_close(&actual.to_device(&Device::Cpu)?, &expected)?;
            assert_close(
                &cuda_cache.conv_state.to_device(&Device::Cpu)?,
                &cpu_cache.conv_state,
            )?;
        }
        Ok(())
    }

    #[test]
    fn causal_conv1d_update_cpu_matches_tensor_path() -> CandleResult<()> {
        let dev = Device::Cpu;
        let dims = dims(2, 4, 5, 3);
        let batch_size = 2;
        let x = Tensor::from_vec(
            patterned(batch_size * dims.conv_dim, 7, 0.08, 0.01),
            (batch_size, 1, dims.conv_dim),
            &dev,
        )?;
        let weight = Tensor::from_vec(
            patterned(dims.conv_dim * dims.conv_kernel_size, 8, 0.05, -0.01),
            (dims.conv_dim, 1, dims.conv_kernel_size),
            &dev,
        )?;
        let initial_state = Tensor::from_vec(
            patterned(
                batch_size * dims.conv_dim * dims.conv_kernel_size,
                9,
                0.03,
                0.0,
            ),
            (batch_size, dims.conv_dim, dims.conv_kernel_size),
            &dev,
        )?;
        let recurrent_state = Tensor::zeros(
            (
                batch_size,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            ),
            DType::F32,
            &dev,
        )?;
        let mut fast_cache = GdnLayerCache {
            conv_state: initial_state.clone(),
            recurrent_state: recurrent_state.clone(),
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        };
        let mut reference_cache = GdnLayerCache {
            conv_state: initial_state,
            recurrent_state,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        };

        let fast = causal_conv1d_update_cpu(&x, &weight, &dims, &mut fast_cache)?;
        let reference = causal_conv1d_update_reference(&x, &weight, &dims, &mut reference_cache)?;

        assert_close(&fast, &reference)?;
        assert_close(&fast_cache.conv_state, &reference_cache.conv_state)
    }

    // a decode step on views of one pool row writes that row in place and matches the same step on a copy of it
    #[test]
    fn pool_row_views_update_in_place() -> CandleResult<()> {
        const ROWS: usize = 3;
        const ROW: usize = 1;
        let dev = Device::Cpu;
        let dims = dims(2, 4, 5, 3);
        let x = Tensor::from_vec(
            patterned(dims.conv_dim, 7, 0.08, 0.01),
            (1, 1, dims.conv_dim),
            &dev,
        )?;
        let weight = Tensor::from_vec(
            patterned(dims.conv_dim * dims.conv_kernel_size, 8, 0.05, -0.01),
            (dims.conv_dim, 1, dims.conv_kernel_size),
            &dev,
        )?;
        let gates = |seed| {
            Tensor::from_vec(
                patterned(dims.num_v_heads, seed, 0.2, 0.1),
                (1, 1, dims.num_v_heads),
                &dev,
            )
        };
        let (b, a) = (gates(2)?, gates(3)?);
        let a_log = Tensor::from_vec(
            patterned(dims.num_v_heads, 4, 0.05, -0.2),
            (dims.num_v_heads,),
            &dev,
        )?;
        let dt_bias = Tensor::from_vec(
            patterned(dims.num_v_heads, 5, 0.1, 0.3),
            (dims.num_v_heads,),
            &dev,
        )?;
        let conv_pool = Tensor::from_vec(
            patterned(ROWS * dims.conv_dim * dims.conv_kernel_size, 9, 0.03, 0.0),
            (ROWS, dims.conv_dim, dims.conv_kernel_size),
            &dev,
        )?;
        let state_pool = Tensor::from_vec(
            patterned(
                ROWS * dims.num_v_heads * dims.head_k_dim * dims.head_v_dim,
                6,
                0.02,
                0.0,
            ),
            (ROWS, dims.num_v_heads, dims.head_k_dim, dims.head_v_dim),
            &dev,
        )?;
        let (conv_before, state_before) = (conv_pool.copy()?, state_pool.copy()?);
        let step = |cache: &mut GdnLayerCache| -> CandleResult<Tensor> {
            let convolved = causal_conv1d_update_cpu(&x, &weight, &dims, cache)?;
            recurrence_cpu_from_convolved(
                &convolved,
                &b,
                &a,
                &a_log,
                &dt_bias,
                &dims,
                1,
                1,
                cache,
                DType::F32,
            )
        };

        let mut copied = GdnLayerCache::gathered(
            conv_pool.narrow(0, ROW, 1)?.copy()?,
            state_pool.narrow(0, ROW, 1)?.copy()?,
            RecurrentStateLayout::GdnKeyMajor,
        );
        let expected = step(&mut copied)?;
        let mut views = GdnLayerCache::gathered(
            conv_pool.narrow(0, ROW, 1)?,
            state_pool.narrow(0, ROW, 1)?,
            RecurrentStateLayout::GdnKeyMajor,
        );
        let ids = [views.conv_state.id(), views.recurrent_state.id()];
        views.pool_views = Some(ids);
        let actual = step(&mut views)?;

        assert_close(&actual, &expected)?;
        assert_eq!([views.conv_state.id(), views.recurrent_state.id()], ids);
        assert_close(&conv_pool.narrow(0, ROW, 1)?, &copied.conv_state)?;
        assert_close(&state_pool.narrow(0, ROW, 1)?, &copied.recurrent_state)?;
        for row in [0, 2] {
            assert_close(
                &conv_pool.narrow(0, row, 1)?,
                &conv_before.narrow(0, row, 1)?,
            )?;
            assert_close(
                &state_pool.narrow(0, row, 1)?,
                &state_before.narrow(0, row, 1)?,
            )?;
        }
        Ok(())
    }

    #[test]
    fn causal_conv1d_prefill_continues_from_existing_state() -> CandleResult<()> {
        let dev = Device::Cpu;
        let dims = dims(2, 4, 5, 3);
        let batch_size = 2;
        let seq_len = 7;
        let split = 3;
        let x = Tensor::from_vec(
            patterned(batch_size * seq_len * dims.conv_dim, 10, 0.08, 0.01),
            (batch_size, seq_len, dims.conv_dim),
            &dev,
        )?;
        let weight = Tensor::from_vec(
            patterned(dims.conv_dim * dims.conv_kernel_size, 11, 0.05, -0.01),
            (dims.conv_dim, 1, dims.conv_kernel_size),
            &dev,
        )?;
        let conv_state = Tensor::from_vec(
            patterned(
                batch_size * dims.conv_dim * dims.conv_kernel_size,
                12,
                0.03,
                0.0,
            ),
            (batch_size, dims.conv_dim, dims.conv_kernel_size),
            &dev,
        )?;
        let recurrent_state = Tensor::zeros(
            (
                batch_size,
                dims.num_v_heads,
                dims.head_k_dim,
                dims.head_v_dim,
            ),
            DType::F32,
            &dev,
        )?;
        let mut one_shot_cache = GdnLayerCache {
            conv_state: conv_state.clone(),
            recurrent_state: recurrent_state.clone(),
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        };
        let mut stepped_cache = GdnLayerCache {
            conv_state: conv_state.clone(),
            recurrent_state: recurrent_state.clone(),
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        };
        let mut chunked_cache = GdnLayerCache {
            conv_state,
            recurrent_state,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: None,
            pending_transitions: None,
            deferred_state: None,
            pool_views: None,
        };

        let one_shot = causal_conv1d_full(&x, &weight, &dims, &mut one_shot_cache)?;
        let first =
            causal_conv1d_full(&x.narrow(1, 0, split)?, &weight, &dims, &mut chunked_cache)?;
        let second = causal_conv1d_full(
            &x.narrow(1, split, seq_len - split)?,
            &weight,
            &dims,
            &mut chunked_cache,
        )?;

        // the decode update is a separate implementation, one token at a time
        let stepped = (0..seq_len)
            .map(|t| {
                causal_conv1d_update_cpu(&x.narrow(1, t, 1)?, &weight, &dims, &mut stepped_cache)
            })
            .collect::<CandleResult<Vec<_>>>()?;

        assert_close(&one_shot, &Tensor::cat(&[first, second], 1)?)?;
        assert_close(&one_shot, &Tensor::cat(&stepped, 1)?)?;
        assert_close(&one_shot_cache.conv_state, &stepped_cache.conv_state)?;
        assert_close(&one_shot_cache.conv_state, &chunked_cache.conv_state)
    }

    #[test]
    fn recurrence_cpu_matches_tensor_path() -> CandleResult<()> {
        for seq_len in [1, 5] {
            run_recurrence_case(dims(2, 4, 5, 3), 2, seq_len)?;
            run_recurrence_case(dims(3, 3, 4, 2), 1, seq_len)?;
            run_recurrence_case(
                dims_with_layout(2, 4, 5, 3, GdnVHeadLayout::Tiled),
                2,
                seq_len,
            )?;
        }
        Ok(())
    }
}
