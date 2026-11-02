// prompts this long take the chunked SSD matmuls instead of the sequential scan
#[cfg(feature = "cuda")]
const SSD_MIN_TOKENS: usize = 128;
// chunks of one sequence a pass of the SSD kernels covers, bounding its scratch (~40 MB a chunk at 64 heads)
#[cfg(feature = "cuda")]
const SSD_WINDOW_CHUNKS: usize = 2;

/// Mamba-2's selective scan on CUDA (a prompt of `SSD_MIN_TOKENS` or more with no `slots` as chunked SSD), over the
/// inputs [`crate::ops::SelectiveScan`] describes (f32, B and C by group). `state` is `(batch, heads, head_dim,
/// d_state)`, or with `slots` (u32, one per batch row) a recurrent pool whose row `slots[b]` it updates in place, a
/// slot past `i32::MAX` marking a padding row that leaves the pool and scores 0. Returns `y` `(batch, seq_len, heads,
/// head_dim)`, f32.
#[cfg(feature = "cuda")]
pub fn selective_scan_cuda(
    scan: crate::ops::SelectiveScan<'_>,
    state: &mut inference_tensor::Tensor,
    slots: Option<&inference_tensor::Tensor>,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;

    let [x, dt, a, b, c, d, dt_bias] = [
        scan.x,
        scan.dt,
        scan.a,
        scan.b,
        scan.c,
        scan.d,
        scan.dt_bias,
    ]
    .map(inference_tensor::Tensor::contiguous);
    let (x, dt, a, b, c, d, dt_bias) = (x?, dt?, a?, b?, c?, d?, dt_bias?);
    let (batch_size, seq_len, n_heads, head_dim) = x.dims4()?;
    let (_, _, n_groups, d_state) = b.dims4()?;
    if !n_heads.is_multiple_of(n_groups) {
        inference_tensor::bail!(
            "selective_scan_cuda: {n_heads} heads do not split over {n_groups} groups"
        );
    }
    let dev = x.device().as_cuda_device()?;

    macro_rules! get_ptr {
        ($tensor:expr, $ty:ty) => {{
            let (s, l) = $tensor.storage_and_layout();
            let s = match &*s {
                inference_tensor::Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?,
                _ => inference_tensor::bail!("selective_scan_cuda: tensor must be on CUDA"),
            };
            s.slice(l.start_offset()..).device_ptr(s.stream()).0
        }};
    }

    let state_ptr = get_ptr!(state, f32) as *mut f32;
    let slots = slots
        .map(inference_tensor::Tensor::contiguous)
        .transpose()?;
    let slot_ptr = match &slots {
        Some(slots) => get_ptr!(slots, u32) as *const i32,
        None => std::ptr::null(),
    };

    let y_buf = unsafe { dev.alloc::<f32>(batch_size * seq_len * n_heads * head_dim) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    if slot_ptr.is_null() && seq_len >= SSD_MIN_TOKENS {
        let shape = SsdShape {
            batch_size,
            seq_len,
            n_heads,
            n_groups,
            head_dim,
            d_state,
        };
        let inputs = [
            get_ptr!(x, f32),
            get_ptr!(dt, f32),
            get_ptr!(a, f32),
            get_ptr!(b, f32),
            get_ptr!(c, f32),
            get_ptr!(d, f32),
            get_ptr!(dt_bias, f32),
        ]
        .map(|ptr| ptr as *const f32);
        let y_ptr = y_buf.device_ptr(y_buf.stream()).0 as *mut f32;
        ssd_scan(
            dev,
            &shape,
            inputs,
            state_ptr,
            y_ptr,
            [scan.dt_min, scan.dt_max],
        )?;
    } else {
        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            crate::cuda::ffi::selective_scan_cuda(
                get_ptr!(x, f32) as *const f32,
                get_ptr!(dt, f32) as *const f32,
                get_ptr!(a, f32) as *const f32,
                get_ptr!(b, f32) as *const f32,
                get_ptr!(c, f32) as *const f32,
                get_ptr!(d, f32) as *const f32,
                get_ptr!(dt_bias, f32) as *const f32,
                state_ptr,
                y_buf.device_ptr(y_buf.stream()).0 as *mut f32,
                slot_ptr,
                batch_size as i32,
                n_heads as i32,
                n_groups as i32,
                head_dim as i32,
                d_state as i32,
                seq_len as i32,
                scan.dt_min,
                scan.dt_max,
                stream,
            );
        }
    }

    let y_storage = inference_tensor::CudaStorage::wrap_cuda_slice(y_buf, dev.clone());
    Ok(inference_tensor::Tensor::from((
        inference_tensor::Storage::Cuda(y_storage),
        (batch_size, seq_len, n_heads, head_dim),
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn selective_scan_cuda(
    _scan: crate::ops::SelectiveScan<'_>,
    _state: &mut inference_tensor::Tensor,
    _slots: Option<&inference_tensor::Tensor>,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    inference_tensor::bail!("selective_scan_cuda requires the cuda feature")
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct SsdParams {
    inputs: [*const f32; 7],
    state: *mut f32,
    y: *mut f32,
    scratch: [*mut f32; 12],
    seq_len: i32,
    first_chunk: i32,
    n_chunks: i32,
    n_heads: i32,
    n_groups: i32,
    head_dim: i32,
    d_state: i32,
    dt_min: f32,
    dt_max: f32,
}

#[cfg(feature = "cuda")]
struct SsdShape {
    batch_size: usize,
    seq_len: usize,
    n_heads: usize,
    n_groups: usize,
    head_dim: usize,
    d_state: usize,
}

// One column-major strided-batched F32 GEMM, `[a, b, c]` as (pointer, leading dim, batch stride).
#[cfg(feature = "cuda")]
struct Gemm {
    trans: [bool; 2],
    mnk: [usize; 3],
    operands: [(*const f32, usize, usize); 3],
    batch: usize,
}

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn ssd_gemm(
    blas: &inference_tensor::cuda_backend::cudarc::cublas::CudaBlas,
    g: Gemm,
) -> inference_tensor::Result<()> {
    use inference_tensor::cuda_backend::cudarc::cublas::{result, sys};
    let op = |t: bool| {
        if t {
            sys::cublasOperation_t::CUBLAS_OP_T
        } else {
            sys::cublasOperation_t::CUBLAS_OP_N
        }
    };
    let [m, n, k] = g.mnk.map(|v| v as i32);
    let [(a, lda, sa), (b, ldb, sb), (c, ldc, sc)] = g.operands;
    let (alpha, beta) = (1.0f32, 0.0f32);
    let f32_ty = sys::cudaDataType_t::CUDA_R_32F;
    unsafe {
        result::gemm_strided_batched_ex(
            *blas.handle(),
            op(g.trans[0]),
            op(g.trans[1]),
            m,
            n,
            k,
            (&raw const alpha).cast(),
            a.cast(),
            f32_ty,
            lda as i32,
            sa as i64,
            b.cast(),
            f32_ty,
            ldb as i32,
            sb as i64,
            (&raw const beta).cast(),
            c.cast_mut().cast(),
            f32_ty,
            ldc as i32,
            sc as i64,
            g.batch as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .map_err(|e| inference_tensor::Error::Cuda(Box::new(e)))
}

// Mamba-2's prompt scan as chunked SSD (the layouts are in ssm.cu): each sequence in windows of chunks, its state
// carried from window to window in place.
#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn ssd_scan(
    dev: &inference_tensor::CudaDevice,
    shape: &SsdShape,
    inputs: [*const f32; 7],
    state: *mut f32,
    y: *mut f32,
    [dt_min, dt_max]: [f32; 2],
) -> inference_tensor::Result<()> {
    use crate::cuda::ffi;
    use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;

    let &SsdShape {
        batch_size,
        seq_len,
        n_heads: h,
        n_groups: g,
        head_dim: p,
        d_state: n,
    } = shape;
    let l = unsafe { ffi::ssd_chunk_length() } as usize;
    let chunks = seq_len.div_ceil(l);
    let window = chunks.min(SSD_WINDOW_CHUNKS);
    let hp = h * p;
    let hpn = hp * n;
    let per_chunk = [
        h * l,
        h,
        l * hp,
        l * hp,
        g * l * n,
        g * l * n,
        g * l * l,
        h * l * l,
        hpn,
        hpn,
        l * hp,
        l * hp,
    ];
    let buf = unsafe { dev.alloc::<f32>(per_chunk.iter().sum::<usize>() * window) }?;
    let base = buf.device_ptr(buf.stream()).0 as *mut f32;
    let mut scratch = [std::ptr::null_mut(); 12];
    let mut at = 0;
    for (s, size) in scratch.iter_mut().zip(per_chunk) {
        *s = unsafe { base.add(at) };
        at += size * window;
    }
    let [_, _, xdt, xw, bg, cg, cb, m, ds, sprev, yi, ys] = scratch.map(|s| s.cast_const());
    let blas = dev.cublas_handle();
    let stream = dev.cuda_stream().cu_stream() as i64;
    let hpg = h / g;

    for row in 0..batch_size {
        let [x, dt, a, b, c, d, dt_bias] = inputs;
        let tok = row * seq_len;
        let row_inputs = unsafe {
            [
                x.add(tok * hp),
                dt.add(tok * h),
                a,
                b.add(tok * g * n),
                c.add(tok * g * n),
                d,
                dt_bias,
            ]
        };
        for first in (0..chunks).step_by(window) {
            let k_n = window.min(chunks - first);
            let params = SsdParams {
                inputs: row_inputs,
                state: unsafe { state.add(row * hpn) },
                y: unsafe { y.add(tok * hp) },
                scratch,
                seq_len: seq_len as i32,
                first_chunk: first as i32,
                n_chunks: k_n as i32,
                n_heads: h as i32,
                n_groups: g as i32,
                head_dim: p as i32,
                d_state: n as i32,
                dt_min,
                dt_max,
            };
            let params_ptr = (&raw const params).cast();
            unsafe { ffi::ssd_prepare_cuda(params_ptr, stream) };
            // C.B per group: cb[t_in][t_out]
            ssd_gemm(
                &blas,
                Gemm {
                    trans: [true, false],
                    mnk: [l, l, n],
                    operands: [(cg, n, l * n), (bg, n, l * n), (cb, l, l * l)],
                    batch: k_n * g,
                },
            )?;
            unsafe { ffi::ssd_mask_cuda(params_ptr, stream) };
            for k in 0..k_n {
                let (xk, yk) = (k * l * hp, k * l * hp);
                // the chunk's own outputs, per head: yi = xdt m^T
                ssd_gemm(
                    &blas,
                    Gemm {
                        trans: [false, true],
                        mnk: [p, l, l],
                        operands: [
                            (unsafe { xdt.add(xk) }, hp, p),
                            (unsafe { m.add(k * h * l * l) }, l, l * l),
                            (unsafe { yi.add(yk) }, hp, p),
                        ],
                        batch: h,
                    },
                )?;
                // what the chunk adds to the state, per group: ds = bg xw^T
                ssd_gemm(
                    &blas,
                    Gemm {
                        trans: [false, true],
                        mnk: [n, hpg * p, l],
                        operands: [
                            (unsafe { bg.add(k * g * l * n) }, n, l * n),
                            (unsafe { xw.add(xk) }, hp, hpg * p),
                            (unsafe { ds.add(k * hpn) }, n, hpg * p * n),
                        ],
                        batch: g,
                    },
                )?;
            }
            unsafe { ffi::ssd_state_pass_cuda(params_ptr, stream) };
            for k in 0..k_n {
                // the carried state's outputs, per group: ys = sprev^T cg
                ssd_gemm(
                    &blas,
                    Gemm {
                        trans: [true, false],
                        mnk: [hpg * p, l, n],
                        operands: [
                            (unsafe { sprev.add(k * hpn) }, n, hpg * p * n),
                            (unsafe { cg.add(k * g * l * n) }, n, l * n),
                            (unsafe { ys.add(k * l * hp) }, hp, hpg * p),
                        ],
                        batch: g,
                    },
                )?;
            }
            unsafe { ffi::ssd_finish_cuda(params_ptr, stream) };
        }
    }
    Ok(())
}
