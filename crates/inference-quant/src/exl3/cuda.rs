//! An EXL3 linear kept as its trellis on a CUDA device, multiplied by exllamav3's kernels.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use half::f16;
use inference_tensor::{CudaStorage, DType, Device, Result, Shape, Storage, Tensor};

use super::{Exl3Codebook, dense_weight, ffi};
use crate::utils::slice_ptr;
use crate::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, UnquantLinear,
    plan_weight_isq,
};

#[derive(Debug)]
pub struct Exl3Layer {
    // `(in / 16, out / 16, words)` i16 words on the device, with f16 `suh` (in) and `svh` (out)
    trellis: Tensor,
    suh: Tensor,
    svh: Tensor,
    codebook: Exl3Codebook,
    bias: Option<Tensor>,
    // the model's dtype, which a dense reconstruction takes
    dtype: DType,
    // the m buckets whose first launch (which may time the kernel shapes) has run outside a capture
    tuned: AtomicU8,
    // as the first of a fused projection group (q of q, k, v): the group's device tables, built on first use
    group: std::sync::Mutex<Option<GroupTables>>,
    group_tuned: AtomicU8,
}

// the device tables of one input through several EXL3 layers: their trellis, suh and svh addresses and widths
#[derive(Debug)]
struct GroupTables {
    // the other layers' addresses, so a table built for one group is not taken for another
    members: Vec<usize>,
    // each slice's trellis address (at its first column), each layer's suh, each slice's svh (at its first column)
    tables: Tensor,
    // per slice: its width, its layer's width, its layer, its first column, its layer's output offset in columns
    ints: Tensor,
    widths: Vec<usize>,
    slices: usize,
    slice_width: usize,
}

// a fused projection group takes up to this many rows (decode, small batches); larger ones run layer by layer
const GROUP_MAX_ROWS: usize = 16;
// mul1 tensors up to this rate (a half-integer one counting as the next whole) take the int8 GEMV alone, faster
// than any fused launch
const INT8_MAX_K: usize = 5;

// the launcher's autotune buckets: powers of two up to 16 rows, then 16-row steps capped at 64
fn tune_bucket(m: usize) -> u8 {
    match m.max(2) {
        m @ ..=16 => m.next_power_of_two().trailing_zeros() as u8,
        m => 4 + m.div_ceil(16).min(4) as u8 - 1,
    }
}

impl Exl3Layer {
    pub fn new_on_device(
        trellis: Tensor,
        suh: Tensor,
        svh: Tensor,
        codebook: Exl3Codebook,
        bias: Option<Tensor>,
        dtype: DType,
    ) -> Result<Self> {
        ffi::ensure_loaded().map_err(inference_tensor::Error::msg)?;
        Ok(Self {
            trellis: trellis.to_dtype(DType::I16)?.contiguous()?,
            suh: suh.to_dtype(DType::F16)?.contiguous()?,
            svh: svh.to_dtype(DType::F16)?.contiguous()?,
            codebook,
            bias,
            dtype,
            tuned: AtomicU8::new(0),
            group: std::sync::Mutex::new(None),
            group_tuned: AtomicU8::new(0),
        })
    }

    fn dims(&self) -> (usize, usize, usize) {
        let d = self.trellis.dims();
        (d[0] * 16, d[1] * 16, d[2])
    }

    /// `xs` through this layer and `others` in one launch (q, k and v: one input, their own rotations and widths),
    /// each output shaped as its layer's own; None where the layers or the batch do not fuse.
    pub fn fused_group(&self, others: &[&Self], xs: &Tensor) -> Result<Option<Vec<Tensor>>> {
        let Device::Cuda(dev) = xs.device() else {
            return Ok(None);
        };
        let layers = std::iter::once(self)
            .chain(others.iter().copied())
            .collect::<Vec<_>>();
        let (k, _, words) = self.dims();
        let rows = xs.elem_count() / k;
        let int8 = self.codebook == Exl3Codebook::Mul1 && words.div_ceil(16) <= INT8_MAX_K;
        if int8
            || layers.iter().any(|l| {
                let (lk, _, lw) = l.dims();
                lk != k || lw != words || l.codebook != self.codebook || l.dtype != self.dtype
            })
        {
            return Ok(None);
        }
        let capturing = dev.cuda_stream().capture_status().map_err(inference_tensor::Error::wrap)?
            != inference_tensor::cuda::cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let members = others
            .iter()
            .map(|l| *l as *const Self as usize)
            .collect::<Vec<_>>();
        let mut group = self.group.lock().unwrap();
        if group.as_ref().is_none_or(|g| g.members != members) {
            // building copies the tables to the device, which a capture must not record
            if capturing {
                return Ok(None);
            }
            let widths = layers.iter().map(|l| l.dims().1).collect::<Vec<_>>();
            // equal slices of the narrowest layer's width spread the layers evenly over the grid
            let slice_width = widths.iter().copied().min().unwrap_or(0);
            if widths.iter().any(|w| w % slice_width != 0) || slice_width % 128 != 0 {
                return Ok(None);
            }
            let (mut trellis, mut svh) = (Vec::new(), Vec::new());
            let mut ints = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut prefix = 0;
            for (src, (l, &width)) in layers.iter().zip(&widths).enumerate() {
                let (t, v) = (
                    device_address::<i16>(&l.trellis)?,
                    device_address::<f16>(&l.svh)?,
                );
                for col in (0..width).step_by(slice_width) {
                    // a tile column is `words` uint16 apart within a row of tiles
                    trellis.push(t + (col / 16 * words * 2) as i64);
                    svh.push(v + (col * 2) as i64);
                    for (list, value) in ints.iter_mut().zip([slice_width, width, src, col, prefix])
                    {
                        list.push(value as i32);
                    }
                }
                prefix += width;
            }
            let slices = trellis.len();
            let mut addresses = trellis;
            for l in &layers {
                addresses.push(device_address::<f16>(&l.suh)?);
            }
            addresses.extend(svh);
            let len = addresses.len();
            *group = Some(GroupTables {
                members,
                tables: Tensor::from_vec(addresses, len, xs.device())?,
                ints: Tensor::from_vec(ints.concat(), 5 * slices, xs.device())?,
                widths,
                slices,
                slice_width,
            });
        }
        // built at any size, so a prefill builds what the decode graphs, captured later, launch with
        if rows > GROUP_MAX_ROWS {
            return Ok(None);
        }
        let group = group.as_ref().unwrap();
        let x = xs.reshape((rows, k))?.to_dtype(DType::F16)?.contiguous()?;
        let outs = if self.dtype == DType::F16 {
            self.group_matmul::<f16>(group, &x, rows, false)?
        } else {
            self.group_matmul::<f32>(group, &x, rows, true)?
        };
        let lead = &xs.dims()[..xs.rank() - 1];
        let mut start = 0;
        let mut shaped = Vec::with_capacity(layers.len());
        for (l, &width) in layers.iter().zip(&group.widths) {
            let len = rows * width;
            let mut y = outs.narrow(0, start, len)?.to_dtype(xs.dtype())?;
            start += len;
            let mut dims = lead.to_vec();
            dims.push(width);
            y = y.reshape(dims)?;
            if let Some(bias) = &l.bias {
                y = y.broadcast_add(&bias.to_dtype(xs.dtype())?)?;
            }
            shaped.push(y);
        }
        Ok(Some(shaped))
    }

    // the group's outputs end to end (each `rows x width`), in `T`
    fn group_matmul<T>(
        &self,
        group: &GroupTables,
        x: &Tensor,
        rows: usize,
        c_fp32: bool,
    ) -> Result<Tensor>
    where
        T: inference_tensor::cuda_backend::CudaDType
            + inference_tensor::cuda::cudarc::driver::DeviceRepr,
    {
        let Device::Cuda(dev) = x.device() else {
            inference_tensor::bail!("an EXL3 layer multiplies on CUDA");
        };
        let (k, _, words) = self.dims();
        let (sources, slices) = (group.widths.len(), group.slices);
        let total = group.widths.iter().sum::<usize>();
        let (x_s, x_l) = x.storage_and_layout();
        let (t_s, t_l) = group.tables.storage_and_layout();
        let (i_s, i_l) = group.ints.storage_and_layout();
        let (Storage::Cuda(x_c), Storage::Cuda(t_c), Storage::Cuda(i_c)) = (&*x_s, &*t_s, &*i_s)
        else {
            inference_tensor::bail!("an EXL3 layer's tensors live on CUDA");
        };
        // SAFETY: the kernels write each layer's rotated input before reading it, every output, and every address
        let a_had = unsafe { dev.alloc::<f16>(sources * rows * k)? };
        // SAFETY: as above
        let out = unsafe { dev.alloc::<T>(rows * total)? };
        // SAFETY: as above
        let c_ptrs = unsafe { dev.alloc::<u64>(slices)? };
        let (a, _a) = slice_ptr(x_c.as_cuda_slice::<f16>()?, x_l.start_offset());
        let tables = t_c.as_cuda_slice::<i64>()?;
        let (b_list, _b) = slice_ptr(tables, t_l.start_offset());
        let (suh_list, _u) = slice_ptr(tables, t_l.start_offset() + slices);
        let (svh_list, _v) = slice_ptr(tables, t_l.start_offset() + slices + sources);
        let (ints, _i) = slice_ptr(i_c.as_cuda_slice::<i32>()?, i_l.start_offset());
        let (had, _h) = slice_ptr(&a_had, 0);
        let (c, _c) = slice_ptr(&out, 0);
        let (ptrs, _p) = slice_ptr(&c_ptrs, 0);
        let bucket = 1u8 << tune_bucket(rows);
        // a first launch may time candidates (synchronizing, allocating), which no other thread's capture may see
        let scope = (self.group_tuned.load(Ordering::Relaxed) & bucket == 0)
            .then(inference_tensor::cuda_backend::context_wide_scope)
            .flatten();
        // SAFETY: the pointers are the live device buffers above (A rows x k, the tables 2 x slices + sources, the
        // ints 5 x slices, the scratch sources x rows x k, C rows x total, c_ptrs slices), and the tables address
        // the group's own tensors
        let ran = unsafe {
            ffi::launch_exl3_mgemm_sliced(
                a as *const f16,
                b_list as *const u64,
                suh_list as *const u64,
                svh_list as *const u64,
                ints as *const i32,
                sources as i32,
                slices as i32,
                group.slice_width as i32,
                c as *mut std::ffi::c_void,
                ptrs as *mut u64,
                rows as i32,
                k as i32,
                words as i32,
                self.codebook as i32,
                c_fp32,
                had as *mut f16,
                dev.cuda_stream().cu_stream(),
            )
        };
        if ran < 0 {
            inference_tensor::bail!(
                "no EXL3 kernel fits {slices} fused {k}-wide slices at {words} words a tile"
            );
        }
        if scope.is_some() {
            self.group_tuned.fetch_or(bucket, Ordering::Relaxed);
        }
        drop((scope, _a, _b, _u, _v, _i, _h, _c, _p));
        Ok(Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            Shape::from(rows * total),
        )))
    }

    // `x` (`m x in`, f16, contiguous) times the weight, `m x out` in f16 for an f16 model (as the kernels write it),
    // else f32
    fn matmul(&self, x: &Tensor) -> Result<Tensor> {
        if self.dtype == DType::F16 {
            self.matmul_into::<f16>(x, false)
        } else {
            self.matmul_into::<f32>(x, true)
        }
    }

    fn matmul_into<T>(&self, x: &Tensor, c_fp32: bool) -> Result<Tensor>
    where
        T: inference_tensor::cuda_backend::CudaDType
            + inference_tensor::cuda::cudarc::driver::DeviceRepr,
    {
        let Device::Cuda(dev) = x.device() else {
            inference_tensor::bail!("an EXL3 layer multiplies on CUDA");
        };
        let (k, n, words) = self.dims();
        let m = x.dim(0)?;
        let (x_s, x_l) = x.storage_and_layout();
        let (t_s, t_l) = self.trellis.storage_and_layout();
        let (u_s, u_l) = self.suh.storage_and_layout();
        let (v_s, v_l) = self.svh.storage_and_layout();
        let (Storage::Cuda(x_c), Storage::Cuda(t_c), Storage::Cuda(u_c), Storage::Cuda(v_c)) =
            (&*x_s, &*t_s, &*u_s, &*v_s)
        else {
            inference_tensor::bail!("an EXL3 layer's tensors live on CUDA");
        };
        // SAFETY: the kernels write all of the input's rotation before reading it, and all of C
        let a_had = unsafe { dev.alloc::<f16>(m * k)? };
        // SAFETY: as above
        let out = unsafe { dev.alloc::<T>(m * n)? };
        let (a, _a) = slice_ptr(x_c.as_cuda_slice::<f16>()?, x_l.start_offset());
        let (b, _b) = slice_ptr(t_c.as_cuda_slice::<i16>()?, t_l.start_offset());
        let (suh, _u) = slice_ptr(u_c.as_cuda_slice::<f16>()?, u_l.start_offset());
        let (svh, _v) = slice_ptr(v_c.as_cuda_slice::<f16>()?, v_l.start_offset());
        let (had, _h) = slice_ptr(&a_had, 0);
        let (c, _c) = slice_ptr(&out, 0);
        let bucket = 1u8 << tune_bucket(m);
        // a first launch may time candidates (synchronizing, allocating), which no other thread's capture may see
        let scope = (self.tuned.load(Ordering::Relaxed) & bucket == 0)
            .then(inference_tensor::cuda_backend::context_wide_scope)
            .flatten();
        // SAFETY: the pointers are the live device buffers above, sized as the kernel reads them (A m x k, the trellis
        // k/16 x n/16 x words, suh k, svh n, the scratch m x k, C m x n f32), all on this device's stream
        let ran = unsafe {
            ffi::launch_exl3_gemm(
                a as *const f16,
                b as *const u16,
                c as *mut std::ffi::c_void,
                m as i32,
                k as i32,
                n as i32,
                words as i32,
                self.codebook as i32,
                c_fp32,
                suh as *const f16,
                had as *mut f16,
                svh as *const f16,
                dev.cuda_stream().cu_stream(),
            )
        };
        if ran < 0 {
            inference_tensor::bail!("no EXL3 kernel fits {m} x {k} -> {n} at {words} words a tile");
        }
        if scope.is_some() {
            self.tuned.fetch_or(bucket, Ordering::Relaxed);
        }
        drop(scope);
        drop((_a, _b, _u, _v, _h, _c));
        Ok(Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            Shape::from((m, n)),
        )))
    }

    fn dense(&self, device: &Device) -> Result<Tensor> {
        dense_weight(
            &self.trellis,
            &self.suh,
            &self.svh,
            self.codebook,
            self.dtype,
            device,
        )
    }
}

impl QuantMethod for Exl3Layer {
    fn exl3_layer(&self) -> Option<&Exl3Layer> {
        Some(self)
    }

    fn new(_: QuantMethodConfig) -> Result<Self> {
        inference_tensor::bail!(
            "an EXL3 layer is built from its checkpoint tensors, by exl3::linear_b"
        )
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.dense(self.trellis.device())
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (k, n, _) = self.dims();
        let rows = x.elem_count() / k;
        let x16 = x.reshape((rows, k))?.to_dtype(DType::F16)?.contiguous()?;
        let mut y = self.matmul(&x16)?.to_dtype(x.dtype())?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(&bias.to_dtype(x.dtype())?)?;
        }
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(n);
        y.reshape(out_dims)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        inference_tensor::bail!("an EXL3 layer does not take weight deltas")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.dtype, self.trellis.device().clone())
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        let (k, n, _) = self.dims();
        Ok(plan_weight_isq(
            self.dtype,
            self.trellis.device().clone(),
            vec![n, k],
            request,
            true,
        ))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_none() && device.is_cuda() {
            let bias = self
                .bias
                .as_ref()
                .map(|b| b.to_device(&device))
                .transpose()?;
            return Ok(Arc::new(Self::new_on_device(
                self.trellis.to_device(&device)?,
                self.suh.to_device(&device)?,
                self.svh.to_device(&device)?,
                self.codebook,
                bias,
                self.dtype,
            )?));
        }
        let weight = self.dense(&Device::Cpu)?;
        let bias = self
            .bias
            .as_ref()
            .map(|b| b.to_device(&Device::Cpu))
            .transpose()?;
        Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            inference_tensor::nn::Linear::new(weight, bias),
        ))?)
        .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }
}

// UQFF has no EXL3 form; an ISQ'd copy serializes as its new type
impl QuantizedSerde for Exl3Layer {
    fn name(&self) -> &'static str {
        "exl3-layer"
    }
}

/// One projection of every expert of a MoE layer, each kept as its trellis, multiplied by exllamav3's indexed
/// multi-matrix kernel: slot j takes its token's row times the expert its index names.
#[derive(Debug)]
pub struct Exl3Experts {
    // per expert: `(in / 16, out / 16, words)` i16, f16 `suh` (in), f16 `svh` (out)
    trellis: Vec<Tensor>,
    suh: Vec<Tensor>,
    svh: Vec<Tensor>,
    // the device addresses of the trellises, then the `suh`s, then the `svh`s, as the kernel reads them
    tables: Tensor,
    codebook: Exl3Codebook,
    dtype: DType,
    tuned: AtomicU8,
    // the m buckets the grouped path has launched outside a capture
    grouped_tuned: AtomicU8,
}

// past this many slots (outside a capture) each expert multiplies its tokens' rows together, in groups of at most
// GROUP_ROWS (which bounds the padding to the largest group)
const GROUPED_MIN_SLOTS: usize = 64;
// exllamav3's fused decode MoE takes up to this many tokens (its MAX_BSZN) and 256 slots
const FUSED_MAX_TOKENS: usize = 8;
const FUSED_MAX_SLOTS: usize = 256;
// the kernels' MOE_COOP_ACT_SILU
const FUSED_ACT_SILU: i32 = 0;
const GROUP_ROWS: usize = 64;

impl Exl3Experts {
    /// `parts` holds each expert's trellis, `suh`, `svh` and codebook, on the device.
    pub fn new_on_device(
        parts: Vec<(Tensor, Tensor, Tensor, Exl3Codebook)>,
        dtype: DType,
    ) -> Result<Self> {
        ffi::ensure_loaded().map_err(inference_tensor::Error::msg)?;
        let Some((first, _, _, codebook)) = parts.first() else {
            inference_tensor::bail!("an EXL3 expert projection with no experts");
        };
        let (dims, codebook) = (first.dims().to_vec(), *codebook);
        let device = first.device().clone();
        let (mut trellis, mut suh, mut svh) = (Vec::new(), Vec::new(), Vec::new());
        for (t, u, v, cb) in parts {
            if t.dims() != dims.as_slice() || cb != codebook {
                inference_tensor::bail!(
                    "EXL3 experts of one projection differ ({:?} {cb:?} against {dims:?} {codebook:?})",
                    t.dims()
                );
            }
            trellis.push(t.to_dtype(DType::I16)?.contiguous()?);
            suh.push(u.to_dtype(DType::F16)?.contiguous()?);
            svh.push(v.to_dtype(DType::F16)?.contiguous()?);
        }
        let mut addresses = Vec::with_capacity(3 * trellis.len());
        for t in &trellis {
            addresses.push(device_address::<i16>(t)?);
        }
        for v in suh.iter().chain(&svh) {
            addresses.push(device_address::<f16>(v)?);
        }
        let tables = Tensor::from_vec(addresses, 3 * trellis.len(), &device)?;
        Ok(Self {
            trellis,
            suh,
            svh,
            tables,
            codebook,
            dtype,
            tuned: AtomicU8::new(0),
            grouped_tuned: AtomicU8::new(0),
        })
    }

    fn dims(&self) -> (usize, usize, usize) {
        let d = self.trellis[0].dims();
        (d[0] * 16, d[1] * 16, d[2])
    }

    /// A gated SiLU MoE layer's routed experts for a few tokens in exllamav3's two fused launches: `xs` `(tokens,
    /// hidden)`, `ids` and `weights` `(tokens, k)`; the routing-weighted sum per token, f32 `(tokens, hidden)`. None
    /// for shapes the kernels do not take.
    pub fn fused_silu_moe(
        gate: &Self,
        up: &Self,
        down: &Self,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Option<Tensor>> {
        let Device::Cuda(dev) = xs.device() else {
            return Ok(None);
        };
        let (tokens, topk) = ids.dims2()?;
        let ((hidden, inter, gu_words), (d_in, d_out, d_words)) = (gate.dims(), down.dims());
        let experts = gate.trellis.len();
        if tokens > FUSED_MAX_TOKENS
            || up.dims() != gate.dims()
            || (d_in, d_out) != (inter, hidden)
            || [up, down]
                .iter()
                .any(|p| p.trellis.len() != experts || p.codebook != gate.codebook)
        {
            return Ok(None);
        }
        let slots_max = (FUSED_MAX_TOKENS * topk).min(FUSED_MAX_SLOTS);
        if tokens * topk > slots_max {
            return Ok(None);
        }
        let x = xs.to_dtype(DType::F16)?.contiguous()?;
        let sel = ids.to_dtype(DType::I64)?.contiguous()?;
        let rw = weights.to_dtype(DType::F16)?.contiguous()?;
        // SAFETY: ffi takes plain sizes
        let ctr_len = unsafe {
            ffi::exl3_moe_ctr_len(slots_max as i32, tokens as i32, inter as i32, hidden as i32)
        };
        // the counters reset themselves as the kernels finish; zeroed here, in stream order, for the first
        let ctr = dev.alloc_zeros::<i32>(ctr_len as usize)?;
        // SAFETY: the kernels write every slot's scratch rows before reading them, and every output row
        let (had_g, had_u) = unsafe {
            (
                dev.alloc::<f16>(slots_max * hidden)?,
                dev.alloc::<f16>(slots_max * hidden)?,
            )
        };
        // SAFETY: as above
        let (gu_g, gu_u) = unsafe {
            (
                dev.alloc::<f32>(slots_max * inter)?,
                dev.alloc::<f32>(slots_max * inter)?,
            )
        };
        // SAFETY: as above
        let (act_out, down_out) = unsafe {
            (
                dev.alloc::<f16>(slots_max * inter)?,
                dev.alloc::<f32>(slots_max * hidden)?,
            )
        };
        // SAFETY: as above
        let out = unsafe { dev.alloc::<f32>(tokens * hidden)? };
        let (x_s, x_l) = x.storage_and_layout();
        let x_o = x_l.start_offset();
        let (s_s, s_l) = sel.storage_and_layout();
        let s_o = s_l.start_offset();
        let (r_s, r_l) = rw.storage_and_layout();
        let r_o = r_l.start_offset();
        let (g_s, g_l) = gate.tables.storage_and_layout();
        let g_o = g_l.start_offset();
        let (u_s, u_l) = up.tables.storage_and_layout();
        let u_o = u_l.start_offset();
        let (d_s, d_l) = down.tables.storage_and_layout();
        let d_o = d_l.start_offset();
        let (
            Storage::Cuda(x_c),
            Storage::Cuda(s_c),
            Storage::Cuda(r_c),
            Storage::Cuda(g_c),
            Storage::Cuda(u_c),
            Storage::Cuda(d_c),
        ) = (&*x_s, &*s_s, &*r_s, &*g_s, &*u_s, &*d_s)
        else {
            inference_tensor::bail!("EXL3 experts' tensors live on CUDA");
        };
        let (x_p, _x) = slice_ptr(x_c.as_cuda_slice::<f16>()?, x_o);
        let (sel_p, _s) = slice_ptr(s_c.as_cuda_slice::<i64>()?, s_o);
        let (rw_p, _r) = slice_ptr(r_c.as_cuda_slice::<f16>()?, r_o);
        let (g_p, _g) = slice_ptr(g_c.as_cuda_slice::<i64>()?, g_o);
        let (u_p, _u) = slice_ptr(u_c.as_cuda_slice::<i64>()?, u_o);
        let (d_p, _d) = slice_ptr(d_c.as_cuda_slice::<i64>()?, d_o);
        let (had_g_p, _hg) = slice_ptr(&had_g, 0);
        let (had_u_p, _hu) = slice_ptr(&had_u, 0);
        let (gu_g_p, _gg) = slice_ptr(&gu_g, 0);
        let (gu_u_p, _gu) = slice_ptr(&gu_u, 0);
        let (act_p, _a) = slice_ptr(&act_out, 0);
        let (down_p, _do) = slice_ptr(&down_out, 0);
        let (ctr_p, _c) = slice_ptr(&ctr, 0);
        let (out_p, _o) = slice_ptr(&out, 0);
        let args = ffi::Exl3MoeArgs {
            x: x_p as *const f16,
            sel: sel_p as *const i64,
            rw: rw_p as *const f16,
            gate_tables: g_p as *const i64,
            up_tables: u_p as *const i64,
            down_tables: d_p as *const i64,
            had_g: had_g_p as *mut f16,
            had_u: had_u_p as *mut f16,
            gu_g: gu_g_p as *mut f32,
            gu_u: gu_u_p as *mut f32,
            act_out: act_p as *mut f16,
            d_out: down_p as *mut f32,
            ctr: ctr_p as *mut i32,
            out: out_p as *mut f32,
            bsz: tokens as i32,
            topk: topk as i32,
            hidden: hidden as i32,
            inter: inter as i32,
            experts: experts as i32,
            act: FUSED_ACT_SILU,
            gate_up_words: gu_words as i32,
            down_words: d_words as i32,
            cb: gate.codebook as i32,
            slots_max: slots_max as i32,
        };
        // SAFETY: every pointer is a live device buffer above, sized as the kernels read them, and the tables
        // address the three layers' own tensors
        let ran = unsafe { ffi::launch_exl3_moe(&args, dev.cuda_stream().cu_stream()) };
        drop((_x, _s, _r, _g, _u, _d, _hg, _hu, _gg, _gu, _a, _do, _c, _o));
        drop((x_s, s_s, r_s, g_s, u_s, d_s));
        if ran < 0 {
            return Ok(None);
        }
        Ok(Some(Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            Shape::from((tokens, hidden)),
        ))))
    }

    // `x` (`slots x m x in`, f16, contiguous) times the experts `ids` (`slots`, i64) names, as f32 `slots x m x out`;
    // `tuned` records the buckets that have launched outside a capture
    fn indexed_matmul(
        &self,
        x: &Tensor,
        ids: &Tensor,
        m: usize,
        tuned: &AtomicU8,
    ) -> Result<Tensor> {
        let Device::Cuda(dev) = x.device() else {
            inference_tensor::bail!("EXL3 experts multiply on CUDA");
        };
        let (k, n, words) = self.dims();
        let (slots, experts) = (ids.dim(0)?, self.trellis.len());
        let (x_s, x_l) = x.storage_and_layout();
        let (i_s, i_l) = ids.storage_and_layout();
        let (t_s, t_l) = self.tables.storage_and_layout();
        let (Storage::Cuda(x_c), Storage::Cuda(i_c), Storage::Cuda(t_c)) = (&*x_s, &*i_s, &*t_s)
        else {
            inference_tensor::bail!("EXL3 experts' tensors live on CUDA");
        };
        // SAFETY: the kernel writes each slot's rotated input before reading it, and every slot's output
        let a_had = unsafe { dev.alloc::<f16>(slots * m * k)? };
        // SAFETY: as above
        let out = unsafe { dev.alloc::<f32>(slots * m * n)? };
        let (a, _a) = slice_ptr(x_c.as_cuda_slice::<f16>()?, x_l.start_offset());
        let (idx, _i) = slice_ptr(i_c.as_cuda_slice::<i64>()?, i_l.start_offset());
        let tables = t_c.as_cuda_slice::<i64>()?;
        let (b_list, _b) = slice_ptr(tables, t_l.start_offset());
        let (suh_list, _u) = slice_ptr(tables, t_l.start_offset() + experts);
        let (svh_list, _v) = slice_ptr(tables, t_l.start_offset() + 2 * experts);
        let (had, _h) = slice_ptr(&a_had, 0);
        let (c, _c) = slice_ptr(&out, 0);
        let bucket = 1u8 << tune_bucket(if m == 1 { slots } else { m });
        // a first launch may time candidates (synchronizing, allocating), which no other thread's capture may see
        let scope = (tuned.load(Ordering::Relaxed) & bucket == 0)
            .then(inference_tensor::cuda_backend::context_wide_scope)
            .flatten();
        // SAFETY: the pointers are the live device buffers above (A slots x m x k, the address tables 3 x experts, the
        // indices slots, the scratch slots x m x k, C slots x m x n f32), and the tables address this layer's own
        // tensors
        let ran = unsafe {
            ffi::launch_exl3_mgemm(
                a as *const f16,
                b_list as *const u64,
                c as *mut std::ffi::c_void,
                m as i32,
                k as i32,
                n as i32,
                words as i32,
                self.codebook as i32,
                true,
                suh_list as *const u64,
                had as *mut f16,
                svh_list as *const u64,
                idx as *const i64,
                slots as i32,
                dev.cuda_stream().cu_stream(),
            )
        };
        if ran < 0 {
            inference_tensor::bail!(
                "no EXL3 kernel fits {slots} experts' {k} -> {n} at {words} words a tile"
            );
        }
        if scope.is_some() {
            tuned.fetch_or(bucket, Ordering::Relaxed);
        }
        drop((scope, _a, _i, _b, _u, _v, _h, _c));
        Ok(Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            Shape::from((slots * m, n)),
        )))
    }

    // as one row a slot, the slots grouped by expert into multi-row matrices (padded with a zero row to the largest),
    // so each expert's weights are read once a group (reads the indices on the host)
    fn grouped_matmul(&self, x: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let slots = ids.len();
        let mut by_expert = std::collections::BTreeMap::<u32, Vec<u32>>::new();
        for (j, &e) in ids.iter().enumerate() {
            by_expert.entry(e).or_default().push(j as u32);
        }
        let m = by_expert
            .values()
            .map(Vec::len)
            .max()
            .unwrap_or(1)
            .min(GROUP_ROWS);
        let groups = by_expert
            .iter()
            .flat_map(|(&e, members)| members.chunks(m).map(move |rows| (e, rows)))
            .collect::<Vec<_>>();
        // padding rows read the zero row appended past the slots
        let (mut rows, mut place) = (vec![slots as u32; groups.len() * m], vec![0u32; slots]);
        for (g, (_, members)) in groups.iter().enumerate() {
            for (r, &j) in members.iter().enumerate() {
                rows[g * m + r] = j;
                place[j as usize] = (g * m + r) as u32;
            }
        }
        let experts = groups
            .iter()
            .map(|&(e, _)| i64::from(e))
            .collect::<Vec<_>>();
        let padded = Tensor::cat(&[x, &x.narrow(0, 0, 1)?.zeros_like()?], 0)?
            .index_select(&Tensor::from_vec(rows, groups.len() * m, x.device())?, 0)?
            .contiguous()?;
        let experts = Tensor::from_vec(experts, groups.len(), x.device())?;
        self.indexed_matmul(&padded, &experts, m, &self.grouped_tuned)?
            .index_select(&Tensor::from_vec(place, slots, x.device())?, 0)
    }

    fn dense(&self, device: &Device) -> Result<Tensor> {
        let weights = (0..self.trellis.len())
            .map(|e| {
                dense_weight(
                    &self.trellis[e],
                    &self.suh[e],
                    &self.svh[e],
                    self.codebook,
                    self.dtype,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&weights, 0)
    }
}

fn device_address<
    T: inference_tensor::cuda_backend::CudaDType + inference_tensor::cuda::cudarc::driver::DeviceRepr,
>(
    t: &Tensor,
) -> Result<i64> {
    let (s, l) = t.storage_and_layout();
    let Storage::Cuda(c) = &*s else {
        inference_tensor::bail!("an EXL3 expert's tensor lives on CUDA");
    };
    let (ptr, _guard) = slice_ptr(c.as_cuda_slice::<T>()?, l.start_offset());
    Ok(ptr as i64)
}

impl QuantMethod for Exl3Experts {
    fn exl3_experts(&self) -> Option<&Exl3Experts> {
        Some(self)
    }

    fn new(_: QuantMethodConfig) -> Result<Self> {
        inference_tensor::bail!(
            "EXL3 experts are built from their checkpoint tensors, by exl3::experts_b"
        )
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.dense(self.trellis[0].device())
    }

    fn forward_raw(&self, _: &Tensor) -> Result<Tensor> {
        inference_tensor::bail!("EXL3 experts multiply through gather_forward")
    }

    // `a` is `(tokens, 1 or k, in)`, `indices` `(tokens, k)`; the result `(tokens, k, out)`
    fn gather_forward_raw(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (tokens, per_token) = indices.dims2()?;
        let (k, n, _) = self.dims();
        let x = a
            .broadcast_as((tokens, per_token, k))?
            .reshape((tokens * per_token, k))?
            .to_dtype(DType::F16)?
            .contiguous()?;
        let Device::Cuda(dev) = a.device() else {
            inference_tensor::bail!("EXL3 experts multiply on CUDA");
        };
        let capturing = dev.cuda_stream().capture_status().map_err(inference_tensor::Error::wrap)?
            != inference_tensor::cuda::cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let y = if tokens * per_token > GROUPED_MIN_SLOTS && !capturing {
            self.grouped_matmul(&x, &indices.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?)?
        } else {
            let ids = indices
                .to_dtype(DType::I64)?
                .reshape(tokens * per_token)?
                .contiguous()?;
            self.indexed_matmul(&x, &ids, 1, &self.tuned)?
        };
        y.to_dtype(a.dtype())?.reshape((tokens, per_token, n))
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        inference_tensor::bail!("EXL3 experts do not take weight deltas")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.dtype, self.trellis[0].device().clone())
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        let (k, n, _) = self.dims();
        Ok(plan_weight_isq(
            self.dtype,
            self.trellis[0].device().clone(),
            vec![self.trellis.len(), n, k],
            request,
            true,
        ))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_none() && device.is_cuda() {
            if device.same_device(self.trellis[0].device()) {
                return Ok(self);
            }
            let parts = (0..self.trellis.len())
                .map(|e| {
                    Ok((
                        self.trellis[e].to_device(&device)?,
                        self.suh[e].to_device(&device)?,
                        self.svh[e].to_device(&device)?,
                        self.codebook,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Arc::new(Self::new_on_device(parts, self.dtype)?));
        }
        Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            inference_tensor::nn::Linear::new(self.dense(&Device::Cpu)?, None),
        ))?)
        .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }
}

// UQFF has no EXL3 form; an ISQ'd copy serializes as its new type
impl QuantizedSerde for Exl3Experts {
    fn name(&self) -> &'static str {
        "exl3-experts"
    }
}

#[cfg(test)]
mod tests {
    use inference_tensor::{DType, Device, Result, Tensor};

    use super::{Exl3Experts, Exl3Layer};
    use crate::QuantMethod;
    use crate::exl3::{Exl3Codebook, dense_weight};

    // of the largest output: f16 accumulation measures 3.4e-4 .. 8e-4, int8 activations (mul1 at m <= 2) 6e-3
    const F16_BOUND: f32 = 2e-3;
    const INT8_BOUND: f32 = 1.5e-2;

    fn lcg(n: usize, seed: u32) -> Vec<u32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                s
            })
            .collect()
    }

    // a random `(k / 16, n / 16, words)` trellis with random signed `suh` and `svh` (f16), on the CPU
    fn random_linear(
        k: usize,
        n: usize,
        words: usize,
        seed: u32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let trellis = lcg(k * n / 256 * words, seed)
            .iter()
            .map(|s| (s >> 16) as u16 as i16)
            .collect::<Vec<_>>();
        let trellis = Tensor::from_vec(trellis, (k / 16, n / 16, words), &Device::Cpu)?;
        let scale = |len: usize, seed: u32| {
            let v = lcg(len, seed)
                .iter()
                .map(|s| if s >> 31 == 0 { 0.05f32 } else { -0.05 } * (1.0 + (s >> 8 & 0xff) as f32 / 512.0))
                .collect::<Vec<_>>();
            Tensor::from_vec(v, len, &Device::Cpu)?.to_dtype(DType::F16)
        };
        Ok((trellis, scale(k, seed + 10)?, scale(n, seed + 20)?))
    }

    fn random_rows(m: usize, k: usize, seed: u32) -> Result<Tensor> {
        let x = lcg(m * k, seed)
            .iter()
            .map(|s| (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5)
            .collect::<Vec<_>>();
        Tensor::from_vec(x, (m, k), &Device::Cpu)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)
    }

    fn relative_error(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = actual.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let err = (&actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        Ok(err / expected.abs()?.max_all()?.to_scalar::<f32>()?)
    }

    // the kernels against the CPU reconstruction: random trellises (every codebook, a half-integer rate), random
    // signed scales, one and two rows (the GEMVs; mul1's quantize activations to int8) and 32 (the GEMM)
    #[test]
    fn the_kernels_match_the_reconstructed_weight() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (k, n) = (512, 512);
        let mut ratios = Vec::new();
        let mut failed = false;
        for (words, codebook, seed) in [
            (48, Exl3Codebook::ThreeInst, 1),
            (64, Exl3Codebook::Mcg, 2),
            (32, Exl3Codebook::Mul1, 3),
            (40, Exl3Codebook::Mul1, 4),
        ] {
            let (trellis, suh, svh) = random_linear(k, n, words, seed)?;
            let weight = dense_weight(&trellis, &suh, &svh, codebook, DType::F32, &Device::Cpu)?;
            let layer = Exl3Layer::new_on_device(
                trellis.to_device(&device)?,
                suh.to_device(&device)?,
                svh.to_device(&device)?,
                codebook,
                None,
                DType::F16,
            )?;
            for m in [1, 2, 32] {
                let x = random_rows(m, k, seed + 30)?;
                let expected = x.matmul(&weight.t()?)?;
                let actual = layer.forward_raw(&x.to_dtype(DType::F16)?.to_device(&device)?)?;
                let ratio = relative_error(&actual, &expected)?;
                ratios.push(format!("{codebook:?} {words} m {m}: {ratio}"));
                let int8 = codebook == Exl3Codebook::Mul1 && m <= 2;
                failed |= ratio > if int8 { INT8_BOUND } else { F16_BOUND };
            }
        }
        assert!(!failed, "{ratios:?}");
        Ok(())
    }

    // the experts against each slot's own reconstructed expert: shared rows (gate and up take one row per token) and
    // a row per slot (down), with repeated experts, at one and twelve tokens (the indexed kernel), forty and two hundred (grouped, then split past GROUP_ROWS)
    #[test]
    fn experts_match_their_reconstructed_weights() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (experts, per_token, k, n, words) = (6, 2, 256, 384, 48);
        let parts = (0..experts)
            .map(|e| random_linear(k, n, words, 100 + e as u32))
            .collect::<Result<Vec<_>>>()?;
        let weights = parts
            .iter()
            .map(|(t, u, v)| {
                dense_weight(t, u, v, Exl3Codebook::ThreeInst, DType::F32, &Device::Cpu)
            })
            .collect::<Result<Vec<_>>>()?;
        let layer = Exl3Experts::new_on_device(
            parts
                .iter()
                .map(|(t, u, v)| {
                    Ok((
                        t.to_device(&device)?,
                        u.to_device(&device)?,
                        v.to_device(&device)?,
                        Exl3Codebook::ThreeInst,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
            DType::F32,
        )?;
        let mut ratios = Vec::new();
        for tokens in [1, 12, 40, 200] {
            let ids = (0..tokens * per_token)
                .map(|j| ((j * 5 + j / 3) % experts) as u32)
                .collect::<Vec<_>>();
            for shared in [true, false] {
                let rows = random_rows(tokens * if shared { 1 } else { per_token }, k, 7)?;
                let expected = (0..tokens * per_token)
                    .map(|j| {
                        let row = if shared { j / per_token } else { j };
                        rows.narrow(0, row, 1)?
                            .matmul(&weights[ids[j] as usize].t()?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let expected = Tensor::cat(&expected, 0)?.reshape((tokens, per_token, n))?;
                let a = rows.reshape((tokens, if shared { 1 } else { per_token }, k))?;
                let indices = Tensor::from_vec(ids.clone(), (tokens, per_token), &device)?;
                let actual = layer.gather_forward(&a.to_device(&device)?, &indices)?;
                ratios.push((tokens, shared, relative_error(&actual, &expected)?));
            }
        }
        assert!(ratios.iter().all(|r| r.2 < F16_BOUND), "{ratios:?}");
        Ok(())
    }

    // the fused MoE against the dense layer it computes: per token, the routing-weighted sum over its experts of
    // down(silu(gate(x)) * up(x)), gate and up at one rate and down at another, at one token and five
    #[test]
    fn the_fused_moe_matches_its_dense_layer() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (experts, topk, hidden, inter) = (6, 2, 256, 384);
        let projection =
            |k: usize, n: usize, words: usize, seed: u32| -> Result<(Exl3Experts, Vec<Tensor>)> {
                let parts = (0..experts)
                    .map(|e| random_linear(k, n, words, seed + 7 * e as u32))
                    .collect::<Result<Vec<_>>>()?;
                let dense = parts
                    .iter()
                    .map(|(t, u, v)| {
                        dense_weight(t, u, v, Exl3Codebook::ThreeInst, DType::F32, &Device::Cpu)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let layer = Exl3Experts::new_on_device(
                    parts
                        .iter()
                        .map(|(t, u, v)| {
                            Ok((
                                t.to_device(&device)?,
                                u.to_device(&device)?,
                                v.to_device(&device)?,
                                Exl3Codebook::ThreeInst,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?,
                    DType::F32,
                )?;
                Ok((layer, dense))
            };
        let (gate, gate_w) = projection(hidden, inter, 48, 200)?;
        let (up, up_w) = projection(hidden, inter, 48, 300)?;
        let (down, down_w) = projection(inter, hidden, 64, 400)?;
        let mut ratios = Vec::new();
        for tokens in [1, 5] {
            let xs = random_rows(tokens, hidden, 9)?;
            let ids = (0..tokens * topk)
                .map(|j| ((j * 5 + 1) % experts) as u32)
                .collect::<Vec<_>>();
            let weights = (0..tokens * topk)
                .map(|j| 0.25 + 0.5 * (j % topk) as f32)
                .collect::<Vec<_>>();
            let mut expected = Vec::new();
            for t in 0..tokens {
                let x = xs.narrow(0, t, 1)?;
                let mut sum = Tensor::zeros((1, hidden), DType::F32, &Device::Cpu)?;
                for j in t * topk..(t + 1) * topk {
                    let e = ids[j] as usize;
                    let h = (x.matmul(&gate_w[e].t()?)?.silu()? * x.matmul(&up_w[e].t()?)?)?;
                    sum = (sum + (h.matmul(&down_w[e].t()?)? * f64::from(weights[j]))?)?;
                }
                expected.push(sum);
            }
            let expected = Tensor::cat(&expected, 0)?;
            let actual = Exl3Experts::fused_silu_moe(
                &gate,
                &up,
                &down,
                &xs.to_device(&device)?,
                &Tensor::from_vec(ids, (tokens, topk), &device)?,
                &Tensor::from_vec(weights, (tokens, topk), &device)?,
            )?
            .expect("the fused kernels take these shapes");
            ratios.push((tokens, relative_error(&actual, &expected)?));
        }
        assert!(ratios.iter().all(|r| r.1 < F16_BOUND), "{ratios:?}");
        Ok(())
    }

    // a fused group (q, k and v: one input, three widths, their own rotations) against each layer alone, in f32 and
    // f16 models, at one row and four
    #[test]
    fn a_fused_group_matches_its_layers() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let k = 256;
        let mut ratios = Vec::new();
        for dtype in [DType::F32, DType::F16] {
            let mut layers = Vec::new();
            let mut weights = Vec::new();
            for (n, seed) in [(384, 500), (128, 600), (256, 700)] {
                let (t, u, v) = random_linear(k, n, 48, seed)?;
                weights.push(dense_weight(
                    &t,
                    &u,
                    &v,
                    Exl3Codebook::ThreeInst,
                    DType::F32,
                    &Device::Cpu,
                )?);
                layers.push(Exl3Layer::new_on_device(
                    t.to_device(&device)?,
                    u.to_device(&device)?,
                    v.to_device(&device)?,
                    Exl3Codebook::ThreeInst,
                    None,
                    dtype,
                )?);
            }
            for rows in [1, 4] {
                let x = random_rows(rows, k, 11)?;
                let outs = layers[0]
                    .fused_group(
                        &[&layers[1], &layers[2]],
                        &x.to_dtype(dtype)?.to_device(&device)?,
                    )?
                    .expect("these layers fuse");
                for (out, w) in outs.iter().zip(&weights) {
                    ratios.push((dtype, rows, relative_error(out, &x.matmul(&w.t()?)?)?));
                }
            }
        }
        assert!(ratios.iter().all(|r| r.2 < F16_BOUND), "{ratios:?}");
        Ok(())
    }
}
