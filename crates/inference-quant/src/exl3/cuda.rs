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
}

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
        })
    }

    fn dims(&self) -> (usize, usize, usize) {
        let d = self.trellis.dims();
        (d[0] * 16, d[1] * 16, d[2])
    }

    // `x` (`m x in`, f16, contiguous) times the weight, as f32 `m x out`
    fn matmul(&self, x: &Tensor) -> Result<Tensor> {
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
        let out = unsafe { dev.alloc::<f32>(m * n)? };
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
                true,
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

#[cfg(test)]
mod tests {
    use inference_tensor::{DType, Device, Result, Tensor};

    use super::Exl3Layer;
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
            let (suh, svh) = (scale(k, seed + 10)?, scale(n, seed + 20)?);
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
                let x = lcg(m * k, seed + 30)
                    .iter()
                    .map(|s| (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5)
                    .collect::<Vec<_>>();
                let x = Tensor::from_vec(x, (m, k), &Device::Cpu)?
                    .to_dtype(DType::F16)?
                    .to_dtype(DType::F32)?;
                let expected = x.matmul(&weight.t()?)?;
                let actual = layer.forward_raw(&x.to_dtype(DType::F16)?.to_device(&device)?)?;
                let actual = actual.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                let err = (&actual - &expected)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                let norm = expected.abs()?.max_all()?.to_scalar::<f32>()?;
                ratios.push(format!("{codebook:?} {words} m {m}: {}", err / norm));
                let int8 = codebook == Exl3Codebook::Mul1 && m <= 2;
                failed |= err / norm > if int8 { INT8_BOUND } else { F16_BOUND };
            }
        }
        assert!(!failed, "{ratios:?}");
        Ok(())
    }
}
