use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

/// `(b, c, h, w)` x `(c, 1, k, k)` depthwise conv plus per-channel bias, zero padded.
pub fn depthwise_conv2d(
    xs: &Tensor,
    w: &Tensor,
    b: &Tensor,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    xs.contiguous()?
        .apply_op3_no_bwd(w, b, &DepthwiseConv { stride, padding })
}

struct DepthwiseConv {
    stride: usize,
    padding: usize,
}

struct Geom {
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    ho: usize,
    wo: usize,
}

impl DepthwiseConv {
    fn geom(&self, lx: &Layout, lw: &Layout) -> Result<Geom> {
        let (b, c, h, w) = lx.shape().dims4()?;
        let (cw, one, k, k2) = lw.shape().dims4()?;
        if cw != c || one != 1 || k != k2 {
            candle_core::bail!(
                "depthwise weight {:?} does not match input {:?}",
                lw.shape(),
                lx.shape()
            );
        }
        if !lx.is_contiguous() || !lw.is_contiguous() {
            candle_core::bail!("depthwise conv expects contiguous input and weight");
        }
        let ho = (h + 2 * self.padding - k) / self.stride + 1;
        let wo = (w + 2 * self.padding - k) / self.stride + 1;
        Ok(Geom {
            b,
            c,
            h,
            w,
            k,
            ho,
            wo,
        })
    }
}

impl CustomOp3 for DepthwiseConv {
    fn name(&self) -> &'static str {
        "depthwise-conv2d"
    }

    fn cpu_fwd(
        &self,
        sx: &CpuStorage,
        lx: &Layout,
        sw: &CpuStorage,
        lw: &Layout,
        sb: &CpuStorage,
        lb: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let g = self.geom(lx, lw)?;
        let (CpuStorage::F32(x), CpuStorage::F32(wt), CpuStorage::F32(bias)) = (sx, sw, sb) else {
            candle_core::bail!("depthwise conv CPU path is f32 only");
        };
        let x = &x[lx.start_offset()..];
        let wt = &wt[lw.start_offset()..];
        let bias = &bias[lb.start_offset()..];
        let (s, p) = (self.stride as isize, self.padding as isize);
        let mut out = vec![0f32; g.b * g.c * g.ho * g.wo];
        out.par_chunks_mut(g.ho * g.wo)
            .enumerate()
            .for_each(|(plane, o)| {
                let c = plane % g.c;
                let xp = &x[plane * g.h * g.w..(plane + 1) * g.h * g.w];
                let wk = &wt[c * g.k * g.k..(c + 1) * g.k * g.k];
                for oy in 0..g.ho {
                    for ox in 0..g.wo {
                        let mut acc = bias[c];
                        for ky in 0..g.k {
                            let iy = oy as isize * s - p + ky as isize;
                            if iy < 0 || iy >= g.h as isize {
                                continue;
                            }
                            let row = &xp[iy as usize * g.w..(iy as usize + 1) * g.w];
                            for kx in 0..g.k {
                                let ix = ox as isize * s - p + kx as isize;
                                if ix >= 0 && ix < g.w as isize {
                                    acc += row[ix as usize] * wk[ky * g.k + kx];
                                }
                            }
                        }
                        o[oy * g.wo + ox] = acc;
                    }
                }
            });
        Ok((CpuStorage::F32(out), Shape::from((g.b, g.c, g.ho, g.wo))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        sx: &candle_core::CudaStorage,
        lx: &Layout,
        sw: &candle_core::CudaStorage,
        lw: &Layout,
        sb: &candle_core::CudaStorage,
        lb: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::{
            cudarc::driver::{LaunchConfig, PushKernelArg},
            CudaStorageSlice, WrapErr,
        };

        let g = self.geom(lx, lw)?;
        let dev = &sx.device;
        let x = sx.as_cuda_slice::<f32>()?.slice(lx.start_offset()..);
        let wt = sw.as_cuda_slice::<f32>()?.slice(lw.start_offset()..);
        let bias = sb.as_cuda_slice::<f32>()?.slice(lb.start_offset()..);
        let n = g.b * g.c * g.ho * g.wo;
        let mut out = unsafe { dev.alloc::<f32>(n)? };
        let func = dev.get_or_load_custom_func(
            crate::cuda_kernels::DEPTHWISE,
            crate::cuda_kernels::MODULE,
            crate::cuda_kernels::ptx()?,
        )?;
        let dims = [g.c, g.h, g.w, g.ho, g.wo, g.k, self.stride, self.padding].map(|v| v as i32);
        let n_i32 = i32::try_from(n)?;
        let mut builder = func.builder();
        builder.arg(&x);
        builder.arg(&wt);
        builder.arg(&bias);
        builder.arg(&mut out);
        builder.arg(&n_i32);
        for d in &dims {
            builder.arg(d);
        }
        unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            },
            Shape::from((g.b, g.c, g.ho, g.wo)),
        ))
    }
}
