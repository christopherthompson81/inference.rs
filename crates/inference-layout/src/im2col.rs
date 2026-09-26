use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
const BLOCK: usize = 256;
#[cfg(feature = "cuda")]
const MAX_GRID_Y: usize = 65535;

/// `(b, c, h, w)` -> `(b, c*k*k + 1, ho*wo)` zero-padded patches plus a ones row, so a biased conv is one matmul.
pub fn im2col(xs: &Tensor, kernel: usize, stride: usize, padding: usize) -> Result<Tensor> {
    xs.contiguous()?.apply_op1_no_bwd(&Im2Col {
        kernel,
        stride,
        padding,
    })
}

struct Im2Col {
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl Im2Col {
    fn out_hw(&self, h: usize, w: usize) -> (usize, usize) {
        let ho = (h + 2 * self.padding - self.kernel) / self.stride + 1;
        let wo = (w + 2 * self.padding - self.kernel) / self.stride + 1;
        (ho, wo)
    }
}

impl CustomOp1 for Im2Col {
    fn name(&self) -> &'static str {
        "im2col-cols-last"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, c, h, w) = l.shape().dims4()?;
        let CpuStorage::F32(x) = s else {
            candle_core::bail!("im2col CPU path is f32 only");
        };
        let x = &x[l.start_offset()..];
        let (ho, wo) = self.out_hw(h, w);
        let k = self.kernel;
        let (st, p) = (self.stride as isize, self.padding as isize);
        let rows = c * k * k + 1;
        let mut out = vec![0f32; b * rows * ho * wo];
        // one output row per (batch, channel, ky, kx), then the ones row
        out.par_chunks_mut(ho * wo)
            .enumerate()
            .for_each(|(row, o)| {
                let (n, r) = (row / rows, row % rows);
                if r == rows - 1 {
                    o.fill(1.);
                    return;
                }
                let kx = r % k;
                let ky = (r / k) % k;
                let plane = n * c + r / (k * k);
                let xp = &x[plane * h * w..(plane + 1) * h * w];
                for oy in 0..ho {
                    let iy = oy as isize * st - p + ky as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for ox in 0..wo {
                        let ix = ox as isize * st - p + kx as isize;
                        if ix >= 0 && ix < w as isize {
                            o[oy * wo + ox] = xp[iy as usize * w + ix as usize];
                        }
                    }
                }
            });
        Ok((CpuStorage::F32(out), Shape::from((b, rows, ho * wo))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::{
            cudarc::driver::{LaunchConfig, PushKernelArg},
            CudaStorageSlice, WrapErr,
        };

        let (b, c, h, w) = l.shape().dims4()?;
        let (ho, wo) = self.out_hw(h, w);
        let k = self.kernel;
        let dev = &s.device;
        let x = s.as_cuda_slice::<f32>()?.slice(l.start_offset()..);
        let rows = c * k * k + 1;
        let mut out = unsafe { dev.alloc::<f32>(b * rows * ho * wo)? };
        let func = dev.get_or_load_custom_func(
            crate::cuda_kernels::IM2COL,
            crate::cuda_kernels::MODULE,
            crate::cuda_kernels::ptx()?,
        )?;
        let n_rows = b * rows;
        let dims = [rows, n_rows, c, h, w, ho, wo, k, self.stride, self.padding]
            .map(i32::try_from)
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut builder = func.builder();
        builder.arg(&x);
        builder.arg(&mut out);
        for d in &dims {
            builder.arg(d);
        }
        let cfg = LaunchConfig {
            grid_dim: (
                (ho * wo).div_ceil(BLOCK) as u32,
                n_rows.min(MAX_GRID_Y) as u32,
                1,
            ),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.w()?;
        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            },
            Shape::from((b, rows, ho * wo)),
        ))
    }
}
