use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
use gemm::Parallelism;
use rayon::prelude::*;

/// CPU dense conv as one accumulating GEMM per kernel tap over shifted views of the padded input (no im2col buffer).
pub fn conv2d(
    xs: &Tensor,
    w: &Tensor,
    b: &Tensor,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    xs.contiguous()?.apply_op3_no_bwd(
        &w.contiguous()?,
        &b.contiguous()?,
        &ImplicitConv { stride, padding },
    )
}

struct ImplicitConv {
    stride: usize,
    padding: usize,
}

impl CustomOp3 for ImplicitConv {
    fn name(&self) -> &'static str {
        "implicit-conv2d"
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
        let (CpuStorage::F32(x), CpuStorage::F32(wt), CpuStorage::F32(bias)) = (sx, sw, sb) else {
            candle_core::bail!("implicit conv is f32 only");
        };
        let (x, wt, bias) = (
            &x[lx.start_offset()..],
            &wt[lw.start_offset()..],
            &bias[lb.start_offset()..],
        );
        let (bn, c, h, w) = lx.shape().dims4()?;
        let (o, ci, k, k2) = lw.shape().dims4()?;
        if ci != c || k != k2 {
            candle_core::bail!(
                "implicit conv: weight {:?} vs input {:?}",
                lw.shape(),
                lx.shape()
            );
        }
        let (s, p) = (self.stride, self.padding);
        let ho = (h + 2 * p - k) / s + 1;
        let wo = (w + 2 * p - k) / s + 1;
        // padded extent rounded up to a stride multiple, split into s*s phase planes of (hs, ws)
        let hs = (h + 2 * p).div_ceil(s);
        let ws = (w + 2 * p).div_ceil(s);
        let plane = hs * ws;
        let n = ho * ws;
        let par = Parallelism::Rayon(candle_core::utils::get_num_threads());
        let mut out = vec![0f32; bn * o * ho * wo];
        // the last tap's view runs up to k/s elements past its plane; the slack keeps that read in bounds
        let mut phases = vec![0f32; s * s * c * plane + k];
        let mut full = vec![0f32; o * n];
        for bi in 0..bn {
            let xb = &x[bi * c * h * w..(bi + 1) * c * h * w];
            phases[..s * s * c * plane]
                .par_chunks_mut(plane)
                .enumerate()
                .for_each(|(pc, dst)| {
                    let (ph, ch) = (pc / c, pc % c);
                    let (py, px) = (ph / s, ph % s);
                    let src = &xb[ch * h * w..(ch + 1) * h * w];
                    for (r, row) in dst.chunks_mut(ws).enumerate() {
                        let iy = (r * s + py) as isize - p as isize;
                        for (col, v) in row.iter_mut().enumerate() {
                            let ix = (col * s + px) as isize - p as isize;
                            *v = if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                src[iy as usize * w + ix as usize]
                            } else {
                                0.
                            };
                        }
                    }
                });
            for ky in 0..k {
                for kx in 0..k {
                    let first = ky == 0 && kx == 0;
                    let ph = (ky % s) * s + kx % s;
                    let rhs = ph * c * plane + (ky / s) * ws + kx / s;
                    // SAFETY: w[:, :, ky, kx], a phase-plane view (slack covers the overhang) and scratch, all in bounds.
                    unsafe {
                        gemm::gemm(
                            o,
                            n,
                            c,
                            full.as_mut_ptr(),
                            1,
                            n as isize,
                            !first,
                            wt.as_ptr().add(ky * k + kx),
                            (k * k) as isize,
                            (c * k * k) as isize,
                            phases.as_ptr().add(rhs),
                            1,
                            plane as isize,
                            1.,
                            1.,
                            false,
                            false,
                            false,
                            par,
                        );
                    }
                }
            }
            let ob = &mut out[bi * o * ho * wo..(bi + 1) * o * ho * wo];
            ob.par_chunks_mut(ho * wo)
                .enumerate()
                .for_each(|(oc, dst)| {
                    for oy in 0..ho {
                        let src = &full[oc * n + oy * ws..oc * n + oy * ws + wo];
                        for (d, v) in dst[oy * wo..(oy + 1) * wo].iter_mut().zip(src) {
                            *d = v + bias[oc];
                        }
                    }
                });
        }
        Ok((CpuStorage::F32(out), Shape::from((bn, o, ho, wo))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    #[test]
    fn matches_candle_conv_batched() -> Result<()> {
        let dev = Device::Cpu;
        for (k, s, p, h, w) in [
            (3, 1, 1, 9, 7),
            (3, 2, 1, 10, 11),
            (2, 1, 0, 6, 5),
            (5, 2, 2, 9, 9),
        ] {
            let x = Tensor::randn(0f32, 1., (3, 20, h, w), &dev)?;
            let wt = Tensor::randn(0f32, 1., (7, 20, k, k), &dev)?;
            let b = Tensor::randn(0f32, 1., 7, &dev)?;
            let want = x
                .conv2d(&wt, p, s, 1, 1)?
                .broadcast_add(&b.reshape((1, 7, 1, 1))?)?;
            let got = conv2d(&x, &wt, &b, s, p)?;
            assert_eq!(got.dims(), want.dims());
            let err = (got - want)?
                .abs()?
                .flatten_all()?
                .max(D::Minus1)?
                .to_scalar::<f32>()?;
            assert!(err < 1e-4, "k={k} s={s} {h}x{w} err={err}");
        }
        Ok(())
    }
}
