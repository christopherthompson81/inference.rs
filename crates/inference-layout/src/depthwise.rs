use inference_tensor::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

/// `(b, c, h, w)` x `(c, 1, k, k)` depthwise conv plus per-channel bias, zero padded.
pub fn depthwise_conv2d(
    xs: &Tensor,
    w: &Tensor,
    b: &Tensor,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    if xs.device().is_cuda() {
        let c = w.dim(0)?;
        return xs
            .conv2d(w, padding, stride, 1, c)?
            .broadcast_add(&b.reshape((1, c, 1, 1))?);
    }
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
            inference_tensor::bail!(
                "depthwise weight {:?} does not match input {:?}",
                lw.shape(),
                lx.shape()
            );
        }
        if !lx.is_contiguous() || !lw.is_contiguous() {
            inference_tensor::bail!("depthwise conv expects contiguous input and weight");
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
        let [x, wt, bias] = crate::cpu_direct::f32_slices([(sx, lx), (sw, lw), (sb, lb)])?;
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
}
