use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

pub const MAX_LEVELS: usize = 4;

/// Deformable-attn sampling (bilinear, zero pad): `v (b,S,h,d)`, `loc (b,q,h,l,p,2)`, `attn (b,q,h,l,p)` -> `(b,q,h*d)`.
pub fn ms_deform_attn(
    value: &Tensor,
    loc: &Tensor,
    attn: &Tensor,
    levels: &[(usize, usize)],
) -> Result<Tensor> {
    if levels.len() > MAX_LEVELS {
        candle_core::bail!("ms_deform_attn supports up to {MAX_LEVELS} levels");
    }
    value.contiguous()?.apply_op3_no_bwd(
        &loc.contiguous()?,
        &attn.contiguous()?,
        &MsDeformAttn {
            levels: levels.to_vec(),
        },
    )
}

struct MsDeformAttn {
    levels: Vec<(usize, usize)>,
}

#[derive(Clone, Copy)]
struct Dims {
    b: usize,
    s: usize,
    h: usize,
    d: usize,
    q: usize,
    l: usize,
    p: usize,
}

impl MsDeformAttn {
    fn dims(&self, lv: &Layout, ll: &Layout, la: &Layout) -> Result<Dims> {
        let (b, s, h, d) = lv.shape().dims4()?;
        let ld = ll.shape().dims();
        let [lb, q, lh, l, p, two] = ld[..] else {
            candle_core::bail!("ms_deform_attn: loc must be rank 6, got {ld:?}");
        };
        if lb != b
            || lh != h
            || l != self.levels.len()
            || two != 2
            || la.shape().dims() != [b, q, h, l, p]
        {
            candle_core::bail!(
                "ms_deform_attn: shape mismatch {:?} {:?} {:?}",
                lv.shape(),
                ll.shape(),
                la.shape()
            );
        }
        if self.levels.iter().map(|(lh, lw)| lh * lw).sum::<usize>() != s {
            candle_core::bail!("ms_deform_attn: level shapes do not cover {s} tokens");
        }
        Ok(Dims {
            b,
            s,
            h,
            d,
            q,
            l,
            p,
        })
    }
}

/// Bilinear taps of one sample: `(flat pixel within level, weight)` for each in-bounds corner.
fn corners(lx: f32, ly: f32, lh: usize, lw: usize) -> impl Iterator<Item = (usize, f32)> {
    let x = lx * lw as f32 - 0.5;
    let y = ly * lh as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    [
        (0., 0., (1. - fx) * (1. - fy)),
        (1., 0., fx * (1. - fy)),
        (0., 1., (1. - fx) * fy),
        (1., 1., fx * fy),
    ]
    .into_iter()
    .filter_map(move |(dx, dy, w)| {
        let (xc, yc) = (x0 + dx, y0 + dy);
        (xc >= 0. && yc >= 0. && xc < lw as f32 && yc < lh as f32)
            .then(|| (yc as usize * lw + xc as usize, w))
    })
}

impl CustomOp3 for MsDeformAttn {
    fn name(&self) -> &'static str {
        "ms-deform-attn"
    }

    fn cpu_fwd(
        &self,
        sv: &CpuStorage,
        lv: &Layout,
        sl: &CpuStorage,
        ll: &Layout,
        sa: &CpuStorage,
        la: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let g = self.dims(lv, ll, la)?;
        let (CpuStorage::F32(v), CpuStorage::F32(loc), CpuStorage::F32(attn)) = (sv, sl, sa) else {
            candle_core::bail!("ms_deform_attn CPU path is f32 only");
        };
        let (v, loc, attn) = (
            &v[lv.start_offset()..],
            &loc[ll.start_offset()..],
            &attn[la.start_offset()..],
        );
        let mut starts = Vec::with_capacity(g.l);
        let mut acc_start = 0;
        for (lh, lw) in &self.levels {
            starts.push(acc_start);
            acc_start += lh * lw;
        }
        let mut out = vec![0f32; g.b * g.q * g.h * g.d];
        // one chunk per (batch, query, head)
        out.par_chunks_mut(g.d).enumerate().for_each(|(bqh, o)| {
            let (bq, hh) = (bqh / g.h, bqh % g.h);
            let bi = bq / g.q;
            for (lvl, (&(lh, lw), &start)) in self.levels.iter().zip(&starts).enumerate() {
                for pt in 0..g.p {
                    let sidx = ((bqh * g.l) + lvl) * g.p + pt;
                    let a = attn[sidx];
                    for (pix, w) in corners(loc[sidx * 2], loc[sidx * 2 + 1], lh, lw) {
                        let row = ((bi * g.s + start + pix) * g.h + hh) * g.d;
                        for (od, vd) in o.iter_mut().zip(&v[row..row + g.d]) {
                            *od += a * w * vd;
                        }
                    }
                }
            }
        });
        Ok((CpuStorage::F32(out), Shape::from((g.b, g.q, g.h * g.d))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        sv: &candle_core::CudaStorage,
        lv: &Layout,
        sl: &candle_core::CudaStorage,
        ll: &Layout,
        sa: &candle_core::CudaStorage,
        la: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::{
            cudarc::driver::{LaunchConfig, PushKernelArg},
            CudaStorageSlice, WrapErr,
        };

        let g = self.dims(lv, ll, la)?;
        let dev = &sv.device;
        let v = sv.as_cuda_slice::<f32>()?.slice(lv.start_offset()..);
        let loc = sl.as_cuda_slice::<f32>()?.slice(ll.start_offset()..);
        let attn = sa.as_cuda_slice::<f32>()?.slice(la.start_offset()..);
        let n = g.b * g.q * g.h * g.d;
        let mut out = unsafe { dev.alloc::<f32>(n)? };
        let func = dev.get_or_load_custom_func(
            crate::cuda_kernels::MS_DEFORM_ATTN,
            crate::cuda_kernels::MODULE,
            crate::cuda_kernels::ptx()?,
        )?;
        let mut lvl = [0i32; 3 * MAX_LEVELS];
        let mut start = 0;
        for (i, &(lh, lw)) in self.levels.iter().enumerate() {
            lvl[i] = i32::try_from(lh)?;
            lvl[MAX_LEVELS + i] = i32::try_from(lw)?;
            lvl[2 * MAX_LEVELS + i] = i32::try_from(start)?;
            start += lh * lw;
        }
        let dims = [n, g.s, g.q, g.h, g.d, g.l, g.p]
            .map(i32::try_from)
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut builder = func.builder();
        builder.arg(&v);
        builder.arg(&loc);
        builder.arg(&attn);
        builder.arg(&mut out);
        for x in dims.iter().chain(&lvl) {
            builder.arg(x);
        }
        unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            },
            Shape::from((g.b, g.q, g.h * g.d)),
        ))
    }
}
