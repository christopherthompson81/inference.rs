//! A direct grouped convolution: each output plane sums its group's input channels in place, with no per-group split.

use rayon::prelude::*;

use crate::conv::{ParamsConv1D, ParamsConv2D};
use crate::cpu_backend::Map2;
use crate::{Layout, Result, WithDType};

/// One group's 2D convolution geometry; a 1D convolution is one row high.
pub(super) struct GroupedConv {
    b_size: usize,
    groups: usize,
    c_in: usize,
    c_out: usize,
    i_h: usize,
    i_w: usize,
    k_h: usize,
    k_w: usize,
    out_h: usize,
    out_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride: usize,
    dilation: usize,
}

impl GroupedConv {
    pub(super) fn conv1d(p: &ParamsConv1D) -> Self {
        Self {
            b_size: p.b_size,
            groups: p.groups,
            c_in: p.c_in,
            c_out: p.c_out,
            i_h: 1,
            i_w: p.l_in,
            k_h: 1,
            k_w: p.k_size,
            out_h: 1,
            out_w: p.l_out(),
            pad_h: 0,
            pad_w: p.padding,
            stride: p.stride,
            dilation: p.dilation,
        }
    }

    pub(super) fn conv2d(p: &ParamsConv2D) -> Self {
        Self {
            b_size: p.b_size,
            groups: p.groups,
            c_in: p.c_in,
            c_out: p.c_out,
            i_h: p.i_h,
            i_w: p.i_w,
            k_h: p.k_h,
            k_w: p.k_w,
            out_h: p.out_h(),
            out_w: p.out_w(),
            pad_h: p.padding,
            pad_w: p.padding,
            stride: p.stride,
            dilation: p.dilation,
        }
    }

    // the output positions along one axis whose tap `k` lands inside the input
    fn valid(&self, k: usize, pad: usize, len: usize, out: usize) -> std::ops::Range<usize> {
        let offset = k * self.dilation;
        let first = pad.saturating_sub(offset).div_ceil(self.stride);
        let last = (len + pad)
            .saturating_sub(offset)
            .div_ceil(self.stride)
            .min(out);
        first..last.max(first)
    }
}

// rank-3 (conv1d) layouts gain a unit height axis
fn strides4(l: &Layout) -> [usize; 4] {
    match *l.stride() {
        [s0, s1, s2] => [s0, s1, 0, s2],
        [s0, s1, s2, s3] => [s0, s1, s2, s3],
        _ => unreachable!("conv layouts are rank 3 or 4"),
    }
}

impl Map2 for GroupedConv {
    const OP: &'static str = "grouped-conv";
    fn f<T: WithDType + num_traits::Num + Copy + 'static>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        let (inp, k) = (&inp[inp_l.start_offset()..], &k[k_l.start_offset()..]);
        let [is0, is1, is2, is3] = strides4(inp_l);
        let [ks0, ks1, ks2, ks3] = strides4(k_l);
        let plane = self.out_h * self.out_w;
        let c_out = self.c_out * self.groups;
        let mut dst = vec![T::zero(); self.b_size * c_out * plane];
        dst.par_chunks_mut(plane)
            .enumerate()
            .for_each(|(idx, out)| {
                let (b, o) = (idx / c_out, idx % c_out);
                let group = o / self.c_out;
                // f64 keeps half-precision sums from rounding at every tap
                let mut acc = vec![0f64; plane];
                for i in 0..self.c_in {
                    let src = &inp[b * is0 + (group * self.c_in + i) * is1..];
                    for kh in 0..self.k_h {
                        let rows = self.valid(kh, self.pad_h, self.i_h, self.out_h);
                        for kw in 0..self.k_w {
                            let w = k[o * ks0 + i * ks1 + kh * ks2 + kw * ks3].to_f64();
                            let cols = self.valid(kw, self.pad_w, self.i_w, self.out_w);
                            for oy in rows.clone() {
                                let iy = oy * self.stride + kh * self.dilation - self.pad_h;
                                let acc = &mut acc[oy * self.out_w..];
                                for ox in cols.clone() {
                                    let ix = ox * self.stride + kw * self.dilation - self.pad_w;
                                    acc[ox] += w * src[iy * is2 + ix * is3].to_f64();
                                }
                            }
                        }
                    }
                }
                for (d, a) in out.iter_mut().zip(acc) {
                    *d = T::from_f64(a);
                }
            });
        Ok(dst)
    }
}
