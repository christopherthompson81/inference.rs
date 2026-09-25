use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor, D};

/// Min-reduction fill for pixels outside the mask in the tensor-op fallback.
const MASK_MIN_FILL: f64 = 1e9;
use rayon::prelude::*;

/// `(b, q, h*w)` mask logits -> `(b, q, 4)` normalized cxcywh of each `> 0` region's pixel bbox (zeros if empty).
pub fn mask_to_box(masks: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    if !crate::has_kernels(masks.device()) {
        return mask_to_box_ops(masks, h, w);
    }
    masks.contiguous()?.apply_op1_no_bwd(&MaskToBox { h, w })
}

/// Tensor-op version for devices without the custom kernel.
fn mask_to_box_ops(masks: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let dev = masks.device();
    let coords = |n: usize, shape: (usize, usize)| -> Result<Tensor> {
        Tensor::arange(0u32, n as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape(shape)?
            .broadcast_as((h, w))?
            .contiguous()?
            .reshape((1, 1, h * w))
    };
    let (xs, ys) = (coords(w, (1, w))?, coords(h, (h, 1))?);
    let m = masks.gt(0.)?.to_dtype(DType::F32)?;
    let fill = (m.affine(-1., 1.)? * MASK_MIN_FILL)?;
    let mx = m.broadcast_mul(&xs)?;
    let my = m.broadcast_mul(&ys)?;
    let x0 = (&mx + &fill)?.min_keepdim(D::Minus1)?;
    let y0 = (&my + &fill)?.min_keepdim(D::Minus1)?;
    let x1 = (mx.max_keepdim(D::Minus1)? + 1.)?;
    let y1 = (my.max_keepdim(D::Minus1)? + 1.)?;
    let norm = Tensor::new(&[w as f32, h as f32, w as f32, h as f32], dev)?;
    let xyxy = Tensor::cat(&[x0, y0, x1, y1], D::Minus1)?
        .broadcast_mul(&m.max_keepdim(D::Minus1)?)?
        .broadcast_div(&norm)?;
    let part = |i| xyxy.narrow(D::Minus1, i, 1);
    let (x0, y0, x1, y1) = (part(0)?, part(1)?, part(2)?, part(3)?);
    Tensor::cat(
        &[
            ((&x0 + &x1)? / 2.)?,
            ((&y0 + &y1)? / 2.)?,
            (x1 - x0)?,
            (y1 - y0)?,
        ],
        D::Minus1,
    )
}

struct MaskToBox {
    h: usize,
    w: usize,
}

/// HF `mask_to_box_coordinate`: `[x_min, y_min, x_max + 1, y_max + 1] / [w, h, w, h]`, then xyxy -> cxcywh.
fn finish(bounds: Option<[usize; 4]>, h: usize, w: usize) -> [f32; 4] {
    let Some([x0, y0, x1, y1]) = bounds else {
        return [0.; 4];
    };
    let (w, h) = (w as f32, h as f32);
    let (x0, y0, x1, y1) = (
        x0 as f32 / w,
        y0 as f32 / h,
        (x1 + 1) as f32 / w,
        (y1 + 1) as f32 / h,
    );
    [(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0]
}

impl CustomOp1 for MaskToBox {
    fn name(&self) -> &'static str {
        "mask-to-box"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, q, n) = l.shape().dims3()?;
        let CpuStorage::F32(m) = s else {
            candle_core::bail!("mask_to_box CPU path is f32 only");
        };
        let m = &m[l.start_offset()..];
        let mut out = vec![0f32; b * q * 4];
        out.par_chunks_mut(4).enumerate().for_each(|(row, o)| {
            let mut bounds: Option<[usize; 4]> = None;
            for (i, &v) in m[row * n..(row + 1) * n].iter().enumerate() {
                if v > 0. {
                    let (y, x) = (i / self.w, i % self.w);
                    let bb = bounds.get_or_insert([x, y, x, y]);
                    *bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x), bb[3].max(y)];
                }
            }
            o.copy_from_slice(&finish(bounds, self.h, self.w));
        });
        Ok((CpuStorage::F32(out), Shape::from((b, q, 4))))
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

        let (b, q, n) = l.shape().dims3()?;
        if n != self.h * self.w {
            candle_core::bail!("mask_to_box: {n} pixels for a {}x{} grid", self.h, self.w);
        }
        let dev = &s.device;
        let m = s.as_cuda_slice::<f32>()?.slice(l.start_offset()..);
        let mut out = unsafe { dev.alloc::<f32>(b * q * 4)? };
        let func = dev.get_or_load_custom_func(
            crate::cuda_kernels::MASK_TO_BOX,
            crate::cuda_kernels::MODULE,
            crate::cuda_kernels::ptx()?,
        )?;
        let (h, w) = (i32::try_from(self.h)?, i32::try_from(self.w)?);
        let mut builder = func.builder();
        builder.arg(&m);
        builder.arg(&mut out);
        builder.arg(&h);
        builder.arg(&w);
        let cfg = LaunchConfig {
            grid_dim: (u32::try_from(b * q)?, 1, 1),
            block_dim: (crate::cuda_kernels::MASK_TO_BOX_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.w()?;
        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            },
            Shape::from((b, q, 4)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn boxes_match_reference_formula() -> Result<()> {
        let (h, w) = (5, 7);
        let mut m = vec![-1f32; 2 * h * w];
        // row 0: pixels (x=2,y=1) and (x=4,y=3); row 1 empty
        m[w + 2] = 1.;
        m[3 * w + 4] = 1.;
        let t = Tensor::from_vec(m, (1, 2, h * w), &Device::Cpu)?;
        let want = [
            [
                (2. / 7. + 5. / 7.) / 2.,
                (1. / 5. + 4. / 5.) / 2.,
                3. / 7.,
                3. / 5.,
            ],
            [0.; 4],
        ];
        let mut results = vec![("fallback".to_string(), mask_to_box_ops(&t, h, w)?)];
        for dev in crate::test_util::devices()? {
            results.push((format!("{dev:?}"), mask_to_box(&t.to_device(&dev)?, h, w)?));
        }
        for (name, got) in results {
            let got = got.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
            for (g, e) in got[0].iter().zip(&want) {
                for (a, b) in g.iter().zip(e) {
                    assert!((a - b).abs() < 1e-6, "{name} {g:?} vs {e:?}");
                }
            }
        }
        Ok(())
    }
}
