//! 1D and 2D Convolutions
//!
use crate::{op::BackpropOp, op::Op, Error, Result, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv1D {
    pub(crate) b_size: usize,
    // Maybe we should have a version without l_in as this bit depends on the input and not only on
    // the weights.
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    // `c_in` and `c_out` count one group's channels
    pub(crate) groups: usize,
}

impl ParamsConv1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in + 2 * self.padding - self.dilation * (self.k_size - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out * self.groups, l_out]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose1D {
    pub(crate) b_size: usize,
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.k_size - 1)
            + self.output_padding
            + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    // `c_in` and `c_out` count one group's channels
    pub(crate) groups: usize,
}

impl ParamsConv2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h + 2 * self.padding - self.dilation * (self.k_h - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w + 2 * self.padding - self.dilation * (self.k_w - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![
            self.b_size,
            self.c_out * self.groups,
            self.out_h(),
            self.out_w(),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h - 1) * self.stride + self.dilation * (self.k_h - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w - 1) * self.stride + self.dilation * (self.k_w - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

impl Tensor {
    // CUDA runs many small groups in one direct kernel (few wide groups are faster as per-group GEMMs, 3090: 2x256 ch
    // 10x slower direct, 32x16 7x faster); the conv backprop ops carry no group count, so tracked graphs split
    fn native_grouped_conv(&self, kernel: &Self, groups: usize, c_in_per_group: usize) -> bool {
        groups >= c_in_per_group
            && self.device().is_cuda()
            && !self.track_op()
            && !kernel.track_op()
    }

    fn conv1d_single_group(&self, kernel: &Self, params: &ParamsConv1D) -> Result<Self> {
        let storage =
            self.storage()
                .conv1d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv1D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (c_out, c_in_k, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k * groups {
            Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "the number of in-channels on the input doesn't match the kernel size",
            }
            .bt())?
        }
        if c_out % groups != 0 {
            crate::bail!("out_channel {c_out} is not divisible by the number of groups {groups}")
        }

        let params = ParamsConv1D {
            b_size,
            l_in,
            c_out: c_out / groups,
            c_in: c_in / groups,
            k_size,
            padding,
            stride,
            dilation,
            groups,
        };
        if groups == 1 || self.native_grouped_conv(kernel, groups, c_in_k) {
            self.conv1d_single_group(kernel, &params)
        } else {
            let params = ParamsConv1D {
                groups: 1,
                ..params
            };
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv_transpose1d_single_group(
        &self,
        kernel: &Self,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        let storage = self.storage().conv_transpose1d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose1D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D transposed convolution over the input tensor.
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (c_in_k, c_out, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        if c_in % groups != 0 {
            crate::bail!("in_channel {c_in} is not divisible by the number of groups")
        }
        let params = ParamsConvTranspose1D {
            b_size,
            l_in,
            k_size,
            c_out,
            c_in: c_in / groups,
            padding,
            output_padding,
            stride,
            dilation,
        };
        if groups == 1 {
            self.conv_transpose1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv_transpose1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv2d_single_group(&self, kernel: &Self, params: &ParamsConv2D) -> Result<Self> {
        let storage =
            self.storage()
                .conv2d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv2D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_out, c_in_k, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k * groups {
            crate::bail!(
                "in_channel mismatch between input ({c_in}, groups {groups}) and kernel ({c_in_k})"
            )
        }
        if c_out % groups != 0 {
            crate::bail!("out_channel {c_out} is not divisible by the number of groups {groups}")
        }
        let params = ParamsConv2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out: c_out / groups,
            c_in: c_in / groups,
            padding,
            stride,
            dilation,
            groups,
        };
        if groups == 1 || self.native_grouped_conv(kernel, groups, c_in_k) {
            self.conv2d_single_group(kernel, &params)
        } else {
            let params = ParamsConv2D {
                groups: 1,
                ..params
            };
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv2d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    /// Applies a 2D transposed convolution over the input tensor.
    pub fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_in_k, c_out, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        let params = ParamsConvTranspose2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out,
            c_in,
            padding,
            output_padding,
            stride,
            dilation,
        };
        let storage = self.storage().conv_transpose2d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            &params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose2D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Result, Tensor};

    // (dtype, max abs error against the f32 CPU reference)
    const DTYPES: [(DType, f32); 3] = [(DType::F32, 1e-4), (DType::F16, 2e-2), (DType::BF16, 8e-2)];

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        a.to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .sub(b)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()
    }

    #[test]
    fn grouped_conv1d_on_cuda_matches_the_per_group_split() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        // (channels, groups, out per group, kernel, padding, stride, dilation)
        let cases = [
            (64, 64, 1, 15, 7, 1, 1),
            (32, 32, 2, 5, 4, 2, 2),
            (24, 4, 3, 3, 1, 1, 1),
        ];
        for (c, groups, m, k, padding, stride, dilation) in cases {
            // A transposed input is not contiguous
            let x = Tensor::randn(0f32, 1., (2, 37, c), &Device::Cpu)?.transpose(1, 2)?;
            let w = Tensor::randn(0f32, 1., (groups * m, c / groups, k), &Device::Cpu)?;
            let expected = x.conv1d(&w, padding, stride, dilation, groups)?;
            for (dtype, tolerance) in DTYPES {
                let (xc, wc) = (
                    x.to_dtype(dtype)?.to_device(&cuda)?,
                    w.to_dtype(dtype)?.to_device(&cuda)?,
                );
                let got = xc.conv1d(&wc, padding, stride, dilation, groups)?;
                assert_eq!(got.dims(), expected.dims());
                let diff = max_abs_diff(&got, &expected)?;
                assert!(diff <= tolerance, "{dtype:?} c={c} groups={groups}: {diff}");
            }
        }
        Ok(())
    }

    #[test]
    fn grouped_conv_rejects_out_channels_that_do_not_split_into_groups() -> Result<()> {
        let x = Tensor::zeros((1, 4, 8), DType::F32, &Device::Cpu)?;
        assert!(x
            .conv1d(
                &Tensor::zeros((6, 1, 3), DType::F32, &Device::Cpu)?,
                1,
                1,
                1,
                4
            )
            .is_err());
        let x = x.unsqueeze(3)?;
        assert!(x
            .conv2d(
                &Tensor::zeros((2, 1, 3, 1), DType::F32, &Device::Cpu)?,
                1,
                1,
                1,
                4
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn grouped_conv2d_on_cuda_matches_the_per_group_split() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        // (channels, groups, out per group, kernel, padding, stride, dilation)
        let cases = [
            (64, 64, 1, 3, 1, 1, 1),
            (16, 16, 2, 5, 2, 2, 1),
            (12, 4, 2, 3, 2, 1, 2),
        ];
        for (c, groups, m, k, padding, stride, dilation) in cases {
            let x = Tensor::randn(0f32, 1., (2, 13, 11, c), &Device::Cpu)?.permute((0, 3, 1, 2))?;
            let w = Tensor::randn(0f32, 1., (groups * m, c / groups, k, k), &Device::Cpu)?;
            let expected = x.conv2d(&w, padding, stride, dilation, groups)?;
            for (dtype, tolerance) in DTYPES {
                let (xc, wc) = (
                    x.to_dtype(dtype)?.to_device(&cuda)?,
                    w.to_dtype(dtype)?.to_device(&cuda)?,
                );
                let got = xc.conv2d(&wc, padding, stride, dilation, groups)?;
                assert_eq!(got.dims(), expected.dims());
                let diff = max_abs_diff(&got, &expected)?;
                assert!(diff <= tolerance, "{dtype:?} c={c} groups={groups}: {diff}");
            }
        }
        Ok(())
    }

    fn time_ms(cuda: &Device, iters: u32, f: impl Fn() -> Result<Tensor>) -> Result<f64> {
        f()?;
        cuda.synchronize()?;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            f()?;
        }
        cuda.synchronize()?;
        Ok(start.elapsed().as_secs_f64() * 1e3 / f64::from(iters))
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_depthwise_conv_split_vs_direct() -> Result<()> {
        const ITERS: u32 = 20;
        let cuda = Device::new_cuda(0)?;
        let x2 = Tensor::randn(0f32, 1., (1, 640, 32, 32), &cuda)?.to_dtype(DType::BF16)?;
        let w2 = Tensor::randn(0f32, 1., (640, 1, 3, 3), &cuda)?.to_dtype(DType::BF16)?;
        let x1 = Tensor::randn(0f32, 1., (1, 1024, 1500), &cuda)?.to_dtype(DType::BF16)?;
        let w1 = Tensor::randn(0f32, 1., (1024, 1, 15), &cuda)?.to_dtype(DType::BF16)?;
        // A tracked input takes the per-group split
        let (x2v, x1v) = (crate::Var::from_tensor(&x2)?, crate::Var::from_tensor(&x1)?);
        let split2 = time_ms(&cuda, ITERS, || x2v.as_tensor().conv2d(&w2, 1, 1, 1, 640))?;
        let direct2 = time_ms(&cuda, ITERS, || x2.conv2d(&w2, 1, 1, 1, 640))?;
        let split1 = time_ms(&cuda, ITERS, || x1v.as_tensor().conv1d(&w1, 7, 1, 1, 1024))?;
        let direct1 = time_ms(&cuda, ITERS, || x1.conv1d(&w1, 7, 1, 1, 1024))?;
        println!("conv2d dw 3x3 640ch 32x32 bf16: split {split2:.3} ms, routed {direct2:.3} ms");
        println!("conv1d dw k15 1024ch L1500 bf16: split {split1:.3} ms, routed {direct1:.3} ms");
        for (groups, c) in [(2usize, 512usize), (8, 512), (32, 512)] {
            let xg = Tensor::randn(0f32, 1., (1, c, 32, 32), &cuda)?.to_dtype(DType::BF16)?;
            let wg =
                Tensor::randn(0f32, 1., (c, c / groups, 3, 3), &cuda)?.to_dtype(DType::BF16)?;
            let xgv = crate::Var::from_tensor(&xg)?;
            let split = time_ms(&cuda, ITERS, || {
                xgv.as_tensor().conv2d(&wg, 1, 1, 1, groups)
            })?;
            let direct = time_ms(&cuda, ITERS, || xg.conv2d(&wg, 1, 1, 1, groups))?;
            println!("conv2d 3x3 {c}ch groups {groups} 32x32 bf16: split {split:.3} ms, routed {direct:.3} ms");
        }
        Ok(())
    }
}
