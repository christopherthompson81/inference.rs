use float8::F8E4M3;
use inference_tensor::{CpuStorage, CustomOp1, DType, Result, Tensor};

struct Fp8ToDtype {
    target_dtype: DType,
}

impl CustomOp1 for Fp8ToDtype {
    fn name(&self) -> &'static str {
        "fp8-to-dtype"
    }

    fn cpu_fwd(
        &self,
        input_s: &inference_tensor::CpuStorage,
        input_l: &inference_tensor::Layout,
    ) -> inference_tensor::Result<(inference_tensor::CpuStorage, inference_tensor::Shape)> {
        let CpuStorage::F8E4M3(input) = input_s else {
            inference_tensor::bail!("Expected F8E4M3 input!");
        };
        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            inference_tensor::bail!("Expected input to have start offset 0, continuous");
        }

        let output = match self.target_dtype {
            DType::F32 => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    output.push(val.to_f32());
                }
                CpuStorage::F32(output)
            }
            DType::F16 => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    output.push(half::f16::from_f32(val.to_f32()));
                }
                CpuStorage::F16(output)
            }
            DType::BF16 => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    output.push(half::bf16::from_f32(val.to_f32()));
                }
                CpuStorage::BF16(output)
            }
            other => {
                inference_tensor::bail!("Unsupported target dtype for FP8 conversion: {other:?}")
            }
        };

        Ok((output, input_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        input_s: &inference_tensor::MetalStorage,
        input_l: &inference_tensor::Layout,
    ) -> Result<(inference_tensor::MetalStorage, inference_tensor::Shape)> {
        use inference_tensor::backend::BackendStorage;

        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            inference_tensor::bail!("Expected input to have start offset 0, continuous");
        }

        let device = input_s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("fp8-to-dtype");

        let num_elements = input_l.shape().elem_count();
        let out_shape = input_l.shape().clone();

        let output = device.new_buffer(num_elements, self.target_dtype, "fp8-to-dtype-output")?;

        crate::metal_kernels::call_fp8_to_dtype(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            self.target_dtype,
            input_s.buffer(),
            &output,
            num_elements,
        )
        .map_err(inference_tensor::Error::wrap)?;

        let newstorage = inference_tensor::MetalStorage::new(
            output,
            device.clone(),
            num_elements,
            self.target_dtype,
        );
        Ok((newstorage, out_shape))
    }
}

struct DtypeToFp8 {
    source_dtype: DType,
}

impl CustomOp1 for DtypeToFp8 {
    fn name(&self) -> &'static str {
        "dtype-to-fp8"
    }

    fn cpu_fwd(
        &self,
        input_s: &inference_tensor::CpuStorage,
        input_l: &inference_tensor::Layout,
    ) -> inference_tensor::Result<(inference_tensor::CpuStorage, inference_tensor::Shape)> {
        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            inference_tensor::bail!("Expected input to have start offset 0, continuous");
        }

        let output = match (self.source_dtype, input_s) {
            (DType::F32, CpuStorage::F32(input)) => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    let clamped = val.clamp(-448.0, 448.0);
                    output.push(F8E4M3::from_f32(clamped));
                }
                CpuStorage::F8E4M3(output)
            }
            (DType::F16, CpuStorage::F16(input)) => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    let f32_val = val.to_f32();
                    let clamped = f32_val.clamp(-448.0, 448.0);
                    output.push(F8E4M3::from_f32(clamped));
                }
                CpuStorage::F8E4M3(output)
            }
            (DType::BF16, CpuStorage::BF16(input)) => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    let f32_val = val.to_f32();
                    let clamped = f32_val.clamp(-448.0, 448.0);
                    output.push(F8E4M3::from_f32(clamped));
                }
                CpuStorage::F8E4M3(output)
            }
            _ => inference_tensor::bail!("Mismatched source dtype and storage type"),
        };

        Ok((output, input_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        input_s: &inference_tensor::MetalStorage,
        input_l: &inference_tensor::Layout,
    ) -> Result<(inference_tensor::MetalStorage, inference_tensor::Shape)> {
        use inference_tensor::backend::BackendStorage;

        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            inference_tensor::bail!("Expected input to have start offset 0, continuous");
        }

        let device = input_s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("dtype-to-fp8");

        let num_elements = input_l.shape().elem_count();
        let out_shape = input_l.shape().clone();

        let output = device.new_buffer(num_elements, DType::F8E4M3, "dtype-to-fp8-output")?;

        crate::metal_kernels::call_dtype_to_fp8(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            self.source_dtype,
            input_s.buffer(),
            &output,
            num_elements,
        )
        .map_err(inference_tensor::Error::wrap)?;

        let newstorage = inference_tensor::MetalStorage::new(
            output,
            device.clone(),
            num_elements,
            DType::F8E4M3,
        );
        Ok((newstorage, out_shape))
    }
}

/// Convert an FP8 tensor to another dtype.
pub(crate) fn fp8_to_dtype(input: &Tensor, target_dtype: DType) -> Result<Tensor> {
    if input.dtype() != DType::F8E4M3 {
        inference_tensor::bail!("Input tensor must be F8E4M3, got {:?}", input.dtype());
    }
    // candle's CUDA cast is the CUDA conversion; the op below covers CPU and Metal
    if input.device().is_cuda() {
        return input.to_dtype(target_dtype);
    }
    input.apply_op1_no_bwd(&Fp8ToDtype { target_dtype })
}

/// Convert a tensor to FP8.
pub(crate) fn dtype_to_fp8(input: &Tensor) -> Result<Tensor> {
    let source_dtype = input.dtype();
    if !matches!(source_dtype, DType::F32 | DType::F16 | DType::BF16) {
        inference_tensor::bail!(
            "Input tensor must be F32, F16, or BF16, got {:?}",
            source_dtype
        );
    }
    if input.device().is_cuda() {
        return input.to_dtype(DType::F8E4M3);
    }
    input.apply_op1_no_bwd(&DtypeToFp8 { source_dtype })
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use float8::F8E4M3;
    use inference_tensor::{DType, Device, IndexOp, Result, Tensor};

    fn bits(t: &Tensor) -> Result<Vec<Vec<u8>>> {
        let rows = t.to_device(&Device::Cpu)?.to_vec2::<F8E4M3>()?;
        Ok(rows
            .into_iter()
            .map(|row| row.iter().map(F8E4M3::to_bits).collect())
            .collect())
    }

    #[test]
    fn cuda_fp8_casts_match_the_cpu_bit_for_bit() -> Result<()> {
        let device = Device::new_cuda(0)?;
        // spans subnormals, signed zeros, the rounding grid and both saturation edges (448 is E4M3's largest finite)
        let values = Tensor::arange(0f32, 2048f32, &Device::Cpu)?
            .affine(0.37, -380.0)?
            .reshape((32, 64))?;
        let mut edges = vec![
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            448.0,
            -448.0,
            464.0,
            -1e-9,
        ];
        edges.resize(64, 1.0);
        let edges = Tensor::from_vec(edges, (1, 64), &Device::Cpu)?;
        let values = Tensor::cat(&[&values, &(&values * 1e-3)?, &(&values * 3.0)?, &edges], 0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let cpu = values.to_dtype(dtype)?;
            let expected = super::dtype_to_fp8(&cpu)?;
            let actual = super::dtype_to_fp8(&cpu.to_device(&device)?)?;
            assert_eq!(bits(&actual)?, bits(&expected)?, "{dtype:?} to fp8");
            let back = super::fp8_to_dtype(&expected.to_device(&device)?, dtype)?;
            assert_eq!(back.dtype(), dtype);
            assert_eq!(
                back.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                super::fp8_to_dtype(&expected, dtype)?
                    .to_dtype(DType::F32)?
                    .to_vec2::<f32>()?,
                "fp8 to {dtype:?}"
            );
            // a strided FP8 CUDA view copies to contiguous
            let fp8 = expected.to_device(&device)?.t()?.contiguous()?;
            assert_eq!(bits(&fp8)?, bits(&expected.t()?.contiguous()?)?);
            // a strided CUDA view converts too
            let strided = super::dtype_to_fp8(&cpu.to_device(&device)?.t()?)?;
            assert_eq!(
                bits(&strided.i((.., 0..1))?.t()?.contiguous()?)?,
                bits(&expected.i(0..1)?)?
            );
        }
        let nan = Tensor::new(&[[f32::NAN, -f32::NAN]], &device)?;
        let nan = super::dtype_to_fp8(&nan)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        assert!(nan[0].iter().all(|v| v.is_nan()), "{nan:?}");
        Ok(())
    }
}
