use std::sync::{Arc, Mutex, atomic::AtomicUsize};

use inference_tensor::nn::Linear;
use inference_tensor::{DType, Device, Result, Tensor};

use crate::Fp8WeightScaleLayout;
use crate::{
    Fp8ActivationMode, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde,
    UnquantLinear,
};

/// E4M3 linear layer with checkpoint-provided scales.
#[derive(Debug)]
pub struct PerTensorFP8Linear {
    weight: Option<Tensor>,
    weight_shape: [usize; 2],
    device: Device,
    weight_scale_inv: Tensor,
    weight_scale_layout: Fp8WeightScaleLayout,
    #[cfg_attr(not(all(feature = "cuda", feature = "cutile")), allow(dead_code))]
    activation_mode: Fp8ActivationMode,
    #[cfg_attr(not(all(feature = "cuda", feature = "cutile")), allow(dead_code))]
    activation_scale: Option<Tensor>,
    bias: Option<Tensor>,
    dequant_dtype: DType,
    dequantized_weight: Mutex<Option<Tensor>>,
}

struct PerTensorFp8Parts {
    weight: Tensor,
    weight_scale_inv: Tensor,
    weight_scale_layout: Fp8WeightScaleLayout,
    activation_mode: Fp8ActivationMode,
    activation_scale: Option<Tensor>,
    bias: Option<Tensor>,
    dequant_dtype: DType,
}

impl PerTensorFP8Linear {
    fn from_parts(parts: PerTensorFp8Parts) -> Result<Self> {
        let PerTensorFp8Parts {
            weight,
            weight_scale_inv,
            weight_scale_layout,
            activation_mode,
            activation_scale,
            bias,
            dequant_dtype,
        } = parts;
        let (n, k) = weight.dims2()?;
        let device = weight.device().clone();
        let mut layer = Self {
            weight: Some(weight),
            weight_shape: [n, k],
            device,
            weight_scale_inv,
            weight_scale_layout,
            activation_mode,
            activation_scale,
            bias,
            dequant_dtype,
            dequantized_weight: Mutex::new(None),
        };
        layer.validate()?;
        if !layer.register_cutile() {
            let weight = layer.dequantize_uncached()?;
            layer.weight = None;
            layer.dequantized_weight = Mutex::new(Some(weight));
        }
        Ok(layer)
    }

    pub fn from_w8a16(
        weight: Tensor,
        weight_scale_inv: Tensor,
        weight_scale_layout: Fp8WeightScaleLayout,
        bias: Option<Tensor>,
        dequant_dtype: DType,
    ) -> Result<Self> {
        Self::from_parts(PerTensorFp8Parts {
            weight,
            weight_scale_inv,
            weight_scale_layout,
            activation_mode: Fp8ActivationMode::None,
            activation_scale: None,
            bias,
            dequant_dtype,
        })
    }

    fn validate(&self) -> Result<()> {
        let weight = self
            .weight
            .as_ref()
            .ok_or_else(|| inference_tensor::Error::msg("FP8 weight is not retained"))?;
        let [n, k] = self.weight_shape;
        if weight.dtype() != DType::F8E4M3 || self.weight_scale_inv.dtype() != DType::F32 {
            inference_tensor::bail!("FP8 linear requires E4M3 weights and F32 scales")
        }
        let valid = match self.weight_scale_layout {
            Fp8WeightScaleLayout::Tensor => self.weight_scale_inv.elem_count() == 1,
            Fp8WeightScaleLayout::Channel => self.weight_scale_inv.elem_count() == n,
            Fp8WeightScaleLayout::Block([block_n, block_k]) => {
                block_n != 0
                    && block_k != 0
                    && self.weight_scale_inv.dims() == [n.div_ceil(block_n), k.div_ceil(block_k)]
            }
        };
        if !valid {
            inference_tensor::bail!(
                "FP8 scale shape {:?} does not match {:?} for weight [{n}, {k}]",
                self.weight_scale_inv.dims(),
                self.weight_scale_layout
            )
        }
        if !weight.device().same_device(self.weight_scale_inv.device()) {
            inference_tensor::bail!("FP8 weight and scale must be on the same device")
        }
        match self.activation_mode {
            Fp8ActivationMode::None | Fp8ActivationMode::DynamicToken => {
                if self.activation_scale.is_some() {
                    inference_tensor::bail!(
                        "dynamic or A16 FP8 linear cannot have an activation scale"
                    )
                }
            }
            Fp8ActivationMode::StaticTensor => {
                let scale = self.activation_scale.as_ref().ok_or_else(|| {
                    inference_tensor::Error::msg("static FP8 W8A8 requires one activation scale")
                })?;
                if scale.dtype() != DType::F32 || scale.elem_count() != 1 {
                    inference_tensor::bail!("static FP8 W8A8 requires one F32 activation scale")
                }
                if !weight.device().same_device(scale.device()) {
                    inference_tensor::bail!(
                        "FP8 weight and activation scale must be on the same device"
                    )
                }
            }
            Fp8ActivationMode::DynamicBlock(_) => {
                inference_tensor::bail!(
                    "retained FP8 linear does not support dynamic-block activations"
                )
            }
        }
        Ok(())
    }

    fn register_cutile(&self) -> bool {
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if let Some(weight) = self.weight.as_ref()
            && let Device::Cuda(dev) = weight.device()
        {
            let [n, k] = self.weight_shape;
            if self.activation_mode == Fp8ActivationMode::None {
                if crate::cutile::fp8_w8a16_supported(dev, n, k, self.dequant_dtype)
                    && matches!(
                        self.weight_scale_layout,
                        Fp8WeightScaleLayout::Tensor
                            | Fp8WeightScaleLayout::Channel
                            | Fp8WeightScaleLayout::Block([128, 128])
                    )
                {
                    crate::cutile::register_fp8_w8a16_shape(
                        weight,
                        &self.weight_scale_inv,
                        self.weight_scale_layout,
                        self.dequant_dtype,
                    );
                    return true;
                }
            } else if matches!(
                self.activation_mode,
                Fp8ActivationMode::StaticTensor | Fp8ActivationMode::DynamicToken
            ) && matches!(
                self.weight_scale_layout,
                Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel
            ) {
                let scheme = crate::cutile::Fp8W8A8Scheme {
                    weight_scale: self.weight_scale_layout,
                    activation: self.activation_mode,
                    output_dtype: self.dequant_dtype,
                };
                if crate::cutile::fp8_w8a8_supported(dev, n, k, self.dequant_dtype, scheme) {
                    crate::cutile::register_fp8_w8a8_shape(weight, &self.weight_scale_inv, scheme);
                    return true;
                }
            }
        }
        false
    }

    fn dequantize_uncached(&self) -> Result<Tensor> {
        let quantized = self
            .weight
            .as_ref()
            .ok_or_else(|| inference_tensor::Error::msg("FP8 weight is not retained"))?
            .force_contiguous()?;
        let [n, k] = self.weight_shape;
        if n == 0 || k == 0 {
            return Tensor::zeros(&self.weight_shape, self.dequant_dtype, &self.device);
        }
        let (scales, block) = match self.weight_scale_layout {
            Fp8WeightScaleLayout::Tensor => (self.weight_scale_inv.reshape((1, 1))?, [n, k]),
            Fp8WeightScaleLayout::Channel => (self.weight_scale_inv.reshape((n, 1))?, [1, k]),
            Fp8WeightScaleLayout::Block(block) => (self.weight_scale_inv.clone(), block),
        };
        #[cfg(all(feature = "cuda", not(has_blockwise_fp8_kernels)))]
        if matches!(&self.device, Device::Cuda(_)) {
            let device = Device::Cpu;
            let weight = quantized.to_device(&device)?;
            let scales = scales.to_device(&device)?;
            let weight = crate::blockwise_fp8::fp8_blockwise_dequantize(
                &weight,
                &scales,
                block.to_vec(),
                self.dequant_dtype,
            )?;
            return weight.to_device(&self.device);
        }
        crate::blockwise_fp8::fp8_blockwise_dequantize(
            &quantized,
            &scales,
            block.to_vec(),
            self.dequant_dtype,
        )
    }

    fn empty_output(&self, input: &Tensor) -> Result<Option<Tensor>> {
        if input.rank() == 0 || input.dims()[..input.rank() - 1].iter().all(|dim| *dim != 0) {
            return Ok(None);
        }
        let [n, k] = self.weight_shape;
        if input.dim(input.rank() - 1)? != k {
            return Ok(None);
        }
        if !input.device().same_device(&self.device) {
            inference_tensor::bail!("FP8 weight and activation must be on the same device")
        }
        let mut shape = input.dims().to_vec();
        *shape.last_mut().unwrap() = n;
        Tensor::zeros(shape, input.dtype(), input.device()).map(Some)
    }

    #[cfg(all(feature = "cuda", feature = "cutile"))]
    fn try_cutile_w8a16(&self, input: &Tensor) -> Result<Option<Tensor>> {
        if self.activation_mode != Fp8ActivationMode::None
            || input.rank() == 0
            || !matches!(input.dtype(), DType::BF16 | DType::F16)
        {
            return Ok(None);
        }
        let Device::Cuda(dev) = input.device() else {
            return Ok(None);
        };
        let Some(weight) = self.weight.as_ref() else {
            return Ok(None);
        };
        let [n, k] = self.weight_shape;
        if !crate::cutile::fp8_w8a16_supported(dev, n, k, input.dtype())
            || !matches!(
                self.weight_scale_layout,
                Fp8WeightScaleLayout::Tensor
                    | Fp8WeightScaleLayout::Channel
                    | Fp8WeightScaleLayout::Block([128, 128])
            )
        {
            return Ok(None);
        }
        let source_shape = input.dims().to_vec();
        let rows = source_shape[..source_shape.len() - 1]
            .iter()
            .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
            .ok_or_else(|| {
                inference_tensor::Error::msg("FP8 W8A16 activation shape overflows usize")
            })?;
        let input = input.reshape((rows, k))?;
        let result = crate::cutile::cutile_fp8_w8a16(
            &input,
            weight,
            &self.weight_scale_inv,
            self.weight_scale_layout,
        )?;
        let mut output_shape = source_shape[..source_shape.len() - 1].to_vec();
        output_shape.push(n);
        let result = result.reshape(output_shape)?;
        match &self.bias {
            Some(bias) => result.broadcast_add(bias).map(Some),
            None => Ok(Some(result)),
        }
    }

    #[cfg(all(feature = "cuda", feature = "cutile"))]
    fn try_cutile_w8a8(&self, input: &Tensor) -> Result<Option<Tensor>> {
        if !matches!(
            self.activation_mode,
            Fp8ActivationMode::StaticTensor | Fp8ActivationMode::DynamicToken
        ) || input.rank() == 0
            || !matches!(
                self.weight_scale_layout,
                Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel
            )
            || !matches!(input.dtype(), DType::BF16 | DType::F16)
        {
            return Ok(None);
        }
        let Device::Cuda(dev) = input.device() else {
            return Ok(None);
        };
        let Some(weight) = self.weight.as_ref() else {
            return Ok(None);
        };
        let [n, k] = self.weight_shape;
        let scheme = crate::cutile::Fp8W8A8Scheme {
            weight_scale: self.weight_scale_layout,
            activation: self.activation_mode,
            output_dtype: input.dtype(),
        };
        if !crate::cutile::fp8_w8a8_supported(dev, n, k, input.dtype(), scheme) {
            return Ok(None);
        }
        let result = crate::cutile::cutile_fp8_w8a8(
            input,
            crate::cutile::CutileFp8W8A8Args {
                weight,
                weight_scales: &self.weight_scale_inv,
                scheme,
                activation_scale: self.activation_scale.as_ref(),
            },
        )?;
        match &self.bias {
            Some(bias) => result.broadcast_add(bias).map(Some),
            None => Ok(Some(result)),
        }
    }
}

fn normalize_weight_scale(
    weight: &Tensor,
    weight_scale: Tensor,
    layout: Fp8WeightScaleLayout,
) -> Result<Tensor> {
    let (n, k) = weight.dims2()?;
    let weight_scale = weight_scale.to_dtype(DType::F32)?;
    match layout {
        Fp8WeightScaleLayout::Tensor => {
            if weight_scale.elem_count() != 1 {
                inference_tensor::bail!("FP8 tensor scale must contain one element")
            }
            weight_scale.reshape(())
        }
        Fp8WeightScaleLayout::Channel => {
            if weight_scale.elem_count() != n {
                inference_tensor::bail!(
                    "FP8 channel scale has {} elements, expected {n}",
                    weight_scale.elem_count()
                )
            }
            weight_scale.reshape(n)
        }
        Fp8WeightScaleLayout::Block([block_n, block_k]) => {
            if block_n == 0 || block_k == 0 {
                inference_tensor::bail!("FP8 block scale dimensions must be nonzero")
            }
            let shape = (n.div_ceil(block_n), k.div_ceil(block_k));
            if weight_scale.elem_count() != shape.0 * shape.1 {
                inference_tensor::bail!(
                    "FP8 block scale has {} elements, expected {} for shape {:?}",
                    weight_scale.elem_count(),
                    shape.0 * shape.1,
                    shape
                )
            }
            weight_scale.reshape(shape)
        }
    }
}

pub fn fp8_w8a16_linear(
    weight: Tensor,
    weight_scale: Tensor,
    weight_scale_layout: Fp8WeightScaleLayout,
    bias: Option<Tensor>,
    dequant_dtype: DType,
) -> Result<Arc<dyn QuantMethod>> {
    let weight_scale = normalize_weight_scale(&weight, weight_scale, weight_scale_layout)?;
    Ok(Arc::new(PerTensorFP8Linear::from_w8a16(
        weight,
        weight_scale,
        weight_scale_layout,
        bias,
        dequant_dtype,
    )?))
}

pub struct Fp8W8A8LinearArgs {
    pub weight: Tensor,
    pub weight_scale: Tensor,
    pub weight_scale_layout: Fp8WeightScaleLayout,
    pub activation_mode: Fp8ActivationMode,
    pub activation_scale: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub dequant_dtype: DType,
}

pub fn fp8_w8a8_linear(args: Fp8W8A8LinearArgs) -> Result<Arc<dyn QuantMethod>> {
    let Fp8W8A8LinearArgs {
        weight,
        weight_scale,
        weight_scale_layout,
        activation_mode,
        activation_scale,
        bias,
        dequant_dtype,
    } = args;
    let weight_scale = normalize_weight_scale(&weight, weight_scale, weight_scale_layout)?;
    let activation_scale = activation_scale
        .map(|scale| scale.to_dtype(DType::F32)?.reshape(()))
        .transpose()?;
    Ok(Arc::new(PerTensorFP8Linear::from_parts(
        PerTensorFp8Parts {
            weight,
            weight_scale_inv: weight_scale,
            weight_scale_layout,
            activation_mode,
            activation_scale,
            bias,
            dequant_dtype,
        },
    )?))
}

impl QuantMethod for PerTensorFP8Linear {
    fn new(method: QuantMethodConfig) -> inference_tensor::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::PerTensorFP8 {
                weight,
                weight_scale_inv,
                activation_mode,
                activation_scale,
                bias,
                dequant_dtype,
            } => Self::from_parts(PerTensorFp8Parts {
                weight,
                weight_scale_inv,
                weight_scale_layout: Fp8WeightScaleLayout::Tensor,
                activation_mode,
                activation_scale,
                bias,
                dequant_dtype,
            }),
            _ => unreachable!(),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        let cached = self.dequantized_weight.lock().unwrap();
        if let Some(weight) = cached.as_ref() {
            return Ok(weight.clone());
        }
        drop(cached);
        self.dequantize_uncached()
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(result) = self.empty_output(x)? {
            return Ok(result);
        }
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if let Some(result) = self.try_cutile_w8a16(x)? {
            return Ok(result);
        }
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if let Some(result) = self.try_cutile_w8a8(x)? {
            return Ok(result);
        }
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            self.dequantize_w()?,
            self.bias.clone(),
        )))?;
        unquant.forward(x)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        inference_tensor::bail!("PerTensorFP8Linear does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F8E4M3, self.device.clone())
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            self.dequant_dtype,
            self.device.clone(),
            self.weight_shape.to_vec(),
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
        let weight = self.dequantize_w()?;
        Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(weight, self.bias.clone()),
        ))?)
        .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }
}

// Serialization structure (same as UnquantLinear):
//
impl QuantizedSerde for PerTensorFP8Linear {
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "pertensor-fp8-linear"
    }
}

#[cfg(test)]
mod tests {
    use float8::F8E4M3;
    use inference_tensor::{DType, Device, Result, Tensor};

    use super::{Fp8W8A8LinearArgs, PerTensorFP8Linear, fp8_w8a8_linear, fp8_w8a16_linear};
    use crate::{Fp8ActivationMode, Fp8WeightScaleLayout};

    #[test]
    fn w8a16_constructor_normalizes_checkpoint_scale_ranks() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![F8E4M3::from_f32(1.0); 16], (4, 4), &device)?;
        let channel = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (4, 1), &device)?;
        let layer = fp8_w8a16_linear(
            weight.clone(),
            channel,
            Fp8WeightScaleLayout::Channel,
            None,
            DType::F32,
        )?;
        let values = layer.dequantize_w()?.to_vec2::<f32>()?;
        assert_eq!(values[0], vec![1.0; 4]);
        assert_eq!(values[3], vec![4.0; 4]);

        let block = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 1, 2, 1), &device)?;
        let layer = fp8_w8a16_linear(
            weight,
            block,
            Fp8WeightScaleLayout::Block([2, 2]),
            None,
            DType::F32,
        )?;
        let values = layer.dequantize_w()?.to_vec2::<f32>()?;
        assert_eq!(values[0], vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(values[3], vec![3.0, 3.0, 4.0, 4.0]);
        Ok(())
    }

    #[test]
    fn w8a8_constructor_enforces_activation_contract() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![F8E4M3::from_f32(1.0); 16], (4, 4), &device)?;
        let weight_scale = Tensor::new(1f32, &device)?;
        let args = |activation_mode, activation_scale| Fp8W8A8LinearArgs {
            weight: weight.clone(),
            weight_scale: weight_scale.clone(),
            weight_scale_layout: Fp8WeightScaleLayout::Tensor,
            activation_mode,
            activation_scale,
            bias: None,
            dequant_dtype: DType::F32,
        };
        assert!(fp8_w8a8_linear(args(Fp8ActivationMode::StaticTensor, None)).is_err());
        assert!(
            fp8_w8a8_linear(args(
                Fp8ActivationMode::DynamicToken,
                Some(Tensor::new(1f32, &device)?),
            ))
            .is_err()
        );
        assert!(fp8_w8a8_linear(args(Fp8ActivationMode::DynamicBlock(4), None)).is_err());
        Ok(())
    }

    #[test]
    fn unsupported_runtime_keeps_only_dequantized_weight() -> Result<()> {
        let device = Device::Cpu;
        let layer = PerTensorFP8Linear::from_w8a16(
            Tensor::from_vec(vec![F8E4M3::from_f32(1.0); 16], (4, 4), &device)?,
            Tensor::new(1f32, &device)?,
            Fp8WeightScaleLayout::Tensor,
            None,
            DType::F32,
        )?;
        assert!(layer.weight.is_none());
        assert!(layer.dequantized_weight.lock().unwrap().is_some());
        Ok(())
    }

    #[test]
    fn retained_fp8_linear_preserves_empty_leading_dimensions() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![F8E4M3::from_f32(1.0); 12], (3, 4), &device)?;
        let layer = fp8_w8a16_linear(
            weight,
            Tensor::new(1f32, &device)?,
            Fp8WeightScaleLayout::Tensor,
            Some(Tensor::zeros(3, DType::F32, &device)?),
            DType::F32,
        )?;
        let output = layer.forward(&Tensor::zeros((2, 0, 4), DType::F32, &device)?)?;
        assert_eq!(output.dims(), [2, 0, 3]);
        Ok(())
    }

    #[test]
    fn apply_isq_requantizes_the_dequantized_weight() -> Result<()> {
        let device = Device::Cpu;
        let layer = fp8_w8a16_linear(
            Tensor::from_vec(vec![F8E4M3::from_f32(1.5); 4 * 32], (4, 32), &device)?,
            Tensor::new(2f32, &device)?,
            Fp8WeightScaleLayout::Tensor,
            Some(Tensor::ones(4, DType::F32, &device)?),
            DType::F32,
        )?;
        crate::unquantized::assert_requantizes_its_dequantized_weight(layer)
    }
}
