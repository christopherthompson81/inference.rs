use super::*;

/// Elementwise multiply and activation. The following activations are supported:
/// - `gelu`
/// - `silu`
/// - `relu`
///
/// This is equivalent to:
/// `act(a) * b`
///
/// With supported dtypes (F16, BF16, F32) and fused activations,
/// this uses a fused kernel for better performance by eliminating intermediate
/// memory allocation. Optimized implementations are available for:
/// - CUDA: Custom CUDA kernel with vec4 optimization
/// - Metal: Native Metal kernel
/// - CPU: Rayon-parallelized implementation
fn glu_activation_type(act: Activation) -> Option<inference_quant::GluActivationType> {
    match act {
        Activation::Silu | Activation::Swish => Some(inference_quant::GluActivationType::Silu),
        Activation::NewGelu | Activation::GeluPytorchTanh => {
            Some(inference_quant::GluActivationType::Gelu)
        }
        Activation::Gelu => Some(inference_quant::GluActivationType::GeluErf),
        Activation::Relu => Some(inference_quant::GluActivationType::Relu),
        Activation::Sigmoid => Some(inference_quant::GluActivationType::Sigmoid),
        _ => None,
    }
}

fn candle_glu_activation_type(
    act: candle_nn::Activation,
) -> Option<inference_quant::GluActivationType> {
    match act {
        candle_nn::Activation::Silu | candle_nn::Activation::Swish => {
            Some(inference_quant::GluActivationType::Silu)
        }
        candle_nn::Activation::NewGelu | candle_nn::Activation::GeluPytorchTanh => {
            Some(inference_quant::GluActivationType::Gelu)
        }
        candle_nn::Activation::Gelu => Some(inference_quant::GluActivationType::GeluErf),
        candle_nn::Activation::Relu => Some(inference_quant::GluActivationType::Relu),
        candle_nn::Activation::Sigmoid => Some(inference_quant::GluActivationType::Sigmoid),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatedActivationOrder {
    GateUp,
    UpGate,
}

pub fn mul_and_act(a: &Tensor, b: &Tensor, act: Activation) -> Result<Tensor> {
    // Check if we can use the fused kernel (works on CUDA, Metal, and CPU)
    if matches!(a.dtype(), DType::F16 | DType::BF16 | DType::F32) && a.dtype() == b.dtype() {
        if let Some(activation_type) = glu_activation_type(act) {
            return inference_quant::fused_glu(a, b, activation_type);
        }
    }

    a.apply(&act)? * b
}

pub fn try_fused_gated_projection(
    gate: &Tensor,
    value: &Tensor,
    act: Activation,
    projection: &dyn inference_quant::QuantMethod,
) -> Result<Option<Tensor>> {
    let Some(activation) = glu_activation_type(act) else {
        return Ok(None);
    };
    inference_quant::try_forward_fused_quantized_glu(gate, value, projection, activation)
}

pub fn mul_and_candle_act(a: &Tensor, b: &Tensor, act: candle_nn::Activation) -> Result<Tensor> {
    // Check if we can use the fused kernel (works on CUDA, Metal, and CPU)
    if matches!(a.dtype(), DType::F16 | DType::BF16 | DType::F32) && a.dtype() == b.dtype() {
        if let Some(activation_type) = candle_glu_activation_type(act) {
            return inference_quant::fused_glu(a, b, activation_type);
        }
    }

    a.apply(&act)? * b
}

pub fn split_mul_and_act(xs: &Tensor, split_size: usize, act: Activation) -> Result<Tensor> {
    split_mul_and_act_order(xs, split_size, act, GatedActivationOrder::GateUp)
}

pub fn try_fused_split_glu_quantized_forward(
    xs: &Tensor,
    split_size: usize,
    act: Activation,
    projection: &dyn inference_quant::QuantMethod,
) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    {
        let Some(activation) = glu_activation_type(act) else {
            return Ok(None);
        };
        projection.try_forward_fused_split_glu(xs, split_size, activation)
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (xs, split_size, act, projection);
        Ok(None)
    }
}

pub fn split_mul_and_act_order(
    xs: &Tensor,
    split_size: usize,
    act: Activation,
    order: GatedActivationOrder,
) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let Some(expected_last_dim) = split_size.checked_mul(2) else {
        candle_core::bail!("split_mul_and_act split size overflow: {split_size}");
    };
    if last_dim != expected_last_dim {
        candle_core::bail!(
            "split_mul_and_act expected last dim {expected_last_dim}, got {last_dim}"
        );
    }
    if order == GatedActivationOrder::GateUp
        && matches!(xs.dtype(), DType::F16 | DType::BF16 | DType::F32)
    {
        if let Some(activation_type) = glu_activation_type(act) {
            return inference_quant::fused_split_glu(xs, split_size, activation_type);
        }
    }

    let first = xs.narrow(D::Minus1, 0, split_size)?;
    let second = xs.narrow(D::Minus1, split_size, split_size)?;
    match order {
        GatedActivationOrder::GateUp => mul_and_act(&first, &second, act),
        GatedActivationOrder::UpGate => mul_and_act(&second, &first, act),
    }
}

#[derive(Clone)]
pub struct MergedDenseProjection {
    proj: Arc<dyn inference_quant::QuantMethod>,
    originals: Vec<Arc<dyn inference_quant::QuantMethod>>,
    output_dims: Vec<usize>,
}

impl MergedDenseProjection {
    /// Wrap a packed projection group: `packed` owns the fused weight, `constituents` are its
    /// view-backed layers used for the dynamic-LoRA fallback path.
    pub fn from_packed(group: &inference_quant::PackedColumnParallel) -> Self {
        Self {
            proj: group.packed.clone(),
            originals: group.constituents.clone(),
            output_dims: group.rows_per_rank.clone(),
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let Some(ys) = self.forward_packed(xs)? else {
            return self.originals.iter().map(|proj| proj.forward(xs)).collect();
        };
        let mut parts = Vec::with_capacity(self.output_dims.len());
        let mut offset = 0;
        for &dim in &self.output_dims {
            parts.push(ys.narrow(D::Minus1, offset, dim)?);
            offset += dim;
        }
        Ok(parts)
    }

    pub fn forward_packed(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        if self
            .originals
            .iter()
            .any(|proj| proj.is_dynamic_lora_active())
        {
            Ok(None)
        } else {
            self.proj.forward(xs).map(Some)
        }
    }

    pub fn activation_quantization_scheme_for(
        &self,
        xs: &Tensor,
    ) -> Option<inference_quant::ActivationQuantizationScheme> {
        if self
            .originals
            .iter()
            .any(|proj| proj.is_dynamic_lora_active())
        {
            None
        } else {
            self.proj.activation_quantization_scheme_for(xs)
        }
    }

    pub fn preferred_activation_scale_layout_for(
        &self,
        xs: &Tensor,
    ) -> Option<inference_quant::ActivationScaleLayout> {
        if self
            .originals
            .iter()
            .any(|proj| proj.is_dynamic_lora_active())
        {
            None
        } else {
            self.proj.preferred_activation_scale_layout_for(xs)
        }
    }

    pub fn forward_quantized_packed(
        &self,
        activation: &inference_quant::QuantizedActivation,
    ) -> Result<Tensor> {
        if self
            .originals
            .iter()
            .any(|proj| proj.is_dynamic_lora_active())
        {
            candle_core::bail!(
                "packed projection cannot use a prequantized activation with active dynamic LoRA"
            )
        }
        self.proj.forward_quantized(activation)
    }

    pub fn forward_quantized(
        &self,
        activation: &inference_quant::QuantizedActivation,
    ) -> Result<Vec<Tensor>> {
        let ys = self.forward_quantized_packed(activation)?;
        let mut parts = Vec::with_capacity(self.output_dims.len());
        let mut offset = 0;
        for &dim in &self.output_dims {
            parts.push(ys.narrow(D::Minus1, offset, dim)?);
            offset += dim;
        }
        Ok(parts)
    }
}

/// Feed-forward path for quantized gate/up/down projections.
pub fn quantized_ffn(
    xs: &Tensor,
    gate: &dyn inference_quant::QuantMethod,
    up: &dyn inference_quant::QuantMethod,
    down: &dyn inference_quant::QuantMethod,
    act: Activation,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(activation_type) = glu_activation_type(act) {
        if let Some(out) =
            inference_quant::try_fused_quantized_ffn(xs, gate, up, down, activation_type)?
        {
            return Ok(out);
        }
        if let Some(inter) =
            inference_quant::try_fused_quantized_gate_up(xs, gate, up, activation_type)?
        {
            return down.forward(&inter);
        }
    }

    #[cfg(feature = "metal")]
    if let Some(activation_type) = glu_activation_type(act) {
        if let Some(inter) =
            inference_quant::try_fused_gate_up_metal(xs, gate, up, activation_type)?
        {
            return down.forward(&inter);
        }
    }

    if xs.device().is_cpu() {
        if let Some(mut out) = inference_quant::try_fused_gemv_shared_lhs_cpu(xs, &[gate, up])? {
            let rhs = out.pop().unwrap();
            let lhs = out.pop().unwrap();
            let inter = mul_and_act(&lhs, &rhs, act)?;
            return down.forward(&inter);
        }
    }

    let lhs = gate.forward(xs)?;
    let rhs = up.forward(xs)?;
    if let Some(output) = try_fused_gated_projection(&lhs, &rhs, act, down)? {
        return Ok(output);
    }
    let inter = mul_and_act(&lhs, &rhs, act)?;
    down.forward(&inter)
}

pub fn qkv_projections(
    xs: &Tensor,
    q_proj: &dyn inference_quant::QuantMethod,
    k_proj: &dyn inference_quant::QuantMethod,
    v_proj: &dyn inference_quant::QuantMethod,
) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if let Some(qkv) = inference_quant::try_fused_quantized_qkv(xs, q_proj, k_proj, v_proj)? {
        return Ok(qkv);
    }

    #[cfg(feature = "metal")]
    if let Some(qkv) = inference_quant::try_fused_qkv_metal(xs, q_proj, k_proj, v_proj)? {
        return Ok(qkv);
    }

    if xs.device().is_cpu() {
        if let Some(mut out) =
            inference_quant::try_fused_gemv_shared_lhs_cpu(xs, &[q_proj, k_proj, v_proj])?
        {
            let v = out.pop().unwrap();
            let k = out.pop().unwrap();
            let q = out.pop().unwrap();
            return Ok((q, k, v));
        }
    }

    Ok((
        q_proj.forward(xs)?,
        k_proj.forward(xs)?,
        v_proj.forward(xs)?,
    ))
}
