use inference_tensor::Result;

use crate::{QuantMethod, QuantizeOntoGuard, QuantizedSerde};

#[derive(Debug, Clone)]
pub struct DummyLayerInfo {
    pub context: String,
    pub prefix: String,
    pub missing_tensors: Vec<String>,
}

impl DummyLayerInfo {
    pub fn unknown() -> Self {
        Self {
            context: "unknown".to_string(),
            prefix: "<unknown>".to_string(),
            missing_tensors: Vec::new(),
        }
    }

    pub fn message(&self, action: &str) -> String {
        let missing = if self.missing_tensors.is_empty() {
            "<unknown>".to_string()
        } else {
            self.missing_tensors.join(", ")
        };
        format!(
            "DummyLayer reached {action} for {} at prefix `{}`. Missing tensor path(s): {missing}. Dummy layers are only valid as temporary UQFF placeholders and must be replaced before inference.",
            self.context, self.prefix
        )
    }
}

#[derive(Debug, Clone)]
pub struct DummyLayer {
    info: DummyLayerInfo,
}

impl DummyLayer {
    pub fn placeholder(info: DummyLayerInfo) -> Self {
        Self { info }
    }

    pub fn info(&self) -> &DummyLayerInfo {
        &self.info
    }
}

impl QuantMethod for DummyLayer {
    fn new(_method: crate::QuantMethodConfig) -> inference_tensor::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            info: DummyLayerInfo::unknown(),
        })
    }
    fn dequantize_w(&self) -> Result<inference_tensor::Tensor> {
        inference_tensor::bail!("{}", self.info.message("dequantization"))
    }
    fn add_delta_w(
        &self,
        _delta: &inference_tensor::Tensor,
    ) -> inference_tensor::Result<std::sync::Arc<dyn QuantMethod>> {
        inference_tensor::bail!("{}", self.info.message("LoRA delta application"))
    }
    fn apply_isq(
        self: std::sync::Arc<Self>,
        _dtype: Option<crate::IsqType>,
        _device: inference_tensor::Device,
        _n_quantized: &std::sync::atomic::AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> inference_tensor::Result<std::sync::Arc<dyn QuantMethod>> {
        // This is necessary for the immediate ISQ
        Ok(self)
    }
    fn dtype_and_device(&self) -> (inference_tensor::DType, inference_tensor::Device) {
        (inference_tensor::DType::F32, inference_tensor::Device::Cpu)
    }
    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            inference_tensor::DType::F32,
            inference_tensor::Device::Cpu,
            Vec::new(),
            request,
            false,
        ))
    }
    fn forward_raw(
        &self,
        _a: &inference_tensor::Tensor,
    ) -> inference_tensor::Result<inference_tensor::Tensor> {
        inference_tensor::bail!("{}", self.info.message("forward pass"))
    }
    fn quantized_act_type(&self) -> Option<inference_tensor::DType> {
        None
    }

    fn dummy_info(&self) -> Option<crate::DummyLayerInfo> {
        Some(self.info.clone())
    }
}

impl QuantizedSerde for DummyLayer {
    fn name(&self) -> &'static str {
        "dummy"
    }
}
