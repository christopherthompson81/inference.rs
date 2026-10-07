use inference_tensor::{DType, Device, Tensor, quantized::ggml_file};

use crate::kv_cache::EitherCache;

/// A quantized model built from a GGML file.
pub trait FromGGML {
    fn from_ggml(
        ct: ggml_file::Content,
        gqa: usize,
        dtype: DType,
    ) -> Result<Self, inference_tensor::Error>
    where
        Self: Sized;
}

/// A quantized model loaded straight from a GGML file, as its pipeline drives it.
pub trait QuantizedModel: Send + Sync {
    fn forward_step(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
    ) -> inference_tensor::Result<Tensor>;
    fn cache(&self) -> &EitherCache;
    fn device(&self) -> &Device;
    fn max_seq_len(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
}
