pub(crate) use inference_models_gemma::{gemma, gemma2};
pub(crate) use inference_models_llama::{llama, mistral, mixtral, smollm3};
pub(crate) use inference_models_other::{
    deepseek2, deepseek3, glm4, glm4_moe, glm4_moe_lite, gpt_oss, granite, hunyuan_v1_dense,
    hunyuan_v1_moe, lfm2, starcoder2,
};
pub(crate) use inference_models_phi::{phi2, phi3, phi3_5_moe};
pub(crate) use inference_models_qwen::{qwen2, qwen3, qwen3_moe, qwen3_next};

pub(crate) mod quantized_llama;
