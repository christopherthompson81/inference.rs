use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::DType;
use std::fs;

use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::AutoDeviceMapParams;
use crate::gguf::Content;
use crate::matformer::MatformerSliceConfig;
use crate::paged_attention::ModelConfigLike;
use crate::pipeline::DeviceMappedModelLoader;
use crate::GGUFArchitecture;
pub(crate) use inference_nn::gguf::metadata::{ContentConfig, ContentMetadata};

fn info_bytes(info: &gguf_file::TensorInfo) -> usize {
    info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size()
}

pub struct GgufDeviceMapLoaderInner<'a, 'f> {
    pub model: &'a Content<'f, fs::File>,
    pub arch: GGUFArchitecture,
}

impl GgufDeviceMapLoaderInner<'_, '_> {
    fn tensor_bytes(&self, name: &str) -> Result<usize> {
        Ok(info_bytes(self.model.tensor_info(name)?))
    }

    fn tensor_bytes_as(&self, name: &str, dtype: DType) -> Result<usize> {
        Ok(self.model.tensor_info(name)?.shape.elem_count() * dtype.size_in_bytes())
    }
}

impl DeviceMappedModelLoader for GgufDeviceMapLoaderInner<'_, '_> {
    fn mapped_max_act_size_elems(
        &self,
        _config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };
        let num_heads = self.model.get_metadata()[&format!("{}.attention.head_count", self.arch)]
            .to_u32()? as usize;
        Ok(max_batch_size * num_heads * max_seq_len.min(&ATTENTION_CHUNK_SIZE))
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        _config: &str,
        _dtype: DType,
        _weight_pack_factor: usize,
        _quantization: Option<&crate::pipeline::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let size_in_bytes = match self.arch {
            GGUFArchitecture::Llama | GGUFArchitecture::Mistral3 => {
                let token_embd = self.tensor_bytes_as("token_embd.weight", DType::F32)?;
                let output_norm = self.tensor_bytes_as("output_norm.weight", DType::F32)?;
                let output = if !self.model.has_tensor("output.weight") {
                    self.tensor_bytes("token_embd.weight")?
                } else {
                    self.tensor_bytes("output.weight")?
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Phi2 => {
                let token_embd = self.tensor_bytes_as("token_embd.weight", DType::F32)?;
                let output_norm = self.tensor_bytes_as("output_norm.weight", DType::F32)?
                    + self.tensor_bytes("output_norm.bias")?;
                let output = if !self.model.has_tensor("output.weight") {
                    self.tensor_bytes("token_embd.weight")?
                } else {
                    self.tensor_bytes("output.weight")?
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Phi3 => {
                let token_embd = self.tensor_bytes_as("token_embd.weight", DType::F32)?;
                let output_norm = self.tensor_bytes_as("output_norm.weight", DType::F32)?;
                let output = if !self.model.has_tensor("output.weight") {
                    self.tensor_bytes("token_embd.weight")?
                } else {
                    self.tensor_bytes("output.weight")?
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Qwen2 | GGUFArchitecture::Qwen3 | GGUFArchitecture::Qwen3MoE => {
                let token_embd = self.tensor_bytes_as("token_embd.weight", DType::F32)?;
                let output_norm = self.tensor_bytes_as("output_norm.weight", DType::F32)?;
                let output = if !self.model.has_tensor("output.weight") {
                    self.tensor_bytes("token_embd.weight")?
                } else {
                    self.tensor_bytes("output.weight")?
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Starcoder2 => {
                let token_embd = self.tensor_bytes_as("token_embd.weight", DType::F32)?;
                let output_norm = self.tensor_bytes_as("output_norm.weight", DType::F32)?
                    + self.tensor_bytes("output_norm.bias")?;
                let output = if !self.model.has_tensor("output.weight") {
                    self.tensor_bytes("token_embd.weight")?
                } else {
                    self.tensor_bytes("output.weight")?
                };
                token_embd + output_norm + output
            }
            _ => unimplemented!(),
        };
        Ok(size_in_bytes)
    }
    fn num_layers(&self, _config: &str) -> Result<usize> {
        Ok(self.model.get_metadata()[&format!("{}.block_count", self.arch)].to_u32()? as usize)
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        _dtype: DType,
        _weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let size_in_bytes = match self.arch {
            GGUFArchitecture::Llama => {
                let attn_norm = self.tensor_bytes_as("blk.0.attn_norm.weight", DType::F32)?;
                let ffn_norm = self.tensor_bytes_as("blk.0.ffn_norm.weight", DType::F32)?;

                let attn_q = self.tensor_bytes("blk.0.attn_q.weight")?;
                let attn_k = self.tensor_bytes("blk.0.attn_k.weight")?;
                let attn_v = self.tensor_bytes("blk.0.attn_v.weight")?;
                let attn_output = self.tensor_bytes("blk.0.attn_output.weight")?;

                // MoE or Mlp
                #[allow(clippy::cast_possible_truncation)]
                let n_expert = self
                    .model
                    .get_metadata()
                    .get("expert_count")
                    .map(|x| x.to_u64().unwrap() as usize)
                    .unwrap_or(0);
                let moe_or_mlp = if n_expert <= 1 {
                    let ffn_gate = self.tensor_bytes("blk.0.ffn_gate.weight")?;
                    let ffn_up = self.tensor_bytes("blk.0.ffn_up.weight")?;
                    let ffn_down = self.tensor_bytes("blk.0.ffn_down.weight")?;
                    ffn_gate + ffn_up + ffn_down
                } else {
                    let mut moe_count = 0;
                    moe_count += self.tensor_bytes("blk.0.ffn_gate_inp.weight")?;
                    match self.model.tensor_info("blk.0.ffn_gate_exps.weight") {
                        Ok(feed_forward_gate_exps) => {
                            moe_count += info_bytes(feed_forward_gate_exps);
                            moe_count += self.tensor_bytes("blk.0.ffn_down_exps.weight")?;
                            moe_count += self.tensor_bytes("blk.0.ffn_up_exps.weight")?;
                        }
                        Err(_) => {
                            for i in 0..n_expert {
                                moe_count +=
                                    self.tensor_bytes(&format!("blk.0.ffn_gate.{i}.weight"))?;
                                moe_count +=
                                    self.tensor_bytes(&format!("blk.0.ffn_down.{i}.weight"))?;
                                moe_count +=
                                    self.tensor_bytes(&format!("blk.0.ffn_up.{i}.weight"))?;
                            }
                        }
                    }

                    moe_count
                };
                attn_norm + ffn_norm + attn_q + attn_k + attn_v + attn_output + moe_or_mlp
            }
            GGUFArchitecture::Phi2 => {
                let attn_norm = self.tensor_bytes_as("blk.0.attn_norm.weight", DType::F32)?
                    + self.tensor_bytes("blk.0.attn_norm.bias")?;

                let attn_qkv = self.tensor_bytes("blk.0.attn_qkv.weight")?;
                let attn_output = self.tensor_bytes("blk.0.attn_output.weight")?;

                let ffn_up = self.tensor_bytes("blk.0.ffn_up.weight")?;
                let ffn_down = self.tensor_bytes("blk.0.ffn_down.weight")?;

                attn_norm + attn_qkv + attn_output + ffn_up + ffn_down
            }
            GGUFArchitecture::Phi3 => {
                let attn_norm = self.tensor_bytes_as("blk.0.attn_norm.weight", DType::F32)?;
                let ffn_norm = self.tensor_bytes_as("blk.0.ffn_norm.weight", DType::F32)?;

                let attn_qkv = self.tensor_bytes("blk.0.attn_qkv.weight")?;
                let attn_output = self.tensor_bytes("blk.0.attn_output.weight")?;

                let ffn_up = self.tensor_bytes("blk.0.ffn_up.weight")?;
                let ffn_down = self.tensor_bytes("blk.0.ffn_down.weight")?;

                attn_norm + ffn_norm + attn_qkv + attn_output + ffn_up + ffn_down
            }
            GGUFArchitecture::Qwen2 | GGUFArchitecture::Qwen3 | GGUFArchitecture::Qwen3MoE => {
                let attn_norm = self.tensor_bytes_as("blk.0.attn_norm.weight", DType::F32)?;
                let ffn_norm = self.tensor_bytes_as("blk.0.ffn_norm.weight", DType::F32)?;

                let mut attn_q = self.tensor_bytes("blk.0.attn_q.weight")?;
                if let GGUFArchitecture::Qwen2 = self.arch {
                    attn_q += self.tensor_bytes("blk.0.attn_q.bias")?;
                }
                let mut attn_k = self.tensor_bytes("blk.0.attn_k.weight")?;
                if let GGUFArchitecture::Qwen2 = self.arch {
                    attn_k += self.tensor_bytes("blk.0.attn_k.bias")?;
                }

                let mut attn_v = self.tensor_bytes("blk.0.attn_v.weight")?;
                if let GGUFArchitecture::Qwen2 = self.arch {
                    attn_v += self.tensor_bytes("blk.0.attn_v.bias")?;
                }

                let attn_output = self.tensor_bytes("blk.0.attn_output.weight")?;

                let ffn_gate = if let GGUFArchitecture::Qwen3MoE = self.arch {
                    self.tensor_bytes("blk.0.ffn_gate_exps.weight")?
                } else {
                    self.tensor_bytes("blk.0.ffn_gate.weight")?
                };

                let ffn_up = if let GGUFArchitecture::Qwen3MoE = self.arch {
                    self.tensor_bytes("blk.0.ffn_up_exps.weight")?
                } else {
                    self.tensor_bytes("blk.0.ffn_up.weight")?
                };

                let ffn_down = if let GGUFArchitecture::Qwen3MoE = self.arch {
                    self.tensor_bytes("blk.0.ffn_down_exps.weight")?
                } else {
                    self.tensor_bytes("blk.0.ffn_down.weight")?
                };

                attn_norm
                    + ffn_norm
                    + attn_q
                    + attn_k
                    + attn_v
                    + attn_output
                    + ffn_gate
                    + ffn_up
                    + ffn_down
            }
            GGUFArchitecture::Starcoder2 => {
                let attn_norm = self.tensor_bytes_as("blk.0.attn_norm.weight", DType::F32)?
                    + self.tensor_bytes("blk.0.attn_norm.bias")?;
                let ffn_norm = self.tensor_bytes_as("blk.0.ffn_norm.weight", DType::F32)?
                    + self.tensor_bytes("blk.0.ffn_norm.bias")?;

                let attn_q = self.tensor_bytes("blk.0.attn_q.weight")?
                    + self.tensor_bytes("blk.0.attn_q.bias")?;
                let attn_k = self.tensor_bytes("blk.0.attn_k.weight")?
                    + self.tensor_bytes("blk.0.attn_k.bias")?;
                let attn_v = self.tensor_bytes("blk.0.attn_v.weight")?
                    + self.tensor_bytes("blk.0.attn_v.bias")?;
                let attn_output = self.tensor_bytes("blk.0.attn_output.weight")?
                    + self.tensor_bytes("blk.0.attn_output.bias")?;

                let ffn_up = self.tensor_bytes("blk.0.ffn_up.weight")?
                    + self.tensor_bytes("blk.0.ffn_up.bias")?;
                let ffn_down = self.tensor_bytes("blk.0.ffn_down.weight")?
                    + self.tensor_bytes("blk.0.ffn_down.bias")?;

                attn_norm + ffn_norm + attn_q + attn_k + attn_v + attn_output + ffn_up + ffn_down
            }
            GGUFArchitecture::Mistral3 => {
                let attn_norm = self.tensor_bytes_as("blk.0.attn_norm.weight", DType::F32)?;

                let attn_q = self.tensor_bytes("blk.0.attn_q.weight")?;
                let attn_k = self.tensor_bytes("blk.0.attn_k.weight")?;
                let attn_v = self.tensor_bytes("blk.0.attn_v.weight")?;

                let attn_output = self.tensor_bytes("blk.0.attn_output.weight")?;

                let ffn_norm = self.tensor_bytes_as("blk.0.ffn_norm.weight", DType::F32)?;
                let ffn_up = self.tensor_bytes("blk.0.ffn_up.weight")?;
                let ffn_down = self.tensor_bytes("blk.0.ffn_down.weight")?;
                let ffn_gate = self.tensor_bytes("blk.0.ffn_gate.weight")?;

                attn_norm
                    + attn_q
                    + attn_k
                    + attn_v
                    + attn_output
                    + ffn_norm
                    + ffn_up
                    + ffn_down
                    + ffn_gate
            }

            _ => unimplemented!(),
        };
        Ok(vec![size_in_bytes; self.num_layers(config)?])
    }
    fn model_config(&self, _config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let model_config_metadata: ContentConfig = self.model.into();
        Ok(Box::new(model_config_metadata))
    }
}
