//! Loads diffusion-model weights from a GGUF file, with linears served quantized and other tensors materialized.

use std::{collections::HashMap, path::Path, sync::Arc};

use candle_core::{DType, Device, Result};
use inference_quant::{
    GgufArchive, GgufBindingMap, GgufTensorBinding, GgufWeightSource, ShardedVarBuilder,
};

pub const GGUF_EXTENSION: &str = "gguf";
// GGUFs converted from ComfyUI checkpoints keep its wrapper prefix on every FLUX tensor.
const COMFY_FLUX_PREFIX: &str = "model.diffusion_model.";

/// A var builder over `path`, naming each GGUF tensor by `native_name` (tensors it maps to `None` are skipped).
pub fn var_builder(
    path: &Path,
    native_name: impl Fn(&str) -> Option<String>,
    dtype: DType,
    device: &Device,
) -> Result<(ShardedVarBuilder, HashMap<String, Vec<usize>>)> {
    let archive = Arc::new(GgufArchive::open_file(path)?);
    let mut bindings = GgufBindingMap::new();
    for gguf_name in archive.tensors().keys() {
        if let Some(native) = native_name(gguf_name) {
            bindings.insert(native, GgufTensorBinding::tensor(gguf_name.clone()));
        }
    }
    let source = Arc::new(GgufWeightSource::new(archive, &bindings, dtype)?);
    let shapes = source.tensor_shapes().clone();
    Ok((source.sharded_var_builder(device.clone()), shapes))
}

/// A FLUX GGUF's tensor names as the model loads them (BFL naming).
pub fn flux_native_name(gguf_name: &str) -> Option<String> {
    Some(
        gguf_name
            .strip_prefix(COMFY_FLUX_PREFIX)
            .unwrap_or(gguf_name)
            .to_string(),
    )
}

/// llama.cpp's `t5encoder` tensor names as the Hugging Face T5 encoder names the T5 model loads.
pub fn t5_native_name(gguf_name: &str) -> Option<String> {
    match gguf_name {
        "token_embd.weight" => return Some("shared.weight".to_string()),
        "enc.output_norm.weight" => return Some("encoder.final_layer_norm.weight".to_string()),
        _ => {}
    }
    let rest = gguf_name.strip_prefix("enc.blk.")?;
    let (block, tensor) = rest.split_once('.')?;
    let native = match tensor {
        "attn_q.weight" => "layer.0.SelfAttention.q.weight",
        "attn_k.weight" => "layer.0.SelfAttention.k.weight",
        "attn_v.weight" => "layer.0.SelfAttention.v.weight",
        "attn_o.weight" => "layer.0.SelfAttention.o.weight",
        "attn_rel_b.weight" => "layer.0.SelfAttention.relative_attention_bias.weight",
        "attn_norm.weight" => "layer.0.layer_norm.weight",
        "ffn_gate.weight" => "layer.1.DenseReluDense.wi_0.weight",
        "ffn_up.weight" => "layer.1.DenseReluDense.wi_1.weight",
        "ffn_down.weight" => "layer.1.DenseReluDense.wo.weight",
        "ffn_norm.weight" => "layer.1.layer_norm.weight",
        _ => return None,
    };
    Some(format!("encoder.block.{block}.{native}"))
}

#[cfg(test)]
mod tests {
    use super::{flux_native_name, t5_native_name};

    #[test]
    fn flux_names_drop_the_comfy_wrapper_prefix() {
        for name in ["img_in.weight", "model.diffusion_model.img_in.weight"] {
            assert_eq!(flux_native_name(name).as_deref(), Some("img_in.weight"));
        }
    }

    #[test]
    fn llama_cpp_t5_names_map_onto_the_hf_encoder() {
        let cases = [
            ("token_embd.weight", "shared.weight"),
            ("enc.output_norm.weight", "encoder.final_layer_norm.weight"),
            (
                "enc.blk.0.attn_rel_b.weight",
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            ),
            (
                "enc.blk.23.ffn_gate.weight",
                "encoder.block.23.layer.1.DenseReluDense.wi_0.weight",
            ),
            (
                "enc.blk.7.attn_o.weight",
                "encoder.block.7.layer.0.SelfAttention.o.weight",
            ),
        ];
        for (gguf, native) in cases {
            assert_eq!(t5_native_name(gguf).as_deref(), Some(native), "{gguf}");
        }
        assert_eq!(t5_native_name("dec.blk.0.attn_q.weight"), None);
    }
}
