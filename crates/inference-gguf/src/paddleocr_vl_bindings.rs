use anyhow::Result;
use inference_quant::{GgufArchive, GgufBindingMap};

use super::multimodal_binding_utils::{
    TensorInventory, bind_llama_text, bind_required_linear, bind_siglip_vision,
    validate_architecture, validate_projector,
};

const FAMILY: &str = "PaddleOCR-VL";
pub const ARCHITECTURE: &str = "paddleocr";
pub const PROJECTOR: &str = "paddleocr";

/// llama.cpp's PaddleOCR-VL: Ernie 4.5 text under Llama names, the SigLIP tower and its `mlp_AR` connector.
pub fn build_paddleocr_vl_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, ARCHITECTURE)?;
    validate_projector(archive, PROJECTOR)?;
    build_paddleocr_vl_bindings_from_inventory(&TensorInventory::from_archive(archive))
}

fn build_paddleocr_vl_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_llama_text(inventory, &mut bindings, "model", "lm_head", FAMILY)?;
    // the converter drops the tower's packing position table and pooling head, which the model never reads
    bind_siglip_vision(inventory, &mut bindings, "visual.vision_model", FAMILY)?;
    for (native, source) in [
        ("mlp_AR.pre_norm", "mm.input_norm"),
        ("mlp_AR.linear_1", "mm.1"),
        ("mlp_AR.linear_2", "mm.2"),
    ] {
        bind_required_linear(inventory, &mut bindings, native, source)?;
    }
    Ok(bindings)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use inference_quant::GgufTensorBinding;

    use super::*;
    use crate::multimodal_binding_utils::{binding_sources, siglip_test_tensors};

    #[test]
    fn maps_complete_paddleocr_vl_inventory() {
        let tensors = tensor_inventory();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_paddleocr_vl_bindings_from_inventory(&inventory).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        assert_eq!(
            bindings.get("mlp_AR.pre_norm.bias"),
            Some(&GgufTensorBinding::tensor("mm.input_norm.bias"))
        );
        assert_eq!(
            bindings.get("model.layers.0.self_attn.o_proj.weight"),
            Some(&GgufTensorBinding::tensor("blk.0.attn_output.weight"))
        );
        assert_eq!(
            bindings.get("visual.vision_model.encoder.layers.0.mlp.fc1.weight"),
            Some(&GgufTensorBinding::tensor("v.blk.0.ffn_up.weight"))
        );
    }

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 8]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![8, 8]),
        ];
        for (name, shape) in [
            ("mm.input_norm", vec![8]),
            ("mm.1", vec![32, 32]),
            ("mm.2", vec![32, 8]),
        ] {
            tensors.push((format!("{name}.weight"), shape));
            tensors.push((format!("{name}.bias"), vec![8]));
        }
        for role in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            tensors.push((format!("blk.0.{role}.weight"), vec![8, 8]));
        }
        for role in ["attn_norm", "ffn_norm"] {
            tensors.push((format!("blk.0.{role}.weight"), vec![8]));
        }
        tensors.extend(siglip_test_tensors());
        tensors
    }
}
