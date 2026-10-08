use anyhow::Result;
use inference_quant::{GgufArchive, GgufBindingMap};

use super::multimodal_binding_utils::{
    TensorInventory, bind_llama_text, bind_required, bind_siglip_vision, validate_architecture,
    validate_projector,
};

const FAMILY: &str = "Idefics3/SmolVLM";

pub fn build_idefics3_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, "llama")?;
    validate_projector(archive, "idefics3")?;
    build_idefics3_bindings_from_inventory(&TensorInventory::from_archive(archive))
}

fn build_idefics3_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_llama_text(
        inventory,
        &mut bindings,
        "model.text_model",
        "lm_head",
        FAMILY,
    )?;
    bind_required(
        inventory,
        &mut bindings,
        "model.connector.modality_projection.proj.weight",
        "mm.model.fc.weight",
    )?;
    bind_siglip_vision(inventory, &mut bindings, "model.vision_model", FAMILY)?;
    Ok(bindings)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use inference_quant::GgufTensorBinding;

    use super::*;
    use crate::multimodal_binding_utils::{binding_sources, siglip_test_tensors};

    #[test]
    fn maps_complete_idefics3_inventory() {
        let tensors = tensor_inventory();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_idefics3_bindings_from_inventory(&inventory).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(binding_sources(&bindings), expected);
        assert_eq!(
            bindings.get("model.connector.modality_projection.proj.weight"),
            Some(&GgufTensorBinding::tensor("mm.model.fc.weight"))
        );
        assert_eq!(
            bindings.get("model.vision_model.encoder.layers.0.self_attn.q_proj.bias"),
            Some(&GgufTensorBinding::tensor("v.blk.0.attn_q.bias"))
        );
    }

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 8]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![8, 8]),
            ("mm.model.fc.weight".to_string(), vec![8, 32]),
        ];
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
