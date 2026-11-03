//! Nemotron-H from llama.cpp's `nemotron_h` / `nemotron_h_moe` GGUFs, bound to transformers' names (`model.*`, each
//! layer's `norm` and `mixer`). The MTP block llama.cpp appends (`nextn`) stays unbound.

use anyhow::Result;
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use inference_tensor::DType;

use super::normal_registry::CanonicalGgufArchitecture;

const ROOT: [(&str, &str); 3] = [
    ("token_embd.weight", "model.embeddings.weight"),
    ("output_norm.weight", "model.norm_f.weight"),
    ("output.weight", "lm_head.weight"),
];

pub fn build_nemotron_h_bindings(
    archive: &GgufArchive,
    architecture: CanonicalGgufArchitecture,
) -> Result<GgufBindingMap> {
    let prefix = architecture.as_str();
    let block_count =
        super::normal_bindings::metadata_usize(archive, &format!("{prefix}.block_count"))?
            .saturating_sub(
                super::normal_bindings::metadata_optional_usize(
                    archive,
                    &format!("{prefix}.nextn_predict_layers"),
                )?
                .unwrap_or(0),
            );
    let mut bindings = GgufBindingMap::new();
    for (source, native) in ROOT {
        if archive.tensors().contains_key(source) {
            bindings.insert(native.to_string(), GgufTensorBinding::tensor(source));
        }
    }
    for name in archive.tensors().keys() {
        let Some((layer, role, suffix)) = super::normal_bindings::parse_block_tensor(name) else {
            continue;
        };
        if layer >= block_count {
            continue;
        }
        let p = format!("model.layers.{layer}");
        let target = match role {
            "attn_norm" | "norm" => format!("{p}.norm.{suffix}"),
            "ssm_in" => format!("{p}.mixer.in_proj.{suffix}"),
            "ssm_conv1d" => format!("{p}.mixer.conv1d.{suffix}"),
            "ssm_dt" => format!("{p}.mixer.dt_bias"),
            "ssm_a" => format!("{p}.mixer.A_log"),
            "ssm_d" => format!("{p}.mixer.D"),
            "ssm_norm" => format!("{p}.mixer.norm.{suffix}"),
            "ssm_out" => format!("{p}.mixer.out_proj.{suffix}"),
            "attn_q" => format!("{p}.mixer.q_proj.{suffix}"),
            "attn_k" => format!("{p}.mixer.k_proj.{suffix}"),
            "attn_v" => format!("{p}.mixer.v_proj.{suffix}"),
            "attn_output" => format!("{p}.mixer.o_proj.{suffix}"),
            "ffn_gate_inp" => format!("{p}.mixer.gate.{suffix}"),
            "exp_probs_b" => format!("{p}.mixer.gate.e_score_correction_bias"),
            "ffn_up_exps" => format!("{p}.mixer.experts.up_proj.{suffix}"),
            "ffn_down_exps" => format!("{p}.mixer.experts.down_proj.{suffix}"),
            "ffn_up_shexp" => format!("{p}.mixer.shared_experts.up_proj.{suffix}"),
            "ffn_down_shexp" => format!("{p}.mixer.shared_experts.down_proj.{suffix}"),
            "ffn_latent_down" => format!("{p}.mixer.fc1_latent_proj.{suffix}"),
            "ffn_latent_up" => format!("{p}.mixer.fc2_latent_proj.{suffix}"),
            "ffn_up" => format!("{p}.mixer.up_proj.{suffix}"),
            "ffn_down" => format!("{p}.mixer.down_proj.{suffix}"),
            _ => continue,
        };
        let mut binding = GgufTensorBinding::tensor(name);
        if matches!(role, "ssm_a" | "ssm_d" | "ssm_norm" | "ssm_conv1d") {
            binding = super::normal_bindings::granite_ssm_binding(archive, role, name, binding)?;
        }
        // transformers routes in f32
        if matches!(role, "ffn_gate_inp" | "exp_probs_b") {
            binding = binding.cast(DType::F32);
        }
        bindings.insert(target, binding);
    }
    Ok(bindings)
}
