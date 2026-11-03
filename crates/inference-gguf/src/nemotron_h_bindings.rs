//! Nemotron-H from llama.cpp's `nemotron_h` / `nemotron_h_moe` GGUFs, bound to transformers' names (`model.*`, each
//! layer's `norm` and `mixer`). The MTP block llama.cpp folds in after the stack (`nextn`) binds to the checkpoint's
//! `mtp.layers.0` (input fusion and attention) and `mtp.layers.1` (MoE and final norm).

use anyhow::Result;
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use inference_tensor::DType;

use super::normal_registry::CanonicalGgufArchitecture;

// the MTP block's attention half and its MoE half, as the HF checkpoint names them
const MTP_MIXER: &str = "mtp.layers.0";
const MTP_FFN: &str = "mtp.layers.1";

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
        let nextn = layer == block_count;
        if layer > block_count {
            continue;
        }
        // `<weight>.scale`, NVFP4's tensor-wide scale, which the shared parser leaves on the role
        let (role, suffix) = match role.strip_suffix(".scale") {
            Some(base) if suffix.is_empty() => (base, "scale"),
            _ => (role, suffix),
        };
        // a trunk layer holds one mixer or one feed-forward under one name; the MTP block holds both, under two
        let (m, f) = match nextn {
            true => (MTP_MIXER.to_string(), MTP_FFN.to_string()),
            false => {
                let p = format!("model.layers.{layer}");
                (p.clone(), p)
            }
        };
        let target = match role {
            "attn_norm" | "norm" => format!("{m}.norm.{suffix}"),
            "post_attention_norm" if nextn => format!("{f}.norm.{suffix}"),
            "nextn.enorm" => format!("{m}.enorm.{suffix}"),
            "nextn.hnorm" => format!("{m}.hnorm.{suffix}"),
            "nextn.eh_proj" => format!("{m}.eh_proj.{suffix}"),
            "nextn.shared_head_norm" => format!("{f}.final_layernorm.{suffix}"),
            "ssm_in" => format!("{m}.mixer.in_proj.{suffix}"),
            "ssm_conv1d" => format!("{m}.mixer.conv1d.{suffix}"),
            "ssm_dt" => format!("{m}.mixer.dt_bias"),
            "ssm_a" => format!("{m}.mixer.A_log"),
            "ssm_d" => format!("{m}.mixer.D"),
            "ssm_norm" => format!("{m}.mixer.norm.{suffix}"),
            "ssm_out" => format!("{m}.mixer.out_proj.{suffix}"),
            "attn_q" => format!("{m}.mixer.q_proj.{suffix}"),
            "attn_k" => format!("{m}.mixer.k_proj.{suffix}"),
            "attn_v" => format!("{m}.mixer.v_proj.{suffix}"),
            "attn_output" => format!("{m}.mixer.o_proj.{suffix}"),
            "ffn_gate_inp" => format!("{f}.mixer.gate.{suffix}"),
            "exp_probs_b" => format!("{f}.mixer.gate.e_score_correction_bias"),
            "ffn_up_exps" => format!("{f}.mixer.experts.up_proj.{suffix}"),
            "ffn_down_exps" => format!("{f}.mixer.experts.down_proj.{suffix}"),
            "ffn_up_shexp" => format!("{f}.mixer.shared_experts.up_proj.{suffix}"),
            "ffn_down_shexp" => format!("{f}.mixer.shared_experts.down_proj.{suffix}"),
            "ffn_latent_down" => format!("{f}.mixer.fc1_latent_proj.{suffix}"),
            "ffn_latent_up" => format!("{f}.mixer.fc2_latent_proj.{suffix}"),
            "ffn_up" => format!("{f}.mixer.up_proj.{suffix}"),
            "ffn_down" => format!("{f}.mixer.down_proj.{suffix}"),
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
