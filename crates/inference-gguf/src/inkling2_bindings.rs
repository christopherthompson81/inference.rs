//! Inkling 2 from llama.cpp's `inkling2_loop` GGUFs, bound to the names `inference_models_other::inkling2` reads
//! (transformers' module names where it has them, `loop.*` for the looped layers' message queries and controllers).

use anyhow::Result;
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use inference_tensor::DType;

const ROOT: [(&str, &str); 4] = [
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("token_embd_norm.weight", "model.embed_norm.weight"),
    ("output_norm.weight", "model.norm.weight"),
    ("output.weight", "lm_head.weight"),
];
const FINAL_QUERY: &str = "output_res_query.weight";

pub fn build_inkling2_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    for (source, native) in ROOT {
        bindings.insert(native.to_string(), GgufTensorBinding::tensor(source));
    }
    let row = |source: &str| -> Result<GgufTensorBinding> {
        let width = archive
            .tensor_info(source)?
            .shape()
            .iter()
            .product::<usize>();
        Ok(GgufTensorBinding::tensor(source).reshape(vec![1, width]))
    };
    bindings.insert("model.final_input.query".to_string(), row(FINAL_QUERY)?);
    for name in archive.tensors().keys() {
        let Some((layer, role, suffix)) = super::normal_bindings::parse_block_tensor(name) else {
            continue;
        };
        let p = format!("model.layers.{layer}");
        let a = format!("{p}.self_attn");
        let target = match role {
            "attn_q" => format!("{a}.q_proj.{suffix}"),
            "attn_k" => format!("{a}.k_proj.{suffix}"),
            "attn_v" => format!("{a}.v_proj.{suffix}"),
            "attn_output" => format!("{a}.o_proj.{suffix}"),
            "attn_gate" => format!("{a}.gate_proj.{suffix}"),
            "attn_rel" => format!("{a}.r_proj.{suffix}"),
            "attn_sink_proj" => format!("{a}.sink_proj.{suffix}"),
            "attn_q_clip" => format!("{a}.q_l2_clip.scale"),
            "attn_rel_clip" => format!("{a}.rel_l2_clip.scale"),
            "attn_k_conv" => format!("{a}.k_sconv.weight"),
            "attn_v_conv" => format!("{a}.v_sconv.weight"),
            "attn_k_conv_proj" => format!("{a}.k_sconv.weight_proj.{suffix}"),
            "attn_v_conv_proj" => format!("{a}.v_sconv.weight_proj.{suffix}"),
            "attn_post_conv" => format!("{p}.post_attention_convolution.weight"),
            "ffn_post_conv" => format!("{p}.post_mlp_convolution.weight"),
            "attn_norm" => format!("{p}.input_layernorm.{suffix}"),
            "attn_q_inp_norm" => format!("{p}.q_input_layernorm.{suffix}"),
            "attn_k_inp_norm" => format!("{p}.k_input_layernorm.{suffix}"),
            "attn_v_inp_norm" => format!("{p}.v_input_layernorm.{suffix}"),
            "ffn_norm" => format!("{p}.post_attention_layernorm.{suffix}"),
            "ffn_gate_inp" => format!("{p}.mlp.gate.{suffix}"),
            "exp_probs_b" => format!("{p}.mlp.gate.e_score_correction_bias"),
            "ffn_down_exps" => format!("{p}.mlp.experts.down_proj.{suffix}"),
            "loop_msg_query" => format!("{p}.loop.msg_query"),
            "attn_rel_bias" | "attn_res_query" | "ffn_res_query" | "ffn_gate_up_exps" => {
                bind_reshaped(archive, &p, role, name, &mut bindings)?;
                continue;
            }
            _ => match role
                .strip_prefix("loop_")
                .and_then(|rest| rest.split_once("_ctrl_"))
            {
                Some((controller, part)) => format!("{p}.loop.{controller}.{part}.{suffix}"),
                None => continue,
            },
        };
        let mut binding = GgufTensorBinding::tensor(name);
        // transformers routes in f32
        if matches!(role, "ffn_gate_inp" | "exp_probs_b") {
            binding = binding.cast(DType::F32);
        }
        bindings.insert(target, binding);
    }
    Ok(bindings)
}

// the tensors whose GGUF layout is not the model's: a transposed table, one-row queries and the fused experts
fn bind_reshaped(
    archive: &GgufArchive,
    p: &str,
    role: &str,
    name: &str,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let dims = archive.tensor_info(name)?.shape().to_vec();
    let tensor = GgufTensorBinding::tensor(name);
    match role {
        // llama.cpp keeps the profiles as rows of `[extent, d_rel]`
        "attn_rel_bias" => {
            bindings.insert(
                format!("{p}.self_attn.rel_logits_proj.proj"),
                tensor.permute(vec![1, 0]),
            );
        }
        "attn_res_query" | "ffn_res_query" => {
            let queries = if dims.len() == 1 { 1 } else { dims[0] };
            let width = *dims.last().expect("a query has a width");
            let module = if role == "attn_res_query" {
                "attention_input"
            } else {
                "mlp_input"
            };
            bindings.insert(
                format!("{p}.{module}.query"),
                tensor.reshape(vec![queries, width]),
            );
        }
        _ => {
            // `[experts, gate rows then up rows, hidden]`
            let rows = dims[1] / 2;
            let e = format!("{p}.mlp.experts");
            bindings.insert(
                format!("{e}.gate_proj.weight"),
                tensor.clone().slice(1, 0, rows),
            );
            bindings.insert(format!("{e}.up_proj.weight"), tensor.slice(1, rows, rows));
        }
    }
    Ok(())
}
