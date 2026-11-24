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
            // each NVFP4 expert's second-level scale (llama.cpp's `<weight>.scale`), the gate's and up's halves sharing
            // the fused stack's
            "ffn_down_exps.scale" => format!("{p}.mlp.experts.down_proj.scale"),
            "ffn_gate_up_exps.scale" => {
                for half in ["gate_proj", "up_proj"] {
                    bindings.insert(
                        format!("{p}.mlp.experts.{half}.scale"),
                        GgufTensorBinding::tensor(name),
                    );
                }
                continue;
            }
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
    bind_groups(archive, &mut bindings);
    Ok(bindings)
}

// a layer's projections of one input concatenated, so decode runs one product: where the parts share a block type
fn bind_groups(archive: &GgufArchive, bindings: &mut GgufBindingMap) {
    let groups = [
        (
            "q_group",
            &["attn_q", "attn_gate", "attn_rel", "attn_sink_proj"][..],
        ),
        ("k_group", &["attn_k", "attn_k_conv_proj"][..]),
        ("v_group", &["attn_v", "attn_v_conv_proj"][..]),
    ];
    for layer in 0.. {
        if !archive
            .tensors()
            .contains_key(&format!("blk.{layer}.attn_q.weight"))
        {
            break;
        }
        for (group, roles) in groups {
            let names = roles
                .iter()
                .map(|role| format!("blk.{layer}.{role}.weight"))
                .filter(|name| archive.tensors().contains_key(name))
                .collect::<Vec<_>>();
            let types = names
                .iter()
                .filter_map(|name| {
                    archive
                        .tensor_info(name)
                        .ok()
                        .map(|info| info.dtype().raw())
                })
                .collect::<std::collections::HashSet<_>>();
            if names.len() < 2 || types.len() != 1 {
                continue;
            }
            let parts = names.into_iter().map(GgufTensorBinding::tensor).collect();
            bindings.insert(
                format!("model.layers.{layer}.self_attn.{group}.weight"),
                GgufTensorBinding::concat(parts, 0),
            );
        }
    }
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

// the GGUF roles of a layer's tensors by the model's names under the layer, as `build_inkling2_bindings` binds them
const LAYER_ROLES: [(&str, &str); 22] = [
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("self_attn.gate_proj.weight", "attn_gate.weight"),
    ("self_attn.r_proj.weight", "attn_rel.weight"),
    ("self_attn.sink_proj.weight", "attn_sink_proj.weight"),
    ("self_attn.q_l2_clip.scale", "attn_q_clip.weight"),
    ("self_attn.rel_l2_clip.scale", "attn_rel_clip.weight"),
    ("self_attn.k_sconv.weight", "attn_k_conv.weight"),
    ("self_attn.v_sconv.weight", "attn_v_conv.weight"),
    (
        "self_attn.k_sconv.weight_proj.weight",
        "attn_k_conv_proj.weight",
    ),
    (
        "self_attn.v_sconv.weight_proj.weight",
        "attn_v_conv_proj.weight",
    ),
    ("post_attention_convolution.weight", "attn_post_conv.weight"),
    ("post_mlp_convolution.weight", "ffn_post_conv.weight"),
    ("input_layernorm.weight", "attn_norm.weight"),
    ("q_input_layernorm.weight", "attn_q_inp_norm.weight"),
    ("k_input_layernorm.weight", "attn_k_inp_norm.weight"),
    ("v_input_layernorm.weight", "attn_v_inp_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("mlp.gate.weight", "ffn_gate_inp.weight"),
    ("mlp.gate.e_score_correction_bias", "exp_probs_b.bias"),
];

/// A tensor the model reads, as llama.cpp's GGUF stores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inkling2GgufTensor {
    pub name: String,
    pub dims: Vec<usize>,
    /// Stored as the transpose of the model's `[rows, cols]`.
    pub transposed: bool,
}

/// The GGUF tensor `build_inkling2_bindings` binds to the model's `name` of `shape`, the routed experts aside: the
/// inverse of its bindings, for writing a GGUF from a checkpoint read by the model's names.
pub fn inkling2_gguf_tensor(name: &str, shape: &[usize]) -> Option<Inkling2GgufTensor> {
    let plain = |gguf: String| {
        Some(Inkling2GgufTensor {
            name: gguf,
            dims: shape.to_vec(),
            transposed: false,
        })
    };
    // a query of one row is stored as its vector
    let query = |gguf: String| {
        let dims = match shape {
            [1, width] => vec![*width],
            _ => shape.to_vec(),
        };
        Some(Inkling2GgufTensor {
            name: gguf,
            dims,
            transposed: false,
        })
    };
    if let Some((gguf, _)) = ROOT.iter().find(|(_, native)| *native == name) {
        return plain(gguf.to_string());
    }
    if name == "model.final_input.query" {
        return query(FINAL_QUERY.to_string());
    }
    let (layer, tail) = name.strip_prefix("model.layers.")?.split_once('.')?;
    let blk = |role: &str| format!("blk.{layer}.{role}");
    if let Some((_, role)) = LAYER_ROLES.iter().find(|(native, _)| *native == tail) {
        return plain(blk(role));
    }
    match tail {
        "self_attn.rel_logits_proj.proj" => {
            let &[rows, cols] = shape else { return None };
            Some(Inkling2GgufTensor {
                name: blk("attn_rel_bias.weight"),
                dims: vec![cols, rows],
                transposed: true,
            })
        }
        "attention_input.query" => query(blk("attn_res_query.weight")),
        "mlp_input.query" => query(blk("ffn_res_query.weight")),
        "loop.msg_query" => plain(blk("loop_msg_query.weight")),
        _ => {
            let (controller, part) = tail
                .strip_prefix("loop.")?
                .strip_suffix(".weight")?
                .split_once('.')?;
            plain(blk(&format!("loop_{controller}_ctrl_{part}.weight")))
        }
    }
}

// the official checkpoint's names: a layer's experts, its fused gate/up stack (gate and up rows interleaved) and its
// down stack
const OFFICIAL_EXPERTS: &str = "model.llm.layers";
const OFFICIAL_MTP: &str = "mtp.safetensors";

/// The official checkpoint's stack of a GGUF expert tensor (`blk.N.ffn_gate_up_exps.weight`, `blk.N.ffn_down_exps.weight`).
fn official_stack(tensor: &str) -> Option<inference_quant::nvfp4::gguf_blocks::ModeloptStack> {
    let (layer, role, suffix) = super::normal_bindings::parse_block_tensor(tensor)?;
    if suffix != "weight" {
        return None;
    }
    let (stored, interleaved) = match role {
        "ffn_gate_up_exps" => ("w13", true),
        "ffn_down_exps" => ("w2", false),
        _ => return None,
    };
    Some(inference_quant::nvfp4::gguf_blocks::ModeloptStack {
        weight: format!("{OFFICIAL_EXPERTS}.{layer}.mlp.experts.{stored}_weight"),
        interleaved,
    })
}

/// An Inkling 2 checkpoint's expert stacks (thinkingmachines' releases), for a GGUF quantizer to read in place of a
/// GGUF's: NVFP4 ones as stored, MXFP8 ones as values; `None` for another checkpoint.
pub fn official_checkpoint_experts(
    dir: &std::path::Path,
) -> Result<Option<inference_quant::nvfp4::gguf_blocks::ModeloptExperts>> {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let inkling = config["text_config"]["model_type"]
        .as_str()
        .is_some_and(|ty| ty.starts_with("inkling2"));
    if !inkling {
        return Ok(None);
    }
    let mut paths = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .filter(|p| p.file_name().is_none_or(|name| name != OFFICIAL_MTP))
        .collect::<Vec<_>>();
    paths.sort();
    // SAFETY: the checkpoint's files are only read
    let experts = unsafe {
        inference_quant::nvfp4::gguf_blocks::ModeloptExperts::open(
            &paths,
            Box::new(official_stack),
        )?
    };
    Ok(Some(experts))
}

#[cfg(test)]
mod tests {
    // each tensor the model reads goes back to the GGUF name and layout its binding reads it from
    #[test]
    fn the_models_names_go_back_to_the_gguf_tensors_they_bind() {
        use super::{Inkling2GgufTensor, inkling2_gguf_tensor};
        let gguf = |name: &str, shape: &[usize]| inkling2_gguf_tensor(name, shape).unwrap();
        let tensor = |name: &str, dims: &[usize], transposed| Inkling2GgufTensor {
            name: name.to_string(),
            dims: dims.to_vec(),
            transposed,
        };
        assert_eq!(
            gguf("lm_head.weight", &[8, 4]),
            tensor("output.weight", &[8, 4], false)
        );
        assert_eq!(
            gguf("model.final_input.query", &[1, 4]),
            tensor("output_res_query.weight", &[4], false)
        );
        assert_eq!(
            gguf("model.layers.3.self_attn.sink_proj.weight", &[2, 4]),
            tensor("blk.3.attn_sink_proj.weight", &[2, 4], false)
        );
        assert_eq!(
            gguf("model.layers.3.self_attn.rel_logits_proj.proj", &[16, 1024]),
            tensor("blk.3.attn_rel_bias.weight", &[1024, 16], true)
        );
        assert_eq!(
            gguf("model.layers.5.attention_input.query", &[3, 4]),
            tensor("blk.5.attn_res_query.weight", &[3, 4], false)
        );
        assert_eq!(
            gguf("model.layers.4.attention_input.query", &[1, 4]),
            tensor("blk.4.attn_res_query.weight", &[4], false)
        );
        assert_eq!(
            gguf("model.layers.7.loop.ffn.forget.weight", &[1, 4]),
            tensor("blk.7.loop_ffn_ctrl_forget.weight", &[1, 4], false)
        );
        assert_eq!(
            gguf("model.layers.7.mlp.gate.e_score_correction_bias", &[8]),
            tensor("blk.7.exp_probs_b.bias", &[8], false)
        );
        assert!(
            inkling2_gguf_tensor("model.layers.7.mlp.experts.gate_proj.weight", &[2, 4, 4])
                .is_none()
        );
    }

    #[test]
    fn gguf_expert_stacks_name_the_official_stacks() {
        let stack = super::official_stack("blk.7.ffn_gate_up_exps.weight").unwrap();
        assert_eq!(stack.weight, "model.llm.layers.7.mlp.experts.w13_weight");
        assert!(stack.interleaved);
        let stack = super::official_stack("blk.7.ffn_down_exps.weight").unwrap();
        assert_eq!(stack.weight, "model.llm.layers.7.mlp.experts.w2_weight");
        assert!(!stack.interleaved);
        assert!(super::official_stack("blk.7.ffn_down_exps.scale").is_none());
        assert!(super::official_stack("blk.7.attn_q.weight").is_none());
    }
}
