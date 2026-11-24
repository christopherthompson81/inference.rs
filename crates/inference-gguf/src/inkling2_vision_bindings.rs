//! Inkling 2 with its vision tower from llama.cpp's GGUFs: an `inkling2_loop` text GGUF beside an `inkling2v`
//! projector, the tower bound to the checkpoint's own `model.visual.*` names.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::NormalLoaderType;
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

use super::multimodal_binding_utils::{
    TensorInventory, bind_required, bind_required_linear, bind_required_with, map_usize,
    validate_projector,
};
use super::normal_config::synthesize_normal_config_value;
use super::whisper_audio_bindings::token_id;

pub const PROJECTOR: &str = "inkling2v";
pub const TEXT_ARCHITECTURE: &str = "inkling2_loop";
const FAMILY: &str = "Inkling 2";
const IMAGE_PAD: &str = "<|unused_200054|>";
const ROOT: &str = "model.visual";
const CHANNELS: usize = 3;
const LAYER_NORM_EPS: &str = "clip.vision.attention.layer_norm_epsilon";

pub fn build_inkling2_vision_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_projector(archive, PROJECTOR)?;
    let mut bindings = super::inkling2_bindings::build_inkling2_bindings(archive)?;
    bind_tower(&TensorInventory::from_archive(archive), &mut bindings)?;
    Ok(bindings)
}

fn bind_tower(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    let emb = format!("{ROOT}.backbone.embeddings");
    bind_required(
        inventory,
        bindings,
        format!("{emb}.class_embedding"),
        "v.class_embd",
    )?;
    bind_required(
        inventory,
        bindings,
        format!("{emb}.cls_position_embedding"),
        "v.class_position_embd.weight",
    )?;
    bind_required(
        inventory,
        bindings,
        format!("{emb}.position_embedding"),
        "v.position_embd.weight",
    )?;
    // llama.cpp lays the linear patch embedding out as a conv (out, c, py, px); the tower reads (py, px, c) rows
    bind_required_with(
        inventory,
        bindings,
        format!("{emb}.patch_embedding.weight"),
        "v.patch_embd.weight",
        |source, shape| {
            let &[out, c, py, px] = shape else {
                anyhow::bail!("{FAMILY} patch embedding is {shape:?}, not a conv");
            };
            anyhow::ensure!(c == CHANNELS, "{FAMILY} patch embedding reads {c} channels");
            Ok(GgufTensorBinding::tensor(source)
                .permute(vec![0, 2, 3, 1])
                .reshape(vec![out, py * px * c]))
        },
    )?;
    bind_required_linear(
        inventory,
        bindings,
        &format!("{ROOT}.backbone.ln_pre"),
        "v.pre_ln",
    )?;
    for layer in 0.. {
        let src = format!("v.blk.{layer}");
        if !inventory.contains(&format!("{src}.ln1.weight")) {
            break;
        }
        let dst = format!("{ROOT}.backbone.layers.{layer}");
        for (target, role) in [
            ("ln_1", "ln1"),
            ("ln_2", "ln2"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.out_proj", "attn_out"),
            ("mlp.c_fc", "ffn_up"),
            ("mlp.c_proj", "ffn_down"),
        ] {
            bind_required_linear(
                inventory,
                bindings,
                &format!("{dst}.{target}"),
                &format!("{src}.{role}"),
            )?;
        }
        for (target, role) in [("ls_1", "ls1"), ("ls_2", "ls2")] {
            bind_required(
                inventory,
                bindings,
                format!("{dst}.{target}.gamma"),
                format!("{src}.{role}.weight"),
            )?;
        }
    }
    bind_required_linear(inventory, bindings, &format!("{ROOT}.proj_in"), "mm.1")?;
    bind_required_linear(inventory, bindings, &format!("{ROOT}.proj_out"), "mm.2")
}

/// The `Inkling2LoopForConditionalGeneration` config the merged text and projector GGUFs describe.
pub fn synthesize_inkling2_vision_config(archive: &GgufArchive) -> Result<String> {
    let names = archive.tensors().keys().cloned().collect::<Vec<_>>();
    let mm = archive.tensor_info("mm.1.weight")?.shape().to_vec();
    inkling2_vision_config(archive.metadata(), &names, &mm)
}

fn inkling2_vision_config(
    metadata: &HashMap<String, Value>,
    names: &[String],
    proj_in: &[usize],
) -> Result<String> {
    let text = synthesize_normal_config_value(&NormalLoaderType::Inkling2Loop, metadata, names)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let hidden = map_usize(metadata, "clip.vision.embedding_length")?;
    let ffn = map_usize(metadata, "clip.vision.feed_forward_length")?;
    let patch = map_usize(metadata, "clip.vision.patch_size")?;
    let merge = map_usize(metadata, "clip.vision.spatial_merge_size")?;
    let grid = map_usize(metadata, "clip.vision.image_size")? / patch;
    let eps = match metadata.get(LAYER_NORM_EPS) {
        Some(Value::F32(eps)) => f64::from(*eps),
        Some(Value::F64(eps)) => *eps,
        _ => anyhow::bail!("{FAMILY} projector has no `{LAYER_NORM_EPS}`"),
    };
    let &[projection_hidden, _] = proj_in else {
        anyhow::bail!("{FAMILY} projector's first linear is {proj_in:?}");
    };
    let config = json!({
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "image_token_id": token_id(metadata, IMAGE_PAD, FAMILY)?,
        "vision_config": {
            "hidden_size": hidden,
            "num_hidden_layers": map_usize(metadata, "clip.vision.block_count")?,
            "num_attention_heads": map_usize(metadata, "clip.vision.attention.head_count")?,
            "mlp_ratio": ffn as f64 / hidden as f64,
            "native_patch_size": patch,
            "adapter_merge_size": merge,
            "posemb_grid_size": grid,
            "projection_hidden_dim": projection_hidden,
            "text_hidden_size": map_usize(metadata, "clip.vision.projection_dim")?,
            "layer_norm_eps": eps,
        },
        "text_config": text,
    });
    Ok(config.to_string())
}
