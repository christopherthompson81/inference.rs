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

const CHECKPOINT_TOWER: &str = "model.visual";
// the processor's own pixel normalization: v / 127.5 - 1
const IMAGE_MEAN: f32 = 0.5;
const IMAGE_STD: f32 = 0.5;

/// llama.cpp's `inkling2v` projector from the official checkpoint's `model.visual.*`, laid out as its converter does.
pub fn inkling2_mmproj_spec(
    checkpoint: &crate::mmproj_writer::Checkpoint,
) -> Result<crate::mmproj_writer::MmprojSpec> {
    use crate::mmproj_writer::{MmprojSpec, MmprojTensor, Role, Source};
    let vision = &checkpoint.config["vision_config"];
    let get = |key: &str| -> Result<usize> {
        vision[key]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| anyhow::anyhow!("{FAMILY} `vision_config` has no `{key}`"))
    };
    let float = |key: &str| -> Result<f64> {
        vision[key]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("{FAMILY} `vision_config` has no `{key}`"))
    };
    let (hidden, layers, patch) = (
        get("hidden_size")?,
        get("num_hidden_layers")?,
        get("native_patch_size")?,
    );
    let ffn = (hidden as f64 * float("mlp_ratio")?) as usize;
    let u32_of = |v: usize| Value::U32(v as u32);
    let mean = Value::Array(vec![Value::F32(IMAGE_MEAN); CHANNELS]);
    let std = Value::Array(vec![Value::F32(IMAGE_STD); CHANNELS]);
    let mut spec = MmprojSpec {
        metadata: vec![
            ("clip.has_vision_encoder".into(), Value::Bool(true)),
            (
                super::multimodal_binding_utils::VISION_PROJECTOR_TYPE.into(),
                Value::String(PROJECTOR.into()),
            ),
            (
                "clip.vision.projection_dim".into(),
                u32_of(get("text_hidden_size")?),
            ),
            (
                "clip.vision.image_size".into(),
                u32_of(get("posemb_grid_size")? * patch),
            ),
            ("clip.vision.patch_size".into(), u32_of(patch)),
            ("clip.vision.embedding_length".into(), u32_of(hidden)),
            ("clip.vision.feed_forward_length".into(), u32_of(ffn)),
            ("clip.vision.block_count".into(), u32_of(layers)),
            (
                "clip.vision.attention.head_count".into(),
                u32_of(get("num_attention_heads")?),
            ),
            ("clip.vision.image_mean".into(), mean),
            ("clip.vision.image_std".into(), std),
            (
                LAYER_NORM_EPS.into(),
                Value::F32(float("layer_norm_eps")? as f32),
            ),
            (
                "clip.vision.spatial_merge_size".into(),
                u32_of(get("adapter_merge_size")?),
            ),
        ],
        tensors: Vec::new(),
    };
    let mut put = |name: &str, source: &str, role: Role| {
        spec.tensors
            .push(MmprojTensor::new(name, Source::tensor(source), role));
    };
    let emb = format!("{CHECKPOINT_TOWER}.backbone.embeddings");
    put(
        "v.class_embd",
        &format!("{emb}.class_embedding"),
        Role::Float,
    );
    put(
        "v.class_position_embd.weight",
        &format!("{emb}.cls_position_embedding"),
        Role::Float,
    );
    put(
        "v.position_embd.weight",
        &format!("{emb}.position_embedding"),
        Role::Float,
    );
    let backbone = format!("{CHECKPOINT_TOWER}.backbone");
    for part in ["weight", "bias"] {
        put(
            &format!("v.pre_ln.{part}"),
            &format!("{backbone}.ln_pre.{part}"),
            Role::Float,
        );
    }
    for layer in 0..layers {
        let (src, dst) = (
            format!("{backbone}.layers.{layer}"),
            format!("v.blk.{layer}"),
        );
        for (role, target) in [
            ("ln1", "ln_1"),
            ("ln2", "ln_2"),
            ("attn_q", "self_attn.q_proj"),
            ("attn_k", "self_attn.k_proj"),
            ("attn_v", "self_attn.v_proj"),
            ("attn_out", "self_attn.out_proj"),
            ("ffn_up", "mlp.c_fc"),
            ("ffn_down", "mlp.c_proj"),
        ] {
            let weight = if role.starts_with("ln") {
                Role::Float
            } else {
                Role::Matrix
            };
            put(
                &format!("{dst}.{role}.weight"),
                &format!("{src}.{target}.weight"),
                weight,
            );
            put(
                &format!("{dst}.{role}.bias"),
                &format!("{src}.{target}.bias"),
                Role::Float,
            );
        }
        for (role, target) in [("ls1", "ls_1"), ("ls2", "ls_2")] {
            put(
                &format!("{dst}.{role}.weight"),
                &format!("{src}.{target}.gamma"),
                Role::Float,
            );
        }
    }
    for (name, source) in [
        ("mm.1.weight", "proj_in.weight"),
        ("mm.2.weight", "proj_out.weight"),
    ] {
        put(name, &format!("{CHECKPOINT_TOWER}.{source}"), Role::Matrix);
    }
    // the linear patch embedding's (py, px, c) rows as the conv llama.cpp's graph reads
    let patch_name = format!("{emb}.patch_embedding.weight");
    let &[out, _] = checkpoint.shape(&patch_name)?.as_slice() else {
        anyhow::bail!("{FAMILY} patch embedding is not a matrix");
    };
    spec.tensors.push(MmprojTensor::new(
        "v.patch_embd.weight",
        Source::tensor(patch_name)
            .reshape(vec![out, patch, patch, CHANNELS])
            .permute(vec![0, 3, 1, 2]),
        Role::Float,
    ));
    Ok(spec)
}
