//! Nemotron-H Omni (Nemotron 3.5 Bolt) from NVIDIA's GGUFs: a `nemotron_h_moe` text GGUF beside a `nemotron_omni`
//! projector. The projector is the checkpoint's C-RADIO tower and `mlp1` under llama.cpp's names, so both bind back
//! onto the checkpoint's: the decoder under `language_model.`, q/k/v joined into RADIO's `attn.qkv`.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use inference_nn::loaders::NormalLoaderType;
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

use super::mmproj_writer::{Checkpoint, MmprojSpec, MmprojTensor, Role, Source};
use super::multimodal_binding_utils::{
    TensorInventory, bind_required, map_usize, validate_projector,
};
use super::normal_config::synthesize_normal_config_value;
use super::normal_registry::CanonicalGgufArchitecture;
use super::parakeet_audio_bindings as parakeet;
use super::whisper_audio_bindings::token_id;

pub const PROJECTOR: &str = "nemotron_omni";
pub const TEXT_ARCHITECTURE: &str = "nemotron_h_moe";
/// The dense decoder a checkpoint projector also pairs with (Nano VL 12B's Nemotron Nano 2).
pub const DENSE_TEXT_ARCHITECTURE: &str = "nemotron_h";
const FAMILY: &str = "Nemotron-H Omni";
const IMAGE_TOKEN: &str = "<image>";
const LANGUAGE_MODEL: &str = "language_model";
const RADIO: &str = "vision_model.radio_model.model";
const PATCH_EMBED: &str = "v.patch_embd.weight";
const VIDEO_PATCH_EMBED: &str = "v.patch_embd.weight.1";
const POSITIONS: &str = "v.position_embd.weight";
const PREFIX_TOKENS: &str = "v.class_embd";
const PROJECTOR_IN: &str = "mm.model.mlp.1.weight";
// a final LayerNorm on the tower, which the reference adds for decoders with MTP
const POST_NORM: &str = "v.post_ln.weight";
const BLOCK_ROLES: [(&str, &str); 6] = [
    ("attn.proj", "attn_out"),
    ("mlp.fc1", "ffn_up"),
    ("mlp.fc2", "ffn_down"),
    ("norm1", "ln1"),
    ("norm2", "ln2"),
    ("attn.qkv", ""),
];
const PROJECTOR_LAYERS: [&str; 3] = ["0", "1", "3"];
const SOUND_TOKEN: &str = "<so_embedding>";
const SOUND_ENCODER: &str = "sound_encoder.encoder";
const SOUND_PROJECTION: &str = "sound_projection";
// the size llama.cpp warms its graph at; images run at their own grids
const WARMUP_IMAGE_SIZE: u32 = 512;
// the towers a checkpoint projector (its safetensors shard) carries under their own names
const CHECKPOINT_TOWERS: [&str; 4] = [
    "vision_model.",
    "mlp1.",
    "sound_encoder.",
    "sound_projection.",
];

pub fn build_nemotron_omni_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_projector(archive, PROJECTOR)?;
    let text = super::nemotron_h_bindings::build_nemotron_h_bindings(
        archive,
        CanonicalGgufArchitecture::NemotronHMoe,
    )?;
    let mut bindings = GgufBindingMap::new();
    for (native, binding) in text.iter() {
        bindings.insert(format!("{LANGUAGE_MODEL}.{native}"), binding.clone());
    }
    for (native, binding) in build_nemotron_omni_tower_bindings(archive)?.iter() {
        bindings.insert(native.to_string(), binding.clone());
    }
    Ok(bindings)
}

/// The projector's towers alone under the checkpoint's names: C-RADIO and `mlp1`, and the sound tower when the
/// projector carries `parakeet` audio.
pub fn build_nemotron_omni_tower_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    let inventory = TensorInventory::from_archive(archive);
    let mut bindings = GgufBindingMap::new();
    bind_vision(&inventory, &mut bindings)?;
    if parakeet::has_audio(archive.metadata()) {
        parakeet::bind_audio(&inventory, &mut bindings, SOUND_ENCODER, SOUND_PROJECTION)?;
    }
    Ok(bindings)
}

/// A `nemotron_h_moe` or `nemotron_h` text GGUF beside the checkpoint's own shard of towers (Nano Omni's vision and
/// sound, Nano VL's vision): the decoder as `build_nemotron_omni_bindings` binds it, every tower tensor under its own
/// name.
pub fn build_nemotron_omni_checkpoint_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_projector(archive, inference_quant::CHECKPOINT_PROJECTOR)?;
    let architecture = match archive.metadata_value("general.architecture") {
        Some(Value::String(name)) if name == DENSE_TEXT_ARCHITECTURE => {
            CanonicalGgufArchitecture::NemotronH
        }
        _ => CanonicalGgufArchitecture::NemotronHMoe,
    };
    let text = super::nemotron_h_bindings::build_nemotron_h_bindings(archive, architecture)?;
    let mut bindings = GgufBindingMap::new();
    for (native, binding) in text.iter() {
        bindings.insert(format!("{LANGUAGE_MODEL}.{native}"), binding.clone());
    }
    let towers = archive
        .tensors()
        .keys()
        .filter(|name| {
            CHECKPOINT_TOWERS
                .iter()
                .any(|prefix| name.starts_with(prefix))
        })
        .collect::<Vec<_>>();
    if towers.is_empty() {
        bail!("{FAMILY} checkpoint projector has no tower tensors");
    }
    for name in towers {
        bindings.insert(name.clone(), GgufTensorBinding::tensor(name.clone()));
    }
    Ok(bindings)
}

fn bind_vision(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    let patches = format!("{RADIO}.patch_generator");
    for (native, source) in [
        ("embedder.weight", PATCH_EMBED),
        ("video_embedder.weight", VIDEO_PATCH_EMBED),
    ] {
        if !inventory.contains(source) {
            continue;
        }
        // a conv kernel (out, c, p, p) in the GGUF, the checkpoint's linear over flattened patches
        let shape = inventory.shape(source)?;
        let rows = shape[0];
        let cols = shape[1..].iter().product();
        bindings.insert(
            format!("{patches}.{native}"),
            GgufTensorBinding::tensor(source).reshape(vec![rows, cols]),
        );
    }
    bind_required(
        inventory,
        bindings,
        format!("{patches}.pos_embed"),
        POSITIONS,
    )?;
    bind_required(
        inventory,
        bindings,
        format!("{patches}.cls_token.token"),
        PREFIX_TOKENS,
    )?;
    for layer in inventory.require_layers("v.blk.", FAMILY)? {
        let ours = format!("{RADIO}.blocks.{layer}");
        let theirs = format!("v.blk.{layer}");
        for (native, source) in BLOCK_ROLES {
            for suffix in ["weight", "bias"] {
                let binding = if source.is_empty() {
                    GgufTensorBinding::concat(
                        ["attn_q", "attn_k", "attn_v"]
                            .iter()
                            .map(|part| {
                                let name = format!("{theirs}.{part}.{suffix}");
                                inventory.shape(&name)?;
                                Ok(GgufTensorBinding::tensor(name))
                            })
                            .collect::<Result<Vec<_>>>()?,
                        0,
                    )
                } else {
                    let name = format!("{theirs}.{source}.{suffix}");
                    inventory.shape(&name)?;
                    GgufTensorBinding::tensor(name)
                };
                bindings.insert(format!("{ours}.{native}.{suffix}"), binding);
            }
        }
    }
    for layer in PROJECTOR_LAYERS {
        bind_required(
            inventory,
            bindings,
            format!("mlp1.{layer}.weight"),
            format!("mm.model.mlp.{layer}.weight"),
        )?;
    }
    Ok(())
}

fn json_usize(value: &serde_json::Value, key: &str) -> Result<usize> {
    value[key]
        .as_u64()
        .map(|v| v as usize)
        .with_context(|| format!("{FAMILY} config has no `{key}`"))
}

fn json_f64(value: &serde_json::Value, key: &str) -> Result<f64> {
    value[key]
        .as_f64()
        .with_context(|| format!("{FAMILY} config has no `{key}`"))
}

fn float_list(value: &serde_json::Value, key: &str) -> Result<Value> {
    let list = value[key]
        .as_array()
        .with_context(|| format!("{FAMILY} config has no `{key}`"))?;
    Ok(Value::Array(
        list.iter()
            .map(|v| v.as_f64().map(|v| Value::F32(v as f32)))
            .collect::<Option<Vec<_>>>()
            .with_context(|| format!("{FAMILY} config `{key}` is not a float list"))?,
    ))
}

/// Nano Omni's projector, as `bind_vision` and the Parakeet binding read it back: C-RADIO with its full position table
/// under projector type `nemotron_omni` (Bolt's), the sound tower as llama.cpp's `parakeet`.
pub fn nano_omni_mmproj_spec(checkpoint: &Checkpoint) -> Result<MmprojSpec> {
    let config = &checkpoint.config;
    let mut vision = config["vision_config"].clone();
    if vision.get("hidden_size").is_none() {
        inference_nn::vision::radio::fill_radio_shape(&mut vision)?;
    }
    let h = json_usize(&vision, "hidden_size")?;
    let layers = json_usize(&vision, "num_hidden_layers")?;
    let ffn = (h as f64 * json_f64(&vision, "mlp_ratio")?) as usize;
    let patch = json_usize(config, "patch_size")?;
    let merge = (1.0 / json_f64(config, "downsample_ratio")?).round() as usize;
    let lm_hidden = json_usize(&config["llm_config"], "hidden_size")?;
    let pixels = |key: &str| -> Result<Value> {
        Ok(Value::U32(
            (json_usize(&vision, key)? * patch * patch) as u32,
        ))
    };
    let mut spec = MmprojSpec {
        metadata: vec![
            ("clip.has_vision_encoder".into(), Value::Bool(true)),
            (
                super::multimodal_binding_utils::VISION_PROJECTOR_TYPE.into(),
                Value::String(PROJECTOR.into()),
            ),
            ("clip.vision.nemotron_omni".into(), Value::Bool(true)),
            (
                "clip.vision.projection_dim".into(),
                Value::U32(lm_hidden as u32),
            ),
            (
                "clip.vision.image_size".into(),
                Value::U32(WARMUP_IMAGE_SIZE),
            ),
            ("clip.vision.patch_size".into(), Value::U32(patch as u32)),
            ("clip.vision.embedding_length".into(), Value::U32(h as u32)),
            (
                "clip.vision.feed_forward_length".into(),
                Value::U32(ffn as u32),
            ),
            ("clip.vision.block_count".into(), Value::U32(layers as u32)),
            (
                "clip.vision.attention.head_count".into(),
                Value::U32(json_usize(&vision, "num_attention_heads")? as u32),
            ),
            (
                "clip.vision.image_mean".into(),
                float_list(config, "norm_mean")?,
            ),
            (
                "clip.vision.image_std".into(),
                float_list(config, "norm_std")?,
            ),
            (
                "clip.vision.attention.layer_norm_epsilon".into(),
                Value::F32(json_f64(&vision, "layer_norm_eps")? as f32),
            ),
            ("clip.use_gelu".into(), Value::Bool(true)),
            (
                "clip.vision.projector.scale_factor".into(),
                Value::U32(merge as u32),
            ),
            (
                "clip.vision.image_min_pixels".into(),
                pixels("min_num_patches")?,
            ),
            (
                "clip.vision.image_max_pixels".into(),
                pixels("max_num_patches")?,
            ),
        ],
        tensors: Vec::new(),
    };
    let patches = format!("{RADIO}.patch_generator");
    let kernel = |native: &str| -> Result<Source> {
        let shape = checkpoint.shape(native)?;
        let channels = shape[1] / (patch * patch);
        Ok(Source::tensor(native).reshape(vec![shape[0], channels, patch, patch]))
    };
    spec.tensors.push(MmprojTensor::new(
        PATCH_EMBED,
        kernel(&format!("{patches}.embedder.weight"))?,
        Role::Float,
    ));
    let video = format!("{patches}.video_embedder.weight");
    if checkpoint.contains(&video) {
        spec.tensors.push(MmprojTensor::new(
            VIDEO_PATCH_EMBED,
            kernel(&video)?,
            Role::Float,
        ));
    }
    let positions = format!("{patches}.pos_embed");
    let grid = checkpoint.shape(&positions)?;
    spec.tensors.push(MmprojTensor::new(
        POSITIONS,
        Source::tensor(positions).reshape(vec![grid[grid.len() - 2], h]),
        Role::Float,
    ));
    spec.tensors.push(MmprojTensor::new(
        PREFIX_TOKENS,
        Source::tensor(format!("{patches}.cls_token.token")),
        Role::Float,
    ));
    for layer in 0..layers {
        let ours = format!("{RADIO}.blocks.{layer}");
        let theirs = format!("v.blk.{layer}");
        for (native, source) in BLOCK_ROLES {
            for suffix in ["weight", "bias"] {
                let role = if suffix == "weight" && !native.starts_with("norm") {
                    Role::Matrix
                } else {
                    Role::Float
                };
                let name = format!("{ours}.{native}.{suffix}");
                if !source.is_empty() {
                    spec.tensors.push(MmprojTensor::new(
                        format!("{theirs}.{source}.{suffix}"),
                        Source::tensor(name),
                        role,
                    ));
                    continue;
                }
                // RADIO's fused qkv as llama.cpp's three
                for (part, gguf) in ["attn_q", "attn_k", "attn_v"].iter().enumerate() {
                    spec.tensors.push(MmprojTensor::new(
                        format!("{theirs}.{gguf}.{suffix}"),
                        Source::tensor(name.clone()).rows(part * h, h),
                        role,
                    ));
                }
            }
        }
    }
    for layer in PROJECTOR_LAYERS {
        let role = if layer == "0" {
            Role::Float
        } else {
            Role::Matrix
        };
        spec.tensors.push(MmprojTensor::new(
            format!("mm.model.mlp.{layer}.weight"),
            Source::tensor(format!("mlp1.{layer}.weight")),
            role,
        ));
    }
    if config
        .get("sound_config")
        .is_some_and(|sound| !sound.is_null())
    {
        let audio = parakeet::audio_spec(
            checkpoint,
            &config["sound_config"],
            SOUND_ENCODER,
            SOUND_PROJECTION,
            lm_hidden,
        )?;
        spec.metadata.extend(audio.metadata);
        spec.tensors.extend(audio.tensors);
    }
    Ok(spec)
}

/// The config the text and projector GGUFs describe: the decoder from the `nemotron_h_moe` metadata, the tower and
/// projector from the projector's.
pub fn synthesize_nemotron_omni_config(archive: &GgufArchive) -> Result<String> {
    let shapes = archive
        .tensors()
        .iter()
        .map(|(name, info)| (name.clone(), info.shape().to_vec()))
        .collect();
    nemotron_omni_config(archive.metadata(), &shapes)
}

fn map_f64(metadata: &HashMap<String, Value>, key: &str) -> Result<f64> {
    match metadata.get(key) {
        Some(Value::F32(value)) => Ok(f64::from(*value)),
        Some(Value::F64(value)) => Ok(*value),
        Some(_) => bail!("GGUF metadata `{key}` must be a float"),
        None => bail!("GGUF metadata `{key}` is required"),
    }
}

fn map_f32_list(metadata: &HashMap<String, Value>, key: &str) -> Result<Vec<f32>> {
    let Some(Value::Array(values)) = metadata.get(key) else {
        bail!("GGUF metadata `{key}` must be a float list");
    };
    values
        .iter()
        .map(|value| match value {
            Value::F32(value) => Ok(*value),
            _ => bail!("GGUF metadata `{key}` must be a float list"),
        })
        .collect()
}

fn nemotron_omni_config(
    metadata: &HashMap<String, Value>,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<String> {
    let tensor_names = shapes.keys().cloned().collect::<Vec<_>>();
    let text =
        synthesize_normal_config_value(&NormalLoaderType::NemotronH, metadata, &tensor_names)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
    let mut config = vision_config(metadata, shapes)?;
    config["llm_config"] = text;
    if parakeet::has_audio(metadata) {
        config["sound_config"] = parakeet::sound_config(metadata, shapes)?;
        config["sound_context_token_id"] = json!(token_id(metadata, SOUND_TOKEN, FAMILY)?);
    }
    Ok(config.to_string())
}

// everything but the decoder's config: the tower, the projector and the image token
fn vision_config(
    metadata: &HashMap<String, Value>,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<serde_json::Value> {
    let shape = |name: &str| {
        shapes
            .get(name)
            .with_context(|| format!("{FAMILY} projector has no `{name}`"))
    };
    let patch = map_usize(metadata, "clip.vision.patch_size")?;
    let hidden = map_usize(metadata, "clip.vision.embedding_length")?;
    let positions = shape(POSITIONS)?[0];
    let side = (positions as f64).sqrt() as usize;
    if side * side != positions {
        bail!("{FAMILY} position grid of {positions} is not square");
    }
    let per_patch = patch * patch;
    Ok(json!({
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "model_type": "nemotron_h_omni",
        "vision_config": {
            "hidden_size": hidden,
            "num_attention_heads": map_usize(metadata, "clip.vision.attention.head_count")?,
            "num_hidden_layers": map_usize(metadata, "clip.vision.block_count")?,
            "layer_norm_eps": map_f64(metadata, "clip.vision.attention.layer_norm_epsilon")?,
            "mlp_ratio": map_usize(metadata, "clip.vision.feed_forward_length")? as f64 / hidden as f64,
            // the GGUF keeps cls and register tokens as one table; only their sum matters
            "num_cls_tokens": shape(PREFIX_TOKENS)?[0],
            "num_registers": 0,
            "max_img_size": side * patch,
            "min_num_patches": map_usize(metadata, "clip.vision.image_min_pixels")? / per_patch,
            "max_num_patches": map_usize(metadata, "clip.vision.image_max_pixels")? / per_patch,
        },
        "downsample_ratio": 1.0 / map_usize(metadata, "clip.vision.projector.scale_factor")? as f64,
        "img_context_token_id": token_id(metadata, IMAGE_TOKEN, FAMILY)?,
        "norm_mean": map_f32_list(metadata, "clip.vision.image_mean")?,
        "norm_std": map_f32_list(metadata, "clip.vision.image_std")?,
        "patch_size": patch,
        "projector_hidden_size": shape(PROJECTOR_IN)?[0],
        "vision_final_layernorm": shapes.contains_key(POST_NORM),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal_binding_utils::binding_sources;
    use crate::whisper_audio_bindings::TOKENS_KEY;

    const HIDDEN: usize = 8;
    const FFN: usize = 32;
    const PATCH: usize = 16;
    const GRID: usize = 4;
    const PREFIX: usize = 3;
    const PROJECTOR_HIDDEN: usize = 12;
    const IMAGE_ID: usize = 7;

    // llama.cpp's mmproj inventory, as the archive reports shapes (out, in)
    fn mmproj_tensors() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            (PATCH_EMBED.to_string(), vec![HIDDEN, 3, PATCH, PATCH]),
            (VIDEO_PATCH_EMBED.to_string(), vec![HIDDEN, 6, PATCH, PATCH]),
            (POSITIONS.to_string(), vec![GRID * GRID, HIDDEN]),
            (PREFIX_TOKENS.to_string(), vec![PREFIX, HIDDEN]),
            ("mm.model.mlp.0.weight".to_string(), vec![4 * HIDDEN]),
            (PROJECTOR_IN.to_string(), vec![PROJECTOR_HIDDEN, 4 * HIDDEN]),
            (
                "mm.model.mlp.3.weight".to_string(),
                vec![HIDDEN, PROJECTOR_HIDDEN],
            ),
        ];
        for layer in 0..2 {
            for (name, rows, cols) in [
                ("attn_q", HIDDEN, HIDDEN),
                ("attn_k", HIDDEN, HIDDEN),
                ("attn_v", HIDDEN, HIDDEN),
                ("attn_out", HIDDEN, HIDDEN),
                ("ffn_up", FFN, HIDDEN),
                ("ffn_down", HIDDEN, FFN),
            ] {
                tensors.push((format!("v.blk.{layer}.{name}.weight"), vec![rows, cols]));
                tensors.push((format!("v.blk.{layer}.{name}.bias"), vec![rows]));
            }
            for norm in ["ln1", "ln2"] {
                for part in ["weight", "bias"] {
                    tensors.push((format!("v.blk.{layer}.{norm}.{part}"), vec![HIDDEN]));
                }
            }
        }
        tensors
    }

    #[test]
    fn every_mmproj_tensor_binds_to_the_checkpoint_names() {
        let tensors = mmproj_tensors();
        let inventory =
            TensorInventory::new(tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice())));
        let mut bindings = GgufBindingMap::new();
        bind_vision(&inventory, &mut bindings).unwrap();
        assert_eq!(
            binding_sources(&bindings),
            tensors.iter().map(|(name, _)| name.clone()).collect()
        );
        for native in [
            "vision_model.radio_model.model.patch_generator.embedder.weight",
            "vision_model.radio_model.model.patch_generator.pos_embed",
            "vision_model.radio_model.model.patch_generator.cls_token.token",
            "vision_model.radio_model.model.blocks.1.attn.qkv.weight",
            "vision_model.radio_model.model.blocks.1.attn.qkv.bias",
            "vision_model.radio_model.model.blocks.0.mlp.fc2.bias",
            "mlp1.3.weight",
        ] {
            assert!(bindings.get(native).is_some(), "`{native}` is not bound");
        }
    }

    #[test]
    fn the_projector_metadata_describes_the_tower() {
        let shapes = mmproj_tensors().into_iter().collect::<HashMap<_, _>>();
        let mut metadata = HashMap::new();
        for (key, value) in [
            ("clip.vision.patch_size", PATCH),
            ("clip.vision.embedding_length", HIDDEN),
            ("clip.vision.attention.head_count", 2),
            ("clip.vision.block_count", 2),
            ("clip.vision.feed_forward_length", FFN),
            ("clip.vision.image_min_pixels", 4 * PATCH * PATCH),
            ("clip.vision.image_max_pixels", 64 * PATCH * PATCH),
            ("clip.vision.projector.scale_factor", 2),
        ] {
            metadata.insert(key.to_string(), Value::U32(value as u32));
        }
        metadata.insert(
            "clip.vision.attention.layer_norm_epsilon".to_string(),
            Value::F32(1e-6),
        );
        for key in ["clip.vision.image_mean", "clip.vision.image_std"] {
            metadata.insert(key.to_string(), Value::Array(vec![Value::F32(0.5); 3]));
        }
        let mut tokens = (0..10)
            .map(|i| Value::String(format!("<t{i}>")))
            .collect::<Vec<_>>();
        tokens[IMAGE_ID] = Value::String(IMAGE_TOKEN.to_string());
        metadata.insert(TOKENS_KEY.to_string(), Value::Array(tokens));
        let config = vision_config(&metadata, &shapes).unwrap();
        let vision = &config["vision_config"];
        assert_eq!(vision["max_img_size"], GRID * PATCH);
        assert_eq!(vision["num_cls_tokens"], PREFIX);
        assert_eq!(vision["mlp_ratio"], (FFN / HIDDEN) as f64);
        assert_eq!(vision["min_num_patches"], 4);
        assert_eq!(vision["max_num_patches"], 64);
        assert_eq!(config["downsample_ratio"], 0.5);
        assert_eq!(config["img_context_token_id"], IMAGE_ID);
        assert_eq!(config["projector_hidden_size"], PROJECTOR_HIDDEN);
        assert_eq!(config["vision_final_layernorm"], false);
    }
}
