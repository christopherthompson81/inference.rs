use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use inference_quant::{GgufArchive, GgufBindingMap};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

use super::multimodal_binding_utils::{
    TensorInventory, bind, bind_required, bind_required_linear, bind_required_linears, map_usize,
    validate_architecture, validate_projector,
};
use super::whisper_audio_bindings::token_id;

const FAMILY: &str = "DeepSeek-OCR";
const IMAGE_TOKEN: &str = "<image>";
pub const TEXT_ARCHITECTURE: &str = "deepseek2-ocr";
pub const PROJECTOR: &str = "deepseekocr";
// llama.cpp keeps these fixed for the projector type (`global_attn_indices`, the global view, tile counts)
const SAM_GLOBAL_LAYERS: [usize; 4] = [2, 5, 8, 11];
const GLOBAL_VIEW: usize = 1024;
const DEFAULT_MAX_TILES: usize = 9;
// CLIP-L/14 at 224, which the mmproj describes only by width, heads and depth
const CLIP_IMAGE_SIZE: usize = 224;
const CLIP_PATCH_SIZE: usize = 14;

/// llama.cpp's DeepSeek-OCR: the non-MLA DeepSeek-V2 text stack, then the SAM and CLIP towers and the projector.
pub fn build_deepseek_ocr_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, TEXT_ARCHITECTURE)?;
    validate_projector(archive, PROJECTOR)?;
    build_from_inventory(&TensorInventory::from_archive(archive))
}

fn build_from_inventory(inventory: &TensorInventory<'_>) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_text(inventory, &mut bindings)?;
    bind_sam(inventory, &mut bindings)?;
    bind_clip(inventory, &mut bindings)?;
    bind_required_linear(
        inventory,
        &mut bindings,
        "model.projector.layers",
        "mm.model.fc",
    )?;
    for (native, source) in [
        ("model.image_newline", "v.image_newline"),
        ("model.view_seperator", "v.view_seperator"),
    ] {
        bind_required(inventory, &mut bindings, native, source)?;
    }
    Ok(bindings)
}

fn bind_text(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    bind_required(
        inventory,
        bindings,
        "model.embed_tokens.weight",
        "token_embd.weight",
    )?;
    bind_required(
        inventory,
        bindings,
        "model.norm.weight",
        "output_norm.weight",
    )?;
    bind(inventory, bindings, "lm_head.weight", "output.weight");
    for layer in inventory.require_layers("blk.", FAMILY)? {
        let native = format!("model.layers.{layer}");
        let source = format!("blk.{layer}");
        bind_required_linears(
            inventory,
            bindings,
            &native,
            &source,
            &[
                ("self_attn.q_proj", "attn_q"),
                ("self_attn.k_proj", "attn_k"),
                ("self_attn.v_proj", "attn_v"),
                ("self_attn.o_proj", "attn_output"),
            ],
        )?;
        bind_required(
            inventory,
            bindings,
            format!("{native}.input_layernorm.weight"),
            format!("{source}.attn_norm.weight"),
        )?;
        bind_required(
            inventory,
            bindings,
            format!("{native}.post_attention_layernorm.weight"),
            format!("{source}.ffn_norm.weight"),
        )?;
        let roles: &[(&str, &str)] = if inventory.contains(&format!("{source}.ffn_gate_inp.weight"))
        {
            &[
                ("mlp.gate", "ffn_gate_inp"),
                ("mlp.experts.gate_proj", "ffn_gate_exps"),
                ("mlp.experts.up_proj", "ffn_up_exps"),
                ("mlp.experts.down_proj", "ffn_down_exps"),
                ("mlp.shared_experts.gate_proj", "ffn_gate_shexp"),
                ("mlp.shared_experts.up_proj", "ffn_up_shexp"),
                ("mlp.shared_experts.down_proj", "ffn_down_shexp"),
            ]
        } else {
            &[
                ("mlp.gate_proj", "ffn_gate"),
                ("mlp.up_proj", "ffn_up"),
                ("mlp.down_proj", "ffn_down"),
            ]
        };
        bind_required_linears(inventory, bindings, &native, &source, roles)?;
    }
    Ok(())
}

fn bind_sam(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    let root = "model.sam_model";
    bind_required_linear(
        inventory,
        bindings,
        &format!("{root}.patch_embed.proj"),
        "v.sam.patch_embd",
    )?;
    bind_required(
        inventory,
        bindings,
        format!("{root}.pos_embed"),
        "v.sam.pos_embd.weight",
    )?;
    for layer in inventory.require_layers("v.sam.blk.", FAMILY)? {
        let native = format!("{root}.blocks.{layer}");
        let source = format!("v.sam.blk.{layer}");
        bind_required_linears(
            inventory,
            bindings,
            &native,
            &source,
            &[
                ("norm1", "pre_ln"),
                ("norm2", "post_ln"),
                ("attn.qkv", "attn.qkv"),
                ("attn.proj", "attn.out"),
                ("mlp.lin1", "mlp.lin1"),
                ("mlp.lin2", "mlp.lin2"),
            ],
        )?;
        for (native_name, source_name) in [("rel_pos_h", "pos_h"), ("rel_pos_w", "pos_w")] {
            bind_required(
                inventory,
                bindings,
                format!("{native}.attn.{native_name}"),
                format!("{source}.attn.{source_name}.weight"),
            )?;
        }
    }
    for index in 0..4 {
        bind_required_linear(
            inventory,
            bindings,
            &format!("{root}.neck.{index}"),
            &format!("v.sam.neck.{index}"),
        )?;
    }
    for net in ["net_2", "net_3"] {
        bind_required(
            inventory,
            bindings,
            format!("{root}.{net}.weight"),
            format!("v.sam.{net}.weight"),
        )?;
    }
    Ok(())
}

fn bind_clip(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    let root = "model.vision_model";
    for (native, source) in [
        ("embeddings.class_embedding", "v.class_embd"),
        (
            "embeddings.position_embedding.weight",
            "v.position_embd.weight",
        ),
    ] {
        bind_required(inventory, bindings, format!("{root}.{native}"), source)?;
    }
    // the patch embedding SAM's features replace, kept so the tower's tensors all bind
    bind(
        inventory,
        bindings,
        format!("{root}.embeddings.patch_embedding.weight"),
        "v.patch_embd.weight",
    );
    bind_required_linear(
        inventory,
        bindings,
        &format!("{root}.pre_layrnorm"),
        "v.pre_ln",
    )?;
    for layer in inventory.require_layers("v.blk.", FAMILY)? {
        bind_required_linears(
            inventory,
            bindings,
            &format!("{root}.transformer.layers.{layer}"),
            &format!("v.blk.{layer}"),
            &[
                ("layer_norm1", "ln1"),
                ("layer_norm2", "ln2"),
                ("self_attn.qkv_proj", "attn_qkv"),
                ("self_attn.out_proj", "attn_out"),
                ("mlp.fc1", "ffn_up"),
                ("mlp.fc2", "ffn_down"),
            ],
        )?;
    }
    Ok(())
}

fn map_f32(metadata: &HashMap<String, Value>, key: &str) -> Result<f32> {
    match metadata.get(key) {
        Some(Value::F32(value)) => Ok(*value),
        Some(Value::F64(value)) => Ok(*value as f32),
        Some(_) => bail!("GGUF metadata `{key}` must be a float"),
        None => bail!("GGUF metadata `{key}` is required"),
    }
}

/// The checkpoint config llama.cpp's metadata describes: the text stack, the SAM and CLIP widths and the projector.
pub fn synthesize_deepseek_ocr_config(archive: &GgufArchive) -> Result<String> {
    let shapes = archive
        .tensors()
        .keys()
        .map(|name| Ok((name.clone(), archive.tensor_info(name)?.shape().to_vec())))
        .collect::<Result<HashMap<_, _>>>()?;
    deepseek_ocr_config(archive.metadata(), &shapes)
}

fn deepseek_ocr_config(
    metadata: &HashMap<String, Value>,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<String> {
    let text = |key: &str| map_usize(metadata, &format!("{TEXT_ARCHITECTURE}.{key}"));
    let shape = |name: &str| {
        shapes
            .get(name)
            .with_context(|| format!("{FAMILY} GGUF lacks `{name}`"))
    };
    // shapes in torch order: a conv's output channels first
    let out_channels = |name: &str| -> Result<usize> {
        shape(name)?
            .first()
            .copied()
            .with_context(|| format!("{FAMILY} GGUF `{name}` has no dims"))
    };
    let projector = shape("mm.model.fc.weight")?;
    let [n_embed, input_dim] = projector[..] else {
        bail!("{FAMILY} projector `mm.model.fc.weight` must be 2D, not {projector:?}");
    };
    let max_tiles =
        map_usize(metadata, "clip.vision.preproc_max_tiles").unwrap_or(DEFAULT_MAX_TILES);
    let config = json!({
        "architectures": ["DeepseekOCRForCausalLM"],
        "language_config": {
            "vocab_size": text("vocab_size")?,
            "hidden_size": text("embedding_length")?,
            "intermediate_size": text("feed_forward_length")?,
            "moe_intermediate_size": text("expert_feed_forward_length")?,
            "num_hidden_layers": text("block_count")?,
            "num_attention_heads": text("attention.head_count")?,
            "num_key_value_heads": text("attention.head_count_kv")?,
            "n_routed_experts": text("expert_count")?,
            "n_shared_experts": text("expert_shared_count")?,
            "num_experts_per_tok": text("expert_used_count")?,
            "first_k_dense_replace": text("leading_dense_block_count").unwrap_or(0),
            "max_position_embeddings": text("context_length")?,
            "rms_norm_eps": map_f32(metadata, &format!("{TEXT_ARCHITECTURE}.attention.layer_norm_rms_epsilon"))?,
            "tie_word_embeddings": !shapes.contains_key("output.weight"),
            "topk_method": "greedy",
            "use_mla": false,
        },
        "vision_config": {
            "image_size": GLOBAL_VIEW,
            "width": {
                "clip-l-14-224": {
                    "heads": map_usize(metadata, "clip.vision.attention.head_count")?,
                    "image_size": CLIP_IMAGE_SIZE,
                    "layers": map_usize(metadata, "clip.vision.block_count")?,
                    "patch_size": CLIP_PATCH_SIZE,
                    "width": map_usize(metadata, "clip.vision.embedding_length")?,
                },
                "sam_vit_b": {
                    "downsample_channels": [out_channels("v.sam.net_2.weight")?, out_channels("v.sam.net_3.weight")?],
                    "global_attn_indexes": SAM_GLOBAL_LAYERS,
                    "heads": map_usize(metadata, "clip.vision.sam.head_count")?,
                    "layers": map_usize(metadata, "clip.vision.sam.block_count")?,
                    "width": map_usize(metadata, "clip.vision.sam.embedding_length")?,
                },
            },
        },
        "projector_config": {"input_dim": input_dim, "n_embed": n_embed},
        "max_tiles": max_tiles,
        "image_token_id": token_id(metadata, IMAGE_TOKEN, FAMILY)?,
    });
    Ok(config.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use inference_quant::GgufTensorBinding;

    use super::*;
    use crate::multimodal_binding_utils::binding_sources;

    fn tensor_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 8]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![8, 8]),
            ("mm.model.fc.weight".to_string(), vec![8, 16]),
            ("mm.model.fc.bias".to_string(), vec![8]),
            ("v.image_newline".to_string(), vec![8]),
            ("v.view_seperator".to_string(), vec![8]),
            ("v.sam.patch_embd.weight".to_string(), vec![4, 3, 2, 2]),
            ("v.sam.patch_embd.bias".to_string(), vec![4]),
            ("v.sam.pos_embd.weight".to_string(), vec![1, 2, 2, 4]),
            ("v.sam.net_2.weight".to_string(), vec![6, 4, 3, 3]),
            ("v.sam.net_3.weight".to_string(), vec![8, 6, 3, 3]),
            ("v.class_embd".to_string(), vec![4]),
            ("v.patch_embd.weight".to_string(), vec![4, 3, 2, 2]),
            ("v.position_embd.weight".to_string(), vec![5, 4]),
            ("v.pre_ln.weight".to_string(), vec![4]),
            ("v.pre_ln.bias".to_string(), vec![4]),
        ];
        for index in 0..4 {
            tensors.push((format!("v.sam.neck.{index}.weight"), vec![4]));
            if index % 2 == 1 {
                tensors.push((format!("v.sam.neck.{index}.bias"), vec![4]));
            }
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
        for role in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate_inp",
            "ffn_gate_exps",
            "ffn_up_exps",
            "ffn_down_exps",
            "ffn_gate_shexp",
            "ffn_up_shexp",
            "ffn_down_shexp",
        ] {
            tensors.push((format!("blk.1.{role}.weight"), vec![8, 8]));
        }
        for layer in 0..2 {
            for role in ["attn_norm", "ffn_norm"] {
                tensors.push((format!("blk.{layer}.{role}.weight"), vec![8]));
            }
        }
        for role in [
            "pre_ln", "post_ln", "attn.qkv", "attn.out", "mlp.lin1", "mlp.lin2",
        ] {
            tensors.push((format!("v.sam.blk.0.{role}.weight"), vec![4, 4]));
            tensors.push((format!("v.sam.blk.0.{role}.bias"), vec![4]));
        }
        for role in ["pos_h", "pos_w"] {
            tensors.push((format!("v.sam.blk.0.attn.{role}.weight"), vec![3, 2]));
        }
        for role in ["ln1", "ln2", "attn_qkv", "attn_out", "ffn_up", "ffn_down"] {
            tensors.push((format!("v.blk.0.{role}.weight"), vec![4, 4]));
            tensors.push((format!("v.blk.0.{role}.bias"), vec![4]));
        }
        tensors
    }

    #[test]
    fn maps_complete_deepseek_ocr_inventory() {
        let tensors = tensor_inventory();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let bindings = build_from_inventory(&inventory).unwrap();
        let expected = tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(binding_sources(&bindings), expected);
        for (native, source) in [
            (
                "model.layers.0.mlp.gate_proj.weight",
                "blk.0.ffn_gate.weight",
            ),
            (
                "model.layers.1.mlp.gate.weight",
                "blk.1.ffn_gate_inp.weight",
            ),
            (
                "model.layers.1.mlp.experts.up_proj.weight",
                "blk.1.ffn_up_exps.weight",
            ),
            (
                "model.layers.1.mlp.shared_experts.down_proj.weight",
                "blk.1.ffn_down_shexp.weight",
            ),
            (
                "model.sam_model.blocks.0.attn.rel_pos_h",
                "v.sam.blk.0.attn.pos_h.weight",
            ),
            (
                "model.sam_model.blocks.0.norm1.bias",
                "v.sam.blk.0.pre_ln.bias",
            ),
            (
                "model.vision_model.transformer.layers.0.self_attn.qkv_proj.weight",
                "v.blk.0.attn_qkv.weight",
            ),
            ("model.view_seperator", "v.view_seperator"),
        ] {
            assert_eq!(
                bindings.get(native),
                Some(&GgufTensorBinding::tensor(source)),
                "{native}"
            );
        }
    }
}
