use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

use super::multimodal_binding_utils::{
    TensorInventory, bind, bind_required, bind_required_linear, bind_required_linears, map_usize,
    projector_type, validate_architecture,
};
use super::whisper_audio_bindings::token_id;

const FAMILY: &str = "DeepSeek-OCR";
const IMAGE_TOKEN: &str = "<image>";
pub const TEXT_ARCHITECTURE: &str = "deepseek2-ocr";
pub const PROJECTOR: &str = "deepseekocr";
/// DeepSeek-OCR-2's: SAM read by the Qwen2 query encoder.
pub const PROJECTOR_2: &str = "deepseekocr2";
// the query table OCR-2's projector carries for the global view
const GLOBAL_QUERIES: &str = "v.resample_query_1024.weight";
const VIEW_SEPARATOR: &str = "v.view_seperator";
// llama.cpp keeps these fixed for the projector type (`global_attn_indices`, the global view, tile counts)
const SAM_GLOBAL_LAYERS: [usize; 4] = [2, 5, 8, 11];
const GLOBAL_VIEW: usize = 1024;
const DEFAULT_MAX_TILES: usize = 9;
const OCR2_DEFAULT_MAX_TILES: usize = 6;
// CLIP-L/14 at 224, which the mmproj describes only by width, heads and depth
const CLIP_IMAGE_SIZE: usize = 224;
const CLIP_PATCH_SIZE: usize = 14;

/// llama.cpp's DeepSeek-OCR and OCR-2: the non-MLA DeepSeek-V2 text stack, then SAM, CLIP (OCR) or the query
/// encoder (OCR-2), and the projector.
pub fn build_deepseek_ocr_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, TEXT_ARCHITECTURE)?;
    let projector =
        projector_type(archive)?.context("GGUF vision projector metadata is required")?;
    if ![PROJECTOR, PROJECTOR_2].contains(&projector) {
        bail!("expected a `{PROJECTOR}` or `{PROJECTOR_2}` vision projector, found `{projector}`");
    }
    build_from_inventory(&TensorInventory::from_archive(archive))
}

fn build_from_inventory(inventory: &TensorInventory<'_>) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_text(inventory, &mut bindings)?;
    bind_sam(inventory, &mut bindings)?;
    if inventory.contains(GLOBAL_QUERIES) {
        bind_query_encoder(inventory, &mut bindings)?;
    } else {
        bind_clip(inventory, &mut bindings)?;
        bind_required(
            inventory,
            &mut bindings,
            "model.image_newline",
            "v.image_newline",
        )?;
    }
    bind_required_linear(
        inventory,
        &mut bindings,
        "model.projector.layers",
        "mm.model.fc",
    )?;
    // newer converters store it as (1, n)
    let separator = inventory.shape(VIEW_SEPARATOR)?.iter().product();
    bindings.insert(
        "model.view_seperator".to_string(),
        GgufTensorBinding::Reshape {
            input: Box::new(GgufTensorBinding::tensor(VIEW_SEPARATOR)),
            dims: vec![separator],
        },
    );
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

fn bind_query_encoder(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let root = "model.qwen2_model";
    for layer in inventory.require_layers("v.blk.", FAMILY)? {
        bind_required_linears(
            inventory,
            bindings,
            &format!("{root}.model.model.layers.{layer}"),
            &format!("v.blk.{layer}"),
            &[
                ("input_layernorm", "ln1"),
                ("post_attention_layernorm", "ln2"),
                ("self_attn.q_proj", "attn_q"),
                ("self_attn.k_proj", "attn_k"),
                ("self_attn.v_proj", "attn_v"),
                ("self_attn.o_proj", "attn_out"),
                ("mlp.gate_proj", "ffn_gate"),
                ("mlp.up_proj", "ffn_up"),
                ("mlp.down_proj", "ffn_down"),
            ],
        )?;
    }
    bind_required_linear(
        inventory,
        bindings,
        &format!("{root}.model.model.norm"),
        "v.post_ln",
    )?;
    for side in ["768", "1024"] {
        bind_required_linear(
            inventory,
            bindings,
            &format!("{root}.query_{side}"),
            &format!("v.resample_query_{side}"),
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
    let ocr2 = shapes.contains_key(GLOBAL_QUERIES);
    let max_tiles = map_usize(metadata, "clip.vision.preproc_max_tiles").unwrap_or(if ocr2 {
        OCR2_DEFAULT_MAX_TILES
    } else {
        DEFAULT_MAX_TILES
    });
    let vision = |key: &str| map_usize(metadata, &format!("clip.vision.{key}"));
    let (architecture, reader_name, reader) = if ocr2 {
        (
            "DeepseekOCR2ForCausalLM",
            "qwen2-0-5b",
            json!({
                "dim": vision("embedding_length")?,
                "layers": vision("block_count")?,
                "heads": vision("attention.head_count")?,
                "kv_heads": vision("attention.head_count_kv")?,
                "intermediate_size": vision("feed_forward_length")?,
            }),
        )
    } else {
        (
            "DeepseekOCRForCausalLM",
            "clip-l-14-224",
            json!({
                "heads": vision("attention.head_count")?,
                "image_size": CLIP_IMAGE_SIZE,
                "layers": vision("block_count")?,
                "patch_size": CLIP_PATCH_SIZE,
                "width": vision("embedding_length")?,
            }),
        )
    };
    let config = json!({
        "architectures": [architecture],
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
                reader_name: reader,
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
        ] {
            assert_eq!(
                bindings.get(native),
                Some(&GgufTensorBinding::tensor(source)),
                "{native}"
            );
        }
        assert_eq!(
            bindings.get("model.view_seperator"),
            Some(&GgufTensorBinding::Reshape {
                input: Box::new(GgufTensorBinding::tensor("v.view_seperator")),
                dims: vec![8],
            })
        );
    }

    // OCR-2's projector: the query encoder's layers, norm and tables in place of CLIP's tower and the newline
    fn ocr2_inventory() -> Vec<(String, Vec<usize>)> {
        let mut tensors = tensor_inventory();
        tensors.retain(|(name, _)| {
            !(name.starts_with("v.blk.")
                || name.starts_with("v.class_embd")
                || name.starts_with("v.patch_embd")
                || name.starts_with("v.position_embd")
                || name.starts_with("v.pre_ln")
                || name == "v.image_newline")
        });
        for (name, shape) in &mut tensors {
            if name == VIEW_SEPARATOR {
                *shape = vec![1, 8];
            }
        }
        for role in [
            "ln1", "ln2", "attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up",
            "ffn_down",
        ] {
            tensors.push((format!("v.blk.0.{role}.weight"), vec![8, 8]));
        }
        for role in ["attn_q", "attn_k", "attn_v"] {
            tensors.push((format!("v.blk.0.{role}.bias"), vec![8]));
        }
        tensors.push(("v.post_ln.weight".to_string(), vec![8]));
        tensors.push(("v.resample_query_768.weight".to_string(), vec![144, 8]));
        tensors.push((GLOBAL_QUERIES.to_string(), vec![256, 8]));
        tensors
    }

    #[test]
    fn maps_the_ocr2_query_encoder_and_names_it_in_the_config() {
        let tensors = ocr2_inventory();
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
                "model.qwen2_model.model.model.layers.0.self_attn.k_proj.bias",
                "v.blk.0.attn_k.bias",
            ),
            (
                "model.qwen2_model.model.model.layers.0.mlp.gate_proj.weight",
                "v.blk.0.ffn_gate.weight",
            ),
            (
                "model.qwen2_model.model.model.norm.weight",
                "v.post_ln.weight",
            ),
            ("model.qwen2_model.query_1024.weight", GLOBAL_QUERIES),
        ] {
            assert_eq!(
                bindings.get(native),
                Some(&GgufTensorBinding::tensor(source)),
                "{native}"
            );
        }
        assert!(bindings.get("model.image_newline").is_none());

        let text = |key: &str| format!("{TEXT_ARCHITECTURE}.{key}");
        let mut metadata = HashMap::new();
        for (key, value) in [
            (text("vocab_size"), 8),
            (text("embedding_length"), 8),
            (text("feed_forward_length"), 8),
            (text("expert_feed_forward_length"), 8),
            (text("block_count"), 2),
            (text("attention.head_count"), 2),
            (text("attention.head_count_kv"), 2),
            (text("expert_count"), 4),
            (text("expert_shared_count"), 1),
            (text("expert_used_count"), 2),
            (text("context_length"), 64),
            ("clip.vision.embedding_length".to_string(), 8),
            ("clip.vision.block_count".to_string(), 1),
            ("clip.vision.attention.head_count".to_string(), 2),
            ("clip.vision.attention.head_count_kv".to_string(), 1),
            ("clip.vision.feed_forward_length".to_string(), 8),
            ("clip.vision.sam.head_count".to_string(), 2),
            ("clip.vision.sam.block_count".to_string(), 1),
            ("clip.vision.sam.embedding_length".to_string(), 4),
        ] {
            metadata.insert(key, Value::U32(value));
        }
        metadata.insert(text("attention.layer_norm_rms_epsilon"), Value::F32(1e-6));
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(vec![Value::String(IMAGE_TOKEN.to_string())]),
        );
        let shapes = tensors.into_iter().collect::<HashMap<_, _>>();
        let config: serde_json::Value =
            serde_json::from_str(&deepseek_ocr_config(&metadata, &shapes).unwrap()).unwrap();
        assert_eq!(config["architectures"][0], "DeepseekOCR2ForCausalLM");
        assert_eq!(config["max_tiles"], OCR2_DEFAULT_MAX_TILES);
        let widths = &config["vision_config"]["width"];
        assert!(widths.get("clip-l-14-224").is_none());
        assert_eq!(widths["qwen2-0-5b"]["kv_heads"], 1);
    }
}
