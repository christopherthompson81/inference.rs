//! llama.cpp's Granite 4 Vision (`granite4_vision`) and Granite Speech (`granite_speech`) projector GGUFs, with their
//! Granite text model, under the Hugging Face names the safetensors checkpoints use.

use anyhow::Result;
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

use inference_nn::loaders::NormalLoaderType;

use super::{
    multimodal_binding_utils::{
        TensorInventory, bind, bind_required, bind_required_linear, bind_required_linears,
        bind_required_with, bind_siglip_vision, validate_architecture, validate_projector,
    },
    normal_bindings::build_normal_bindings,
    normal_registry::CanonicalGgufArchitecture,
};

pub const TEXT_ARCHITECTURE: &str = "granite";
pub const VISION_PROJECTOR: &str = "granite4_vision";
pub const SPEECH_PROJECTOR: &str = "granite_speech";
const VISION_FAMILY: &str = "Granite 4 Vision";
const SPEECH_FAMILY: &str = "Granite Speech";
// per projector block: its corner for a spatial one, -1 for a layerwise one
const VISION_SPATIAL_OFFSETS_KEY: &str = "clip.vision.projector.spatial_offsets";
// per projector block: the tower layer it reads
const VISION_FEATURE_LAYER_KEY: &str = "clip.vision.feature_layer";
// the BLIP-2 Q-Former layer's linears and norms, as llama.cpp names them
const QFORMER_ROLES: &[(&str, &str)] = &[
    ("attention.attention.query", "self_attn_q"),
    ("attention.attention.key", "self_attn_k"),
    ("attention.attention.value", "self_attn_v"),
    ("attention.output.dense", "self_attn_out"),
    ("attention.output.LayerNorm", "self_attn_norm"),
    ("crossattention.attention.query", "cross_attn_q"),
    ("crossattention.attention.key", "cross_attn_k"),
    ("crossattention.attention.value", "cross_attn_v"),
    ("crossattention.output.dense", "cross_attn_out"),
    ("crossattention.output.LayerNorm", "cross_attn_norm"),
    ("intermediate_query.dense", "ffn_up"),
    ("output_query.dense", "ffn_down"),
    ("output_query.LayerNorm", "ffn_norm"),
];

/// The Granite text bindings with `model.` re-rooted under `text_root` and the head under `head`.
fn text_bindings(archive: &GgufArchive, text_root: &str, head: &str) -> Result<GgufBindingMap> {
    let text = build_normal_bindings(
        archive,
        &NormalLoaderType::Granite,
        CanonicalGgufArchitecture::Granite,
    )?;
    Ok(text
        .iter()
        .map(|(native, binding)| {
            let native = match native.strip_prefix("model.") {
                Some(rest) => format!("{text_root}.{rest}"),
                None => native.replacen("lm_head", head, 1),
            };
            (native, binding.clone())
        })
        .collect())
}

fn bind_qformer_layer(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
) -> Result<()> {
    bind_required_linears(inventory, bindings, native, source, QFORMER_ROLES)
}

pub fn build_granite4_vision_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, TEXT_ARCHITECTURE)?;
    validate_projector(archive, VISION_PROJECTOR)?;
    let mut bindings = text_bindings(archive, "model.language_model", "lm_head")?;
    let inventory = TensorInventory::from_archive(archive);
    bind_siglip_vision(
        &inventory,
        &mut bindings,
        "model.vision_tower.vision_model",
        VISION_FAMILY,
    )?;
    bind(
        &inventory,
        &mut bindings,
        "model.image_newline",
        "v.image_newline",
    );
    let offsets = i32_array(archive, VISION_SPATIAL_OFFSETS_KEY)?;
    let feature_layers = i32_array(archive, VISION_FEATURE_LAYER_KEY)?;
    bind_granite4_vision_projectors(&inventory, &offsets, &feature_layers, &mut bindings)?;
    Ok(bindings)
}

fn bind_granite4_vision_projectors(
    inventory: &TensorInventory<'_>,
    offsets: &[i32],
    feature_layers: &[i32],
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    if feature_layers.len() != offsets.len() {
        anyhow::bail!(
            "{VISION_FAMILY} GGUF names {} feature layers for {} projectors",
            feature_layers.len(),
            offsets.len()
        );
    }
    // llama.cpp writes the layerwise projectors from the deepest tower layer up; transformers' configs list them from
    // the shallowest, so a block's index is its feature layer's rank among them
    let mut layerwise = (0..offsets.len())
        .filter(|&block| offsets[block] < 0)
        .collect::<Vec<_>>();
    layerwise.sort_by_key(|&block| feature_layers[block]);
    for (block, &offset) in offsets.iter().enumerate() {
        let native = match usize::try_from(offset) {
            Ok(corner) => format!("model.spatial_projectors.{corner}"),
            Err(_) => format!(
                "model.layerwise_projectors.{}",
                layerwise
                    .iter()
                    .position(|&b| b == block)
                    .expect("a layerwise block")
            ),
        };
        let source = format!("v.proj_blk.{block}");
        bind_required_linear(
            inventory,
            bindings,
            &format!("{native}.norm"),
            &format!("{source}.norm"),
        )?;
        bind_required_linear(
            inventory,
            bindings,
            &format!("{native}.out_linear"),
            &format!("{source}.linear"),
        )?;
        bind_required_linear(
            inventory,
            bindings,
            &format!("{native}.qformer.layernorm"),
            &format!("{source}.post_norm"),
        )?;
        bind_required(
            inventory,
            bindings,
            format!("{native}.query"),
            format!("{source}.query"),
        )?;
        bind_required(
            inventory,
            bindings,
            format!("{native}.image_positions"),
            format!("{source}.img_pos"),
        )?;
        bind_qformer_layer(
            inventory,
            bindings,
            &format!("{native}.qformer.encoder.layer.0"),
            &source,
        )?;
    }
    Ok(())
}

fn i32_array(archive: &GgufArchive, key: &str) -> Result<Vec<i32>> {
    match archive.metadata().get(key) {
        Some(inference_tensor::quantized::gguf_file::Value::Array(values)) => values
            .iter()
            .map(|value| value.to_i32().map_err(anyhow::Error::from))
            .collect(),
        _ => anyhow::bail!("{VISION_FAMILY} GGUF lacks `{key}`"),
    }
}

pub fn build_granite_speech_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_architecture(archive, TEXT_ARCHITECTURE)?;
    validate_projector(archive, SPEECH_PROJECTOR)?;
    let mut bindings = text_bindings(archive, "language_model.model", "language_model.lm_head")?;
    bind_granite_speech_audio(&TensorInventory::from_archive(archive), &mut bindings)?;
    Ok(bindings)
}

fn bind_granite_speech_audio(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    bind_required_linear(
        inventory,
        bindings,
        "encoder.input_linear",
        "a.input_projection",
    )?;
    bind_required_linear(inventory, bindings, "encoder.out", "a.enc_ctc_out")?;
    bind_required_linear(inventory, bindings, "encoder.out_mid", "a.enc_ctc_out_mid")?;
    for layer in inventory.require_layers("a.blk.", SPEECH_FAMILY)? {
        let native = format!("encoder.layers.{layer}");
        let source = format!("a.blk.{layer}");
        bind_required_linears(
            inventory,
            bindings,
            &native,
            &source,
            &[
                ("attn.pre_norm", "ln1"),
                ("attn.to_q", "attn_q"),
                ("attn.to_out", "attn_out"),
                ("ff1.pre_norm", "ffn_norm"),
                ("ff1.up_proj", "ffn_up"),
                ("ff1.down_proj", "ffn_down"),
                ("ff2.pre_norm", "ffn_norm_1"),
                ("ff2.up_proj", "ffn_up_1"),
                ("ff2.down_proj", "ffn_down_1"),
                ("conv.norm", "norm_conv"),
                ("post_norm", "ln2"),
                // llama.cpp folds the running statistics into these: a scale and a shift
                ("conv.batch_norm", "conv_norm"),
            ],
        )?;
        bindings.insert(
            format!("{native}.attn.to_kv.weight"),
            GgufTensorBinding::concat(
                vec![
                    GgufTensorBinding::tensor(format!("{source}.attn_k.weight")),
                    GgufTensorBinding::tensor(format!("{source}.attn_v.weight")),
                ],
                0,
            ),
        );
        bind_required(
            inventory,
            bindings,
            format!("{native}.attn.rel_pos_emb.weight"),
            format!("{source}.attn_rel_pos_emb"),
        )?;
        // llama.cpp squeezes the convolutions' unit dimensions
        for (name, role, unit_dim) in [("up_conv", "conv_pw1", 2), ("down_conv", "conv_pw2", 2)] {
            bind_required_with(
                inventory,
                bindings,
                format!("{native}.conv.{name}.weight"),
                format!("{source}.{role}.weight"),
                |source, shape| {
                    let mut dims = shape.to_vec();
                    dims.insert(unit_dim, 1);
                    Ok(GgufTensorBinding::tensor(source).reshape(dims))
                },
            )?;
            bind_required(
                inventory,
                bindings,
                format!("{native}.conv.{name}.bias"),
                format!("{source}.{role}.bias"),
            )?;
        }
        bind_required_with(
            inventory,
            bindings,
            format!("{native}.conv.depth_conv.conv.weight"),
            format!("{source}.conv_dw.weight"),
            |source, shape| {
                Ok(GgufTensorBinding::tensor(source).reshape(vec![shape[0], 1, shape[1]]))
            },
        )?;
    }
    bind_required_linear(inventory, bindings, "projector.linear", "a.proj_linear")?;
    bind_required_linear(
        inventory,
        bindings,
        "projector.qformer.layernorm",
        "a.proj_norm",
    )?;
    bind_required(inventory, bindings, "projector.query", "a.proj_query")?;
    for layer in inventory.require_layers("a.proj_blk.", SPEECH_FAMILY)? {
        bind_qformer_layer(
            inventory,
            bindings,
            &format!("projector.qformer.encoder.layer.{layer}"),
            &format!("a.proj_blk.{layer}"),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HIDDEN: usize = 8;

    fn linear(name: &str, out: usize, inp: usize) -> Vec<(String, Vec<usize>)> {
        vec![
            (format!("{name}.weight"), vec![out, inp]),
            (format!("{name}.bias"), vec![out]),
        ]
    }

    fn qformer_layer(prefix: &str) -> Vec<(String, Vec<usize>)> {
        QFORMER_ROLES
            .iter()
            .flat_map(|(_, role)| linear(&format!("{prefix}.{role}"), HIDDEN, HIDDEN))
            .collect()
    }

    fn vision_block(block: usize) -> Vec<(String, Vec<usize>)> {
        let p = format!("v.proj_blk.{block}");
        let mut tensors = qformer_layer(&p);
        for name in ["norm", "linear", "post_norm"] {
            tensors.extend(linear(&format!("{p}.{name}"), HIDDEN, HIDDEN));
        }
        tensors.push((format!("{p}.query"), vec![1, 4, HIDDEN]));
        tensors.push((format!("{p}.img_pos"), vec![1, 16, HIDDEN]));
        tensors
    }

    #[test]
    fn layerwise_projectors_take_transformers_order_from_their_feature_layers() -> Result<()> {
        let tensors = (0..3).flat_map(vision_block).collect::<Vec<_>>();
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let mut bindings = GgufBindingMap::new();
        // block 0 reads the deeper layer, as llama.cpp writes them; block 2 is the first spatial projector
        bind_granite4_vision_projectors(&inventory, &[-1, -1, 0], &[26, 8, 26], &mut bindings)?;
        for (native, source) in [
            ("model.layerwise_projectors.1.query", "v.proj_blk.0.query"),
            ("model.layerwise_projectors.0.query", "v.proj_blk.1.query"),
            ("model.spatial_projectors.0.query", "v.proj_blk.2.query"),
            (
                "model.layerwise_projectors.0.qformer.layernorm.weight",
                "v.proj_blk.1.post_norm.weight",
            ),
            (
                "model.layerwise_projectors.0.qformer.encoder.layer.0.crossattention.attention.key.bias",
                "v.proj_blk.1.cross_attn_k.bias",
            ),
        ] {
            assert_eq!(
                bindings.get(native),
                Some(&GgufTensorBinding::tensor(source)),
                "{native}"
            );
        }
        Ok(())
    }

    #[test]
    fn speech_kv_and_squeezed_convolutions_return_to_transformers_layouts() -> Result<()> {
        let (inner, kernel) = (2 * HIDDEN, 3);
        let mut tensors = Vec::new();
        for name in [
            "a.input_projection",
            "a.enc_ctc_out",
            "a.enc_ctc_out_mid",
            "a.proj_linear",
            "a.proj_norm",
        ] {
            tensors.extend(linear(name, HIDDEN, HIDDEN));
        }
        tensors.push(("a.proj_query".to_string(), vec![1, 3, HIDDEN]));
        tensors.extend(qformer_layer("a.proj_blk.0"));
        let b = "a.blk.0";
        for role in [
            "ln1",
            "attn_out",
            "ffn_norm",
            "ffn_up",
            "ffn_down",
            "ffn_norm_1",
            "ffn_up_1",
            "ffn_down_1",
            "norm_conv",
            "ln2",
            "conv_norm",
        ] {
            tensors.extend(linear(&format!("{b}.{role}"), HIDDEN, HIDDEN));
        }
        for role in ["attn_q", "attn_k", "attn_v"] {
            tensors.push((format!("{b}.{role}.weight"), vec![HIDDEN, HIDDEN]));
        }
        tensors.push((format!("{b}.attn_rel_pos_emb"), vec![9, 4]));
        tensors.extend(linear(&format!("{b}.conv_pw1"), 2 * inner, HIDDEN));
        tensors.extend(linear(&format!("{b}.conv_pw2"), HIDDEN, inner));
        tensors.push((format!("{b}.conv_dw.weight"), vec![inner, kernel]));
        let inventory = TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        let mut bindings = GgufBindingMap::new();
        bind_granite_speech_audio(&inventory, &mut bindings)?;
        assert_eq!(
            bindings.get("encoder.layers.0.attn.to_kv.weight"),
            Some(&GgufTensorBinding::concat(
                vec![
                    GgufTensorBinding::tensor("a.blk.0.attn_k.weight"),
                    GgufTensorBinding::tensor("a.blk.0.attn_v.weight"),
                ],
                0,
            ))
        );
        assert_eq!(
            bindings.get("encoder.layers.0.conv.up_conv.weight"),
            Some(
                &GgufTensorBinding::tensor("a.blk.0.conv_pw1.weight").reshape(vec![
                    2 * inner,
                    HIDDEN,
                    1
                ])
            )
        );
        assert_eq!(
            bindings.get("encoder.layers.0.conv.depth_conv.conv.weight"),
            Some(
                &GgufTensorBinding::tensor("a.blk.0.conv_dw.weight")
                    .reshape(vec![inner, 1, kernel])
            )
        );
        assert_eq!(
            bindings.get("projector.qformer.encoder.layer.0.output_query.LayerNorm.weight"),
            Some(&GgufTensorBinding::tensor("a.proj_blk.0.ffn_norm.weight"))
        );
        Ok(())
    }
}
