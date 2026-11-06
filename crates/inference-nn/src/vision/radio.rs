//! C-RADIO's tower shape from its training config (`vision_config.args`, as Nemotron Nano Omni ships it), in the keys
//! a config that states the shape uses.

// timm ViTs a RADIO training config names in `args.model`: (name, hidden, heads, layers, mlp ratio)
const RADIO_VITS: [(&str, usize, usize, usize, f64); 1] =
    [("vit_huge_patch16_224", 1280, 16, 32, 4.0)];
// timm's ViT LayerNorm
const RADIO_LAYER_NORM_EPS: f64 = 1e-6;

/// Fills a RADIO training config's tower shape (Nano Omni's `vision_config`) in as Bolt's config states it.
pub fn fill_radio_shape(vision: &mut serde_json::Value) -> anyhow::Result<()> {
    let args = &vision["args"];
    let model = args["model"].as_str().unwrap_or_default();
    let Some(&(_, hidden, heads, layers, mlp_ratio)) =
        RADIO_VITS.iter().find(|(name, ..)| *name == model)
    else {
        anyhow::bail!("RADIO ViT `{model}` is not supported");
    };
    // one cls token per teacher, registers padding the prefix to `register_multiple`
    let cls = if args["cls_token_per_teacher"].as_bool() == Some(true) {
        args["teachers"].as_array().map_or(1, Vec::len)
    } else {
        1
    };
    let multiple = args["register_multiple"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0);
    let registers = if multiple > 0 {
        multiple - cls % multiple
    } else {
        0
    };
    let shape = serde_json::json!({
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_hidden_layers": layers,
        "mlp_ratio": mlp_ratio,
        "layer_norm_eps": RADIO_LAYER_NORM_EPS,
        "num_cls_tokens": cls,
        "num_registers": registers,
        "max_img_size": args["cpe_max_size"],
    });
    let object = vision
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("`vision_config` is not an object"))?;
    for (key, value) in shape.as_object().into_iter().flatten() {
        object.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::fill_radio_shape;

    // Nano Omni's `vision_config`: RADIO's training args where Bolt's config states the tower's shape
    #[test]
    fn a_radio_training_config_gives_bolts_shape() {
        let mut vision = serde_json::json!({
            "args": {
                "model": "vit_huge_patch16_224",
                "cpe_max_size": 2048,
                "register_multiple": 10,
                "cls_token_per_teacher": true,
                "teachers": [{"name": "clip"}, {"name": "siglip"}, {"name": "dino_v2"}, {"name": "sam"}],
            },
            "min_num_patches": 1024,
            "max_num_patches": 13312,
        });
        fill_radio_shape(&mut vision).unwrap();
        for (key, want) in [
            ("hidden_size", 1280.0),
            ("num_attention_heads", 16.0),
            ("num_hidden_layers", 32.0),
            ("mlp_ratio", 4.0),
            ("num_cls_tokens", 4.0),
            ("num_registers", 6.0),
            ("max_img_size", 2048.0),
        ] {
            assert_eq!(vision[key].as_f64(), Some(want), "{key}");
        }
        // the mmproj of the same checkpoint keeps cls and registers as one table of 10
        let prefix =
            vision["num_cls_tokens"].as_u64().unwrap() + vision["num_registers"].as_u64().unwrap();
        assert_eq!(prefix, 10);
        assert_eq!(vision["min_num_patches"], 1024);
    }

    #[test]
    fn an_unknown_radio_vit_is_refused() {
        let mut vision = serde_json::json!({"args": {"model": "vit_tiny_patch16_224"}});
        let error = fill_radio_shape(&mut vision).unwrap_err();
        assert!(
            error.to_string().contains("vit_tiny_patch16_224"),
            "{error}"
        );
    }
}
