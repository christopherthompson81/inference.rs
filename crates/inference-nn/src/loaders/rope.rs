use anyhow::Result;

use crate::model::RopePairing;

pub const QK_ROPE_LAYOUT_CONFIG_KEY: &str = "_inference_qk_rope_layout";

pub fn qk_rope_layout_from_config(config: &str) -> Result<Option<RopePairing>> {
    let config: serde_json::Value = serde_json::from_str(config)?;
    let Some(layout) = config
        .get(QK_ROPE_LAYOUT_CONFIG_KEY)
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    match layout {
        "adjacent" => Ok(Some(
            RopePairing::Adjacent,
        )),
        "half_split" => Ok(Some(
            RopePairing::HalfSplit,
        )),
        layout => anyhow::bail!(
            "model config `{QK_ROPE_LAYOUT_CONFIG_KEY}` must be `adjacent` or `half_split`, got `{layout}`"
        ),
    }
}
