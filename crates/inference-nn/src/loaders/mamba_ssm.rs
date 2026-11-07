//! state-spaces' original `mamba_ssm` checkpoints (`MambaLMHeadModel`): a config of `d_model`, `n_layer` and
//! `ssm_cfg` with no `architectures`, read as transformers' Mamba or Mamba-2 config.

use serde_json::{Value, json};

/// Set on a config translated from `mamba_ssm`'s, whose embeddings are `backbone.embedding` (transformers' are
/// `backbone.embeddings`).
pub const MAMBA_SSM_CONFIG_KEY: &str = "_inference_mamba_ssm";

// `mamba_ssm`'s defaults (`MambaConfig`, `Mamba`, `Mamba2`)
const PAD_VOCAB_MULTIPLE: u64 = 8;
const MAMBA1_STATE: u64 = 16;
const MAMBA2_STATE: u64 = 128;
const MAMBA2_HEAD_DIM: u64 = 64;
const MAMBA2_GROUPS: u64 = 1;
const MAMBA2_CHUNK: u64 = 256;
const CONV_KERNEL: u64 = 4;
const EXPAND: u64 = 2;
const NORM_EPS: f64 = 1e-5;
const MAMBA2_LAYER: &str = "Mamba2";

/// transformers' config for a `mamba_ssm` one, or `None` when `config` is not one.
#[allow(clippy::cast_precision_loss)]
pub fn mamba_ssm_config(config: &Value) -> Option<Value> {
    let object = config.as_object()?;
    if object.contains_key("architectures") || object.contains_key("model_type") {
        return None;
    }
    let (d_model, n_layer, ssm) = (
        object.get("d_model")?.as_u64()?,
        object.get("n_layer")?.as_u64()?,
        object.get("ssm_cfg")?,
    );
    let ssm_u64 = |key: &str, default: u64| ssm.get(key).and_then(Value::as_u64).unwrap_or(default);
    let multiple = object
        .get("pad_vocab_size_multiple")
        .and_then(Value::as_u64)
        .unwrap_or(PAD_VOCAB_MULTIPLE);
    // the embeddings are padded to the multiple
    let vocab = object.get("vocab_size")?.as_u64()?.div_ceil(multiple) * multiple;
    let expand = ssm_u64("expand", EXPAND);
    let mut translated = json!({
        "hidden_size": d_model,
        "num_hidden_layers": n_layer,
        "vocab_size": vocab,
        "conv_kernel": ssm_u64("d_conv", CONV_KERNEL),
        "expand": expand,
        "layer_norm_epsilon": object.get("norm_epsilon").and_then(Value::as_f64).unwrap_or(NORM_EPS),
        "use_bias": ssm.get("bias").and_then(Value::as_bool).unwrap_or(false),
        "use_conv_bias": ssm.get("conv_bias").and_then(Value::as_bool).unwrap_or(true),
        "tie_word_embeddings": object.get("tie_embeddings").and_then(Value::as_bool).unwrap_or(true),
        MAMBA_SSM_CONFIG_KEY: true,
    });
    let fields = translated.as_object_mut()?;
    if ssm.get("layer").and_then(Value::as_str) == Some(MAMBA2_LAYER) {
        let head_dim = ssm_u64("headdim", MAMBA2_HEAD_DIM);
        fields.insert("architectures".into(), json!(["Mamba2ForCausalLM"]));
        fields.insert("model_type".into(), json!("mamba2"));
        fields.insert("state_size".into(), json!(ssm_u64("d_state", MAMBA2_STATE)));
        fields.insert("head_dim".into(), json!(head_dim));
        fields.insert("num_heads".into(), json!(expand * d_model / head_dim));
        fields.insert("n_groups".into(), json!(ssm_u64("ngroups", MAMBA2_GROUPS)));
        fields.insert(
            "chunk_size".into(),
            json!(ssm_u64("chunk_size", MAMBA2_CHUNK)),
        );
    } else {
        fields.insert("architectures".into(), json!(["MambaForCausalLM"]));
        fields.insert("model_type".into(), json!("mamba"));
        fields.insert("state_size".into(), json!(ssm_u64("d_state", MAMBA1_STATE)));
        if let Some(rank) = ssm.get("dt_rank").filter(|rank| rank.is_u64()) {
            fields.insert("time_step_rank".into(), rank.clone());
        }
    }
    Some(translated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mamba_ssm_configs_read_as_transformers_ones() {
        let mamba = mamba_ssm_config(&json!({
            "d_model": 768, "n_layer": 24, "vocab_size": 50277, "ssm_cfg": {}, "pad_vocab_size_multiple": 8
        }))
        .expect("a mamba_ssm config");
        assert_eq!(mamba["architectures"], json!(["MambaForCausalLM"]));
        assert_eq!(mamba["vocab_size"], 50280);
        assert_eq!(mamba["state_size"], 16);
        assert_eq!(mamba[MAMBA_SSM_CONFIG_KEY], true);
        let mamba2 = mamba_ssm_config(&json!({
            "d_model": 768, "n_layer": 24, "vocab_size": 50277, "ssm_cfg": {"layer": "Mamba2"},
            "pad_vocab_size_multiple": 16, "tie_embeddings": true
        }))
        .expect("a mamba_ssm config");
        assert_eq!(mamba2["architectures"], json!(["Mamba2ForCausalLM"]));
        assert_eq!(mamba2["vocab_size"], 50288);
        assert_eq!(mamba2["num_heads"], 24);
        assert_eq!(mamba2["n_groups"], 1);
        assert!(
            mamba_ssm_config(&json!({"architectures": ["MambaForCausalLM"], "d_model": 768}))
                .is_none()
        );
    }
}
