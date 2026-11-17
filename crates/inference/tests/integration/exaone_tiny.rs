//! Tiny random-weight EXAONE checkpoints load through their Hugging Face classes and generate: EXAONE 4.0 in the 32B's
//! layout (sliding layers beside NoPE global ones) and the 1.2B's (no window, RoPE on every layer, tied embeddings),
//! and EXAONE 3.5 under its checkpoint's names.

use std::{collections::HashMap, path::Path};

use inference::{ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "../support/recording.rs"]
mod recording;

const LLAMA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llama_tiny");
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
const LAYERS: usize = 4;
// shorter than the chat prompt, so the sliding layers drop keys
const SLIDING_WINDOW: usize = 4;
const MAX_LEN: usize = 4;

fn config(sliding: bool, tied: bool) -> serde_json::Value {
    let (window, layer_types) = if sliding {
        (
            serde_json::json!(SLIDING_WINDOW),
            serde_json::json!([
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ]),
        )
    } else {
        (
            serde_json::Value::Null,
            serde_json::json!(vec!["full_attention"; LAYERS]),
        )
    };
    serde_json::json!({
        "architectures": ["Exaone4ForCausalLM"],
        "model_type": "exaone4",
        "hidden_size": 32,
        "intermediate_size": 48,
        "vocab_size": 266,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "rope_scaling": {
            "factor": 16.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 64, "rope_type": "llama3"
        },
        "hidden_act": "silu",
        "sliding_window": window,
        "sliding_window_pattern": if sliding { serde_json::json!("LLLG") } else { serde_json::Value::Null },
        "layer_types": layer_types,
        "tie_word_embeddings": tied,
    })
}

fn exaone3_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["ExaoneForCausalLM"],
        "model_type": "exaone",
        "hidden_size": 32,
        "intermediate_size": 48,
        "vocab_size": 266,
        "num_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 512,
        "layer_norm_epsilon": 1e-5,
        "rope_theta": 1000000.0,
        "rope_scaling": {
            "factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 64, "rope_type": "llama3"
        },
        "activation_function": "silu",
        "tie_word_embeddings": true,
    })
}

async fn loads_and_generates(
    config: serde_json::Value,
    spec: inference_nn::decoder::DecoderSpec,
    expected: &[&str],
    unexpected: &str,
) -> anyhow::Result<()> {
    let staging = tempfile::tempdir()?;
    let config_path = staging.path().join("config.json");
    std::fs::write(&config_path, config.to_string())?;
    let mut files = recording::fixture_files(LLAMA)?;
    files.retain(|path| path.file_name().is_some_and(|name| name != "config.json"));
    files.push(config_path);
    files.push(TOKENIZER.into());
    let files = files
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<&Path>>();
    let dir = recording::record_checkpoint_seeded_with_shapes(
        &files,
        LAYERS,
        // llama.cpp's precomputed llama3 factors; a safetensors checkpoint has none
        &["model.rope_freqs.weight", "transformer.rope_freqs.weight"],
        HashMap::new(),
        |vb, metadata| {
            inference_nn::decoder::CausalLm::new(
                &spec,
                vb,
                true,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )?;
    let tensors = inference_tensor::safetensors::load(
        dir.path().join("model.safetensors"),
        &inference_tensor::Device::Cpu,
    )?;
    for name in expected {
        anyhow::ensure!(tensors.contains_key(*name), "no {name}");
    }
    anyhow::ensure!(!tensors.contains_key(unexpected), "{unexpected} loaded");

    let model = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let request = RequestBuilder::from(
        TextMessages::new().add_message(TextMessageRole::User, "hello there, how are you today"),
    )
    .set_sampler_max_len(MAX_LEN)
    .set_sampler_topk(1);
    let response = model.send_chat_request(request).await?;
    anyhow::ensure!(
        response.usage.completion_tokens > 0,
        "the model generated nothing"
    );
    Ok(())
}

#[tokio::test]
async fn exaone4_sliding_and_windowless_layouts_load_and_generate() -> anyhow::Result<()> {
    let sliding =
        serde_json::from_value::<inference_models_other::exaone4::Config>(config(true, false))?
            .decoder_spec()?;
    assert_eq!(sliding.no_rope_layers, vec![3]);
    assert_eq!(sliding.layer_windows[0], Some(SLIDING_WINDOW));
    assert_eq!(sliding.layer_windows[3], None);
    let windowless =
        serde_json::from_value::<inference_models_other::exaone4::Config>(config(false, true))?
            .decoder_spec()?;
    assert!(windowless.no_rope_layers.is_empty());

    for (sliding, tied) in [(true, false), (false, true)] {
        let config = config(sliding, tied);
        let spec =
            serde_json::from_value::<inference_models_other::exaone4::Config>(config.clone())?
                .decoder_spec()?;
        loads_and_generates(
            config,
            spec,
            &[
                "model.layers.0.self_attn.q_norm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.post_feedforward_layernorm.weight",
            ],
            "model.layers.0.input_layernorm.weight",
        )
        .await
        .map_err(|error| anyhow::anyhow!("sliding {sliding}: {error}"))?;
    }
    Ok(())
}

#[tokio::test]
async fn exaone3_loads_under_its_checkpoint_names_and_generates() -> anyhow::Result<()> {
    let config = exaone3_config();
    let spec = serde_json::from_value::<inference_models_other::exaone::Config>(config.clone())?
        .decoder_spec()?;
    loads_and_generates(
        config,
        spec,
        &[
            "transformer.wte.weight",
            "transformer.ln_f.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.ln_2.weight",
            "transformer.h.0.attn.attention.q_proj.weight",
            "transformer.h.0.attn.attention.out_proj.weight",
            "transformer.h.0.mlp.c_fc_0.weight",
            "transformer.h.0.mlp.c_fc_1.weight",
            "transformer.h.0.mlp.c_proj.weight",
        ],
        "lm_head.weight",
    )
    .await
}
