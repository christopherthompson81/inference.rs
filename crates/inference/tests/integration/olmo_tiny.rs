//! Tiny random-weight OLMo 1, OLMo 2, Olmo 3 (sliding layers on plain RoPE beside YaRN full ones) and OLMoE
//! checkpoints load through their Hugging Face classes and generate.

use std::{collections::HashMap, path::Path};

use inference::{ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder};
use inference_nn::{decoder::DecoderSpec, paged_attention::AttentionImplementation};

#[path = "../support/recording.rs"]
mod recording;

const LLAMA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llama_tiny");
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
const LAYERS: usize = 4;
const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 48;
const EXPERTS: usize = 4;
// shorter than the chat prompt, so the sliding layers drop keys
const SLIDING_WINDOW: usize = 4;
const MAX_LEN: usize = 4;

fn common(architecture: &str, model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "architectures": [architecture],
        "model_type": model_type,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": 266,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
    })
}

// the per-expert layout transformers saves, which the experts loader reads shapes of before loading
fn expert_shapes(experts: usize) -> HashMap<String, Vec<usize>> {
    (0..LAYERS)
        .flat_map(|layer| {
            (0..experts).flat_map(move |expert| {
                let p = format!("model.layers.{layer}.mlp.experts.{expert}");
                [
                    (format!("{p}.gate_proj.weight"), vec![INTERMEDIATE, HIDDEN]),
                    (format!("{p}.up_proj.weight"), vec![INTERMEDIATE, HIDDEN]),
                    (format!("{p}.down_proj.weight"), vec![HIDDEN, INTERMEDIATE]),
                ]
            })
        })
        .collect()
}

fn checkpoint(
    config: serde_json::Value,
    spec: DecoderSpec,
    shapes: HashMap<String, Vec<usize>>,
) -> anyhow::Result<tempfile::TempDir> {
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
    recording::record_checkpoint_seeded_with_shapes(&files, LAYERS, &[], shapes, |vb, metadata| {
        inference_models_other::olmo::Model::new(
            &spec,
            vb,
            false,
            metadata,
            AttentionImplementation::Eager,
        )
        .map(|_| ())
    })
}

async fn generates(dir: &tempfile::TempDir) -> anyhow::Result<()> {
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
async fn every_olmo_class_loads_and_generates() -> anyhow::Result<()> {
    let mut olmo = common("OlmoForCausalLM", "olmo");
    olmo["clip_qkv"] = 8.0.into();
    let olmo_spec =
        serde_json::from_value::<inference_models_other::olmo::OlmoConfig>(olmo.clone())?
            .decoder_spec();

    let mut olmo2 = common("Olmo2ForCausalLM", "olmo2");
    olmo2["rms_norm_eps"] = 1e-6.into();
    let olmo2_spec =
        serde_json::from_value::<inference_models_other::olmo::Olmo2Config>(olmo2.clone())?
            .decoder_spec()?;

    let mut olmo3 = olmo2.clone();
    olmo3["architectures"] = serde_json::json!(["Olmo3ForCausalLM"]);
    olmo3["model_type"] = "olmo3".into();
    olmo3["sliding_window"] = SLIDING_WINDOW.into();
    olmo3["layer_types"] = serde_json::json!([
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ]);
    olmo3["rope_scaling"] = serde_json::json!({
        "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 256,
        "beta_fast": 32.0, "beta_slow": 1.0, "attention_factor": 1.1,
    });
    let olmo3_spec =
        serde_json::from_value::<inference_models_other::olmo::Olmo2Config>(olmo3.clone())?
            .decoder_spec()?;
    assert!(olmo3_spec.local_rope.is_some());
    assert_eq!(olmo3_spec.layer_windows[0], Some(SLIDING_WINDOW));
    assert_eq!(olmo3_spec.layer_windows[3], None);

    let mut olmoe = common("OlmoeForCausalLM", "olmoe");
    olmoe["rms_norm_eps"] = 1e-5.into();
    olmoe["num_experts"] = EXPERTS.into();
    olmoe["num_experts_per_tok"] = 2.into();
    let olmoe_spec =
        serde_json::from_value::<inference_models_other::olmo::OlmoeConfig>(olmoe.clone())?
            .decoder_spec();

    for (config, spec, shapes) in [
        (olmo, olmo_spec, HashMap::new()),
        (olmo2, olmo2_spec, HashMap::new()),
        (olmo3, olmo3_spec, HashMap::new()),
        (olmoe, olmoe_spec, expert_shapes(EXPERTS)),
    ] {
        let name = config["model_type"].clone();
        let dir = checkpoint(config, spec, shapes)?;
        generates(&dir)
            .await
            .map_err(|error| anyhow::anyhow!("{name}: {error}"))?;
    }
    Ok(())
}
