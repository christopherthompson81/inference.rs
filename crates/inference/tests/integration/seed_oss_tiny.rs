//! A tiny random-weight Seed-OSS checkpoint (q/k/v bias without an o_proj bias, a head_dim apart from
//! hidden_size / heads) loads through its Hugging Face class and generates.

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
const LAYERS: usize = 2;
const HIDDEN: usize = 32;
const HEADS: usize = 4;
// wider than HIDDEN / HEADS, as the 36B's 80 x 128 heads are wider than its 5120 hidden
const HEAD_DIM: usize = 16;
const MAX_LEN: usize = 4;

fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["SeedOssForCausalLM"],
        "model_type": "seed_oss",
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "vocab_size": 266,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": 2,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "rope_scaling": {"rope_type": "default"},
        "hidden_act": "silu",
        "attention_bias": true,
        "attention_out_bias": false,
        "mlp_bias": false,
        "tie_word_embeddings": false,
    })
}

#[tokio::test]
async fn seed_oss_loads_and_generates() -> anyhow::Result<()> {
    let config = config();
    let spec = serde_json::from_value::<inference_models_other::seed_oss::Config>(config.clone())?
        .decoder_spec()?;
    assert!(spec.qkv_bias && !spec.o_bias);
    assert_eq!(spec.head_dim, HEAD_DIM);

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
        &[],
        HashMap::new(),
        |vb, metadata| {
            inference_models_other::seed_oss::Model::new(
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
    anyhow::ensure!(tensors.contains_key("model.layers.0.self_attn.k_proj.bias"));
    anyhow::ensure!(!tensors.contains_key("model.layers.0.self_attn.o_proj.bias"));

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
