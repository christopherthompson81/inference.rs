//! A Llama whose config sets head_dim apart from hidden_size / num_attention_heads, as some Llama-architecture
//! checkpoints do: the projections take the checkpoint's shapes and the model generates.

use inference::{
    Model, ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};
use inference_models_llama::llama::{Config, Llama};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "../support/recording.rs"]
mod recording;

const LLAMA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llama_tiny");
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
const ROPE_FREQS: &str = "model.rope_freqs.weight";
// hidden_size 32 over 2 heads would be 16
const HEAD_DIM: usize = 8;
const PROMPT: &str = "hello";
const MAX_LEN: usize = 4;

async fn generate(model: &Model) -> anyhow::Result<usize> {
    let request =
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT))
            .set_sampler_max_len(MAX_LEN)
            .set_sampler_topk(1);
    let response = model.send_chat_request(request).await?;
    Ok(response.usage.completion_tokens)
}

#[tokio::test]
async fn a_llama_with_an_explicit_head_dim_loads_its_projections_and_generates()
-> anyhow::Result<()> {
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{LLAMA}/config.json"))?)?;
    config["head_dim"] = HEAD_DIM.into();
    let cfg: Config = serde_json::from_value(config.clone())?;
    let staging = tempfile::tempdir()?;
    let config_path = staging.path().join("config.json");
    std::fs::write(&config_path, serde_json::to_string(&config)?)?;

    let mut files = recording::fixture_files(LLAMA)?;
    files.retain(|path| path.file_name().is_some_and(|name| name != "config.json"));
    files.push(config_path);
    files.push(TOKENIZER.into());
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    let checkpoint = recording::record_checkpoint(
        &files,
        cfg.num_hidden_layers,
        &[ROPE_FREQS],
        |vb, metadata| {
            Llama::new(
                &cfg.decoder_spec(),
                vb,
                true,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )?;

    let weights = inference_tensor::safetensors::load(
        checkpoint.path().join("model.safetensors"),
        &inference_tensor::Device::Cpu,
    )?;
    let shape = |name: &str| weights[name].dims().to_vec();
    let q_rows = cfg.num_attention_heads * HEAD_DIM;
    let kv_rows = cfg.num_key_value_heads * HEAD_DIM;
    assert_eq!(
        shape("model.layers.0.self_attn.q_proj.weight"),
        [q_rows, cfg.hidden_size]
    );
    assert_eq!(
        shape("model.layers.0.self_attn.k_proj.weight"),
        [kv_rows, cfg.hidden_size]
    );
    assert_eq!(
        shape("model.layers.0.self_attn.o_proj.weight"),
        [cfg.hidden_size, q_rows]
    );

    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    anyhow::ensure!(generate(&model).await? > 0, "the model generated nothing");
    Ok(())
}
