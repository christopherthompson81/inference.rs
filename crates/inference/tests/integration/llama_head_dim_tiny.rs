//! A Llama whose config sets head_dim apart from hidden_size / num_attention_heads, as some Llama-architecture
//! checkpoints do: the projections take the checkpoint's shapes and the model generates.

use inference::{
    Model, ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};
use inference_models_llama::llama::Config;

#[path = "../support/llama_tiny.rs"]
mod support;

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
    let checkpoint =
        support::tiny_llama_checkpoint_with(serde_json::json!({ "head_dim": HEAD_DIM }))?;
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(
        checkpoint.path().join("config.json"),
    )?)?;

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
