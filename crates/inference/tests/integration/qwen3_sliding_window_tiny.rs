//! Qwen3 and Qwen3-MoE checkpoints ship a `sliding_window` with `use_sliding_window: false`; the window reaches a
//! layer only when sliding is on and the layer is past `max_window_layers`.

use std::collections::HashMap;
use std::path::Path;

use inference::{
    Model, ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};
use inference_models_qwen::{qwen3, qwen3_moe};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "../support/recording.rs"]
mod recording;

// The tiny Qwen3 shape, with the tiny Llama's tokenizer files and chat template
const QWEN3_CONFIG: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen3_embedding_tiny/config.json"
);
const LLAMA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llama_tiny");
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
// Shorter than the templated prompt, so applying it changes what the prompt attends to
const WINDOW: usize = 2;
const PROMPT: &str = "hello there";
const MAX_LEN: usize = 6;
const LOGPROB_TOLERANCE: f32 = 1e-5;
const MOE_ARCHITECTURE: &str = "Qwen3MoeForCausalLM";
const MOE_MODEL_TYPE: &str = "qwen3_moe";
const MOE_INTERMEDIATE: usize = 16;
const MOE_EXPERTS: usize = 4;
const MOE_EXPERTS_PER_TOKEN: usize = 2;

async fn greedy_trace(dir: &Path) -> anyhow::Result<Vec<(u32, f32)>> {
    let model: Model = TextModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let request =
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT))
            .set_sampler_max_len(MAX_LEN)
            .set_sampler_topk(1)
            .return_logprobs(true)
            .set_sampler_topn_logprobs(1);
    let response = model.send_chat_request(request).await?;
    let trace = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| (t.top_logprobs[0].token, t.top_logprobs[0].logprob))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    anyhow::ensure!(!trace.is_empty(), "the model generated nothing");
    Ok(trace)
}

fn tiny_config() -> anyhow::Result<serde_json::Value> {
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(QWEN3_CONFIG)?)?;
    config["tie_word_embeddings"] = false.into();
    Ok(config)
}

type Build<'a> = &'a dyn Fn(
    inference_quant::ShardedVarBuilder,
    inference_nn::model::NormalLoadingMetadata,
) -> inference_tensor::Result<()>;

// HF stores Qwen3-MoE experts one by one; the layout detection reads these shapes before any tensor.
fn expert_shapes(num_layers: usize, hidden: usize) -> HashMap<String, Vec<usize>> {
    (0..num_layers)
        .flat_map(|layer| (0..MOE_EXPERTS).map(move |expert| (layer, expert)))
        .flat_map(|(layer, expert)| {
            let p = format!("model.layers.{layer}.mlp.experts.{expert}");
            [
                (
                    format!("{p}.gate_proj.weight"),
                    vec![MOE_INTERMEDIATE, hidden],
                ),
                (
                    format!("{p}.up_proj.weight"),
                    vec![MOE_INTERMEDIATE, hidden],
                ),
                (
                    format!("{p}.down_proj.weight"),
                    vec![hidden, MOE_INTERMEDIATE],
                ),
            ]
        })
        .collect()
}

// Records one checkpoint, then decodes it under configs that name a window: only one that applies it may differ.
async fn assert_window_reaches_only_sliding_layers(
    config: serde_json::Value,
    num_layers: usize,
    shapes: HashMap<String, Vec<usize>>,
    build: Build<'_>,
) -> anyhow::Result<()> {
    let staging = tempfile::tempdir()?;
    let config_path = staging.path().join("config.json");
    std::fs::write(&config_path, serde_json::to_string(&config)?)?;

    let mut files = recording::fixture_files(LLAMA)?;
    files.retain(|path| path.file_name().is_some_and(|name| name != "config.json"));
    files.push(config_path);
    files.push(TOKENIZER.into());
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    let checkpoint =
        recording::record_checkpoint_with_shapes(&files, num_layers, &[], shapes, build)?;
    let expected = greedy_trace(checkpoint.path()).await?;

    // (use_sliding_window, max_window_layers, the window reaches some layer)
    for (enabled, max_window_layers, applies) in [
        (false, 0, false),
        (true, num_layers, false),
        (true, 0, true),
    ] {
        let mut windowed = config.clone();
        windowed["sliding_window"] = WINDOW.into();
        windowed["use_sliding_window"] = enabled.into();
        windowed["max_window_layers"] = max_window_layers.into();
        std::fs::write(
            checkpoint.path().join("config.json"),
            serde_json::to_string(&windowed)?,
        )?;
        let actual = greedy_trace(checkpoint.path()).await?;
        let same = expected.len() == actual.len()
            && expected
                .iter()
                .zip(&actual)
                .all(|(e, a)| e.0 == a.0 && (e.1 - a.1).abs() < LOGPROB_TOLERANCE);
        anyhow::ensure!(
            same != applies,
            "use_sliding_window {enabled}, max_window_layers {max_window_layers}: {actual:?} vs {expected:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_qwen3_window_reaches_only_its_sliding_layers() -> anyhow::Result<()> {
    let config = tiny_config()?;
    let cfg: qwen3::Config = serde_json::from_value(config.clone())?;
    assert_window_reaches_only_sliding_layers(
        config,
        cfg.num_hidden_layers,
        HashMap::new(),
        &|vb, metadata| {
            qwen3::Model::new(
                &cfg.decoder_spec(),
                vb,
                true,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )
    .await
}

#[tokio::test]
async fn a_qwen3_moe_window_reaches_only_its_sliding_layers() -> anyhow::Result<()> {
    let mut config = tiny_config()?;
    config["architectures"] = serde_json::json!([MOE_ARCHITECTURE]);
    config["model_type"] = MOE_MODEL_TYPE.into();
    config["moe_intermediate_size"] = MOE_INTERMEDIATE.into();
    config["num_experts"] = MOE_EXPERTS.into();
    config["num_experts_per_tok"] = MOE_EXPERTS_PER_TOKEN.into();
    config["mlp_only_layers"] = serde_json::json!([]);
    config["decoder_sparse_step"] = 1.into();
    config["norm_topk_prob"] = true.into();
    let cfg: qwen3_moe::Config = serde_json::from_value(config.clone())?;
    let shapes = expert_shapes(cfg.num_hidden_layers, cfg.hidden_size);
    assert_window_reaches_only_sliding_layers(
        config,
        cfg.num_hidden_layers,
        shapes,
        &|vb, metadata| {
            qwen3_moe::Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
                .map(|_| ())
        },
    )
    .await
}
