//! Tiny random-weight RoPE encoders (Nomic BERT, ModernBERT, NeoBERT, bidirectional Llama): detection, ModernBERT's
//! sliding window, bidirectional Llama layers, pooling without `modules.json`, ISQ and UQFF.

use std::path::Path;

use inference::{
    EmbeddingModelBuilder, EmbeddingRequestBuilder, IsqType, Model, ModelDType,
    UqffEmbeddingModelBuilder,
};

#[path = "../support/rope_encoders_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::{
    nomic_moe_config, prefix_weights, tiny_llama_embedding_checkpoint, tiny_modern_bert_checkpoint,
    tiny_neo_bert_checkpoint, tiny_nomic_bert_checkpoint,
};
use traces::uqff_files;

const PROMPT: &str = "hello there";
const HIDDEN_SIZE: usize = 32;
// The Normalize module scales every embedding to unit length.
const NORM_TOLERANCE: f32 = 1e-4;
const EMBEDDING_TOLERANCE: f32 = 1e-5;
// The fixture's local_attention of 4 attends 2 tokens each way; 3 tokens all see each other, 9 do not.
const INSIDE_WINDOW: [u32; 3] = [40, 41, 42];
const PAST_WINDOW: [u32; 9] = [40, 41, 42, 43, 44, 45, 46, 47, 48];
const WIDE_LOCAL_ATTENTION: usize = 64;
// Prompts that differ only after their first token: CLS pooling sees the change only through bidirectional attention.
const CHANGED_LATER: [[u32; 3]; 2] = [[40, 41, 42], [40, 41, 43]];

fn cpu_embedding_builder(dir: &Path) -> EmbeddingModelBuilder {
    EmbeddingModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

async fn embed_tokens(model: &Model, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
    let request = EmbeddingRequestBuilder::new()
        .add_tokens(tokens.to_vec())
        .build()?;
    let response = serde_json::to_value(model.generate_embeddings(request).await?)?;
    Ok(serde_json::from_value(
        response["data"][0]["embedding"].clone(),
    )?)
}

async fn assert_unit_embedding(dir: &Path) -> anyhow::Result<()> {
    let model = cpu_embedding_builder(dir).build().await?;
    let embedding = model.generate_embedding(PROMPT).await?;
    assert_eq!(embedding.len(), HIDDEN_SIZE);
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < NORM_TOLERANCE, "norm {norm}");
    Ok(())
}

async fn assert_uqff_reload_matches(dir: &Path) -> anyhow::Result<()> {
    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_embedding_builder(dir)
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let expected = quantized.generate_embedding(PROMPT).await?;
    drop(quantized);
    let written = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(!written.is_empty(), "no UQFF written");
    let reloaded = UqffEmbeddingModelBuilder::new(dir.to_string_lossy(), vec![written[0].clone()])
        .into_inner()
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    assert_eq!(reloaded.generate_embedding(PROMPT).await?, expected);
    Ok(())
}

#[tokio::test]
async fn rope_encoders_are_detected_by_class_and_embed_to_unit_length() -> anyhow::Result<()> {
    assert_unit_embedding(tiny_nomic_bert_checkpoint(|_| {})?.path()).await?;
    assert_unit_embedding(tiny_modern_bert_checkpoint(|_| {})?.path()).await
}

#[tokio::test]
async fn modern_bert_sliding_layers_see_only_their_window() -> anyhow::Result<()> {
    let narrow = tiny_modern_bert_checkpoint(|_| {})?;
    let narrow = cpu_embedding_builder(narrow.path()).build().await?;
    // the same weights, with a window wider than any prompt here
    let wide = tiny_modern_bert_checkpoint(|config| {
        config["local_attention"] = WIDE_LOCAL_ATTENTION.into();
    })?;
    let wide = cpu_embedding_builder(wide.path()).build().await?;

    let inside = max_diff(
        &embed_tokens(&narrow, &INSIDE_WINDOW).await?,
        &embed_tokens(&wide, &INSIDE_WINDOW).await?,
    );
    assert!(
        inside < EMBEDDING_TOLERANCE,
        "a prompt inside the window moved by {inside}"
    );
    let past = max_diff(
        &embed_tokens(&narrow, &PAST_WINDOW).await?,
        &embed_tokens(&wide, &PAST_WINDOW).await?,
    );
    assert!(
        past > EMBEDDING_TOLERANCE,
        "the window left a long prompt unchanged"
    );
    Ok(())
}

#[tokio::test]
async fn an_isq_nomic_bert_and_a_reload_of_its_uqff_embed_alike() -> anyhow::Result<()> {
    assert_uqff_reload_matches(tiny_nomic_bert_checkpoint(|_| {})?.path()).await
}

#[tokio::test]
async fn a_prefixed_isq_modern_bert_and_a_reload_of_its_uqff_embed_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_modern_bert_checkpoint(|config| {
        config["architectures"] = serde_json::json!(["ModernBertForMaskedLM"]);
    })?;
    prefix_weights(checkpoint.path(), "model")?;
    assert_uqff_reload_matches(checkpoint.path()).await
}

#[tokio::test]
async fn a_bidirectional_llama_lets_its_first_token_see_later_ones() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_embedding_checkpoint(|_| {})?;
    assert_unit_embedding(checkpoint.path()).await?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let diff = max_diff(
        &embed_tokens(&model, &CHANGED_LATER[0]).await?,
        &embed_tokens(&model, &CHANGED_LATER[1]).await?,
    );
    assert!(
        diff > EMBEDDING_TOLERANCE,
        "a later token left the first token's state unchanged"
    );
    Ok(())
}

#[tokio::test]
async fn a_prefixed_isq_euro_bert_and_a_reload_of_its_uqff_embed_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_embedding_checkpoint(|config| {
        config["architectures"] = serde_json::json!(["EuroBertForMaskedLM"]);
    })?;
    prefix_weights(checkpoint.path(), "model")?;
    assert_uqff_reload_matches(checkpoint.path()).await
}

#[tokio::test]
async fn a_neo_bert_without_modules_mean_pools_and_reloads_from_uqff() -> anyhow::Result<()> {
    let checkpoint = tiny_neo_bert_checkpoint()?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let embedding = model.generate_embedding(PROMPT).await?;
    assert_eq!(embedding.len(), HIDDEN_SIZE);
    drop(model);
    prefix_weights(checkpoint.path(), "model")?;
    assert_uqff_reload_matches(checkpoint.path()).await
}

#[tokio::test]
async fn a_nomic_moe_routes_tokens_and_reloads_from_uqff() -> anyhow::Result<()> {
    let top_two = tiny_nomic_bert_checkpoint(nomic_moe_config)?;
    assert_unit_embedding(top_two.path()).await?;
    // the same weights routed to one expert per token
    let top_one = tiny_nomic_bert_checkpoint(|config| {
        nomic_moe_config(config);
        config["moe_top_k"] = 1.into();
    })?;
    let diff = max_diff(
        &embed_tokens(
            &cpu_embedding_builder(top_two.path()).build().await?,
            &PAST_WINDOW,
        )
        .await?,
        &embed_tokens(
            &cpu_embedding_builder(top_one.path()).build().await?,
            &PAST_WINDOW,
        )
        .await?,
    );
    assert!(
        diff > EMBEDDING_TOLERANCE,
        "top-k left the embedding unchanged"
    );
    assert_uqff_reload_matches(top_two.path()).await
}

const CLASSIFIER_TOLERANCE: f32 = 1e-4;
const HEAD_STD: f32 = 0.5;
const MODERN_NORM_EPS: f64 = 1e-5;

/// The checkpoint's pooled last hidden state, through a lone Pooling module of `mode`.
async fn pooled_state(dir: &Path, mode: &str) -> anyhow::Result<Vec<f32>> {
    std::fs::create_dir_all(dir.join("1_Pooling"))?;
    std::fs::write(
        dir.join("1_Pooling/config.json"),
        serde_json::json!({
            "word_embedding_dimension": HIDDEN_SIZE,
            "pooling_mode_cls_token": mode == "cls",
            "pooling_mode_mean_tokens": mode == "mean",
            "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false,
        })
        .to_string(),
    )?;
    std::fs::write(
        dir.join("modules.json"),
        serde_json::json!([
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        ])
        .to_string(),
    )?;
    let state = embed_tokens(&cpu_embedding_builder(dir).build().await?, &PAST_WINDOW).await?;
    std::fs::remove_file(dir.join("modules.json"))?;
    std::fs::remove_dir_all(dir.join("1_Pooling"))?;
    Ok(state)
}

/// Makes `dir` a one-label `class` with the encoder under `model.` and `head` beside it.
fn as_classifier(
    dir: &Path,
    class: &str,
    head: Vec<(&str, inference_tensor::Tensor)>,
    edit: impl FnOnce(&mut serde_json::Value),
) -> anyhow::Result<()> {
    prefix_weights(dir, "model")?;
    let path = dir.join("model.safetensors");
    let mut weights = inference_tensor::safetensors::load(&path, &inference_tensor::Device::Cpu)?;
    weights.extend(head.into_iter().map(|(name, t)| (name.to_string(), t)));
    inference_tensor::safetensors::save(&weights, &path)?;
    let config_path = dir.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    config["architectures"] = serde_json::json!([class]);
    config["id2label"] = serde_json::json!({"0": "LABEL_0"});
    edit(&mut config);
    std::fs::write(config_path, config.to_string())?;
    Ok(())
}

fn random(shape: &[usize]) -> anyhow::Result<inference_tensor::Tensor> {
    Ok(inference_tensor::Tensor::randn(
        0f32,
        HEAD_STD,
        shape,
        &inference_tensor::Device::Cpu,
    )?)
}

#[tokio::test]
async fn a_modern_bert_classifier_scores_its_mean_state_through_its_head() -> anyhow::Result<()> {
    use inference_tensor::{D, Device, Tensor};
    let checkpoint = tiny_modern_bert_checkpoint(|_| {})?;
    let mean = pooled_state(checkpoint.path(), "mean").await?;
    let (dense, norm, out, out_bias) = (
        random(&[HIDDEN_SIZE, HIDDEN_SIZE])?,
        random(&[HIDDEN_SIZE])?,
        random(&[1, HIDDEN_SIZE])?,
        random(&[1])?,
    );
    // dense (no bias), exact GELU, a bias-free LayerNorm, then the label projection
    let xs = Tensor::from_vec(mean, (1, HIDDEN_SIZE), &Device::Cpu)?
        .matmul(&dense.t()?)?
        .gelu_erf()?;
    let centered = xs.broadcast_sub(&xs.mean_keepdim(D::Minus1)?)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = centered
        .broadcast_div(&(var + MODERN_NORM_EPS)?.sqrt()?)?
        .broadcast_mul(&norm)?;
    let expected = normed
        .matmul(&out.t()?)?
        .broadcast_add(&out_bias)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    as_classifier(
        checkpoint.path(),
        "ModernBertForSequenceClassification",
        vec![
            ("head.dense.weight", dense),
            ("head.norm.weight", norm),
            ("classifier.weight", out),
            ("classifier.bias", out_bias),
        ],
        |config| config["classifier_pooling"] = "mean".into(),
    )?;
    let logit = embed_tokens(
        &cpu_embedding_builder(checkpoint.path()).build().await?,
        &PAST_WINDOW,
    )
    .await?;
    let diff = max_diff(&expected, &logit);
    assert!(
        diff < CLASSIFIER_TOLERANCE,
        "logit {logit:?}, expected {expected:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_neo_bert_classifier_scores_its_first_token_through_dense_and_tanh() -> anyhow::Result<()>
{
    use inference_tensor::{Device, Tensor};
    let checkpoint = tiny_neo_bert_checkpoint()?;
    let first = pooled_state(checkpoint.path(), "cls").await?;
    let (dense, dense_bias, out, out_bias) = (
        random(&[HIDDEN_SIZE, HIDDEN_SIZE])?,
        random(&[HIDDEN_SIZE])?,
        random(&[1, HIDDEN_SIZE])?,
        random(&[1])?,
    );
    let expected = Tensor::from_vec(first, (1, HIDDEN_SIZE), &Device::Cpu)?
        .matmul(&dense.t()?)?
        .broadcast_add(&dense_bias)?
        .tanh()?
        .matmul(&out.t()?)?
        .broadcast_add(&out_bias)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    as_classifier(
        checkpoint.path(),
        "NeoBERTForSequenceClassification",
        vec![
            ("dense.weight", dense),
            ("dense.bias", dense_bias),
            ("classifier.weight", out),
            ("classifier.bias", out_bias),
        ],
        |_| {},
    )?;
    let logit = embed_tokens(
        &cpu_embedding_builder(checkpoint.path()).build().await?,
        &PAST_WINDOW,
    )
    .await?;
    let diff = max_diff(&expected, &logit);
    assert!(
        diff < CLASSIFIER_TOLERANCE,
        "logit {logit:?}, expected {expected:?}"
    );
    Ok(())
}
