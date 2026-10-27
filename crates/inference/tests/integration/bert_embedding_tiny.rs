//! A tiny random-weight BERT embedder and cross-encoder: encoder detection, bidirectional attention, tokenizer and
//! length limits, ISQ, and reranking with pair segment ids.

use std::path::Path;

use inference::{
    EmbeddingModelBuilder, EmbeddingRequestBuilder, IsqType, Model, ModelDType, RerankRequest,
    RerankResponse, UqffEmbeddingModelBuilder,
};

#[path = "../support/bert_embedding_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::{
    JINA_V3_INSTRUCTION, JINA_V3_TASKS, prefix_encoder_weights, tiny_bert_checkpoint,
    tiny_jina_v3_checkpoint,
};
use traces::uqff_files;

const HIDDEN_SIZE: usize = 32;
// The Normalize module scales every embedding to unit length.
const NORM_TOLERANCE: f32 = 1e-4;
const EMBEDDING_TOLERANCE: f32 = 1e-5;
// Q8_0 on 32-wide layers keeps the embedding close, not equal
const ISQ_MIN_COSINE: f32 = 0.99;
// Prompts that differ only after their first token: CLS pooling sees the change only through bidirectional attention.
const PROMPT: &str = "hello there";
const PROMPT_CHANGED_LATER: &str = "hello where";
// A tokenizer padding as sentence-transformers exports it, longer than any prompt here.
const SAVED_PADDING: usize = 48;
const SENTENCE_MAX_SEQ_LENGTH: usize = 8;
const TOKENS: [u32; 9] = [40, 41, 42, 43, 44, 45, 46, 47, 48];
// RoBERTa positions start at pad_token_id + 1, so 10 position rows hold 8 tokens.
const ROBERTA_PAD_TOKEN_ID: usize = 1;
const ROBERTA_POSITIONS: usize = 10;
const QUERY: &str = "hello";
const DOCUMENTS: [&str; 3] = ["hello there", "goodbye", "hello hello"];
const TOP_N: usize = 2;
const SCORE_TOLERANCE: f32 = 1e-5;

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

async fn embed_tokens(model: &Model, tokens: &[u32], truncate: bool) -> anyhow::Result<Vec<f32>> {
    let request = EmbeddingRequestBuilder::new()
        .add_tokens(tokens.to_vec())
        .with_truncate_sequence(truncate)
        .build()?;
    let response = serde_json::to_value(model.generate_embeddings(request).await?)?;
    Ok(serde_json::from_value(
        response["data"][0]["embedding"].clone(),
    )?)
}

#[tokio::test]
async fn an_encoder_class_is_detected_and_embeds_to_unit_length() -> anyhow::Result<()> {
    // no config_sentence_transformers.json: the BertModel class alone routes to the embedding loader
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let embedding = model.generate_embedding(PROMPT).await?;
    assert_eq!(embedding.len(), HIDDEN_SIZE);
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < NORM_TOLERANCE, "norm {norm}");
    Ok(())
}

#[tokio::test]
async fn the_first_token_attends_to_later_ones() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let first = model.generate_embedding(PROMPT).await?;
    let changed = model.generate_embedding(PROMPT_CHANGED_LATER).await?;
    let diff = max_diff(&first, &changed);
    assert!(
        diff > EMBEDDING_TOLERANCE,
        "a later token left the CLS embedding unchanged"
    );
    Ok(())
}

#[tokio::test]
async fn a_saved_tokenizer_padding_does_not_reach_the_prompt() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let expected = cpu_embedding_builder(checkpoint.path())
        .build()
        .await?
        .generate_embedding(PROMPT)
        .await?;

    let tokenizer_path = checkpoint.path().join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&tokenizer_path)?)?;
    tokenizer["padding"] = serde_json::json!({
        "strategy": {"Fixed": SAVED_PADDING},
        "direction": "Right",
        "pad_to_multiple_of": null,
        "pad_id": 0,
        "pad_type_id": 0,
        "pad_token": "<unk>"
    });
    std::fs::write(&tokenizer_path, serde_json::to_string(&tokenizer)?)?;
    let actual = cpu_embedding_builder(checkpoint.path())
        .build()
        .await?
        .generate_embedding(PROMPT)
        .await?;
    let diff = max_diff(&expected, &actual);
    assert!(
        diff < EMBEDDING_TOLERANCE,
        "the saved padding moved the embedding by {diff}"
    );
    Ok(())
}

#[tokio::test]
async fn sentence_transformers_max_seq_length_limits_the_prompt() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    std::fs::write(
        checkpoint.path().join("sentence_bert_config.json"),
        serde_json::to_string(&serde_json::json!({"max_seq_length": SENTENCE_MAX_SEQ_LENGTH}))?,
    )?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    assert!(
        embed_tokens(&model, &TOKENS, false).await.is_err(),
        "a prompt past max_seq_length was accepted"
    );
    let truncated = embed_tokens(&model, &TOKENS, true).await?;
    let head = embed_tokens(&model, &TOKENS[..SENTENCE_MAX_SEQ_LENGTH], false).await?;
    assert!(max_diff(&truncated, &head) < EMBEDDING_TOLERANCE);
    Ok(())
}

#[tokio::test]
async fn roberta_positions_leave_room_for_the_pad_offset() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|config| {
        config["architectures"] = serde_json::json!(["RobertaModel"]);
        config["model_type"] = "roberta".into();
        config["pad_token_id"] = ROBERTA_PAD_TOKEN_ID.into();
        config["max_position_embeddings"] = ROBERTA_POSITIONS.into();
    })?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let longest = ROBERTA_POSITIONS - ROBERTA_PAD_TOKEN_ID - 1;
    assert_eq!(
        embed_tokens(&model, &TOKENS[..longest], false).await?.len(),
        HIDDEN_SIZE
    );
    assert!(
        embed_tokens(&model, &TOKENS[..=longest], false)
            .await
            .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn an_isq_encoder_and_a_reload_of_its_uqff_embed_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let unquantized = cpu_embedding_builder(checkpoint.path())
        .build()
        .await?
        .generate_embedding(PROMPT)
        .await?;
    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_embedding_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let quantized_embedding = quantized.generate_embedding(PROMPT).await?;
    drop(quantized);
    let cosine: f32 = unquantized
        .iter()
        .zip(&quantized_embedding)
        .map(|(a, b)| a * b)
        .sum();
    assert!(cosine > ISQ_MIN_COSINE, "Q8_0 cosine {cosine}");

    let written = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(!written.is_empty(), "no UQFF written");
    let reloaded = UqffEmbeddingModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        vec![written[0].clone()],
    )
    .into_inner()
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;
    assert_eq!(
        reloaded.generate_embedding(PROMPT).await?,
        quantized_embedding
    );
    Ok(())
}

// A GGUF-type ISQ keeps its biases in F32; a BF16 CUDA matmul has to add them in its own dtype.
#[tokio::test]
async fn a_bf16_gpu_encoder_with_biased_isq_layers_embeds() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let model = EmbeddingModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_isq(IsqType::Q8_0)
        .build()
        .await?;
    assert_eq!(model.generate_embedding(PROMPT).await?.len(), HIDDEN_SIZE);
    Ok(())
}

fn classifier(labels: usize) -> impl FnOnce(&mut serde_json::Value) {
    move |config| {
        config["architectures"] = serde_json::json!(["BertForSequenceClassification"]);
        config["id2label"] = (0..labels)
            .map(|i| (i.to_string(), format!("LABEL_{i}").into()))
            .collect::<serde_json::Map<_, _>>()
            .into();
    }
}

async fn rerank(model: &Model, request: serde_json::Value) -> anyhow::Result<RerankResponse> {
    let request: RerankRequest = serde_json::from_value(request)?;
    Ok(model.rerank(request).await?)
}

fn raw_scores(response: &RerankResponse) -> Vec<f32> {
    let mut scores = vec![0f32; response.results.len()];
    for result in &response.results {
        scores[result.index] = result.relevance_score;
    }
    scores
}

#[tokio::test]
async fn a_cross_encoder_ranks_documents_most_relevant_first() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(classifier(1))?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let scored = rerank(
        &model,
        serde_json::json!({"query": QUERY, "documents": DOCUMENTS, "return_documents": true}),
    )
    .await?;
    let scores: Vec<f32> = scored.results.iter().map(|r| r.relevance_score).collect();
    assert!(
        scores.windows(2).all(|pair| pair[0] >= pair[1]),
        "{scores:?}"
    );
    assert!(scores.iter().all(|score| (0.0..=1.0).contains(score)));
    for result in &scored.results {
        let text = result.document.as_ref().map(|d| d.text.as_str());
        assert_eq!(text, Some(DOCUMENTS[result.index]));
    }

    let raw = rerank(
        &model,
        serde_json::json!({"query": QUERY, "documents": DOCUMENTS, "raw_scores": true}),
    )
    .await?;
    for (logit, result) in raw_scores(&raw).iter().zip(raw_scores(&scored)) {
        let sigmoid = 1.0 / (1.0 + (-logit).exp());
        assert!((sigmoid - result).abs() < SCORE_TOLERANCE);
    }

    let top = rerank(
        &model,
        serde_json::json!({"query": QUERY, "documents": DOCUMENTS, "top_n": TOP_N}),
    )
    .await?;
    assert_eq!(top.results.len(), TOP_N);
    assert_eq!(top.results[0].index, scored.results[0].index);
    Ok(())
}

#[tokio::test]
async fn a_pair_reaches_the_model_with_its_segment_ids() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(classifier(1))?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let tokenizer = tokenizers::Tokenizer::from_file(checkpoint.path().join("tokenizer.json"))
        .map_err(anyhow::Error::msg)?;
    let pair = tokenizer
        .encode((QUERY, DOCUMENTS[0]), true)
        .map_err(anyhow::Error::msg)?;
    assert!(
        pair.get_type_ids().contains(&1),
        "the pair has no second segment"
    );

    let reranked = rerank(
        &model,
        serde_json::json!({"query": QUERY, "documents": [DOCUMENTS[0]], "raw_scores": true}),
    )
    .await?;
    // the same ids as one segment
    let single_segment = embed_tokens(&model, pair.get_ids(), false).await?;
    let diff = (reranked.results[0].relevance_score - single_segment[0]).abs();
    assert!(
        diff > SCORE_TOLERANCE,
        "the segment ids left the score unchanged"
    );
    Ok(())
}

#[tokio::test]
async fn a_prefixed_cross_encoder_and_a_reload_of_its_uqff_score_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(classifier(1))?;
    prefix_encoder_weights(checkpoint.path(), "bert")?;
    let request = serde_json::json!({"query": QUERY, "documents": DOCUMENTS, "raw_scores": true});
    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_embedding_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let expected = raw_scores(&rerank(&quantized, request.clone()).await?);
    drop(quantized);

    let written = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(!written.is_empty(), "no UQFF written");
    let reloaded = UqffEmbeddingModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        vec![written[0].clone()],
    )
    .into_inner()
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;
    assert_eq!(raw_scores(&rerank(&reloaded, request).await?), expected);
    Ok(())
}

#[tokio::test]
async fn only_a_single_logit_classifier_reranks() -> anyhow::Result<()> {
    let request = serde_json::json!({"query": QUERY, "documents": DOCUMENTS});
    let embedder = tiny_bert_checkpoint(|_| {})?;
    let two_labels = tiny_bert_checkpoint(classifier(2))?;
    for checkpoint in [&embedder, &two_labels] {
        let model = cpu_embedding_builder(checkpoint.path()).build().await?;
        let error = rerank(&model, request.clone()).await.unwrap_err();
        assert!(error.to_string().contains("one logit"), "{error}");
    }
    Ok(())
}

fn jina_v2(config: &mut serde_json::Value) {
    config["architectures"] = serde_json::json!(["JinaBertModel"]);
    config["position_embedding_type"] = "alibi".into();
    config["feed_forward_type"] = "geglu".into();
}

// with no position table, a mean pool of a bidirectional encoder sees a bag of tokens; ALiBi makes order count
#[tokio::test]
async fn jina_v2_alibi_makes_token_order_count_and_reloads_from_uqff() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(jina_v2)?;
    let pooling = checkpoint.path().join("1_Pooling/config.json");
    let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&pooling)?)?;
    config["pooling_mode_cls_token"] = false.into();
    config["pooling_mode_mean_tokens"] = true.into();
    std::fs::write(&pooling, serde_json::to_string(&config)?)?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let forward = embed_tokens(&model, &TOKENS[..4], false).await?;
    // not a reversal: that keeps every |i - j|, which symmetric ALiBi cannot tell apart
    let swapped =
        embed_tokens(&model, &[TOKENS[1], TOKENS[0], TOKENS[2], TOKENS[3]], false).await?;
    assert!(
        max_diff(&forward, &swapped) > EMBEDDING_TOLERANCE,
        "swapping two tokens left the embedding unchanged"
    );
    drop(model);

    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_embedding_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let expected = quantized.generate_embedding(PROMPT).await?;
    drop(quantized);
    let written = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(!written.is_empty(), "no UQFF written");
    let reloaded = UqffEmbeddingModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        vec![written[0].clone()],
    )
    .into_inner()
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;
    assert_eq!(reloaded.generate_embedding(PROMPT).await?, expected);
    Ok(())
}

async fn embed_with_task(
    model: &Model,
    text: &str,
    task: Option<&str>,
) -> anyhow::Result<Vec<f32>> {
    let mut request = EmbeddingRequestBuilder::new().add_prompt(text);
    if let Some(task) = task {
        request = request.with_task(task);
    }
    let response = serde_json::to_value(model.generate_embeddings(request.build()?).await?)?;
    Ok(serde_json::from_value(
        response["data"][0]["embedding"].clone(),
    )?)
}

#[tokio::test]
async fn a_task_picks_its_adapter_and_prepends_its_instruction() -> anyhow::Result<()> {
    let checkpoint = tiny_jina_v3_checkpoint()?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let base = embed_with_task(&model, PROMPT, None).await?;
    // the second task has no instruction, so only its adapter moves the embedding
    let adapted = embed_with_task(&model, PROMPT, Some(JINA_V3_TASKS[1])).await?;
    assert!(
        max_diff(&base, &adapted) > EMBEDDING_TOLERANCE,
        "the adapter changed nothing"
    );

    // the first task's text request equals its tokens with the instruction already in them
    let tokenizer = tokenizers::Tokenizer::from_file(checkpoint.path().join("tokenizer.json"))
        .map_err(anyhow::Error::msg)?;
    let instructed = tokenizer
        .encode(format!("{JINA_V3_INSTRUCTION}{PROMPT}"), true)
        .map_err(anyhow::Error::msg)?;
    let request = EmbeddingRequestBuilder::new()
        .add_tokens(instructed.get_ids().to_vec())
        .with_task(JINA_V3_TASKS[0])
        .build()?;
    let response = serde_json::to_value(model.generate_embeddings(request).await?)?;
    let from_tokens: Vec<f32> = serde_json::from_value(response["data"][0]["embedding"].clone())?;
    let from_text = embed_with_task(&model, PROMPT, Some(JINA_V3_TASKS[0])).await?;
    assert!(max_diff(&from_text, &from_tokens) < EMBEDDING_TOLERANCE);
    Ok(())
}

#[tokio::test]
async fn requests_with_different_tasks_embed_as_they_do_alone() -> anyhow::Result<()> {
    let checkpoint = tiny_jina_v3_checkpoint()?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let tasks = [None, Some(JINA_V3_TASKS[0]), Some(JINA_V3_TASKS[1])];
    let mut alone = Vec::new();
    for task in tasks {
        alone.push(embed_with_task(&model, PROMPT, task).await?);
    }
    // the same three at once, which the scheduler may batch together
    let together = futures::future::try_join_all(
        tasks
            .iter()
            .map(|task| embed_with_task(&model, PROMPT, *task)),
    )
    .await?;
    for (a, t) in alone.iter().zip(&together) {
        assert!(max_diff(a, t) < EMBEDDING_TOLERANCE);
    }
    Ok(())
}

#[tokio::test]
async fn an_unknown_task_or_a_model_without_adapters_is_rejected() -> anyhow::Result<()> {
    let jina = tiny_jina_v3_checkpoint()?;
    let model = cpu_embedding_builder(jina.path()).build().await?;
    let error = embed_with_task(&model, PROMPT, Some("summarization"))
        .await
        .unwrap_err();
    assert!(error.to_string().contains(JINA_V3_TASKS[0]), "{error}");

    let bert = tiny_bert_checkpoint(|_| {})?;
    let model = cpu_embedding_builder(bert.path()).build().await?;
    let error = embed_with_task(&model, PROMPT, Some(JINA_V3_TASKS[0]))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("no task adapters"), "{error}");
    Ok(())
}
