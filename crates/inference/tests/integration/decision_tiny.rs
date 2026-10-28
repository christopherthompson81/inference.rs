//! A tiny random-weight decision model in laya's layout (an encoder folder, a tokenizer folder, the decision config
//! beside them): detection by its files, each question type's answer, the temperatures, and request validation.

use std::collections::HashMap;
use std::path::Path;

use inference::{DecisionAnswer, DecisionRequest, EmbeddingModelBuilder, Model, ModelDType};
use inference_tensor::{Device, Tensor};

#[path = "../support/rope_encoders_tiny.rs"]
mod rope;

const HIDDEN: usize = 32;
const HEAD_LAYERS: usize = 1;
// the fixture's position table
const MAX_LEN: usize = 64;
const HEAD_MAX_LEN: usize = 32;
// specials the shared tiny tokenizer has, standing in for [CLS] / [SEP] / [MASK]
const CLS: &str = "<s>";
const SEP: &str = "</s>";
const MASK: &str = "<|IMAGE_PLACEHOLDER|>";
const WEIGHT_STD: f32 = 0.5;
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;
// a batched pass sums in another order than a lone one; on a GPU that moves an answer by about 1e-6
const BATCH_TOLERANCE: f64 = 1e-5;
// a choice of three falls in laya's `choice:3-5` bucket, which takes precedence over the per-type temperature
const CHOICE_3_5_TEMPERATURE: f64 = 2.5;

fn random(shape: &[usize]) -> anyhow::Result<Tensor> {
    Ok(Tensor::randn(0f32, WEIGHT_STD, shape, &Device::Cpu)?)
}

/// The decision head's weights under torch's names: one pre-norm block, the type embedding and the scorer.
fn head_weights() -> anyhow::Result<HashMap<String, Tensor>> {
    let mut weights = HashMap::new();
    let mut put = |name: &str, shape: &[usize]| -> anyhow::Result<()> {
        weights.insert(name.to_string(), random(shape)?);
        Ok(())
    };
    for i in 0..HEAD_LAYERS {
        let block = format!("head.layers.{i}");
        put(
            &format!("{block}.self_attn.in_proj_weight"),
            &[3 * HIDDEN, HIDDEN],
        )?;
        put(&format!("{block}.self_attn.in_proj_bias"), &[3 * HIDDEN])?;
        put(
            &format!("{block}.self_attn.out_proj.weight"),
            &[HIDDEN, HIDDEN],
        )?;
        put(&format!("{block}.self_attn.out_proj.bias"), &[HIDDEN])?;
        put(&format!("{block}.linear1.weight"), &[4 * HIDDEN, HIDDEN])?;
        put(&format!("{block}.linear1.bias"), &[4 * HIDDEN])?;
        put(&format!("{block}.linear2.weight"), &[HIDDEN, 4 * HIDDEN])?;
        put(&format!("{block}.linear2.bias"), &[HIDDEN])?;
        for norm in ["norm1", "norm2"] {
            put(&format!("{block}.{norm}.weight"), &[HIDDEN])?;
            put(&format!("{block}.{norm}.bias"), &[HIDDEN])?;
        }
    }
    put("type_emb.weight", &[3, HIDDEN])?;
    put("scorer.0.weight", &[HIDDEN])?;
    put("scorer.0.bias", &[HIDDEN])?;
    put("scorer.1.weight", &[HIDDEN, HIDDEN])?;
    put("scorer.1.bias", &[HIDDEN])?;
    put("scorer.3.weight", &[1, HIDDEN])?;
    put("scorer.3.bias", &[1])?;
    Ok(weights)
}

/// A laya-layout checkpoint: the tiny ModernBERT under `encoder.`, a random head, and `temperatures` as laya's
/// `rl_agent_config.json` writes them.
fn tiny_decision_checkpoint(
    temperatures: serde_json::Value,
) -> anyhow::Result<(tempfile::TempDir, tempfile::TempDir)> {
    let encoder = rope::tiny_modern_bert_checkpoint(|_| {})?;
    let dir = tempfile::tempdir()?;
    let mut weights = inference_tensor::safetensors::load(
        encoder.path().join("model.safetensors"),
        &Device::Cpu,
    )?
    .into_iter()
    .map(|(name, tensor)| (format!("encoder.{name}"), tensor))
    .collect::<HashMap<_, _>>();
    weights.extend(head_weights()?);
    inference_tensor::safetensors::save(&weights, dir.path().join("model.safetensors"))?;
    std::fs::create_dir(dir.path().join("encoder"))?;
    std::fs::copy(
        encoder.path().join("config.json"),
        dir.path().join("encoder/config.json"),
    )?;
    std::fs::create_dir(dir.path().join("tokenizer"))?;
    std::fs::copy(
        encoder.path().join("tokenizer.json"),
        dir.path().join("tokenizer/tokenizer.json"),
    )?;
    std::fs::write(
        dir.path().join("tokenizer/tokenizer_config.json"),
        serde_json::json!({"cls_token": CLS, "sep_token": SEP, "mask_token": MASK}).to_string(),
    )?;
    let mut config = serde_json::json!({
        "head_layers": HEAD_LAYERS,
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
    });
    config
        .as_object_mut()
        .expect("an object")
        .extend(temperatures.as_object().cloned().unwrap_or_default());
    std::fs::write(dir.path().join("rl_agent_config.json"), config.to_string())?;
    Ok((dir, encoder))
}

async fn model(dir: &Path) -> anyhow::Result<Model> {
    Ok(EmbeddingModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?)
}

fn request(questions: serde_json::Value) -> anyhow::Result<DecisionRequest> {
    Ok(serde_json::from_value(serde_json::json!({
        "state": {"ticket": "the app closes on upload", "plan": "pro"},
        "questions": questions,
    }))?)
}

fn questions() -> serde_json::Value {
    serde_json::json!({
        "route": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"billing": null, "shipping": "parcels", "technical": null}},
        "angry": {"type": "noul", "instructions": "Is the writer angry?"},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["can wait", "this week", "today", "now"]},
    })
}

fn choice_probabilities(answer: &DecisionAnswer) -> Vec<f64> {
    match answer {
        DecisionAnswer::Choice { probabilities, .. } => probabilities.values().copied().collect(),
        other => panic!("not a choice answer: {other:?}"),
    }
}

#[tokio::test]
async fn a_decision_checkpoint_answers_each_question_type() -> anyhow::Result<()> {
    let (dir, _encoder) = tiny_decision_checkpoint(serde_json::json!({}))?;
    let model = model(dir.path()).await?;
    let response = model.decide(request(questions())?).await?;
    let ids = response
        .answers
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        ["route", "angry", "urgency"],
        "answers keep the request's order"
    );
    match &response.answers["route"] {
        DecisionAnswer::Choice {
            choice,
            probabilities,
            confidence,
        } => {
            let keys = probabilities.keys().map(String::as_str).collect::<Vec<_>>();
            assert_eq!(keys, ["billing", "shipping", "technical"]);
            let best = probabilities
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(key, _)| key.clone());
            assert_eq!(Some(choice.clone()), best);
            assert!((probabilities.values().sum::<f64>() - 1.0).abs() < PROBABILITY_SUM_TOLERANCE);
            assert!((0.0..=1.0).contains(confidence));
        }
        other => panic!("route: {other:?}"),
    }
    match &response.answers["angry"] {
        DecisionAnswer::Noul { noul } => assert!((0.0..=1.0).contains(noul)),
        other => panic!("angry: {other:?}"),
    }
    match &response.answers["urgency"] {
        DecisionAnswer::Score {
            score,
            legend,
            probabilities,
            ..
        } => {
            assert_eq!(legend["3"], "now");
            let expected = probabilities
                .iter()
                .map(|(level, p)| level.parse::<f64>().unwrap() * p)
                .sum::<f64>();
            assert!((score - expected).abs() < PROBABILITY_SUM_TOLERANCE);
        }
        other => panic!("urgency: {other:?}"),
    }
    assert!(response.usage.input_tokens > 0 && response.usage.output_tokens == 0);
    Ok(())
}

// the same weights with a `choice:3-5` temperature: each probability is the uncalibrated one raised to 1 / T,
// renormalized
#[tokio::test]
async fn a_choice_takes_its_option_count_temperature() -> anyhow::Result<()> {
    let (dir, _encoder) = tiny_decision_checkpoint(serde_json::json!({}))?;
    let plain = choice_probabilities(
        &model(dir.path())
            .await?
            .decide(request(questions())?)
            .await?
            .answers["route"],
    );
    std::fs::write(
        dir.path().join("rl_agent_config.json"),
        serde_json::json!({
            "head_layers": HEAD_LAYERS, "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN,
            "temperature": [9.0, 9.0, 9.0],
            "temperature_by_options": {"choice:3-5": CHOICE_3_5_TEMPERATURE},
        })
        .to_string(),
    )?;
    let calibrated = choice_probabilities(
        &model(dir.path())
            .await?
            .decide(request(questions())?)
            .await?
            .answers["route"],
    );
    let scaled = plain
        .iter()
        .map(|p| p.powf(1.0 / CHOICE_3_5_TEMPERATURE))
        .collect::<Vec<_>>();
    let total = scaled.iter().sum::<f64>();
    for (got, want) in calibrated.iter().zip(scaled.iter().map(|p| p / total)) {
        assert!((got - want).abs() < 1e-5, "{calibrated:?} against {want}");
    }
    Ok(())
}

// a batch's f32 matmuls sum in another order than one row's
fn same_answer(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            (a.as_f64().unwrap_or(f64::NAN) - b.as_f64().unwrap_or(f64::NAN)).abs()
                < BATCH_TOLERANCE
        }
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|((ka, va), (kb, vb))| ka == kb && same_answer(va, vb))
        }
        _ => a == b,
    }
}

// questions batched together answer as they do alone
#[tokio::test]
async fn each_question_answers_as_it_does_alone() -> anyhow::Result<()> {
    let (dir, _encoder) = tiny_decision_checkpoint(serde_json::json!({}))?;
    let model = model(dir.path()).await?;
    let together = model.decide(request(questions())?).await?;
    for (id, question) in questions().as_object().expect("an object") {
        let alone = model
            .decide(request(
                serde_json::json!({ id.clone(): question.clone() }),
            )?)
            .await?;
        let (alone, together) = (
            serde_json::to_value(&alone.answers[id])?,
            serde_json::to_value(&together.answers[id])?,
        );
        assert!(
            same_answer(&alone, &together),
            "{id}: {alone} vs {together}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn malformed_questions_and_non_decision_models_are_refused() -> anyhow::Result<()> {
    let (dir, encoder) = tiny_decision_checkpoint(serde_json::json!({}))?;
    let model = model(dir.path()).await?;
    for (questions, expected) in [
        (
            serde_json::json!({"q": {"type": "choice", "instructions": "x", "criteria": {"only": null}}}),
            "2 to 255 options",
        ),
        (
            serde_json::json!({"q": {"type": "score", "instructions": "x", "criteria": ["a"]}}),
            "2 to 10 levels",
        ),
        (
            serde_json::json!({"q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}}),
            "not `maybe`",
        ),
        (serde_json::json!({}), "at least one question"),
    ] {
        let error = model.decide(request(questions)?).await.unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    // the same encoder loaded as an embedder answers no questions
    let embedder = self::model(encoder.path()).await?;
    let error = embedder.decide(request(questions())?).await.unwrap_err();
    assert!(
        error.to_string().contains("not a decision model"),
        "{error}"
    );
    Ok(())
}
