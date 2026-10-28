//! A tiny random-weight chat model as a GGUF with a llama.cpp decision profile: each question renders through the
//! `systemone` template, and its probabilities are the softmax of the label tokens' logits at the prompt's end.

use inference::{
    DecisionAnswer, DecisionRequest, GgufModelBuilder, LogitsOutput, Model, ModelDType,
    PromptInput, PromptLogitsRequest,
};
use inference_api::operations::TokenizeRequest;
use inference_tensor::quantized::gguf_file::Value;

#[path = "../support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;
#[path = "../support/llama_tiny_gguf.rs"]
mod tiny_gguf;
use tiny_gguf::write_tiny_gguf;

const GGUF_FILE: &str = "decision.gguf";
const TEMPLATE: &str = "{{ state if state is string else state | tojson }}\n{{ instructions }}\
{% for o in options %}\n{{ o.label }}: {{ o.key }}{% endfor %}\nCode:";
const CHOICE_TEMPERATURE: f32 = 2.0;
const STATE: &str = "the app closes on upload";
const PROBABILITY_TOLERANCE: f64 = 1e-5;

async fn model(family: &str) -> anyhow::Result<(Model, tempfile::TempDir)> {
    let checkpoint = tiny_llama_checkpoint()?;
    let dir = tempfile::tempdir()?;
    write_tiny_gguf(
        checkpoint.path(),
        &dir.path().join(GGUF_FILE),
        &[
            ("llama.decision.type", Value::String(family.into())),
            (
                "llama.decision.temperature.choice",
                Value::F32(CHOICE_TEMPERATURE),
            ),
            (
                "tokenizer.chat_template.systemone",
                Value::String(TEMPLATE.into()),
            ),
        ],
    )?;
    let model = GgufModelBuilder::new(dir.path().to_string_lossy(), vec![GGUF_FILE])
        .with_tok_model_id(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    Ok((model, checkpoint))
}

async fn tokens(model: &Model, text: &str) -> anyhow::Result<Vec<u32>> {
    Ok(model
        .tokenize(TokenizeRequest {
            model: None,
            text: text.to_string(),
            add_special_tokens: false,
        })
        .await?
        .tokens)
}

/// The softmax at `temperature` of the label logits at the end of `prompt`, labels `A`, `B`, ... in order.
async fn label_softmax(
    model: &Model,
    prompt: &str,
    n: usize,
    temperature: f64,
) -> anyhow::Result<Vec<f64>> {
    let labels = ('A'..)
        .take(n)
        .map(|c| vec![c.to_string()])
        .collect::<Vec<_>>();
    group_softmax(model, prompt, &labels, temperature).await
}

/// As [`label_softmax`], each output scored by the best logit of its spellings that are one token.
async fn group_softmax(
    model: &Model,
    prompt: &str,
    groups: &[Vec<String>],
    temperature: f64,
) -> anyhow::Result<Vec<f64>> {
    let ids = tokens(model, prompt).await?;
    let scored = model
        .prompt_logits(PromptLogitsRequest {
            model: None,
            prompt: PromptInput::Tokens(ids.clone()),
            output: LogitsOutput::Logits,
        })
        .await?;
    let logits = scored.logits.expect("the request asked for logits");
    let last = &logits[(ids.len() - 1) * scored.vocab_size..];
    let mut scores = Vec::with_capacity(groups.len());
    for group in groups {
        let mut best = f64::NEG_INFINITY;
        for form in group {
            if let [id] = tokens(model, form).await?[..] {
                best = best.max(f64::from(last[id as usize]));
            }
        }
        anyhow::ensure!(best.is_finite(), "no spelling of {group:?} is one token");
        scores.push(best / temperature);
    }
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
    let total = exp.iter().sum::<f64>();
    Ok(exp.iter().map(|e| e / total).collect())
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert!(
        got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(g, w)| (g - w).abs() < PROBABILITY_TOLERANCE),
        "{what}: {got:?} vs {want:?}"
    );
}

#[tokio::test]
async fn a_gguf_decision_profile_answers_by_label_logits() -> anyhow::Result<()> {
    let (model, _checkpoint) = model("nimble").await?;
    let request: DecisionRequest = serde_json::from_value(serde_json::json!({
        "state": STATE,
        "questions": {
            "route": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": null, "shipping": "parcels", "technical": null}},
            "angry": {"type": "noul", "instructions": "Is the writer angry?"},
        },
    }))?;
    let response = model.decide(request).await?;

    let route = format!("{STATE}\nWhich team?\nA: billing\nB: shipping\nC: technical\nCode:");
    let want = label_softmax(&model, &route, 3, f64::from(CHOICE_TEMPERATURE)).await?;
    match &response.answers["route"] {
        DecisionAnswer::Choice { probabilities, .. } => {
            let keys = probabilities.keys().map(String::as_str).collect::<Vec<_>>();
            assert_eq!(keys, ["billing", "shipping", "technical"]);
            assert_close(
                &probabilities.values().copied().collect::<Vec<_>>(),
                &want,
                "route",
            );
        }
        other => panic!("route: {other:?}"),
    }
    // nimble lists noul's options as false, true, with no temperature for noul
    let angry = format!("{STATE}\nIs the writer angry?\nA: false\nB: true\nCode:");
    let want = label_softmax(&model, &angry, 2, 1.0).await?;
    match &response.answers["angry"] {
        DecisionAnswer::Noul { noul } => assert_close(&[*noul], &want[1..], "angry"),
        other => panic!("angry: {other:?}"),
    }
    Ok(())
}

// lfm2-d1 picks its labels per question: a choice's single-letter keys are its codes, a score reads its level digits
#[tokio::test]
async fn per_question_labels_score_by_their_best_spelling() -> anyhow::Result<()> {
    let (model, _checkpoint) = model("lfm2-d1").await?;
    let request: DecisionRequest = serde_json::from_value(serde_json::json!({
        "state": STATE,
        "questions": {
            "pick": {"type": "choice", "instructions": "Which?", "criteria": {"x": null, "y": null}},
            "level": {"type": "score", "instructions": "How bad?", "criteria": ["fine", "bad"]},
        },
    }))?;
    let response = model.decide(request).await?;
    let spelled = |code: &str| vec![code.to_string(), format!(" {code}")];
    let pick = format!("{STATE}\nWhich?\nx: x\ny: y\nCode:");
    let want = group_softmax(
        &model,
        &pick,
        &[spelled("x"), spelled("y")],
        f64::from(CHOICE_TEMPERATURE),
    )
    .await?;
    match &response.answers["pick"] {
        DecisionAnswer::Choice { probabilities, .. } => assert_close(
            &probabilities.values().copied().collect::<Vec<_>>(),
            &want,
            "pick",
        ),
        other => panic!("pick: {other:?}"),
    }
    let level = format!("{STATE}\nHow bad?\n0: 0\n1: 1\nCode:");
    let digits = [vec!["0".to_string()], vec!["1".to_string()]];
    let want = group_softmax(&model, &level, &digits, 1.0).await?;
    match &response.answers["level"] {
        DecisionAnswer::Score { probabilities, .. } => assert_close(
            &probabilities.values().copied().collect::<Vec<_>>(),
            &want,
            "level",
        ),
        other => panic!("level: {other:?}"),
    }
    // a noul reads yes / no, which this vocabulary has no single tokens for
    let request: DecisionRequest = serde_json::from_value(serde_json::json!({
        "state": STATE,
        "questions": {"angry": {"type": "noul", "instructions": "Angry?"}},
    }))?;
    let err = model.decide(request).await.unwrap_err().to_string();
    assert!(err.contains("not a single token"), "{err}");
    Ok(())
}
