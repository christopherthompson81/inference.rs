//! Built-in MTP drafting on a real Qwen3.5 checkpoint; skips unless `INFERENCE_TEST_QWEN3_5_MODEL` is a local dir and
//! a GPU is present (the MTP head reads and writes paged KV).

use std::path::Path;

use inference::{Model, ModelDType, MultimodalModelBuilder, RequestBuilder, TextMessageRole};

const MODEL_ENV: &str = "INFERENCE_TEST_QWEN3_5_MODEL";
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));
const MAX_LEN: usize = 64;
const N_PREDICT: usize = 2;
// Plain prose a trained head predicts well; the reasoning preamble is off so the reply starts at once.
const PROMPTS: &[&str] = &[
    "Count from one to twenty in words, separated by commas.",
    "Write the first four lines of a nursery rhyme about a star.",
];
// Real heads land most of their drafts on prose this predictable; a broken head or verifier lands next to none.
const MIN_ACCEPT_RATE: f64 = 0.3;

fn model_dir() -> Option<String> {
    std::env::var(MODEL_ENV)
        .ok()
        .filter(|d| Path::new(d).exists())
}

async fn build(dir: &str, mtp: bool) -> anyhow::Result<Model> {
    let builder = MultimodalModelBuilder::new(dir)
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?);
    let builder = if mtp {
        builder.with_builtin_mtp(Some(N_PREDICT))
    } else {
        builder
    };
    Ok(builder.build().await?)
}

async fn greedy_ids(model: &Model, prompt: &str) -> anyhow::Result<Vec<u32>> {
    let request = RequestBuilder::new()
        .add_message(TextMessageRole::User, prompt)
        .enable_thinking(false)
        .set_sampler_max_len(MAX_LEN)
        .set_sampler_topk(1)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(1);
    let response = model.send_chat_request(request).await?;
    let ids = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| toks.iter().map(|t| t.top_logprobs[0].token).collect())
        .unwrap_or_default();
    Ok(ids)
}

#[tokio::test]
async fn builtin_mtp_accepts_drafts_and_keeps_greedy_output() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: {MODEL_ENV} is not a local checkpoint dir");
        return Ok(());
    };
    if !ON_GPU {
        eprintln!("SKIP: built-in MTP needs a GPU build");
        return Ok(());
    }
    let plain = build(&dir, false).await?;
    let mtp = build(&dir, true).await?;
    for prompt in PROMPTS {
        let expected = greedy_ids(&plain, prompt).await?;
        let drafted = greedy_ids(&mtp, prompt).await?;
        let agreed = expected
            .iter()
            .zip(&drafted)
            .take_while(|(e, d)| e == d)
            .count();
        eprintln!("{prompt:?}: {agreed} of {} ids agree", expected.len());
        anyhow::ensure!(
            !expected.is_empty(),
            "the model generated nothing for {prompt:?}"
        );
        anyhow::ensure!(
            agreed == expected.len() && drafted.len() == expected.len(),
            "MTP drafting changed the greedy output of {prompt:?} at step {agreed}: {drafted:?} vs {expected:?}"
        );
    }
    let stats = mtp.speculative_stats()?;
    let (proposed, accepted) = stats.data.iter().fold((0, 0), |(p, a), m| {
        (p + m.draft_tokens_proposed, a + m.draft_tokens_accepted)
    });
    let rate = accepted as f64 / proposed.max(1) as f64;
    eprintln!("MTP accepted {accepted} of {proposed} draft tokens ({rate:.2}): {stats:?}");
    anyhow::ensure!(
        rate >= MIN_ACCEPT_RATE,
        "MTP accepted {accepted} of {proposed} draft tokens"
    );
    Ok(())
}
