//! Inkling 2 with an external DFlash 2 drafter on real checkpoints; skips unless `INFERENCE_TEST_INKLING2_MODEL` and
//! `INFERENCE_TEST_INKLING2_DFLASH2` are local dirs and a CUDA device is present.

use std::path::Path;

use inference::{
    MemoryGpuConfig, Model, ModelDType, RequestBuilder, TextMessageRole, TextMessages,
    TextModelBuilder,
};

const MODEL_ENV: &str = "INFERENCE_TEST_INKLING2_MODEL";
const DRAFTER_ENV: &str = "INFERENCE_TEST_INKLING2_DFLASH2";
const CONTEXT: usize = 8192;
const N_PREDICT: usize = 7;
const MAX_LEN: usize = 96;
const SEED: u64 = 1;
const PROMPT: &str = "Count from 1 to 60, separated by commas.";
// a working drafter lands most of a count; drafts it cannot read (an embedding the target never makes) land none
const MIN_GREEDY_ACCEPT_RATE: f64 = 0.5;
const MIN_SAMPLED_ACCEPT_RATE: f64 = 0.3;

fn local(env: &str) -> Option<String> {
    std::env::var(env)
        .ok()
        .filter(|dir| Path::new(dir).exists())
}

fn request(temperature: f64) -> RequestBuilder {
    RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT))
        .enable_thinking(false)
        .set_sampler_temperature(temperature)
        .set_sampler_seed(SEED)
        .set_sampler_max_len(MAX_LEN)
}

fn totals(model: &Model) -> anyhow::Result<(usize, usize)> {
    let stats = model.speculative_stats()?;
    Ok(stats.data.iter().fold((0, 0), |(p, a), m| {
        (p + m.draft_tokens_proposed, a + m.draft_tokens_accepted)
    }))
}

// what one request's drafts landed, from the counters' change over it
async fn accept_rate(model: &Model, temperature: f64) -> anyhow::Result<f64> {
    let (proposed, accepted) = totals(model)?;
    model.send_chat_request(request(temperature)).await?;
    let (proposed_after, accepted_after) = totals(model)?;
    let (proposed, accepted) = (proposed_after - proposed, accepted_after - accepted);
    anyhow::ensure!(proposed > 0, "the drafter never drafted");
    Ok(accepted as f64 / proposed as f64)
}

// the drafts are read through Inkling's normed embedding on every path (a captured draft once took the raw table and
// drafted noise), and sampled drafts stay inside the target's unpadded vocabulary
#[tokio::test]
async fn dflash2_drafts_land_greedy_and_sampled_on_cuda() -> anyhow::Result<()> {
    let (Some(dir), Some(drafter)) = (local(MODEL_ENV), local(DRAFTER_ENV)) else {
        eprintln!("SKIP: {MODEL_ENV} and {DRAFTER_ENV} are not both local checkpoint dirs");
        return Ok(());
    };
    if !cfg!(feature = "cuda") || inference_tensor::Device::new_cuda(0).is_err() {
        eprintln!("SKIP: drafting needs a CUDA device");
        return Ok(());
    }
    let paged = inference::PagedAttentionMetaBuilder::default()
        .with_gpu_memory(MemoryGpuConfig::ContextSize(CONTEXT))
        .build()?;
    let model = TextModelBuilder::new(dir)
        .with_dtype(ModelDType::BF16)
        .with_prefix_cache_n(None)
        .with_paged_attn(paged)
        .with_mtp_model(drafter, Some(N_PREDICT))
        .build()
        .await?;
    let greedy = accept_rate(&model, 0.0).await?;
    let sampled = accept_rate(&model, 1.0).await?;
    eprintln!("accepted a draft token: greedy {greedy:.2}, sampled {sampled:.2}");
    anyhow::ensure!(
        greedy >= MIN_GREEDY_ACCEPT_RATE,
        "greedy drafts landed {greedy:.2}"
    );
    anyhow::ensure!(
        sampled >= MIN_SAMPLED_ACCEPT_RATE,
        "sampled drafts landed {sampled:.2}"
    );
    Ok(())
}
