//! A tiny random-weight Llama with a tiny random EAGLE-3 drafter (SpecForge layout, a reduced vocabulary mapped through
//! `d2t`): drafting from three of the target's layers leaves greedy decoding as it is without the drafter.

use inference::{ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder};

#[path = "../support/llama_tiny.rs"]
mod support;
#[path = "../support/tiny_eagle3.rs"]
mod tiny_eagle3;
#[path = "../support/traces.rs"]
mod traces;

// the committed tiny Llama's widths, with layers enough for three distinct taps
const LAYERS: usize = 4;
const HIDDEN: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
const INTERMEDIATE: usize = 64;
const VOCAB: usize = 266;
const DRAFT_VOCAB: usize = 64;
const TAP_LAYER_IDS: [usize; 3] = [1, 2, 4];
const N_PREDICT: usize = 2;
const MAX_LEN: usize = 16;
// the verify step's rows run batched where plain decode runs them one by one
const LOGPROB_TOLERANCE: f32 = 0.05;
const SHAPE: tiny_eagle3::Eagle3Shape = tiny_eagle3::Eagle3Shape {
    hidden: HIDDEN,
    heads: HEADS,
    kv_heads: KV_HEADS,
    head_dim: HEAD_DIM,
    intermediate: INTERMEDIATE,
    vocab: VOCAB,
    draft_vocab: DRAFT_VOCAB,
    tap_layer_ids: TAP_LAYER_IDS,
    eagle31: false,
};
const PROMPT: &str =
    "an EAGLE-3 drafter reads three of the target's layers and drafts in its own small vocabulary";

// Greedy decoding with the drafter matches decoding without it, and the drafter drafted.
#[tokio::test]
async fn an_eagle3_drafter_drafts_without_changing_greedy_output() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = support::tiny_llama_checkpoint_with(serde_json::json!({
        "num_hidden_layers": LAYERS
    }))?;
    let drafter = tiny_eagle3::tiny_eagle3(&SHAPE)?;
    let builder = || -> anyhow::Result<TextModelBuilder> {
        Ok(TextModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::BF16)
            .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?))
    };
    let request =
        || RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT));
    let drafting = builder()?
        .with_mtp_model(drafter.path().to_string_lossy(), Some(N_PREDICT))
        .build()
        .await?;
    let (actual, _) = traces::greedy(&drafting, request(), MAX_LEN).await?;
    let drafts: usize = drafting
        .speculative_stats()?
        .data
        .iter()
        .map(|m| m.drafts)
        .sum();
    drop(drafting);
    let plain = builder()?.build().await?;
    let (expected, _) = traces::greedy(&plain, request(), MAX_LEN).await?;
    anyhow::ensure!(drafts > 0, "the EAGLE-3 drafter never drafted");
    anyhow::ensure!(
        traces::close(&expected, &actual, LOGPROB_TOLERANCE),
        "drafting changed greedy decoding: {expected:?} vs {actual:?}"
    );
    Ok(())
}

// llama.cpp's GGUF of the same drafter (llama.cpp's names, permuted q/k, target-id `d2t`) loads, drafts and leaves
// greedy decoding as the checkpoint does; how many steps draft follows scheduling, so only that both do is checked
#[tokio::test]
async fn a_gguf_drafter_drafts_as_its_checkpoint_does() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = support::tiny_llama_checkpoint_with(serde_json::json!({
        "num_hidden_layers": LAYERS
    }))?;
    let drafter = tiny_eagle3::tiny_eagle3(&SHAPE)?;
    let gguf = tiny_eagle3::tiny_eagle3_gguf(&SHAPE, drafter.path())?;
    let run = |mtp: String| {
        let checkpoint = checkpoint.path().to_string_lossy().into_owned();
        async move {
            let model = TextModelBuilder::new(checkpoint)
                .with_dtype(ModelDType::BF16)
                .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
                .with_mtp_model(mtp, Some(N_PREDICT))
                .build()
                .await?;
            let request = RequestBuilder::from(
                TextMessages::new().add_message(TextMessageRole::User, PROMPT),
            );
            let (trace, _) = traces::greedy(&model, request, MAX_LEN).await?;
            let drafts: usize = model
                .speculative_stats()?
                .data
                .iter()
                .map(|m| m.drafts)
                .sum();
            anyhow::Ok((trace, drafts))
        }
    };
    let (expected, drafts) = run(drafter.path().to_string_lossy().into_owned()).await?;
    let (actual, gguf_drafts) = run(gguf.path().to_string_lossy().into_owned()).await?;
    anyhow::ensure!(
        drafts > 0 && gguf_drafts > 0,
        "the GGUF drafter drafted {gguf_drafts} times, the checkpoint {drafts}"
    );
    anyhow::ensure!(
        traces::close(&expected, &actual, LOGPROB_TOLERANCE),
        "the GGUF drafter changed decoding: {expected:?} vs {actual:?}"
    );
    Ok(())
}
