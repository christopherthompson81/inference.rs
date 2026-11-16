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
    let drafter = tiny_eagle3::tiny_eagle3(&tiny_eagle3::Eagle3Shape {
        hidden: HIDDEN,
        heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        intermediate: INTERMEDIATE,
        vocab: VOCAB,
        draft_vocab: DRAFT_VOCAB,
        tap_layer_ids: TAP_LAYER_IDS,
        eagle31: false,
    })?;
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
