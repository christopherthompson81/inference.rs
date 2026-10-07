//! Engine behavior on a tiny random-weight PaddleOCR-VL: the checkpoint is built at test time, so these run everywhere.

use std::path::{Path, PathBuf};

use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};
use inference_api::models::{ModelOperationRequest, ModelStatus};

#[path = "../support/paddleocr_vl_tiny.rs"]
mod support;
use support::{FIXTURES, tiny_checkpoint};

const OCR_PROMPT: &str = "OCR:";
// Long enough that a shared prefix runs past the image into whole paged blocks; paged hits never end inside an image.
const LONG_PROMPT: &str = "OCR: transcribe every line of this page exactly, keeping the original line breaks and punctuation in place.";
const TEXT_PROMPT: &str = "Reply with the single word: ok";
const MAX_LEN: usize = 8;
// A scheduler spin never completes either request, so the mixed-batch test fails on this instead of hanging.
const MIXED_BATCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
// Cached KV comes from a different prefill than a full recompute, so logprobs match only to rounding.
const LOGPROB_TOLERANCE: f32 = 1e-3;
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));

async fn build(dir: &Path) -> anyhow::Result<Model> {
    let mut builder =
        MultimodalModelBuilder::new(dir.to_string_lossy()).with_dtype(ModelDType::F32);
    if !ON_GPU {
        builder = builder.with_force_cpu();
    }
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        builder = builder.with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?);
    }
    // The server runs with the prefix cacher on.
    Ok(builder.with_prefix_cache_n(Some(16)).build().await?)
}

fn fixture(name: &str) -> anyhow::Result<image::DynamicImage> {
    Ok(image::open(PathBuf::from(FIXTURES).join(name))?)
}

fn image_request(names: &[&str]) -> anyhow::Result<RequestBuilder> {
    prompted_image_request(names, OCR_PROMPT)
}

fn prompted_image_request(names: &[&str], prompt: &str) -> anyhow::Result<RequestBuilder> {
    let images = names
        .iter()
        .map(|name| fixture(name))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(
        RequestBuilder::from(MultimodalMessages::new().add_image_message(
            TextMessageRole::User,
            prompt,
            images,
        ))
        .set_sampler_max_len(MAX_LEN)
        .set_sampler_topk(1)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(1),
    )
}

fn greedy_ids(resp: &inference::ChatCompletionResponse) -> Vec<u32> {
    resp.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| toks.iter().map(|t| t.top_logprobs[0].token).collect())
        .unwrap_or_default()
}

// Greedy ids of one image and of two, pinned so the text stack's numerics can't drift unnoticed; GPU runs paged flash.
#[tokio::test]
async fn image_decodes_are_pinned() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let one = model
        .send_chat_request(image_request(&["ocr.png"])?)
        .await?;
    let two = model
        .send_chat_request(image_request(&["ocr.png", "table.png"])?)
        .await?;
    let traces = (greedy_ids(&one), greedy_ids(&two));
    let expected: (Vec<u32>, Vec<u32>) = if ON_GPU {
        (
            vec![135, 219, 156, 129, 129, 129, 175, 129],
            vec![69, 119, 15, 86, 255, 8, 129, 129],
        )
    } else {
        (
            vec![172, 129, 129, 129, 129, 255, 8, 51],
            vec![69, 119, 15, 86, 255, 8, 129, 129],
        )
    };
    assert_eq!(traces, expected);
    Ok(())
}

#[tokio::test]
async fn mixed_text_and_image_batch_makes_progress() -> anyhow::Result<()> {
    let dir = tiny_checkpoint()?;
    let model = build(dir.path()).await?;
    let alone = greedy_ids(
        &model
            .send_chat_request(image_request(&["page_00.png"])?)
            .await?,
    );
    let image = image_request(&["page_00.png"])?;
    let (batched, text_only) = tokio::time::timeout(MIXED_BATCH_TIMEOUT, async {
        tokio::join!(
            model.send_chat_request(image),
            model.send_chat_request(
                RequestBuilder::new()
                    .add_message(TextMessageRole::User, TEXT_PROMPT)
                    .set_sampler_max_len(MAX_LEN)
            )
        )
    })
    .await?;
    text_only?;
    assert!(
        !alone.is_empty(),
        "the image request produced no tokens, so the comparison proves nothing"
    );
    assert_eq!(
        alone,
        greedy_ids(&batched?),
        "image output changed when a text-only request shared the batch"
    );
    Ok(())
}

fn trace(resp: &inference::ChatCompletionResponse) -> Vec<(u32, f32)> {
    resp.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| (t.top_logprobs[0].token, t.top_logprobs[0].logprob))
                .collect()
        })
        .unwrap_or_default()
}

fn cached_tokens(resp: &inference::ChatCompletionResponse) -> usize {
    resp.usage
        .prompt_tokens_details
        .as_ref()
        .map_or(0, |details| details.cached_tokens)
}

fn same_decode(a: &[(u32, f32)], b: &[(u32, f32)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < LOGPROB_TOLERANCE)
}

// Same-size pages give byte-identical prompts, so only the registered image span keeps their KV blocks apart.
#[tokio::test]
async fn prefix_cache_does_not_serve_one_image_for_another() -> anyhow::Result<()> {
    let dir = tiny_checkpoint()?;
    let fresh = build(dir.path()).await?;
    let fresh_page_01 = trace(
        &fresh
            .send_chat_request(prompted_image_request(&["page_01.png"], LONG_PROMPT)?)
            .await?,
    );
    let fresh_page_00 = trace(
        &fresh
            .send_chat_request(prompted_image_request(&["page_00.png"], LONG_PROMPT)?)
            .await?,
    );

    let model = build(dir.path()).await?;
    model
        .send_chat_request(prompted_image_request(&["page_00.png"], LONG_PROMPT)?)
        .await?;
    let page_01 = trace(
        &model
            .send_chat_request(prompted_image_request(&["page_01.png"], LONG_PROMPT)?)
            .await?,
    );
    assert!(!page_01.is_empty());
    assert!(
        same_decode(&page_01, &fresh_page_01),
        "page_01 was served page_00's cached blocks: {page_01:?}"
    );
    let resp = model
        .send_chat_request(prompted_image_request(&["page_00.png"], LONG_PROMPT)?)
        .await?;
    assert!(
        cached_tokens(&resp) > 0,
        "page_00 was not served from the prefix cache"
    );
    let cached = trace(&resp);
    assert!(
        same_decode(&cached, &fresh_page_00),
        "prefix cache reuse changed page_00: {fresh_page_00:?} vs {cached:?}"
    );
    Ok(())
}

// The second request reuses the shared first image's blocks and must still decode exactly as a fresh request.
#[tokio::test]
async fn partial_prefix_hit_matches_a_fresh_two_image_decode() -> anyhow::Result<()> {
    let dir = tiny_checkpoint()?;
    let both = ["page_00.png", "page_01.png"];
    let fresh = trace(
        &build(dir.path())
            .await?
            .send_chat_request(image_request(&both)?)
            .await?,
    );
    assert!(!fresh.is_empty());
    let warm = build(dir.path()).await?;
    warm.send_chat_request(image_request(&["page_00.png"])?)
        .await?;
    let resp = warm.send_chat_request(image_request(&both)?).await?;
    // Paged hits are whole blocks and the shared prefix ends at the first image's end token, so only the
    // token-granular non-paged cacher can serve half of this prompt.
    if !ON_GPU {
        assert!(
            cached_tokens(&resp) > 0,
            "the shared first image was not served from the prefix cache"
        );
    }
    let cached = trace(&resp);
    assert!(same_decode(&cached, &fresh), "{fresh:?} vs {cached:?}");
    Ok(())
}

// Reload rebuilds from the config the SDK stored at first load, so it must decode exactly as the first load did.
#[tokio::test]
async fn reload_decodes_like_the_first_load() -> anyhow::Result<()> {
    let dir = tiny_checkpoint()?;
    let model = build(dir.path()).await?;
    // Unload and reload take the real model id; the `default` alias is not registered for them.
    let models = model.models()?.data;
    let ids = models
        .iter()
        .filter(|m| m.default == Some(true))
        .map(|m| m.id.clone())
        .collect::<Vec<_>>();
    let [id] = ids.as_slice() else {
        anyhow::bail!("expected exactly one default model, got {models:?}");
    };
    let op = || ModelOperationRequest {
        model_id: id.clone(),
    };
    let first = trace(
        &model
            .send_chat_request(image_request(&["page_00.png"])?)
            .await?,
    );
    assert!(!first.is_empty());

    model.unload_model(op())?;
    assert_eq!(model.model_status(op())?.status, ModelStatus::Unloaded);
    model.reload_model(op()).await?;
    assert_eq!(model.model_status(op())?.status, ModelStatus::Loaded);

    let reloaded = trace(
        &model
            .send_chat_request(image_request(&["page_00.png"])?)
            .await?,
    );
    assert!(
        same_decode(&first, &reloaded),
        "reload decoded differently: {first:?} vs {reloaded:?}"
    );
    Ok(())
}
