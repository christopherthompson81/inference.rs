//! PaddleOCR-VL end-to-end tests; they skip unless `INFERENCE_TEST_PADDLEOCR_VL_MODEL` is a local checkpoint dir.

use std::path::Path;

use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};

const MODEL_ENV: &str = "INFERENCE_TEST_PADDLEOCR_VL_MODEL";
// Synthetic images from make_fixtures.py; goldens from make_goldens.py on 1.6 (same in f32 and bf16).
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/paddleocr_vl");
const OCR_PROMPT: &str = "OCR:";
const TABLE_PROMPT: &str = "Table Recognition:";
const EOS: u32 = 2;
const MAX_LEN: usize = 64;

const OCR_GOLDEN: &[u32] = &[700, 6105, 19836, 67586, 56692, 93919, 7, 5, 3655, 93937, 2];
// A bare-string text turn has to reach the typed-content template, or this decodes as byte-fallback garbage.
const TEXT_ONLY_PROMPT: &str = "Reply with the single word: ok";
const TEXT_ONLY_GOLDEN: &[u32] = &[715, 275, 318, 290, 93919, 5, 3, 364, 315, 6644, 93937, 2];
const TEXT_GOLDENS: &[(&str, &str, &str)] = &[
    ("ocr.png", OCR_PROMPT, "The quick brown fox jumps 42 times."),
    (
        "page_00.png",
        OCR_PROMPT,
        "Weekly garden notes\nTomatoes are ripening early.\nWater the beans twice.",
    ),
    (
        "page_01.png",
        OCR_PROMPT,
        "Library opening hours\nMonday to Friday: 9 to 5.\nClosed on public holidays.",
    ),
    // OTSL cell tokens are non-special added tokens; detok must keep them or tables collapse into run-on text.
    (
        "table.png",
        TABLE_PROMPT,
        "<fcel>Fruit<fcel>Colour<fcel>Count<nl><fcel>Apple<fcel>Red<fcel>12<nl><fcel>Lemon<fcel>Yellow<fcel>7<nl>",
    ),
];

fn golden(fixture_name: &str) -> (&'static str, &'static str) {
    TEXT_GOLDENS
        .iter()
        .find(|(name, _, _)| *name == fixture_name)
        .map(|&(_, prompt, golden)| (prompt, golden))
        .expect("fixture has a golden")
}

fn model_dir() -> Option<String> {
    std::env::var(MODEL_ENV)
        .ok()
        .filter(|d| Path::new(d).exists())
}

fn fixture(name: &str) -> anyhow::Result<image::DynamicImage> {
    Ok(image::open(format!("{FIXTURES}/{name}"))?)
}

fn strip_eos(ids: &[u32]) -> &[u32] {
    match ids.last() {
        Some(&EOS) => &ids[..ids.len() - 1],
        _ => ids,
    }
}

fn greedy_ids(resp: &inference::ChatCompletionResponse) -> Vec<u32> {
    resp.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| toks.iter().map(|t| t.top_logprobs[0].token).collect())
        .unwrap_or_default()
}

fn text(resp: &inference::ChatCompletionResponse) -> String {
    resp.choices[0].message.content.clone().unwrap_or_default()
}

fn image_request(images: Vec<image::DynamicImage>, prompt: &str, max_len: usize) -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        prompt,
        images,
    ))
    .set_sampler_max_len(max_len)
    .set_sampler_topk(1)
}

// GPU builds run the parity tests in bf16 on the device (the goldens match in both dtypes); CPU builds use f32.
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));

async fn build(paged: bool) -> anyhow::Result<Model> {
    let dir = model_dir().expect("checked by caller");
    let mut builder = MultimodalModelBuilder::new(&dir).with_dtype(if ON_GPU {
        ModelDType::BF16
    } else {
        ModelDType::F32
    });
    if !ON_GPU {
        builder = builder.with_force_cpu();
    }
    #[cfg(any(feature = "cuda", feature = "metal"))]
    if paged {
        // The prefix cacher is off by default but on in the server, and it is where a block-hash collision would show.
        builder = builder
            .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
            .with_prefix_cache_n(Some(16));
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let _ = paged;
    builder.build().await
}

macro_rules! skip_unless_model {
    ($what:literal) => {
        if model_dir().is_none() {
            eprintln!(
                "SKIP {}: set {} to a local checkpoint dir",
                $what, MODEL_ENV
            );
            return Ok(());
        }
    };
}

#[tokio::test]
async fn greedy_ids_match_transformers() -> anyhow::Result<()> {
    skip_unless_model!("greedy parity");
    let model = build(false).await?;
    let resp = model
        .send_chat_request(
            image_request(vec![fixture("ocr.png")?], OCR_PROMPT, OCR_GOLDEN.len())
                .return_logprobs(true)
                .set_sampler_topn_logprobs(1),
        )
        .await?;
    assert_eq!(strip_eos(&greedy_ids(&resp)), strip_eos(OCR_GOLDEN));
    Ok(())
}

async fn assert_text_golden(name: &str) -> anyhow::Result<()> {
    let (prompt, golden) = golden(name);
    let model = build(false).await?;
    let resp = model
        .send_chat_request(image_request(vec![fixture(name)?], prompt, MAX_LEN))
        .await?;
    assert_eq!(text(&resp), golden, "{name} [{prompt}]");
    Ok(())
}

// One test per fixture so the CPU f32 decodes run in parallel instead of as one long pole.
macro_rules! text_golden_test {
    ($name:ident, $fixture:literal) => {
        #[tokio::test]
        async fn $name() -> anyhow::Result<()> {
            skip_unless_model!("text parity");
            assert_text_golden($fixture).await
        }
    };
}

text_golden_test!(ocr_text_matches_transformers, "ocr.png");
text_golden_test!(page_00_text_matches_transformers, "page_00.png");
text_golden_test!(page_01_text_matches_transformers, "page_01.png");
text_golden_test!(table_otsl_matches_transformers, "table.png");

#[tokio::test]
async fn text_only_matches_transformers() -> anyhow::Result<()> {
    skip_unless_model!("text-only parity");
    let model = build(false).await?;
    let resp = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, TEXT_ONLY_PROMPT)
                .return_logprobs(true)
                .set_sampler_topn_logprobs(1)
                .set_sampler_topk(1)
                .set_sampler_max_len(TEXT_ONLY_GOLDEN.len()),
        )
        .await?;
    assert_eq!(strip_eos(&greedy_ids(&resp)), strip_eos(TEXT_ONLY_GOLDEN));
    Ok(())
}

// Every image needs its own grid row; transformers also transcribes only the first page of a two-image message.
#[tokio::test]
async fn two_images_in_one_message_match_transformers() -> anyhow::Result<()> {
    skip_unless_model!("multi-image");
    let model = build(false).await?;
    let pages = vec![fixture("page_00.png")?, fixture("page_01.png")?];
    let resp = model
        .send_chat_request(image_request(pages, OCR_PROMPT, 2 * MAX_LEN))
        .await?;
    assert_eq!(text(&resp), golden("page_00.png").1);
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
mod gpu {
    use super::*;

    // A scheduler spin never completes either request, so the mixed-batch test fails on this instead of hanging.
    const MIXED_BATCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

    // Same-size pages give byte-identical prompts, so only the registered image span keeps their KV blocks apart.
    #[tokio::test]
    async fn prefix_cache_does_not_serve_one_image_for_another() -> anyhow::Result<()> {
        skip_unless_model!("paged prefix cache");
        let model = build(true).await?;
        let run = async |name: &str| -> anyhow::Result<String> {
            Ok(text(
                &model
                    .send_chat_request(image_request(vec![fixture(name)?], OCR_PROMPT, MAX_LEN))
                    .await?,
            ))
        };
        let first = run("page_00.png").await?;
        assert_eq!(first, golden("page_00.png").1);
        assert_eq!(run("page_01.png").await?, golden("page_01.png").1);
        assert_eq!(
            run("page_00.png").await?,
            first,
            "prefix cache reuse changed page_00"
        );
        Ok(())
    }

    #[tokio::test]
    async fn isq_q8_0_keeps_ocr_text() -> anyhow::Result<()> {
        skip_unless_model!("ISQ Q8_0");
        let model = MultimodalModelBuilder::new(model_dir().expect("checked above"))
            .with_dtype(ModelDType::BF16)
            .with_isq(inference::IsqType::Q8_0)
            .build()
            .await?;
        for name in ["ocr.png", "page_00.png", "page_01.png"] {
            let (prompt, golden) = golden(name);
            let resp = model
                .send_chat_request(image_request(vec![fixture(name)?], prompt, MAX_LEN))
                .await?;
            assert_eq!(text(&resp), golden, "{name}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn mixed_text_and_image_batch_makes_progress() -> anyhow::Result<()> {
        skip_unless_model!("paged mixed batch");
        let model = build(true).await?;
        let image = image_request(vec![fixture("page_00.png")?], OCR_PROMPT, MAX_LEN);
        let alone = text(&model.send_chat_request(image.clone()).await?);
        let (batched, text_only) = tokio::time::timeout(MIXED_BATCH_TIMEOUT, async {
            tokio::join!(
                model.send_chat_request(image),
                model.send_chat_request(
                    RequestBuilder::new()
                        .add_message(TextMessageRole::User, TEXT_ONLY_PROMPT)
                        .set_sampler_max_len(8)
                )
            )
        })
        .await?;
        text_only?;
        assert_eq!(
            alone,
            text(&batched?),
            "OCR output changed when a text-only request shared the batch"
        );
        Ok(())
    }
}
