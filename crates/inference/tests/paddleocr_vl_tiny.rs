//! Engine behavior on a tiny random-weight PaddleOCR-VL: the checkpoint is built at test time, so these run everywhere.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Result as CandleResult, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init};
use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};
use inference_models_other::paddleocr_vl::{config::Config, PaddleOcrVlModel};
use inference_nn::{
    device_map::DeviceMapSetting, model::NormalLoadingMetadata,
    paged_attention::AttentionImplementation,
};
use inference_quant::{ShardedSafeTensors, TensorShapes};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny"
);
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/paddleocr_vl");
const OCR_PROMPT: &str = "OCR:";
// Long enough that a shared prefix runs past the image into whole paged blocks; paged hits never end inside an image.
const LONG_PROMPT: &str =
    "OCR: transcribe every line of this page exactly, keeping the original line breaks and punctuation in place.";
const TEXT_PROMPT: &str = "Reply with the single word: ok";
const MAX_LEN: usize = 8;
// Large enough that the image moves the logits, small enough to stay finite through two layers.
const WEIGHT_STD: f32 = 0.5;
// Fixed so a failure reproduces with the same weights; the constructor requests tensors in a fixed order.
const WEIGHT_SEED: u64 = 0x0CE1_2024;
// A scheduler spin never completes either request, so the mixed-batch test fails on this instead of hanging.
const MIXED_BATCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
// Cached KV comes from a different prefill than a full recompute, so logprobs match only to rounding.
const LOGPROB_TOLERANCE: f32 = 1e-3;
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));

/// Hands out random tensors for whatever the model constructor asks for, and keeps them to write a checkpoint.
#[derive(Clone)]
struct RecordingWeights(Arc<Mutex<(StdRng, HashMap<String, Tensor>)>>);

impl RecordingWeights {
    fn new() -> Self {
        Self(Arc::new(Mutex::new((
            StdRng::seed_from_u64(WEIGHT_SEED),
            HashMap::new(),
        ))))
    }
}

impl SimpleBackend for RecordingWeights {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        dev: &Device,
    ) -> CandleResult<Tensor> {
        let mut guard = self.0.lock().unwrap();
        let (rng, seen) = &mut *guard;
        if let Some(t) = seen.get(name) {
            return Ok(t.clone());
        }
        let normal = Normal::new(0f32, WEIGHT_STD).map_err(candle_core::Error::wrap)?;
        let data = (0..s.elem_count())
            .map(|_| normal.sample(rng))
            .collect::<Vec<_>>();
        let t = Tensor::from_vec(data, s, &Device::Cpu)?.to_dtype(dtype)?;
        seen.insert(name.to_string(), t.clone());
        t.to_device(dev)
    }

    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> CandleResult<Tensor> {
        candle_core::bail!("no shape for {name}")
    }

    fn contains_tensor(&self, _: &str) -> bool {
        true
    }
}

impl TensorShapes for RecordingWeights {
    fn tensor_shapes(&self) -> HashMap<String, Vec<usize>> {
        HashMap::new()
    }
}

/// The committed tiny config, tokenizer and templates plus random weights for exactly the tensors the model loads.
fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for entry in std::fs::read_dir(TINY)? {
        let path = entry?.path();
        std::fs::copy(&path, dir.path().join(path.file_name().unwrap()))?;
    }
    let cfg: Config =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json"))?)?;
    let weights = RecordingWeights::new();
    let num_layers = cfg.text_config().num_hidden_layers;
    let metadata = NormalLoadingMetadata {
        mapper: DeviceMapSetting::dummy().into_mapper(
            num_layers,
            &Device::Cpu,
            None,
            &[Device::Cpu],
        )?,
        loading_isq: false,
        real_device: Device::Cpu,
        multi_progress: Arc::new(indicatif::MultiProgress::new()),
        matformer_slicing_config: None,
        rope_pairing: None,
    };
    let vb = ShardedSafeTensors::wrap(weights.clone(), DType::F32, Device::Cpu);
    PaddleOcrVlModel::new(&cfg, vb, metadata, AttentionImplementation::Eager)?;
    let tensors = std::mem::take(&mut weights.0.lock().unwrap().1);
    candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

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
    builder.with_prefix_cache_n(Some(16)).build().await
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
    let ids = model.list_models()?;
    let [id] = ids.as_slice() else {
        anyhow::bail!("expected exactly one model, got {ids:?}");
    };
    let first = trace(
        &model
            .send_chat_request(image_request(&["page_00.png"])?)
            .await?,
    );
    assert!(!first.is_empty());

    model.unload_model(id)?;
    assert!(!model.is_model_loaded(id)?);
    model.reload_model(id).await?;
    assert!(model.is_model_loaded(id)?);

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
