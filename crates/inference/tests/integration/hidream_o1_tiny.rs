//! HiDream-O1 through the engine on a tiny random-weight checkpoint: detection, size snapping, seeds and Dev's limits.

use base64::Engine as _;
use inference::{
    DiffusionLoaderType, DiffusionModelBuilder, ImageGenerationRequest,
    ImageGenerationResponseFormat, Model, ModelBuilder, ModelDType,
};

#[path = "../support/hidream_o1_tiny.rs"]
mod support;
use support::tiny_hidream_o1;

const PROMPT: &str = "a small red cube on a table";
const SEED: u64 = 7;
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));

fn request(steps: Option<usize>, seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        response_format: ImageGenerationResponseFormat::B64Json,
        // a wide request snaps to the model's own 16:9 size
        width: 640,
        height: 360,
        steps,
        seed: Some(seed),
        ..ImageGenerationRequest::new(PROMPT)
    }
}

fn decode(choice: &inference_api::response::ImageChoice) -> anyhow::Result<image::RgbImage> {
    let b64 = choice.b64_json.as_deref().expect("a base64 image");
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64)?;
    Ok(image::load_from_memory(&bytes)?.to_rgb8())
}

async fn image(model: &Model, request: ImageGenerationRequest) -> anyhow::Result<image::RgbImage> {
    decode(&model.generate_image(request).await?.data[0])
}

fn error_text<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Ok(_) => panic!("the request should have failed"),
        Err(err) => err.to_string(),
    }
}

#[tokio::test]
async fn hidream_o1_is_detected_and_seeds_repeat_at_the_snapped_size() -> anyhow::Result<()> {
    let (_dir, path) = tiny_hidream_o1("tiny-hidream-o1")?;
    let mut builder = ModelBuilder::new(path.to_string_lossy()).with_dtype(ModelDType::F32);
    if !ON_GPU {
        builder = builder.with_force_cpu();
    }
    let model = builder.build().await?;
    let first = image(&model, request(Some(2), SEED)).await?;
    assert_eq!(first.dimensions(), (2560, 1440));
    assert_eq!(first, image(&model, request(Some(2), SEED)).await?);
    let other_seed = image(&model, request(Some(2), SEED + 1)).await?;
    assert_ne!(first, other_seed);
    // choice i of n takes seed + i
    let pair = model
        .generate_image(ImageGenerationRequest {
            n_choices: 2,
            ..request(Some(2), SEED)
        })
        .await?;
    let decoded = pair
        .data
        .iter()
        .map(decode)
        .collect::<anyhow::Result<Vec<_>>>()?;
    // compared whole, so a failure prints no pixel bytes
    assert!(first != other_seed, "two seeds gave the same image");
    assert!(
        decoded == [first, other_seed],
        "choice i should take seed + i"
    );
    assert!(error_text(model.generate_image(request(Some(0), SEED)).await).contains("steps"));
    Ok(())
}

#[tokio::test]
async fn hidream_o1_dev_samples_its_fixed_schedule_only() -> anyhow::Result<()> {
    let (_dir, path) = tiny_hidream_o1("tiny-hidream-o1-dev")?;
    let mut builder =
        DiffusionModelBuilder::new(path.to_string_lossy(), DiffusionLoaderType::HiDreamO1Dev)
            .with_dtype(ModelDType::F32);
    if !ON_GPU {
        builder = builder.with_force_cpu();
    }
    let model = builder.build().await?;
    let text = error_text(model.generate_image(request(Some(2), SEED)).await);
    assert!(text.contains("fixed 28-step"), "{text}");
    let guided = ImageGenerationRequest {
        guidance_scale: Some(3.0),
        ..request(None, SEED)
    };
    assert!(error_text(model.generate_image(guided).await).contains("without guidance"));
    let first = image(&model, request(None, SEED)).await?;
    assert_eq!(first.dimensions(), (2560, 1440));
    assert_eq!(first, image(&model, request(None, SEED)).await?);
    Ok(())
}
