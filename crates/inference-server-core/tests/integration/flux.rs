//! Real-weight FLUX through the engine API (what the C ABI calls) and /v1/images/generations, on a single-file layout.
#![cfg(feature = "cuda")]

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL_ENV: &str = "INFERENCE_TEST_FLUX_DIR";
const PROMPT: &str = "A red apple on a wooden table";
// Small enough to generate in seconds, a multiple of 16 as FLUX's patching needs.
const SIDE: u32 = 256;
// A wrong text encoder or config still yields a PNG of the right size, but not a red subject with any detail.
const MIN_LUMA_STD_DEV: f64 = 20.0;

fn request(response_format: &str) -> Value {
    json!({
        "prompt": PROMPT,
        "response_format": response_format,
        "height": SIDE,
        "width": SIDE,
    })
}

fn assert_side(image: &image::DynamicImage) {
    assert_eq!((image.width(), image.height()), (SIDE, SIDE));
}

fn assert_red_subject(image: &image::DynamicImage) {
    let pixels = image.to_rgb8();
    let (lo, hi) = (SIDE / 4, 3 * SIDE / 4);
    let mut sums = [0f64; 3];
    for y in lo..hi {
        for x in lo..hi {
            for (sum, channel) in sums.iter_mut().zip(pixels.get_pixel(x, y).0) {
                *sum += f64::from(channel);
            }
        }
    }
    let [red, green, blue] = sums;
    assert!(red > green && red > blue, "centre channel sums {sums:?}");
    let luma = image.to_luma8();
    let n = luma.len() as f64;
    let mean = luma.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = luma
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    assert!(var.sqrt() > MIN_LUMA_STD_DEV, "luma std dev {}", var.sqrt());
}

#[tokio::test(flavor = "multi_thread")]
async fn flux_gguf_generates_images_through_the_engine_and_http() -> anyhow::Result<()> {
    let Ok(dir) = std::env::var(MODEL_ENV) else {
        eprintln!("SKIP: {MODEL_ENV} is not set");
        return Ok(());
    };
    // Not `skip_without_cuda!`: its lock guard would be held across the awaits below.
    if inference_tensor::Device::new_cuda(0).is_err() {
        eprintln!("SKIP: no CUDA device");
        return Ok(());
    }
    let dir = std::fs::canonicalize(dir)?;

    let spec = serde_json::from_value(json!({
        "model": {"DiffusionPlain": {"model_id": dir.to_string_lossy(), "arch": "flux", "dtype": "bf16"}},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;

    let body = engine
        .image_generation_json(request("b64_json").to_string().as_bytes())
        .await?;
    let response: Value = serde_json::from_str(&body)?;
    let payload = response["data"][0]["b64_json"]
        .as_str()
        .expect("b64_json requested");
    let image = image::load_from_memory(&STANDARD.decode(payload)?)?;
    assert_side(&image);
    assert_red_subject(&image);

    let app = InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .build()
        .await?;
    let http = Request::post("/v1/images/generations")
        .header("content-type", "application/json")
        .body(Body::from(request("url").to_string()))?;
    let response = app.clone().oneshot(http).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let response: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await?)?;
    assert!(response["data"][0]["b64_json"].is_null());
    let url = response["data"][0]["url"].as_str().expect("url requested");
    let content = app.oneshot(Request::get(url).body(Body::empty())?).await?;
    assert_eq!(content.status(), StatusCode::OK);
    assert_eq!(content.headers()["content-type"], "image/png");
    let png = to_bytes(content.into_body(), usize::MAX).await?;
    assert_side(&image::load_from_memory(&png)?);
    Ok(())
}
