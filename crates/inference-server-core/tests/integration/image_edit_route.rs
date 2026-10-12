//! /v1/images/edits end to end on a tiny random-weight HiDream-O1: OpenAI's multipart form, the JSON body and errors.

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::{chat_route::body_text, hidream_support};

const BOUNDARY: &str = "inference-image-edit-boundary";
const ROUTE: &str = "/v1/images/edits";
// a 4:3 reference, which an edit without a size matches at about 2048 x 2048's area
const REFERENCE: (u32, u32) = (640, 480);
const MATCHED: (u32, u32) = (2336, 1760);

async fn router(dir: &std::path::Path) -> anyhow::Result<axum::Router> {
    let spec = serde_json::from_value(json!({
        "model": {"DiffusionPlain": {"model_id": dir.to_string_lossy(), "arch": "hidream-o1", "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .build()
        .await
}

fn png() -> Vec<u8> {
    let (w, h) = REFERENCE;
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x % 251) as u8, (y % 241) as u8, 9])
    });
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Png,
    )
    .unwrap();
    bytes
}

// a multipart form: text fields in order, then each upload as (field, bytes)
fn form(fields: &[(&str, &str)], files: &[(&str, &[u8])]) -> Request<Body> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    for (name, bytes) in files {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"in.png\"\r\nContent-Type: image/png\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    Request::post(ROUTE)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap()
}

fn decoded(response: &Value) -> image::RgbImage {
    let b64 = response["data"][0]["b64_json"]
        .as_str()
        .expect("a base64 image");
    image::load_from_memory(&STANDARD.decode(b64).unwrap())
        .unwrap()
        .to_rgb8()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_form_and_the_json_body_edit_alike() -> anyhow::Result<()> {
    let (_dir, path) = hidream_support::tiny_hidream_o1("tiny-hidream-o1")?;
    let app = router(&path).await?;
    let image = png();
    let fields = [
        ("prompt", "make it blue"),
        ("size", "auto"),
        ("steps", "1"),
        ("seed", "7"),
        ("response_format", "b64_json"),
    ];
    let response = app
        .clone()
        .oneshot(form(&fields, &[("image", &image)]))
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let from_form = decoded(&serde_json::from_str(&body_text(response).await?)?);
    assert_eq!(from_form.dimensions(), MATCHED);

    let url = format!("data:image/png;base64,{}", STANDARD.encode(&image));
    let body = json!({
        "prompt": "make it blue",
        "images": [{"image_url": url}],
        "steps": 1,
        "seed": 7,
        "response_format": "b64_json",
    });
    let request = Request::post(ROUTE)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))?;
    let response = app.clone().oneshot(request).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let from_json = decoded(&serde_json::from_str(&body_text(response).await?)?);
    // compared whole, so a failure prints no pixel bytes
    assert!(
        from_form == from_json,
        "the form and the JSON body edited differently"
    );

    // `image[]` repeats for subject references, and a size picks the output's aspect
    let sized = [
        ("prompt", "a scene"),
        ("size", "640x360"),
        ("steps", "1"),
        ("response_format", "b64_json"),
    ];
    let response = app
        .clone()
        .oneshot(form(&sized, &[("image[]", &image), ("image[]", &image)]))
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let scene = decoded(&serde_json::from_str(&body_text(response).await?)?);
    assert_eq!(scene.dimensions(), (2560, 1440));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn masks_bad_sizes_and_missing_images_are_refused() -> anyhow::Result<()> {
    let (_dir, path) = hidream_support::tiny_hidream_o1("tiny-hidream-o1")?;
    let app = router(&path).await?;
    let image = png();
    for (fields, files, param) in [
        (
            vec![("prompt", "x")],
            vec![("image", image.as_slice()), ("mask", image.as_slice())],
            "mask",
        ),
        (
            vec![("prompt", "x"), ("size", "big")],
            vec![("image", image.as_slice())],
            "size",
        ),
        (
            vec![("prompt", "x"), ("size", "640x360"), ("width", "640")],
            vec![("image", image.as_slice())],
            "size",
        ),
        (vec![("prompt", "x")], vec![], "images"),
    ] {
        let response = app.clone().oneshot(form(&fields, &files)).await?;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{param}");
        let error: Value = serde_json::from_str(&body_text(response).await?)?;
        assert_eq!(error["error"]["param"], param, "{error}");
    }
    Ok(())
}
