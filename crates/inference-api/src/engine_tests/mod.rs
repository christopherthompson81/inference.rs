//! The engine on a tiny random-weight PaddleOCR-VL, where a test seeds or reads store state no request reaches.

use serde_json::json;

use crate::Engine;

#[path = "../../../inference/tests/support/paddleocr_vl_tiny.rs"]
mod support;

mod logits_processors;
mod responses_stream;
mod store;

async fn tiny_engine() -> anyhow::Result<(tempfile::TempDir, Engine)> {
    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = Engine::load(spec).await?;
    Ok((dir, engine))
}
