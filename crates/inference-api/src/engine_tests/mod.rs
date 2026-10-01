//! The engine on a tiny random-weight PaddleOCR-VL, where a test seeds or reads store state no request reaches.

use serde_json::json;

use crate::Engine;

#[path = "../../../inference/tests/support/paddleocr_vl_tiny.rs"]
mod support;

mod logits_processors;
mod parallel_tools;
mod responses_stream;
mod store;

async fn tiny_engine() -> anyhow::Result<(tempfile::TempDir, Engine)> {
    tiny_engine_with(crate::engine::EngineCallbacks::default()).await
}

async fn tiny_engine_with(
    callbacks: crate::engine::EngineCallbacks,
) -> anyhow::Result<(tempfile::TempDir, Engine)> {
    tiny_engine_from(json!({}), callbacks).await
}

/// The tiny engine with `extra` merged into its spec's top level.
async fn tiny_engine_from(
    extra: serde_json::Value,
    callbacks: crate::engine::EngineCallbacks,
) -> anyhow::Result<(tempfile::TempDir, Engine)> {
    let dir = support::tiny_checkpoint()?;
    let mut spec = json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    });
    if let (Some(spec), Some(extra)) = (spec.as_object_mut(), extra.as_object()) {
        spec.extend(extra.clone());
    }
    let engine = Engine::load_with_callbacks(serde_json::from_value(spec)?, callbacks).await?;
    Ok((dir, engine))
}
