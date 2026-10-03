//! Builds tiny random-weight LLaVA 1.5 (Llama text) and LLaVA-NeXT (Mistral text) checkpoints at test time.
#![allow(dead_code)]

use inference_models_llama::llava::{config::Config, llava_next, llava15};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

const LLAVA15: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llava/llava15");
const LLAVA_NEXT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/llava/llava_next"
);
// From the fixtures' text_config, written by make_tiny.py.
const TEXT_LAYERS: usize = 2;
// Optional tensors the text constructors probe for (GGUF-converted Llama 3 RoPE frequencies); real LLaVA has none.
const ABSENT: &[&str] = &["language_model.model.rope_freqs.weight"];

fn record(
    dir: &str,
    build: impl FnOnce(
        &Config,
        inference_quant::ShardedVarBuilder,
        inference_nn::model::NormalLoadingMetadata,
    ) -> candle_core::Result<()>,
) -> anyhow::Result<tempfile::TempDir> {
    let cfg: Config =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let files = recording::fixture_files(dir)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(&files, TEXT_LAYERS, ABSENT, |vb, metadata| {
        build(&cfg, vb, metadata)
    })
}

pub fn tiny_llava15() -> anyhow::Result<tempfile::TempDir> {
    record(LLAVA15, |cfg, vb, metadata| {
        llava15::Model::new(cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}

pub fn tiny_llava_next() -> anyhow::Result<tempfile::TempDir> {
    record(LLAVA_NEXT, |cfg, vb, metadata| {
        llava_next::Model::new(cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}
