//! A tiny random-weight Voxtral Realtime in both layouts it ships in: transformers' (renamed, Q and K half-split)
//! decodes as Mistral's consolidated one, an ISQ load of it writes a UQFF that reloads alike, and repacked as an
//! audio.cpp GGUF it decodes as its safetensors.

use inference::{
    AudioInput, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole,
};

#[path = "../support/audiocpp_package.rs"]
mod package;
#[path = "../support/voxtral_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::{consolidated_checkpoint, transformers_checkpoint};

const PROMPT: &str = "Hi";
const MAX_LEN: usize = 4;
const SAMPLE_RATE: u32 = 16_000;
const SAMPLES: usize = 32_000;
const TONE_HZ: f32 = 440.0;
// both layouts hold the same weights, so only the RoPE rotation's rounding parts them
const LAYOUT_TOLERANCE: f32 = 1e-4;
const UQFF_TOLERANCE: f32 = 1e-3;

fn builder(dir: &std::path::Path) -> MultimodalModelBuilder {
    MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

fn request() -> RequestBuilder {
    let clip = AudioInput {
        samples: (0..SAMPLES)
            .map(|i| {
                0.3 * (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect(),
        sample_rate: SAMPLE_RATE,
        channels: 1,
    };
    RequestBuilder::from(MultimodalMessages::new().add_audio_message(
        TextMessageRole::User,
        PROMPT,
        vec![clip],
    ))
}

async fn trace(dir: &std::path::Path) -> anyhow::Result<traces::Trace> {
    let model = builder(dir).build().await?;
    Ok(traces::greedy(&model, request(), MAX_LEN).await?.0)
}

#[tokio::test]
async fn transformers_layout_decodes_as_the_consolidated_one() -> anyhow::Result<()> {
    let consolidated = consolidated_checkpoint()?;
    let transformers = transformers_checkpoint(consolidated.path())?;
    let (a, b) = (
        trace(consolidated.path()).await?,
        trace(transformers.path()).await?,
    );
    anyhow::ensure!(
        traces::close(&a, &b, LAYOUT_TOLERANCE),
        "the layouts decode apart: {b:?} vs {a:?}"
    );
    Ok(())
}

// the UQFF carries the residual tensors under transformers' names, which the reload reads back
#[tokio::test]
async fn an_isq_load_of_the_transformers_layout_reloads_from_its_uqff() -> anyhow::Result<()> {
    let consolidated = consolidated_checkpoint()?;
    let transformers = transformers_checkpoint(consolidated.path())?;
    let (written, read) =
        traces::isq_and_uqff_reload_traces(transformers.path(), builder, request, MAX_LEN).await?;
    anyhow::ensure!(
        traces::close(&written, &read, UQFF_TOLERANCE),
        "the UQFF reload moved: {read:?} vs {written:?}"
    );
    Ok(())
}

#[tokio::test]
async fn an_audiocpp_package_decodes_as_its_safetensors() -> anyhow::Result<()> {
    let consolidated = consolidated_checkpoint()?;
    let transformers = transformers_checkpoint(consolidated.path())?;
    let dir = tempfile::tempdir()?;
    let gguf = package::repack_all(
        transformers.path(),
        "voxtral_realtime",
        &[("", "model.safetensors")],
        &dir.path().join("tiny-f32.gguf"),
    )?;
    let (a, b) = (trace(transformers.path()).await?, trace(&gguf).await?);
    anyhow::ensure!(a == b, "the package decodes apart: {b:?} vs {a:?}");
    Ok(())
}
