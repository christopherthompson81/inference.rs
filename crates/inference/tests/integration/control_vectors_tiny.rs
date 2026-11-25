//! Control vectors on a tiny Llama (the shared decoder): what they add, and changing them on a loaded model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inference::{
    ControlVectorFile, ControlVectors, DerivedControlVector, ModelDType, PromptInput,
    PromptLogitsRequest, SetControlVectorsRequest, TextModelBuilder,
};
use inference_tensor::{Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

#[path = "../support/decode_graphs.rs"]
mod decode_graphs;
#[path = "../support/llama_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;

// the tiny Llama's width and its last layer, which llama.cpp's numbering (from 1) can steer
const HIDDEN: usize = 32;
const STEERED_LAYER: usize = 1;
const DIRECTION_STD: f32 = 0.5;
const DIRECTION_SEED: u64 = 0x6376_6563;
const PROMPT_TOKENS: u32 = 12;
// the flash decode kernels' head dim, so the GPU model decodes in CUDA graphs
const GRAPH_HEAD_DIM: usize = 64;
const MAX_LEN: usize = 6;
const GRAPH_PROMPT: &str = "the quick brown fox jumps";
// llama.cpp's derived directions are unit length; the tiny model's states are about this wide
const DERIVED_SCALE: f32 = 4.0;

fn direction(scale: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(DIRECTION_SEED);
    let normal = Normal::new(0f32, DIRECTION_STD).expect("a valid normal");
    (0..HIDDEN)
        .map(|_| normal.sample(&mut rng) * scale)
        .collect()
}

// a control vector file holding `values` as the steered layer's direction
fn vector_file(dir: &Path, name: &str, values: &[f32]) -> anyhow::Result<PathBuf> {
    let path = dir.join(name);
    let tensors = HashMap::from([(
        format!("direction.{STEERED_LAYER}"),
        Tensor::new(values, &Device::Cpu)?,
    )]);
    inference_tensor::safetensors::save(&tensors, &path)?;
    Ok(path)
}

fn vectors(path: &Path, scale: f32, layer_range: Option<[usize; 2]>) -> ControlVectors {
    ControlVectors {
        files: vec![ControlVectorFile {
            path: path.to_string_lossy().into_owned(),
            scale,
        }],
        derived: Vec::new(),
        layer_range,
        loop_guidance: None,
    }
}

fn builder(checkpoint: &tempfile::TempDir) -> TextModelBuilder {
    TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_prefix_cache_n(None)
}

// bit patterns, so the first token's missing logprob (NaN) compares equal to itself
async fn logprobs(model: &inference::Model) -> anyhow::Result<Vec<u32>> {
    Ok(model
        .prompt_logits(PromptLogitsRequest {
            reference: None,
            model: None,
            prompt: PromptInput::Tokens((0..PROMPT_TOKENS).collect()),
            output: Default::default(),
            audio: Vec::new(),
        })
        .await?
        .token_logprobs
        .into_iter()
        .map(|logprob| logprob.unwrap_or(f32::NAN).to_bits())
        .collect())
}

async fn steered(
    checkpoint: &tempfile::TempDir,
    vectors: Option<&ControlVectors>,
) -> anyhow::Result<Vec<u32>> {
    let mut builder = builder(checkpoint);
    for file in vectors.iter().flat_map(|v| &v.files) {
        builder = builder.with_control_vector(&file.path, file.scale);
    }
    if let Some([first, last]) = vectors.and_then(|v| v.layer_range) {
        builder = builder.with_control_vector_layer_range(first, last);
    }
    logprobs(&builder.build().await?).await
}

// a zero direction, and a direction outside the layer range, add nothing: the scores are the unsteered model's bits
#[tokio::test]
async fn a_zero_vector_or_an_unsteered_layer_leaves_the_scores_bitwise_unchanged()
-> anyhow::Result<()> {
    let checkpoint = support::tiny_llama_checkpoint()?;
    let dir = tempfile::tempdir()?;
    let zero = vector_file(dir.path(), "zero.safetensors", &[0.0; HIDDEN])?;
    let real = vector_file(dir.path(), "real.safetensors", &direction(1.0))?;
    let plain = steered(&checkpoint, None).await?;
    assert_eq!(
        steered(&checkpoint, Some(&vectors(&zero, 3.0, None))).await?,
        plain
    );
    assert_eq!(
        steered(&checkpoint, Some(&vectors(&real, 1.0, Some([0, 0])))).await?,
        plain
    );
    assert_ne!(
        steered(&checkpoint, Some(&vectors(&real, 1.0, None))).await?,
        plain
    );
    Ok(())
}

// the layer adds scale x direction: a direction at scale 2 scores as the doubled direction at scale 1
#[tokio::test]
async fn a_scale_multiplies_the_direction_the_layer_adds() -> anyhow::Result<()> {
    let checkpoint = support::tiny_llama_checkpoint()?;
    let dir = tempfile::tempdir()?;
    let once = vector_file(dir.path(), "once.safetensors", &direction(1.0))?;
    let twice = vector_file(dir.path(), "twice.safetensors", &direction(2.0))?;
    assert_eq!(
        steered(&checkpoint, Some(&vectors(&once, 2.0, None))).await?,
        steered(&checkpoint, Some(&vectors(&twice, 1.0, None))).await?
    );
    Ok(())
}

// set on a loaded model, the vectors steer its next request as a load with them does, and clearing them restores it
#[tokio::test]
async fn a_loaded_model_takes_new_vectors_from_its_next_request() -> anyhow::Result<()> {
    let checkpoint = support::tiny_llama_checkpoint()?;
    let dir = tempfile::tempdir()?;
    let path = vector_file(dir.path(), "real.safetensors", &direction(1.0))?;
    let plain = steered(&checkpoint, None).await?;
    let half = steered(&checkpoint, Some(&vectors(&path, 0.5, None))).await?;
    let model = builder(&checkpoint).build().await?;
    let set = |vectors| SetControlVectorsRequest {
        model: None,
        vectors,
    };
    let answered = model
        .set_control_vectors(set(vectors(&path, 0.5, None)))
        .await?;
    assert_eq!(answered.vectors, vectors(&path, 0.5, None));
    assert_eq!(logprobs(&model).await?, half);
    model
        .set_control_vectors(set(ControlVectors::default()))
        .await?;
    assert_eq!(logprobs(&model).await?, plain);
    assert_eq!(
        model.control_vectors(None).await?.vectors,
        ControlVectors::default()
    );
    Ok(())
}

// a model that runs each layer once refuses loop guidance, at load and live
#[tokio::test]
async fn a_model_without_a_loop_refuses_loop_guidance() -> anyhow::Result<()> {
    let checkpoint = support::tiny_llama_checkpoint()?;
    let refused = builder(&checkpoint).with_loop_guidance(0.3).build().await;
    anyhow::ensure!(refused.is_err(), "a Llama took loop guidance at load");
    let model = builder(&checkpoint).build().await?;
    let guided = ControlVectors {
        loop_guidance: Some(0.3),
        ..Default::default()
    };
    let live = model
        .set_control_vectors(SetControlVectorsRequest {
            model: None,
            vectors: guided,
        })
        .await;
    anyhow::ensure!(live.is_err(), "a Llama took loop guidance live");
    Ok(())
}

// changing a steered model's scale rewrites the vectors its captured decode graphs read: no capture, the new output
#[tokio::test]
async fn a_new_scale_keeps_the_captured_decode_graphs_on_cuda() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let snapshotter = decode_graphs::recorder();
    let checkpoint =
        support::tiny_llama_checkpoint_with(serde_json::json!({ "head_dim": GRAPH_HEAD_DIM }))?;
    let dir = tempfile::tempdir()?;
    let path = vector_file(dir.path(), "real.safetensors", &direction(1.0))?;
    let gpu = |scale| {
        TextModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::BF16)
            .with_paged_attn(
                inference::PagedAttentionMetaBuilder::default()
                    .build()
                    .expect("default paged attention"),
            )
            .with_control_vector(&path, scale)
    };
    let model = gpu(1.0).build().await?;
    decode_graphs::prompt_trace(&model, GRAPH_PROMPT, MAX_LEN).await?;
    let captures = |counters: &decode_graphs::Counters| {
        counters.total(decode_graphs::EVENTS, &[("event", "capture")])
    };
    // each take drains the counters: the first holds the captures, the second what the rescaled decode did
    anyhow::ensure!(
        captures(&decode_graphs::Counters::take(&snapshotter)) > 0,
        "the decode steps captured no graph"
    );
    model
        .set_control_vectors(SetControlVectorsRequest {
            model: None,
            vectors: vectors(&path, 2.0, None),
        })
        .await?;
    let rescaled = decode_graphs::prompt_trace(&model, GRAPH_PROMPT, MAX_LEN).await?;
    let after = decode_graphs::Counters::take(&snapshotter);
    assert_eq!(
        captures(&after),
        0,
        "the new scale captured graphs again: {after:?}"
    );
    anyhow::ensure!(
        after.total(decode_graphs::EVENTS, &[("event", "replay")]) > 0,
        "the rescaled decode replayed no graph: {after:?}"
    );
    drop(model);
    let fresh =
        decode_graphs::prompt_trace(&gpu(2.0).build().await?, GRAPH_PROMPT, MAX_LEN).await?;
    assert_eq!(
        rescaled.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
        fresh.iter().map(|(t, _)| *t).collect::<Vec<_>>()
    );
    Ok(())
}

// a vector derived on the model from contrasting texts is written once to the cache and read from it after; it steers,
// and texts that do not differ derive nothing
#[tokio::test]
async fn a_vector_derived_from_texts_is_cached_and_steers() -> anyhow::Result<()> {
    let cache = tempfile::tempdir()?;
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference::CONTROL_VECTOR_DIR_ENV, cache.path()) };
    let checkpoint = support::tiny_llama_checkpoint()?;
    let model = builder(&checkpoint).build().await?;
    let plain = logprobs(&model).await?;
    let derived = |positive: &str, negative: &str| ControlVectors {
        derived: vec![DerivedControlVector {
            positive: vec![positive.to_string(), format!("{positive} again")],
            negative: vec![negative.to_string(), format!("{negative} again")],
            scale: DERIVED_SCALE,
            method: None,
        }],
        ..Default::default()
    };
    let set = |vectors| SetControlVectorsRequest {
        model: None,
        vectors,
    };
    let answered = model
        .set_control_vectors(set(derived("a b c", "x y z")))
        .await?;
    let [file] = answered.vectors.files.as_slice() else {
        anyhow::bail!("one derived file, answered {:?}", answered.vectors);
    };
    anyhow::ensure!(Path::new(&file.path).starts_with(cache.path()) && file.scale == DERIVED_SCALE);
    let modified = std::fs::metadata(&file.path)?.modified()?;
    let steered = logprobs(&model).await?;
    anyhow::ensure!(steered != plain, "the derived vector did not steer");
    // the same texts again read the cached file
    model
        .set_control_vectors(set(derived("a b c", "x y z")))
        .await?;
    anyhow::ensure!(std::fs::metadata(&file.path)?.modified()? == modified);
    assert_eq!(logprobs(&model).await?, steered);
    assert!(
        model
            .set_control_vectors(set(derived("same", "same")))
            .await
            .is_err()
    );
    Ok(())
}
