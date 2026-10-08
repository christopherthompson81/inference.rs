//! A tiny random-weight Llama: its load paths (plain, ISQ, UQFF, LoRA), in-place requantization, prefix cache and graphs.

use std::collections::HashMap;
use std::path::Path;

use inference::{
    IsqType, LoraModelBuilder, Model, ModelDType, RequestBuilder, TextMessageRole, TextMessages,
    TextModelBuilder, UqffTextModelBuilder,
};
use inference_tensor::{Device, Tensor};

#[path = "../support/decode_graphs.rs"]
mod decode_graphs;
#[path = "../support/llama_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::{tiny_llama_checkpoint, tiny_llama_checkpoint_with};
use traces::{close, greedy, uqff_files};

const PROMPT: &str = "hello";
// Long enough to fill whole paged blocks, so the paged prefix cache has something to match.
const LONG_PROMPT: &str =
    "tell me everything about the quick brown fox and the lazy dog it jumps over, from the start.";
const PREFIX_CACHE_SEQS: usize = 16;
const CALIBRATION_TEXT: &str = "hello world. the quick brown fox jumps over the lazy dog.";
const MAX_LEN: usize = 8;
const ADAPTER: &str = "q-proj-adapter";
const ADAPTER_RANK: usize = 2;
// From tests/fixtures/llama_tiny/config.json; q_proj is hidden x hidden there.
const TINY_LAYERS: usize = 2;
const TINY_HIDDEN: usize = 32;
// f32 on both sides: the GPU's summation order moves logprobs ~1e-6, which these large random weights grow to ~3e-3
const F32_LOGPROB_TOLERANCE: f32 = 1e-2;
// A CPU run repeats exactly.
const CPU_PIN_TOLERANCE: f32 = 1e-5;
// Cached KV comes from a different prefill than a recompute, so logprobs match only to rounding.
const CACHED_LOGPROB_TOLERANCE: f32 = 1e-3;
// Part of the engine's refusal to requantize a model with no tracked layers.
const NOT_AN_ISQ_LOAD: &str = "loaded with ISQ";

fn cpu_text_builder(dir: &Path) -> TextModelBuilder {
    TextModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

fn prompt_request() -> RequestBuilder {
    RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT))
}

async fn greedy_ids(model: &Model) -> anyhow::Result<Vec<u32>> {
    let (trace, _) = greedy(model, prompt_request(), MAX_LEN).await?;
    Ok(trace.into_iter().map(|(id, _)| id).collect())
}

// (token, logprob) per greedy step, so an adapter that moves the logits shows even when the argmax holds.
async fn greedy_trace(model: &Model, adapter: Option<&str>) -> anyhow::Result<traces::Trace> {
    let request = match adapter {
        Some(adapter) => prompt_request().set_adapter(adapter),
        None => prompt_request(),
    };
    Ok(greedy(model, request, MAX_LEN).await?.0)
}

// A rank-2 PEFT adapter on every layer's q_proj, large enough to move the tiny model's logits.
fn write_q_proj_adapter(dir: &Path) -> anyhow::Result<()> {
    std::fs::write(
        dir.join("adapter_config.json"),
        format!(
            r#"{{"r":{ADAPTER_RANK},"lora_alpha":{ADAPTER_RANK},"target_modules":["q_proj"]}}"#
        ),
    )?;
    let ramp = |rows: usize, cols: usize, scale: f32| {
        let data = (0..rows * cols)
            .map(|i| ((i % 7) as f32 - 3.0) * scale)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, (rows, cols), &Device::Cpu)
    };
    let mut tensors = HashMap::new();
    for layer in 0..TINY_LAYERS {
        let prefix = format!("base_model.model.model.layers.{layer}.self_attn.q_proj");
        tensors.insert(
            format!("{prefix}.lora_A.weight"),
            ramp(ADAPTER_RANK, TINY_HIDDEN, 0.3)?,
        );
        tensors.insert(
            format!("{prefix}.lora_B.weight"),
            ramp(TINY_HIDDEN, ADAPTER_RANK, 0.5)?,
        );
    }
    inference_tensor::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;
    Ok(())
}

#[tokio::test]
async fn a_text_checkpoint_loads_and_decodes_deterministically() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let model = cpu_text_builder(checkpoint.path()).build().await?;
    assert_eq!(greedy_ids(&model).await?, greedy_ids(&model).await?);
    Ok(())
}

#[tokio::test]
async fn an_isq_load_and_a_reload_of_its_uqff_decode_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_text_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let quantized_ids = greedy_ids(&quantized).await?;
    drop(quantized);

    let written = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(
        !written.is_empty(),
        "no UQFF written to {}",
        uqff_dir.path().display()
    );
    let reloaded = UqffTextModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        vec![written[0].clone()],
    )
    .into_inner()
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;
    assert_eq!(greedy_ids(&reloaded).await?, quantized_ids);
    Ok(())
}

// Requantizing an ISQ load in place works from the loaded weights, so Q8_0 then HQQ4 is pinned, not an HQQ4 load.
#[tokio::test]
async fn an_isq_load_requantizes_in_place_as_recorded() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let model = cpu_text_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .build()
        .await?;
    let before = greedy_trace(&model, None).await?;
    model.re_isq_model(IsqType::HQQ4).await?;
    let after = greedy_trace(&model, None).await?;
    let expected_before: &[(u32, f32)] = &[
        (155, -2.0968606),
        (210, -1.9850466),
        (120, -1.8510517),
        (18, -3.1753128),
        (202, -2.1321208),
        (210, -1.7131561),
        (120, -1.5091397),
        (189, -3.0938122),
    ];
    let expected_after: &[(u32, f32)] = &[
        (155, -1.9171935),
        (210, -2.1728337),
        (120, -2.0395246),
        (18, -3.3132489),
        (202, -2.1687129),
        (210, -1.8857474),
        (155, -2.4020855),
        (210, -1.3533674),
    ];
    anyhow::ensure!(
        close(&before, expected_before, CPU_PIN_TOLERANCE),
        "Q8_0 decode moved: {before:?}"
    );
    anyhow::ensure!(
        close(&after, expected_after, CPU_PIN_TOLERANCE),
        "requantized decode moved: {after:?}"
    );
    Ok(())
}

#[tokio::test]
async fn re_isq_without_an_isq_load_fails() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let model = cpu_text_builder(checkpoint.path()).build().await?;
    let error = model.re_isq_model(IsqType::Q8_0).await.err();
    anyhow::ensure!(
        error
            .as_ref()
            .is_some_and(|e| e.to_string().contains(NOT_AN_ISQ_LOAD)),
        "re-ISQ of an unquantized load: {error:?}"
    );
    Ok(())
}

// The repeat of a prompt reuses the first run's cached prefix and decodes the same.
#[tokio::test]
async fn a_repeated_prompt_hits_the_prefix_cache() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let builder = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_prefix_cache_n(Some(PREFIX_CACHE_SEQS));
    // GPU builds take the paged path, whose prefix cache matches whole blocks
    let builder = if cfg!(any(feature = "cuda", feature = "metal")) {
        builder.with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
    } else {
        builder.with_force_cpu()
    };
    let model = builder.build().await?;
    let request = || {
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, LONG_PROMPT))
    };
    let (first, first_cached) = greedy(&model, request(), MAX_LEN).await?;
    let (second, second_cached) = greedy(&model, request(), MAX_LEN).await?;
    anyhow::ensure!(
        first_cached == 0 && second_cached > 0,
        "cached prompt tokens: {first_cached} then {second_cached}"
    );
    anyhow::ensure!(
        close(&first, &second, CACHED_LOGPROB_TOLERANCE),
        "a prefix hit changed decoding: {first:?} vs {second:?}"
    );
    Ok(())
}

// A GPU load that writes UQFF serves the written file on the GPU, as a later load of it does, calibrated or not.
#[tokio::test]
async fn a_gpu_isq_load_that_writes_uqff_serves_it() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = tiny_llama_checkpoint()?;
    let calibration = checkpoint.path().join("calibration.txt");
    std::fs::write(&calibration, CALIBRATION_TEXT)?;
    let gpu_builder =
        || TextModelBuilder::new(checkpoint.path().to_string_lossy()).with_dtype(ModelDType::F32);
    for calibrate in [false, true] {
        let uqff_dir = tempfile::tempdir()?;
        let builder = gpu_builder()
            .with_isq(IsqType::Q8_0)
            .write_uqff(uqff_dir.path().join("model.uqff"));
        let builder = if calibrate {
            builder.with_calibration_file(calibration.clone())
        } else {
            builder
        };
        let written = builder.build().await?;
        let served = greedy_trace(&written, None).await?;
        drop(written);
        let reloaded = gpu_builder()
            .from_uqff(uqff_files(uqff_dir.path())?)
            .build()
            .await?;
        let expected = greedy_trace(&reloaded, None).await?;
        anyhow::ensure!(
            close(&served, &expected, CACHED_LOGPROB_TOLERANCE),
            "calibrated {calibrate}: the written model decoded {served:?}, a load of its UQFF {expected:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn writing_uqff_while_loading_from_uqff_is_refused() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    // only resolved, never read: the conflict is refused first
    std::fs::write(checkpoint.path().join("in.uqff"), [])?;
    let result = cpu_text_builder(checkpoint.path())
        .from_uqff(vec!["in.uqff".into()])
        .write_uqff(checkpoint.path().join("out.uqff"))
        .build()
        .await;
    let error = result.err().map(|e| e.to_string()).unwrap_or_default();
    anyhow::ensure!(
        error.contains("while loading from UQFF"),
        "expected the write/read conflict, got {error:?}"
    );
    Ok(())
}

#[tokio::test]
async fn an_isq_load_can_calibrate_on_a_text_file_first() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let calibration = checkpoint.path().join("calibration.txt");
    std::fs::write(&calibration, CALIBRATION_TEXT)?;
    let model = cpu_text_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .with_calibration_file(calibration)
        .build()
        .await?;
    greedy_ids(&model).await?;
    Ok(())
}

#[tokio::test]
async fn a_lora_adapter_applies_only_when_a_request_selects_it() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let adapter = tempfile::tempdir()?;
    write_q_proj_adapter(adapter.path())?;
    let base = cpu_text_builder(checkpoint.path()).build().await?;
    let base_trace = greedy_trace(&base, None).await?;
    let lora = LoraModelBuilder::from_text_model_builder(cpu_text_builder(checkpoint.path()))
        .with_adapter(ADAPTER, adapter.path().to_string_lossy())
        .build()
        .await?;
    assert_eq!(greedy_trace(&lora, None).await?, base_trace);
    let adapted = greedy_trace(&lora, Some(ADAPTER)).await?;
    assert_ne!(adapted, base_trace, "the adapter did not change the decode");
    Ok(())
}

// fattn reads no head dim 16 cache, so a CUDA build decodes this one through the gather.
#[tokio::test]
async fn paged_gpu_decode_through_the_gather_matches_the_cpu() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = tiny_llama_checkpoint()?;
    let cpu = greedy_trace(&cpu_text_builder(checkpoint.path()).build().await?, None).await?;
    let gpu = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;
    let gpu = greedy_trace(&gpu, None).await?;
    let close = gpu.len() == cpu.len()
        && gpu
            .iter()
            .zip(&cpu)
            .all(|(g, c)| g.0 == c.0 && (g.1 - c.1).abs() < F32_LOGPROB_TOLERANCE);
    anyhow::ensure!(close, "GPU decode {gpu:?} differs from CPU {cpu:?}");
    Ok(())
}

// the committed config's 16 decodes through the gather, which a graph cannot capture
const GRAPH_HEAD_DIM: usize = 64;

// Each prompt's trace, per round of concurrent requests, at a head dim the flash decode kernels serve.
async fn paged_gpu_rounds() -> anyhow::Result<decode_graphs::Rounds> {
    let checkpoint = tiny_llama_checkpoint_with(serde_json::json!({ "head_dim": GRAPH_HEAD_DIM }))?;
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;
    decode_graphs::rounds(&model, MAX_LEN).await
}

// Each GRAPH_PROMPTS entry's greedy (token, logprob) per step.
const GRAPH_TRACES: [&[(u32, f32)]; decode_graphs::GRAPH_PROMPTS.len()] = [
    &[
        (121, -1.3651954),
        (134, -2.2569206),
        (230, -2.519556),
        (230, -1.539979),
        (169, -2.4789698),
        (134, -2.802221),
        (121, -1.9325787),
        (189, -1.9721572),
    ],
    &[
        (13, -2.5397305),
        (169, -2.925844),
        (134, -3.2152035),
        (121, -2.468901),
        (189, -2.3651981),
        (8, -2.78733),
        (121, -1.138626),
        (189, -2.3949344),
    ],
    &[
        (121, -1.5961162),
        (189, -2.2986858),
        (144, -2.1111917),
        (214, -2.229178),
        (123, -2.2322695),
        (5, -1.7348924),
        (214, -2.205392),
        (123, -2.4888473),
    ],
    &[
        (17, -2.6776829),
        (148, -2.6463716),
        (225, -2.8080559),
        (225, -2.8393962),
        (225, -2.7868881),
        (225, -2.7372813),
        (225, -2.7477746),
        (134, -2.7548532),
    ],
    &[
        (121, -2.3032737),
        (157, -2.968076),
        (6, -2.433681),
        (121, -2.014543),
        (189, -2.6645021),
        (8, -2.385579),
        (121, -1.1709063),
        (189, -2.7256854),
    ],
];

// Decode steps replay captured graphs and give the eager output.
#[tokio::test]
async fn paged_gpu_decode_replays_cuda_graphs() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    decode_graphs::assert_rounds_replay(paged_gpu_rounds, &GRAPH_TRACES).await
}

#[tokio::test]
async fn paged_gpu_decode_without_cuda_graphs_matches() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    decode_graphs::assert_rounds_without_graphs(paged_gpu_rounds, &GRAPH_TRACES).await
}
