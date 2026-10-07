//! The text-model load paths on a tiny random-weight Llama: plain, in-situ quantized, and reloaded from the UQFF it wrote.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inference::{
    IsqType, LoraModelBuilder, Model, ModelDType, RequestBuilder, TextMessageRole, TextMessages,
    TextModelBuilder, UqffTextModelBuilder,
};
use inference_tensor::{Device, Tensor};

#[path = "../support/decode_graphs.rs"]
mod decode_graphs;
#[path = "../support/llama_tiny.rs"]
mod support;
use support::{tiny_llama_checkpoint, tiny_llama_checkpoint_with};

const PROMPT: &str = "hello";
const MAX_LEN: usize = 8;
const UQFF_EXTENSION: &str = "uqff";
const ADAPTER: &str = "q-proj-adapter";
const ADAPTER_RANK: usize = 2;
// From tests/fixtures/llama_tiny/config.json; q_proj is hidden x hidden there.
const TINY_LAYERS: usize = 2;
const TINY_HIDDEN: usize = 32;
// f32 on both sides: the GPU's summation order moves logprobs ~1e-6, which these large random weights grow to ~3e-3
const F32_LOGPROB_TOLERANCE: f32 = 1e-2;

fn cpu_text_builder(dir: &Path) -> TextModelBuilder {
    TextModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

async fn greedy_ids(model: &Model) -> anyhow::Result<Vec<u32>> {
    let request =
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT))
            .set_sampler_max_len(MAX_LEN)
            .set_sampler_topk(1)
            .return_logprobs(true)
            .set_sampler_topn_logprobs(1);
    let response = model.send_chat_request(request).await?;
    let tokens = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| t.top_logprobs[0].token)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    anyhow::ensure!(!tokens.is_empty(), "the model generated nothing");
    Ok(tokens)
}

// (token, logprob) per greedy step, so an adapter that moves the logits shows even when the argmax holds.
async fn greedy_trace(model: &Model, adapter: Option<&str>) -> anyhow::Result<Vec<(u32, f32)>> {
    let mut request =
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT))
            .set_sampler_max_len(MAX_LEN)
            .set_sampler_topk(1)
            .return_logprobs(true)
            .set_sampler_topn_logprobs(1);
    if let Some(adapter) = adapter {
        request = request.set_adapter(adapter);
    }
    let response = model.send_chat_request(request).await?;
    let trace = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| (t.top_logprobs[0].token, t.top_logprobs[0].logprob))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    anyhow::ensure!(!trace.is_empty(), "the model generated nothing");
    Ok(trace)
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

fn uqff_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|path| path.extension().is_some_and(|ext| ext == UQFF_EXTENSION));
    files.sort();
    Ok(files)
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

#[tokio::test]
async fn an_isq_load_can_calibrate_on_a_text_file_first() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let calibration = checkpoint.path().join("calibration.txt");
    std::fs::write(
        &calibration,
        "hello world. the quick brown fox jumps over the lazy dog.",
    )?;
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
