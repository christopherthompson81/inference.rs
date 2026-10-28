//! Runtime LoRA on a GGUF whose Q/K rows a converter reordered to adjacent RoPE pairs, against the safetensors source.

use std::collections::HashMap;
use std::path::Path;

use inference::{
    GgufModelBuilder, LoraModelBuilder, Model, ModelDType, RequestBuilder, TextMessageRole,
    TextMessages, TextModelBuilder,
};
use inference_tensor::{Device, Tensor};

#[path = "../support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;
#[path = "../support/llama_tiny_gguf.rs"]
mod tiny_gguf;
use tiny_gguf::{HEAD_DIM, HEADS, HIDDEN, KV_HEADS, LAYERS, write_tiny_gguf};

const PROMPT: &str = "hello";
const MAX_LEN: usize = 8;
const ADAPTER: &str = "qk-adapter";
const ADAPTER_RANK: usize = 2;
const GGUF_FILE: &str = "tiny.gguf";
// Adjacent and half-split RoPE are the same rotation; only f32 summation order differs between the two loads
const LOGPROB_TOLERANCE: f32 = 1e-3;

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

fn assert_same_trace(
    gguf: &[(u32, f32)],
    safetensors: &[(u32, f32)],
    what: &str,
) -> anyhow::Result<()> {
    let same = gguf.len() == safetensors.len()
        && gguf
            .iter()
            .zip(safetensors)
            .all(|(g, s)| g.0 == s.0 && (g.1 - s.1).abs() < LOGPROB_TOLERANCE);
    anyhow::ensure!(
        same,
        "{what}: GGUF {gguf:?} differs from safetensors {safetensors:?}"
    );
    Ok(())
}

// Touches both Q (two heads) and K (one KV head), in the half-split layout a Hugging Face adapter is trained in.
fn write_qk_adapter(dir: &Path) -> anyhow::Result<()> {
    std::fs::write(
        dir.join("adapter_config.json"),
        format!(
            r#"{{"r":{ADAPTER_RANK},"lora_alpha":{ADAPTER_RANK},"target_modules":["q_proj","k_proj"]}}"#
        ),
    )?;
    let ramp = |rows: usize, cols: usize, scale: f32| {
        let data = (0..rows * cols)
            .map(|i| ((i % 7) as f32 - 3.0) * scale)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, (rows, cols), &Device::Cpu)
    };
    let mut tensors = HashMap::new();
    for layer in 0..LAYERS {
        for (proj, out) in [
            ("q_proj", HEADS * HEAD_DIM),
            ("k_proj", KV_HEADS * HEAD_DIM),
        ] {
            let prefix = format!("base_model.model.model.layers.{layer}.self_attn.{proj}");
            tensors.insert(
                format!("{prefix}.lora_A.weight"),
                ramp(ADAPTER_RANK, HIDDEN, 0.3)?,
            );
            tensors.insert(
                format!("{prefix}.lora_B.weight"),
                ramp(out, ADAPTER_RANK, 0.5)?,
            );
        }
    }
    inference_tensor::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;
    Ok(())
}

#[tokio::test]
async fn a_lora_adapter_on_an_adjacent_rope_gguf_matches_the_safetensors_model()
-> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let gguf_dir = tempfile::tempdir()?;
    write_tiny_gguf(checkpoint.path(), &gguf_dir.path().join(GGUF_FILE), &[])?;
    let adapter = tempfile::tempdir()?;
    write_qk_adapter(adapter.path())?;

    let safetensors = LoraModelBuilder::from_text_model_builder(
        TextModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu(),
    )
    .with_adapter(ADAPTER, adapter.path().to_string_lossy())
    .build()
    .await?;
    let gguf = GgufModelBuilder::new(gguf_dir.path().to_string_lossy(), vec![GGUF_FILE])
        .with_tok_model_id(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_lora_adapter(ADAPTER, adapter.path().to_string_lossy())
        .build()
        .await?;

    let base = greedy_trace(&safetensors, None).await?;
    assert_same_trace(&greedy_trace(&gguf, None).await?, &base, "base model")?;
    let adapted = greedy_trace(&safetensors, Some(ADAPTER)).await?;
    anyhow::ensure!(adapted != base, "the adapter did not change the decode");
    assert_same_trace(
        &greedy_trace(&gguf, Some(ADAPTER)).await?,
        &adapted,
        "with the adapter",
    )
}
