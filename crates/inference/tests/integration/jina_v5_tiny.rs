//! Jina v5's task adapters over tiny Qwen3 and bidirectional Llama bases: each task embeds as its adapter merged
//! into the weights does, with the task's prefix on the text.

use std::collections::HashMap;
use std::path::Path;

use inference::{EmbeddingModelBuilder, EmbeddingRequestBuilder, Model, ModelDType};
use inference_tensor::{Device, Tensor};

#[path = "../support/qwen3_embedding_tiny.rs"]
mod qwen3;
#[path = "../support/rope_encoders_tiny.rs"]
mod rope;

const PROMPT: &str = "hello there, how are you";
const ADAPTERS: [&str; 2] = ["retrieval", "text-matching"];
// (task, the adapter it applies, the prefix Jina's code puts on its text)
const TASKS: [(&str, &str, &str); 3] = [
    ("retrieval.query", "retrieval", "Query: "),
    ("retrieval.passage", "retrieval", "Document: "),
    ("text-matching", "text-matching", "Document: "),
];
const TARGETS: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];
const RANK: usize = 4;
// alpha / rank of 2, so a dropped scale shows
const ALPHA: usize = 8;
const LORA_STD: f64 = 0.1;
const TOLERANCE: f32 = 1e-4;
const WEIGHTS: &str = "model.safetensors";
const MODULES: &str = "modules.json";
const POOLING_DIR: &str = "1_Pooling";

type Weights = HashMap<String, Tensor>;

fn load(dir: &Path) -> anyhow::Result<Weights> {
    Ok(inference_tensor::safetensors::load(
        dir.join(WEIGHTS),
        &Device::Cpu,
    )?)
}

fn edit_config(dir: &Path, edit: impl FnOnce(&mut serde_json::Value)) -> anyhow::Result<()> {
    let path = dir.join("config.json");
    let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    edit(&mut config);
    std::fs::write(path, config.to_string())?;
    Ok(())
}

fn write_modules(dir: &Path, modules: serde_json::Value) -> anyhow::Result<()> {
    std::fs::write(dir.join(MODULES), modules.to_string())?;
    Ok(())
}

/// A random PEFT adapter on every targeted projection, as `base_model.model.<module>.lora_{A,B}.weight`.
fn random_adapter(weights: &Weights) -> anyhow::Result<Weights> {
    let mut adapter = Weights::new();
    for (name, weight) in weights {
        let Some(module) = name.strip_suffix(".weight") else {
            continue;
        };
        if !TARGETS.iter().any(|target| module.ends_with(target)) {
            continue;
        }
        let (out_dim, in_dim) = weight.dims2()?;
        let peft = format!("base_model.model.{module}");
        let a = Tensor::randn(0f32, LORA_STD as f32, (RANK, in_dim), &Device::Cpu)?;
        let b = Tensor::randn(0f32, LORA_STD as f32, (out_dim, RANK), &Device::Cpu)?;
        adapter.insert(format!("{peft}.lora_A.weight"), a);
        adapter.insert(format!("{peft}.lora_B.weight"), b);
    }
    Ok(adapter)
}

/// The base with `adapter` folded in: `W + alpha / r * B A`.
fn merged(weights: &Weights, adapter: &Weights) -> anyhow::Result<Weights> {
    let scale = ALPHA as f64 / RANK as f64;
    weights
        .iter()
        .map(|(name, weight)| {
            let peft = format!("base_model.model.{}", name.trim_end_matches(".weight"));
            let weight = match adapter.get(&format!("{peft}.lora_A.weight")) {
                Some(a) => {
                    let b = &adapter[&format!("{peft}.lora_B.weight")];
                    (weight + (b.matmul(a)? * scale)?.to_dtype(weight.dtype())?)?
                }
                None => weight.clone(),
            };
            Ok((name.clone(), weight))
        })
        .collect()
}

/// Copies `base` with its weights replaced, pooled and normalized as the merged per-task repos are.
fn merged_checkpoint(
    base: &Path,
    weights: &Weights,
    architecture: &str,
) -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for file in ["config.json", "tokenizer.json"] {
        std::fs::copy(base.join(file), dir.path().join(file))?;
    }
    inference_tensor::safetensors::save(weights, dir.path().join(WEIGHTS))?;
    edit_config(dir.path(), |config| {
        config["architectures"] = serde_json::json!([architecture]);
    })?;
    std::fs::create_dir(dir.path().join(POOLING_DIR))?;
    let hidden = weights["embed_tokens.weight"].dim(1)?;
    std::fs::write(
        dir.path().join(POOLING_DIR).join("config.json"),
        serde_json::json!({
            "word_embedding_dimension": hidden,
            "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": false,
            "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false,
            "pooling_mode_lasttoken": true,
        })
        .to_string(),
    )?;
    write_modules(
        dir.path(),
        serde_json::json!([
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": POOLING_DIR, "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ]),
    )?;
    Ok(dir)
}

/// Turns `dir` into a v5 base with its adapters under `adapters/`; returns each merged checkpoint.
fn as_jina_v5(
    dir: &Path,
    merged_architecture: &str,
    edit: impl FnOnce(&mut serde_json::Value),
) -> anyhow::Result<HashMap<&'static str, tempfile::TempDir>> {
    let weights = load(dir)?;
    let mut merged_dirs = HashMap::new();
    for name in ADAPTERS {
        let adapter = random_adapter(&weights)?;
        let adapter_dir = dir.join("adapters").join(name);
        std::fs::create_dir_all(&adapter_dir)?;
        inference_tensor::safetensors::save(
            &adapter,
            adapter_dir.join("adapter_model.safetensors"),
        )?;
        std::fs::write(
            adapter_dir.join("adapter_config.json"),
            serde_json::json!({
                "peft_type": "LORA",
                "r": RANK,
                "lora_alpha": ALPHA,
                "target_modules": TARGETS,
                "bias": "none",
                "fan_in_fan_out": false,
            })
            .to_string(),
        )?;
        let merged = merged(&weights, &adapter)?;
        merged_dirs.insert(name, merged_checkpoint(dir, &merged, merged_architecture)?);
    }
    edit_config(dir, |config| {
        config["architectures"] = serde_json::json!(["JinaEmbeddingsV5Model"]);
        config["model_type"] = "jina_embeddings_v5".into();
        config["task_names"] = serde_json::json!(ADAPTERS);
        edit(config);
    })?;
    // Jina's own module, which pools in its code rather than through a Pooling module
    let _ = std::fs::remove_dir_all(dir.join(POOLING_DIR));
    write_modules(
        dir,
        serde_json::json!([{"idx": 0, "name": "transformer", "path": "", "type": "custom_st.Transformer"}]),
    )?;
    Ok(merged_dirs)
}

async fn cpu_model(dir: &Path) -> anyhow::Result<Model> {
    Ok(EmbeddingModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?)
}

async fn embed(model: &Model, text: &str, task: Option<&str>) -> anyhow::Result<Vec<f32>> {
    let mut request = EmbeddingRequestBuilder::new().add_prompt(text);
    if let Some(task) = task {
        request = request.with_task(task);
    }
    let response = serde_json::to_value(model.generate_embeddings(request.build()?).await?)?;
    Ok(serde_json::from_value(
        response["data"][0]["embedding"].clone(),
    )?)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max)
}

async fn assert_tasks_embed_as_merged_adapters(
    base: &Path,
    merged: &HashMap<&'static str, tempfile::TempDir>,
) -> anyhow::Result<()> {
    let model = cpu_model(base).await?;
    let mut alone = Vec::new();
    for (task, adapter, prefix) in TASKS {
        let expected = embed(
            &cpu_model(merged[adapter].path()).await?,
            &format!("{prefix}{PROMPT}"),
            None,
        )
        .await?;
        let embedding = embed(&model, PROMPT, Some(task)).await?;
        let diff = max_diff(&expected, &embedding);
        anyhow::ensure!(
            diff < TOLERANCE,
            "{task}: max diff {diff} from the merged adapter"
        );
        alone.push(embedding);
    }
    // the same tasks at once, which the scheduler may batch together
    let together = futures::future::try_join_all(
        TASKS
            .iter()
            .map(|(task, ..)| embed(&model, PROMPT, Some(task))),
    )
    .await?;
    for ((task, ..), (a, t)) in TASKS.iter().zip(alone.iter().zip(&together)) {
        let diff = max_diff(a, t);
        anyhow::ensure!(diff < TOLERANCE, "{task} batched: max diff {diff}");
    }
    Ok(())
}

#[tokio::test]
async fn a_jina_v5_qwen3_base_embeds_each_task_as_its_merged_adapter() -> anyhow::Result<()> {
    let base = qwen3::tiny_embedding_checkpoint()?;
    let merged = as_jina_v5(base.path(), "Qwen3Model", |_| {})?;
    assert_tasks_embed_as_merged_adapters(base.path(), &merged).await
}

#[tokio::test]
async fn a_jina_v5_bidirectional_base_embeds_each_task_as_its_merged_adapter() -> anyhow::Result<()>
{
    let base = rope::tiny_llama_embedding_checkpoint(|_| {})?;
    let merged = as_jina_v5(base.path(), "LlamaBidirectionalModel", |config| {
        // EuroBERT's flag, which picks the bidirectional base
        config["is_decoder"] = false.into();
    })?;
    assert_tasks_embed_as_merged_adapters(base.path(), &merged).await
}

#[tokio::test]
async fn a_jina_v5_base_without_its_adapters_is_refused() -> anyhow::Result<()> {
    let base = qwen3::tiny_embedding_checkpoint()?;
    as_jina_v5(base.path(), "Qwen3Model", |_| {})?;
    std::fs::remove_dir_all(base.path().join("adapters").join(ADAPTERS[1]))?;
    let Err(error) = cpu_model(base.path()).await else {
        anyhow::bail!("a v5 base without its adapters loaded");
    };
    anyhow::ensure!(
        error.to_string().contains("task adapter is missing"),
        "{error}"
    );
    Ok(())
}
