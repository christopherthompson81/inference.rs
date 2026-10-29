//! Granite 3.x / 4.1 (dense, MoE, MoE with a shared expert) and Granite SWA on the Granite 4 model: a dense
//! checkpoint's `mlp` scores as its weights fused into `shared_mlp` do, the MoE classes generate, and SWA's
//! transformers v5 expert names score as Granite 4's.

use std::collections::HashMap;

use inference::{
    LogitsOutput, Model, ModelDType, PromptInput, PromptLogitsRequest, RequestBuilder,
    TextMessageRole, TextMessages, TextModelBuilder,
};
use inference_tensor::{DType, Device, Tensor};

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llama_tiny");
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
const FIXTURE_FILES: [&str; 3] = [
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer_config.json",
];
const VOCAB: usize = 266;
const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 48;
const SHARED_INTERMEDIATE: usize = 40;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const WEIGHT_STD: f32 = 0.2;
const PROMPT: [u32; 5] = [40, 41, 42, 43, 44];
const MAX_LEN: usize = 4;
const LOGIT_TOLERANCE: f32 = 1e-4;
// shorter than PROMPT
const SLIDING_WINDOW: usize = 2;

fn config(architecture: &str, model_type: &str, moe: bool, shared: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "architectures": [architecture],
        "model_type": model_type,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 512,
        "tie_word_embeddings": true,
        "embedding_multiplier": 12.0,
        "attention_multiplier": 0.0625,
        "residual_multiplier": 0.22,
        "logits_scaling": 8.0,
    });
    if moe {
        config["num_local_experts"] = EXPERTS.into();
        config["num_experts_per_tok"] = TOP_K.into();
    }
    if shared {
        config["shared_intermediate_size"] = SHARED_INTERMEDIATE.into();
    }
    config
}

fn random(shape: &[usize]) -> anyhow::Result<Tensor> {
    Ok(Tensor::randn(0f32, WEIGHT_STD, shape, &Device::Cpu)?)
}

// Everything but the feed-forward weights, which each test lays out its own way.
fn common_weights() -> anyhow::Result<HashMap<String, Tensor>> {
    let head_dim = HIDDEN / HEADS;
    let mut weights = HashMap::from([
        (
            "model.embed_tokens.weight".to_string(),
            random(&[VOCAB, HIDDEN])?,
        ),
        (
            "model.norm.weight".to_string(),
            Tensor::ones(HIDDEN, DType::F32, &Device::Cpu)?,
        ),
    ]);
    for layer in 0..LAYERS {
        let prefix = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            weights.insert(
                format!("{prefix}.{norm}.weight"),
                Tensor::ones(HIDDEN, DType::F32, &Device::Cpu)?,
            );
        }
        for (name, rows, cols) in [
            ("q_proj", HEADS * head_dim, HIDDEN),
            ("k_proj", KV_HEADS * head_dim, HIDDEN),
            ("v_proj", KV_HEADS * head_dim, HIDDEN),
            ("o_proj", HIDDEN, HEADS * head_dim),
        ] {
            weights.insert(
                format!("{prefix}.self_attn.{name}.weight"),
                random(&[rows, cols])?,
            );
        }
    }
    Ok(weights)
}

fn checkpoint(
    config: serde_json::Value,
    weights: &HashMap<String, Tensor>,
) -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for file in FIXTURE_FILES {
        std::fs::copy(format!("{FIXTURE}/{file}"), dir.path().join(file))?;
    }
    std::fs::copy(TOKENIZER, dir.path().join("tokenizer.json"))?;
    std::fs::write(dir.path().join("config.json"), config.to_string())?;
    inference_tensor::safetensors::save(weights, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

async fn load(dir: &tempfile::TempDir) -> anyhow::Result<Model> {
    Ok(TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?)
}

async fn prompt_logits(dir: &tempfile::TempDir) -> anyhow::Result<Vec<f32>> {
    let scored = load(dir)
        .await?
        .prompt_logits(PromptLogitsRequest {
            model: None,
            prompt: PromptInput::Tokens(PROMPT.to_vec()),
            output: LogitsOutput::Logits,
        })
        .await?;
    Ok(scored.logits.expect("the request asked for logits"))
}

async fn generates(dir: &tempfile::TempDir) -> anyhow::Result<()> {
    let request =
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, "hello"))
            .set_sampler_max_len(MAX_LEN)
            .set_sampler_topk(1);
    let response = load(dir).await?.send_chat_request(request).await?;
    anyhow::ensure!(
        response.usage.completion_tokens > 0,
        "the model generated nothing"
    );
    Ok(())
}

#[tokio::test]
async fn a_dense_granite_mlp_scores_as_its_fused_shared_mlp() -> anyhow::Result<()> {
    let common = common_weights()?;
    let (mut split, mut fused) = (common.clone(), common);
    for layer in 0..LAYERS {
        let prefix = format!("model.layers.{layer}");
        let gate = random(&[INTERMEDIATE, HIDDEN])?;
        let up = random(&[INTERMEDIATE, HIDDEN])?;
        let down = random(&[HIDDEN, INTERMEDIATE])?;
        fused.insert(
            format!("{prefix}.shared_mlp.input_linear.weight"),
            Tensor::cat(&[&gate, &up], 0)?,
        );
        fused.insert(
            format!("{prefix}.shared_mlp.output_linear.weight"),
            down.clone(),
        );
        split.insert(format!("{prefix}.mlp.gate_proj.weight"), gate);
        split.insert(format!("{prefix}.mlp.up_proj.weight"), up);
        split.insert(format!("{prefix}.mlp.down_proj.weight"), down);
    }
    let dense = checkpoint(
        config("GraniteForCausalLM", "granite", false, false),
        &split,
    )?;
    let hybrid = checkpoint(
        config(
            "GraniteMoeHybridForCausalLM",
            "granitemoehybrid",
            false,
            false,
        ),
        &fused,
    )?;

    let (dense, hybrid) = (prompt_logits(&dense).await?, prompt_logits(&hybrid).await?);
    assert_eq!(dense.len(), hybrid.len());
    let worst = dense
        .iter()
        .zip(&hybrid)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < LOGIT_TOLERANCE, "logits differ by {worst}");
    assert!(dense.iter().any(|logit| logit.abs() > LOGIT_TOLERANCE));
    Ok(())
}

#[tokio::test]
async fn granite_moe_checkpoints_with_and_without_a_shared_expert_generate() -> anyhow::Result<()> {
    for (architecture, model_type, shared) in [
        ("GraniteMoeForCausalLM", "granitemoe", false),
        ("GraniteMoeSharedForCausalLM", "granitemoeshared", true),
    ] {
        let mut weights = common_weights()?;
        for layer in 0..LAYERS {
            let prefix = format!("model.layers.{layer}");
            let moe = format!("{prefix}.block_sparse_moe");
            weights.insert(
                format!("{moe}.input_linear.weight"),
                random(&[EXPERTS, 2 * INTERMEDIATE, HIDDEN])?,
            );
            weights.insert(
                format!("{moe}.output_linear.weight"),
                random(&[EXPERTS, HIDDEN, INTERMEDIATE])?,
            );
            weights.insert(
                format!("{moe}.router.layer.weight"),
                random(&[EXPERTS, HIDDEN])?,
            );
            if shared {
                weights.insert(
                    format!("{prefix}.shared_mlp.input_linear.weight"),
                    random(&[2 * SHARED_INTERMEDIATE, HIDDEN])?,
                );
                weights.insert(
                    format!("{prefix}.shared_mlp.output_linear.weight"),
                    random(&[HIDDEN, SHARED_INTERMEDIATE])?,
                );
            }
        }
        let dir = checkpoint(config(architecture, model_type, true, shared), &weights)?;
        generates(&dir).await?;
    }
    Ok(())
}

#[tokio::test]
async fn granite_swa_v5_expert_names_score_as_granite_4_names() -> anyhow::Result<()> {
    let mut v5 = common_weights()?;
    let mut granite4 = v5.clone();
    for layer in 0..LAYERS {
        let prefix = format!("model.layers.{layer}");
        let sinks = random(&[HEADS])?;
        v5.insert(format!("{prefix}.self_attn.sinks"), sinks.clone());
        granite4.insert(format!("{prefix}.self_attn.sinks"), sinks);
        let gate_up = random(&[EXPERTS, 2 * INTERMEDIATE, HIDDEN])?;
        let down = random(&[EXPERTS, HIDDEN, INTERMEDIATE])?;
        let router = random(&[EXPERTS, HIDDEN])?;
        let moe = format!("{prefix}.block_sparse_moe");
        v5.insert(format!("{moe}.experts.gate_up_proj"), gate_up.clone());
        v5.insert(format!("{moe}.experts.down_proj"), down.clone());
        v5.insert(format!("{moe}.router.weight"), router.clone());
        granite4.insert(format!("{moe}.input_linear.weight"), gate_up);
        granite4.insert(format!("{moe}.output_linear.weight"), down);
        granite4.insert(format!("{moe}.router.layer.weight"), router);
    }
    let mut swa = config("GraniteMoeSWAForCausalLM", "granitemoe_swa", true, false);
    swa["sliding_window"] = SLIDING_WINDOW.into();
    swa["layer_types"] = serde_json::json!(["full_attention", "sliding_attention"]);
    let mut renamed = swa.clone();
    renamed["architectures"] = serde_json::json!(["GraniteMoeForCausalLM"]);
    renamed["model_type"] = "granitemoe".into();
    // the same layer kinds, under the names Granite 4 reads
    renamed["layer_types"] = serde_json::json!(["attention", "sliding_attention"]);

    let full = {
        let mut full = swa.clone();
        full["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
        checkpoint(full, &v5)?
    };
    let (v5, granite4) = (checkpoint(swa, &v5)?, checkpoint(renamed, &granite4)?);
    let (v5, granite4, full) = (
        prompt_logits(&v5).await?,
        prompt_logits(&granite4).await?,
        prompt_logits(&full).await?,
    );
    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    };
    assert!(
        worst(&v5, &granite4) < LOGIT_TOLERANCE,
        "logits differ by {}",
        worst(&v5, &granite4)
    );
    // the prompt outruns the window, so the sliding layer must see less than full attention does
    assert!(worst(&v5, &full) > LOGIT_TOLERANCE);
    Ok(())
}
