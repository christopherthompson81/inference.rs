//! A tiny random-weight Nemotron-H (Mamba-2, NoPE attention, sigmoid-routed squared-ReLU experts with a shared
//! expert, a dense squared-ReLU MLP, and a Mamba layer with no feed-forward after it): decoding step by step through
//! its Mamba state and KV cache scores the tokens as one prefill over them does (quantized, with a prompt long enough
//! to run the experts grouped, too, and on CUDA), and calibrating an ISQ load collects statistics for every
//! expert projection.

use std::collections::HashMap;

use inference::{
    ChatRequest, IsqType, ModelDType, PromptInput, PromptLogitsRequest, RequestBuilder,
    TextMessageRole, TextMessages, TextModelBuilder,
};
use inference_tensor::{DType, Device, Tensor};

#[path = "../support/traces.rs"]
mod traces;

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
// Mamba+MoE, Mamba alone, attention+MoE, Mamba+MLP
const LAYERS: [&str; 7] = ["mamba", "moe", "mamba", "attention", "moe", "mamba", "mlp"];
const VOCAB: usize = 266;
const HIDDEN: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
const MAMBA_HEADS: usize = 4;
const MAMBA_HEAD_DIM: usize = 8;
const GROUPS: usize = 2;
const STATE: usize = 8;
const CONV: usize = 4;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
// enough prompt tokens for the CPU expert kernel to decode an expert's rows into a tile
const LONG_PROMPT_REPEATS: usize = 12;
// whole Q8_0 blocks, so a quantized load runs its experts from their blocks on the CPU
const EXPERT_WIDTH: usize = 32;
const SHARED_WIDTH: usize = 40;
const MLP_WIDTH: usize = 48;
const WEIGHT_STD: f32 = 0.2;
const MAX_LEN: usize = 6;
const LOGPROB_TOLERANCE: f32 = 1e-4;
// cuBLAS and the paged attention kernels add up in other orders than the CPU's
const CUDA_LOGPROB_TOLERANCE: f32 = 1e-3;
const CIMATRIX: &str = "collected.cimatrix";

fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "layers_block_type": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": MLP_WIDTH,
        "mlp_hidden_act": "relu2",
        "ssm_state_size": STATE,
        "mamba_num_heads": MAMBA_HEADS,
        "mamba_head_dim": MAMBA_HEAD_DIM,
        "n_groups": GROUPS,
        "conv_kernel": CONV,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": EXPERT_WIDTH,
        "moe_shared_expert_intermediate_size": SHARED_WIDTH,
        "routed_scaling_factor": 2.5,
        "layer_norm_epsilon": 1e-5,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": false,
    })
}

fn random(shape: &[usize]) -> anyhow::Result<Tensor> {
    Ok(Tensor::randn(0f32, WEIGHT_STD, shape, &Device::Cpu)?)
}

fn near_one(shape: &[usize]) -> anyhow::Result<Tensor> {
    Ok((Tensor::randn(0f32, WEIGHT_STD, shape, &Device::Cpu)? + 1.0)?)
}

// NVIDIA's `backbone.*` names, as the Hub checkpoints keep them
fn weights() -> anyhow::Result<HashMap<String, Tensor>> {
    let inner = MAMBA_HEADS * MAMBA_HEAD_DIM;
    let conv_dim = inner + 2 * GROUPS * STATE;
    let mut weights = HashMap::from([
        (
            "backbone.embeddings.weight".to_string(),
            random(&[VOCAB, HIDDEN])?,
        ),
        ("backbone.norm_f.weight".to_string(), near_one(&[HIDDEN])?),
        ("lm_head.weight".to_string(), random(&[VOCAB, HIDDEN])?),
    ]);
    for (i, kind) in LAYERS.iter().enumerate() {
        let layer = format!("backbone.layers.{i}");
        weights.insert(format!("{layer}.norm.weight"), near_one(&[HIDDEN])?);
        let mixer = format!("{layer}.mixer");
        let mut tensors: Vec<(String, Tensor)> = match *kind {
            "mamba" => vec![
                (
                    "in_proj.weight".into(),
                    random(&[inner + conv_dim + MAMBA_HEADS, HIDDEN])?,
                ),
                ("conv1d.weight".into(), random(&[conv_dim, 1, CONV])?),
                ("conv1d.bias".into(), random(&[conv_dim])?),
                ("dt_bias".into(), random(&[MAMBA_HEADS])?),
                ("A_log".into(), random(&[MAMBA_HEADS])?),
                ("D".into(), near_one(&[MAMBA_HEADS])?),
                ("norm.weight".into(), near_one(&[inner])?),
                ("out_proj.weight".into(), random(&[HIDDEN, inner])?),
            ],
            "attention" => vec![
                ("q_proj.weight".into(), random(&[HEADS * HEAD_DIM, HIDDEN])?),
                (
                    "k_proj.weight".into(),
                    random(&[KV_HEADS * HEAD_DIM, HIDDEN])?,
                ),
                (
                    "v_proj.weight".into(),
                    random(&[KV_HEADS * HEAD_DIM, HIDDEN])?,
                ),
                ("o_proj.weight".into(), random(&[HIDDEN, HEADS * HEAD_DIM])?),
            ],
            "moe" => {
                let mut moe = vec![
                    ("gate.weight".into(), random(&[EXPERTS, HIDDEN])?),
                    ("gate.e_score_correction_bias".into(), random(&[EXPERTS])?),
                    (
                        "shared_experts.up_proj.weight".into(),
                        random(&[SHARED_WIDTH, HIDDEN])?,
                    ),
                    (
                        "shared_experts.down_proj.weight".into(),
                        random(&[HIDDEN, SHARED_WIDTH])?,
                    ),
                ];
                for e in 0..EXPERTS {
                    moe.push((
                        format!("experts.{e}.up_proj.weight"),
                        random(&[EXPERT_WIDTH, HIDDEN])?,
                    ));
                    moe.push((
                        format!("experts.{e}.down_proj.weight"),
                        random(&[HIDDEN, EXPERT_WIDTH])?,
                    ));
                }
                moe
            }
            _ => vec![
                ("up_proj.weight".into(), random(&[MLP_WIDTH, HIDDEN])?),
                ("down_proj.weight".into(), random(&[HIDDEN, MLP_WIDTH])?),
            ],
        };
        for (name, tensor) in tensors.drain(..) {
            weights.insert(format!("{mixer}.{name}"), tensor);
        }
    }
    Ok(weights)
}

fn checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for file in FIXTURE_FILES {
        std::fs::copy(format!("{FIXTURE}/{file}"), dir.path().join(file))?;
    }
    std::fs::copy(TOKENIZER, dir.path().join("tokenizer.json"))?;
    std::fs::write(dir.path().join("config.json"), config().to_string())?;
    let weights = weights()?
        .into_iter()
        .map(|(name, tensor)| Ok((name, tensor.to_dtype(DType::F32)?)))
        .collect::<anyhow::Result<HashMap<_, _>>>()?;
    inference_tensor::safetensors::save(&weights, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

#[tokio::test]
async fn a_step_by_step_decode_scores_as_one_prefill() -> anyhow::Result<()> {
    decode_scores_as_prefill(None, "say something long", true).await
}

// quantized experts on the CPU run from their blocks, a long prompt's tiled and a decode step's per row
#[tokio::test]
async fn a_long_prompt_on_quantized_experts_scores_as_its_decode() -> anyhow::Result<()> {
    decode_scores_as_prefill(
        Some(IsqType::Q8_0),
        &"say something long ".repeat(LONG_PROMPT_REPEATS),
        true,
    )
    .await
}

// on CUDA through the paged hybrid cache and the scan kernel at a state narrower than a warp (grouped mmq against the
// gather path is a GPU case of the experts' backends: quantized, prefill and decode differ by the activations' rounding)
#[tokio::test]
async fn a_long_prompt_on_cuda_scores_as_its_decode() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    decode_scores_as_prefill(
        None,
        &"say something long ".repeat(LONG_PROMPT_REPEATS),
        false,
    )
    .await
}

async fn decode_scores_as_prefill(
    isq: Option<IsqType>,
    prompt: &str,
    cpu: bool,
) -> anyhow::Result<()> {
    let dir = checkpoint()?;
    let mut builder =
        TextModelBuilder::new(dir.path().to_string_lossy()).with_dtype(ModelDType::F32);
    if cpu {
        builder = builder.with_force_cpu();
    }
    if let Some(isq) = isq {
        builder = builder.with_isq(isq);
    }
    let model = builder.build().await?;
    let chat =
        || RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, prompt));
    let (decoded, _) = traces::greedy(&model, chat(), MAX_LEN).await?;
    let mut tokens = model
        .tokenize_chat(ChatRequest::from(chat()).request)
        .await?
        .tokens;
    tokens.extend(decoded.iter().map(|(token, _)| *token));
    let scored = model
        .prompt_logits(PromptLogitsRequest {
            model: None,
            prompt: PromptInput::Tokens(tokens),
            output: Default::default(),
            audio: Vec::new(),
        })
        .await?
        .token_logprobs;
    let prefilled = scored[scored.len() - decoded.len()..]
        .iter()
        .map(|logprob| logprob.unwrap_or(f32::NAN));
    for (step, ((_, decoded), prefilled)) in decoded.iter().zip(prefilled).enumerate() {
        anyhow::ensure!(
            (decoded - prefilled).abs()
                < if cpu {
                    LOGPROB_TOLERANCE
                } else {
                    CUDA_LOGPROB_TOLERANCE
                },
            "step {step}: decoded {decoded}, one prefill {prefilled}"
        );
    }
    Ok(())
}

// the experts' up and down stacks are tracked as one module each per MoE layer, under the canonical expert keys
#[tokio::test]
async fn calibrating_an_isq_load_collects_every_expert_projection() -> anyhow::Result<()> {
    let dir = checkpoint()?;
    let model = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_isq(IsqType::Q8_0)
        .build()
        .await?;
    model.calibration_start_json(b"{}").await?;
    let chat = RequestBuilder::from(
        TextMessages::new().add_message(TextMessageRole::User, "say something long"),
    );
    traces::greedy(&model, chat, MAX_LEN).await?;
    let saved = dir.path().join(CIMATRIX);
    model
        .calibration_apply_json(
            serde_json::json!({ "save_cimatrix": saved })
                .to_string()
                .as_bytes(),
        )
        .await?;
    let collected = inference_quant::CollectedImatrixData::load_imatrix(&saved)?.0;
    for (i, kind) in LAYERS.iter().enumerate() {
        if *kind != "moe" {
            continue;
        }
        for projection in ["up_proj", "down_proj"] {
            let key = format!("backbone.layers.{i}.mixer.experts.{projection}");
            let stats = collected
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("no statistics for `{key}`"))?;
            anyhow::ensure!(
                stats.iter().any(|v| *v > 0.0),
                "`{key}` collected only zeros"
            );
        }
    }
    Ok(())
}
