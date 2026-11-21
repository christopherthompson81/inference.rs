//! A tiny random-weight Inkling 2 (sliding and global layers with relative-position biases and per-query sinks, short
//! convolutions with their state, block attention residuals, a looped core run three times under its controllers,
//! softcapped SwiGLU experts and a padded vocabulary): decoding step by step through the KV and convolution states of
//! every pass scores the tokens as one prefill over them does, on the CPU and on Vulkan, and a prompt past the window
//! and the bias's extent scores on Vulkan as on the CPU.

use std::collections::HashMap;

use inference::{
    ChatRequest, ModelDType, PromptInput, PromptLogitsRequest, RequestBuilder, TextMessageRole,
    TextMessages, TextModelBuilder,
};
use inference_tensor::{DType, Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

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
const VOCAB: usize = 266;
// the embedding and head rows past the tokenizer's, as the checkpoints pad them
const PADDED_VOCAB: usize = 272;
const HIDDEN: usize = 32;
const LAYERS: usize = 6;
const BLOCK: usize = 2;
// the looped core, layers 2 and 3, run three times
const CORE: (usize, usize) = (2, 4);
const PASSES: usize = 3;
const GATE_RANK: usize = 8;
const SLIDING: [usize; 4] = [0, 1, 3, 4];
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
const D_REL: usize = 4;
// short, so the chat prompt runs past both
const REL_EXTENT: usize = 8;
const WINDOW: usize = 4;
const CONV: usize = 4;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const EXPERT_WIDTH: usize = 32;
const WEIGHT_STD: f32 = 0.2;
const WEIGHT_SEED: u64 = 0x696e_6b6c;
const MAX_LEN: usize = 6;
const LOGPROB_TOLERANCE: f32 = 1e-4;
// Vulkan's ops sum in other orders than the CPU's
const VULKAN_LOGPROB_TOLERANCE: f32 = 1e-3;
const LONG_PROMPT_REPEATS: usize = 6;

fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Inkling2LoopForCausalLM"],
        "model_type": "inkling2_loop_text",
        "vocab_size": PADDED_VOCAB,
        "unpadded_vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "d_rel": D_REL,
        "rel_extent": REL_EXTENT,
        // a floor the prompt passes, so the global layers' log scale moves
        "log_scaling_n_floor": 8.0,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 4.0,
        "rel_l2_clip_threshold": 2.0,
        "local_layer_ids": SLIDING,
        "sconv_kernel_size": CONV,
        "final_logit_softcapping": 30.0,
        "sliding_window_size": WINDOW,
        "n_layers_per_attn_res_block": BLOCK,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "intermediate_size": EXPERT_WIDTH,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "loop_config": {
            "core_start": CORE.0,
            "core_end": CORE.1,
            "num_passes": PASSES,
            "gate_rank": GATE_RANK,
        },
        "max_position_embeddings": 4096,
    })
}

fn random(rng: &mut StdRng, shape: &[usize]) -> anyhow::Result<Tensor> {
    let normal = Normal::new(0f32, WEIGHT_STD)?;
    let count = shape.iter().product::<usize>();
    let data = (0..count).map(|_| normal.sample(rng)).collect::<Vec<_>>();
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?)
}

fn near_one(rng: &mut StdRng, shape: &[usize]) -> anyhow::Result<Tensor> {
    Ok((random(rng, shape)? + 1.0)?)
}

fn weights() -> anyhow::Result<HashMap<String, Tensor>> {
    let rng = &mut StdRng::seed_from_u64(WEIGHT_SEED);
    let (q_dim, kv_dim) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let mut w = HashMap::new();
    let mut put = |name: String, t: Tensor| {
        w.insert(name, t);
    };
    put(
        "model.embed_tokens.weight".into(),
        random(rng, &[PADDED_VOCAB, HIDDEN])?,
    );
    put("model.embed_norm.weight".into(), near_one(rng, &[HIDDEN])?);
    put("model.norm.weight".into(), near_one(rng, &[HIDDEN])?);
    put("model.final_input.query".into(), random(rng, &[1, HIDDEN])?);
    put(
        "lm_head.weight".into(),
        random(rng, &[PADDED_VOCAB, HIDDEN])?,
    );
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        let a = format!("{p}.self_attn");
        let sliding = SLIDING.contains(&i);
        if i > 0 {
            let queries = if sliding { 1 } else { 3 };
            put(
                format!("{p}.attention_input.query"),
                random(rng, &[queries, HIDDEN])?,
            );
        }
        put(format!("{p}.mlp_input.query"), random(rng, &[1, HIDDEN])?);
        if sliding {
            put(
                format!("{p}.input_layernorm.weight"),
                near_one(rng, &[HIDDEN])?,
            );
        } else {
            for norm in ["q", "k", "v"] {
                put(
                    format!("{p}.{norm}_input_layernorm.weight"),
                    near_one(rng, &[HIDDEN])?,
                );
            }
            for conv in ["k_sconv", "v_sconv"] {
                put(
                    format!("{a}.{conv}.weight_proj.weight"),
                    random(rng, &[KV_HEADS * CONV, HIDDEN])?,
                );
            }
        }
        for (name, rows, cols) in [
            ("q_proj", q_dim, HIDDEN),
            ("k_proj", kv_dim, HIDDEN),
            ("v_proj", kv_dim, HIDDEN),
            ("gate_proj", q_dim, HIDDEN),
            ("r_proj", HEADS * D_REL, HIDDEN),
            ("sink_proj", HEADS, HIDDEN),
            ("o_proj", HIDDEN, q_dim),
        ] {
            put(format!("{a}.{name}.weight"), random(rng, &[rows, cols])?);
        }
        let extent = if sliding { WINDOW } else { REL_EXTENT };
        put(
            format!("{a}.q_l2_clip.scale"),
            near_one(rng, &[HEADS, HEAD_DIM])?,
        );
        put(
            format!("{a}.rel_l2_clip.scale"),
            near_one(rng, &[HEADS, D_REL])?,
        );
        put(
            format!("{a}.rel_logits_proj.proj"),
            random(rng, &[D_REL, extent])?,
        );
        put(format!("{a}.k_sconv.weight"), random(rng, &[kv_dim, CONV])?);
        put(format!("{a}.v_sconv.weight"), random(rng, &[kv_dim, CONV])?);
        put(
            format!("{p}.post_attention_convolution.weight"),
            random(rng, &[HIDDEN, CONV])?,
        );
        put(
            format!("{p}.post_mlp_convolution.weight"),
            random(rng, &[HIDDEN, CONV])?,
        );
        put(
            format!("{p}.post_attention_layernorm.weight"),
            near_one(rng, &[HIDDEN])?,
        );
        put(
            format!("{p}.mlp.gate.weight"),
            random(rng, &[EXPERTS, HIDDEN])?,
        );
        put(
            format!("{p}.mlp.gate.e_score_correction_bias"),
            random(rng, &[EXPERTS])?,
        );
        for e in 0..EXPERTS {
            let x = format!("{p}.mlp.experts.{e}");
            put(
                format!("{x}.gate_proj.weight"),
                random(rng, &[EXPERT_WIDTH, HIDDEN])?,
            );
            put(
                format!("{x}.up_proj.weight"),
                random(rng, &[EXPERT_WIDTH, HIDDEN])?,
            );
            put(
                format!("{x}.down_proj.weight"),
                random(rng, &[HIDDEN, EXPERT_WIDTH])?,
            );
        }
        if (CORE.0..CORE.1).contains(&i) {
            put(format!("{p}.loop.msg_query"), random(rng, &[4, HIDDEN])?);
            for c in ["q", "k", "v", "ffn"] {
                let l = format!("{p}.loop.{c}");
                put(
                    format!("{l}.down.weight"),
                    random(rng, &[GATE_RANK, 2 * HIDDEN])?,
                );
                put(
                    format!("{l}.gain.weight"),
                    random(rng, &[HIDDEN, GATE_RANK])?,
                );
                put(
                    format!("{l}.forget.weight"),
                    random(rng, &[HIDDEN, GATE_RANK])?,
                );
            }
        }
    }
    Ok(w)
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

fn builder(dir: &tempfile::TempDir, device: Option<&str>) -> TextModelBuilder {
    // no prefix cache, so a scoring request is one prefill rather than a suffix over the decode steps' state
    let builder = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_prefix_cache_n(None);
    match device {
        Some(device) => builder.with_device(device),
        None => builder.with_force_cpu(),
    }
}

fn chat(prompt: &str) -> RequestBuilder {
    RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, prompt))
}

async fn prompt_logprobs(model: &inference::Model, tokens: Vec<u32>) -> anyhow::Result<Vec<f32>> {
    Ok(model
        .prompt_logits(PromptLogitsRequest {
            reference: None,
            model: None,
            prompt: PromptInput::Tokens(tokens),
            output: Default::default(),
            audio: Vec::new(),
        })
        .await?
        .token_logprobs
        .into_iter()
        .map(|logprob| logprob.unwrap_or(f32::NAN))
        .collect())
}

async fn decode_scores_as_prefill(device: Option<&str>, tolerance: f32) -> anyhow::Result<()> {
    let dir = checkpoint()?;
    let model = builder(&dir, device).build().await?;
    let prompt = "say something long";
    let (decoded, _) = traces::greedy(&model, chat(prompt), MAX_LEN).await?;
    let mut tokens = model
        .tokenize_chat(ChatRequest::from(chat(prompt)).request)
        .await?
        .tokens;
    anyhow::ensure!(
        tokens.len() > REL_EXTENT,
        "the prompt stops short of the bias's extent"
    );
    tokens.extend(decoded.iter().map(|(token, _)| *token));
    let scored = prompt_logprobs(&model, tokens).await?;
    let prefilled = &scored[scored.len() - decoded.len()..];
    for (step, ((_, decoded), prefilled)) in decoded.iter().zip(prefilled).enumerate() {
        anyhow::ensure!(
            (decoded - prefilled).abs() < tolerance,
            "step {step}: decoded {decoded}, one prefill {prefilled}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_step_by_step_decode_scores_as_one_prefill() -> anyhow::Result<()> {
    decode_scores_as_prefill(None, LOGPROB_TOLERANCE).await
}

// f32 throughout: Vulkan's prompt matmuls on matrix cores take f16 inputs
fn no_matrix_cores() {
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_tensor::vulkan::NO_MATRIX_CORES_ENV, "1") };
}

#[tokio::test]
async fn a_vulkan_decode_scores_as_one_prefill() -> anyhow::Result<()> {
    no_matrix_cores();
    if !cfg!(feature = "vulkan") || Device::new_vulkan(0).is_err() {
        return Ok(());
    }
    decode_scores_as_prefill(Some("vulkan:0"), VULKAN_LOGPROB_TOLERANCE).await
}

#[tokio::test]
async fn a_long_prompt_scores_on_vulkan_as_on_the_cpu() -> anyhow::Result<()> {
    no_matrix_cores();
    if !cfg!(feature = "vulkan") || Device::new_vulkan(0).is_err() {
        return Ok(());
    }
    // the CPU stores KV in f16 by default and Vulkan in f32
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_nn::kv_cache::CPU_KV_F32_ENV, "1") };
    let dir = checkpoint()?;
    let prompt = "say something long ".repeat(LONG_PROMPT_REPEATS);
    let mut scores = Vec::new();
    for device in [None, Some("vulkan:0")] {
        let model = builder(&dir, device).build().await?;
        let tokens = model
            .tokenize_chat(ChatRequest::from(chat(&prompt)).request)
            .await?
            .tokens;
        scores.push(prompt_logprobs(&model, tokens).await?);
    }
    for (position, (cpu, vulkan)) in scores[0].iter().zip(&scores[1]).enumerate().skip(1) {
        anyhow::ensure!(
            (cpu - vulkan).abs() < VULKAN_LOGPROB_TOLERANCE,
            "position {position}: CPU {cpu}, Vulkan {vulkan}"
        );
    }
    Ok(())
}
