//! Tiny random-weight Mamba stacks with tied embeddings: a Mamba-2 (two groups), a Falcon-Mamba (Mamba 1's selection
//! with weightless dt, B and C norms, A per channel and state item) and a Jamba (Mamba 1 with weighted norms around a
//! NoPE attention layer, softmax-routed SwiGLU experts beside a dense MLP). Decoding step by step through the state
//! scores the tokens as one prefill over them does, a repeated prompt resumes from the prefix cache to the same text,
//! and a long prompt (Mamba-2's chunked scan, Mamba 1's sequential one) scores on CUDA as on the CPU, its decode (a
//! replayed CUDA graph) scoring as a CPU prefill does.

use std::collections::HashMap;

use inference::{
    ChatRequest, ModelDType, PromptInput, PromptLogitsRequest, RequestBuilder, TextMessageRole,
    TextMessages, TextModelBuilder,
};
use inference_tensor::{DType, Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

#[macro_use]
#[path = "../support/concurrent_cases.rs"]
mod concurrent_cases;
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
const HIDDEN: usize = 32;
const LAYERS: usize = 3;
const HEADS: usize = 8;
const HEAD_DIM: usize = 8;
const INNER: usize = HEADS * HEAD_DIM;
const GROUPS: usize = 2;
const STATE: usize = 16;
const CONV: usize = 4;
const CONV_DIM: usize = INNER + 2 * GROUPS * STATE;
const PROJECTION: usize = INNER + CONV_DIM + HEADS;
const DT_RANK: usize = 4;
const ATTENTION_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const ATTENTION_HEAD_DIM: usize = HIDDEN / ATTENTION_HEADS;
const FFN: usize = 48;
const EXPERTS: usize = 4;
const EXPERTS_PER_TOKEN: usize = 2;
// (period, offset): layer 1 attention, layers 0 and 2 experts
const JAMBA_ATTENTION: (usize, usize) = (3, 1);
const JAMBA_EXPERTS: (usize, usize) = (2, 0);
const WEIGHT_STD: f32 = 0.2;
const WEIGHT_SEED: u64 = 23;
const MAX_LEN: usize = 8;
const LOGPROB_TOLERANCE: f32 = 1e-3;
const CUDA_LOGPROB_TOLERANCE: f32 = 5e-3;
// past the 128 tokens from which CUDA takes the chunked scan
const LONG_PROMPT_REPEATS: usize = 12;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Mamba2,
    FalconMamba,
    Jamba,
}

const KINDS: [Kind; 3] = [Kind::Mamba2, Kind::FalconMamba, Kind::Jamba];

// each weight's name and shape, as the kind's checkpoint names them
fn weight_shapes(kind: Kind) -> Vec<(String, Vec<usize>, bool)> {
    let (root, embeddings, final_norm) = match kind {
        Kind::Jamba => ("model", "embed_tokens", "final_layernorm"),
        _ => ("backbone", "embeddings", "norm_f"),
    };
    // (name, shape, all ones)
    let mut shapes = vec![
        (
            format!("{root}.{embeddings}.weight"),
            vec![VOCAB, HIDDEN],
            false,
        ),
        (format!("{root}.{final_norm}.weight"), vec![HIDDEN], true),
    ];
    let mlp = |shapes: &mut Vec<(String, Vec<usize>, bool)>, p: &str| {
        shapes.push((format!("{p}.gate_proj.weight"), vec![FFN, HIDDEN], false));
        shapes.push((format!("{p}.up_proj.weight"), vec![FFN, HIDDEN], false));
        shapes.push((format!("{p}.down_proj.weight"), vec![HIDDEN, FFN], false));
    };
    for layer in 0..LAYERS {
        let l = format!("{root}.layers.{layer}");
        let mixer = match kind {
            Kind::Jamba => {
                shapes.push((format!("{l}.input_layernorm.weight"), vec![HIDDEN], true));
                shapes.push((format!("{l}.pre_ff_layernorm.weight"), vec![HIDDEN], true));
                let ffn = format!("{l}.feed_forward");
                if layer % JAMBA_EXPERTS.0 == JAMBA_EXPERTS.1 {
                    shapes.push((format!("{ffn}.router.weight"), vec![EXPERTS, HIDDEN], false));
                    for expert in 0..EXPERTS {
                        mlp(&mut shapes, &format!("{ffn}.experts.{expert}"));
                    }
                } else {
                    mlp(&mut shapes, &ffn);
                }
                if layer % JAMBA_ATTENTION.0 == JAMBA_ATTENTION.1 {
                    let a = format!("{l}.self_attn");
                    let kv = KV_HEADS * ATTENTION_HEAD_DIM;
                    for (name, rows, cols) in [
                        ("q_proj", HIDDEN, HIDDEN),
                        ("k_proj", kv, HIDDEN),
                        ("v_proj", kv, HIDDEN),
                        ("o_proj", HIDDEN, HIDDEN),
                    ] {
                        shapes.push((format!("{a}.{name}.weight"), vec![rows, cols], false));
                    }
                    continue;
                }
                format!("{l}.mamba")
            }
            _ => {
                shapes.push((format!("{l}.norm.weight"), vec![HIDDEN], true));
                format!("{l}.mixer")
            }
        };
        let p = mixer;
        shapes.push((format!("{p}.out_proj.weight"), vec![HIDDEN, INNER], false));
        match kind {
            Kind::Mamba2 => {
                for (name, shape, ones) in [
                    ("in_proj.weight", vec![PROJECTION, HIDDEN], false),
                    ("conv1d.weight", vec![CONV_DIM, 1, CONV], false),
                    ("conv1d.bias", vec![CONV_DIM], false),
                    ("dt_bias", vec![HEADS], false),
                    ("A_log", vec![HEADS], false),
                    ("D", vec![HEADS], true),
                    ("norm.weight", vec![INNER], true),
                ] {
                    shapes.push((format!("{p}.{name}"), shape, ones));
                }
            }
            Kind::FalconMamba | Kind::Jamba => {
                for (name, shape, ones) in [
                    ("in_proj.weight", vec![2 * INNER, HIDDEN], false),
                    ("conv1d.weight", vec![INNER, 1, CONV], false),
                    ("conv1d.bias", vec![INNER], false),
                    ("x_proj.weight", vec![DT_RANK + 2 * STATE, INNER], false),
                    ("dt_proj.weight", vec![INNER, DT_RANK], false),
                    ("dt_proj.bias", vec![INNER], false),
                    ("A_log", vec![INNER, STATE], false),
                    ("D", vec![INNER], true),
                ] {
                    shapes.push((format!("{p}.{name}"), shape, ones));
                }
                if let Kind::Jamba = kind {
                    for (name, n) in [
                        ("dt_layernorm", DT_RANK),
                        ("b_layernorm", STATE),
                        ("c_layernorm", STATE),
                    ] {
                        shapes.push((format!("{p}.{name}.weight"), vec![n], false));
                    }
                }
            }
        }
    }
    shapes
}

fn checkpoint(kind: Kind) -> anyhow::Result<tempfile::TempDir> {
    let rng = &mut StdRng::seed_from_u64(WEIGHT_SEED);
    let normal = Normal::new(0f32, WEIGHT_STD)?;
    let mut weights = HashMap::new();
    for (name, shape, ones) in weight_shapes(kind) {
        let tensor = if ones {
            Tensor::ones(shape.as_slice(), DType::F32, &Device::Cpu)?
        } else {
            let data = (0..shape.iter().product::<usize>())
                .map(|_| normal.sample(rng))
                .collect::<Vec<_>>();
            Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
        };
        weights.insert(name, tensor);
    }
    let config = match kind {
        Kind::Mamba2 => serde_json::json!({
            "architectures": ["Mamba2ForCausalLM"],
            "model_type": "mamba2",
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
            "expand": INNER / HIDDEN,
            "n_groups": GROUPS,
            "chunk_size": 256,
            "state_size": STATE,
            "conv_kernel": CONV,
            "layer_norm_epsilon": 1e-5,
        }),
        Kind::FalconMamba => serde_json::json!({
            "architectures": ["FalconMambaForCausalLM"],
            "model_type": "falcon_mamba",
            "intermediate_size": INNER,
            "time_step_rank": DT_RANK,
            "state_size": STATE,
            "conv_kernel": CONV,
            "layer_norm_epsilon": 1e-5,
        }),
        Kind::Jamba => serde_json::json!({
            "architectures": ["JambaForCausalLM"],
            "model_type": "jamba",
            "intermediate_size": FFN,
            "num_attention_heads": ATTENTION_HEADS,
            "num_key_value_heads": KV_HEADS,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "attn_layer_period": JAMBA_ATTENTION.0,
            "attn_layer_offset": JAMBA_ATTENTION.1,
            "expert_layer_period": JAMBA_EXPERTS.0,
            "expert_layer_offset": JAMBA_EXPERTS.1,
            "num_experts": EXPERTS,
            "num_experts_per_tok": EXPERTS_PER_TOKEN,
            "mamba_d_state": STATE,
            "mamba_d_conv": CONV,
            "mamba_expand": INNER / HIDDEN,
            "mamba_dt_rank": DT_RANK,
        }),
    };
    let mut config = config.as_object().cloned().expect("an object");
    for (key, value) in [
        ("vocab_size", serde_json::json!(VOCAB)),
        ("hidden_size", serde_json::json!(HIDDEN)),
        ("num_hidden_layers", serde_json::json!(LAYERS)),
        ("tie_word_embeddings", serde_json::json!(true)),
    ] {
        config.insert(key.to_string(), value);
    }
    let dir = tempfile::tempdir()?;
    for file in FIXTURE_FILES {
        std::fs::copy(format!("{FIXTURE}/{file}"), dir.path().join(file))?;
    }
    std::fs::copy(TOKENIZER, dir.path().join("tokenizer.json"))?;
    // as Python's json writes Mamba-2's default dt limit
    let mut text = serde_json::Value::Object(config).to_string();
    if matches!(kind, Kind::Mamba2) {
        text = text.replace(
            r#""tie_word_embeddings""#,
            r#""time_step_limit":[0.0,Infinity],"tie_word_embeddings""#,
        );
    }
    std::fs::write(dir.path().join("config.json"), text)?;
    inference_tensor::safetensors::save(&weights, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

fn chat(prompt: &str) -> RequestBuilder {
    RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, prompt))
}

async fn scores(model: &inference::Model, tokens: Vec<u32>) -> anyhow::Result<Vec<Option<f32>>> {
    let request = PromptLogitsRequest {
        model: None,
        prompt: PromptInput::Tokens(tokens),
        output: Default::default(),
        audio: Vec::new(),
    };
    Ok(model.prompt_logits(request).await?.token_logprobs)
}

async fn a_step_by_step_decode_scores_as_one_prefill() -> anyhow::Result<()> {
    for kind in KINDS {
        step_by_step_decode_scores_as_one_prefill(kind).await?;
    }
    Ok(())
}

async fn step_by_step_decode_scores_as_one_prefill(kind: Kind) -> anyhow::Result<()> {
    let dir = checkpoint(kind)?;
    // no prefix cache, so the scoring request is one prefill rather than a suffix over the decode steps' state
    let model = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_prefix_cache_n(None)
        .build()
        .await?;
    let prompt = "say something";
    let (decoded, _) = traces::greedy(&model, chat(prompt), MAX_LEN).await?;
    let mut tokens = model
        .tokenize_chat(ChatRequest::from(chat(prompt)).request)
        .await?
        .tokens;
    tokens.extend(decoded.iter().map(|(token, _)| *token));
    let scored = scores(&model, tokens).await?;
    let prefilled = scored[scored.len() - decoded.len()..]
        .iter()
        .map(|logprob| logprob.unwrap_or(f32::NAN));
    for (step, ((_, decoded), prefilled)) in decoded.iter().zip(prefilled).enumerate() {
        anyhow::ensure!(
            (decoded - prefilled).abs() < LOGPROB_TOLERANCE,
            "{kind:?} step {step}: decoded {decoded}, one prefill {prefilled}"
        );
    }
    Ok(())
}

async fn a_repeated_prompt_resumes_from_the_prefix_cache_to_the_same_text() -> anyhow::Result<()> {
    for kind in KINDS {
        repeated_prompt_resumes_to_the_same_text(kind).await?;
    }
    Ok(())
}

async fn repeated_prompt_resumes_to_the_same_text(kind: Kind) -> anyhow::Result<()> {
    let dir = checkpoint(kind)?;
    let model = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let request = || {
        chat("hello there")
            .set_sampler_max_len(MAX_LEN)
            .set_sampler_topk(1)
    };
    let first = model.send_chat_request(request()).await?;
    let second = model.send_chat_request(request()).await?;
    let text = |r: &inference::ChatCompletionResponse| {
        r.choices[0].message.content.clone().unwrap_or_default()
    };
    anyhow::ensure!(
        first.usage.completion_tokens > 0,
        "the model generated nothing"
    );
    assert_eq!(text(&first), text(&second), "{kind:?}");
    Ok(())
}

concurrent_cases!(
    concurrent_cases,
    [
        a_step_by_step_decode_scores_as_one_prefill: 0,
        a_repeated_prompt_resumes_from_the_prefix_cache_to_the_same_text: 0,
    ]
);

// CUDA's conv and scan kernels (Mamba-2's chunked one with the grouped norm, Mamba 1's sequential one with A per
// item), with no attention layer in the cache
#[tokio::test]
async fn a_long_prompt_scores_on_cuda_as_on_the_cpu() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    for kind in KINDS {
        long_prompt_scores_on_cuda_as_on_the_cpu(kind).await?;
    }
    Ok(())
}

async fn long_prompt_scores_on_cuda_as_on_the_cpu(kind: Kind) -> anyhow::Result<()> {
    let dir = checkpoint(kind)?;
    let build = |cpu: bool| {
        let builder =
            TextModelBuilder::new(dir.path().to_string_lossy()).with_dtype(ModelDType::F32);
        if cpu {
            builder.with_force_cpu()
        } else {
            builder
        }
        .build()
    };
    let (cpu, gpu) = (build(true).await?, build(false).await?);
    let tokens = cpu
        .tokenize_chat(
            ChatRequest::from(chat(&"say something long ".repeat(LONG_PROMPT_REPEATS))).request,
        )
        .await?
        .tokens;
    let (expected, actual) = (
        scores(&cpu, tokens.clone()).await?,
        scores(&gpu, tokens).await?,
    );
    for (position, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
        if let (Some(expected), Some(actual)) = (expected, actual) {
            anyhow::ensure!(
                (expected - actual).abs() < CUDA_LOGPROB_TOLERANCE,
                "{kind:?} position {position}: CPU {expected}, CUDA {actual}"
            );
        }
    }
    // the CUDA decode (replaying its captured step) scores each token as the CPU scores the whole sequence
    let (decoded, _) = traces::greedy(&gpu, chat("say something"), MAX_LEN).await?;
    anyhow::ensure!(!decoded.is_empty(), "{kind:?}: CUDA decoded nothing");
    let mut tokens = cpu
        .tokenize_chat(ChatRequest::from(chat("say something")).request)
        .await?
        .tokens;
    tokens.extend(decoded.iter().map(|(token, _)| *token));
    let scored = scores(&cpu, tokens).await?;
    let prefilled = scored[scored.len() - decoded.len()..]
        .iter()
        .map(|logprob| logprob.unwrap_or(f32::NAN));
    for (step, ((_, decoded), prefilled)) in decoded.iter().zip(prefilled).enumerate() {
        anyhow::ensure!(
            (decoded - prefilled).abs() < CUDA_LOGPROB_TOLERANCE,
            "{kind:?} step {step}: CUDA decoded {decoded}, CPU prefill {prefilled}"
        );
    }
    Ok(())
}
