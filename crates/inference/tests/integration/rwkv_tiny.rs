//! Tiny random-weight RWKV stacks (two layers, two heads): an RWKV-7 in flash-linear-attention's layout and an RWKV-6
//! in RWKV's HF layout. Decoding step by step through the token shifts and WKV state scores the tokens as one prefill
//! over them does, a repeated prompt resumes from the prefix cache to the same text, and a long prompt scores on CUDA
//! (the fused kernels) as on the CPU.

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
// the CUDA kernel runs heads of 64
const HEAD_DIM: usize = 64;
const HEADS: usize = 2;
const HIDDEN: usize = HEADS * HEAD_DIM;
const FFN: usize = 192;
const LAYERS: usize = 2;
const RANK: usize = 16;
const MIX_RANK: usize = 8;
const WEIGHT_STD: f32 = 0.1;
const WEIGHT_SEED: u64 = 31;
const MAX_LEN: usize = 8;
const LOGPROB_TOLERANCE: f32 = 1e-3;
const CUDA_LOGPROB_TOLERANCE: f32 = 5e-3;
const LONG_PROMPT_REPEATS: usize = 12;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Rwkv7,
    Rwkv6,
}

const KINDS: [Kind; 2] = [Kind::Rwkv7, Kind::Rwkv6];

// (name, shape, all ones / all zeros / random)
#[derive(Clone, Copy)]
enum Init {
    Ones,
    Zeros,
    Random,
}

fn weight_shapes(kind: Kind) -> Vec<(String, Vec<usize>, Init)> {
    use Init::*;
    let mut shapes = Vec::new();
    let norm = |shapes: &mut Vec<(String, Vec<usize>, Init)>, p: &str| {
        shapes.push((format!("{p}.weight"), vec![HIDDEN], Ones));
        shapes.push((format!("{p}.bias"), vec![HIDDEN], Zeros));
    };
    match kind {
        Kind::Rwkv7 => {
            shapes.push((
                "model.embeddings.weight".into(),
                vec![VOCAB, HIDDEN],
                Random,
            ));
            norm(&mut shapes, "model.norm");
            shapes.push(("lm_head.weight".into(), vec![VOCAB, HIDDEN], Random));
            for layer in 0..LAYERS {
                let l = format!("model.layers.{layer}");
                if layer == 0 {
                    norm(&mut shapes, &format!("{l}.pre_norm"));
                }
                norm(&mut shapes, &format!("{l}.attn_norm"));
                norm(&mut shapes, &format!("{l}.ffn_norm"));
                let a = format!("{l}.attn");
                for proj in ["r_proj", "k_proj", "v_proj", "o_proj"] {
                    shapes.push((format!("{a}.{proj}.weight"), vec![HIDDEN, HIDDEN], Random));
                }
                let loras: &[&str] = if layer == 0 {
                    &["w", "a", "g"]
                } else {
                    &["w", "a", "v", "g"]
                };
                for lora in loras {
                    let p = format!("{a}.{lora}_lora.lora");
                    shapes.push((format!("{p}.0.weight"), vec![RANK, HIDDEN], Random));
                    shapes.push((format!("{p}.2.weight"), vec![HIDDEN, RANK], Random));
                    if *lora != "g" {
                        shapes.push((format!("{p}.2.bias"), vec![HIDDEN], Random));
                    }
                }
                for mix in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
                    shapes.push((format!("{a}.{mix}"), vec![1, 1, HIDDEN], Random));
                }
                shapes.push((format!("{a}.k_k"), vec![HIDDEN], Random));
                shapes.push((format!("{a}.k_a"), vec![HIDDEN], Random));
                shapes.push((format!("{a}.r_k"), vec![HEADS, HEAD_DIM], Random));
                norm(&mut shapes, &format!("{a}.g_norm"));
                shapes.push((format!("{l}.ffn.x_k"), vec![HIDDEN], Random));
                shapes.push((format!("{l}.ffn.key.weight"), vec![FFN, HIDDEN], Random));
                shapes.push((format!("{l}.ffn.value.weight"), vec![HIDDEN, FFN], Random));
            }
        }
        Kind::Rwkv6 => {
            shapes.push(("rwkv.embeddings.weight".into(), vec![VOCAB, HIDDEN], Random));
            norm(&mut shapes, "rwkv.ln_out");
            shapes.push(("head.weight".into(), vec![VOCAB, HIDDEN], Random));
            for layer in 0..LAYERS {
                let l = format!("rwkv.blocks.{layer}");
                if layer == 0 {
                    norm(&mut shapes, &format!("{l}.pre_ln"));
                }
                norm(&mut shapes, &format!("{l}.ln1"));
                norm(&mut shapes, &format!("{l}.ln2"));
                let a = format!("{l}.attention");
                for mix in ["x", "w", "k", "v", "r", "g"] {
                    shapes.push((format!("{a}.time_maa_{mix}"), vec![1, 1, HIDDEN], Random));
                }
                shapes.push((
                    format!("{a}.time_maa_w1"),
                    vec![HIDDEN, 5 * MIX_RANK],
                    Random,
                ));
                shapes.push((
                    format!("{a}.time_maa_w2"),
                    vec![5, MIX_RANK, HIDDEN],
                    Random,
                ));
                shapes.push((format!("{a}.time_decay"), vec![1, 1, HIDDEN], Random));
                shapes.push((format!("{a}.time_decay_w1"), vec![HIDDEN, RANK], Random));
                shapes.push((format!("{a}.time_decay_w2"), vec![RANK, HIDDEN], Random));
                shapes.push((format!("{a}.time_faaaa"), vec![HEADS, HEAD_DIM], Random));
                for proj in ["receptance", "key", "value", "gate", "output"] {
                    shapes.push((format!("{a}.{proj}.weight"), vec![HIDDEN, HIDDEN], Random));
                }
                norm(&mut shapes, &format!("{a}.ln_x"));
                let f = format!("{l}.feed_forward");
                shapes.push((format!("{f}.time_maa_k"), vec![1, 1, HIDDEN], Random));
                shapes.push((format!("{f}.time_maa_r"), vec![1, 1, HIDDEN], Random));
                shapes.push((format!("{f}.key.weight"), vec![FFN, HIDDEN], Random));
                shapes.push((
                    format!("{f}.receptance.weight"),
                    vec![HIDDEN, HIDDEN],
                    Random,
                ));
                shapes.push((format!("{f}.value.weight"), vec![HIDDEN, FFN], Random));
            }
        }
    }
    shapes
}

fn checkpoint(kind: Kind) -> anyhow::Result<tempfile::TempDir> {
    let rng = &mut StdRng::seed_from_u64(WEIGHT_SEED);
    let normal = Normal::new(0f32, WEIGHT_STD)?;
    let mut weights = HashMap::new();
    for (name, shape, init) in weight_shapes(kind) {
        let tensor = match init {
            Init::Ones => Tensor::ones(shape.as_slice(), DType::F32, &Device::Cpu)?,
            Init::Zeros => Tensor::zeros(shape.as_slice(), DType::F32, &Device::Cpu)?,
            Init::Random => {
                let data = (0..shape.iter().product::<usize>())
                    .map(|_| normal.sample(rng))
                    .collect::<Vec<_>>();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
        };
        weights.insert(name, tensor);
    }
    let config = match kind {
        Kind::Rwkv7 => serde_json::json!({
            "architectures": ["RWKV7ForCausalLM"],
            "model_type": "rwkv7",
            "head_dim": HEAD_DIM,
            "norm_eps": 1e-5,
            "decay_low_rank_dim": RANK,
            "a_low_rank_dim": RANK,
            "v_low_rank_dim": RANK,
            "gate_low_rank_dim": RANK,
        }),
        Kind::Rwkv6 => serde_json::json!({
            "architectures": ["Rwkv6ForCausalLM"],
            "model_type": "rwkv6",
            "head_size": HEAD_DIM,
            "layer_norm_epsilon": 1e-5,
            "rescale_every": 6,
        }),
    };
    let mut config = config.as_object().cloned().expect("an object");
    for (key, value) in [
        ("vocab_size", serde_json::json!(VOCAB)),
        ("hidden_size", serde_json::json!(HIDDEN)),
        ("num_hidden_layers", serde_json::json!(LAYERS)),
        ("intermediate_size", serde_json::json!(FFN)),
        ("tie_word_embeddings", serde_json::json!(false)),
    ] {
        config.insert(key.to_string(), value);
    }
    let dir = tempfile::tempdir()?;
    for file in FIXTURE_FILES {
        std::fs::copy(format!("{FIXTURE}/{file}"), dir.path().join(file))?;
    }
    std::fs::copy(TOKENIZER, dir.path().join("tokenizer.json"))?;
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::Value::Object(config).to_string(),
    )?;
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
    let (decoded, _) = traces::greedy(&model, chat("say something"), MAX_LEN).await?;
    let mut tokens = model
        .tokenize_chat(ChatRequest::from(chat("say something")).request)
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
    let (decoded, _) = traces::greedy(&gpu, chat("say something"), MAX_LEN).await?;
    anyhow::ensure!(!decoded.is_empty(), "CUDA decoded nothing");
    Ok(())
}
