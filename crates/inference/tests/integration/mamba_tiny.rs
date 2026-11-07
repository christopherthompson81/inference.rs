//! Tiny random-weight Mamba stacks with no attention anywhere, a Mamba-2 (two groups) and a Falcon-Mamba (Mamba 1's
//! selection with its dt, B and C norms, A per channel and state item), both with tied embeddings: decoding step by
//! step through the state scores the tokens as one prefill over them does, a repeated prompt resumes from the prefix
//! cache to the same text, and a long prompt (Mamba-2's chunked scan, Mamba 1's sequential one) scores on CUDA as on
//! the CPU.

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
}

const KINDS: [Kind; 2] = [Kind::Mamba2, Kind::FalconMamba];

fn checkpoint(kind: Kind) -> anyhow::Result<tempfile::TempDir> {
    let rng = &mut StdRng::seed_from_u64(WEIGHT_SEED);
    let normal = Normal::new(0f32, WEIGHT_STD)?;
    let mut random = |shape: &[usize]| {
        let data = (0..shape.iter().product::<usize>())
            .map(|_| normal.sample(rng))
            .collect::<Vec<_>>();
        Tensor::from_vec(data, shape, &Device::Cpu)
    };
    let ones = |n: usize| Tensor::ones(n, DType::F32, &Device::Cpu);
    let mut weights = HashMap::from([
        (
            "backbone.embeddings.weight".to_string(),
            random(&[VOCAB, HIDDEN])?,
        ),
        ("backbone.norm_f.weight".to_string(), ones(HIDDEN)?),
    ]);
    for layer in 0..LAYERS {
        let p = format!("backbone.layers.{layer}.mixer");
        weights.insert(
            format!("backbone.layers.{layer}.norm.weight"),
            ones(HIDDEN)?,
        );
        weights.insert(format!("{p}.out_proj.weight"), random(&[HIDDEN, INNER])?);
        match kind {
            Kind::Mamba2 => {
                weights.insert(
                    format!("{p}.in_proj.weight"),
                    random(&[PROJECTION, HIDDEN])?,
                );
                weights.insert(format!("{p}.conv1d.weight"), random(&[CONV_DIM, 1, CONV])?);
                weights.insert(format!("{p}.conv1d.bias"), random(&[CONV_DIM])?);
                weights.insert(format!("{p}.dt_bias"), random(&[HEADS])?);
                weights.insert(format!("{p}.A_log"), random(&[HEADS])?);
                weights.insert(format!("{p}.D"), ones(HEADS)?);
                weights.insert(format!("{p}.norm.weight"), ones(INNER)?);
            }
            Kind::FalconMamba => {
                weights.insert(format!("{p}.in_proj.weight"), random(&[2 * INNER, HIDDEN])?);
                weights.insert(format!("{p}.conv1d.weight"), random(&[INNER, 1, CONV])?);
                weights.insert(format!("{p}.conv1d.bias"), random(&[INNER])?);
                weights.insert(
                    format!("{p}.x_proj.weight"),
                    random(&[DT_RANK + 2 * STATE, INNER])?,
                );
                weights.insert(format!("{p}.dt_proj.weight"), random(&[INNER, DT_RANK])?);
                weights.insert(format!("{p}.dt_proj.bias"), random(&[INNER])?);
                weights.insert(format!("{p}.A_log"), random(&[INNER, STATE])?);
                weights.insert(format!("{p}.D"), ones(INNER)?);
            }
        }
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
        }),
        Kind::FalconMamba => serde_json::json!({
            "architectures": ["FalconMambaForCausalLM"],
            "model_type": "falcon_mamba",
            "intermediate_size": INNER,
            "time_step_rank": DT_RANK,
        }),
    };
    let mut config = config.as_object().cloned().expect("an object");
    for (key, value) in [
        ("vocab_size", serde_json::json!(VOCAB)),
        ("hidden_size", serde_json::json!(HIDDEN)),
        ("num_hidden_layers", serde_json::json!(LAYERS)),
        ("state_size", serde_json::json!(STATE)),
        ("conv_kernel", serde_json::json!(CONV)),
        ("layer_norm_epsilon", serde_json::json!(1e-5)),
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
    let (decoded, _) = traces::greedy(&gpu, chat("say something"), MAX_LEN).await?;
    anyhow::ensure!(!decoded.is_empty(), "{kind:?}: CUDA decoded nothing");
    Ok(())
}
