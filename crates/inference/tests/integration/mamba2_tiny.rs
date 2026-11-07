//! A tiny random-weight Mamba-2 (two groups, tied embeddings, no attention anywhere): decoding step by step through
//! its state scores the tokens as one prefill over them does, a repeated prompt resumes from the prefix cache to the
//! same text, and a prompt long enough for the chunked scan scores on CUDA as on the CPU.

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
const WEIGHT_STD: f32 = 0.2;
const WEIGHT_SEED: u64 = 23;
const MAX_LEN: usize = 8;
const LOGPROB_TOLERANCE: f32 = 1e-3;
const CUDA_LOGPROB_TOLERANCE: f32 = 5e-3;
// past the 128 tokens from which CUDA takes the chunked scan
const LONG_PROMPT_REPEATS: usize = 12;

fn checkpoint() -> anyhow::Result<tempfile::TempDir> {
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
        let p = format!("backbone.layers.{layer}");
        weights.insert(format!("{p}.norm.weight"), ones(HIDDEN)?);
        weights.insert(
            format!("{p}.mixer.in_proj.weight"),
            random(&[PROJECTION, HIDDEN])?,
        );
        weights.insert(
            format!("{p}.mixer.out_proj.weight"),
            random(&[HIDDEN, INNER])?,
        );
        weights.insert(
            format!("{p}.mixer.conv1d.weight"),
            random(&[CONV_DIM, 1, CONV])?,
        );
        weights.insert(format!("{p}.mixer.conv1d.bias"), random(&[CONV_DIM])?);
        weights.insert(format!("{p}.mixer.dt_bias"), random(&[HEADS])?);
        weights.insert(format!("{p}.mixer.A_log"), random(&[HEADS])?);
        weights.insert(format!("{p}.mixer.D"), ones(HEADS)?);
        weights.insert(format!("{p}.mixer.norm.weight"), ones(INNER)?);
    }
    let config = serde_json::json!({
        "architectures": ["Mamba2ForCausalLM"],
        "model_type": "mamba2",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_heads": HEADS,
        "head_dim": HEAD_DIM,
        "expand": INNER / HIDDEN,
        "n_groups": GROUPS,
        "state_size": STATE,
        "conv_kernel": CONV,
        "layer_norm_epsilon": 1e-5,
        "chunk_size": 256,
        "tie_word_embeddings": true,
    });
    let dir = tempfile::tempdir()?;
    for file in FIXTURE_FILES {
        std::fs::copy(format!("{FIXTURE}/{file}"), dir.path().join(file))?;
    }
    std::fs::copy(TOKENIZER, dir.path().join("tokenizer.json"))?;
    // as Python's json writes the default dt limit
    let text = config.to_string().replace(
        r#""tie_word_embeddings""#,
        r#""time_step_limit":[0.0,Infinity],"tie_word_embeddings""#,
    );
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
    let dir = checkpoint()?;
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
            "step {step}: decoded {decoded}, one prefill {prefilled}"
        );
    }
    Ok(())
}

async fn a_repeated_prompt_resumes_from_the_prefix_cache_to_the_same_text() -> anyhow::Result<()> {
    let dir = checkpoint()?;
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
    assert_eq!(text(&first), text(&second));
    Ok(())
}

concurrent_cases!(
    concurrent_cases,
    [
        a_step_by_step_decode_scores_as_one_prefill: 0,
        a_repeated_prompt_resumes_from_the_prefix_cache_to_the_same_text: 0,
    ]
);

// CUDA's conv and chunked scan kernels with the grouped norm, on a paged cache with no attention layer in it
#[tokio::test]
async fn a_long_prompt_scores_on_cuda_as_on_the_cpu() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let dir = checkpoint()?;
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
                "position {position}: CPU {expected}, CUDA {actual}"
            );
        }
    }
    let (decoded, _) = traces::greedy(&gpu, chat("say something"), MAX_LEN).await?;
    anyhow::ensure!(!decoded.is_empty(), "CUDA decoded nothing");
    Ok(())
}
