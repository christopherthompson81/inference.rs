//! A tiny random-weight Kimi-Linear (five layers): KDA layers with a dense first layer and routed experts, around two
//! MLA layers whose rope part carries no positions. Decoding step by step through the KDA state and the MLA cache
//! scores the tokens as one prefill over them does, a repeated prompt resumes from the prefix cache to the same text,
//! and a long prompt scores on CUDA (the KDA kernel, the paged latent cache) as on the CPU, its decode (a replayed CUDA
//! graph) scoring as a CPU prefill does. `scripts/kimi_linear_tiny_parity.py` checks the same shapes against
//! transformers.

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
const HIDDEN: usize = 128;
const FFN: usize = 96;
const LAYERS: usize = 5;
// one-based, as the checkpoint numbers them
const FULL_ATTN_LAYERS: [usize; 2] = [3, 5];
// the CUDA kernel runs heads of 64
const KDA_HEADS: usize = 2;
const KDA_HEAD_DIM: usize = 64;
const KDA_WIDTH: usize = KDA_HEADS * KDA_HEAD_DIM;
const CONV: usize = 4;
const MLA_HEADS: usize = 2;
// the CUDA MLA decode kernel runs latent and rope widths of 512 and 64, as Kimi-Linear has them
const KV_LORA: usize = 512;
const NOPE: usize = 16;
const ROPE: usize = 64;
const V_HEAD: usize = 16;
const EXPERTS: usize = 4;
const MOE_FFN: usize = 32;
const WEIGHT_STD: f32 = 0.1;
const WEIGHT_SEED: u64 = 31;
const MAX_LEN: usize = 8;
const LOGPROB_TOLERANCE: f32 = 1e-3;
const CUDA_LOGPROB_TOLERANCE: f32 = 5e-3;
const LONG_PROMPT_REPEATS: usize = 12;

#[derive(Clone, Copy, Debug)]
struct Kind;

const KIND: Kind = Kind;

// (name, shape, all ones / random)
#[derive(Clone, Copy)]
enum Init {
    Ones,
    Random,
}

fn weight_shapes() -> Vec<(String, Vec<usize>, Init)> {
    use Init::*;
    let mut shapes = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![VOCAB, HIDDEN],
            Random,
        ),
        ("model.norm.weight".to_string(), vec![HIDDEN], Ones),
        ("lm_head.weight".to_string(), vec![VOCAB, HIDDEN], Random),
    ];
    for layer in 0..LAYERS {
        let l = format!("model.layers.{layer}");
        let mut push =
            |name: String, shape: Vec<usize>, init: Init| shapes.push((name, shape, init));
        push(format!("{l}.input_layernorm.weight"), vec![HIDDEN], Ones);
        push(
            format!("{l}.post_attention_layernorm.weight"),
            vec![HIDDEN],
            Ones,
        );
        let a = format!("{l}.self_attn");
        if FULL_ATTN_LAYERS.contains(&(layer + 1)) {
            push(
                format!("{a}.q_proj.weight"),
                vec![MLA_HEADS * (NOPE + ROPE), HIDDEN],
                Random,
            );
            push(
                format!("{a}.kv_a_proj_with_mqa.weight"),
                vec![KV_LORA + ROPE, HIDDEN],
                Random,
            );
            push(format!("{a}.kv_a_layernorm.weight"), vec![KV_LORA], Ones);
            push(
                format!("{a}.kv_b_proj.weight"),
                vec![MLA_HEADS * (NOPE + V_HEAD), KV_LORA],
                Random,
            );
            push(
                format!("{a}.o_proj.weight"),
                vec![HIDDEN, MLA_HEADS * V_HEAD],
                Random,
            );
        } else {
            for proj in ["q_proj", "k_proj", "v_proj"] {
                push(
                    format!("{a}.{proj}.weight"),
                    vec![KDA_WIDTH, HIDDEN],
                    Random,
                );
            }
            for conv in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                push(
                    format!("{a}.{conv}.weight"),
                    vec![KDA_WIDTH, 1, CONV],
                    Random,
                );
            }
            for low_rank in ['f', 'g'] {
                push(
                    format!("{a}.{low_rank}_a_proj.weight"),
                    vec![KDA_HEAD_DIM, HIDDEN],
                    Random,
                );
                push(
                    format!("{a}.{low_rank}_b_proj.weight"),
                    vec![KDA_WIDTH, KDA_HEAD_DIM],
                    Random,
                );
            }
            push(
                format!("{a}.b_proj.weight"),
                vec![KDA_HEADS, HIDDEN],
                Random,
            );
            push(format!("{a}.dt_bias"), vec![KDA_WIDTH], Random);
            push(format!("{a}.A_log"), vec![1, 1, KDA_HEADS, 1], Random);
            push(format!("{a}.o_norm.weight"), vec![KDA_HEAD_DIM], Ones);
            push(
                format!("{a}.o_proj.weight"),
                vec![HIDDEN, KDA_WIDTH],
                Random,
            );
        }
        if layer == 0 {
            for (proj, shape) in [
                ("gate_proj", [FFN, HIDDEN]),
                ("up_proj", [FFN, HIDDEN]),
                ("down_proj", [HIDDEN, FFN]),
            ] {
                push(format!("{l}.mlp.{proj}.weight"), shape.to_vec(), Random);
            }
            continue;
        }
        let m = format!("{l}.block_sparse_moe");
        push(format!("{m}.gate.weight"), vec![EXPERTS, HIDDEN], Random);
        push(
            format!("{m}.gate.e_score_correction_bias"),
            vec![EXPERTS],
            Random,
        );
        for expert in 0..EXPERTS {
            let e = format!("{m}.experts.{expert}");
            push(format!("{e}.w1.weight"), vec![MOE_FFN, HIDDEN], Random);
            push(format!("{e}.w3.weight"), vec![MOE_FFN, HIDDEN], Random);
            push(format!("{e}.w2.weight"), vec![HIDDEN, MOE_FFN], Random);
        }
        for (proj, shape) in [
            ("gate_proj", [MOE_FFN, HIDDEN]),
            ("up_proj", [MOE_FFN, HIDDEN]),
            ("down_proj", [HIDDEN, MOE_FFN]),
        ] {
            push(
                format!("{m}.shared_experts.{proj}.weight"),
                shape.to_vec(),
                Random,
            );
        }
    }
    shapes
}

fn checkpoint(_kind: Kind) -> anyhow::Result<tempfile::TempDir> {
    let rng = &mut StdRng::seed_from_u64(WEIGHT_SEED);
    let normal = Normal::new(0f32, WEIGHT_STD)?;
    let mut weights = HashMap::new();
    for (name, shape, init) in weight_shapes() {
        let tensor = match init {
            Init::Ones => Tensor::ones(shape.as_slice(), DType::F32, &Device::Cpu)?,
            Init::Random => {
                let data = (0..shape.iter().product::<usize>())
                    .map(|_| normal.sample(rng))
                    .collect::<Vec<_>>();
                Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?
            }
        };
        weights.insert(name, tensor);
    }
    let config = serde_json::json!({
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": MLA_HEADS,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": false,
        "q_lora_rank": null,
        "kv_lora_rank": KV_LORA,
        "qk_nope_head_dim": NOPE,
        "qk_rope_head_dim": ROPE,
        "v_head_dim": V_HEAD,
        "mla_use_nope": true,
        "num_experts": EXPERTS,
        "num_experts_per_token": 2,
        "moe_intermediate_size": MOE_FFN,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": 2.5,
        "model_max_length": 4096,
        "linear_attn_config": {
            "full_attn_layers": FULL_ATTN_LAYERS,
            "head_dim": KDA_HEAD_DIM,
            "num_heads": KDA_HEADS,
            "short_conv_kernel_size": CONV,
        },
    });
    let dir = tempfile::tempdir()?;
    for file in FIXTURE_FILES {
        std::fs::copy(format!("{FIXTURE}/{file}"), dir.path().join(file))?;
    }
    std::fs::copy(TOKENIZER, dir.path().join("tokenizer.json"))?;
    std::fs::write(dir.path().join("config.json"), config.to_string())?;
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
    step_by_step_decode_scores_as_one_prefill(KIND).await
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
    repeated_prompt_resumes_to_the_same_text(KIND).await
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
    long_prompt_scores_on_cuda_as_on_the_cpu(KIND).await
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
