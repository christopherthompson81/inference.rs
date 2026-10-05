//! Tiny gpt-oss (sliding window, sinks) and Llama 4 (chunked) checkpoints whose paged GPU decode must match the CPU.

use std::collections::HashMap;
use std::path::Path;

use inference::{
    Model, ModelDType, MultimodalModelBuilder, RequestBuilder, TextMessageRole, TextModelBuilder,
};
use inference_models_llama::llama4::{Llama4Config, Llama4Model};
use inference_models_other::gpt_oss;
use inference_nn::paged_attention::AttentionImplementation;

#[path = "../support/recording.rs"]
mod recording;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
// Byte-level, so every character is a token: the prompt alone is several windows long.
const PROMPTS: [&str; 3] = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
];
const MAX_LEN: usize = 24;
// Head dim 64 is the smallest every CUDA attention backend serves.
const HEAD_DIM: usize = 64;
const VOCAB: usize = 266;
const WINDOW: usize = 8;
const GPT_OSS_LAYERS: usize = 2;
const GPT_OSS_HIDDEN: usize = 64;
const GPT_OSS_EXPERT_INTER: usize = 64;
const GPT_OSS_EXPERTS: usize = 2;
// Near the top attention scores of these random weights, so a backend that drops the sinks moves the output.
const SINK_LOGIT: f64 = 8.0;
const LLAMA4_LAYERS: usize = 4;
const LLAMA4_HIDDEN: usize = 2 * HEAD_DIM;
const LLAMA4_EXPERT_INTER: usize = 64;
const LLAMA4_EXPERTS: usize = 2;
const LLAMA4_MOE_STEP: usize = 2;
// The image token is last, as image_token_index expects.
const LLAMA4_IMAGE_TOKENS: [&str; 6] = [
    "<|image_start|>",
    "<|image_end|>",
    "<|patch|>",
    "<|tile_x_separator|>",
    "<|tile_y_separator|>",
    "<|image|>",
];
const LLAMA4_VOCAB: usize = VOCAB + LLAMA4_IMAGE_TOKENS.len();
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));
const LOGPROB_TOLERANCE: f32 = 0.5;
// A gap the rounding drift above could close.
const TIE_MARGIN: f32 = LOGPROB_TOLERANCE;
// Decode steps matched across the prompts: several windows and chunk edges.
const MIN_AGREED: usize = 3 * WINDOW;

fn tokenizer() -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::from_str(&std::fs::read_to_string(format!(
        "{FIXTURES}/paddleocr_vl/tiny/tokenizer.json"
    ))?)?)
}

// The input processor looks its image tokens up even for a text-only prompt.
fn llama4_tokenizer() -> anyhow::Result<serde_json::Value> {
    let mut tokenizer = tokenizer()?;
    let added = tokenizer["added_tokens"].as_array_mut().unwrap();
    for (offset, content) in LLAMA4_IMAGE_TOKENS.iter().enumerate() {
        added.push(serde_json::json!({
            "id": VOCAB + offset,
            "content": content,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }));
    }
    Ok(tokenizer)
}

fn record(
    config: &serde_json::Value,
    tokenizer: &serde_json::Value,
    num_layers: usize,
    absent: &[&str],
    shapes: HashMap<String, Vec<usize>>,
    build: impl FnOnce(
        inference_quant::ShardedVarBuilder,
        inference_nn::model::NormalLoadingMetadata,
    ) -> candle_core::Result<()>,
) -> anyhow::Result<tempfile::TempDir> {
    let scratch = tempfile::tempdir()?;
    let config_path = scratch.path().join("config.json");
    std::fs::write(&config_path, config.to_string())?;
    let tokenizer_path = scratch.path().join("tokenizer.json");
    std::fs::write(&tokenizer_path, tokenizer.to_string())?;
    let files = [
        config_path,
        tokenizer_path,
        format!("{FIXTURES}/llama_tiny/tokenizer_config.json").into(),
        format!("{FIXTURES}/llama_tiny/chat_template.jinja").into(),
    ];
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_with_shapes(&files, num_layers, absent, shapes, build)
}

fn tiny_gpt_oss() -> anyhow::Result<tempfile::TempDir> {
    // Alternating from a sliding layer, as the released checkpoints do.
    let layer_types = (0..GPT_OSS_LAYERS)
        .map(|layer| ["sliding_attention", "full_attention"][layer % 2])
        .collect::<Vec<_>>();
    let config = serde_json::json!({
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "vocab_size": VOCAB,
        "hidden_size": GPT_OSS_HIDDEN,
        "intermediate_size": GPT_OSS_EXPERT_INTER,
        "num_hidden_layers": GPT_OSS_LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 150000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": false
        },
        "sliding_window": WINDOW,
        "layer_types": layer_types,
        "num_local_experts": GPT_OSS_EXPERTS,
        // every expert per token, so a rounding change cannot flip the routing
        "num_experts_per_tok": GPT_OSS_EXPERTS,
        "attention_bias": true,
        "tie_word_embeddings": false
    });
    let cfg: gpt_oss::Config = serde_json::from_value(config.clone())?;
    // Reported absent so the experts load as split dense projections rather than packed MXFP4 blocks.
    let packed = (0..GPT_OSS_LAYERS)
        .map(|layer| format!("model.layers.{layer}.mlp.experts.gate_up_proj_blocks"))
        .collect::<Vec<_>>();
    let packed = packed.iter().map(String::as_str).collect::<Vec<_>>();
    let checkpoint = record(
        &config,
        &tokenizer()?,
        GPT_OSS_LAYERS,
        &packed,
        HashMap::new(),
        |vb, metadata| {
            gpt_oss::Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
                .map(|_| ())
        },
    )?;
    let weights = checkpoint.path().join("model.safetensors");
    let mut tensors = candle_core::safetensors::load(&weights, &candle_core::Device::Cpu)?;
    for (name, tensor) in tensors.iter_mut() {
        if name.ends_with(".self_attn.sinks") {
            *tensor = tensor.ones_like()?.affine(SINK_LOGIT, 0.0)?;
        }
    }
    candle_core::safetensors::save(&tensors, &weights)?;
    Ok(checkpoint)
}

// Llama 4 stacks each layer's experts transposed, [E, in, out]; the layout detection reads these shapes first.
fn llama4_expert_shapes() -> HashMap<String, Vec<usize>> {
    (LLAMA4_MOE_STEP - 1..LLAMA4_LAYERS)
        .step_by(LLAMA4_MOE_STEP)
        .flat_map(|layer| {
            let p = format!("language_model.model.layers.{layer}.feed_forward.experts");
            [
                (
                    format!("{p}.gate_up_proj"),
                    vec![LLAMA4_EXPERTS, LLAMA4_HIDDEN, 2 * LLAMA4_EXPERT_INTER],
                ),
                (
                    format!("{p}.down_proj"),
                    vec![LLAMA4_EXPERTS, LLAMA4_EXPERT_INTER, LLAMA4_HIDDEN],
                ),
            ]
        })
        .collect()
}

fn tiny_llama4() -> anyhow::Result<tempfile::TempDir> {
    let config = serde_json::json!({
        "architectures": ["Llama4ForConditionalGeneration"],
        "model_type": "llama4",
        "image_token_index": LLAMA4_VOCAB - 1,
        "text_config": {
            "hidden_act": "silu",
            "hidden_size": LLAMA4_HIDDEN,
            "intermediate_size": LLAMA4_EXPERT_INTER,
            "intermediate_size_mlp": 128,
            "vocab_size": LLAMA4_VOCAB,
            "num_hidden_layers": LLAMA4_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 512,
            "use_qk_norm": true,
            "interleave_moe_layer_step": LLAMA4_MOE_STEP,
            "num_local_experts": LLAMA4_EXPERTS,
            "num_experts_per_tok": LLAMA4_EXPERTS,
            "attention_chunk_size": WINDOW,
            // Every fourth layer is the global NoPE one; a small floor scale makes its temperature tuning act.
            "floor_scale": 4.0,
            "tie_word_embeddings": false
        },
        "vision_config": {
            "hidden_size": 32,
            "hidden_act": "gelu",
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_channels": 3,
            "intermediate_size": 64,
            "vision_output_dim": 32,
            "image_size": 28,
            "patch_size": 14,
            "norm_eps": 1e-5,
            "pixel_shuffle_ratio": 0.5,
            "projector_input_dim": 32,
            "projector_output_dim": 32,
            "vision_feature_layer": -1,
            "rope_theta": 10000.0
        }
    });
    let cfg: Llama4Config = serde_json::from_value(config.clone())?;
    record(
        &config,
        &llama4_tokenizer()?,
        LLAMA4_LAYERS,
        &[],
        llama4_expert_shapes(),
        |vb, metadata| {
            Llama4Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}

#[derive(Debug, PartialEq)]
struct Step {
    token: u32,
    logprob: f32,
    runner_up: u32,
    runner_up_logprob: f32,
}

async fn trace(model: &Model, prompt: &str) -> anyhow::Result<Vec<Step>> {
    let request = RequestBuilder::new()
        .add_message(TextMessageRole::User, prompt)
        .set_sampler_max_len(MAX_LEN)
        // random weights may pick the end token, and every step past it still checks attention
        .set_sampler_ignore_eos(true)
        .set_sampler_topk(1)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(2);
    let response = model.send_chat_request(request).await?;
    let steps = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .ok_or_else(|| anyhow::anyhow!("no logprobs"))?;
    let trace = steps
        .iter()
        .map(|step| match step.top_logprobs.as_slice() {
            [top, runner_up, ..] => Ok(Step {
                token: top.token,
                logprob: top.logprob,
                runner_up: runner_up.token,
                runner_up_logprob: runner_up.logprob,
            }),
            other => anyhow::bail!("expected two top logprobs, got {other:?}"),
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(trace.len() == MAX_LEN, "decode stopped early: {trace:?}");
    Ok(trace)
}

fn paged_cache() -> inference::PagedCacheSpec {
    inference::PagedAttentionMetaBuilder::default()
        .build()
        .unwrap()
}

async fn gpt_oss(dir: &Path, gpu: bool) -> anyhow::Result<Model> {
    // f16 rounds finer than bf16, so another kernel's rounding flips fewer of the random weights' near ties
    let builder = TextModelBuilder::new(dir.to_string_lossy()).with_dtype(ModelDType::F16);
    let builder = if gpu {
        builder.with_paged_attn(paged_cache())
    } else {
        builder.with_force_cpu()
    };
    Ok(builder.build().await?)
}

async fn llama4(dir: &Path, gpu: bool) -> anyhow::Result<Model> {
    // its random weights overflow f16
    let builder = MultimodalModelBuilder::new(dir.to_string_lossy()).with_dtype(ModelDType::BF16);
    let builder = if gpu {
        builder.with_paged_attn(paged_cache())
    } else {
        builder.with_force_cpu()
    };
    Ok(builder.build().await?)
}

async fn traces(model: &Model) -> anyhow::Result<Vec<Vec<Step>>> {
    let mut traces = Vec::with_capacity(PROMPTS.len());
    for prompt in PROMPTS {
        traces.push(trace(model, prompt).await?);
    }
    Ok(traces)
}

// Random weights leave near ties that rounding may flip, so a trace need only match up to one, where the GPU
// must take the CPU's runner-up.
fn ensure_close(gpu: &[Vec<Step>], cpu: &[Vec<Step>]) -> anyhow::Result<()> {
    let mut agreed = 0;
    for (gpu, cpu) in gpu.iter().zip(cpu) {
        let matched = gpu
            .iter()
            .zip(cpu)
            .take_while(|(g, c)| g.token == c.token)
            .count();
        let close = gpu[..matched].iter().zip(cpu).all(|(g, c)| {
            (g.logprob - c.logprob).abs() < LOGPROB_TOLERANCE
                && (g.runner_up_logprob - c.runner_up_logprob).abs() < LOGPROB_TOLERANCE
        });
        let split_at_a_tie = gpu.get(matched).zip(cpu.get(matched)).is_none_or(|(g, c)| {
            g.token == c.runner_up && c.logprob - c.runner_up_logprob < TIE_MARGIN
        });
        anyhow::ensure!(
            close && split_at_a_tie,
            "GPU decode {gpu:?} differs from CPU {cpu:?}"
        );
        agreed += matched;
    }
    anyhow::ensure!(
        agreed >= MIN_AGREED,
        "only {agreed} steps agreed before near ties; GPU {gpu:?}, CPU {cpu:?}"
    );
    Ok(())
}

#[tokio::test]
async fn gpt_oss_decodes_alike_on_cpu_and_paged_gpu() -> anyhow::Result<()> {
    let checkpoint = tiny_gpt_oss()?;
    let cpu = traces(&gpt_oss(checkpoint.path(), false).await?).await?;
    assert_eq!(
        cpu,
        traces(&gpt_oss(checkpoint.path(), false).await?).await?
    );
    if ON_GPU {
        ensure_close(
            &traces(&gpt_oss(checkpoint.path(), true).await?).await?,
            &cpu,
        )?;
    }
    Ok(())
}

#[tokio::test]
async fn llama4_decodes_alike_on_cpu_and_paged_gpu() -> anyhow::Result<()> {
    let checkpoint = tiny_llama4()?;
    let cpu = traces(&llama4(checkpoint.path(), false).await?).await?;
    assert_eq!(cpu, traces(&llama4(checkpoint.path(), false).await?).await?);
    if ON_GPU {
        ensure_close(
            &traces(&llama4(checkpoint.path(), true).await?).await?,
            &cpu,
        )?;
    }
    Ok(())
}
