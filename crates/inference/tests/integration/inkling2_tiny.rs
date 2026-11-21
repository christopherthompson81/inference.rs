//! A tiny random-weight Inkling 2 (sliding and global layers with relative-position biases and per-query sinks, short
//! convolutions with their state, block attention residuals, a looped core run three times under its controllers,
//! softcapped SwiGLU experts and a padded vocabulary): decoding step by step through the KV and convolution states of
//! every pass scores the tokens as one prefill over them does, on the CPU, Vulkan and CUDA (also through the paged
//! cache), a prompt past the window and the bias's extent scores on Vulkan and CUDA as on the CPU, and paged decode
//! steps replay captured CUDA graphs and give the eager output.

use std::collections::HashMap;

use inference::{
    ChatRequest, ModelDType, PromptInput, PromptLogitsRequest, RequestBuilder, TextMessageRole,
    TextMessages, TextModelBuilder,
};
use inference_tensor::{DType, Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

#[path = "../support/decode_graphs.rs"]
mod decode_graphs;
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
// the widths the tests run at; the NVFP4 experts' rows need whole 64-value blocks
pub(crate) struct Dims {
    pub(crate) hidden: usize,
    pub(crate) expert_width: usize,
}
const TINY: Dims = Dims {
    hidden: 32,
    expert_width: 32,
};
const LAYERS: usize = 6;
const BLOCK: usize = 2;
// the looped core, layers 2 and 3, run three times
const CORE: (usize, usize) = (2, 4);
const PASSES: usize = 3;
const GATE_RANK: usize = 8;
const SLIDING: [usize; 4] = [0, 1, 3, 4];
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
// a head size the fused CUDA attention and the decode graphs' paged check take
const HEAD_DIM: usize = 64;
const D_REL: usize = 4;
// short, so the chat prompt runs past both
const REL_EXTENT: usize = 8;
const WINDOW: usize = 4;
const CONV: usize = 4;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const WEIGHT_STD: f32 = 0.2;
const WEIGHT_SEED: u64 = 0x696e_6b6c;
const MAX_LEN: usize = 6;
const LOGPROB_TOLERANCE: f32 = 1e-4;
// Vulkan's ops sum in other orders than the CPU's
const VULKAN_LOGPROB_TOLERANCE: f32 = 1e-3;
// CUDA's matmuls and the fused attention sum in other orders too
const CUDA_LOGPROB_TOLERANCE: f32 = 1e-3;
const LONG_PROMPT_REPEATS: usize = 6;
// decode steps a graph round takes a prompt
const GRAPH_MAX_LEN: usize = 8;

pub(crate) fn config(d: &Dims) -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Inkling2LoopForCausalLM"],
        "model_type": "inkling2_loop_text",
        "vocab_size": PADDED_VOCAB,
        "unpadded_vocab_size": VOCAB,
        "hidden_size": d.hidden,
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
        "intermediate_size": d.expert_width,
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

pub(crate) fn weights(d: &Dims) -> anyhow::Result<HashMap<String, Tensor>> {
    let rng = &mut StdRng::seed_from_u64(WEIGHT_SEED);
    let (q_dim, kv_dim) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let mut w = HashMap::new();
    let mut put = |name: String, t: Tensor| {
        w.insert(name, t);
    };
    put(
        "model.embed_tokens.weight".into(),
        random(rng, &[PADDED_VOCAB, d.hidden])?,
    );
    put(
        "model.embed_norm.weight".into(),
        near_one(rng, &[d.hidden])?,
    );
    put("model.norm.weight".into(), near_one(rng, &[d.hidden])?);
    put(
        "model.final_input.query".into(),
        random(rng, &[1, d.hidden])?,
    );
    put(
        "lm_head.weight".into(),
        random(rng, &[PADDED_VOCAB, d.hidden])?,
    );
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        let a = format!("{p}.self_attn");
        let sliding = SLIDING.contains(&i);
        if i > 0 {
            let queries = if sliding { 1 } else { 3 };
            put(
                format!("{p}.attention_input.query"),
                random(rng, &[queries, d.hidden])?,
            );
        }
        put(format!("{p}.mlp_input.query"), random(rng, &[1, d.hidden])?);
        if sliding {
            put(
                format!("{p}.input_layernorm.weight"),
                near_one(rng, &[d.hidden])?,
            );
        } else {
            for norm in ["q", "k", "v"] {
                put(
                    format!("{p}.{norm}_input_layernorm.weight"),
                    near_one(rng, &[d.hidden])?,
                );
            }
            for conv in ["k_sconv", "v_sconv"] {
                put(
                    format!("{a}.{conv}.weight_proj.weight"),
                    random(rng, &[KV_HEADS * CONV, d.hidden])?,
                );
            }
        }
        for (name, rows, cols) in [
            ("q_proj", q_dim, d.hidden),
            ("k_proj", kv_dim, d.hidden),
            ("v_proj", kv_dim, d.hidden),
            ("gate_proj", q_dim, d.hidden),
            ("r_proj", HEADS * D_REL, d.hidden),
            ("sink_proj", HEADS, d.hidden),
            ("o_proj", d.hidden, q_dim),
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
            random(rng, &[d.hidden, CONV])?,
        );
        put(
            format!("{p}.post_mlp_convolution.weight"),
            random(rng, &[d.hidden, CONV])?,
        );
        put(
            format!("{p}.post_attention_layernorm.weight"),
            near_one(rng, &[d.hidden])?,
        );
        put(
            format!("{p}.mlp.gate.weight"),
            random(rng, &[EXPERTS, d.hidden])?,
        );
        put(
            format!("{p}.mlp.gate.e_score_correction_bias"),
            random(rng, &[EXPERTS])?,
        );
        for e in 0..EXPERTS {
            let x = format!("{p}.mlp.experts.{e}");
            put(
                format!("{x}.gate_proj.weight"),
                random(rng, &[d.expert_width, d.hidden])?,
            );
            put(
                format!("{x}.up_proj.weight"),
                random(rng, &[d.expert_width, d.hidden])?,
            );
            put(
                format!("{x}.down_proj.weight"),
                random(rng, &[d.hidden, d.expert_width])?,
            );
        }
        if (CORE.0..CORE.1).contains(&i) {
            put(format!("{p}.loop.msg_query"), random(rng, &[4, d.hidden])?);
            for c in ["q", "k", "v", "ffn"] {
                let l = format!("{p}.loop.{c}");
                put(
                    format!("{l}.down.weight"),
                    random(rng, &[GATE_RANK, 2 * d.hidden])?,
                );
                put(
                    format!("{l}.gain.weight"),
                    random(rng, &[d.hidden, GATE_RANK])?,
                );
                put(
                    format!("{l}.forget.weight"),
                    random(rng, &[d.hidden, GATE_RANK])?,
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
    std::fs::write(dir.path().join("config.json"), config(&TINY).to_string())?;
    let weights = weights(&TINY)?
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
    // f32 KV, so the decode reads the keys the prefill computed (the CPU stores f16 by default)
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_nn::kv_cache::CPU_KV_F32_ENV, "1") };
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

// a prompt past the window and the bias's extent scores on `device` as on the CPU
async fn long_prompt_scores_as_on_the_cpu(device: &str, tolerance: f32) -> anyhow::Result<()> {
    // the CPU stores KV in f16 by default and the GPUs in f32
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_nn::kv_cache::CPU_KV_F32_ENV, "1") };
    let dir = checkpoint()?;
    let prompt = "say something long ".repeat(LONG_PROMPT_REPEATS);
    let mut scores = Vec::new();
    for device in [None, Some(device)] {
        let model = builder(&dir, device).build().await?;
        let tokens = model
            .tokenize_chat(ChatRequest::from(chat(&prompt)).request)
            .await?
            .tokens;
        scores.push(prompt_logprobs(&model, tokens).await?);
    }
    for (position, (cpu, gpu)) in scores[0].iter().zip(&scores[1]).enumerate().skip(1) {
        anyhow::ensure!(
            (cpu - gpu).abs() < tolerance,
            "position {position}: CPU {cpu}, {device} {gpu}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_long_prompt_scores_on_vulkan_as_on_the_cpu() -> anyhow::Result<()> {
    no_matrix_cores();
    if !cfg!(feature = "vulkan") || Device::new_vulkan(0).is_err() {
        return Ok(());
    }
    long_prompt_scores_as_on_the_cpu("vulkan:0", VULKAN_LOGPROB_TOLERANCE).await
}

#[tokio::test]
async fn a_cuda_decode_scores_as_one_prefill() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    decode_scores_as_prefill(Some("cuda:0"), CUDA_LOGPROB_TOLERANCE).await
}

#[tokio::test]
async fn a_long_prompt_scores_on_cuda_as_on_the_cpu() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    long_prompt_scores_as_on_the_cpu("cuda:0", CUDA_LOGPROB_TOLERANCE).await
}

async fn paged_model(dtype: ModelDType) -> anyhow::Result<(tempfile::TempDir, inference::Model)> {
    let dir = checkpoint()?;
    let model = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(dtype)
        .with_prefix_cache_n(None)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;
    Ok((dir, model))
}

#[tokio::test]
async fn a_paged_cuda_decode_scores_as_one_prefill() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let (_dir, model) = paged_model(ModelDType::F32).await?;
    let prompt = "say something long";
    let (decoded, _) = traces::greedy(&model, chat(prompt), MAX_LEN).await?;
    let mut tokens = model
        .tokenize_chat(ChatRequest::from(chat(prompt)).request)
        .await?
        .tokens;
    tokens.extend(decoded.iter().map(|(token, _)| *token));
    let scored = prompt_logprobs(&model, tokens).await?;
    let prefilled = &scored[scored.len() - decoded.len()..];
    for (step, ((_, decoded), prefilled)) in decoded.iter().zip(prefilled).enumerate() {
        anyhow::ensure!(
            (decoded - prefilled).abs() < CUDA_LOGPROB_TOLERANCE,
            "step {step}: decoded {decoded}, one prefill {prefilled}"
        );
    }
    Ok(())
}

async fn paged_rounds() -> anyhow::Result<decode_graphs::Rounds> {
    let (_dir, model) = paged_model(ModelDType::BF16).await?;
    decode_graphs::rounds(&model, GRAPH_MAX_LEN).await
}

// Each GRAPH_PROMPTS entry's greedy (token, logprob) per step, eager (BF16)
const GRAPH_TRACES: [&[(u32, f32)]; decode_graphs::GRAPH_PROMPTS.len()] = [
    &[
        (48, -2.9551888),
        (150, -3.2265258),
        (137, -3.194851),
        (191, -2.4267263),
        (119, -3.4348319),
        (149, -3.3291705),
        (141, -2.8890543),
        (77, -2.9204497),
    ],
    &[
        (16, -3.3449197),
        (262, -1.2122244),
        (38, -3.0433576),
        (61, -3.1591885),
        (226, -3.1263769),
        (262, -2.958307),
        (192, -2.863158),
        (150, -2.1006672),
    ],
    &[
        (48, -2.497094),
        (185, -3.548465),
        (257, -3.0858254),
        (87, -3.3803632),
        (48, -2.3982816),
        (74, -2.3380835),
        (178, -2.2719083),
        (102, -3.9094074),
    ],
    &[
        (185, -3.2546346),
        (161, -3.1738234),
        (64, -3.339417),
        (2, -3.548018),
    ],
    &[
        (48, -3.009561),
        (150, -3.1070476),
        (172, -3.3455296),
        (160, -3.1594114),
        (81, -2.7371683),
        (195, -3.1146226),
        (112, -3.4764118),
        (10, -3.338034),
    ],
];

// paged decode steps (attention slots and convolution states) replay captured graphs and give the eager output; the
// random weights' flat logits leave BF16 near-ties a batch breaks either way
#[tokio::test]
async fn paged_cuda_decode_replays_cuda_graphs() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let snapshotter = decode_graphs::recorder();
    let rounds = paged_rounds().await?;
    decode_graphs::assert_replayed(&decode_graphs::Counters::take(&snapshotter));
    decode_graphs::assert_traces_up_to_ties(&rounds, &GRAPH_TRACES);
    Ok(())
}

#[tokio::test]
async fn paged_cuda_decode_without_cuda_graphs_matches() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    decode_graphs::disable_graphs();
    let snapshotter = decode_graphs::recorder();
    let rounds = paged_rounds().await?;
    decode_graphs::assert_disabled(&decode_graphs::Counters::take(&snapshotter));
    decode_graphs::assert_traces_up_to_ties(&rounds, &GRAPH_TRACES);
    Ok(())
}

// Inkling 2's own layout (`model.llm.*`), its experts NVFP4 before `MXFP8_FROM` and MXFP8 from it, as the released
// checkpoints store them
const OFFICIAL: Dims = Dims {
    hidden: 64,
    expert_width: 64,
};
const MXFP8_FROM: usize = 4;
const QUANT_SEED: u64 = 0x6e76_6670;
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
// NVFP4 block scales of 0.25 to 0.47 and MXFP8 values of 1/32 to 1.9 (E4M3 bytes), MXFP8 scales of 2^-7 to 2^-4
const NVFP4_SCALE_BYTES: std::ops::Range<u8> = 0x28..0x30;
const MXFP8_VALUE_BYTES: std::ops::Range<u8> = 0x10..0x3f;
const MXFP8_EXPONENTS: std::ops::Range<u8> = 120..124;
const NVFP4_DENOMINATOR: f32 = 8.0;
// each NVFP4 expert's second-level scale, a power of two that its block scales' exponents can take instead
const NVFP4_EXPERT_EXPONENTS: std::ops::RangeInclusive<i8> = -2..=1;
const E4M3_EXPONENT_SHIFT: u32 = 3;
const NVFP4_BLOCK: usize = 16;
const MXFP8_BLOCK: usize = 32;

fn e4m3(byte: u8) -> f32 {
    let (exponent, mantissa) = (i32::from((byte >> 3) & 0x0f), f32::from(byte & 0x07));
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    if byte & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

struct Quantized {
    // the stored tensors by their name's suffix after `{stored}_weight`
    stored: Vec<(&'static str, Tensor)>,
    // `[experts, rows, cols]`, as they decode
    values: Tensor,
}

// `fold`: each NVFP4 expert's scale in its block scales, its second level 1
fn quantize(
    rng: &mut StdRng,
    nvfp4: bool,
    fold: bool,
    (experts, rows, cols): (usize, usize, usize),
) -> anyhow::Result<Quantized> {
    use rand::Rng;
    let dev = &Device::Cpu;
    let mut values = vec![0f32; experts * rows * cols];
    if nvfp4 {
        let packed = (0..values.len() / 2)
            .map(|_| rng.random::<u8>())
            .collect::<Vec<_>>();
        let mut scales = (0..values.len() / NVFP4_BLOCK)
            .map(|_| rng.random_range(NVFP4_SCALE_BYTES))
            .collect::<Vec<_>>();
        let exponents = (0..experts)
            .map(|_| rng.random_range(NVFP4_EXPERT_EXPONENTS))
            .collect::<Vec<_>>();
        let mut amax = exponents
            .iter()
            .map(|&k| NVFP4_DENOMINATOR * 2f32.powi(i32::from(k)))
            .collect::<Vec<f32>>();
        if fold {
            let per_expert = rows * cols / NVFP4_BLOCK;
            for (block, scale) in scales.iter_mut().enumerate() {
                let k = exponents[block / per_expert];
                *scale = scale.wrapping_add_signed(k << E4M3_EXPONENT_SHIFT);
            }
            amax.fill(NVFP4_DENOMINATOR);
        }
        for (i, value) in values.iter_mut().enumerate() {
            let nibble = (packed[i / 2] >> (4 * (i % 2))) & 0x0f;
            let global = amax[i / (rows * cols)] / NVFP4_DENOMINATOR;
            *value = E2M1[usize::from(nibble)] * e4m3(scales[i / NVFP4_BLOCK]) * global;
        }
        return Ok(Quantized {
            stored: vec![
                (
                    "",
                    Tensor::from_raw_buffer(&packed, DType::U8, &[experts, rows, cols / 2], dev)?,
                ),
                (
                    "_scale",
                    Tensor::from_raw_buffer(
                        &scales,
                        DType::F8E4M3,
                        &[experts, rows, cols / NVFP4_BLOCK],
                        dev,
                    )?,
                ),
                ("_amax", Tensor::from_vec(amax, experts, dev)?),
                (
                    "_outer_scale_denominator",
                    Tensor::from_vec(vec![NVFP4_DENOMINATOR], 1, dev)?,
                ),
            ],
            values: Tensor::from_vec(values, (experts, rows, cols), dev)?,
        });
    }
    let bytes = (0..values.len())
        .map(|_| rng.random_range(MXFP8_VALUE_BYTES) | if rng.random() { 0x80 } else { 0 })
        .collect::<Vec<_>>();
    let exponents = (0..values.len() / MXFP8_BLOCK)
        .map(|_| rng.random_range(MXFP8_EXPONENTS))
        .collect::<Vec<_>>();
    for (i, value) in values.iter_mut().enumerate() {
        let scale = 2f32.powi(i32::from(exponents[i / MXFP8_BLOCK]) - 127);
        *value = e4m3(bytes[i]) * scale;
    }
    Ok(Quantized {
        stored: vec![
            (
                "",
                Tensor::from_raw_buffer(&bytes, DType::F8E4M3, &[experts, rows, cols], dev)?,
            ),
            (
                "_scale",
                Tensor::from_raw_buffer(
                    &exponents,
                    DType::U8,
                    &[experts, rows, cols / MXFP8_BLOCK],
                    dev,
                )?,
            ),
        ],
        values: Tensor::from_vec(values, (experts, rows, cols), dev)?,
    })
}

// the depthwise convs' `[c, w]` taps reversed along `w` and transposed, the dynamic convs' stored `[w, c]`
fn static_taps(taps: &Tensor) -> anyhow::Result<Tensor> {
    let width = taps.dim(1)?;
    let lags = (0..width)
        .rev()
        .map(|lag| taps.narrow(1, lag, 1))
        .collect::<inference_tensor::Result<Vec<_>>>()?;
    Ok(Tensor::cat(&lags, 1)?.t()?.contiguous()?)
}

/// The official checkpoint's tensors, and the model's own (the tiny weights at `OFFICIAL`, the experts as the
/// official ones decode).
fn official_weights(
    fold: bool,
) -> anyhow::Result<(HashMap<String, Tensor>, HashMap<String, Tensor>)> {
    let d = &OFFICIAL;
    let mut ours = weights(d)?;
    let rng = &mut StdRng::seed_from_u64(QUANT_SEED);
    let mut off = HashMap::new();
    let take = |ours: &mut HashMap<String, Tensor>, name: &str| -> anyhow::Result<Tensor> {
        ours.remove(name)
            .ok_or_else(|| anyhow::anyhow!("no tiny tensor `{name}`"))
    };
    let mut theirs = HashMap::new();
    let llm = "model.llm";
    for (ours_name, name) in [
        ("model.embed_tokens.weight", "embed.weight"),
        ("model.embed_norm.weight", "embed_norm.weight"),
        ("model.norm.weight", "norm.weight"),
        ("lm_head.weight", "unembed.weight"),
    ] {
        let t = take(&mut ours, ours_name)?;
        theirs.insert(ours_name.to_string(), t.clone());
        off.insert(format!("{llm}.{name}"), t);
    }
    let query = take(&mut ours, "model.final_input.query")?;
    off.insert(
        format!("{llm}.attn_res.output_query"),
        query.reshape(d.hidden)?,
    );
    theirs.insert("model.final_input.query".to_string(), query);
    let (mut cross_q, mut cross_kv) = (Vec::new(), Vec::new());
    for layer in 0..LAYERS {
        let p = format!("model.layers.{layer}");
        let o = format!("{llm}.layers.{layer}");
        let mut keep =
            |ours: &mut HashMap<String, Tensor>, suffix: &str| -> anyhow::Result<Tensor> {
                let t = take(ours, &format!("{p}.{suffix}"))?;
                theirs.insert(format!("{p}.{suffix}"), t.clone());
                Ok(t)
            };
        if layer > 0 {
            let q = keep(&mut ours, "attention_input.query")?;
            let q = if q.dim(0)? == 1 {
                q.reshape(d.hidden)?
            } else {
                q
            };
            off.insert(format!("{llm}.attn_res.attn_queries.{}", layer - 1), q);
        }
        let q = keep(&mut ours, "mlp_input.query")?.reshape(d.hidden)?;
        off.insert(format!("{llm}.attn_res.mlp_queries.{layer}"), q);
        let renames = [
            ("self_attn.q_proj.weight", "attn.wq_du.weight"),
            ("self_attn.k_proj.weight", "attn.wk_dv.weight"),
            ("self_attn.v_proj.weight", "attn.wv_dv.weight"),
            ("self_attn.o_proj.weight", "attn.wo_ud.weight"),
            ("self_attn.gate_proj.weight", "attn.gate_attn.weight"),
            ("self_attn.r_proj.weight", "attn.wr_du.weight"),
            ("self_attn.sink_proj.weight", "attn.attn_sink_proj.weight"),
            ("self_attn.q_l2_clip.scale", "attn.q_l2_clip.scale"),
            ("self_attn.rel_l2_clip.scale", "attn.rel_l2_clip.scale"),
            (
                "self_attn.rel_logits_proj.proj",
                "attn.rel_logits_proj.proj",
            ),
            (
                "self_attn.k_sconv.weight_proj.weight",
                "attn.k_sconv.weight_proj",
            ),
            (
                "self_attn.v_sconv.weight_proj.weight",
                "attn.v_sconv.weight_proj",
            ),
            ("input_layernorm.weight", "attn_norm.weight"),
            ("q_input_layernorm.weight", "q_input_norm.weight"),
            ("k_input_layernorm.weight", "k_input_norm.weight"),
            ("v_input_layernorm.weight", "v_input_norm.weight"),
            ("post_attention_layernorm.weight", "mlp_norm.weight"),
            ("mlp.gate.weight", "mlp.gate.weight"),
            ("mlp.gate.e_score_correction_bias", "mlp.gate.bias"),
        ];
        for (suffix, name) in renames {
            if ours.contains_key(&format!("{p}.{suffix}")) {
                let t = keep(&mut ours, suffix)?;
                off.insert(format!("{o}.{name}"), t);
            }
        }
        let sliding = SLIDING.contains(&layer);
        for conv in ["k_sconv", "v_sconv"] {
            let taps = keep(&mut ours, &format!("self_attn.{conv}.weight"))?;
            if sliding {
                off.insert(format!("{o}.attn.{conv}.weight"), taps.unsqueeze(1)?);
            } else {
                off.insert(format!("{o}.attn.{conv}.static_w"), static_taps(&taps)?);
            }
        }
        for (conv, name) in [
            ("post_attention_convolution", "attn_sconv"),
            ("post_mlp_convolution", "mlp_sconv"),
        ] {
            let taps = keep(&mut ours, &format!("{conv}.weight"))?;
            off.insert(format!("{o}.{name}.weight"), taps.unsqueeze(1)?);
        }
        if (CORE.0..CORE.1).contains(&layer) {
            let m = keep(&mut ours, "loop.msg_query")?;
            cross_q.extend([m.narrow(0, 0, 1)?, m.narrow(0, 3, 1)?]);
            cross_kv.push(m.narrow(0, 1, 2)?);
            for (ours_c, c) in [("q", "attn"), ("k", "k"), ("v", "v"), ("ffn", "mlp")] {
                for part in ["down", "gain", "forget"] {
                    let t = keep(&mut ours, &format!("loop.{ours_c}.{part}.weight"))?;
                    off.insert(format!("{o}.controllers.{c}.{part}_weight"), t);
                }
            }
        }
        let nvfp4 = layer < MXFP8_FROM;
        let width = d.expert_width;
        let w13 = quantize(rng, nvfp4, fold, (EXPERTS, 2 * width, d.hidden))?;
        let w2 = quantize(rng, nvfp4, fold, (EXPERTS, d.hidden, width))?;
        for (stored, q) in [("w13", &w13), ("w2", &w2)] {
            for (suffix, t) in &q.stored {
                off.insert(
                    format!("{o}.mlp.experts.{stored}_weight{suffix}"),
                    t.clone(),
                );
            }
        }
        // `w13`'s even rows are the gate's, its odd ones the up's
        let gate_up = w13.values.reshape((EXPERTS, width, 2, d.hidden))?;
        for e in 0..EXPERTS {
            let x = format!("{p}.mlp.experts.{e}");
            for (proj, half) in [("gate_proj", 0), ("up_proj", 1)] {
                take(&mut ours, &format!("{x}.{proj}.weight"))?;
                let rows = gate_up.get(e)?.narrow(1, half, 1)?.squeeze(1)?;
                theirs.insert(format!("{x}.{proj}.weight"), rows);
            }
            take(&mut ours, &format!("{x}.down_proj.weight"))?;
            theirs.insert(format!("{x}.down_proj.weight"), w2.values.get(e)?);
        }
    }
    // each looped layer i: its q's queries at rows 2i and 2i + 1 of one, its k's and v's of the other
    off.insert(
        "model.llm.middle_cycle_cross_pass_queries".to_string(),
        Tensor::cat(&cross_q, 0)?,
    );
    off.insert(
        "model.llm.middle_cycle_cross_pass_kv_queries".to_string(),
        Tensor::cat(&cross_kv, 0)?,
    );
    anyhow::ensure!(
        ours.is_empty(),
        "tiny tensors with no official name: {:?}",
        ours.keys()
    );
    Ok((off, theirs))
}

fn write_checkpoint(
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

fn official_checkpoint(official: &HashMap<String, Tensor>) -> anyhow::Result<tempfile::TempDir> {
    let config = serde_json::json!({
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "text_config": config(&OFFICIAL),
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    });
    write_checkpoint(config, official)
}

// the chat prompt's logprobs (more tokens than the experts' matvec takes, so a prompt runs the grouped kernels) and
// the decode steps' after it
async fn chat_logprobs(dir: &tempfile::TempDir, device: Option<&str>) -> anyhow::Result<Vec<f32>> {
    let model = builder(dir, device).build().await?;
    let tokens = model
        .tokenize_chat(ChatRequest::from(chat("say something long")).request)
        .await?
        .tokens;
    let mut scores = prompt_logprobs(&model, tokens).await?;
    let (decoded, _) = traces::greedy(&model, chat("say something long"), MAX_LEN).await?;
    scores.extend(decoded.iter().map(|(_, logprob)| *logprob));
    Ok(scores)
}

fn ensure_close(got: &[f32], want: &[f32], tolerance: f32) -> anyhow::Result<()> {
    for (i, (got, want)) in got.iter().zip(want).enumerate().skip(1) {
        anyhow::ensure!(
            (got - want).abs() < tolerance,
            "token {i}: {got}, expected {want}"
        );
    }
    Ok(())
}

// the official checkpoint's experts load as stored, their second-level scales apart: it scores as its decoded weights
// in the model's own layout
#[tokio::test]
async fn an_official_checkpoint_scores_as_its_decoded_weights() -> anyhow::Result<()> {
    // f32 KV, so both read the keys they computed (the CPU stores f16 by default)
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_nn::kv_cache::CPU_KV_F32_ENV, "1") };
    let (official, decoded) = official_weights(false)?;
    let decoded = decoded
        .into_iter()
        .map(|(name, tensor)| Ok((name, tensor.to_dtype(DType::F32)?)))
        .collect::<anyhow::Result<HashMap<_, _>>>()?;
    let expected = chat_logprobs(&write_checkpoint(config(&OFFICIAL), &decoded)?, None).await?;
    let scored = chat_logprobs(&official_checkpoint(&official)?, None).await?;
    ensure_close(&scored, &expected, LOGPROB_TOLERANCE)
}

// CUDA's NVFP4 kernels quantize their inputs, so they score against the same weights with each expert's scale folded
// into its block scales (a power of two, exact): the kernels apply the second level as the CPU does
#[tokio::test]
async fn official_experts_scales_apply_on_cuda_as_folded_into_their_blocks() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let (scaled, _) = official_weights(false)?;
    let (folded, _) = official_weights(true)?;
    let expected = chat_logprobs(&official_checkpoint(&folded)?, Some("cuda:0")).await?;
    let scored = chat_logprobs(&official_checkpoint(&scaled)?, Some("cuda:0")).await?;
    ensure_close(&scored, &expected, CUDA_LOGPROB_TOLERANCE)
}
