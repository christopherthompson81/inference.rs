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
// the GGUF load's kernels against the checkpoint load's, the same weights (an unquantized recipe scored ~1e-3 off)
const RECIPE_LOGPROB_TOLERANCE: f32 = 5e-3;
const RECIPE_FILE: &str = "recipe.gguf";
const RECIPE_PROMPT_TOKENS: u32 = 24;
// Vulkan's ops sum in other orders than the CPU's
const VULKAN_LOGPROB_TOLERANCE: f32 = 1e-3;
// CUDA's matmuls and the fused attention sum in other orders too
const CUDA_LOGPROB_TOLERANCE: f32 = 1e-3;
const LONG_PROMPT_REPEATS: usize = 6;
// a prompt and a decode past several of the tiny window's rings (its window and a 256-row prefill piece): ~1000
// tokens of prompt
const RING_PROMPT_REPEATS: usize = 40;
const RING_DECODE_LEN: usize = 700;
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
    // the checkpoint as stored: these tests are of its own load, which a recipe would replace
    let builder = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_prefix_cache_n(None)
        .with_load_recipes(false);
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

// paged CUDA keeps each sliding slot's window in a ring of the window and a 256-row prefill piece: a prompt several
// rings long scores as the CPU's whole context does (its prefill in pieces through the ring's wraps), and a decode
// as long scores as one prefill of it does (the decode kernel through them). The CPU isn't the decode's reference:
// near-tied routes flip between it and CUDA for a few positions of a long decode.
#[tokio::test]
async fn windowed_paged_cuda_scores_as_on_the_cpu() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_nn::kv_cache::CPU_KV_F32_ENV, "1") };
    let (dir, paged) = paged_model(ModelDType::F32).await?;
    let cpu = builder(&dir, None).build().await?;
    let prompt = "say something long ".repeat(RING_PROMPT_REPEATS);
    let tokens = cpu
        .tokenize_chat(ChatRequest::from(chat(&prompt)).request)
        .await?
        .tokens;
    let (on_cpu, on_cuda) = (
        prompt_logprobs(&cpu, tokens.clone()).await?,
        prompt_logprobs(&paged, tokens).await?,
    );
    for (position, (cpu, cuda)) in on_cpu.iter().zip(&on_cuda).enumerate().skip(1) {
        anyhow::ensure!(
            (cpu - cuda).abs() < CUDA_LOGPROB_TOLERANCE,
            "prompt position {position}: CPU {cpu}, windowed CUDA {cuda}"
        );
    }
    let request = || chat("say something long").set_sampler_ignore_eos(true);
    let (decoded, _) = traces::greedy(&paged, request(), RING_DECODE_LEN).await?;
    let mut tokens = paged
        .tokenize_chat(ChatRequest::from(request()).request)
        .await?
        .tokens;
    tokens.extend(decoded.iter().map(|(token, _)| *token));
    let scored = prompt_logprobs(&paged, tokens).await?;
    for (step, ((_, decoded), prefilled)) in decoded
        .iter()
        .zip(&scored[scored.len() - decoded.len()..])
        .enumerate()
    {
        anyhow::ensure!(
            (decoded - prefilled).abs() < CUDA_LOGPROB_TOLERANCE,
            "decode step {step}: decoded {decoded}, one prefill {prefilled}"
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

// the official checkpoint loads as its recipe, built once into the recipe directory and read from there after
#[tokio::test]
async fn an_official_checkpoint_loads_as_its_recipe_once_built() -> anyhow::Result<()> {
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_nn::kv_cache::CPU_KV_F32_ENV, "1") };
    let (official, _) = official_weights(false)?;
    let dir = official_checkpoint(&official)?;
    let recipes = tempfile::tempdir()?;
    let built = || -> anyhow::Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(recipes.path())? {
            for file in std::fs::read_dir(entry?.path())? {
                files.push(file?.path());
            }
        }
        Ok(files)
    };
    let recipe = || {
        builder(&dir, None)
            .with_load_recipes(true)
            .with_recipe_dir(recipes.path())
    };
    let model = recipe().build().await?;
    let files = built()?;
    anyhow::ensure!(
        files.len() == 1 && files[0].extension().is_some_and(|e| e == "gguf"),
        "one recipe GGUF, built {files:?}"
    );
    // ids rather than text: the fixture's vocabulary is no gpt2 byte-level one for the GGUF tokenizer to split into
    let tokens = (0..RECIPE_PROMPT_TOKENS).collect::<Vec<u32>>();
    let first = prompt_logprobs(&model, tokens.clone()).await?;
    drop(model);
    let modified = std::fs::metadata(&files[0])?.modified()?;
    let again = recipe().build().await?;
    anyhow::ensure!(
        built()? == files && std::fs::metadata(&files[0])?.modified()? == modified,
        "a second load reads the built recipe"
    );
    ensure_close(
        &prompt_logprobs(&again, tokens).await?,
        &first,
        LOGPROB_TOLERANCE,
    )
}

// the checkpoint written with its dense tensors unquantized, under the GGUF's names and layouts, scores the
// checkpoint's tokens as the checkpoint does (the recipe's mapping; recipe C quantizes the same tensors)
#[tokio::test]
async fn an_unquantized_recipe_scores_as_its_checkpoint() -> anyhow::Result<()> {
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_nn::kv_cache::CPU_KV_F32_ENV, "1") };
    let (official, _) = official_weights(false)?;
    let dir = official_checkpoint(&official)?;
    let stored = builder(&dir, None).build().await?;
    let tokens = stored
        .tokenize_chat(ChatRequest::from(chat("say something long")).request)
        .await?
        .tokens;
    let want = prompt_logprobs(&stored, tokens.clone()).await?;
    drop(stored);
    let out = tempfile::tempdir()?;
    inference_api::gguf_requantize::write_inkling2_recipe(
        dir.path(),
        &out.path().join(RECIPE_FILE),
        inference_api::gguf_requantize::Inkling2Recipe::Unquantized,
        None,
        |_, _, _| {},
    )?;
    let gguf = inference::GgufModelBuilder::new(out.path().to_string_lossy(), vec![RECIPE_FILE])
        .with_dtype(ModelDType::F32)
        .with_prefix_cache_n(None)
        .with_force_cpu()
        .build()
        .await?;
    ensure_close(
        &prompt_logprobs(&gguf, tokens).await?,
        &want,
        RECIPE_LOGPROB_TOLERANCE,
    )
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

// the checkpoint's own drafter (`mtp.safetensors`, random): two layers over the taps before visits 1 and 5 and the
// normed output (the tiny model's 10 visits), blocks of 4
const DRAFT_LAYERS: usize = 2;
const DRAFT_BLOCK: usize = 4;
const DRAFT_RANK: usize = 8;
const DRAFT_TAPS: [usize; 3] = [1, 5, 10];
const DRAFT_SEED: u64 = 0x6d74_7000;
// past the bias's extent and the drafter's window, through many rejected and partly accepted blocks
const DRAFT_DECODE_LEN: usize = 24;
// past the drafting the tuner starts with, a plain stretch and a probe
const AUTO_DECODE_LEN: usize = 160;
const DRAFT_LOGPROB_TOLERANCE: f32 = 0.06;

fn drafter_config() -> serde_json::Value {
    serde_json::json!({
        "hidden_size": OFFICIAL.hidden,
        "num_hidden_layers": DRAFT_LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": OFFICIAL.expert_width,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "local_layer_ids": [0],
        "sliding_window_size": WINDOW,
        "block_size": DRAFT_BLOCK,
        "markov_rank": DRAFT_RANK,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": DRAFT_TAPS,
    })
}

fn drafter_weights() -> anyhow::Result<HashMap<String, Tensor>> {
    let rng = &mut StdRng::seed_from_u64(DRAFT_SEED);
    let (h, r) = (OFFICIAL.hidden, DRAFT_RANK);
    let (q_dim, kv_dim) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let mut w = HashMap::new();
    w.insert(
        "fuse_weight".to_string(),
        random(rng, &[DRAFT_LAYERS + 1, DRAFT_TAPS.len()])?,
    );
    w.insert("mask_embed".to_string(), random(rng, &[h])?);
    w.insert("norm.weight".to_string(), near_one(rng, &[h])?);
    for i in 0..DRAFT_LAYERS {
        let p = format!("layers.{i}");
        for norm in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
            w.insert(format!("{p}.{norm}.weight"), near_one(rng, &[h])?);
        }
        for norm in ["q_norm", "k_norm", "k_norm_ctx"] {
            w.insert(
                format!("{p}.self_attn.{norm}.weight"),
                near_one(rng, &[HEAD_DIM])?,
            );
        }
        for (name, rows, cols) in [
            ("q_proj", q_dim, h),
            ("k_proj", kv_dim, h),
            ("v_proj", kv_dim, h),
            ("k_proj_ctx", kv_dim, h),
            ("v_proj_ctx", kv_dim, h),
            ("o_proj", h, q_dim),
        ] {
            w.insert(
                format!("{p}.self_attn.{name}.weight"),
                random(rng, &[rows, cols])?,
            );
        }
        let width = OFFICIAL.expert_width;
        for (name, rows, cols) in [
            ("gate_proj", width, h),
            ("up_proj", width, h),
            ("down_proj", h, width),
        ] {
            w.insert(
                format!("{p}.mlp.{name}.weight"),
                random(rng, &[rows, cols])?,
            );
        }
    }
    let m = "markov_head";
    w.insert(format!("{m}.dflare_norm.weight"), near_one(rng, &[h])?);
    w.insert(
        format!("{m}.markov_w1.weight"),
        random(rng, &[PADDED_VOCAB, r])?,
    );
    w.insert(
        format!("{m}.rnn_proj.weight"),
        random(rng, &[3 * r, 2 * r + 2 * h])?,
    );
    w.insert(format!("{m}.rnn_proj.bias"), random(rng, &[3 * r])?);
    w.insert(format!("{m}.hidden_out.weight"), random(rng, &[h, r])?);
    w.insert(format!("{m}.hidden_out.bias"), random(rng, &[h])?);
    Ok(w)
}

#[derive(Clone, Copy, PartialEq)]
enum Drafting {
    Off,
    Fixed,
    Auto,
}

async fn paged_trace(
    dir: &tempfile::TempDir,
    drafting: Drafting,
    request: RequestBuilder,
    len: usize,
) -> anyhow::Result<traces::Trace> {
    paged_trace_on(dir, drafting, request, len, inference::PagedCacheType::Auto).await
}

async fn paged_trace_on(
    dir: &tempfile::TempDir,
    drafting: Drafting,
    request: RequestBuilder,
    len: usize,
    cache: inference::PagedCacheType,
) -> anyhow::Result<traces::Trace> {
    // BF16: the drafter's paged attention reads half-precision caches (or q8 ones)
    let mut builder = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_load_recipes(false)
        .with_dtype(ModelDType::BF16)
        .with_prefix_cache_n(None)
        .with_paged_attn(
            inference::PagedAttentionMetaBuilder::default()
                .with_paged_cache_type(cache)
                .build()?,
        );
    if drafting != Drafting::Off {
        builder = builder.with_mtp_model(dir.path().to_string_lossy(), Some(DRAFT_BLOCK - 1));
    }
    if drafting == Drafting::Auto {
        builder = builder.with_mtp_auto_depth();
    }
    let model = builder.build().await?;
    Ok(traces::greedy(&model, request, len).await?.0)
}

fn ensure_same_greedy(drafted: &traces::Trace, plain: &traces::Trace) -> anyhow::Result<()> {
    // a verify step's rows sum in other orders than a decode step's: equal up to a near tie, which may flip
    for (step, (d, p)) in drafted.iter().zip(plain).enumerate() {
        anyhow::ensure!(
            (d.1 - p.1).abs() < DRAFT_LOGPROB_TOLERANCE,
            "step {step}: drafted {drafted:?}, plain {plain:?}"
        );
        if d.0 != p.0 {
            return Ok(());
        }
    }
    anyhow::ensure!(
        drafted.len() == plain.len(),
        "drafted {drafted:?}, plain {plain:?}"
    );
    Ok(())
}

fn drafter_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let (official, _) = official_weights(false)?;
    let config = serde_json::json!({
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "text_config": config(&OFFICIAL),
        "mtp_config": drafter_config(),
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    });
    let dir = write_checkpoint(config, &official)?;
    inference_tensor::safetensors::save(&drafter_weights()?, dir.path().join("mtp.safetensors"))?;
    Ok(dir)
}

// drafting from the checkpoint's own drafter keeps the greedy output: each verify step's rejected rows are cut from
// every slot's convolution states
#[tokio::test]
async fn the_checkpoint_drafter_keeps_greedy_output_on_cuda() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let dir = drafter_checkpoint()?;
    let request = || chat("say something long");
    let plain = paged_trace(&dir, Drafting::Off, request(), DRAFT_DECODE_LEN).await?;
    let drafted = paged_trace(&dir, Drafting::Fixed, request(), DRAFT_DECODE_LEN).await?;
    ensure_same_greedy(&drafted, &plain)
}

// on a q8 cache the drafter's paged layers are q8 too (written quantized, read through rotated queries), and drafting
// still keeps the greedy output
#[tokio::test]
async fn the_checkpoint_drafter_keeps_greedy_output_on_a_q8_cache_on_cuda() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let dir = drafter_checkpoint()?;
    let q8 = inference::PagedCacheType::Quant { k: 8, v: 8 };
    let request = || chat("say something long");
    let plain = paged_trace_on(&dir, Drafting::Off, request(), DRAFT_DECODE_LEN, q8).await?;
    let drafted = paged_trace_on(&dir, Drafting::Fixed, request(), DRAFT_DECODE_LEN, q8).await?;
    ensure_same_greedy(&drafted, &plain)
}

// a packed cache (q4 here) serves plain and drafted decodes alike; K and V at different widths fail the load, not every
// request after it
#[tokio::test]
async fn a_q4_cache_keeps_drafted_output_and_mixed_widths_fail_the_load_on_cuda()
-> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let dir = drafter_checkpoint()?;
    let q4 = inference::PagedCacheType::Quant { k: 4, v: 4 };
    let request = || chat("say something long");
    let plain = paged_trace_on(&dir, Drafting::Off, request(), DRAFT_DECODE_LEN, q4).await?;
    anyhow::ensure!(!plain.is_empty(), "a q4 cache decoded nothing");
    let drafted = paged_trace_on(&dir, Drafting::Fixed, request(), DRAFT_DECODE_LEN, q4).await?;
    ensure_same_greedy(&drafted, &plain)?;
    let mixed = inference::PagedCacheType::Quant { k: 8, v: 4 };
    let error = paged_trace_on(&dir, Drafting::Off, chat("hi"), 1, mixed)
        .await
        .expect_err("a q8/q4 cache loads");
    anyhow::ensure!(format!("{error:#}").contains("not q8/q4"), "{error:#}");
    Ok(())
}

// choosing between drafting and plain decode each step keeps the greedy output: the random drafter's misses switch
// it to plain steps, whose rows the drafter buffers, and probes draft again from them
#[tokio::test]
async fn auto_depth_keeps_greedy_output_through_plain_steps_on_cuda() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let dir = drafter_checkpoint()?;
    // the tiny model soon samples its EOS
    let request = || chat("say something long").set_sampler_ignore_eos(true);
    let plain = paged_trace(&dir, Drafting::Off, request(), AUTO_DECODE_LEN).await?;
    let auto = paged_trace(&dir, Drafting::Auto, request(), AUTO_DECODE_LEN).await?;
    ensure_same_greedy(&auto, &plain)
}

// a tiny tower: 8 px cells (4 px patches merged 2 x 2) over a 4 x 4 position grid, so most pictures resize it
const TOWER_HIDDEN: usize = 64;
const TOWER_LAYERS: usize = 2;
const TOWER_HEADS: usize = 1;
const TOWER_PATCH: usize = 4;
const TOWER_GRID: usize = 4;
const TOWER_MLP_RATIO: usize = 2;
const TOWER_PROJECTION: usize = 64;
const TOWER_SEED: u64 = 0x7669_7369;
// the fixture's last added token (`<|video_pad|>`) renamed, inside the vocabulary as the real pad is
const IMAGE_PAD_ID: u32 = 265;
const IMAGE_PAD: &str = "<|unused_200054|>";
// pictures of 3 x 2 and 2 x 2 cells
const PICTURE: (u32, u32) = (20, 12);
const OTHER_PICTURE: (u32, u32) = (16, 9);
const PICTURE_CELLS: usize = 6;
const OTHER_PICTURE_CELLS: usize = 4;
const PICTURE_PROMPT: &str = "what is in the picture";
const VISION_TEMPLATE: &str = "{% for m in messages %}{% if m.content is string %}{{ m.content }}\
{% else %}{% for p in m.content %}{% if p.type == 'text' %}{{ p.text }}{% else %}<|unused_200054|>{% endif %}\
{% endfor %}{% endif %}{% endfor %}";
// cached KV comes from another prefill than a full recompute, so logprobs match only to rounding
const CACHED_LOGPROB_TOLERANCE: f32 = 1e-3;

fn tower_config(text_hidden: usize) -> serde_json::Value {
    serde_json::json!({
        "hidden_size": TOWER_HIDDEN,
        "num_hidden_layers": TOWER_LAYERS,
        "num_attention_heads": TOWER_HEADS,
        "mlp_ratio": TOWER_MLP_RATIO as f64,
        "native_patch_size": TOWER_PATCH,
        "adapter_merge_size": 2,
        "posemb_grid_size": TOWER_GRID,
        "projection_hidden_dim": TOWER_PROJECTION,
        "projector_bias": false,
        "text_hidden_size": text_hidden,
        "layer_norm_eps": 1e-5,
    })
}

fn tower_weights(text_hidden: usize) -> anyhow::Result<HashMap<String, Tensor>> {
    let rng = &mut StdRng::seed_from_u64(TOWER_SEED);
    let (d, f) = (TOWER_HIDDEN, TOWER_HIDDEN * TOWER_MLP_RATIO);
    let mut w = HashMap::new();
    let e = "model.visual.backbone.embeddings";
    w.insert(format!("{e}.class_embedding"), random(rng, &[d])?);
    w.insert(format!("{e}.cls_position_embedding"), random(rng, &[d])?);
    w.insert(
        format!("{e}.patch_embedding.weight"),
        random(rng, &[d, 3 * TOWER_PATCH * TOWER_PATCH])?,
    );
    w.insert(
        format!("{e}.position_embedding"),
        random(rng, &[TOWER_GRID * TOWER_GRID, d])?,
    );
    for norm in ["weight", "bias"] {
        let t = if norm == "weight" {
            near_one(rng, &[d])?
        } else {
            random(rng, &[d])?
        };
        w.insert(format!("model.visual.backbone.ln_pre.{norm}"), t);
    }
    for i in 0..TOWER_LAYERS {
        let l = format!("model.visual.backbone.layers.{i}");
        for norm in ["ln_1", "ln_2"] {
            w.insert(format!("{l}.{norm}.weight"), near_one(rng, &[d])?);
            w.insert(format!("{l}.{norm}.bias"), random(rng, &[d])?);
        }
        for scale in ["ls_1", "ls_2"] {
            w.insert(format!("{l}.{scale}.gamma"), random(rng, &[d])?);
        }
        for (name, rows, cols) in [
            ("self_attn.q_proj", d, d),
            ("self_attn.k_proj", d, d),
            ("self_attn.v_proj", d, d),
            ("self_attn.out_proj", d, d),
            ("mlp.c_fc", f, d),
            ("mlp.c_proj", d, f),
        ] {
            w.insert(format!("{l}.{name}.weight"), random(rng, &[rows, cols])?);
            w.insert(format!("{l}.{name}.bias"), random(rng, &[rows])?);
        }
    }
    w.insert(
        "model.visual.proj_in.weight".into(),
        random(rng, &[TOWER_PROJECTION, 4 * d])?,
    );
    w.insert(
        "model.visual.proj_out.weight".into(),
        random(rng, &[text_hidden, TOWER_PROJECTION])?,
    );
    Ok(w)
}

// `dir`'s checkpoint given the tower: its config, weights, a tokenizer with the pad and a template that writes it
fn add_tower(
    dir: &tempfile::TempDir,
    weights: &HashMap<String, Tensor>,
    text_hidden: usize,
) -> anyhow::Result<()> {
    let config_path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    if config.get("text_config").is_none() {
        config = serde_json::json!({"text_config": config});
    }
    config["architectures"] = serde_json::json!(["Inkling2LoopForConditionalGeneration"]);
    config["model_type"] = serde_json::json!("inkling2_loop");
    config["vision_config"] = tower_config(text_hidden);
    config["image_token_id"] = serde_json::json!(IMAGE_PAD_ID);
    std::fs::write(&config_path, config.to_string())?;
    let tokenizer_path = dir.path().join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&tokenizer_path)?)?;
    let pad = tokenizer["added_tokens"]
        .as_array_mut()
        .and_then(|tokens| tokens.iter_mut().find(|t| t["id"] == IMAGE_PAD_ID))
        .ok_or_else(|| anyhow::anyhow!("the tokenizer has no added token {IMAGE_PAD_ID}"))?;
    let old = pad["content"].as_str().unwrap_or_default().to_string();
    pad["content"] = serde_json::json!(IMAGE_PAD);
    if let Some(vocab) = tokenizer["model"]["vocab"].as_object_mut()
        && let Some(id) = vocab.remove(&old)
    {
        vocab.insert(IMAGE_PAD.to_string(), id);
    }
    std::fs::write(&tokenizer_path, tokenizer.to_string())?;
    std::fs::write(dir.path().join("chat_template.jinja"), VISION_TEMPLATE)?;
    // the text weights as given (an official checkpoint's experts stay packed bytes)
    let mut all = weights.clone();
    all.extend(tower_weights(text_hidden)?);
    inference_tensor::safetensors::save(&all, dir.path().join("model.safetensors"))?;
    Ok(())
}

fn picture((width, height): (u32, u32)) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([
            (x * 37 % 251) as u8,
            (y * 53 % 241) as u8,
            ((x + y) * 29 % 239) as u8,
        ])
    }))
}

fn picture_chat(size: (u32, u32)) -> RequestBuilder {
    RequestBuilder::from(inference::MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PICTURE_PROMPT,
        vec![picture(size)],
    ))
}

// paged on CUDA (where a repeat reads the prefix cache), else on the CPU
async fn vision_model(dir: &tempfile::TempDir) -> anyhow::Result<(inference::Model, bool)> {
    let mut builder = inference::MultimodalModelBuilder::new(dir.path().to_string_lossy())
        .with_load_recipes(false)
        .with_dtype(ModelDType::F32)
        .with_prefix_cache_n(Some(4));
    let paged = cfg!(feature = "cuda") && Device::new_cuda(0).is_ok();
    if paged {
        builder = builder.with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?);
    } else {
        builder = builder.with_force_cpu();
    }
    Ok((builder.build().await?, paged))
}

// a picture takes one pad a cell, its rows steer the reply, and a repeat served from the prefix cache decodes the same
#[tokio::test]
async fn a_picture_expands_to_its_cells_steers_the_reply_and_repeats_from_the_cache()
-> anyhow::Result<()> {
    let dir = checkpoint()?;
    add_tower(&dir, &weights(&TINY)?, TINY.hidden)?;
    let (model, paged) = vision_model(&dir).await?;
    let text = model
        .send_chat_request(chat(PICTURE_PROMPT).set_sampler_max_len(1))
        .await?
        .usage
        .prompt_tokens;
    let first = model
        .send_chat_request(picture_chat(PICTURE).set_sampler_max_len(1))
        .await?
        .usage
        .prompt_tokens;
    assert_eq!(first - text, PICTURE_CELLS, "the picture's pads");
    let (traced, _) = traces::greedy(&model, picture_chat(PICTURE), MAX_LEN).await?;
    let (repeat, cached) = traces::greedy(&model, picture_chat(PICTURE), MAX_LEN).await?;
    assert!(
        !paged || cached > 0,
        "the repeat read nothing from the prefix cache"
    );
    assert_eq!(traced.len(), repeat.len());
    for ((a, la), (b, lb)) in traced.iter().zip(&repeat) {
        assert_eq!(a, b);
        assert!((la - lb).abs() < CACHED_LOGPROB_TOLERANCE, "{la} vs {lb}");
    }
    let (other, _) = traces::greedy(&model, picture_chat(OTHER_PICTURE), MAX_LEN).await?;
    assert!(
        traced
            .iter()
            .zip(&other)
            .any(|(a, b)| a.0 != b.0 || (a.1 - b.1).abs() > CACHED_LOGPROB_TOLERANCE),
        "another picture left the reply as it was: {traced:?}"
    );
    Ok(())
}

// with no prefix cache to reuse, a repeated picture's rows come from the encoder cache and decode the same
#[tokio::test]
async fn a_repeated_picture_takes_its_rows_from_the_encoder_cache() -> anyhow::Result<()> {
    let dir = checkpoint()?;
    add_tower(&dir, &weights(&TINY)?, TINY.hidden)?;
    let model = inference::MultimodalModelBuilder::new(dir.path().to_string_lossy())
        .with_load_recipes(false)
        .with_dtype(ModelDType::F32)
        .with_prefix_cache_n(None)
        .with_force_cpu()
        .build()
        .await?;
    let encoder_cache = |model: &inference::Model| -> anyhow::Result<(usize, usize)> {
        let stats = model.cache_stats()?;
        let cache = stats.data[0]
            .encoder_cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no encoder cache"))?;
        Ok((cache.hits, cache.misses))
    };
    let (first, _) = traces::greedy(&model, picture_chat(PICTURE), MAX_LEN).await?;
    let (hits, misses) = encoder_cache(&model)?;
    assert!(
        hits == 0 && misses > 0,
        "first picture: {hits} hits, {misses} misses"
    );
    let (second, _) = traces::greedy(&model, picture_chat(PICTURE), MAX_LEN).await?;
    let (repeat_hits, repeat_misses) = encoder_cache(&model)?;
    assert!(
        repeat_hits > 0 && repeat_misses == misses,
        "repeated picture: {repeat_hits} hits, {repeat_misses} misses"
    );
    assert_eq!(first, second, "cached rows changed the reply");
    Ok(())
}

// drafting over a prompt with a picture keeps the greedy output, verify steps rolling the conv states back as without
#[tokio::test]
async fn the_checkpoint_drafter_keeps_greedy_output_after_a_picture_on_cuda() -> anyhow::Result<()>
{
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let dir = drafter_checkpoint()?;
    let (official, _) = official_weights(false)?;
    add_tower(&dir, &official, OFFICIAL.hidden)?;
    let dir = &dir;
    let trace = |drafting: Drafting| async move {
        let mut builder = inference::MultimodalModelBuilder::new(dir.path().to_string_lossy())
            .with_load_recipes(false)
            .with_dtype(ModelDType::BF16)
            .with_prefix_cache_n(None)
            .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?);
        if drafting != Drafting::Off {
            builder = builder.with_mtp_model(dir.path().to_string_lossy(), Some(DRAFT_BLOCK - 1));
        }
        let model = builder.build().await?;
        anyhow::Ok(
            traces::greedy(&model, picture_chat(PICTURE), DRAFT_DECODE_LEN)
                .await?
                .0,
        )
    };
    let plain = trace(Drafting::Off).await?;
    let drafted = trace(Drafting::Fixed).await?;
    ensure_same_greedy(&drafted, &plain)
}

// asked for, the recipe brings the checkpoint's tower: its projector written once beside the recipe, the load reading
// pictures through it
#[tokio::test]
async fn a_recipe_load_asked_for_vision_writes_the_projector_once_and_reads_pictures()
-> anyhow::Result<()> {
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_nn::kv_cache::CPU_KV_F32_ENV, "1") };
    let (official, _) = official_weights(false)?;
    let dir = official_checkpoint(&official)?;
    add_tower(&dir, &official, OFFICIAL.hidden)?;
    let recipes = tempfile::tempdir()?;
    let built = || -> anyhow::Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(recipes.path())? {
            for file in std::fs::read_dir(entry?.path())? {
                files.push(file?.path());
            }
        }
        files.sort();
        Ok(files)
    };
    let recipe = |vision: bool| {
        builder(&dir, None)
            .with_load_recipes(true)
            .with_recipe_vision(vision)
            .with_recipe_dir(recipes.path())
    };
    // text only by default: no projector written
    drop(recipe(false).build().await?);
    anyhow::ensure!(built()?.len() == 1, "a text load wrote {:?}", built()?);
    let model = recipe(true).build().await?;
    let files = built()?;
    anyhow::ensure!(
        files.len() == 2
            && files.iter().any(|f| f
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("mmproj-"))),
        "the recipe and its projector, built {files:?}"
    );
    // pictures against each other: the GGUF tokenizer splits none of the fixture's text
    let mut tokens = Vec::new();
    for size in [PICTURE, OTHER_PICTURE] {
        tokens.push(
            model
                .send_chat_request(picture_chat(size).set_sampler_max_len(1))
                .await?
                .usage
                .prompt_tokens,
        );
    }
    anyhow::ensure!(
        tokens[0] - tokens[1] == PICTURE_CELLS - OTHER_PICTURE_CELLS,
        "prompt tokens {tokens:?}"
    );
    drop(model);
    let modified = files
        .iter()
        .map(|f| Ok(std::fs::metadata(f)?.modified()?))
        .collect::<anyhow::Result<Vec<_>>>()?;
    drop(recipe(true).build().await?);
    let again = built()?
        .iter()
        .map(|f| Ok(std::fs::metadata(f)?.modified()?))
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(again == modified, "a second load reads the built projector");
    Ok(())
}
