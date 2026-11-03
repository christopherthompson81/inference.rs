//! A tiny random-weight Nemotron-H (Mamba-2, NoPE attention, sigmoid-routed squared-ReLU experts with a shared
//! expert, a dense squared-ReLU MLP, and a Mamba layer with no feed-forward after it): decoding step by step through
//! its Mamba state and KV cache scores the tokens as one prefill over them does (quantized, with a prompt long enough
//! to run the experts grouped, too, and on CUDA, and with latent experts), calibrating an ISQ load collects
//! statistics for every expert projection, and its MTP head drafts on the CPU without changing greedy output.

use std::collections::HashMap;

use inference::{
    ChatRequest, IsqType, ModelDType, PromptInput, PromptLogitsRequest, RequestBuilder,
    TextMessageRole, TextMessages, TextModelBuilder,
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
// Mamba+MoE, Mamba alone, attention+MoE, Mamba+MLP
const LAYERS: [&str; 7] = ["mamba", "moe", "mamba", "attention", "moe", "mamba", "mlp"];
const VOCAB: usize = 266;
const HIDDEN: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
// a width FlashInfer decodes, so a BF16 model's paged decode steps capture as CUDA graphs
const GRAPH_HEAD_DIM: usize = 64;
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
// narrower than the model, as Super's and Bolt's latent experts are
const LATENT_WIDTH: usize = 16;
const MLP_WIDTH: usize = 48;
const WEIGHT_STD: f32 = 0.2;
// fixed, so every run checks the same checkpoint: decode and prefill round differently, and an unlucky draw of weights
// (a near-tie in the experts' routing) can put one step past a tolerance
const WEIGHT_SEED: u64 = 0x6e65_6d6f;
const MAX_LEN: usize = 6;
const LOGPROB_TOLERANCE: f32 = 1e-4;
// cuBLAS and the paged attention kernels add up in other orders than the CPU's
const CUDA_LOGPROB_TOLERANCE: f32 = 1e-3;
// Vulkan's ops sum in other orders than the CPU's (its attention, and its kernels as they land), as CUDA's do
const VULKAN_LOGPROB_TOLERANCE: f32 = 1e-3;
// an ISQ checkpoint's prompt reaches lm_head as one row, which Vulkan's matvec takes with f32 activations, where a
// whole-prompt score runs the CPU's kernel on int8 activations (0.013 apart on this checkpoint)
const VULKAN_QUANTIZED_LOGPROB_TOLERANCE: f32 = 0.05;
// BF16 with Q8_0 experts: decode and one prefill round differently (0.023 on this checkpoint), where a replay reading
// stale state is off by nats
const BF16_LOGPROB_TOLERANCE: f32 = 0.1;
const CIMATRIX: &str = "collected.cimatrix";
const MTP_N_PREDICT: usize = 2;
// enough steps that drafts are rejected at several positions, each replaying the Mamba state
const MTP_MAX_LEN: usize = 24;

/// The checkpoint's attention width and, for latent MoE, the routed experts' width.
#[derive(Clone, Copy)]
struct Shape {
    head_dim: usize,
    latent: Option<usize>,
}

impl Shape {
    const BASE: Self = Self {
        head_dim: HEAD_DIM,
        latent: None,
    };
}

fn config(shape: Shape) -> serde_json::Value {
    let head_dim = shape.head_dim;
    let mut config = serde_json::json!({
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "layers_block_type": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": head_dim,
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
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
    });
    if let Some(latent) = shape.latent {
        config["moe_latent_size"] = latent.into();
    }
    config
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

// NVIDIA's `backbone.*` names, as the Hub checkpoints keep them
fn weights(shape: Shape) -> anyhow::Result<HashMap<String, Tensor>> {
    let head_dim = shape.head_dim;
    let expert_in = shape.latent.unwrap_or(HIDDEN);
    let rng = &mut StdRng::seed_from_u64(WEIGHT_SEED);
    let inner = MAMBA_HEADS * MAMBA_HEAD_DIM;
    let conv_dim = inner + 2 * GROUPS * STATE;
    let mut weights = HashMap::from([
        (
            "backbone.embeddings.weight".to_string(),
            random(rng, &[VOCAB, HIDDEN])?,
        ),
        (
            "backbone.norm_f.weight".to_string(),
            near_one(rng, &[HIDDEN])?,
        ),
        ("lm_head.weight".to_string(), random(rng, &[VOCAB, HIDDEN])?),
    ]);
    for (i, kind) in LAYERS.iter().enumerate() {
        let layer = format!("backbone.layers.{i}");
        weights.insert(format!("{layer}.norm.weight"), near_one(rng, &[HIDDEN])?);
        let mixer = format!("{layer}.mixer");
        let mut tensors: Vec<(String, Tensor)> = match *kind {
            "mamba" => vec![
                (
                    "in_proj.weight".into(),
                    random(rng, &[inner + conv_dim + MAMBA_HEADS, HIDDEN])?,
                ),
                ("conv1d.weight".into(), random(rng, &[conv_dim, 1, CONV])?),
                ("conv1d.bias".into(), random(rng, &[conv_dim])?),
                ("dt_bias".into(), random(rng, &[MAMBA_HEADS])?),
                ("A_log".into(), random(rng, &[MAMBA_HEADS])?),
                ("D".into(), near_one(rng, &[MAMBA_HEADS])?),
                ("norm.weight".into(), near_one(rng, &[inner])?),
                ("out_proj.weight".into(), random(rng, &[HIDDEN, inner])?),
            ],
            "attention" => attention_tensors(rng, head_dim)?,
            "moe" => moe_tensors(rng, expert_in, shape.latent)?,
            _ => vec![
                ("up_proj.weight".into(), random(rng, &[MLP_WIDTH, HIDDEN])?),
                (
                    "down_proj.weight".into(),
                    random(rng, &[HIDDEN, MLP_WIDTH])?,
                ),
            ],
        };
        for (name, tensor) in tensors.drain(..) {
            weights.insert(format!("{mixer}.{name}"), tensor);
        }
    }
    // drawn after the stack's, so its weights stay those the other cases were tuned on
    let mtp = [
        ("mtp.layers.0.enorm.weight", near_one(rng, &[HIDDEN])?),
        ("mtp.layers.0.hnorm.weight", near_one(rng, &[HIDDEN])?),
        (
            "mtp.layers.0.eh_proj.weight",
            random(rng, &[HIDDEN, 2 * HIDDEN])?,
        ),
        ("mtp.layers.0.norm.weight", near_one(rng, &[HIDDEN])?),
        ("mtp.layers.1.norm.weight", near_one(rng, &[HIDDEN])?),
        (
            "mtp.layers.1.final_layernorm.weight",
            near_one(rng, &[HIDDEN])?,
        ),
    ];
    weights.extend(mtp.into_iter().map(|(name, t)| (name.to_string(), t)));
    for (name, tensor) in attention_tensors(rng, head_dim)? {
        weights.insert(format!("mtp.layers.0.mixer.{name}"), tensor);
    }
    for (name, tensor) in moe_tensors(rng, expert_in, shape.latent)? {
        weights.insert(format!("mtp.layers.1.mixer.{name}"), tensor);
    }
    Ok(weights)
}

fn attention_tensors(rng: &mut StdRng, head_dim: usize) -> anyhow::Result<Vec<(String, Tensor)>> {
    Ok(vec![
        (
            "q_proj.weight".into(),
            random(rng, &[HEADS * head_dim, HIDDEN])?,
        ),
        (
            "k_proj.weight".into(),
            random(rng, &[KV_HEADS * head_dim, HIDDEN])?,
        ),
        (
            "v_proj.weight".into(),
            random(rng, &[KV_HEADS * head_dim, HIDDEN])?,
        ),
        (
            "o_proj.weight".into(),
            random(rng, &[HIDDEN, HEADS * head_dim])?,
        ),
    ])
}

// experts reading `expert_in`-wide inputs, projected to and from `latent` when the MoE is latent
fn moe_tensors(
    rng: &mut StdRng,
    expert_in: usize,
    latent: Option<usize>,
) -> anyhow::Result<Vec<(String, Tensor)>> {
    let mut moe = vec![
        ("gate.weight".into(), random(rng, &[EXPERTS, HIDDEN])?),
        (
            "gate.e_score_correction_bias".into(),
            random(rng, &[EXPERTS])?,
        ),
        (
            "shared_experts.up_proj.weight".into(),
            random(rng, &[SHARED_WIDTH, HIDDEN])?,
        ),
        (
            "shared_experts.down_proj.weight".into(),
            random(rng, &[HIDDEN, SHARED_WIDTH])?,
        ),
    ];
    for e in 0..EXPERTS {
        moe.push((
            format!("experts.{e}.up_proj.weight"),
            random(rng, &[EXPERT_WIDTH, expert_in])?,
        ));
        moe.push((
            format!("experts.{e}.down_proj.weight"),
            random(rng, &[expert_in, EXPERT_WIDTH])?,
        ));
    }
    // drawn last, so the other checkpoints keep their weights
    if let Some(latent) = latent {
        moe.push((
            "fc1_latent_proj.weight".into(),
            random(rng, &[latent, HIDDEN])?,
        ));
        moe.push((
            "fc2_latent_proj.weight".into(),
            random(rng, &[HIDDEN, latent])?,
        ));
    }
    Ok(moe)
}

fn checkpoint() -> anyhow::Result<tempfile::TempDir> {
    checkpoint_with(Shape::BASE)
}

fn checkpoint_with(shape: Shape) -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for file in FIXTURE_FILES {
        std::fs::copy(format!("{FIXTURE}/{file}"), dir.path().join(file))?;
    }
    std::fs::copy(TOKENIZER, dir.path().join("tokenizer.json"))?;
    std::fs::write(dir.path().join("config.json"), config(shape).to_string())?;
    let weights = weights(shape)?
        .into_iter()
        .map(|(name, tensor)| Ok((name, tensor.to_dtype(DType::F32)?)))
        .collect::<anyhow::Result<HashMap<_, _>>>()?;
    inference_tensor::safetensors::save(&weights, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

#[tokio::test]
async fn a_step_by_step_decode_scores_as_one_prefill() -> anyhow::Result<()> {
    decode_scores_as_prefill(None, "say something long", Target::Cpu).await
}

// latent experts: down-projected before them, up after, with the router and the shared expert at the model's width
#[tokio::test]
async fn a_latent_moe_decode_scores_as_one_prefill() -> anyhow::Result<()> {
    decode_scores_as_prefill(None, "say something long", Target::CpuLatent).await
}

// quantized experts on the CPU run from their blocks, a long prompt's tiled and a decode step's per row
#[tokio::test]
async fn a_long_prompt_on_quantized_experts_scores_as_its_decode() -> anyhow::Result<()> {
    decode_scores_as_prefill(
        Some(IsqType::Q8_0),
        &"say something long ".repeat(LONG_PROMPT_REPEATS),
        Target::Cpu,
    )
    .await
}

// a CPU-only build runs paged attention on the CPU: decode steps through the block tables, the prompt through a gather
#[tokio::test]
async fn a_long_prompt_scores_as_its_cpu_paged_decode() -> anyhow::Result<()> {
    if cfg!(any(feature = "cuda", feature = "metal")) {
        return Ok(());
    }
    decode_scores_as_prefill(
        None,
        &"say something long ".repeat(LONG_PROMPT_REPEATS),
        Target::CpuPaged,
    )
    .await
}

// greedy decoding with the MTP head drafting through the CPU's paged attention matches decoding without it: a verify
// step that keeps fewer rows than it ran replays each Mamba layer's state from its stash over the kept rows
#[tokio::test]
async fn an_mtp_head_drafts_on_the_cpu_without_changing_greedy_output() -> anyhow::Result<()> {
    if cfg!(any(feature = "cuda", feature = "metal")) {
        return Ok(());
    }
    mtp_keeps_greedy_output(TextModelBuilder::with_force_cpu, LOGPROB_TOLERANCE).await
}

// as on the CPU, with Vulkan's paged cache writes and decode attention, its scan steps and its experts
#[tokio::test]
async fn an_mtp_head_drafts_on_vulkan_without_changing_greedy_output() -> anyhow::Result<()> {
    if !cfg!(feature = "vulkan") || Device::new_vulkan(0).is_err() {
        return Ok(());
    }
    mtp_keeps_greedy_output(|b| b.with_device("vulkan:0"), VULKAN_LOGPROB_TOLERANCE).await
}

async fn mtp_keeps_greedy_output(
    place: fn(TextModelBuilder) -> TextModelBuilder,
    tolerance: f32,
) -> anyhow::Result<()> {
    let dir = checkpoint()?;
    let builder = || -> anyhow::Result<TextModelBuilder> {
        Ok(place(
            TextModelBuilder::new(dir.path().to_string_lossy())
                .with_dtype(ModelDType::F32)
                .with_prefix_cache_n(None),
        )
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?))
    };
    let request = || {
        RequestBuilder::from(
            TextMessages::new().add_message(TextMessageRole::User, "say something long"),
        )
    };
    let drafting = builder()?
        .with_builtin_mtp(Some(MTP_N_PREDICT))
        .build()
        .await?;
    let (actual, _) = traces::greedy(&drafting, request(), MTP_MAX_LEN).await?;
    let stats = drafting.speculative_stats()?;
    let proposed: usize = stats.data.iter().map(|m| m.draft_tokens_proposed).sum();
    let accepted: usize = stats.data.iter().map(|m| m.draft_tokens_accepted).sum();
    drop(drafting);
    let plain = builder()?.build().await?;
    let (expected, _) = traces::greedy(&plain, request(), MTP_MAX_LEN).await?;
    anyhow::ensure!(proposed > 0, "the MTP head never drafted");
    anyhow::ensure!(
        accepted < proposed,
        "every draft was accepted ({accepted}), so no replay ran"
    );
    anyhow::ensure!(
        traces::close(&expected, &actual, tolerance),
        "drafting changed greedy decoding ({accepted} of {proposed} drafts accepted): {expected:?} vs {actual:?}"
    );
    Ok(())
}

// on CUDA through the hybrid cache and the scan kernel at a state narrower than a warp (grouped mmq against the gather
// path is a GPU case of the experts' backends: quantized, prefill and decode differ by the activations' rounding)
#[tokio::test]
async fn a_long_prompt_on_cuda_scores_as_its_decode() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    decode_scores_as_prefill(
        None,
        &"say something long ".repeat(LONG_PROMPT_REPEATS),
        Target::Cuda,
    )
    .await
}

// in BF16 with quantized experts (dense ones route on the host) the paged decode steps replay captured CUDA graphs,
// reading and writing the recurrent pool in place
#[tokio::test]
async fn a_long_prompt_scores_as_its_graph_decode() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") || Device::new_cuda(0).is_err() {
        return Ok(());
    }
    let snapshotter = decode_graphs::recorder();
    decode_scores_as_prefill(
        Some(IsqType::Q8_0),
        &"say something long ".repeat(LONG_PROMPT_REPEATS),
        Target::CudaGraphs,
    )
    .await?;
    decode_graphs::assert_replayed(&decode_graphs::Counters::take(&snapshotter));
    Ok(())
}

/// Where a decode-against-prefill check runs, which sets the checkpoint's attention width, the dtype and the tolerance.
#[derive(Clone, Copy)]
enum Target {
    Cpu,
    CpuLatent,
    CpuPaged,
    Cuda,
    CudaGraphs,
    Vulkan,
    VulkanQuantized,
}

impl Target {
    fn shape(self) -> Shape {
        match self {
            Self::CudaGraphs => Shape {
                head_dim: GRAPH_HEAD_DIM,
                latent: None,
            },
            Self::CpuLatent => Shape {
                head_dim: HEAD_DIM,
                latent: Some(LATENT_WIDTH),
            },
            Self::Cpu | Self::CpuPaged | Self::Cuda | Self::Vulkan | Self::VulkanQuantized => {
                Shape::BASE
            }
        }
    }

    fn dtype(self) -> ModelDType {
        match self {
            Self::CudaGraphs => ModelDType::BF16,
            Self::Cpu
            | Self::CpuLatent
            | Self::CpuPaged
            | Self::Cuda
            | Self::Vulkan
            | Self::VulkanQuantized => ModelDType::F32,
        }
    }

    fn tolerance(self) -> f32 {
        match self {
            Self::Cpu | Self::CpuLatent | Self::CpuPaged => LOGPROB_TOLERANCE,
            Self::Cuda => CUDA_LOGPROB_TOLERANCE,
            Self::Vulkan => VULKAN_LOGPROB_TOLERANCE,
            Self::VulkanQuantized => VULKAN_QUANTIZED_LOGPROB_TOLERANCE,
            Self::CudaGraphs => BF16_LOGPROB_TOLERANCE,
        }
    }
}

// the same checkpoint scores a long prompt on CUDA as on the CPU: the GPU's Mamba conv and scan kernels, attention and
// grouped experts against the CPU's
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
    let chat = RequestBuilder::from(TextMessages::new().add_message(
        TextMessageRole::User,
        "say something long ".repeat(LONG_PROMPT_REPEATS),
    ));
    let tokens = cpu
        .tokenize_chat(ChatRequest::from(chat).request)
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
    Ok(())
}

// Vulkan's decode steps (its Mamba conv and scan kernels on the state pools, its matvecs on Q8_0 weights) score as the
// model's own prefill does
#[tokio::test]
async fn a_vulkan_decode_scores_as_one_prefill() -> anyhow::Result<()> {
    if !cfg!(feature = "vulkan") || Device::new_vulkan(0).is_err() {
        return Ok(());
    }
    for (isq, target) in [
        (None, Target::Vulkan),
        (Some(IsqType::Q8_0), Target::VulkanQuantized),
        (Some(IsqType::Q5_0), Target::VulkanQuantized),
    ] {
        decode_scores_as_prefill(isq, "say something long", target)
            .await
            .map_err(|err| err.context(format!("{isq:?}")))?;
    }
    Ok(())
}

// the same checkpoint scores a long prompt on Vulkan as on the CPU, plain and with Q8_0 experts run from their blocks
#[tokio::test]
async fn a_long_prompt_scores_on_vulkan_as_on_the_cpu() -> anyhow::Result<()> {
    if !cfg!(feature = "vulkan") || Device::new_vulkan(0).is_err() {
        return Ok(());
    }
    // the CPU stores KV in f16 by default and Vulkan in f32, a difference a quantized checkpoint amplifies
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_nn::kv_cache::CPU_KV_F32_ENV, "1") };
    let dir = checkpoint()?;
    for isq in [None, Some(IsqType::Q8_0)] {
        let build = |device: &str| {
            let builder = TextModelBuilder::new(dir.path().to_string_lossy())
                .with_dtype(ModelDType::F32)
                .with_device(device);
            match isq {
                Some(isq) => builder.with_isq(isq),
                None => builder,
            }
            .build()
        };
        let (cpu, vulkan) = (build("cpu").await?, build("vulkan:0").await?);
        let chat = RequestBuilder::from(TextMessages::new().add_message(
            TextMessageRole::User,
            "say something long ".repeat(LONG_PROMPT_REPEATS),
        ));
        let tokens = cpu
            .tokenize_chat(ChatRequest::from(chat).request)
            .await?
            .tokens;
        let (expected, actual) = (
            scores(&cpu, tokens.clone()).await?,
            scores(&vulkan, tokens).await?,
        );
        for (position, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            if let (Some(expected), Some(actual)) = (expected, actual) {
                anyhow::ensure!(
                    (expected - actual).abs() < VULKAN_LOGPROB_TOLERANCE,
                    "{isq:?} position {position}: CPU {expected}, Vulkan {actual}"
                );
            }
        }
    }
    Ok(())
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

async fn decode_scores_as_prefill(
    isq: Option<IsqType>,
    prompt: &str,
    target: Target,
) -> anyhow::Result<()> {
    let dir = checkpoint_with(target.shape())?;
    // no prefix cache, so the scoring request is one prefill rather than a suffix over the decode steps' state
    let mut builder = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(target.dtype())
        .with_prefix_cache_n(None);
    match target {
        Target::Cpu | Target::CpuLatent => builder = builder.with_force_cpu(),
        Target::CpuPaged => {
            builder = builder
                .with_force_cpu()
                .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        }
        Target::CudaGraphs => {
            builder =
                builder.with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        }
        Target::Cuda => {}
        Target::Vulkan | Target::VulkanQuantized => builder = builder.with_device("vulkan:0"),
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
            (decoded - prefilled).abs() < target.tolerance(),
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
