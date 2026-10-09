//! A tiny random-weight Qwen3.5 text checkpoint: the one text loader whose runtime config differs from the checkpoint's.

use inference::{
    IsqType, Model, ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};
use inference_models_qwen::qwen3_5::{Qwen3_5TextModel, TextConfig};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "../support/decode_graphs.rs"]
mod decode_graphs;
#[path = "../support/recording.rs"]
mod recording;
#[path = "../support/traces.rs"]
mod traces;

const DECLARED_CONTEXT: u64 = 1024;
const RUNTIME_LIMIT: usize = 256;
const NUM_LAYERS: usize = 4;
const VOCAB: usize = 266;
// the target's hidden size, which a DFlash drafter must share
const HIDDEN: usize = 64;
const HEAD_DIM: usize = 32;
// 32 decodes through the gather, which a graph cannot capture
const GRAPH_HEAD_DIM: usize = 64;
const PARTIAL_ROTARY_FACTOR: f64 = 0.25;
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn tiny_qwen3_5_text(head_dim: usize, mtp: bool) -> anyhow::Result<tempfile::TempDir> {
    let rotary_pairs = (head_dim as f64 * PARTIAL_ROTARY_FACTOR) as usize / 2;
    // split 2:1:1 over the rotated pairs, the shape of Qwen3.5's own sections
    let section = rotary_pairs / 4;
    let mut config = serde_json::json!({
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "head_dim": head_dim,
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 128,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "hidden_act": "silu",
        "max_position_embeddings": DECLARED_CONTEXT,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
            "mrope_section": [2 * section, section, section]
        },
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "tie_word_embeddings": false
    });
    if mtp {
        config["mtp_num_hidden_layers"] = 1.into();
    }
    let scratch = tempfile::tempdir()?;
    let config_path = scratch.path().join("config.json");
    std::fs::write(&config_path, config.to_string())?;
    let files = [
        config_path,
        format!("{FIXTURES}/paddleocr_vl/tiny/tokenizer.json").into(),
        format!("{FIXTURES}/llama_tiny/tokenizer_config.json").into(),
        format!("{FIXTURES}/llama_tiny/chat_template.jinja").into(),
    ];
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    let cfg: TextConfig = serde_json::from_value(config)?;
    recording::record_checkpoint_seeded_by_name(
        &files,
        NUM_LAYERS + usize::from(mtp),
        &[],
        |vb, metadata| {
            Qwen3_5TextModel::new(
                &cfg,
                vb,
                false,
                mtp,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )
}

#[tokio::test]
async fn a_uqff_keeps_the_checkpoint_context_when_written_under_max_model_len() -> anyhow::Result<()>
{
    let checkpoint = tiny_qwen3_5_text(HEAD_DIM, false)?;
    let uqff = tempfile::tempdir()?;
    TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_isq(IsqType::Q8_0)
        .with_max_model_len(RUNTIME_LIMIT)
        .write_uqff(uqff.path().join("model.uqff"))
        .build()
        .await?;
    let written: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(uqff.path().join("config.json"))?)?;
    assert_eq!(written["max_position_embeddings"], DECLARED_CONTEXT);
    Ok(())
}

const GRAPH_MAX_LEN: usize = 8;
const MTP_N_PREDICT: usize = 2;
const DFLASH_INTERMEDIATE: usize = 128;
const DFLASH_HEADS: usize = 2;
const DFLASH_HEAD_DIM: usize = 64;
const DFLASH_BLOCK: usize = 4;
const DFLASH_N_PREDICT: usize = 2;
// the tokenizer's last id, never produced by the prompts
const DFLASH_MASK_TOKEN: u32 = VOCAB as u32 - 1;
const DFLASH_TAPS: [usize; 2] = [1, 3];
const DFLASH_WEIGHT_SCALE: f32 = 0.05;
const DFLASH_WEIGHT_STEP: f32 = 0.37;
// every drafter layer windowed, which its decode graphs need
const DFLASH_WINDOW: usize = 32;
const DFLASH_COMPONENT: &str = "dflash";

async fn paged_gpu_model(mtp: bool) -> anyhow::Result<(tempfile::TempDir, Model)> {
    let checkpoint = tiny_qwen3_5_text(GRAPH_HEAD_DIM, mtp)?;
    let builder = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?);
    let builder = if mtp {
        builder.with_builtin_mtp(Some(MTP_N_PREDICT))
    } else {
        builder
    };
    let model = builder.build().await?;
    Ok((checkpoint, model))
}

async fn paged_gpu_rounds() -> anyhow::Result<decode_graphs::Rounds> {
    let (_checkpoint, model) = paged_gpu_model(false).await?;
    decode_graphs::rounds(&model, GRAPH_MAX_LEN).await
}

// Each GRAPH_PROMPTS entry's greedy (token, logprob) per step.
const GRAPH_TRACES: [&[(u32, f32)]; decode_graphs::GRAPH_PROMPTS.len()] = [
    &[
        (9, -0.4501265),
        (193, -0.76926416),
        (189, -0.28025556),
        (217, -1.1754965),
        (21, -2.0956717),
        (55, -0.82508224),
        (172, -0.029812645),
        (193, -0.99381155),
    ],
    &[
        (9, -0.5188245),
        (193, -0.9457299),
        (189, -0.28283915),
        (217, -1.089729),
        (21, -1.7542188),
        (55, -1.1566712),
        (172, -0.02740049),
        (241, -1.1313995),
    ],
    &[
        (9, -0.5884266),
        (193, -0.8403543),
        (189, -0.24684215),
        (217, -1.145189),
        (125, -1.9596578),
        (14, -0.29733655),
        (240, -1.6873292),
        (68, -0.5975468),
    ],
    &[
        (9, -0.45637178),
        (193, -0.66758424),
        (189, -0.25932452),
        (217, -0.9372047),
        (21, -1.8989222),
        (55, -0.8179172),
        (172, -0.033267055),
        (193, -0.8706901),
    ],
    &[
        (9, -0.64795035),
        (193, -0.70973366),
        (189, -0.32321578),
        (217, -1.0392342),
        (80, -1.9292932),
        (2, -0.9089281),
    ],
];

// Hybrid decode steps (attention and recurrent layers) replay captured graphs and give the eager output.
#[tokio::test]
async fn hybrid_paged_gpu_decode_replays_cuda_graphs() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    decode_graphs::assert_rounds_replay(paged_gpu_rounds, &GRAPH_TRACES).await
}

#[tokio::test]
async fn hybrid_paged_gpu_decode_without_cuda_graphs_matches() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    decode_graphs::assert_rounds_without_graphs(paged_gpu_rounds, &GRAPH_TRACES).await
}

// With a drafter attached, decode graphs capture the fixed-width verify steps and greedy output is unchanged.
#[tokio::test]
async fn hybrid_builtin_mtp_replays_verify_graphs() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let snapshotter = decode_graphs::recorder();
    let (_checkpoint, model) = paged_gpu_model(true).await?;
    let rounds = decode_graphs::rounds(&model, GRAPH_MAX_LEN).await?;
    let counters = decode_graphs::Counters::take(&snapshotter);
    decode_graphs::assert_replayed(&counters);
    decode_graphs::assert_only_prompts_skip(&counters);
    let drafts: usize = model
        .speculative_stats()?
        .data
        .iter()
        .map(|m| m.drafts)
        .sum();
    assert!(drafts > 0, "MTP never drafted");
    decode_graphs::assert_traces(&rounds, &GRAPH_TRACES);
    Ok(())
}

// A v1 DFlash drafter sized to the graph fixture: it reuses the target's embeddings and head, and taps two layers.
fn tiny_dflash() -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    let config = serde_json::json!({
        "architectures": ["DFlashDraftModel"],
        "hidden_size": HIDDEN,
        "intermediate_size": DFLASH_INTERMEDIATE,
        "num_hidden_layers": 1,
        "num_attention_heads": DFLASH_HEADS,
        "num_key_value_heads": 1,
        "head_dim": DFLASH_HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB,
        "rope_theta": 10000,
        "block_size": DFLASH_BLOCK,
        "mask_token_id": DFLASH_MASK_TOKEN,
        "layer_types": ["sliding_attention"],
        "sliding_window": DFLASH_WINDOW,
        "dflash_config": { "target_layer_ids": DFLASH_TAPS }
    });
    std::fs::write(dir.path().join("config.json"), config.to_string())?;
    let device = inference_tensor::Device::Cpu;
    let ramp = |rows: usize, cols: usize| {
        let data = (0..rows * cols)
            .map(|i| ((i as f32) * DFLASH_WEIGHT_STEP).sin() * DFLASH_WEIGHT_SCALE)
            .collect::<Vec<_>>();
        inference_tensor::Tensor::from_vec(data, (rows, cols), &device)
    };
    let ones = |n: usize| inference_tensor::Tensor::ones(n, inference_tensor::DType::F32, &device);
    let (q, kv) = (DFLASH_HEADS * DFLASH_HEAD_DIM, DFLASH_HEAD_DIM);
    let mut tensors = std::collections::HashMap::new();
    let layer = "layers.0";
    for (name, rows, cols) in [
        ("self_attn.q_proj", q, HIDDEN),
        ("self_attn.k_proj", kv, HIDDEN),
        ("self_attn.v_proj", kv, HIDDEN),
        ("self_attn.o_proj", HIDDEN, q),
        ("mlp.gate_proj", DFLASH_INTERMEDIATE, HIDDEN),
        ("mlp.up_proj", DFLASH_INTERMEDIATE, HIDDEN),
        ("mlp.down_proj", HIDDEN, DFLASH_INTERMEDIATE),
    ] {
        tensors.insert(format!("{layer}.{name}.weight"), ramp(rows, cols)?);
    }
    for (name, n) in [
        ("self_attn.q_norm", DFLASH_HEAD_DIM),
        ("self_attn.k_norm", DFLASH_HEAD_DIM),
        ("input_layernorm", HIDDEN),
        ("post_attention_layernorm", HIDDEN),
    ] {
        tensors.insert(format!("{layer}.{name}.weight"), ones(n)?);
    }
    tensors.insert(
        "fc.weight".to_string(),
        ramp(HIDDEN, DFLASH_TAPS.len() * HIDDEN)?,
    );
    tensors.insert("hidden_norm.weight".to_string(), ones(HIDDEN)?);
    tensors.insert("norm.weight".to_string(), ones(HIDDEN)?);
    inference_tensor::safetensors::save(&tensors, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

// An external DFlash drafter on a text model drafts, both its and the target's graphs replay, and greedy output holds.
#[tokio::test]
async fn hybrid_dflash_drafts_and_keeps_greedy_output() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let snapshotter = decode_graphs::recorder();
    let drafter = tiny_dflash()?;
    let checkpoint = tiny_qwen3_5_text(GRAPH_HEAD_DIM, false)?;
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .with_mtp_model(drafter.path().to_string_lossy(), Some(DFLASH_N_PREDICT))
        .build()
        .await?;
    let rounds = decode_graphs::rounds(&model, GRAPH_MAX_LEN).await?;
    let counters = decode_graphs::Counters::take(&snapshotter);
    decode_graphs::assert_replayed(&counters);
    decode_graphs::assert_only_prompts_skip(&counters);
    decode_graphs::assert_component_replayed(&counters, DFLASH_COMPONENT);
    let drafts: usize = model
        .speculative_stats()?
        .data
        .iter()
        .map(|m| m.drafts)
        .sum();
    assert!(drafts > 0, "DFlash never drafted");
    decode_graphs::assert_traces(&rounds, &GRAPH_TRACES);
    Ok(())
}

// On Metal the drafter's RoPE runs through the Metal rotary kernels, with per-batch caches.
#[cfg(feature = "metal")]
#[tokio::test]
async fn metal_dflash_drafts_and_keeps_greedy_output() -> anyhow::Result<()> {
    let drafter = tiny_dflash()?;
    let (_checkpoint, plain) = paged_gpu_model(false).await?;
    let expected = decode_graphs::rounds(&plain, GRAPH_MAX_LEN).await?;
    drop(plain);
    let checkpoint = tiny_qwen3_5_text(GRAPH_HEAD_DIM, false)?;
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .with_mtp_model(drafter.path().to_string_lossy(), Some(DFLASH_N_PREDICT))
        .build()
        .await?;
    let rounds = decode_graphs::rounds(&model, GRAPH_MAX_LEN).await?;
    let drafts: usize = model
        .speculative_stats()?
        .data
        .iter()
        .map(|m| m.drafts)
        .sum();
    assert!(drafts > 0, "DFlash never drafted");
    // the last round decodes every prompt
    let expected: Vec<&[(u32, f32)]> = expected.last().unwrap().iter().map(Vec::as_slice).collect();
    decode_graphs::assert_traces(&rounds, &expected);
    Ok(())
}

const PREFIX_CACHE_SEQS: usize = 4;
const PREFIX_MAX_LEN: usize = 6;
const CACHED_LOGPROB_TOLERANCE: f32 = 1e-3;
// several paged blocks long, so the snapshot boundary falls inside the prompt
const PREFIX_PROMPT: &str = "a hybrid model keeps a recurrent state beside its attention cache, so a prefix hit has \
    to restore that state at the cached boundary before the rest of the prompt runs; this sentence is long enough \
    to fill several blocks of the paged cache and leave a tail after the last full one.";

// The repeat of a prompt restores the paged recurrent snapshot with the cached blocks and decodes the same.
// A non-paged hybrid prefix only matches where its finished sequence ended, which a repeated prompt never reaches.
#[tokio::test]
async fn a_repeated_prompt_restores_the_recurrent_prefix() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = tiny_qwen3_5_text(GRAPH_HEAD_DIM, false)?;
    // the CUDA recurrence kernels take f16/bf16
    let model = TextModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_prefix_cache_n(Some(PREFIX_CACHE_SEQS))
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;
    let request = || {
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PREFIX_PROMPT))
    };
    let (first, first_cached) = traces::greedy(&model, request(), PREFIX_MAX_LEN).await?;
    let (second, second_cached) = traces::greedy(&model, request(), PREFIX_MAX_LEN).await?;
    anyhow::ensure!(
        first_cached == 0 && second_cached > 0,
        "cached prompt tokens: {first_cached} then {second_cached}"
    );
    anyhow::ensure!(
        traces::close(&first, &second, CACHED_LOGPROB_TOLERANCE),
        "a prefix hit changed decoding: {first:?} vs {second:?}"
    );
    Ok(())
}
