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
fn tiny_dflash(window: usize) -> anyhow::Result<tempfile::TempDir> {
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
        "sliding_window": window,
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
    let drafter = tiny_dflash(DFLASH_WINDOW)?;
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
    let drafter = tiny_dflash(DFLASH_WINDOW)?;
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

// wider than a paged block, as real drafters' windows are (2048), so a turn's reply never spans the replay window
const AGENT_DFLASH_WINDOW: usize = 128;
// token-exact turns, so the reply's blocks match on the next turn: the prompt ends mid-block, its snapshot sits at
// the last full block, and the reply carries the cache one block past it
const AGENT_PROMPT_LEN: usize = 400;
const AGENT_REPLY_LEN: usize = 40;
const AGENT_FOLLOW_UP_LEN: usize = 20;
const AGENT_TOKEN_STRIDE: usize = 7;

fn agent_tokens(len: usize, offset: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((offset + i * AGENT_TOKEN_STRIDE) % (VOCAB - 2)) as u32)
        .collect()
}

async fn agent_model(
    checkpoint: &std::path::Path,
    drafter: &std::path::Path,
) -> anyhow::Result<Model> {
    Ok(TextModelBuilder::new(checkpoint.to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_prefix_cache_n(Some(PREFIX_CACHE_SEQS))
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .with_mtp_model(drafter.to_string_lossy(), Some(DFLASH_N_PREDICT))
        .build()
        .await?)
}

// A greedy completion of `prompt` token ids: each generated token with its logprob, and the cached prompt tokens
async fn agent_turn(
    model: &Model,
    prompt: &[u32],
    max_tokens: usize,
) -> anyhow::Result<(traces::Trace, usize)> {
    let request = serde_json::json!({
        "model": "default", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
        "logprobs": 1, "ignore_eos": true,
    });
    let response: serde_json::Value = serde_json::from_str(
        &model
            .completion_json(request.to_string().as_bytes())
            .await?,
    )?;
    let trace = response["choices"][0]["logprobs"]["content"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("no logprobs in {response}"))?
        .iter()
        .map(|step| {
            let top = &step["top_logprobs"][0];
            (
                top["token"].as_u64().unwrap_or_default() as u32,
                top["logprob"].as_f64().unwrap_or_default() as f32,
            )
        })
        .collect();
    let cached = response["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or_default() as usize;
    Ok((trace, cached))
}

// An agent's next turn (prompt, reply, follow-up) caches the reply's blocks past the previous prompt's snapshot; with a
// DFlash drafter that snapshot still restores, drafter checkpoint included, and decodes as a cold run does.
#[tokio::test]
async fn a_dflash_agent_turn_resumes_at_the_previous_prompt_snapshot() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let drafter = tiny_dflash(AGENT_DFLASH_WINDOW)?;
    let checkpoint = tiny_qwen3_5_text(GRAPH_HEAD_DIM, false)?;
    let model = agent_model(checkpoint.path(), drafter.path()).await?;
    let first = agent_tokens(AGENT_PROMPT_LEN, 0);
    let (reply, _) = agent_turn(&model, &first, AGENT_REPLY_LEN).await?;
    let mut next = first;
    next.extend(reply.iter().map(|&(token, _)| token));
    next.extend(agent_tokens(AGENT_FOLLOW_UP_LEN, 1));
    let (warm, cached) = agent_turn(&model, &next, PREFIX_MAX_LEN).await?;
    drop(model);
    let cold_model = agent_model(checkpoint.path(), drafter.path()).await?;
    let (cold, _) = agent_turn(&cold_model, &next, PREFIX_MAX_LEN).await?;
    anyhow::ensure!(cached > 0, "the next turn found no cached prefix");
    anyhow::ensure!(
        traces::close(&warm, &cold, CACHED_LOGPROB_TOLERANCE),
        "a prefix hit changed decoding: {warm:?} vs {cold:?}"
    );
    Ok(())
}

// the writer thread stores a prefix well within this
const STORE_POLL: std::time::Duration = std::time::Duration::from_millis(50);
const STORE_POLLS: usize = 200;

async fn stored_model(
    checkpoint: &std::path::Path,
    drafter: Option<&std::path::Path>,
    store: Option<&std::path::Path>,
) -> anyhow::Result<Model> {
    let mut builder = TextModelBuilder::new(checkpoint.to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_prefix_cache_n(Some(PREFIX_CACHE_SEQS))
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?);
    if let Some(drafter) = drafter {
        builder = builder.with_mtp_model(drafter.to_string_lossy(), Some(DFLASH_N_PREDICT));
    }
    if let Some(store) = store {
        builder = builder.with_prefix_cache_dir(store.to_string_lossy(), None);
    }
    Ok(builder.build().await?)
}

fn stored_entries(store: &std::path::Path) -> usize {
    walkdir_count(store, "key.bin")
}

fn walkdir_count(dir: &std::path::Path, name: &str) -> usize {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                walkdir_count(&path, name)
            } else {
                usize::from(path.file_name().is_some_and(|n| n == name))
            }
        })
        .sum()
}

// A prefix written to the disk store restores in a fresh engine (a restart), and the next turn decodes as a cold run
async fn check_restart_restores(drafter: Option<&std::path::Path>) -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_5_text(GRAPH_HEAD_DIM, false)?;
    let store = tempfile::tempdir()?;
    let first = agent_tokens(AGENT_PROMPT_LEN, 0);
    let mut next = first.clone();
    next.extend(agent_tokens(AGENT_FOLLOW_UP_LEN, 1));
    {
        let model = stored_model(checkpoint.path(), drafter, Some(store.path())).await?;
        agent_turn(&model, &first, PREFIX_MAX_LEN).await?;
        for _ in 0..STORE_POLLS {
            if stored_entries(store.path()) > 0 {
                break;
            }
            tokio::time::sleep(STORE_POLL).await;
        }
    }
    anyhow::ensure!(
        stored_entries(store.path()) > 0,
        "no prefix reached the disk store"
    );
    let restarted = stored_model(checkpoint.path(), drafter, Some(store.path())).await?;
    let (warm, cached) = agent_turn(&restarted, &next, PREFIX_MAX_LEN).await?;
    drop(restarted);
    let cold_model = stored_model(checkpoint.path(), drafter, None).await?;
    let (cold, _) = agent_turn(&cold_model, &next, PREFIX_MAX_LEN).await?;
    anyhow::ensure!(cached > 0, "the restarted engine found no cached prefix");
    anyhow::ensure!(
        traces::close(&warm, &cold, CACHED_LOGPROB_TOLERANCE),
        "a restored prefix changed decoding: {warm:?} vs {cold:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_stored_prefix_restores_after_a_restart() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    check_restart_restores(None).await
}

#[tokio::test]
async fn a_stored_prefix_restores_with_its_dflash_checkpoint_after_a_restart() -> anyhow::Result<()>
{
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let drafter = tiny_dflash(AGENT_DFLASH_WINDOW)?;
    check_restart_restores(Some(drafter.path())).await
}

// a ChatML-style template over the fixture tokenizer's special `<|begin_of_sentence|>`, so each turn opens with one token
const TURN_TEMPLATE: &str = "{%- for message in messages -%}<|begin_of_sentence|>{{ message['role'] }}: \
{{ message['content'] }}\n{%- endfor -%}{%- if add_generation_prompt -%}<|begin_of_sentence|>assistant: {%- endif -%}";
const SHARED_HISTORY_CHARS: usize = 300;
const FINAL_MESSAGE_CHARS: usize = 120;

async fn chat_cached_tokens(model: &Model, history: &str, last: &str) -> anyhow::Result<usize> {
    let request = serde_json::json!({
        "model": "default", "max_tokens": 2, "temperature": 0.0,
        "messages": [
            {"role": "user", "content": history},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": last},
        ],
    });
    let response: serde_json::Value = serde_json::from_str(
        &model
            .chat_json(request.to_string().as_bytes(), Default::default())
            .await?,
    )?;
    Ok(response["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or_default() as usize)
}

// A request that replaces only the final message resumes where that message starts, not at its first token: the
// previous prompt's recurrent state is also kept there, short of the last full block the final message runs into
#[tokio::test]
async fn a_new_final_message_resumes_at_the_previous_one() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = tiny_qwen3_5_text(GRAPH_HEAD_DIM, false)?;
    std::fs::write(checkpoint.path().join("chat_template.jinja"), TURN_TEMPLATE)?;
    let model = stored_model(checkpoint.path(), None, None).await?;
    let history = "abc ".repeat(SHARED_HISTORY_CHARS / 4);
    chat_cached_tokens(&model, &history, &"first ".repeat(FINAL_MESSAGE_CHARS / 6)).await?;
    let cached =
        chat_cached_tokens(&model, &history, &"other ".repeat(FINAL_MESSAGE_CHARS / 6)).await?;
    anyhow::ensure!(
        cached >= SHARED_HISTORY_CHARS / 2,
        "the changed final message reused only {cached} tokens"
    );
    Ok(())
}

// distinct prompts sent before the first is extended: more than a disk-backed drafter keeps on the GPU
const EVICTING_PROMPTS: usize = 4;

// a host store's size in GiB for the tiny model: far more than its prefixes take
const TINY_RAM_STORE_GIB: u64 = 1;

// With a host store the drafter keeps few prefixes on the GPU; one it dropped restores from the store (a directory,
// or memory alone), drafter checkpoint included, and its next turn decodes as a cold run does
async fn check_dropped_prefix_restores(in_memory: bool) -> anyhow::Result<()> {
    let drafter = tiny_dflash(AGENT_DFLASH_WINDOW)?;
    let checkpoint = tiny_qwen3_5_text(GRAPH_HEAD_DIM, false)?;
    let store = tempfile::tempdir()?;
    let model = if in_memory {
        TextModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::BF16)
            .with_prefix_cache_n(Some(PREFIX_CACHE_SEQS))
            .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
            .with_mtp_model(drafter.path().to_string_lossy(), Some(DFLASH_N_PREDICT))
            .with_prefix_cache_ram(TINY_RAM_STORE_GIB)
            .build()
            .await?
    } else {
        stored_model(checkpoint.path(), Some(drafter.path()), Some(store.path())).await?
    };
    let prompts = (0..EVICTING_PROMPTS)
        .map(|i| agent_tokens(AGENT_PROMPT_LEN, 1 + i))
        .collect::<Vec<_>>();
    for prompt in &prompts {
        agent_turn(&model, prompt, PREFIX_MAX_LEN).await?;
    }
    if !in_memory {
        for _ in 0..STORE_POLLS {
            if stored_entries(store.path()) >= EVICTING_PROMPTS {
                break;
            }
            tokio::time::sleep(STORE_POLL).await;
        }
        anyhow::ensure!(
            stored_entries(store.path()) >= EVICTING_PROMPTS,
            "the prompts did not all reach the disk store"
        );
    }
    let mut next = prompts[0].clone();
    next.extend(agent_tokens(AGENT_FOLLOW_UP_LEN, 0));
    let (warm, cached) = agent_turn(&model, &next, PREFIX_MAX_LEN).await?;
    drop(model);
    let cold_model = stored_model(checkpoint.path(), Some(drafter.path()), None).await?;
    let (cold, _) = agent_turn(&cold_model, &next, PREFIX_MAX_LEN).await?;
    anyhow::ensure!(cached > 0, "the dropped prefix did not restore");
    anyhow::ensure!(
        traces::close(&warm, &cold, CACHED_LOGPROB_TOLERANCE),
        "a restored prefix changed decoding: {warm:?} vs {cold:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_prefix_the_drafter_dropped_restores_from_the_disk_store() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    check_dropped_prefix_restores(false).await
}

#[tokio::test]
async fn a_prefix_the_drafter_dropped_restores_from_the_memory_store() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    check_dropped_prefix_restores(true).await
}

// the prompt's snapshot sits at its last full block before the end, the reply's one block further
const PROMPT_SNAPSHOT_TOKENS: usize = 384;
const REPLY_SNAPSHOT_TOKENS: usize = 416;
// The reply's first block is reused as the original request decoded it, where a cold run prefills it: the same tokens,
// logprobs a few hundredths apart (a decoded block reused by plain prefix caching drifts the same way)
const DECODED_KV_LOGPROB_TOLERANCE: f32 = 0.1;
const WARMUP_POLL: std::time::Duration = std::time::Duration::from_millis(20);
const WARMUP_POLLS: usize = 250;

// After a reply, the engine prefills it on top of the prompt's snapshot, so the next turn resumes past the reply
async fn check_reply_prefill(drafter: Option<&std::path::Path>) -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_5_text(GRAPH_HEAD_DIM, false)?;
    let model = stored_model(checkpoint.path(), drafter, None).await?;
    let first = agent_tokens(AGENT_PROMPT_LEN, 0);
    let (reply, _) = agent_turn(&model, &first, AGENT_REPLY_LEN).await?;
    let hits = || -> anyhow::Result<usize> {
        Ok(model
            .cache_stats()?
            .data
            .iter()
            .map(|m| m.prefix_cache_hits)
            .sum())
    };
    for _ in 0..WARMUP_POLLS {
        if hits()? > 0 {
            break;
        }
        tokio::time::sleep(WARMUP_POLL).await;
    }
    // the warm-up hit the prompt's snapshot; give its one decode step time to finish
    tokio::time::sleep(WARMUP_POLL * 10).await;
    let mut next = first;
    next.extend(reply.iter().map(|&(token, _)| token));
    next.extend(agent_tokens(AGENT_FOLLOW_UP_LEN, 1));
    let (warm, cached) = agent_turn(&model, &next, PREFIX_MAX_LEN).await?;
    drop(model);
    let cold_model = stored_model(checkpoint.path(), drafter, None).await?;
    let (cold, _) = agent_turn(&cold_model, &next, PREFIX_MAX_LEN).await?;
    anyhow::ensure!(
        cached == REPLY_SNAPSHOT_TOKENS,
        "the next turn resumed at {cached}, not past the reply ({REPLY_SNAPSHOT_TOKENS}; the prompt's snapshot is at \
         {PROMPT_SNAPSHOT_TOKENS})"
    );
    anyhow::ensure!(
        traces::close(&warm, &cold, DECODED_KV_LOGPROB_TOLERANCE),
        "resuming past the reply changed decoding: {warm:?} vs {cold:?}"
    );
    Ok(())
}

#[tokio::test]
async fn a_finished_reply_is_prefilled_for_the_next_turn() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    check_reply_prefill(None).await
}

#[tokio::test]
async fn a_finished_reply_is_prefilled_with_a_dflash_drafter() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let drafter = tiny_dflash(AGENT_DFLASH_WINDOW)?;
    check_reply_prefill(Some(drafter.path())).await
}
