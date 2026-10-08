//! A tiny random-weight Qwen3.5 text checkpoint: the one text loader whose runtime config differs from the checkpoint's.

use inference::{IsqType, Model, ModelDType, TextModelBuilder};
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
        "vocab_size": 266,
        "hidden_size": 64,
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
    // every step a graph skipped is a prompt: verify steps replay too
    let skipped = counters.total(decode_graphs::DISPATCH, &[("mode", "skipped")]);
    let prompts = counters.total(decode_graphs::DISPATCH, &[("reason", "prefill")]);
    assert_eq!(
        skipped, prompts,
        "a decode or verify step skipped graphs: {counters:?}"
    );
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
