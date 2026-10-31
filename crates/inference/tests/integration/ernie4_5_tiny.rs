//! A tiny random-weight ERNIE 4.5 MoE checkpoint with its MTP layer: drafting with the shared decoder's MTP head
//! leaves greedy decoding as it is without one.

use std::collections::HashMap;

use inference::{ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "../support/decode_graphs.rs"]
mod decode_graphs;
#[path = "../support/recording.rs"]
mod recording;
#[path = "../support/traces.rs"]
mod traces;

const VOCAB: usize = 266;
const HIDDEN: usize = 64;
const NUM_LAYERS: usize = 3;
const EXPERTS: usize = 4;
const MOE_INTERMEDIATE: usize = 32;
const MTP_N_PREDICT: usize = 2;
const MTP_COMPONENT: &str = "mtp";
const MAX_LEN: usize = 24;
// the verify step's rows run batched where plain decode runs them one by one
const LOGPROB_TOLERANCE: f32 = 0.05;
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
const PROMPT: &str = "an MTP head drafts the next tokens from the target's last hidden state and the token it sampled";

fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Ernie4_5_MoeForCausalLM"],
        "model_type": "ernie4_5_moe",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 96,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": true,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "moe_num_experts": EXPERTS,
        "moe_k": 2,
        "moe_num_shared_experts": 2,
        "moe_layer_start_index": 1,
        "num_nextn_predict_layers": 1,
    })
}

// the checkpoint stores the experts one by one, which the loader reads from their shapes
fn expert_shapes() -> HashMap<String, Vec<usize>> {
    (1..NUM_LAYERS)
        .flat_map(|layer| (0..EXPERTS).map(move |expert| (layer, expert)))
        .flat_map(|(layer, expert)| {
            let p = format!("model.layers.{layer}.mlp.experts.{expert}");
            [
                (
                    format!("{p}.gate_proj.weight"),
                    vec![MOE_INTERMEDIATE, HIDDEN],
                ),
                (
                    format!("{p}.up_proj.weight"),
                    vec![MOE_INTERMEDIATE, HIDDEN],
                ),
                (
                    format!("{p}.down_proj.weight"),
                    vec![HIDDEN, MOE_INTERMEDIATE],
                ),
            ]
        })
        .collect()
}

fn tiny_ernie4_5_moe() -> anyhow::Result<tempfile::TempDir> {
    let scratch = tempfile::tempdir()?;
    let config_path = scratch.path().join("config.json");
    std::fs::write(&config_path, config().to_string())?;
    let files = [
        config_path,
        format!("{FIXTURES}/paddleocr_vl/tiny/tokenizer.json").into(),
        format!("{FIXTURES}/llama_tiny/tokenizer_config.json").into(),
        format!("{FIXTURES}/llama_tiny/chat_template.jinja").into(),
    ];
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    // built with the head, so the checkpoint carries its weights
    let mut with_mtp = config();
    with_mtp["_inference_mtp"] = true.into();
    let cfg: inference_models_other::ernie4_5::MoeConfig = serde_json::from_value(with_mtp)?;
    recording::record_checkpoint_seeded_with_shapes(
        &files,
        NUM_LAYERS + 1,
        &[],
        expert_shapes(),
        |vb, metadata| {
            inference_models_other::ernie4_5::Model::new(
                &cfg.decoder_spec()?,
                vb,
                false,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )
}

// Greedy decoding with the MTP head drafting matches decoding without it, the head drafted, and both the verify steps
// (the head's capture handed through their state) and the head's own forwards replayed CUDA graphs.
#[tokio::test]
async fn an_mtp_head_drafts_without_changing_greedy_output() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let snapshotter = decode_graphs::recorder();
    let checkpoint = tiny_ernie4_5_moe()?;
    let builder = || -> anyhow::Result<TextModelBuilder> {
        Ok(TextModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::BF16)
            .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?))
    };
    let request =
        || RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT));
    let drafting = builder()?
        .with_builtin_mtp(Some(MTP_N_PREDICT))
        .build()
        .await?;
    let (actual, _) = traces::greedy(&drafting, request(), MAX_LEN).await?;
    let counters = decode_graphs::Counters::take(&snapshotter);
    decode_graphs::assert_replayed(&counters);
    decode_graphs::assert_only_prompts_skip(&counters);
    decode_graphs::assert_component_replayed(&counters, MTP_COMPONENT);
    let drafts: usize = drafting
        .speculative_stats()?
        .data
        .iter()
        .map(|m| m.drafts)
        .sum();
    drop(drafting);
    let plain = builder()?.build().await?;
    let (expected, _) = traces::greedy(&plain, request(), MAX_LEN).await?;
    anyhow::ensure!(drafts > 0, "the MTP head never drafted");
    anyhow::ensure!(
        traces::close(&expected, &actual, LOGPROB_TOLERANCE),
        "drafting changed greedy decoding: {expected:?} vs {actual:?}"
    );
    Ok(())
}
