//! A tiny random-weight Llama with a tiny random EAGLE-3 drafter (SpecForge layout, a reduced vocabulary mapped through
//! `d2t`): drafting from three of the target's layers leaves greedy decoding as it is without the drafter.

use std::collections::HashMap;

use inference::{ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder};
use inference_tensor::{DType, Device, Tensor};

#[path = "../support/llama_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;

// the committed tiny Llama's widths, with layers enough for three distinct taps
const LAYERS: usize = 4;
const HIDDEN: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
const INTERMEDIATE: usize = 64;
const VOCAB: usize = 266;
const DRAFT_VOCAB: usize = 64;
// draft id i is target id 3i
const DRAFT_ID_STRIDE: i64 = 3;
const TAP_LAYER_IDS: [usize; 3] = [1, 2, 4];
const N_PREDICT: usize = 2;
const WEIGHT_STD: f32 = 0.1;
const MAX_LEN: usize = 16;
// the verify step's rows run batched where plain decode runs them one by one
const LOGPROB_TOLERANCE: f32 = 0.05;
const PROMPT: &str =
    "an EAGLE-3 drafter reads three of the target's layers and drafts in its own small vocabulary";

fn tiny_drafter() -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLMEagle3"],
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_hidden_layers": 1,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
        "draft_vocab_size": DRAFT_VOCAB,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": TAP_LAYER_IDS},
    });
    std::fs::write(dir.path().join("config.json"), config.to_string())?;
    let random = |shape: &[usize]| -> anyhow::Result<Tensor> {
        Ok(Tensor::randn(0f32, WEIGHT_STD, shape, &Device::Cpu)?.to_dtype(DType::BF16)?)
    };
    let ones = |width: usize| Tensor::ones(width, DType::BF16, &Device::Cpu);
    let (q, kv) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let mut tensors = HashMap::from([
        ("fc.weight".to_string(), random(&[HIDDEN, 3 * HIDDEN])?),
        ("norm.weight".to_string(), ones(HIDDEN)?),
        (
            "lm_head.weight".to_string(),
            random(&[DRAFT_VOCAB, HIDDEN])?,
        ),
        (
            "d2t".to_string(),
            Tensor::from_vec(
                (0..DRAFT_VOCAB as i64)
                    .map(|i| i * (DRAFT_ID_STRIDE - 1))
                    .collect::<Vec<_>>(),
                DRAFT_VOCAB,
                &Device::Cpu,
            )?,
        ),
    ]);
    for (name, shape) in [
        ("self_attn.q_proj.weight", vec![q, 2 * HIDDEN]),
        ("self_attn.k_proj.weight", vec![kv, 2 * HIDDEN]),
        ("self_attn.v_proj.weight", vec![kv, 2 * HIDDEN]),
        ("self_attn.o_proj.weight", vec![HIDDEN, q]),
        ("mlp.gate_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("mlp.up_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("mlp.down_proj.weight", vec![HIDDEN, INTERMEDIATE]),
    ] {
        tensors.insert(format!("midlayer.{name}"), random(&shape)?);
    }
    for norm in ["input_layernorm", "hidden_norm", "post_attention_layernorm"] {
        tensors.insert(format!("midlayer.{norm}.weight"), ones(HIDDEN)?);
    }
    inference_tensor::safetensors::save(&tensors, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

// Greedy decoding with the drafter matches decoding without it, and the drafter drafted.
#[tokio::test]
async fn an_eagle3_drafter_drafts_without_changing_greedy_output() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = support::tiny_llama_checkpoint_with(serde_json::json!({
        "num_hidden_layers": LAYERS
    }))?;
    let drafter = tiny_drafter()?;
    let builder = || -> anyhow::Result<TextModelBuilder> {
        Ok(TextModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::BF16)
            .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?))
    };
    let request =
        || RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT));
    let drafting = builder()?
        .with_mtp_model(drafter.path().to_string_lossy(), Some(N_PREDICT))
        .build()
        .await?;
    let (actual, _) = traces::greedy(&drafting, request(), MAX_LEN).await?;
    let drafts: usize = drafting
        .speculative_stats()?
        .data
        .iter()
        .map(|m| m.drafts)
        .sum();
    drop(drafting);
    let plain = builder()?.build().await?;
    let (expected, _) = traces::greedy(&plain, request(), MAX_LEN).await?;
    anyhow::ensure!(drafts > 0, "the EAGLE-3 drafter never drafted");
    anyhow::ensure!(
        traces::close(&expected, &actual, LOGPROB_TOLERANCE),
        "drafting changed greedy decoding: {expected:?} vs {actual:?}"
    );
    Ok(())
}
