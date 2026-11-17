//! Tiny random-weight EXAONE checkpoints load through their Hugging Face classes and generate: EXAONE 4.0 in the 32B's
//! layout (sliding layers beside NoPE global ones) and the 1.2B's (no window, RoPE on every layer, tied embeddings),
//! EXAONE 3.5 under its checkpoint's names, and K-EXAONE's experts and MTP layer.

use std::{collections::HashMap, path::Path};

use inference::{ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "../support/recording.rs"]
mod recording;
#[path = "../support/traces.rs"]
mod traces;

const LLAMA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llama_tiny");
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
const LAYERS: usize = 4;
// shorter than the chat prompt, so the sliding layers drop keys
const SLIDING_WINDOW: usize = 4;
const MAX_LEN: usize = 4;

fn config(sliding: bool, tied: bool) -> serde_json::Value {
    let (window, layer_types) = if sliding {
        (
            serde_json::json!(SLIDING_WINDOW),
            serde_json::json!([
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ]),
        )
    } else {
        (
            serde_json::Value::Null,
            serde_json::json!(vec!["full_attention"; LAYERS]),
        )
    };
    serde_json::json!({
        "architectures": ["Exaone4ForCausalLM"],
        "model_type": "exaone4",
        "hidden_size": 32,
        "intermediate_size": 48,
        "vocab_size": 266,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "rope_scaling": {
            "factor": 16.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 64, "rope_type": "llama3"
        },
        "hidden_act": "silu",
        "sliding_window": window,
        "sliding_window_pattern": if sliding { serde_json::json!("LLLG") } else { serde_json::Value::Null },
        "layer_types": layer_types,
        "tie_word_embeddings": tied,
    })
}

fn exaone3_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["ExaoneForCausalLM"],
        "model_type": "exaone",
        "hidden_size": 32,
        "intermediate_size": 48,
        "vocab_size": 266,
        "num_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 512,
        "layer_norm_epsilon": 1e-5,
        "rope_theta": 1000000.0,
        "rope_scaling": {
            "factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 64, "rope_type": "llama3"
        },
        "activation_function": "silu",
        "tie_word_embeddings": true,
    })
}

async fn loads_and_generates(
    config: serde_json::Value,
    spec: inference_nn::decoder::DecoderSpec,
    expected: &[&str],
    unexpected: &str,
) -> anyhow::Result<()> {
    let staging = tempfile::tempdir()?;
    let config_path = staging.path().join("config.json");
    std::fs::write(&config_path, config.to_string())?;
    let mut files = recording::fixture_files(LLAMA)?;
    files.retain(|path| path.file_name().is_some_and(|name| name != "config.json"));
    files.push(config_path);
    files.push(TOKENIZER.into());
    let files = files
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<&Path>>();
    let dir = recording::record_checkpoint_seeded_with_shapes(
        &files,
        LAYERS,
        // llama.cpp's precomputed llama3 factors; a safetensors checkpoint has none
        &["model.rope_freqs.weight", "transformer.rope_freqs.weight"],
        HashMap::new(),
        |vb, metadata| {
            inference_nn::decoder::CausalLm::new(
                &spec,
                vb,
                true,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )?;
    let tensors = inference_tensor::safetensors::load(
        dir.path().join("model.safetensors"),
        &inference_tensor::Device::Cpu,
    )?;
    for name in expected {
        anyhow::ensure!(tensors.contains_key(*name), "no {name}");
    }
    anyhow::ensure!(!tensors.contains_key(unexpected), "{unexpected} loaded");

    let model = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let request = RequestBuilder::from(
        TextMessages::new().add_message(TextMessageRole::User, "hello there, how are you today"),
    )
    .set_sampler_max_len(MAX_LEN)
    .set_sampler_topk(1);
    let response = model.send_chat_request(request).await?;
    anyhow::ensure!(
        response.usage.completion_tokens > 0,
        "the model generated nothing"
    );
    Ok(())
}

#[tokio::test]
async fn exaone4_sliding_and_windowless_layouts_load_and_generate() -> anyhow::Result<()> {
    let sliding =
        serde_json::from_value::<inference_models_other::exaone4::Config>(config(true, false))?
            .decoder_spec()?;
    assert_eq!(sliding.no_rope_layers, vec![3]);
    assert_eq!(sliding.layer_windows[0], Some(SLIDING_WINDOW));
    assert_eq!(sliding.layer_windows[3], None);
    let windowless =
        serde_json::from_value::<inference_models_other::exaone4::Config>(config(false, true))?
            .decoder_spec()?;
    assert!(windowless.no_rope_layers.is_empty());

    for (sliding, tied) in [(true, false), (false, true)] {
        let config = config(sliding, tied);
        let spec =
            serde_json::from_value::<inference_models_other::exaone4::Config>(config.clone())?
                .decoder_spec()?;
        loads_and_generates(
            config,
            spec,
            &[
                "model.layers.0.self_attn.q_norm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.post_feedforward_layernorm.weight",
            ],
            "model.layers.0.input_layernorm.weight",
        )
        .await
        .map_err(|error| anyhow::anyhow!("sliding {sliding}: {error}"))?;
    }
    Ok(())
}

#[tokio::test]
async fn exaone3_loads_under_its_checkpoint_names_and_generates() -> anyhow::Result<()> {
    let config = exaone3_config();
    let spec = serde_json::from_value::<inference_models_other::exaone::Config>(config.clone())?
        .decoder_spec()?;
    loads_and_generates(
        config,
        spec,
        &[
            "transformer.wte.weight",
            "transformer.ln_f.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.ln_2.weight",
            "transformer.h.0.attn.attention.q_proj.weight",
            "transformer.h.0.attn.attention.out_proj.weight",
            "transformer.h.0.mlp.c_fc_0.weight",
            "transformer.h.0.mlp.c_fc_1.weight",
            "transformer.h.0.mlp.c_proj.weight",
        ],
        "lm_head.weight",
    )
    .await
}

const K_LAYERS: usize = 3;
const K_HIDDEN: usize = 64;
const K_EXPERTS: usize = 4;
// the CUDA expert kernels take widths in multiples of 64
const K_EXPERT_WIDTH: usize = 64;
const K_MAX_LEN: usize = 16;
const K_MTP_N_PREDICT: usize = 2;
// verify rows run batched where plain decode runs them one by one
const K_LOGPROB_TOLERANCE: f32 = 0.05;

fn k_exaone_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["ExaoneMoeForCausalLM"],
        "model_type": "exaone_moe",
        "hidden_size": K_HIDDEN,
        "intermediate_size": 96,
        "vocab_size": 266,
        "num_hidden_layers": K_LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "hidden_act": "silu",
        "sliding_window": SLIDING_WINDOW,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
        "mlp_layer_types": ["dense", "sparse", "sparse"],
        "num_experts": K_EXPERTS,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": K_EXPERT_WIDTH,
        "num_shared_experts": 1,
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "num_nextn_predict_layers": 1,
        "mtp_layer_types": ["full_attention"],
        "tie_word_embeddings": false,
    })
}

// the per-expert layout the checkpoints store, which the experts loader reads shapes of before loading
fn k_exaone_expert_shapes() -> HashMap<String, Vec<usize>> {
    (1..K_LAYERS)
        .flat_map(|layer| {
            (0..K_EXPERTS).flat_map(move |expert| {
                let p = format!("model.layers.{layer}.mlp.experts.{expert}");
                [
                    (
                        format!("{p}.gate_proj.weight"),
                        vec![K_EXPERT_WIDTH, K_HIDDEN],
                    ),
                    (
                        format!("{p}.up_proj.weight"),
                        vec![K_EXPERT_WIDTH, K_HIDDEN],
                    ),
                    (
                        format!("{p}.down_proj.weight"),
                        vec![K_HIDDEN, K_EXPERT_WIDTH],
                    ),
                ]
            })
        })
        .collect()
}

// recorded with the MTP layer, so the checkpoint carries it at its root
fn tiny_k_exaone() -> anyhow::Result<tempfile::TempDir> {
    let staging = tempfile::tempdir()?;
    let config_path = staging.path().join("config.json");
    std::fs::write(&config_path, k_exaone_config().to_string())?;
    let mut files = recording::fixture_files(LLAMA)?;
    files.retain(|path| path.file_name().is_some_and(|name| name != "config.json"));
    files.push(config_path);
    files.push(TOKENIZER.into());
    let files = files
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<&Path>>();
    let mut with_mtp = k_exaone_config();
    with_mtp["_inference_mtp"] = true.into();
    let cfg: inference_models_other::exaone_moe::Config = serde_json::from_value(with_mtp)?;
    recording::record_checkpoint_seeded_with_shapes(
        &files,
        K_LAYERS + 1,
        &["model.rope_freqs.weight"],
        k_exaone_expert_shapes(),
        |vb, metadata| {
            inference_nn::decoder::CausalLm::new(
                &cfg.decoder_spec()?,
                vb,
                true,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )
}

#[tokio::test]
async fn k_exaone_routes_its_experts_and_generates() -> anyhow::Result<()> {
    let spec =
        serde_json::from_value::<inference_models_other::exaone_moe::Config>(k_exaone_config())?
            .decoder_spec()?;
    assert_eq!(
        spec.moe.as_ref().map(|moe| moe.dense_layers.clone()),
        Some(vec![0])
    );
    assert_eq!(spec.no_rope_layers, vec![2]);
    let dir = tiny_k_exaone()?;
    let tensors = inference_tensor::safetensors::load(
        dir.path().join("model.safetensors"),
        &inference_tensor::Device::Cpu,
    )?;
    for name in [
        "model.layers.1.mlp.e_score_correction_bias",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
    ] {
        anyhow::ensure!(tensors.contains_key(name), "no {name}");
    }
    let model = TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let request = RequestBuilder::from(
        TextMessages::new().add_message(TextMessageRole::User, "hello there, how are you today"),
    )
    .set_sampler_max_len(MAX_LEN)
    .set_sampler_topk(1);
    let response = model.send_chat_request(request).await?;
    anyhow::ensure!(
        response.usage.completion_tokens > 0,
        "the model generated nothing"
    );
    Ok(())
}

// Greedy decoding with K-EXAONE's root-level, pre-norm, NoPE MTP layer drafting matches decoding without it.
#[tokio::test]
async fn k_exaone_mtp_drafts_without_changing_greedy_output() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = tiny_k_exaone()?;
    let builder = || -> anyhow::Result<TextModelBuilder> {
        Ok(TextModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::BF16)
            .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?))
    };
    let request = || {
        RequestBuilder::from(
            TextMessages::new()
                .add_message(TextMessageRole::User, "hello there, how are you today"),
        )
    };
    let drafting = builder()?
        .with_builtin_mtp(Some(K_MTP_N_PREDICT))
        .build()
        .await?;
    let (actual, _) = traces::greedy(&drafting, request(), K_MAX_LEN).await?;
    let drafts: usize = drafting
        .speculative_stats()?
        .data
        .iter()
        .map(|m| m.drafts)
        .sum();
    drop(drafting);
    let plain = builder()?.build().await?;
    let (expected, _) = traces::greedy(&plain, request(), K_MAX_LEN).await?;
    anyhow::ensure!(drafts > 0, "the MTP head never drafted");
    anyhow::ensure!(
        traces::close(&expected, &actual, K_LOGPROB_TOLERANCE),
        "drafting changed greedy decoding: {expected:?} vs {actual:?}"
    );
    Ok(())
}
