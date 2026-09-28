//! A tiny random-weight Qwen3.5 text checkpoint: the one text loader whose runtime config differs from the checkpoint's.

use inference::{IsqType, ModelDType, TextModelBuilder};
use inference_models_qwen::qwen3_5::{Qwen3_5TextModel, TextConfig};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "support/recording.rs"]
mod recording;

const DECLARED_CONTEXT: u64 = 1024;
const RUNTIME_LIMIT: usize = 256;
const NUM_LAYERS: usize = 4;
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn tiny_qwen3_5_text() -> anyhow::Result<tempfile::TempDir> {
    let config = serde_json::json!({
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "head_dim": 32,
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
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1]
        },
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "tie_word_embeddings": false
    });
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
    recording::record_checkpoint(&files, NUM_LAYERS, &[], |vb, metadata| {
        Qwen3_5TextModel::new(
            &cfg,
            vb,
            false,
            false,
            metadata,
            AttentionImplementation::Eager,
        )
        .map(|_| ())
    })
}

#[tokio::test]
async fn a_uqff_keeps_the_checkpoint_context_when_written_under_max_model_len() -> anyhow::Result<()>
{
    let checkpoint = tiny_qwen3_5_text()?;
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
