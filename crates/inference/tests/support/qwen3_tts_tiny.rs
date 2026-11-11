//! Builds a tiny random-weight Qwen3-TTS checkpoint (talker, code predictor, codec, BPE tokenizer files) at test time.
#![allow(dead_code)]

use std::path::Path;

use inference_models_speech::qwen3_tts::{CodecConfig, Qwen3Tts, Qwen3TtsConfig};
use serde_json::json;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

#[path = "recording.rs"]
mod recording;

// invented speaker names; the second speaks Chinese in an (invented) dialect
pub const SPEAKERS: [&str; 2] = ["tiny_amber", "tiny_basil"];
pub const LANGUAGES: [&str; 2] = ["english", "chinese"];
/// Audio samples per 80 ms frame in the tiny codec (2 x 2 x 2 upsampling).
pub const SAMPLES_PER_FRAME: usize = 8;
pub const CODE_GROUPS: usize = 4;
const CODES: usize = 64;
// the reference's control region: the codec vocabulary's last 1024 ids
const CONTROL_IDS: usize = 1024;
// the byte-level alphabet's 256 entries come first, then these added tokens
const ADDED: [(&str, bool); 6] = [
    ("<|im_start|>", true),
    ("<|im_end|>", true),
    ("assistant", false),
    ("<tts_pad>", true),
    ("<tts_bos>", true),
    ("<tts_eos>", true),
];
const BYTE_TOKENS: usize = 256;

fn config(model_type: &str, size: &str) -> serde_json::Value {
    let control = |i: usize| CODES + i;
    let predictor = json!({
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "max_position_embeddings": 64, "vocab_size": CODES,
    });
    let languages = json!({
        LANGUAGES[0]: control(7), LANGUAGES[1]: control(8), "tiny_dialect": control(9),
    });
    let speakers = json!({ SPEAKERS[0]: control(10), SPEAKERS[1]: control(11) });
    let dialects = json!({ SPEAKERS[0]: false, SPEAKERS[1]: "tiny_dialect" });
    let mut talker = json!({
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
        "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "max_position_embeddings": 4096, "vocab_size": CODES + CONTROL_IDS,
        "text_hidden_size": 24, "text_vocab_size": BYTE_TOKENS + ADDED.len(),
        "num_code_groups": CODE_GROUPS,
    });
    let ids = json!({
        "codec_pad_id": control(0), "codec_bos_id": control(1), "codec_eos_token_id": control(2),
        "codec_think_id": control(3), "codec_nothink_id": control(4),
        "codec_think_bos_id": control(5), "codec_think_eos_id": control(6),
    });
    for (key, value) in ids.as_object().into_iter().flatten() {
        talker[key] = value.clone();
    }
    talker["codec_language_id"] = languages;
    talker["spk_id"] = speakers;
    talker["spk_is_dialect"] = dialects;
    talker["code_predictor_config"] = predictor;
    json!({
        "architectures": ["Qwen3TTSForConditionalGeneration"],
        "model_type": "qwen3_tts",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "tts_model_type": model_type,
        "tts_model_size": size,
        "tts_pad_token_id": BYTE_TOKENS + 3,
        "tts_bos_token_id": BYTE_TOKENS + 4,
        "tts_eos_token_id": BYTE_TOKENS + 5,
        "talker_config": talker,
    })
}

fn codec_config() -> serde_json::Value {
    json!({
        "output_sample_rate": 24000,
        "decoder_config": {
            "codebook_size": CODES, "codebook_dim": 16, "latent_dim": 16, "decoder_dim": 32,
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
            "hidden_act": "silu", "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "max_position_embeddings": 8000, "sliding_window": 4,
            "num_quantizers": CODE_GROUPS, "num_semantic_quantizers": 1,
            "upsample_rates": [2, 2], "upsampling_ratios": [2],
        },
    })
}

// Qwen2's files: every byte-level character as a token, no merges, and the added tokens after them
fn write_tokenizer(dir: &Path) -> anyhow::Result<()> {
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocab: serde_json::Map<String, serde_json::Value> = alphabet
        .iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i.into()))
        .collect();
    std::fs::write(
        dir.join("vocab.json"),
        serde_json::Value::Object(vocab).to_string(),
    )?;
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\n")?;
    let added: serde_json::Map<String, serde_json::Value> = ADDED
        .iter()
        .enumerate()
        .map(|(i, (content, special))| {
            (
                (BYTE_TOKENS + i).to_string(),
                json!({ "content": content, "special": special }),
            )
        })
        .collect();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        json!({ "added_tokens_decoder": added }).to_string(),
    )?;
    Ok(())
}

/// A tiny checkpoint of `model_type` (`custom_voice` or `voice_design`) at `size` (`0b6` takes no instructions).
pub fn tiny_qwen3_tts_checkpoint(
    model_type: &str,
    size: &str,
) -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    let config = config(model_type, size).to_string();
    let codec = codec_config().to_string();
    std::fs::write(dir.path().join("config.json"), &config)?;
    let codec_dir = dir.path().join("speech_tokenizer");
    std::fs::create_dir(&codec_dir)?;
    std::fs::write(codec_dir.join("config.json"), &codec)?;
    write_tokenizer(dir.path())?;
    let cfg = Qwen3TtsConfig::from_json(&config)?;
    let codec_cfg: CodecConfig = serde_json::from_str(&codec)?;
    let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    let [talker, codec] = recording::record_sharded_pair(|vb, codec_vb| {
        Qwen3Tts::new(
            cfg,
            &codec_cfg,
            &vb,
            &codec_vb,
            tokenizer,
            &inference_tensor::Device::Cpu,
        )
        .map(|_| ())
    })?;
    inference_tensor::safetensors::save(&talker, dir.path().join("model.safetensors"))?;
    inference_tensor::safetensors::save(&codec, codec_dir.join("model.safetensors"))?;
    Ok(dir)
}
