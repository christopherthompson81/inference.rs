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
    ("<tts_text_bos>", true),
    ("<tts_text_eod>", true),
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
    let mut config = json!({
        "architectures": ["Qwen3TTSForConditionalGeneration"],
        "model_type": "qwen3_tts",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "tts_model_type": model_type,
        "tts_model_size": size,
        "tts_pad_token_id": BYTE_TOKENS + 3,
        "tts_bos_token_id": BYTE_TOKENS + 4,
        "tts_eos_token_id": BYTE_TOKENS + 5,
        "talker_config": talker,
    });
    if model_type == "base" {
        // a Base model has no preset voices
        config["talker_config"]["spk_id"] = json!({});
        config["talker_config"]["spk_is_dialect"] = json!({});
        // a speaker encoder as the reference's, narrower; its x-vector is the talker's width
        config["speaker_encoder_config"] = json!({
            "mel_dim": 16, "enc_dim": 32, "enc_channels": [16, 16, 16, 16, 48],
            "enc_attention_channels": 8, "enc_res2net_scale": 4, "enc_se_channels": 8,
            "sample_rate": 24000,
        });
    }
    config
}

fn codec_config() -> serde_json::Value {
    // the decoder's head size, window, RoPE base and eps are the release's, which llama.cpp's GGUF does not store;
    // the encoder's 2 x 2 downsampling and Mimi's own 2 give the decoder's 8 samples per frame
    let encoder = json!({
        "num_filters": 4, "kernel_size": 7, "last_kernel_size": 3, "residual_kernel_size": 3,
        "dilation_growth_rate": 2, "num_residual_layers": 1, "compress": 2, "upsampling_ratios": [2, 2],
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "head_dim": 8, "norm_eps": 1e-5, "rope_theta": 10000.0,
        "max_position_embeddings": 8000, "codebook_size": CODES, "codebook_dim": 8,
        "num_quantizers": CODE_GROUPS + 2, "num_semantic_quantizers": 1,
        "vector_quantization_hidden_dimension": 8,
    });
    json!({
        "output_sample_rate": 24000,
        "encoder_valid_num_quantizers": CODE_GROUPS,
        "encoder_config": encoder,
        "decoder_config": {
            "codebook_size": CODES, "codebook_dim": 16, "latent_dim": 16, "decoder_dim": 32,
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 64,
            "hidden_act": "silu", "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "max_position_embeddings": 8000, "sliding_window": 72,
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

/// A tiny checkpoint of `model_type` (`custom_voice`, `voice_design` or `base`) at `size` (`0b6` takes no instructions).
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

// llama.cpp's token types
const TOKEN_NORMAL: i32 = 1;
const TOKEN_CONTROL: i32 = 3;
const TOKEN_USER_DEFINED: i32 = 4;
const SNAKE_EPS: f64 = 1e-9;
const USAGE_EPS: f64 = 1e-5;

/// A tiny Base checkpoint converted as llama.cpp's `conversion/qwen3tts.py` converts one: a talker GGUF (text
/// projection folded into the token table, codec rows appended and named) and an mmproj (speaker encoder, stacked code
/// predictor tables and heads, the codec decoder with SnakeBeta and the codebooks folded), F32, beside the checkpoint.
pub fn tiny_qwen3_tts_gguf(checkpoint: &Path) -> anyhow::Result<std::path::PathBuf> {
    use inference_tensor::quantized::{GgmlDType, QTensor, gguf_file::Value};
    use inference_tensor::{D, Device, Tensor};
    use std::collections::HashMap;

    let load = |path: std::path::PathBuf| inference_tensor::safetensors::load(path, &Device::Cpu);
    let model = load(checkpoint.join("model.safetensors"))?;
    let codec = load(
        checkpoint
            .join("speech_tokenizer")
            .join("model.safetensors"),
    )?;
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let get = |map: &HashMap<String, Tensor>, name: &str| {
        map.get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no `{name}` in the tiny checkpoint"))
    };
    let talker = &cfg["talker_config"];
    let predictor = &talker["code_predictor_config"];
    let speaker = &cfg["speaker_encoder_config"];
    let n = |v: &serde_json::Value| v.as_u64().map(|v| u32::try_from(v).unwrap()).unwrap();
    let f = |v: &serde_json::Value| v.as_f64().unwrap() as f32;

    // the talker: the text projection folded into the text rows, then the codec rows
    let mut talker_tensors: Vec<(String, Tensor)> = Vec::new();
    let linear = |x: &Tensor, prefix: &str| -> anyhow::Result<Tensor> {
        let w = get(&model, &format!("{prefix}.weight"))?;
        let b = get(&model, &format!("{prefix}.bias"))?;
        Ok(x.matmul(&w.t()?)?.broadcast_add(&b)?)
    };
    let text = get(&model, "talker.model.text_embedding.weight")?;
    let hidden =
        inference_tensor::nn::ops::silu(&linear(&text, "talker.text_projection.linear_fc1")?)?;
    let folded = linear(&hidden, "talker.text_projection.linear_fc2")?;
    let codec_rows = get(&model, "talker.model.codec_embedding.weight")?;
    talker_tensors.push((
        "token_embd.weight".into(),
        Tensor::cat(&[&folded, &codec_rows], 0)?,
    ));
    talker_tensors.push((
        "output.weight".into(),
        get(&model, "talker.codec_head.weight")?,
    ));
    talker_tensors.push((
        "output_norm.weight".into(),
        get(&model, "talker.model.norm.weight")?,
    ));
    let layer_names = [
        ("attn_norm", "input_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
    ];
    for i in 0..n(&talker["num_hidden_layers"]) {
        for (gguf, native) in layer_names {
            talker_tensors.push((
                format!("blk.{i}.{gguf}.weight"),
                get(&model, &format!("talker.model.layers.{i}.{native}.weight"))?,
            ));
        }
    }

    // the mmproj: code predictor, speaker encoder, codec decoder
    let mut mm: Vec<(String, Tensor)> = Vec::new();
    let gen_names = layer_names.map(|(gguf, native)| {
        let gguf = match gguf {
            "attn_norm" => "ln1",
            "attn_output" => "attn_out",
            "ffn_norm" => "ln2",
            other => other,
        };
        (gguf, native)
    });
    let cp = "talker.code_predictor.";
    for i in 0..n(&predictor["num_hidden_layers"]) {
        for (gguf, native) in gen_names {
            mm.push((
                format!("a.gen.code.blk.{i}.{gguf}.weight"),
                get(&model, &format!("{cp}model.layers.{i}.{native}.weight"))?,
            ));
        }
    }
    mm.push((
        "a.gen.code.output_norm.weight".into(),
        get(&model, &format!("{cp}model.norm.weight"))?,
    ));
    for part in ["weight", "bias"] {
        mm.push((
            format!("a.gen.code.proj_in.{part}"),
            get(&model, &format!("{cp}small_to_mtp_projection.{part}"))?,
        ));
    }
    let books = CODE_GROUPS - 1;
    let stack = |name: &dyn Fn(usize) -> String| -> anyhow::Result<Tensor> {
        let parts = (0..books)
            .map(|i| get(&model, &name(i)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Tensor::stack(&parts, 0)?)
    };
    mm.push((
        "a.gen.code.embd.weight".into(),
        stack(&|i| format!("{cp}model.codec_embedding.{i}.weight"))?,
    ));
    mm.push((
        "a.gen.code.head.weight".into(),
        stack(&|i| format!("{cp}lm_head.{i}.weight"))?,
    ));
    let module = |mm: &mut Vec<(String, Tensor)>, gguf: &str, native: &str| -> anyhow::Result<()> {
        for part in ["weight", "bias"] {
            mm.push((
                format!("{gguf}.{part}"),
                get(&model, &format!("{native}.{part}"))?,
            ));
        }
        Ok(())
    };
    let se = "speaker_encoder.";
    module(&mut mm, "a.conv1d.0", &format!("{se}blocks.0.conv"))?;
    let channels = speaker["enc_channels"].as_array().unwrap().len();
    let scale = n(&speaker["enc_res2net_scale"]) as usize;
    for i in 1..channels - 1 {
        let blk = format!("{se}blocks.{i}.");
        module(
            &mut mm,
            &format!("a.blk.{i}.conv_pw1"),
            &format!("{blk}tdnn1.conv"),
        )?;
        module(
            &mut mm,
            &format!("a.blk.{i}.conv_pw2"),
            &format!("{blk}tdnn2.conv"),
        )?;
        module(
            &mut mm,
            &format!("a.blk.{i}.se_conv1"),
            &format!("{blk}se_block.conv1"),
        )?;
        module(
            &mut mm,
            &format!("a.blk.{i}.se_conv2"),
            &format!("{blk}se_block.conv2"),
        )?;
        for j in 0..scale - 1 {
            module(
                &mut mm,
                &format!("a.blk.{i}.res2.{j}"),
                &format!("{blk}res2net_block.blocks.{j}.conv"),
            )?;
        }
    }
    module(&mut mm, "a.conv_out", &format!("{se}mfa.conv"))?;
    module(&mut mm, "a.asp_tdnn", &format!("{se}asp.tdnn.conv"))?;
    module(&mut mm, "a.asp_attn", &format!("{se}asp.conv"))?;
    module(&mut mm, "mm.a.fc", &format!("{se}fc"))?;

    let c = |name: &str| get(&codec, &format!("decoder.{name}"));
    let w = "a.gen.wav.";
    let codec_module =
        |mm: &mut Vec<(String, Tensor)>, gguf: &str, native: &str| -> anyhow::Result<()> {
            for part in ["weight", "bias"] {
                mm.push((format!("{w}{gguf}.{part}"), c(&format!("{native}.{part}"))?));
            }
            Ok(())
        };
    let snake = |mm: &mut Vec<(String, Tensor)>, gguf: &str, native: &str| -> anyhow::Result<()> {
        mm.push((
            format!("{w}{gguf}.alpha"),
            c(&format!("{native}.alpha"))?.exp()?,
        ));
        let beta = ((c(&format!("{native}.beta"))?.exp()? + SNAKE_EPS)?).recip()?;
        mm.push((format!("{w}{gguf}.beta"), beta));
        Ok(())
    };
    let codebook = |prefix: &str, i: usize| -> anyhow::Result<Tensor> {
        let book = format!("quantizer.{prefix}.vq.layers.{i}._codebook");
        let usage = c(&format!("{book}.cluster_usage"))?.clamp(USAGE_EPS, f64::INFINITY)?;
        Ok(c(&format!("{book}.embedding_sum"))?.broadcast_div(&usage.unsqueeze(1)?)?)
    };
    mm.push((
        format!("{w}quant.first.codebook.weight"),
        codebook("rvq_first", 0)?,
    ));
    let rest = (0..books)
        .map(|i| codebook("rvq_rest", i))
        .collect::<anyhow::Result<Vec<_>>>()?;
    mm.push((
        format!("{w}quant.rest.codebook.weight"),
        Tensor::stack(&rest, 0)?,
    ));
    for (gguf, native) in [("first", "rvq_first"), ("rest", "rvq_rest")] {
        mm.push((
            format!("{w}quant.{gguf}.out_proj.weight"),
            c(&format!("quantizer.{native}.output_proj.weight"))?.squeeze(D::Minus1)?,
        ));
    }
    codec_module(&mut mm, "pre_conv", "pre_conv.conv")?;
    codec_module(&mut mm, "tfm.in_proj", "pre_transformer.input_proj")?;
    codec_module(&mut mm, "tfm.out_proj", "pre_transformer.output_proj")?;
    mm.push((
        format!("{w}tfm.output_norm.weight"),
        c("pre_transformer.norm.weight")?,
    ));
    let decoder = &codec_config()["decoder_config"];
    for i in 0..n(&decoder["num_hidden_layers"]) {
        let t = format!("pre_transformer.layers.{i}.");
        for (gguf, native) in gen_names
            .iter()
            .filter(|(g, _)| !g.ends_with("_norm") || g.starts_with("ln"))
        {
            mm.push((
                format!("{w}tfm.blk.{i}.{gguf}.weight"),
                c(&format!("{t}{native}.weight"))?,
            ));
        }
        mm.push((
            format!("{w}tfm.blk.{i}.ls1.weight"),
            c(&format!("{t}self_attn_layer_scale.scale"))?,
        ));
        mm.push((
            format!("{w}tfm.blk.{i}.ls2.weight"),
            c(&format!("{t}mlp_layer_scale.scale"))?,
        ));
    }
    let ups = decoder["upsampling_ratios"].as_array().unwrap().len();
    for i in 0..ups {
        let (g, d) = (format!("up.blk.{i}."), format!("upsample.{i}."));
        codec_module(&mut mm, &format!("{g}conv"), &format!("{d}0.conv"))?;
        codec_module(&mut mm, &format!("{g}dwconv"), &format!("{d}1.dwconv.conv"))?;
        codec_module(&mut mm, &format!("{g}norm"), &format!("{d}1.norm"))?;
        codec_module(&mut mm, &format!("{g}pw1"), &format!("{d}1.pwconv1"))?;
        codec_module(&mut mm, &format!("{g}pw2"), &format!("{d}1.pwconv2"))?;
        mm.push((format!("{w}{g}gamma"), c(&format!("{d}1.gamma"))?));
    }
    codec_module(&mut mm, "dac.entry", "decoder.0.conv")?;
    let blocks = decoder["upsample_rates"].as_array().unwrap().len();
    for i in 0..blocks {
        let (g, d) = (format!("dac.blk.{i}."), format!("decoder.{}.block.", i + 1));
        snake(&mut mm, &format!("{g}snake"), &format!("{d}0"))?;
        codec_module(&mut mm, &format!("{g}conv"), &format!("{d}1.conv"))?;
        for x in 0..3 {
            let (r, u) = (format!("{g}res.{x}."), format!("{d}{}.", x + 2));
            snake(&mut mm, &format!("{r}act1"), &format!("{u}act1"))?;
            codec_module(&mut mm, &format!("{r}conv1"), &format!("{u}conv1.conv"))?;
            snake(&mut mm, &format!("{r}act2"), &format!("{u}act2"))?;
            codec_module(&mut mm, &format!("{r}conv2"), &format!("{u}conv2.conv"))?;
        }
    }
    snake(
        &mut mm,
        "dac.post_snake",
        &format!("decoder.{}", blocks + 1),
    )?;
    codec_module(
        &mut mm,
        "dac.post_conv",
        &format!("decoder.{}.conv", blocks + 2),
    )?;

    // the vocabulary: the bytes, the added tokens, then a name per codec row
    let vocab: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(checkpoint.join("vocab.json"))?)?;
    let mut tokens = vec![String::new(); vocab.len()];
    for (token, id) in vocab {
        tokens[id.as_u64().unwrap() as usize] = token;
    }
    let mut types = vec![TOKEN_NORMAL; tokens.len()];
    for (content, special) in ADDED {
        tokens.push(content.to_string());
        types.push(if special {
            TOKEN_CONTROL
        } else {
            TOKEN_USER_DEFINED
        });
    }
    let codec_vocab = n(&talker["vocab_size"]) as usize;
    let mut names: Vec<String> = (0..codec_vocab).map(|i| format!("<|codec_{i}|>")).collect();
    for (key, value) in talker.as_object().unwrap() {
        let Some(prefix) = key.strip_suffix("_id") else {
            continue;
        };
        match value {
            serde_json::Value::Number(id) => {
                names[id.as_u64().unwrap() as usize] = format!("<|{prefix}|>")
            }
            serde_json::Value::Object(ids) => {
                for (name, id) in ids {
                    names[id.as_u64().unwrap() as usize] = format!("<|{prefix}_{name}|>");
                }
            }
            _ => {}
        }
    }
    types.extend(std::iter::repeat_n(TOKEN_CONTROL, names.len()));
    tokens.extend(names);

    let arch = "qwen3tts";
    let talker_meta = vec![
        (
            "general.architecture".to_string(),
            Value::String(arch.into()),
        ),
        ("general.type".to_string(), Value::String("model".into())),
        (
            format!("{arch}.block_count"),
            Value::U32(n(&talker["num_hidden_layers"])),
        ),
        (
            format!("{arch}.context_length"),
            Value::U32(n(&talker["max_position_embeddings"])),
        ),
        (
            format!("{arch}.embedding_length"),
            Value::U32(n(&talker["hidden_size"])),
        ),
        (
            format!("{arch}.feed_forward_length"),
            Value::U32(n(&talker["intermediate_size"])),
        ),
        (
            format!("{arch}.attention.head_count"),
            Value::U32(n(&talker["num_attention_heads"])),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            Value::U32(n(&talker["num_key_value_heads"])),
        ),
        (
            format!("{arch}.attention.key_length"),
            Value::U32(n(&talker["head_dim"])),
        ),
        (
            format!("{arch}.rope.freq_base"),
            Value::F32(f(&talker["rope_theta"])),
        ),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            Value::F32(f(&talker["rms_norm_eps"])),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(tokens.into_iter().map(Value::String).collect()),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            Value::Array(types.into_iter().map(Value::I32).collect()),
        ),
    ];
    let mm_meta = vec![
        (
            "general.architecture".to_string(),
            Value::String("clip".into()),
        ),
        ("general.type".to_string(), Value::String("mmproj".into())),
        (
            "clip.audio.num_mel_bins".to_string(),
            Value::U32(n(&speaker["mel_dim"])),
        ),
        (
            "clip.audio.projection_dim".to_string(),
            Value::U32(n(&speaker["enc_dim"])),
        ),
        (
            "clip.gen.audio.embedding_length".to_string(),
            Value::U32(n(&predictor["hidden_size"])),
        ),
        (
            "clip.gen.audio.feed_forward_length".to_string(),
            Value::U32(n(&predictor["intermediate_size"])),
        ),
        (
            "clip.gen.audio.block_count".to_string(),
            Value::U32(n(&predictor["num_hidden_layers"])),
        ),
        (
            "clip.gen.audio.attention.head_count".to_string(),
            Value::U32(n(&predictor["num_attention_heads"])),
        ),
        (
            "clip.gen.audio.attention.head_count_kv".to_string(),
            Value::U32(n(&predictor["num_key_value_heads"])),
        ),
        (
            "clip.gen.audio.attention.layer_norm_epsilon".to_string(),
            Value::F32(f(&predictor["rms_norm_eps"])),
        ),
    ];
    let write = |path: &Path,
                 meta: &[(String, Value)],
                 tensors: &[(String, Tensor)]|
     -> anyhow::Result<()> {
        let quantized = tensors
            .iter()
            .map(|(name, t)| {
                Ok((
                    name.clone(),
                    QTensor::quantize(&t.contiguous()?, GgmlDType::F32)?,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let meta = meta
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>();
        let quantized = quantized
            .iter()
            .map(|(k, q)| (k.as_str(), q))
            .collect::<Vec<_>>();
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        inference_tensor::quantized::gguf_file::write(&mut out, &meta, &quantized)?;
        Ok(())
    };
    let talker_path = checkpoint.join("Qwen3-TTS-tiny-F32.gguf");
    write(&talker_path, &talker_meta, &talker_tensors)?;
    write(
        &checkpoint.join("mmproj-Qwen3-TTS-tiny-F32.gguf"),
        &mm_meta,
        &mm,
    )?;
    Ok(talker_path)
}
