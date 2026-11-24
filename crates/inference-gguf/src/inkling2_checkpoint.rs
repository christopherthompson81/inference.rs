//! An Inkling 2 checkpoint's GGUF metadata, as llama.cpp's `conversion/inkling2.py` writes it for `inkling2_loop`: the
//! text config under the architecture's keys, the generation defaults and the gpt2 tokenizer with the chat template.

use std::path::Path;

use anyhow::{Context, Result};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::Value as Json;

const ARCHITECTURE: &str = "inkling2_loop";
// llama.cpp's pre-tokenizer for Inkling 2's split regex
const PRE_TOKENIZER: &str = "inkling2";
const TOKENIZER_MODEL: &str = "gpt2";
const QUANTIZATION_VERSION: u32 = 2;
// llama.cpp's token types: an ordinary vocabulary entry, and an added special one
const TOKEN_NORMAL: i32 = 1;
const TOKEN_CONTROL: i32 = 3;

fn read_json(dir: &Path, file: &str) -> Result<Json> {
    let path = dir.join(file);
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

fn field<'a>(config: &'a Json, key: &str) -> Result<&'a Json> {
    config
        .get(key)
        .with_context(|| format!("Inkling 2's config has no `{key}`"))
}

// a count, which configs sometimes write as a whole float
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn u32_of(config: &Json, key: &str) -> Result<Value> {
    let value = field(config, key)?;
    let count = value
        .as_u64()
        .or_else(|| {
            value
                .as_f64()
                .filter(|v| v.fract() == 0.0 && *v >= 0.0)
                .map(|v| v as u64)
        })
        .with_context(|| format!("`{key}` is not a count"))?;
    Ok(Value::U32(u32::try_from(count)?))
}

#[allow(clippy::cast_possible_truncation)]
fn f32_of(config: &Json, key: &str) -> Result<Value> {
    let value = field(config, key)?
        .as_f64()
        .with_context(|| format!("`{key}` is not a number"))?;
    Ok(Value::F32(value as f32))
}

/// The GGUF metadata llama.cpp writes for the Inkling 2 checkpoint in `dir`: its text config under `inkling2_loop.*`,
/// `general.sampling.*` from `generation_config.json`, and the tokenizer from `tokenizer.json`.
pub fn inkling2_checkpoint_metadata(dir: &Path) -> Result<Vec<(String, Value)>> {
    let config = read_json(dir, "config.json")?;
    let text = field(&config, "text_config")?;
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            Value::String(ARCHITECTURE.to_string()),
        ),
        (
            "general.type".to_string(),
            Value::String("model".to_string()),
        ),
    ];
    if let Ok(generation) = read_json(dir, "generation_config.json")
        && generation.get("temperature").is_some()
    {
        metadata.push((
            "general.sampling.temp".to_string(),
            f32_of(&generation, "temperature")?,
        ));
    }
    let key = |name: &str| format!("{ARCHITECTURE}.{name}");
    let layers = field(text, "num_hidden_layers")?
        .as_u64()
        .context("`num_hidden_layers` is not a count")? as usize;
    let local = field(text, "local_layer_ids")?
        .as_array()
        .context("`local_layer_ids` is not a list")?
        .iter()
        .filter_map(Json::as_u64)
        .collect::<Vec<_>>();
    let per_layer = |name: &str, swa: &str| -> Result<Value> {
        let (all, sliding) = (
            field(text, name)?,
            text.get(swa).unwrap_or(field(text, name)?),
        );
        let count =
            |v: &Json| -> Result<i32> { Ok(i32::try_from(v.as_u64().context("a head count")?)?) };
        let (all, sliding) = (count(all)?, count(sliding)?);
        Ok(Value::Array(
            (0..layers as u64)
                .map(|l| Value::I32(if local.contains(&l) { sliding } else { all }))
                .collect(),
        ))
    };
    let head_dim = u32_of(text, "head_dim")?;
    let swa_head_dim = text
        .get("swa_head_dim")
        .map_or(Ok(head_dim.clone()), |_| u32_of(text, "swa_head_dim"))?;
    let loop_config = field(text, "loop_config")?;
    metadata.extend([
        (key("block_count"), u32_of(text, "num_hidden_layers")?),
        (
            key("context_length"),
            u32_of(text, "model_max_length")
                .or_else(|_| u32_of(text, "max_position_embeddings"))?,
        ),
        (key("embedding_length"), u32_of(text, "hidden_size")?),
        (
            key("feed_forward_length"),
            u32_of(text, "intermediate_size")?,
        ),
        (
            key("attention.head_count"),
            per_layer("num_attention_heads", "swa_num_attention_heads")?,
        ),
        (
            key("attention.head_count_kv"),
            per_layer("num_key_value_heads", "swa_num_key_value_heads")?,
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            f32_of(text, "rms_norm_eps")?,
        ),
        (key("expert_count"), u32_of(text, "n_routed_experts")?),
        (
            key("expert_used_count"),
            u32_of(text, "num_experts_per_tok")?,
        ),
        (key("attention.key_length"), head_dim.clone()),
        (key("attention.value_length"), head_dim),
        (key("vocab_size"), u32_of(text, "vocab_size")?),
        (
            key("vocab_size_unpadded"),
            u32_of(text, "unpadded_vocab_size")?,
        ),
        (
            key("expert_feed_forward_length"),
            u32_of(text, "intermediate_size")?,
        ),
        (key("attention.key_length_swa"), swa_head_dim.clone()),
        (key("attention.value_length_swa"), swa_head_dim),
        (key("rope.dimension_count"), Value::U32(0)),
        (
            key("attention.sliding_window"),
            u32_of(text, "sliding_window_size")?,
        ),
        (
            key("attention.sliding_window_pattern"),
            Value::Array(
                (0..layers as u64)
                    .map(|l| Value::Bool(local.contains(&l)))
                    .collect(),
            ),
        ),
        (key("shortconv.l_cache"), u32_of(text, "sconv_kernel_size")?),
        (key("expert_weights_scale"), f32_of(text, "route_scale")?),
        (
            key("activation.situ_beta"),
            f32_of(text, "mlp_gate_softcap")?,
        ),
        (
            key("activation.situ_linear_beta"),
            f32_of(text, "mlp_up_softcap")?,
        ),
        (
            key("attn_res.block_size"),
            u32_of(text, "n_layers_per_attn_res_block")?,
        ),
        (
            key("final_logit_softcapping"),
            f32_of(text, "final_logit_softcapping")?,
        ),
        (key("attention.relative_length"), u32_of(text, "d_rel")?),
        (
            key("attention.relative_extent"),
            u32_of(text, "rel_extent")?,
        ),
        (
            key("attention.qk_clip"),
            f32_of(text, "qk_l2_clip_threshold")?,
        ),
        (
            key("attention.relative_clip"),
            f32_of(text, "rel_l2_clip_threshold")?,
        ),
        (
            key("attention.log_scale_floor"),
            u32_of(text, "log_scaling_n_floor")?,
        ),
        (
            key("attention.log_scale_alpha"),
            f32_of(text, "log_scaling_alpha")?,
        ),
        (key("loop.start"), u32_of(loop_config, "core_start")?),
        (key("loop.end"), u32_of(loop_config, "core_end")?),
        (key("loop.count"), u32_of(loop_config, "num_passes")?),
        (key("loop.gate_rank"), u32_of(loop_config, "gate_rank")?),
        (
            key("loop.varying"),
            Value::Bool(
                loop_config
                    .get("is_varloop")
                    .and_then(Json::as_bool)
                    .unwrap_or(false),
            ),
        ),
        (
            "general.quantization_version".to_string(),
            Value::U32(QUANTIZATION_VERSION),
        ),
    ]);
    let unpadded = field(text, "unpadded_vocab_size")?
        .as_u64()
        .context("`unpadded_vocab_size` is not a count")? as usize;
    let generation = read_json(dir, "generation_config.json").ok();
    let eos = [Some(&config), Some(text), generation.as_ref()]
        .into_iter()
        .flatten()
        .find_map(|c| c.get("eos_token_id").and_then(Json::as_u64))
        .context("Inkling 2's config names no `eos_token_id`")?;
    let eos = u32::try_from(eos)?;
    metadata.extend(tokenizer_metadata(dir, unpadded, eos)?);
    Ok(metadata)
}

// the gpt2 tokenizer over the vocabulary's first `vocab` ids, its added tokens special (control) ones
fn tokenizer_metadata(dir: &Path, vocab: usize, eos: u32) -> Result<Vec<(String, Value)>> {
    let tokenizer = read_json(dir, "tokenizer.json")?;
    let model = field(&tokenizer, "model")?;
    let mut tokens = vec![None; vocab];
    let mut types = vec![TOKEN_NORMAL; vocab];
    for (token, id) in field(model, "vocab")?
        .as_object()
        .context("a vocabulary map")?
    {
        if let Some(slot) = id.as_u64().and_then(|id| tokens.get_mut(id as usize)) {
            *slot = Some(token.clone());
        }
    }
    let mut pad = None;
    let tokenizer_config = read_json(dir, "tokenizer_config.json")?;
    let pad_token = tokenizer_config.get("pad_token").and_then(Json::as_str);
    for added in field(&tokenizer, "added_tokens")?
        .as_array()
        .context("a list of added tokens")?
    {
        let id = field(added, "id")?
            .as_u64()
            .context("an added token's id")? as usize;
        let content = field(added, "content")?
            .as_str()
            .context("an added token's text")?;
        if Some(content) == pad_token {
            pad = Some(u32::try_from(id)?);
        }
        if id < vocab {
            tokens[id] = Some(content.to_string());
            if added
                .get("special")
                .and_then(Json::as_bool)
                .unwrap_or(false)
            {
                types[id] = TOKEN_CONTROL;
            }
        }
    }
    let tokens = tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| token.with_context(|| format!("the tokenizer has no token {id}")))
        .collect::<Result<Vec<_>>>()?;
    let merges = field(model, "merges")?
        .as_array()
        .context("a list of merges")?
        .iter()
        .map(|merge| match merge {
            Json::String(pair) => Ok(pair.clone()),
            Json::Array(pair) => Ok(pair
                .iter()
                .filter_map(Json::as_str)
                .collect::<Vec<_>>()
                .join(" ")),
            other => anyhow::bail!("a merge of {other}"),
        })
        .collect::<Result<Vec<_>>>()?;
    let template = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .ok()
        .or_else(|| {
            tokenizer_config
                .get("chat_template")
                .and_then(Json::as_str)
                .map(str::to_string)
        });
    let clean_spaces = tokenizer_config
        .get("clean_up_tokenization_spaces")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let mut metadata = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::String(TOKENIZER_MODEL.to_string()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            Value::String(PRE_TOKENIZER.to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(tokens.into_iter().map(Value::String).collect()),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            Value::Array(types.into_iter().map(Value::I32).collect()),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Value::Array(merges.into_iter().map(Value::String).collect()),
        ),
        ("tokenizer.ggml.eos_token_id".to_string(), Value::U32(eos)),
    ];
    if let Some(pad) = pad {
        metadata.push((
            "tokenizer.ggml.padding_token_id".to_string(),
            Value::U32(pad),
        ));
    }
    metadata.extend([
        ("tokenizer.ggml.eot_token_id".to_string(), Value::U32(eos)),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Value::Bool(false),
        ),
        (
            "tokenizer.ggml.add_eos_token".to_string(),
            Value::Bool(false),
        ),
    ]);
    if let Some(template) = template {
        metadata.push((
            "tokenizer.chat_template".to_string(),
            Value::String(template),
        ));
    }
    metadata.push((
        "tokenizer.ggml.clean_spaces".to_string(),
        Value::Bool(clean_spaces),
    ));
    Ok(metadata)
}
