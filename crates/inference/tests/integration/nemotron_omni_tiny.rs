//! A tiny random-weight Nemotron-H Omni (C-RADIO, pixel shuffle, ReLU^2 projector, Nemotron-H): an image expands to
//! `<img>`, one `<image>` per token of its patch grid, `</img>`; images of two sizes in one request take their own
//! counts; and a repeat served from the prefix cache decodes the same tokens. A clip beside them goes through a
//! Parakeet encoder and expands to `<so_start>`, one `<so_embedding>` per encoder frame, `<so_end>`.

use std::collections::HashMap;

use inference::{
    AudioInput, MessageMedia, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder,
    RequestBuilder, TextMessageRole,
};
use inference_tensor::{Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
// `<image>\n` per image part and `<so_embedding>\n` per audio part, as the checkpoints' templates write them
const TEMPLATE: &str = "{%- for message in messages -%}{{- message['role'] + ': ' -}}{%- if message['content'] is string -%}{{- message['content'] -}}{%- else -%}{%- for chunk in message['content'] -%}{%- if chunk['type'] == 'image' -%}{{- '<image>\\n' -}}{%- elif chunk['type'] == 'audio' -%}{{- '<so_embedding>\\n' -}}{%- elif chunk['type'] == 'text' -%}{{- chunk['text'] -}}{%- endif -%}{%- endfor -%}{%- endif -%}{{- '\\n' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- 'assistant: ' -}}{%- endif -%}";
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llama_tiny");
const FIXTURE_FILES: [&str; 2] = ["generation_config.json", "tokenizer_config.json"];
// appended after the tiny tokenizer's own
const IMAGE_TOKENS: [(&str, u32); 3] = [("<image>", 266), ("<img>", 267), ("</img>", 268)];
const SOUND_TOKENS: [(&str, u32); 3] = [
    ("<so_embedding>", 269),
    ("<so_start>", 270),
    ("<so_end>", 271),
];
const VOCAB: usize = 272;
const HIDDEN: usize = 32;
const LAYERS: [&str; 3] = ["mamba", "attention", "mlp"];
const HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const MAMBA_HEADS: usize = 4;
const MAMBA_HEAD_DIM: usize = 8;
const GROUPS: usize = 2;
const STATE: usize = 8;
const CONV: usize = 4;
const MLP_WIDTH: usize = 48;
const VISION_HIDDEN: usize = 32;
const VISION_HEADS: usize = 2;
const VISION_BLOCKS: usize = 2;
const VISION_FFN: usize = 4 * VISION_HIDDEN;
const PATCH: usize = 16;
// a 6 x 6 position grid, smaller than the larger image's 6 x 8 patches, so it is resized and cropped
const MAX_IMG_SIZE: usize = 6 * PATCH;
const CLS_TOKENS: usize = 2;
const REGISTERS: usize = 1;
const MIN_PATCHES: usize = 4;
const MAX_PATCHES: usize = 64;
const PROJECTOR_HIDDEN: usize = 48;
const MERGE: usize = 2;
const SOUND_HIDDEN: usize = 16;
const SOUND_HEADS: usize = 2;
const SOUND_FFN: usize = 32;
const SOUND_CONV: usize = 3;
const SUBSAMPLING_CHANNELS: usize = 4;
const SUBSAMPLING_KERNEL: usize = 3;
const SUBSAMPLING_FACTOR: usize = 8;
const MEL_BINS: usize = 16;
const SOUND_PROJECTOR_HIDDEN: usize = 24;
const SAMPLE_RATE: u32 = 16_000;
const HOP: usize = 160;
const TONE_HZ: f32 = 440.0;
// 0.5 s: 50 mel frames, through three stride-2 convs 25, 13, 7
const CLIP_SAMPLES: usize = 8_000;
const CLIP_TOKENS: usize = 7;
const WEIGHT_STD: f32 = 0.2;
const WEIGHT_SEED: u64 = 0x6f6d_6e69;
const PROMPT: &str = "Describe it.";
const MAX_LEN: usize = 6;
// 100 x 70 sizes to a 6 x 8 patch grid (12 tokens), 40 x 40 to 4 x 4 (4 tokens)
const WIDE: (u32, u32, usize) = (100, 70, 12);
const SQUARE: (u32, u32, usize) = (40, 40, 4);
// `<img>` and `</img>` around each image's tokens, `<so_start>` and `<so_end>` around a clip's
const DELIMITERS: usize = 2;
// cached KV comes from a different prefill than a full recompute, so logprobs match only to rounding
const LOGPROB_TOLERANCE: f32 = 1e-3;

fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "model_type": "nemotron_h_omni",
        "downsample_ratio": 1.0 / MERGE as f64,
        "img_context_token_id": IMAGE_TOKENS[0].1,
        "norm_mean": [0.48, 0.46, 0.41],
        "norm_std": [0.27, 0.26, 0.28],
        "patch_size": PATCH,
        "projector_hidden_size": PROJECTOR_HIDDEN,
        "vision_final_layernorm": false,
        "sound_context_token_id": SOUND_TOKENS[0].1,
        "sound_config": {
            "model_type": "parakeet",
            "hidden_size": SOUND_HIDDEN,
            "num_attention_heads": SOUND_HEADS,
            "num_hidden_layers": 1,
            "intermediate_size": SOUND_FFN,
            "conv_kernel_size": SOUND_CONV,
            "convolution_bias": false,
            "subsampling_conv_channels": SUBSAMPLING_CHANNELS,
            "subsampling_conv_kernel_size": SUBSAMPLING_KERNEL,
            "subsampling_conv_stride": 2,
            "subsampling_factor": SUBSAMPLING_FACTOR,
            "num_mel_bins": MEL_BINS,
            "projection_hidden_size": SOUND_PROJECTOR_HIDDEN,
            "projection_bias": false,
            "sampling_rate": SAMPLE_RATE,
        },
        "vision_config": {
            "hidden_size": VISION_HIDDEN,
            "num_attention_heads": VISION_HEADS,
            "num_hidden_layers": VISION_BLOCKS,
            "layer_norm_eps": 1e-6,
            "mlp_ratio": (VISION_FFN / VISION_HIDDEN) as f64,
            "num_cls_tokens": CLS_TOKENS,
            "num_registers": REGISTERS,
            "max_img_size": MAX_IMG_SIZE,
            "min_num_patches": MIN_PATCHES,
            "max_num_patches": MAX_PATCHES,
        },
        "llm_config": {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "layers_block_type": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": 1,
            "head_dim": HEAD_DIM,
            "intermediate_size": MLP_WIDTH,
            "mlp_hidden_act": "relu2",
            "ssm_state_size": STATE,
            "mamba_num_heads": MAMBA_HEADS,
            "mamba_head_dim": MAMBA_HEAD_DIM,
            "n_groups": GROUPS,
            "conv_kernel": CONV,
            "layer_norm_epsilon": 1e-5,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false,
        },
    })
}

fn random(rng: &mut StdRng, shape: &[usize]) -> anyhow::Result<Tensor> {
    let normal = Normal::new(0f32, WEIGHT_STD)?;
    let count = shape.iter().product::<usize>();
    let data = (0..count).map(|_| normal.sample(rng)).collect::<Vec<_>>();
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?)
}

fn near_one(rng: &mut StdRng, shape: &[usize]) -> anyhow::Result<Tensor> {
    Ok((random(rng, shape)? + 1.0)?)
}

fn text_weights(rng: &mut StdRng, weights: &mut HashMap<String, Tensor>) -> anyhow::Result<()> {
    let root = "language_model.backbone";
    weights.insert(
        format!("{root}.embeddings.weight"),
        random(rng, &[VOCAB, HIDDEN])?,
    );
    weights.insert(format!("{root}.norm_f.weight"), near_one(rng, &[HIDDEN])?);
    weights.insert(
        "language_model.lm_head.weight".into(),
        random(rng, &[VOCAB, HIDDEN])?,
    );
    let inner = MAMBA_HEADS * MAMBA_HEAD_DIM;
    let conv_dim = inner + 2 * GROUPS * STATE;
    for (i, kind) in LAYERS.iter().enumerate() {
        let layer = format!("{root}.layers.{i}");
        weights.insert(format!("{layer}.norm.weight"), near_one(rng, &[HIDDEN])?);
        let tensors: Vec<(&str, Tensor)> = match *kind {
            "mamba" => vec![
                (
                    "in_proj.weight",
                    random(rng, &[inner + conv_dim + MAMBA_HEADS, HIDDEN])?,
                ),
                ("conv1d.weight", random(rng, &[conv_dim, 1, CONV])?),
                ("conv1d.bias", random(rng, &[conv_dim])?),
                ("dt_bias", random(rng, &[MAMBA_HEADS])?),
                ("A_log", random(rng, &[MAMBA_HEADS])?),
                ("D", near_one(rng, &[MAMBA_HEADS])?),
                ("norm.weight", near_one(rng, &[inner])?),
                ("out_proj.weight", random(rng, &[HIDDEN, inner])?),
            ],
            "attention" => vec![
                ("q_proj.weight", random(rng, &[HEADS * HEAD_DIM, HIDDEN])?),
                ("k_proj.weight", random(rng, &[HEAD_DIM, HIDDEN])?),
                ("v_proj.weight", random(rng, &[HEAD_DIM, HIDDEN])?),
                ("o_proj.weight", random(rng, &[HIDDEN, HEADS * HEAD_DIM])?),
            ],
            _ => vec![
                ("up_proj.weight", random(rng, &[MLP_WIDTH, HIDDEN])?),
                ("down_proj.weight", random(rng, &[HIDDEN, MLP_WIDTH])?),
            ],
        };
        for (name, tensor) in tensors {
            weights.insert(format!("{layer}.mixer.{name}"), tensor);
        }
    }
    Ok(())
}

// the released checkpoint's names: timm-style RADIO with a fused qkv, and `mlp1`
fn vision_weights(rng: &mut StdRng, weights: &mut HashMap<String, Tensor>) -> anyhow::Result<()> {
    let root = "vision_model.radio_model.model";
    let h = VISION_HIDDEN;
    let side = MAX_IMG_SIZE / PATCH;
    let patches = format!("{root}.patch_generator");
    weights.insert(
        format!("{patches}.embedder.weight"),
        random(rng, &[h, 3 * PATCH * PATCH])?,
    );
    weights.insert(
        format!("{patches}.pos_embed"),
        random(rng, &[1, side * side, h])?,
    );
    weights.insert(
        format!("{patches}.cls_token.token"),
        random(rng, &[CLS_TOKENS + REGISTERS, h])?,
    );
    for block in 0..VISION_BLOCKS {
        let b = format!("{root}.blocks.{block}");
        for (name, rows, cols) in [
            ("attn.qkv", 3 * h, h),
            ("attn.proj", h, h),
            ("mlp.fc1", VISION_FFN, h),
            ("mlp.fc2", h, VISION_FFN),
        ] {
            weights.insert(format!("{b}.{name}.weight"), random(rng, &[rows, cols])?);
            weights.insert(format!("{b}.{name}.bias"), random(rng, &[rows])?);
        }
        for norm in ["norm1", "norm2"] {
            weights.insert(format!("{b}.{norm}.weight"), near_one(rng, &[h])?);
            weights.insert(format!("{b}.{norm}.bias"), random(rng, &[h])?);
        }
    }
    let shuffled = h * MERGE * MERGE;
    weights.insert("mlp1.0.weight".into(), near_one(rng, &[shuffled])?);
    weights.insert(
        "mlp1.1.weight".into(),
        random(rng, &[PROJECTOR_HIDDEN, shuffled])?,
    );
    weights.insert(
        "mlp1.3.weight".into(),
        random(rng, &[HIDDEN, PROJECTOR_HIDDEN])?,
    );
    Ok(())
}

// transformers' Parakeet names: `sound_encoder.encoder.*`, then the reference's `sound_projection`
fn sound_weights(rng: &mut StdRng, weights: &mut HashMap<String, Tensor>) -> anyhow::Result<()> {
    let root = "sound_encoder.encoder";
    let (h, ch, k) = (SOUND_HIDDEN, SUBSAMPLING_CHANNELS, SUBSAMPLING_KERNEL);
    let sub = format!("{root}.subsampling");
    // module indices: 0 the strided conv, then (depthwise, pointwise) at 2, 3 and 5, 6
    for (layer, shape) in [
        (0, [ch, 1, k, k]),
        (2, [ch, 1, k, k]),
        (3, [ch, ch, 1, 1]),
        (5, [ch, 1, k, k]),
        (6, [ch, ch, 1, 1]),
    ] {
        weights.insert(format!("{sub}.layers.{layer}.weight"), random(rng, &shape)?);
        weights.insert(format!("{sub}.layers.{layer}.bias"), random(rng, &[ch])?);
    }
    let freq = MEL_BINS / SUBSAMPLING_FACTOR;
    weights.insert(
        format!("{sub}.linear.weight"),
        random(rng, &[h, ch * freq])?,
    );
    weights.insert(format!("{sub}.linear.bias"), random(rng, &[h])?);
    let layer = format!("{root}.layers.0");
    for ff in ["feed_forward1", "feed_forward2"] {
        weights.insert(
            format!("{layer}.{ff}.linear1.weight"),
            random(rng, &[SOUND_FFN, h])?,
        );
        weights.insert(
            format!("{layer}.{ff}.linear2.weight"),
            random(rng, &[h, SOUND_FFN])?,
        );
    }
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"] {
        weights.insert(
            format!("{layer}.self_attn.{proj}.weight"),
            random(rng, &[h, h])?,
        );
    }
    for bias in ["bias_u", "bias_v"] {
        weights.insert(
            format!("{layer}.self_attn.{bias}"),
            random(rng, &[SOUND_HEADS, h / SOUND_HEADS])?,
        );
    }
    let conv = format!("{layer}.conv");
    weights.insert(
        format!("{conv}.pointwise_conv1.weight"),
        random(rng, &[2 * h, h, 1])?,
    );
    weights.insert(
        format!("{conv}.depthwise_conv.weight"),
        random(rng, &[h, 1, SOUND_CONV])?,
    );
    weights.insert(
        format!("{conv}.pointwise_conv2.weight"),
        random(rng, &[h, h, 1])?,
    );
    weights.insert(format!("{conv}.norm.weight"), near_one(rng, &[h])?);
    weights.insert(format!("{conv}.norm.bias"), random(rng, &[h])?);
    weights.insert(format!("{conv}.norm.running_mean"), random(rng, &[h])?);
    weights.insert(
        format!("{conv}.norm.running_var"),
        near_one(rng, &[h])?.abs()?,
    );
    for norm in [
        "norm_feed_forward1",
        "norm_self_att",
        "norm_conv",
        "norm_feed_forward2",
        "norm_out",
    ] {
        weights.insert(format!("{layer}.{norm}.weight"), near_one(rng, &[h])?);
        weights.insert(format!("{layer}.{norm}.bias"), random(rng, &[h])?);
    }
    weights.insert("sound_projection.norm.weight".into(), near_one(rng, &[h])?);
    weights.insert(
        "sound_projection.linear1.weight".into(),
        random(rng, &[SOUND_PROJECTOR_HIDDEN, h])?,
    );
    weights.insert(
        "sound_projection.linear2.weight".into(),
        random(rng, &[HIDDEN, SOUND_PROJECTOR_HIDDEN])?,
    );
    Ok(())
}

// the tiny tokenizer with the image and sound tokens appended
fn tokenizer() -> anyhow::Result<String> {
    let mut tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(TOKENIZER)?)?;
    let added = tokenizer["added_tokens"]
        .as_array_mut()
        .ok_or_else(|| anyhow::anyhow!("the tiny tokenizer has no added tokens"))?;
    for (content, id) in IMAGE_TOKENS.into_iter().chain(SOUND_TOKENS) {
        added.push(serde_json::json!({
            "id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true,
        }));
    }
    Ok(tokenizer.to_string())
}

fn checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for file in FIXTURE_FILES {
        std::fs::copy(format!("{FIXTURE}/{file}"), dir.path().join(file))?;
    }
    std::fs::write(dir.path().join("chat_template.jinja"), TEMPLATE)?;
    std::fs::write(dir.path().join("tokenizer.json"), tokenizer()?)?;
    std::fs::write(dir.path().join("config.json"), config().to_string())?;
    let rng = &mut StdRng::seed_from_u64(WEIGHT_SEED);
    let mut weights = HashMap::new();
    text_weights(rng, &mut weights)?;
    vision_weights(rng, &mut weights)?;
    sound_weights(rng, &mut weights)?;
    inference_tensor::safetensors::save(&weights, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

async fn build(dir: &std::path::Path) -> anyhow::Result<Model> {
    Ok(MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_prefix_cache_n(Some(16))
        .build()
        .await?)
}

fn image((width, height, _): (u32, u32, usize)) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x * 5) as u8, (y * 8) as u8, ((x + y) * 3) as u8])
    }))
}

fn request(images: &[(u32, u32, usize)]) -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        images.iter().map(|&size| image(size)).collect(),
    ))
    .set_sampler_max_len(MAX_LEN)
    .set_sampler_topk(1)
    .return_logprobs(true)
    .set_sampler_topn_logprobs(1)
}

fn trace(resp: &inference::ChatCompletionResponse) -> Vec<(u32, f32)> {
    resp.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| (t.top_logprobs[0].token, t.top_logprobs[0].logprob))
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn images_expand_to_their_patch_grids_and_a_cached_repeat_decodes_the_same()
-> anyhow::Result<()> {
    let checkpoint = checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let text_only = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, PROMPT)
                .set_sampler_max_len(MAX_LEN),
        )
        .await?
        .usage
        .prompt_tokens;
    // each image also adds the template's newline after it, one token
    for images in [vec![WIDE], vec![WIDE, SQUARE]] {
        let first = model.send_chat_request(request(&images)).await?;
        let tokens = images
            .iter()
            .map(|&(_, _, n)| n + DELIMITERS)
            .sum::<usize>();
        let added = first.usage.prompt_tokens - text_only;
        assert_eq!(
            added,
            tokens + images.len(),
            "{} images added {added} prompt tokens",
            images.len()
        );
        let first = trace(&first);
        assert!(!first.is_empty(), "the image request generated nothing");
        let second = trace(&model.send_chat_request(request(&images)).await?);
        assert_eq!(first.len(), second.len());
        for ((a, la), (b, lb)) in first.iter().zip(&second) {
            assert_eq!(a, b);
            assert!((la - lb).abs() < LOGPROB_TOLERANCE, "{la} vs {lb}");
        }
    }
    Ok(())
}

fn clip() -> AudioInput {
    AudioInput {
        samples: (0..CLIP_SAMPLES)
            .map(|i| {
                0.3 * (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect(),
        sample_rate: SAMPLE_RATE,
        channels: 1,
    }
}

fn media_request(images: &[(u32, u32, usize)], clips: usize) -> RequestBuilder {
    let media = MessageMedia {
        images: images.iter().map(|&size| image(size)).collect(),
        audios: (0..clips).map(|_| clip()).collect(),
        ..MessageMedia::default()
    };
    RequestBuilder::from(MultimodalMessages::new().add_multimodal_message(
        TextMessageRole::User,
        PROMPT,
        media,
    ))
    .set_sampler_max_len(MAX_LEN)
    .set_sampler_topk(1)
    .return_logprobs(true)
    .set_sampler_topn_logprobs(1)
}

#[tokio::test]
async fn a_clip_expands_to_its_encoder_frames_beside_an_image() -> anyhow::Result<()> {
    assert_eq!(CLIP_SAMPLES / HOP, 50);
    let checkpoint = checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let image_only = model
        .send_chat_request(media_request(&[SQUARE], 0))
        .await?
        .usage
        .prompt_tokens;
    for clips in [1, 2] {
        let first = model
            .send_chat_request(media_request(&[SQUARE], clips))
            .await?;
        // each clip also adds the template's newline after it
        let added = first.usage.prompt_tokens - image_only;
        assert_eq!(
            added,
            clips * (CLIP_TOKENS + DELIMITERS + 1),
            "{clips} clips added {added} tokens"
        );
        let first = trace(&first);
        assert!(!first.is_empty(), "the audio request generated nothing");
        let second = trace(
            &model
                .send_chat_request(media_request(&[SQUARE], clips))
                .await?,
        );
        assert_eq!(first.len(), second.len());
        for ((a, la), (b, lb)) in first.iter().zip(&second) {
            assert_eq!(a, b);
            assert!((la - lb).abs() < LOGPROB_TOLERANCE, "{la} vs {lb}");
        }
    }
    Ok(())
}
