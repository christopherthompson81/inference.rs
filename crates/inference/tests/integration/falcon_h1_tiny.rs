//! Falcon-H1 on Granite's parallel hybrid blocks: a checkpoint's muP multipliers, applied on the activations, score as
//! the same weights with them folded in (as llama.cpp's converter writes a GGUF), and a block with the gated Mamba norm
//! decodes a cached repeat as it decoded the first time.

use std::collections::HashMap;

use inference::{
    LogitsOutput, Model, ModelDType, PromptInput, PromptLogitsRequest, RequestBuilder,
    TextMessageRole, TextMessages, TextModelBuilder,
};
use inference_tensor::{DType, Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
#[macro_use]
#[path = "../support/concurrent_cases.rs"]
mod concurrent_cases;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llama_tiny");
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
const FIXTURE_FILES: [&str; 3] = [
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer_config.json",
];
const VOCAB: usize = 266;
const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 48;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
const MAMBA_HEADS: usize = 4;
const MAMBA_HEAD_DIM: usize = 8;
const D_SSM: usize = MAMBA_HEADS * MAMBA_HEAD_DIM;
const D_STATE: usize = 16;
const CONV: usize = 4;
const CONV_DIM: usize = D_SSM + 2 * D_STATE;
const PROJECTION: usize = D_SSM + CONV_DIM + MAMBA_HEADS;
const WEIGHT_STD: f32 = 0.2;
const WEIGHT_SEED: u64 = 11;
// long enough that the Mamba state carries across tokens
const PROMPT: [u32; 12] = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51];
const MAX_LEN: usize = 8;
const LOGIT_TOLERANCE: f32 = 1e-3;
// past the hybrid cache's growth step beyond a chat prompt
const MAX_POSITIONS: usize = 4096;

// non-trivial everywhere, as the released checkpoints' are
const EMBEDDING: f32 = 5.6;
const LM_HEAD: f32 = 0.04;
const ATTENTION_IN: f32 = 1.3;
const ATTENTION_OUT: f32 = 0.9;
const KEY: f32 = 0.4;
const SSM_IN: f32 = 1.25;
const SSM_OUT: f32 = 0.24;
const SSM: [f32; 5] = [0.35, 0.25, 0.4, 0.5, 0.3];
const MLP: [f32; 2] = [0.88, 0.59];

fn config(gated_norm: bool, mup: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "architectures": ["FalconH1ForCausalLM"],
        "model_type": "falcon_h1",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1e11,
        "max_position_embeddings": MAX_POSITIONS,
        "tie_word_embeddings": false,
        "mamba_d_ssm": D_SSM,
        "mamba_n_heads": MAMBA_HEADS,
        "mamba_d_head": MAMBA_HEAD_DIM,
        "mamba_n_groups": 1,
        "mamba_d_state": D_STATE,
        "mamba_d_conv": CONV,
        "mamba_rms_norm": gated_norm,
        "mamba_norm_before_gate": false,
    });
    if mup {
        for (key, value) in [
            ("embedding_multiplier", EMBEDDING),
            ("lm_head_multiplier", LM_HEAD),
            ("attention_in_multiplier", ATTENTION_IN),
            ("attention_out_multiplier", ATTENTION_OUT),
            ("key_multiplier", KEY),
            ("ssm_in_multiplier", SSM_IN),
            ("ssm_out_multiplier", SSM_OUT),
        ] {
            config[key] = value.into();
        }
        config["ssm_multipliers"] = serde_json::json!(SSM);
        config["mlp_multipliers"] = serde_json::json!(MLP);
    }
    config
}

fn weights(gated_norm: bool) -> anyhow::Result<HashMap<String, Tensor>> {
    let rng = &mut StdRng::seed_from_u64(WEIGHT_SEED);
    let normal = Normal::new(0f32, WEIGHT_STD)?;
    let mut random = |shape: &[usize]| {
        let data = (0..shape.iter().product::<usize>())
            .map(|_| normal.sample(rng))
            .collect::<Vec<_>>();
        Tensor::from_vec(data, shape, &Device::Cpu)
    };
    let ones = |n: usize| Tensor::ones(n, DType::F32, &Device::Cpu);
    let mut weights = HashMap::from([
        (
            "model.embed_tokens.weight".to_string(),
            random(&[VOCAB, HIDDEN])?,
        ),
        ("model.final_layernorm.weight".to_string(), ones(HIDDEN)?),
        ("lm_head.weight".to_string(), random(&[VOCAB, HIDDEN])?),
    ]);
    for layer in 0..LAYERS {
        let p = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "pre_ff_layernorm"] {
            weights.insert(format!("{p}.{norm}.weight"), ones(HIDDEN)?);
        }
        for (name, rows, cols) in [
            ("self_attn.q_proj", HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.o_proj", HIDDEN, HEADS * HEAD_DIM),
            ("feed_forward.gate_proj", INTERMEDIATE, HIDDEN),
            ("feed_forward.up_proj", INTERMEDIATE, HIDDEN),
            ("feed_forward.down_proj", HIDDEN, INTERMEDIATE),
            ("mamba.in_proj", PROJECTION, HIDDEN),
            ("mamba.out_proj", HIDDEN, D_SSM),
        ] {
            weights.insert(format!("{p}.{name}.weight"), random(&[rows, cols])?);
        }
        weights.insert(
            format!("{p}.mamba.conv1d.weight"),
            random(&[CONV_DIM, 1, CONV])?,
        );
        weights.insert(format!("{p}.mamba.conv1d.bias"), random(&[CONV_DIM])?);
        weights.insert(format!("{p}.mamba.dt_bias"), random(&[MAMBA_HEADS])?);
        weights.insert(format!("{p}.mamba.A_log"), random(&[MAMBA_HEADS])?);
        weights.insert(format!("{p}.mamba.D"), ones(MAMBA_HEADS)?);
        if gated_norm {
            weights.insert(format!("{p}.mamba.norm.weight"), ones(D_SSM)?);
        }
    }
    Ok(weights)
}

// the multipliers into the weights, as llama.cpp's converter does
fn folded(weights: &HashMap<String, Tensor>) -> anyhow::Result<HashMap<String, Tensor>> {
    let in_proj_rows = [
        (SSM[0], D_SSM),
        (SSM[1], D_SSM),
        (SSM[2], D_STATE),
        (SSM[3], D_STATE),
        (SSM[4], MAMBA_HEADS),
    ]
    .into_iter()
    .flat_map(|(m, n)| std::iter::repeat_n(m * SSM_IN, n))
    .collect::<Vec<_>>();
    let in_proj_rows = Tensor::from_vec(in_proj_rows, (PROJECTION, 1), &Device::Cpu)?;
    weights
        .iter()
        .map(|(name, w)| {
            let scale = if name.ends_with("embed_tokens.weight") {
                EMBEDDING
            } else if name == "lm_head.weight" {
                LM_HEAD
            } else if name.ends_with("k_proj.weight") {
                ATTENTION_IN * KEY
            } else if name.ends_with("q_proj.weight") || name.ends_with("v_proj.weight") {
                ATTENTION_IN
            } else if name.ends_with("o_proj.weight") {
                ATTENTION_OUT
            } else if name.ends_with("gate_proj.weight") {
                MLP[0]
            } else if name.ends_with("down_proj.weight") {
                MLP[1]
            } else if name.ends_with("out_proj.weight") {
                SSM_OUT
            } else if name.ends_with("in_proj.weight") {
                return Ok((name.clone(), w.broadcast_mul(&in_proj_rows)?));
            } else {
                1.0
            };
            Ok((name.clone(), (w * f64::from(scale))?))
        })
        .collect()
}

fn checkpoint(
    config: serde_json::Value,
    weights: &HashMap<String, Tensor>,
) -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for file in FIXTURE_FILES {
        std::fs::copy(format!("{FIXTURE}/{file}"), dir.path().join(file))?;
    }
    std::fs::copy(TOKENIZER, dir.path().join("tokenizer.json"))?;
    std::fs::write(dir.path().join("config.json"), config.to_string())?;
    inference_tensor::safetensors::save(weights, dir.path().join("model.safetensors"))?;
    Ok(dir)
}

async fn load(dir: &tempfile::TempDir) -> anyhow::Result<Model> {
    Ok(TextModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?)
}

async fn prompt_logits(dir: &tempfile::TempDir) -> anyhow::Result<Vec<f32>> {
    let scored = load(dir)
        .await?
        .prompt_logits(PromptLogitsRequest {
            model: None,
            prompt: PromptInput::Tokens(PROMPT.to_vec()),
            output: LogitsOutput::Logits,
            audio: Vec::new(),
        })
        .await?;
    Ok(scored.logits.expect("the request asked for logits"))
}

async fn mup_multipliers_score_as_the_weights_with_them_folded_in() -> anyhow::Result<()> {
    for gated_norm in [false, true] {
        let weights = weights(gated_norm)?;
        let mup = checkpoint(config(gated_norm, true), &weights)?;
        let folded = checkpoint(config(gated_norm, false), &folded(&weights)?)?;
        let (mup, folded) = (prompt_logits(&mup).await?, prompt_logits(&folded).await?);
        assert_eq!(mup.len(), folded.len());
        let worst = mup
            .iter()
            .zip(&folded)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = mup.iter().map(|v| v.abs()).fold(0f32, f32::max);
        assert!(
            worst < LOGIT_TOLERANCE * scale.max(1.0),
            "gated norm {gated_norm}: logits differ by {worst} at scale {scale}"
        );
    }
    Ok(())
}

async fn a_gated_norm_block_decodes_a_cached_repeat_the_same() -> anyhow::Result<()> {
    let dir = checkpoint(config(true, true), &weights(true)?)?;
    let model = load(&dir).await?;
    let request = || {
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, "hello there"))
            .set_sampler_max_len(MAX_LEN)
            .set_sampler_topk(1)
    };
    let first = model.send_chat_request(request()).await?;
    let second = model.send_chat_request(request()).await?;
    let text = |r: &inference::ChatCompletionResponse| {
        r.choices[0].message.content.clone().unwrap_or_default()
    };
    anyhow::ensure!(
        first.usage.completion_tokens > 0,
        "the model generated nothing"
    );
    assert_eq!(text(&first), text(&second));
    Ok(())
}

concurrent_cases!(
    concurrent_cases,
    [
        mup_multipliers_score_as_the_weights_with_them_folded_in: 0,
        a_gated_norm_block_decodes_a_cached_repeat_the_same: 0,
    ]
);
