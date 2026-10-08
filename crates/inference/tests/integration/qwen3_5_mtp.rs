//! Built-in MTP drafting on a real Qwen3.5 checkpoint; skips unless `INFERENCE_TEST_QWEN3_5_MODEL` is a local dir and
//! a GPU is present (the MTP head reads and writes paged KV).

use std::path::Path;

use inference::{
    GgufModelBuilder, Model, ModelDType, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
    TextModelBuilder,
};

const MODEL_ENV: &str = "INFERENCE_TEST_QWEN3_5_MODEL";
// A Qwen3.5 or Qwen3.8 GGUF that keeps its `nextn` (MTP) blocks, as llama.cpp's converter writes by default.
const GGUF_ENV: &str = "INFERENCE_TEST_QWEN3_5_GGUF";
// The same for a Qwen3.5-MoE GGUF, which loads through the MoE text model
const MOE_GGUF_ENV: &str = "INFERENCE_TEST_QWEN3_5_MOE_GGUF";
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));
const MAX_LEN: usize = 64;
const N_PREDICT: usize = 2;
// Plain prose a trained head predicts well (no reasoning preamble), with the steps each runs before its first near tie.
const PROMPTS: &[(&str, usize)] = &[
    (
        "Count from one to twenty in words, separated by commas.",
        16,
    ),
    (
        "Write the first four lines of a nursery rhyme about a star.",
        5,
    ),
];
// Real heads land most of their drafts on prose this predictable; a broken head or verifier lands next to none.
const MIN_ACCEPT_RATE: f64 = 0.3;
// The two paths move logprobs by up to ~0.12 on Qwen3.5-0.8B (BF16 and Q8_0); a closer top two can swap.
const TIE_MARGIN: f32 = 0.25;

type Step = (u32, f32, u32, f32);
#[cfg(unix)]
const TEXT_VIEW_FILES: [&str; 4] = [
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
];

fn model_dir() -> Option<String> {
    std::env::var(MODEL_ENV)
        .ok()
        .filter(|d| Path::new(d).exists())
}

fn paged() -> anyhow::Result<inference::PagedCacheSpec> {
    Ok(inference::PagedAttentionMetaBuilder::default().build()?)
}

async fn build(dir: &str, mtp: bool) -> anyhow::Result<Model> {
    let builder = MultimodalModelBuilder::new(dir)
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(paged()?);
    let builder = if mtp {
        builder.with_builtin_mtp(Some(N_PREDICT))
    } else {
        builder
    };
    Ok(builder.build().await?)
}

#[cfg(unix)]
async fn build_text(dir: &Path, mtp: bool) -> anyhow::Result<Model> {
    let builder = TextModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(paged()?);
    let builder = if mtp {
        builder.with_builtin_mtp(Some(N_PREDICT))
    } else {
        builder
    };
    Ok(builder.build().await?)
}

// The text backbone alone, as a text-only Qwen3.5 checkpoint: its text config, the same weights and tokenizer.
#[cfg(unix)]
fn text_only_view(dir: &str) -> anyhow::Result<tempfile::TempDir> {
    let view = tempfile::tempdir()?;
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        Path::new(dir).join("config.json"),
    )?)?;
    let mut text = config["text_config"].clone();
    text["architectures"] = serde_json::json!(["Qwen3_5ForCausalLM"]);
    text["model_type"] = "qwen3_5_text".into();
    text["tie_word_embeddings"] = config["tie_word_embeddings"].clone();
    std::fs::write(view.path().join("config.json"), text.to_string())?;
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path.file_name().unwrap_or_default();
        let keep = path.extension().is_some_and(|ext| ext == "safetensors")
            || TEXT_VIEW_FILES.map(std::ffi::OsStr::new).contains(&name);
        if keep {
            std::os::unix::fs::symlink(&path, view.path().join(name))?;
        }
    }
    Ok(view)
}

// Each step's greedy id and logprob, then the runner-up's id and logprob.
async fn greedy_trace(model: &Model, prompt: &str) -> anyhow::Result<Vec<Step>> {
    let request = RequestBuilder::new()
        .add_message(TextMessageRole::User, prompt)
        .enable_thinking(false)
        .set_sampler_max_len(MAX_LEN)
        .set_sampler_topk(1)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(2);
    let response = model.send_chat_request(request).await?;
    let steps = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| {
                    let (runner_up, runner_up_logprob) = t
                        .top_logprobs
                        .get(1)
                        .map_or((u32::MAX, f32::NEG_INFINITY), |r| (r.token, r.logprob));
                    (
                        t.top_logprobs[0].token,
                        t.top_logprobs[0].logprob,
                        runner_up,
                        runner_up_logprob,
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(steps)
}

async fn traces(model: &Model) -> anyhow::Result<Vec<Vec<Step>>> {
    let mut traces = Vec::new();
    for (prompt, _) in PROMPTS {
        traces.push(greedy_trace(model, prompt).await?);
    }
    Ok(traces)
}

async fn check_mtp(plain: &Model, mtp: &Model) -> anyhow::Result<()> {
    check_against(&traces(plain).await?, mtp).await
}

// The verify kernels round differently from decode, so drafting may only swap a near-tied top two.
async fn check_against(expected: &[Vec<Step>], mtp: &Model) -> anyhow::Result<()> {
    for (&(prompt, min_agreed), expected) in PROMPTS.iter().zip(expected) {
        let drafted = greedy_trace(mtp, prompt).await?;
        let agreed = expected
            .iter()
            .zip(&drafted)
            .take_while(|(e, d)| e.0 == d.0)
            .count();
        eprintln!("{prompt:?}: {agreed} of {} ids agree", expected.len());
        // either trace may hold the near tie: the other's pick is its runner-up, close behind its top
        let parted_at_a_tie = match (expected.get(agreed), drafted.get(agreed)) {
            (
                Some(&(kept, top, runner_up, runner_up_logprob)),
                Some(&(swapped, d_top, d_runner_up, d_runner_up_logprob)),
            ) => {
                (swapped == runner_up && top - runner_up_logprob < TIE_MARGIN)
                    || (kept == d_runner_up && d_top - d_runner_up_logprob < TIE_MARGIN)
            }
            (None, None) => true,
            _ => false,
        };
        anyhow::ensure!(
            parted_at_a_tie && agreed >= min_agreed,
            "MTP drafting changed the greedy output of {prompt:?} at step {agreed}: {drafted:?} vs {expected:?}"
        );
    }
    let stats = mtp.speculative_stats()?;
    let (proposed, accepted) = stats.data.iter().fold((0, 0), |(p, a), m| {
        (p + m.draft_tokens_proposed, a + m.draft_tokens_accepted)
    });
    let rate = accepted as f64 / proposed.max(1) as f64;
    eprintln!("MTP accepted {accepted} of {proposed} draft tokens ({rate:.2}): {stats:?}");
    anyhow::ensure!(
        rate >= MIN_ACCEPT_RATE,
        "MTP accepted {accepted} of {proposed} draft tokens"
    );
    Ok(())
}

fn gpu_model_dir() -> Option<String> {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP: {MODEL_ENV} is not a local checkpoint dir");
        return None;
    };
    if !ON_GPU {
        eprintln!("SKIP: built-in MTP needs a GPU build");
        return None;
    }
    Some(dir)
}

#[tokio::test]
async fn builtin_mtp_accepts_drafts_and_keeps_greedy_output() -> anyhow::Result<()> {
    let Some(dir) = gpu_model_dir() else {
        return Ok(());
    };
    check_mtp(&build(&dir, false).await?, &build(&dir, true).await?).await
}

#[cfg(unix)]
#[tokio::test]
async fn text_only_builtin_mtp_accepts_drafts_and_keeps_greedy_output() -> anyhow::Result<()> {
    let Some(dir) = gpu_model_dir() else {
        return Ok(());
    };
    let view = text_only_view(&dir)?;
    check_mtp(
        &build_text(view.path(), false).await?,
        &build_text(view.path(), true).await?,
    )
    .await
}

async fn build_gguf(file: &Path, mtp: bool) -> anyhow::Result<Model> {
    let dir = file.parent().unwrap_or(Path::new("."));
    let name = file.file_name().unwrap_or_default().to_string_lossy();
    let builder =
        GgufModelBuilder::new(dir.to_string_lossy(), vec![name]).with_paged_attn(paged()?);
    let builder = if mtp {
        builder.with_builtin_mtp(Some(N_PREDICT))
    } else {
        builder
    };
    Ok(builder.build().await?)
}

#[tokio::test]
async fn gguf_builtin_mtp_accepts_drafts_and_keeps_greedy_output() -> anyhow::Result<()> {
    check_gguf_mtp(GGUF_ENV).await
}

#[tokio::test]
async fn moe_gguf_builtin_mtp_accepts_drafts_and_keeps_greedy_output() -> anyhow::Result<()> {
    check_gguf_mtp(MOE_GGUF_ENV).await
}

async fn check_gguf_mtp(env: &str) -> anyhow::Result<()> {
    let Some(file) = std::env::var(env).ok().filter(|f| Path::new(f).is_file()) else {
        eprintln!("SKIP: {env} is not a local GGUF file");
        return Ok(());
    };
    if !ON_GPU {
        eprintln!("SKIP: built-in MTP needs a GPU build");
        return Ok(());
    }
    let file = Path::new(&file);
    // one model at a time: a 27B quant leaves no room for a second copy on one card
    let plain = build_gguf(file, false).await?;
    let expected = traces(&plain).await?;
    drop(plain);
    let mtp = build_gguf(file, true).await?;
    check_against(&expected, &mtp).await
}
