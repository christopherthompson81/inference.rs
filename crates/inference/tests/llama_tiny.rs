//! The text-model load paths on a tiny random-weight Llama: plain, in-situ quantized, and reloaded from the UQFF it wrote.

use std::path::{Path, PathBuf};

use inference::{
    IsqType, Model, ModelDType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
    UqffTextModelBuilder,
};

#[path = "support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;

const PROMPT: &str = "hello";
const MAX_LEN: usize = 8;
const UQFF_EXTENSION: &str = "uqff";

fn cpu_text_builder(dir: &Path) -> TextModelBuilder {
    TextModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

async fn greedy_ids(model: &Model) -> anyhow::Result<Vec<u32>> {
    let request =
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT))
            .set_sampler_max_len(MAX_LEN)
            .set_sampler_topk(1)
            .return_logprobs(true)
            .set_sampler_topn_logprobs(1);
    let response = model.send_chat_request(request).await?;
    let tokens = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| t.top_logprobs[0].token)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    anyhow::ensure!(!tokens.is_empty(), "the model generated nothing");
    Ok(tokens)
}

fn uqff_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|path| path.extension().is_some_and(|ext| ext == UQFF_EXTENSION));
    files.sort();
    Ok(files)
}

#[tokio::test]
async fn a_text_checkpoint_loads_and_decodes_deterministically() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let model = cpu_text_builder(checkpoint.path()).build().await?;
    assert_eq!(greedy_ids(&model).await?, greedy_ids(&model).await?);
    Ok(())
}

#[tokio::test]
async fn an_isq_load_and_a_reload_of_its_uqff_decode_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_text_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let quantized_ids = greedy_ids(&quantized).await?;
    drop(quantized);

    let written = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(
        !written.is_empty(),
        "no UQFF written to {}",
        uqff_dir.path().display()
    );
    let reloaded = UqffTextModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        vec![written[0].clone()],
    )
    .into_inner()
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;
    assert_eq!(greedy_ids(&reloaded).await?, quantized_ids);
    Ok(())
}

#[tokio::test]
async fn an_isq_load_can_calibrate_on_a_text_file_first() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let calibration = checkpoint.path().join("calibration.txt");
    std::fs::write(
        &calibration,
        "hello world. the quick brown fox jumps over the lazy dog.",
    )?;
    let model = cpu_text_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .with_calibration_file(calibration)
        .build()
        .await?;
    greedy_ids(&model).await?;
    Ok(())
}

// Well under the 10 s drop timeout, which is what a Terminate stuck behind the queue used to cost.
const PROMPT_DROP_LIMIT: std::time::Duration = std::time::Duration::from_secs(5);
const QUEUED_MAX_TOKENS: usize = 64;

#[tokio::test(flavor = "multi_thread")]
async fn dropping_an_engine_with_a_full_request_queue_stops_it_promptly() -> anyhow::Result<()> {
    use inference::{NormalRequest, Request, RequestMessage, SamplingParams};
    use tokio::sync::mpsc::{channel, error::TrySendError};

    let checkpoint = tiny_llama_checkpoint()?;
    let model = cpu_text_builder(checkpoint.path()).build().await?;
    let sender = model.inner().get_sender(None)?;
    // Held so the engine does not skip the queued requests as abandoned.
    let mut receivers = Vec::new();
    loop {
        let (tx, rx) = channel(1);
        let mut sampling = SamplingParams::deterministic();
        sampling.max_len = Some(QUEUED_MAX_TOKENS);
        let request = NormalRequest::new_simple(
            RequestMessage::Completion {
                text: PROMPT.to_string(),
                echo_prompt: false,
                best_of: None,
            },
            sampling,
            tx,
            0,
            None,
            None,
        );
        match sender.try_send(Request::Normal(Box::new(request))) {
            Ok(()) => receivers.push(rx),
            Err(TrySendError::Full(_)) => break,
            Err(TrySendError::Closed(_)) => {
                anyhow::bail!("the engine stopped while the queue filled")
            }
        }
    }
    drop(sender);

    let started = std::time::Instant::now();
    drop(model);
    let elapsed = started.elapsed();
    assert!(
        elapsed < PROMPT_DROP_LIMIT,
        "drop took {elapsed:?} with {} queued requests",
        receivers.len()
    );
    Ok(())
}
