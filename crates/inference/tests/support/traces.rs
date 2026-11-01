//! Greedy decode traces and UQFF listings shared by the tiny-checkpoint tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use inference::{IsqType, Model, MultimodalModelBuilder, RequestBuilder};

const UQFF_EXTENSION: &str = "uqff";
const UQFF_FILE: &str = "model.uqff";
const WEIGHTS_EXTENSION: &str = "safetensors";

/// The greedy (token, logprob) per step.
pub type Trace = Vec<(u32, f32)>;

/// `request` decoded greedily for up to `max_len` steps: its trace and the prompt tokens served from the prefix cache.
pub async fn greedy(
    model: &Model,
    request: RequestBuilder,
    max_len: usize,
) -> anyhow::Result<(Trace, usize)> {
    let request = request
        .set_sampler_max_len(max_len)
        .set_sampler_topk(1)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(1);
    let response = model.send_chat_request(request).await?;
    let trace: Trace = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| (t.top_logprobs[0].token, t.top_logprobs[0].logprob))
                .collect()
        })
        .unwrap_or_default();
    anyhow::ensure!(!trace.is_empty(), "the model generated nothing");
    let cached = response
        .usage
        .prompt_tokens_details
        .as_ref()
        .map_or(0, |details| details.cached_tokens);
    Ok((trace, cached))
}

/// The same ids, each step's logprob within `tolerance`.
pub fn close(a: &[(u32, f32)], b: &[(u32, f32)], tolerance: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < tolerance)
}

/// The `.uqff` files written into `dir`, sorted.
pub fn uqff_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|path| path.extension().is_some_and(|ext| ext == UQFF_EXTENSION));
    files.sort();
    Ok(files)
}

/// A Q8_0 in-situ load of `checkpoint` that writes UQFF, then a load of the written files alone (as a UQFF repo
/// publishes them, the checkpoint's weights left behind): the greedy trace of `request` from each.
pub async fn isq_and_uqff_reload_traces(
    checkpoint: &Path,
    builder: impl Fn(&Path) -> MultimodalModelBuilder,
    request: impl Fn() -> RequestBuilder,
    max_len: usize,
) -> anyhow::Result<(Trace, Trace)> {
    let uqff_dir = tempfile::tempdir()?;
    let quantized = builder(checkpoint)
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join(UQFF_FILE))
        .build()
        .await?;
    let (written, _) = greedy(&quantized, request(), max_len).await?;
    drop(quantized);
    let files = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(!files.is_empty(), "no UQFF written");
    for entry in std::fs::read_dir(checkpoint)? {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != WEIGHTS_EXTENSION) {
            std::fs::copy(&path, uqff_dir.path().join(path.file_name().unwrap()))?;
        }
    }
    let reloaded = builder(uqff_dir.path())
        .from_uqff(vec![files[0].clone()])
        .build()
        .await?;
    let (read, _) = greedy(&reloaded, request(), max_len).await?;
    Ok((written, read))
}
