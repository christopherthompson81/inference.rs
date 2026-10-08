//! Greedy decode traces and UQFF listings shared by the tiny-checkpoint tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use inference::{Model, RequestBuilder};

const UQFF_EXTENSION: &str = "uqff";

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
