//! IQ-quantized GGUF weights on real checkpoints; skips unless `INFERENCE_TEST_IQ4_XS_GGUF` is a local GGUF file and
//! a GPU build is present.

use std::path::Path;

use inference::{GgufModelBuilder, Model};

const IQ4_XS_ENV: &str = "INFERENCE_TEST_IQ4_XS_GGUF";
// The raw IQ blocks live on CUDA or the CPU; Metal has no kernels for them yet.
const ON_CUDA: bool = cfg!(feature = "cuda");
const PROMPT: &str = "The three primary colors of light are";
const MAX_TOKENS: usize = 32;
// `llama-simple -n 32` on the same Qwen3.8-27B IQ4_XS file, without the space it prints before the continuation
const LLAMA_CPP_CONTINUATION: &str = "red, green, and blue. When these colors are combined in equal intensities, they \
produce white light. This phenomenon is known as additive color mixing. In";
// GDN keeps a full recurrent state per sequence slot, so a 27B on one card has room for few
const MAX_SEQS: usize = 2;

async fn build(file: &Path) -> anyhow::Result<Model> {
    let dir = file.parent().unwrap_or(Path::new("."));
    let name = file.file_name().unwrap_or_default().to_string_lossy();
    Ok(GgufModelBuilder::new(dir.to_string_lossy(), vec![name])
        .with_max_num_seqs(MAX_SEQS)
        .build()
        .await?)
}

#[tokio::test]
async fn iq4_xs_greedy_continuation_matches_llama_cpp() -> anyhow::Result<()> {
    let Some(file) = std::env::var(IQ4_XS_ENV)
        .ok()
        .filter(|f| Path::new(f).is_file())
    else {
        eprintln!("SKIP: {IQ4_XS_ENV} is not a local GGUF file");
        return Ok(());
    };
    if !ON_CUDA {
        eprintln!("SKIP: IQ4_XS kernels need a CUDA build");
        return Ok(());
    }
    let model = build(Path::new(&file)).await?;
    let request = serde_json::json!({
        "model": "default",
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
    });
    let response: serde_json::Value = serde_json::from_str(
        &model
            .completion_json(request.to_string().as_bytes())
            .await?,
    )?;
    let text = response["choices"][0]["text"].as_str().unwrap_or_default();
    assert_eq!(text, LLAMA_CPP_CONTINUATION, "{response}");
    Ok(())
}
