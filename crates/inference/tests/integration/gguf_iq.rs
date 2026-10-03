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

async fn build(file: &Path) -> anyhow::Result<Model> {
    let dir = file.parent().unwrap_or(Path::new("."));
    let name = file.file_name().unwrap_or_default().to_string_lossy();
    Ok(GgufModelBuilder::new(dir.to_string_lossy(), vec![name])
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

// A directory of GGUFs (one per type, named `*-<TYPE>.gguf` as llama-quantize suggests) and the matching
// `llama-perplexity`: each file's perplexity must match the reference implementation's.
struct ParitySuite {
    dir_env: &'static str,
    perplexity_env: &'static str,
    types: &'static [&'static str],
}

const IQ_SUITE: ParitySuite = ParitySuite {
    dir_env: "INFERENCE_TEST_IQ_GGUF_DIR",
    perplexity_env: "INFERENCE_TEST_LLAMA_PERPLEXITY",
    types: &[
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S",
    ],
};
// ik_llama.cpp's trellis types, checked against its own llama-perplexity
const KT_SUITE: ParitySuite = ParitySuite {
    dir_env: "INFERENCE_TEST_KT_GGUF_DIR",
    perplexity_env: "INFERENCE_TEST_IK_LLAMA_PERPLEXITY",
    types: &["IQ1_KT", "IQ2_KT", "IQ3_KT", "IQ4_KT"],
};
// Any text past two windows works; both sides read the same file.
const PERPLEXITY_TEXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md");
// llama-perplexity's `-c 512 --chunks 1`: one 512-token window scored on its second half
const PERPLEXITY_WINDOW: usize = 512;
// Our kernels match the dequantized weights; what drifts is rounding elsewhere, about 1.2% on IQ1_S.
const PERPLEXITY_TOLERANCE: f64 = 0.02;

fn llama_cpp_perplexity(llama_perplexity: &str, file: &Path) -> anyhow::Result<f64> {
    // ik's build writes llama.log into its working directory
    let scratch = tempfile::tempdir()?;
    let output = std::process::Command::new(llama_perplexity)
        .current_dir(scratch.path())
        .arg("-m")
        .arg(file)
        .args([
            "-f",
            PERPLEXITY_TEXT,
            "-c",
            &PERPLEXITY_WINDOW.to_string(),
            "--chunks",
            "1",
            "-ngl",
            "99",
        ])
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "llama-perplexity failed on {}",
        file.display()
    );
    // mainline logs `Final estimate: PPL = x` to stderr, ik `... PPL over 1 chunks for n_ctx=512 = x` to stdout
    let log = String::from_utf8_lossy(&output.stderr) + String::from_utf8_lossy(&output.stdout);
    let value = log
        .lines()
        .find_map(|line| line.split_once("Final estimate: PPL").map(|(_, rest)| rest))
        .and_then(|line| line.rsplit_once(" = "))
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .ok_or_else(|| anyhow::anyhow!("no perplexity in llama-perplexity's output"))?;
    Ok(value.parse()?)
}

// The tokens llama-perplexity scores: those after the window's first half, each given everything before it.
async fn perplexity(model: &Model, text: &str) -> anyhow::Result<f64> {
    let request = serde_json::json!({ "text": text, "add_special_tokens": true });
    let tokenized: serde_json::Value =
        serde_json::from_str(&model.tokenize_json(request.to_string().as_bytes()).await?)?;
    let tokens = tokenized["tokens"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("tokenize returned no tokens"))?;
    // llama-perplexity refuses texts shorter than two windows
    anyhow::ensure!(
        tokens.len() >= 2 * PERPLEXITY_WINDOW,
        "the text is shorter than two windows"
    );
    let request = serde_json::json!({ "prompt": tokens[..PERPLEXITY_WINDOW] });
    let (scored, _) = model
        .prompt_logits_json(request.to_string().as_bytes())
        .await?;
    let scored: serde_json::Value = serde_json::from_str(&scored)?;
    let logprobs = scored["token_logprobs"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("scoring returned no logprobs"))?
        [PERPLEXITY_WINDOW / 2 + 1..]
        .iter()
        .map(|logprob| {
            logprob
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("a scored token has no logprob"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((-logprobs.iter().sum::<f64>() / logprobs.len() as f64).exp())
}

#[tokio::test]
async fn every_iq_gguf_matches_llama_cpp_perplexity() -> anyhow::Result<()> {
    matches_reference_perplexity(&IQ_SUITE).await
}

#[tokio::test]
async fn every_trellis_gguf_matches_ik_llama_cpp_perplexity() -> anyhow::Result<()> {
    matches_reference_perplexity(&KT_SUITE).await
}

async fn matches_reference_perplexity(suite: &ParitySuite) -> anyhow::Result<()> {
    let (Some(dir), Some(llama_perplexity)) = (
        std::env::var(suite.dir_env)
            .ok()
            .filter(|d| Path::new(d).is_dir()),
        std::env::var(suite.perplexity_env)
            .ok()
            .filter(|f| Path::new(f).is_file()),
    ) else {
        eprintln!(
            "SKIP: {} and {} must name a GGUF directory and llama-perplexity",
            suite.dir_env, suite.perplexity_env
        );
        return Ok(());
    };
    if !ON_CUDA {
        eprintln!("SKIP: these kernels need a CUDA build");
        return Ok(());
    }
    let text = std::fs::read_to_string(PERPLEXITY_TEXT)?;
    let mut files = std::fs::read_dir(&dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|path| path.extension().is_some_and(|ext| ext == "gguf"));
    files.sort();
    let missing: Vec<_> = suite
        .types
        .iter()
        .filter(|ty| {
            !files.iter().any(|f| {
                f.file_stem()
                    .is_some_and(|s| s.to_string_lossy().ends_with(*ty))
            })
        })
        .collect();
    anyhow::ensure!(missing.is_empty(), "{dir} has no GGUF for {missing:?}");
    let mut mismatches = Vec::new();
    for file in &files {
        let expected = llama_cpp_perplexity(&llama_perplexity, file)?;
        let actual = perplexity(&build(file).await?, &text).await?;
        let drift = (actual / expected - 1.0).abs();
        eprintln!(
            "{}: ours {actual:.4}, reference {expected:.4}",
            file.display()
        );
        if drift > PERPLEXITY_TOLERANCE {
            mismatches.push(format!(
                "{}: ours {actual:.4}, reference {expected:.4}",
                file.display()
            ));
        }
    }
    anyhow::ensure!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    Ok(())
}
