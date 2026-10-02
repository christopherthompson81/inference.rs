//! Compute perplexity of a text file using a loaded model.
//!
//! Run with: `cargo run --release --example perplexity -p inference-examples`

use std::{fs::read_to_string, path::PathBuf, time::Instant};

use anyhow::Result;
use clap::Parser;
use inference::{
    LogitsOutput, ModelBuilder, PromptInput, PromptLogitsRequest, api::operations::TokenizeRequest,
};

const PROMPT_CHUNKSIZE: usize = 1024;

/// Calculate perplexity of a model. By default, this uses the Llama 3.1 8B model.
#[derive(Parser)]
struct Args {
    /// The model to run.
    #[arg(short, long, default_value = "google/gemma-4-E4B-it")]
    model_id: String,

    /// Filename to text to run the model on. This is recommended to be the Wikitext 2 dataset:
    /// https://huggingface.co/datasets/EricB/wikitext2
    #[arg(short, long)]
    file: String,

    /// ISQ quantization to run with (`4`, `q4k`, ...).
    #[arg(short, long)]
    isq: Option<String>,

    /// Generate and utilize an imatrix to enhance GGUF quantizations.
    #[arg(short, long)]
    calibration_file: Option<PathBuf>,
}

fn tokenize(text: String, add_special_tokens: bool) -> TokenizeRequest {
    TokenizeRequest {
        model: None,
        text,
        add_special_tokens,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let mut model_builder = ModelBuilder::new(&args.model_id).with_logging();
    if let Some(isq) = &args.isq {
        let isq = inference::parse_isq_value(isq, None).map_err(anyhow::Error::msg)?;
        model_builder = model_builder.with_isq(isq);
    }
    if let Some(calibration_file) = &args.calibration_file {
        model_builder = model_builder.with_calibration_file(calibration_file.clone());
    }

    let model = model_builder.build().await?;

    let text = read_to_string(&args.file)?;
    let tokens = model.tokenize(tokenize(text, false)).await?.tokens;
    let bos_token = model
        .tokenize(tokenize(" ".to_string(), true))
        .await?
        .tokens[0];

    println!("Using bos token id `{bos_token}`.");

    let n_chunks = tokens.len().div_ceil(PROMPT_CHUNKSIZE);
    let mut ppl_measurements = Vec::new();
    for (i, chunk) in tokens.chunks(PROMPT_CHUNKSIZE).enumerate() {
        let start = Instant::now();
        let request = PromptLogitsRequest {
            model: None,
            prompt: PromptInput::Tokens([vec![bos_token], chunk.to_vec()].concat()),
            output: LogitsOutput::Logprobs,
        };
        let scored = model.prompt_logits(request).await?;

        // The first token has no prediction; the rest give the mean negative log-likelihood.
        let logprobs: Vec<f32> = scored.token_logprobs.iter().flatten().copied().collect();
        let nll = -logprobs.iter().sum::<f32>() / logprobs.len() as f32;
        let perplexity = nll.exp();
        let end = Instant::now();

        ppl_measurements.push(perplexity);
        println!(
            "Chunk {i}/{n_chunks} ({} tokens): Perplexity for `{}`, ISQ `{:?}`, {}s: {perplexity}",
            scored.tokens.len(),
            args.file,
            args.isq,
            end.duration_since(start).as_secs_f32(),
        );
    }

    let mean = ppl_measurements.iter().sum::<f32>() / ppl_measurements.len() as f32;
    let variance = ppl_measurements
        .iter()
        .map(|e| (mean - e).powf(2.))
        .sum::<f32>()
        / ppl_measurements.len() as f32;
    let std_dev = variance.sqrt();
    println!();
    println!(
        "Final perplexity for `{}`, ISQ `{:?}`: {} +/- {} ppl",
        args.file, args.isq, mean, std_dev
    );

    Ok(())
}
