//! Performance benchmarking command

use anyhow::Result;
use comfy_table::{Cell, Color, ContentArrangement, Table, presets::UTF8_FULL};
use inference_api::{
    Engine, EngineSpec,
    engine::RuntimeSpec,
    engine_completion::{CompletionStream, CompletionStreamEvent},
    openai::CompletionRequest,
};
use inference_core::initialize_logging;
use serde_json::json;
use std::time::{Duration, Instant};
use tracing::info;

use crate::args::{BenchRuntimeOptions, GlobalOptions, ModelType};

use super::normalize_requested_adapter;
use super::serve::{model_spec, mtp_spec, normalize_quant_flags};

/// The engine a benchmark measures: the model as `serve` would load it, run one sequence at a time.
fn bench_spec(
    model_type: &ModelType,
    runtime: &BenchRuntimeOptions,
    global: &GlobalOptions,
) -> Result<EngineSpec> {
    let (model, model_runtime) = model_spec(model_type, &runtime.matformer_selection(), global)?;

    Ok(EngineSpec {
        model: Some(model),
        model_id: None,
        runtime: RuntimeSpec {
            // One sequence, no prefix cache, and exactly gen_len tokens, so each measurement is the same work.
            max_seqs: Some(1),
            prefix_cache_n: Some(0),
            disable_eos_stop: true,
            no_kv_cache: runtime.no_kv_cache,
            mtp: mtp_spec(
                runtime.mtp,
                runtime.mtp_model.clone(),
                runtime.mtp_n_predict,
                runtime.mtp_draft_sampling,
            ),
            log: None,
            ..model_runtime
        },
        ..Default::default()
    })
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

/// Benchmark result for a single test
struct BenchResult {
    test_name: String,
    tok_per_sec: f32,
    std_dev: f32,
    latency_ms: f32,
    latency_kind: BenchLatencyKind,
}

#[derive(Clone, Copy)]
enum BenchLatencyKind {
    Ttft,
    Tpot,
}

struct BenchMeasurement {
    time_to_first_token: Duration,
    decode_duration: Duration,
    decode_intervals: usize,
}

const BENCH_TOKEN_BASE: u32 = 1000;
const BENCH_TOKEN_SPAN: u32 = 2048;
const BENCH_ITER_STRIDE: u32 = 131;
const BENCH_CASE_STRIDE: u32 = 719;
// Greedy decoding, so every iteration of a case generates the same tokens.
const GREEDY_TOP_K: usize = 1;

pub struct BenchRunConfig {
    pub prompt_lens: Vec<usize>,
    pub gen_len: usize,
    pub depths: Vec<usize>,
    pub iterations: usize,
    pub warmup: usize,
    pub adapter: Option<String>,
}

/// Extract model_id from ModelType
fn get_model_id(model_type: &ModelType) -> String {
    match model_type {
        ModelType::Auto { model, .. }
        | ModelType::Text { model, .. }
        | ModelType::Multimodal { model, .. }
        | ModelType::Diffusion { model, .. }
        | ModelType::Speech { model, .. }
        | ModelType::Embedding { model, .. } => model.model_id.clone(),
    }
}

/// Run the benchmark command
pub async fn run_bench(
    mut model_type: ModelType,
    runtime: BenchRuntimeOptions,
    global: GlobalOptions,
    config: BenchRunConfig,
) -> Result<()> {
    initialize_logging();

    let BenchRunConfig {
        prompt_lens,
        gen_len,
        depths,
        iterations,
        warmup,
        adapter: request_adapter,
    } = config;

    if prompt_lens.is_empty() {
        anyhow::bail!("--prompt-len must contain at least one value");
    }
    if depths.is_empty() {
        anyhow::bail!("--depth must contain at least one value");
    }
    if iterations == 0 {
        anyhow::bail!("--iterations must be greater than 0");
    }
    if prompt_lens.iter().all(|prompt_len| *prompt_len == 0) && gen_len <= 1 {
        anyhow::bail!("benchmark must enable at least one TTFT or decode measurement");
    }
    if gen_len > 1 && depths.contains(&0) {
        anyhow::bail!("--depth must be greater than 0 when decode metrics are enabled");
    }
    let request_adapter = normalize_requested_adapter(&model_type, request_adapter.as_deref())?;

    // Get model ID for display
    let model_id = get_model_id(&model_type);
    // Convert args and load model
    normalize_quant_flags(&mut model_type)?;
    info!("Loading model for benchmarking...");
    let spec = bench_spec(&model_type, &runtime, &global)?;
    let engine = Engine::load(spec).await?;
    if let Some(alias) = request_adapter.as_deref() {
        super::run::require_adapter(&engine, alias).await?;
    }
    let max_model_len = engine
        .models()
        .map_err(anyhow::Error::msg)?
        .data
        .into_iter()
        .next()
        .and_then(|model| model.max_model_len);
    if let Some(max_seq_len) = max_model_len {
        let longest_ttft = prompt_lens
            .iter()
            .copied()
            .map(|prompt_len| prompt_len.saturating_add(1))
            .max()
            .unwrap_or_default();
        let longest_decode = if gen_len > 1 {
            depths
                .iter()
                .copied()
                .map(|depth| depth.saturating_add(gen_len))
                .max()
                .unwrap_or_default()
        } else {
            0
        };
        let longest_request = longest_ttft.max(longest_decode);
        if longest_request > max_seq_len {
            anyhow::bail!(
                "benchmark request length {longest_request} exceeds model maximum {max_seq_len}"
            );
        }
    }
    info!("Model loaded.");

    if warmup > 0 {
        info!("Running {warmup} warmup iteration(s) per benchmark case...");
        for (prompt_idx, prompt_len) in prompt_lens.iter().copied().enumerate() {
            if prompt_len == 0 {
                continue;
            }
            for i in 0..warmup {
                let token_start = bench_token_start(i, prompt_idx, 0);
                run_single_bench(&engine, prompt_len, 1, token_start, request_adapter.clone())
                    .await?;
            }
        }
        if gen_len > 1 {
            for (depth_idx, depth) in depths.iter().copied().enumerate() {
                for i in 0..warmup {
                    let token_start = bench_token_start(i, depth_idx, prompt_lens.len());
                    run_single_bench(
                        &engine,
                        depth,
                        gen_len,
                        token_start,
                        request_adapter.clone(),
                    )
                    .await?;
                }
            }
        }
        info!("Warmup complete.");
    }

    // Run benchmarks
    info!(
        "Running {} iteration(s) with prompt lengths {:?}, {} generation tokens, decode depths {:?}...",
        iterations, prompt_lens, gen_len, depths
    );

    #[cfg(feature = "cuda")]
    let cuda_profiler_range = std::env::var_os("INFERENCE_RS_BENCH_CUDA_PROFILER_RANGE").is_some();
    #[cfg(feature = "cuda")]
    if cuda_profiler_range {
        unsafe {
            let _ = cudaProfilerStart();
        }
    }

    let mut ttft_results: Vec<(usize, Vec<(f32, f32)>)> =
        prompt_lens.iter().map(|&len| (len, Vec::new())).collect();
    let mut decode_results: Vec<(usize, Vec<(f32, f32)>)> =
        depths.iter().map(|&depth| (depth, Vec::new())).collect();

    for i in 0..iterations {
        info!("Iteration {}/{}...", i + 1, iterations);

        for (prompt_idx, (prompt_len, results)) in ttft_results.iter_mut().enumerate() {
            if *prompt_len == 0 {
                continue;
            }
            let token_start = bench_token_start(i + warmup, prompt_idx, 0);
            let measurement = run_single_bench(
                &engine,
                *prompt_len,
                1,
                token_start,
                request_adapter.clone(),
            )
            .await?;
            let ttft_seconds = measurement.time_to_first_token.as_secs_f32();
            let tok_per_sec = if ttft_seconds > 0.0 {
                *prompt_len as f32 / ttft_seconds
            } else {
                0.0
            };
            results.push((
                tok_per_sec,
                measurement.time_to_first_token.as_secs_f32() * 1000.0,
            ));
        }

        if gen_len > 1 {
            for (depth_idx, (depth, results)) in decode_results.iter_mut().enumerate() {
                let token_start = bench_token_start(i + warmup, depth_idx, prompt_lens.len());
                let measurement = run_single_bench(
                    &engine,
                    *depth,
                    gen_len,
                    token_start,
                    request_adapter.clone(),
                )
                .await?;
                let decode_seconds = measurement.decode_duration.as_secs_f32();
                let tok_per_sec = if decode_seconds > 0.0 {
                    measurement.decode_intervals as f32 / decode_seconds
                } else {
                    0.0
                };
                let ms_per_tok = if tok_per_sec > 0.0 {
                    1000.0 / tok_per_sec
                } else {
                    0.0
                };
                results.push((tok_per_sec, ms_per_tok));
            }
        }
    }

    #[cfg(feature = "cuda")]
    if cuda_profiler_range {
        unsafe {
            let _ = cudaProfilerStop();
        }
    }

    // Calculate statistics
    let mut results = Vec::new();

    for (prompt_len, ttft_result) in ttft_results {
        if ttft_result.is_empty() {
            continue;
        }
        let tok_per_sec_vals: Vec<f32> = ttft_result.iter().map(|(t, _)| *t).collect();
        let ttft_vals: Vec<f32> = ttft_result.iter().map(|(_, l)| *l).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let (mean_ttft, _) = calculate_stats(&ttft_vals);
        results.push(BenchResult {
            test_name: format!("TTFT ({} input tokens)", prompt_len),
            tok_per_sec: mean_tps,
            std_dev: std_dev_tps,
            latency_ms: mean_ttft,
            latency_kind: BenchLatencyKind::Ttft,
        });
    }

    for (depth, decode_result) in decode_results {
        if decode_result.is_empty() {
            continue;
        }
        let tok_per_sec_vals: Vec<f32> = decode_result.iter().map(|(t, _)| *t).collect();
        let tpot_vals: Vec<f32> = decode_result.iter().map(|(_, l)| *l).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let (mean_tpot, _) = calculate_stats(&tpot_vals);
        results.push(BenchResult {
            test_name: format!("Decode ({} tokens @ d{})", gen_len, depth),
            tok_per_sec: mean_tps,
            std_dev: std_dev_tps,
            latency_ms: mean_tpot,
            latency_kind: BenchLatencyKind::Tpot,
        });
    }

    // Print results
    print_results(&model_id, request_adapter.as_deref(), iterations, &results);

    Ok(())
}

/// Calculate mean and standard deviation
fn calculate_stats(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();
    (mean, std_dev)
}

fn bench_token_start(iteration: usize, case_idx: usize, group_offset: usize) -> u32 {
    ((iteration + 1) as u32 * BENCH_ITER_STRIDE
        + (case_idx + group_offset) as u32 * BENCH_CASE_STRIDE)
        % BENCH_TOKEN_SPAN
}

async fn run_single_bench(
    engine: &Engine,
    prompt_tokens: usize,
    gen_tokens: usize,
    token_start: u32,
    adapter: Option<String>,
) -> Result<BenchMeasurement> {
    measure(
        engine,
        bench_tokens(prompt_tokens, token_start),
        gen_tokens,
        adapter,
    )
    .await
}

async fn measure(
    engine: &Engine,
    prompt: Vec<u32>,
    gen_tokens: usize,
    adapter: Option<String>,
) -> Result<BenchMeasurement> {
    let request = bench_request(prompt, gen_tokens, adapter)?;
    let request_start = Instant::now();
    let stream = engine
        .completion_stream(request)
        .await
        .map_err(anyhow::Error::msg)?;
    recv_measurement(stream, request_start, gen_tokens).await
}

fn bench_request(
    prompt: Vec<u32>,
    gen_tokens: usize,
    adapter: Option<String>,
) -> Result<CompletionRequest> {
    Ok(serde_json::from_value(json!({
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "top_k": GREEDY_TOP_K,
        "adapter": adapter,
    }))?)
}

fn bench_tokens(prompt_tokens: usize, token_start: u32) -> Vec<u32> {
    (0..prompt_tokens)
        .map(|idx| BENCH_TOKEN_BASE + (token_start + idx as u32) % BENCH_TOKEN_SPAN)
        .collect()
}

async fn recv_measurement(
    mut stream: CompletionStream,
    request_start: Instant,
    expected_tokens: usize,
) -> Result<BenchMeasurement> {
    let mut first_token = None;

    let last_token = loop {
        match stream.next_event().await {
            Some(CompletionStreamEvent::Chunk(response)) => {
                let received = Instant::now();
                let finished = response
                    .choices
                    .iter()
                    .any(|choice| choice.finish_reason.is_some());
                if !response.choices.is_empty() {
                    first_token.get_or_insert(received);
                }
                if finished {
                    break received;
                }
            }
            Some(CompletionStreamEvent::Error(error)) => anyhow::bail!("{error}"),
            None => anyhow::bail!("the completion stream ended without a final chunk"),
        }
    };

    let first_token = first_token.expect("finished response must contain a token");
    Ok(BenchMeasurement {
        time_to_first_token: first_token.duration_since(request_start),
        decode_duration: last_token.duration_since(first_token),
        decode_intervals: expected_tokens.saturating_sub(1),
    })
}

/// Print benchmark results in a nice table
#[allow(clippy::cast_precision_loss)]
fn print_results(
    model_id: &str,
    adapter: Option<&str>,
    iterations: usize,
    results: &[BenchResult],
) {
    println!();
    println!("Benchmark Results");
    println!("=================");
    println!();
    println!("Model: {}", model_id);
    println!("Adapter: {}", adapter.unwrap_or("base"));
    println!("Iterations: {}", iterations);
    println!();

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Test"),
            Cell::new("T/s"),
            Cell::new("Latency"),
        ]);

    for result in results {
        let latency_str = match result.latency_kind {
            BenchLatencyKind::Ttft => format!("{:.2} ms TTFT", result.latency_ms),
            BenchLatencyKind::Tpot => format!("{:.2} ms TPOT", result.latency_ms),
        };

        table.add_row(vec![
            Cell::new(&result.test_name),
            Cell::new(format!("{:.1} ± {:.1}", result.tok_per_sec, result.std_dev))
                .fg(Color::Green),
            Cell::new(latency_str),
        ]);
    }

    println!("{table}");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROMPT_TOKENS: usize = 8;
    const GEN_TOKENS: usize = 4;

    #[tokio::test(flavor = "multi_thread")]
    async fn a_measurement_times_a_token_prompt_through_the_engine_api() -> anyhow::Result<()> {
        let dir = crate::commands::tiny_support::tiny_checkpoint()?;
        let spec = serde_json::from_value(json!({
            "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
            "runtime": {"device": "cpu", "disable_eos_stop": true},
        }))?;
        let engine = Engine::load(spec).await?;
        // The tiny vocabulary is smaller than bench's synthetic token range.
        let prompt: Vec<u32> = (1..=PROMPT_TOKENS as u32).collect();

        // Bench divides decode time by max_tokens, so the request must run to it; disable_eos_stop sees to that.
        let full = engine
            .completion(bench_request(prompt.clone(), GEN_TOKENS, None)?)
            .await
            .map_err(anyhow::Error::msg)?;
        assert_eq!(full.usage.completion_tokens, GEN_TOKENS);
        assert_eq!(full.choices[0].finish_reason, "length");

        let measurement = measure(&engine, prompt, GEN_TOKENS, None).await?;
        assert!(measurement.time_to_first_token > Duration::ZERO);
        Ok(())
    }

    #[test]
    fn a_benchmark_runs_one_sequence_to_its_full_length() {
        use clap::Parser;
        let cli = crate::args::Cli::try_parse_from([
            "inference",
            "--seed",
            "3",
            "bench",
            "-m",
            "org/model",
            "--cpu",
            "--mtp",
        ])
        .unwrap();
        let crate::args::Command::Bench {
            model_type,
            default_model,
            runtime,
            ..
        } = cli.command
        else {
            panic!("not a bench command");
        };
        let model_type = crate::args::resolve_model_type(model_type, default_model).unwrap();
        let spec = bench_spec(&model_type, &runtime, &cli.global).unwrap();
        let rt = &spec.runtime;
        assert_eq!((rt.max_seqs, rt.prefix_cache_n), (Some(1), Some(0)));
        assert!(rt.disable_eos_stop);
        assert_eq!((rt.device.as_deref(), rt.seed), (Some("cpu"), Some(3)));
        assert!(rt.mtp.is_some() && rt.log.is_none() && rt.throughput_logging.is_none());
    }
}
