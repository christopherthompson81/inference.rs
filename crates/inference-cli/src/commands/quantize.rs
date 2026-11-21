//! Quantize command implementation for UQFF generation

use std::collections::{BTreeMap, HashSet};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use tracing::{info, warn};

use inference_api::{
    Engine, EngineSpec, asr_gguf,
    engine::{
        IsqType, ModelSelected, NormalLoaderType, RuntimeSpec, UqffWriteConfig, expand_isq_value,
    },
    gguf_requantize, hidream_gguf, initialize_logging,
    uqff::{QuantPolicy, resolve_model_source},
};

use super::serve::{self, extract_device_settings};
use crate::args::{
    AdapterOptions, AsrQuantizeOptions, CacheOptions, DeviceOptions, Exl3QuantizeOptions,
    FormatOptions, GgufQuantizeOptions, GlobalOptions, HessiansOptions, HiDreamQuantizeOptions,
    ImatrixConvertOptions, KldOptions, MatformerSelection, MmprojOptions, ModelSourceOptions,
    ModelType, MultimodalAdapterOptions, MultimodalOptions, NemotronTranscribeAudioOptions,
    QuantizationOptions, QuantizeDeviceOptions, QuantizeModelSourceOptions, QuantizeModelType,
    QuantizeMultimodalOptions, QuantizeQuantizationOptions, QuantizerChoice,
    WorldgenQuantizeOptions,
};

const MIB: f64 = (1u64 << 20) as f64;
// a streamed EXL3 write holds the model on the host, its layers built on the GPU one at a time
const HOST_DEVICE: &str = "cpu";
const GGUF_EXTENSION: &str = "gguf";
const REPORT_HESSIANS_ENV: &str = "INFERENCE_GPTQ_REPORT_HESSIANS";
// the Hessian directory a scoring run names but never writes
const KLD_SCRATCH_DIR: &str = "inference-kld";

/// Extract ISQ values from the QuantizeModelType
fn get_isq_values(model_type: &QuantizeModelType) -> &[String] {
    match model_type {
        QuantizeModelType::Auto { quantization, .. } => &quantization.in_situ_quant,
        QuantizeModelType::Text { quantization, .. } => &quantization.in_situ_quant,
        QuantizeModelType::Multimodal { quantization, .. } => &quantization.in_situ_quant,
        QuantizeModelType::Embedding { quantization, .. } => &quantization.in_situ_quant,
    }
}

/// Extract the output path from the QuantizeModelType
fn get_output_path(model_type: &QuantizeModelType) -> &PathBuf {
    match model_type {
        QuantizeModelType::Auto { output, .. } => &output.output_path,
        QuantizeModelType::Text { output, .. } => &output.output_path,
        QuantizeModelType::Multimodal { output, .. } => &output.output_path,
        QuantizeModelType::Embedding { output, .. } => &output.output_path,
    }
}

/// Extract the model ID from the QuantizeModelType
fn get_model_id(model_type: &QuantizeModelType) -> &str {
    match model_type {
        QuantizeModelType::Auto { model, .. }
        | QuantizeModelType::Text { model, .. }
        | QuantizeModelType::Multimodal { model, .. } => model
            .model_id
            .as_deref()
            .expect("quantize model source was normalized"),
        QuantizeModelType::Embedding { model, .. } => &model.model_id,
    }
}

/// Extract the no_readme flag from the QuantizeModelType
fn get_no_readme(model_type: &QuantizeModelType) -> bool {
    match model_type {
        QuantizeModelType::Auto { output, .. } => output.no_readme,
        QuantizeModelType::Text { output, .. } => output.no_readme,
        QuantizeModelType::Multimodal { output, .. } => output.no_readme,
        QuantizeModelType::Embedding { output, .. } => output.no_readme,
    }
}

/// Extract the README override flags from the QuantizeModelType
fn get_readme_overrides(model_type: &QuantizeModelType) -> (Option<String>, Option<String>) {
    match model_type {
        QuantizeModelType::Auto { output, .. }
        | QuantizeModelType::Text { output, .. }
        | QuantizeModelType::Multimodal { output, .. }
        | QuantizeModelType::Embedding { output, .. } => {
            (output.uqff_base_model.clone(), output.uqff_repo_id.clone())
        }
    }
}

/// Writes a speech recogniser's `.nemo` as an `asr` GGUF.
pub fn run_asr_quantize(options: AsrQuantizeOptions, global: GlobalOptions) -> Result<()> {
    initialize_logging();
    let policy = asr_gguf::QuantPolicy::parse(&options.policy, &options.keep)?;
    let summary = asr_gguf::write_asr_gguf(
        &options.model_id,
        options.revision,
        &global.token_source,
        &options.output_path,
        &policy,
    )?;
    report_gguf(&options.output_path, &summary);
    Ok(())
}

/// Writes a HiDream-O1 checkpoint as a GGUF.
pub fn run_hidream_quantize(options: HiDreamQuantizeOptions, global: GlobalOptions) -> Result<()> {
    initialize_logging();
    let summary = hidream_gguf::write_hidream_gguf(
        &options.model_id,
        options.revision,
        &global.token_source,
        &options.output_path,
        &options.matrices,
        &options.keep,
    )?;
    report_gguf(&options.output_path, &summary);
    Ok(())
}

/// Rewrites a float GGUF with the given per-tensor types.
pub fn run_gguf_quantize(options: GgufQuantizeOptions) -> Result<()> {
    initialize_logging();
    let rules = options
        .tensor_types
        .iter()
        .map(|rule| {
            let Some((pattern, ty)) = rule.rsplit_once('=') else {
                anyhow::bail!("`--tensor-type {rule}` is not `<regex>=<type>`");
            };
            Ok((
                regex::Regex::new(pattern)?,
                gguf_requantize::writable_type(ty)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let imatrix = options
        .imatrix
        .as_deref()
        .map(gguf_requantize::load_imatrix)
        .transpose()?;
    if let Some(imatrix) = &imatrix {
        info!("Loaded an imatrix of {} tensors", imatrix.len());
    }
    let quantizer = match options.quantizer {
        QuantizerChoice::LlamaCpp => gguf_requantize::Quantizer::LlamaCpp,
        QuantizerChoice::IkLlama => gguf_requantize::Quantizer::IkLlama,
    };
    let plan = gguf_requantize::RequantizePlan {
        rules,
        imatrix,
        quantizer,
        reuse: options.reuse.clone(),
        only: options.only.as_deref().map(regex::Regex::new).transpose()?,
        gpu: (!options.cpu || options.hessians.is_some())
            .then(|| inference_tensor::Device::new_cuda(0).ok())
            .flatten()
            .inspect(|device| {
                #[cfg(feature = "cuda")]
                if let inference_tensor::Device::Cuda(dev) = device {
                    // SAFETY: every quantizer runs on the device's one stream, so no use crosses streams; the
                    // per-buffer events otherwise cost a create and destroy each (most of a stack's GPTQ setup)
                    unsafe { dev.disable_event_tracking() };
                }
                #[cfg(not(feature = "cuda"))]
                let _ = device;
            }),
        hessians: options
            .hessians
            .as_deref()
            .map(|path| -> Result<Box<dyn gguf_requantize::HessianSource>> {
                if path.is_dir() {
                    let dir = gguf_requantize::hessians_for_gguf(path, &options.model)?;
                    if !dir.unmatched.is_empty() {
                        warn!(
                            "{} Hessians bind to no tensor of the GGUF: {}",
                            dir.unmatched.len(),
                            dir.unmatched.join(", ")
                        );
                    }
                    info!(
                        "Hessians for {} tensors from `{}`",
                        dir.len(),
                        path.display()
                    );
                    Ok(Box::new(dir))
                } else {
                    Ok(Box::new(gguf_requantize::HessianFile::open(path)?))
                }
            })
            .transpose()?,
        hessian_weights_only: options.hessian_weights_only,
        gptq_damp: options.gptq_damp,
        // a diagnostic: GPTQ's report measured over another calibration's Hessians
        report_hessians: std::env::var_os(REPORT_HESSIANS_ENV)
            .map(|dir| -> Result<Box<dyn gguf_requantize::HessianSource>> {
                Ok(Box::new(gguf_requantize::hessians_for_gguf(
                    Path::new(&dir),
                    &options.model,
                )?))
            })
            .transpose()?,
        checkpoint: options
            .checkpoint
            .as_deref()
            .map(
                |dir| -> Result<Box<dyn gguf_requantize::CheckpointSource>> {
                    let experts =
                        gguf_requantize::official_checkpoint_experts(dir)?.ok_or_else(|| {
                            anyhow::anyhow!(
                                "`{}` is no checkpoint whose expert stacks are known",
                                dir.display()
                            )
                        })?;
                    Ok(Box::new(experts))
                },
            )
            .transpose()?,
    };
    let requantize = || {
        gguf_requantize::requantize_gguf(
            &options.model,
            &options.output_path,
            &plan,
            |i, n, name| info!("[{}/{n}] {name}", i + 1),
        )
    };
    // every logical CPU: the scale searches gain ~1.26x from the hyperthreads, the gemm pool's reason too
    let summary = match inference_tensor::utils::smt_gemm_pool() {
        Some(pool) => pool.install(requantize)?,
        None => requantize()?,
    };
    if !summary.without_imatrix.is_empty() {
        warn!(
            "{} quantized tensors had no imatrix entry: {}",
            summary.without_imatrix.len(),
            summary.without_imatrix.join(", ")
        );
    }
    let types = summary
        .by_type
        .iter()
        .map(|(name, count, bytes)| format!("{count} {name} ({:.1} MiB)", *bytes as f64 / MIB))
        .collect::<Vec<_>>()
        .join(", ");
    info!(
        "Wrote {} ({:.2} GiB from {:.2} GiB, {} of {} tensors quantized, {} reused): {types}",
        options.output_path.display(),
        summary.bytes_out as f64 / MIB / 1024.0,
        summary.bytes_in as f64 / MIB / 1024.0,
        summary.quantized,
        summary.tensors,
        summary.reused,
    );
    Ok(())
}

/// Converts a `.cimatrix` to llama.cpp's imatrix under a GGUF's tensor names.
/// Loading with `exl3_out` set writes the EXL3 model; the engine is shut down once it has.
pub async fn run_exl3_quantize(options: Exl3QuantizeOptions, global: GlobalOptions) -> Result<()> {
    initialize_logging();
    let model = ModelSelected::Run {
        model_id: options.model_id,
        quant: None,
        tokenizer_json: None,
        // the host's auto dtype is f32, twice what a streamed model holds in RAM
        dtype: if options.resident {
            inference_api::engine::ModelDType::Auto
        } else {
            inference_api::engine::ModelDType::BF16
        },
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff: None,
        imatrix: None,
        calibration_file: Some(options.calibration_file),
        hessians_out: None,
        exl3_out: Some(inference_api::engine::Exl3WriteConfig {
            output: options.output_path,
            bits: options.bits,
            head_bits: options.head_bits,
            codebook: options.codebook,
        }),
        max_edge: None,
        max_seq_len: inference_api::engine::AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
        max_batch_size: inference_api::engine::AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
        max_num_images: None,
        max_image_length: None,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    };
    let model = resolve_model_source(model, &global.token_source, false, QuantPolicy::GgufInput)
        .await?
        .model;
    let spec = EngineSpec {
        model: Some(model),
        model_id: None,
        runtime: RuntimeSpec {
            max_seqs: Some(1),
            prefix_cache_n: Some(0),
            paged_attn: Some(false),
            token_source: Some(global.token_source.to_string()),
            device: (!options.resident).then(|| HOST_DEVICE.to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = Engine::load(spec).await?;
    engine.shutdown().await.map_err(anyhow::Error::msg)?;
    Ok(())
}

/// Score against a KL-divergence base, or with `--text` write one, through the streamed load that collects Hessians.
pub async fn run_kld(options: KldOptions, global: GlobalOptions) -> Result<()> {
    if options.resident {
        return run_kld_resident(options, global).await;
    }
    let Some(text) = options.text else {
        // a base given as the calibration file is scored against, writing nothing
        return run_hessians(
            HessiansOptions {
                model_id: options.model_id,
                calibration_file: options.kld_base,
                output_path: std::env::temp_dir().join(KLD_SCRATCH_DIR),
            },
            global,
        )
        .await;
    };
    // a base job given as the calibration file writes the base where the Hessians would go
    let job_path =
        std::env::temp_dir().join(format!("{KLD_SCRATCH_DIR}-job-{}", std::process::id()));
    inference_api::engine_logits::KldBaseJob {
        text,
        n_ctx: options.ctx,
        chunks: options.chunks,
    }
    .write(&job_path)?;
    let written = run_hessians(
        HessiansOptions {
            model_id: options.model_id,
            calibration_file: job_path.clone(),
            output_path: options.kld_base,
        },
        global,
    )
    .await;
    let _ = std::fs::remove_file(&job_path);
    written
}

// `model_id` (a GGUF file or a model) in BF16, streamed through `calibration` (and where it writes) when given
fn bf16_model(model_id: String, streamed: Option<(PathBuf, PathBuf)>) -> ModelSelected {
    let (calibration_file, hessians_out) = streamed.unzip();
    let gguf = Path::new(&model_id);
    if gguf.is_file() && gguf.extension().is_some_and(|e| e == GGUF_EXTENSION) {
        ModelSelected::GGUF {
            tok_model_id: None,
            quantized_model_id: gguf
                .parent()
                .map_or_else(|| ".".into(), |p| p.display().to_string()),
            quantized_filename: gguf
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
            quant: None,
            tokenizer_json: None,
            mmproj_filename: None,
            mmproj_selection: inference_api::engine::MmprojSelection::Given,
            lora_adapters: Vec::new(),
            lora_runtime_config: None,
            dtype: inference_api::engine::ModelDType::BF16,
            topology: None,
            organization: None,
            write_uqff: None,
            imatrix: None,
            calibration_file: calibration_file.clone(),
            hessians_out: hessians_out.clone(),
            max_edge: None,
            max_seq_len: inference_api::engine::AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: inference_api::engine::AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: None,
            max_image_length: None,
            hf_cache_path: None,
            matformer_config_path: None,
            matformer_slice_name: None,
        }
    } else {
        ModelSelected::Run {
            model_id,
            quant: None,
            tokenizer_json: None,
            // the host's auto dtype is f32, twice what a streamed model holds in RAM
            dtype: inference_api::engine::ModelDType::BF16,
            topology: None,
            organization: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file,
            hessians_out,
            exl3_out: None,
            max_edge: None,
            max_seq_len: inference_api::engine::AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: inference_api::engine::AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: None,
            max_image_length: None,
            hf_cache_path: None,
            matformer_config_path: None,
            matformer_slice_name: None,
        }
    }
}

/// `--resident`: the model loaded whole onto its device, its prompt scoring writing the base over `--text` or scoring
/// against it, reported as the streamed scoring does.
async fn run_kld_resident(options: KldOptions, global: GlobalOptions) -> Result<()> {
    use inference_api::engine_logits::{
        KldBase, KldBaseWriter, LogitsOutput, PromptInput, PromptLogitsRequest,
    };
    use inference_api::operations::TokenizeRequest;
    initialize_logging();
    let model = resolve_model_source(
        bf16_model(options.model_id, None),
        &global.token_source,
        false,
        QuantPolicy::GgufInput,
    )
    .await?
    .model;
    let spec = EngineSpec {
        model: Some(model),
        model_id: None,
        runtime: RuntimeSpec {
            max_seqs: Some(1),
            prefix_cache_n: Some(0),
            // one sequence at a time; a paged cache would take the memory the full logits need
            paged_attn: Some(false),
            token_source: Some(global.token_source.to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = Engine::load(spec).await?;
    let tokenize = |text: String| TokenizeRequest {
        model: None,
        text,
        add_special_tokens: true,
    };
    // llama-perplexity starts each chunk with the BOS where encoding adds one
    let bos = engine
        .tokenize(tokenize(String::new()))
        .await
        .map_err(anyhow::Error::msg)?
        .tokens
        .first()
        .copied();
    let score = |tokens: Vec<u32>,
                 reference: Option<inference_api::engine_logits::LogitsReference>| {
        PromptLogitsRequest {
            model: None,
            prompt: PromptInput::Tokens(tokens),
            output: if reference.is_some() {
                LogitsOutput::Logprobs
            } else {
                LogitsOutput::Logits
            },
            audio: Vec::new(),
            reference,
        }
    };
    let chunk_of = |chunk: &[u32]| {
        let mut chunk = chunk.to_vec();
        if let Some(bos) = bos {
            chunk[0] = bos;
        }
        chunk
    };
    if let Some(text) = options.text {
        let tokens = engine
            .tokenize(tokenize(std::fs::read_to_string(&text)?))
            .await
            .map_err(anyhow::Error::msg)?
            .tokens;
        let n_chunk = options.chunks.min(tokens.len() / options.ctx);
        anyhow::ensure!(
            n_chunk > 0,
            "{} holds {} tokens, less than one chunk of {}",
            text.display(),
            tokens.len(),
            options.ctx
        );
        let tokens = &tokens[..n_chunk * options.ctx];
        let mut writer = None;
        for chunk in tokens.chunks(options.ctx) {
            let scored = engine
                .prompt_logits(score(chunk_of(chunk), None))
                .await
                .map_err(anyhow::Error::msg)?;
            let vocab = scored.vocab_size;
            let logits = scored
                .logits
                .ok_or_else(|| anyhow::anyhow!("prompt scoring returned no logits"))?;
            let writer = match &mut writer {
                Some(writer) => writer,
                None => writer.insert(KldBaseWriter::create(
                    &options.kld_base,
                    options.ctx,
                    vocab,
                    tokens,
                )?),
            };
            // the second half's rows but the last, each predicting the token after it
            let first = options.ctx / 2;
            writer.write_chunk(&logits[first * vocab..(options.ctx - 1) * vocab])?;
        }
        writer.expect("at least one chunk").finish()?;
        info!(
            "Wrote a base of {n_chunk} chunks of {} tokens to `{}`.",
            options.ctx,
            options.kld_base.display()
        );
    } else {
        let mut base = KldBase::open(&options.kld_base)?;
        let chunks = base
            .tokens
            .chunks(base.n_ctx)
            .map(chunk_of)
            .collect::<Vec<_>>();
        let (mut klds, mut same, mut nll, mut nll_base) = (Vec::new(), 0usize, 0f64, 0f64);
        for chunk in chunks {
            let reference = base.next_chunk()?;
            let first = reference.first_row;
            let scored = engine
                .prompt_logits(score(chunk, Some(reference)))
                .await
                .map_err(anyhow::Error::msg)?;
            let divergence = scored
                .divergence
                .ok_or_else(|| anyhow::anyhow!("prompt scoring returned no divergence"))?;
            let rows = divergence.kld.len();
            // token_logprobs[i] scores token i, the one row i - 1 predicts
            nll -= scored.token_logprobs[first + 1..first + 1 + rows]
                .iter()
                .map(|l| f64::from(l.unwrap_or(f32::NAN)))
                .sum::<f64>();
            nll_base -= divergence
                .reference_logprobs
                .iter()
                .map(|&l| f64::from(l))
                .sum::<f64>();
            same += divergence.same_top.iter().filter(|&&s| s).count();
            klds.extend(divergence.kld);
        }
        let n = f64::from(u32::try_from(klds.len())?);
        let mean = klds.iter().map(|&k| f64::from(k)).sum::<f64>() / n;
        klds.sort_by(f32::total_cmp);
        let pick = |hundredths: usize| klds[(klds.len() - 1) * hundredths / 100];
        info!(
            "KLD over {} positions ({} chunks of {}): mean {mean:.5}, median {:.5}, p99 {:.4}, same top {:.2}%, \
             PPL {:.4} (base {:.4})",
            klds.len(),
            base.n_chunk,
            base.n_ctx,
            pick(50),
            pick(99),
            100.0 * f64::from(u32::try_from(same)?) / n,
            (nll / n).exp(),
            (nll_base / n).exp()
        );
    }
    engine.shutdown().await.map_err(anyhow::Error::msg)?;
    Ok(())
}

pub async fn run_hessians(options: HessiansOptions, global: GlobalOptions) -> Result<()> {
    initialize_logging();
    let model = bf16_model(
        options.model_id,
        Some((options.calibration_file, options.output_path)),
    );
    let model = resolve_model_source(model, &global.token_source, false, QuantPolicy::GgufInput)
        .await?
        .model;
    let spec = EngineSpec {
        model: Some(model),
        model_id: None,
        runtime: RuntimeSpec {
            max_seqs: Some(1),
            prefix_cache_n: Some(0),
            paged_attn: Some(false),
            token_source: Some(global.token_source.to_string()),
            device: Some(HOST_DEVICE.to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = Engine::load(spec).await?;
    engine.shutdown().await.map_err(anyhow::Error::msg)?;
    Ok(())
}

pub fn run_imatrix_convert(options: ImatrixConvertOptions) -> Result<()> {
    initialize_logging();
    let mapping = gguf_requantize::cimatrix_for_gguf(&options.cimatrix, &options.gguf)?;
    if !mapping.unmatched.is_empty() {
        info!(
            "{} entries bind to no tensor of the GGUF: {}",
            mapping.unmatched.len(),
            mapping.unmatched.join(", ")
        );
    }
    if mapping.unseen_experts > 0 {
        warn!(
            "{} expert slices saw no tokens; they take llama.cpp's stand-in importance",
            mapping.unseen_experts
        );
    }
    gguf_requantize::save_legacy_imatrix(&mapping.entries, &options.dataset, &options.output_path)?;
    info!(
        "Wrote {} entries to {}",
        mapping.entries.len(),
        options.output_path.display()
    );
    Ok(())
}

/// Writes Nemotron 3.5 Transcribe's audio GGUF from its checkpoint.
pub fn run_nemotron_transcribe_audio(options: NemotronTranscribeAudioOptions) -> Result<()> {
    initialize_logging();
    let written = gguf_requantize::write_nemotron_transcribe_audio_gguf(
        &options.model,
        &options.output_path,
    )?;
    info!(
        "Wrote {} audio tensors to {} ({:.1} MiB)",
        written,
        options.output_path.display(),
        std::fs::metadata(&options.output_path)?.len() as f64 / MIB
    );
    Ok(())
}

/// Writes a checkpoint's towers as one mmproj GGUF.
pub fn run_mmproj_quantize(options: MmprojOptions) -> Result<()> {
    initialize_logging();
    let summary = gguf_requantize::write_mmproj(
        &options.model,
        &options.output_path,
        options.matrix_type.as_deref(),
    )?;
    info!(
        "Wrote {} {} projector tensors to {} ({:.1} MiB)",
        summary.tensors,
        summary.architecture,
        options.output_path.display(),
        summary.bytes as f64 / MIB
    );
    Ok(())
}

/// Lays the world-generation checkpoints out for distribution.
pub fn run_worldgen_quantize(options: WorldgenQuantizeOptions) -> Result<()> {
    initialize_logging();
    let files = inference_jobs::worldgen_package::write_worldgen_package(
        &options.models,
        &options.output_path,
        &options.matrices,
    )?;
    for file in &files {
        match &file.gguf {
            Some(summary) => report_gguf(&file.path, summary),
            None => info!(
                "Placed {} ({:.1} MiB)",
                file.path.display(),
                file.bytes as f64 / MIB
            ),
        }
    }
    let total: u64 = files.iter().map(|f| f.bytes).sum();
    info!(
        "{} files, {:.1} GiB",
        files.len(),
        total as f64 / MIB / 1024.0
    );
    Ok(())
}

fn report_gguf(path: &Path, summary: &inference_api::GgufWriteSummary) {
    let types = summary
        .types
        .iter()
        .map(|(name, count)| format!("{count} {name}"))
        .collect::<Vec<_>>()
        .join(", ");
    info!(
        "Wrote {} ({:.1} MiB): {types}",
        path.display(),
        summary.bytes as f64 / MIB
    );
}

/// Run UQFF quantization and generation, supporting multiple ISQ types.
pub async fn run_quantize(model_type: QuantizeModelType, global: GlobalOptions) -> Result<()> {
    initialize_logging();

    let isq_values = get_isq_values(&model_type);
    let base_output = get_output_path(&model_type).clone();
    let file_mode = base_output.extension().is_some_and(|ext| ext == "uqff");
    let model_id = get_model_id(&model_type).to_string();
    let no_readme = get_no_readme(&model_type);
    let (flag_base_model, flag_repo_id) = get_readme_overrides(&model_type);

    // Expand numeric ISQ shorthands into concrete variants (both Metal and non-Metal),
    // then deduplicate by IsqType.
    let mut seen_strings = HashSet::new();
    let mut seen_types = HashSet::new();
    let mut expanded_isq: Vec<IsqType> = Vec::new();
    for val in isq_values {
        if !seen_strings.insert(val.to_lowercase()) {
            warn!("Duplicate --isq value '{}'; skipping.", val);
            continue;
        }
        let types = expand_isq_value(val)?;
        for tp in types {
            if seen_types.insert(tp) {
                expanded_isq.push(tp);
            }
        }
    }

    // Multiple expanded ISQ types require directory output mode
    if expanded_isq.len() > 1 && file_mode {
        anyhow::bail!(
            "Cannot use multiple --isq values with a .uqff output path (ISQ setting produced multiple expanded ISQ values). \
             Use a directory path (e.g., -o output/) to auto-name files per ISQ type."
        );
    }

    let effective_output = if file_mode {
        base_output.clone()
    } else if expanded_isq.len() == 1 {
        base_output.join(format!("{}.uqff", expanded_isq[0]))
    } else {
        base_output.clone()
    };
    let write_uqff = UqffWriteConfig::with_types(effective_output.clone(), expanded_isq.clone())
        .artifact_only()
        .with_report_metadata(
            Some(flag_base_model.clone().unwrap_or_else(|| model_id.clone())),
            flag_repo_id.clone(),
        );
    let requested = expanded_isq
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");

    let (model_selected, cpu, device_layers) = convert_to_model_selected(&model_type, write_uqff)?;
    let model_selected = resolve_model_source(
        model_selected,
        &global.token_source,
        cpu,
        QuantPolicy::GgufInput,
    )
    .await?
    .model;
    if !file_mode {
        std::fs::create_dir_all(&base_output)?;
    }
    info!(
        "Starting UQFF generation for ISQ=[{}] -> `{}`",
        requested,
        effective_output.display()
    );
    let is_multimodal = matches!(&model_type, QuantizeModelType::Multimodal { .. })
        || matches!(
            &model_selected,
            ModelSelected::GGUF {
                mmproj_filename: Some(_),
                ..
            }
        );

    // Loading with write_uqff set writes the files; the engine is shut down once it has.
    let spec = EngineSpec {
        model: Some(model_selected),
        model_id: None,
        runtime: RuntimeSpec {
            device: cpu.then(|| "cpu".to_string()),
            max_seqs: Some(1),
            prefix_cache_n: Some(0),
            paged_attn: Some(false),
            token_source: Some(global.token_source.to_string()),
            device_layers,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = Engine::load(spec).await?;
    engine.shutdown().await.map_err(anyhow::Error::msg)?;

    info!("UQFF generation for ISQ=[{}] complete!", requested);

    // Generate README.md model card and upload hint in directory mode
    if !file_mode {
        let (base_model, repo_id) = if no_readme {
            (model_id.clone(), flag_repo_id)
        } else if flag_base_model.is_some() || flag_repo_id.is_some() {
            // CLI flags provided, skip interactive prompts
            (
                flag_base_model.unwrap_or_else(|| model_id.clone()),
                flag_repo_id,
            )
        } else {
            prompt_readme_details(&model_id)
        };

        if !no_readme
            && let Err(e) =
                generate_model_card(&base_output, &base_model, repo_id.as_deref(), is_multimodal)
        {
            warn!("Failed to generate README.md: {}", e);
        }

        print_upload_hint(&base_output, repo_id.as_deref(), &model_id);
    }

    Ok(())
}

/// Prompt the user for base model and upload destination to populate the README.
fn prompt_readme_details(default_model_id: &str) -> (String, Option<String>) {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    // Ask for base model
    eprintln!();
    eprint!("Base model for the README (press Enter for '{default_model_id}'): ",);
    io::stderr().flush().ok();
    let base_model = lines
        .next()
        .and_then(|l| l.ok())
        .map(|l| l.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_model_id.to_string());

    // Ask for upload destination
    eprint!("HF repo where this will be uploaded (e.g. 'user/model-UQFF', press Enter to skip): ");
    io::stderr().flush().ok();
    let repo_id = lines
        .next()
        .and_then(|l| l.ok())
        .map(|l| l.trim().to_string())
        .filter(|s| !s.is_empty());

    eprintln!();
    (base_model, repo_id)
}

/// Generate a README.md model card in the UQFF output directory.
fn generate_model_card(
    output_dir: &Path,
    base_model: &str,
    repo_id: Option<&str>,
    is_multimodal: bool,
) -> Result<()> {
    // Scan the output directory for .uqff files and group by prefix
    let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("uqff") {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                // Group shards: strip trailing numeric suffix (e.g., "q4k-0" -> "q4k")
                let key = if let Some((pre, suf)) = stem.rsplit_once('-') {
                    if suf.chars().all(|c| c.is_ascii_digit()) {
                        pre.to_string()
                    } else {
                        stem.to_string()
                    }
                } else {
                    stem.to_string()
                };
                groups.entry(key).or_default().push(path);
            }
        }
    }

    if groups.is_empty() {
        warn!("No .uqff files found in output directory, skipping README.md generation");
        return Ok(());
    }

    let repo_display = repo_id.unwrap_or("<REPO_ID>");

    let has_afq = groups.keys().any(|k| k.to_lowercase().starts_with("afq"));
    let afq_note = if has_afq {
        "**Note:** AFQ variants are optimized for Apple Silicon / Metal."
    } else {
        ""
    };

    let mut output = format!(
        r#"---
tags:
  - uqff
  - inference.rs
base_model: {base_model}
base_model_relation: quantized
---

# `{base_model}`, UQFF quantization

Generated with [inference.rs](https://github.com/christopherthompson81/inference.rs) {inference_version}. Documentation: [UQFF docs](https://github.com/christopherthompson81/inference.rs/blob/master/docs/src/content/docs/guides/quantization/uqff.mdx).

1) **Flexible** 🌀: Multiple quantization formats in *one* file format with *one* framework to run them all.
2) **Versioned**: Embedded semantic-version metadata lets inference.rs detect incompatible artifacts before loading.
3) **Easy** 🤗: Download UQFF models *easily* and *quickly* from Hugging Face, or use a local file.
4) **Customizable** 🛠️: Make and publish your own UQFF files in minutes.

## Install

Install [inference.rs](https://github.com/christopherthompson81/inference.rs) ([full guide](https://github.com/christopherthompson81/inference.rs/blob/master/docs/src/content/docs/quickstart.mdx)):

**Linux/macOS:**
```
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/christopherthompson81/inference.rs/master/install.sh | sh
```

**Windows (PowerShell):**
```
irm https://raw.githubusercontent.com/christopherthompson81/inference.rs/master/install.ps1 | iex
```

## Examples

{afq_note}

|Quantization|Command|
|--|--|
"#,
        inference_version = inference_api::INFERENCE_RS_VERSION,
    );

    let model_type = if is_multimodal { "multimodal " } else { "" };

    for (prefix, paths) in &groups {
        // Sort shards by numeric suffix
        let mut paths_sorted = paths.clone();
        paths_sorted.sort_by_key(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
            if let Some((_, suf)) = stem.rsplit_once('-') {
                suf.parse::<u64>().unwrap_or(u64::MAX)
            } else {
                u64::MAX
            }
        });

        // Use only the first shard file (auto-discovery handles the rest)
        let first_file = paths_sorted[0]
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();

        let quant_name = prefix.to_uppercase();
        output += &format!(
            "|{quant_name}|`inference run {model_type}-m {repo_display} --from-uqff {first_file}`|\n"
        );
    }

    let readme_path = output_dir.join("README.md");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&readme_path)?;
    file.write_all(output.as_bytes())?;

    info!("Generated model card at `{}`", readme_path.display());
    Ok(())
}

/// Print the hf cli upload command for the user.
fn print_upload_hint(output_dir: &Path, repo_id: Option<&str>, model_id: &str) {
    let repo = if let Some(id) = repo_id {
        id.to_string()
    } else {
        let model_name = Path::new(model_id)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(model_id);
        format!("<YOUR_USERNAME>/{model_name}-UQFF")
    };

    info!("To upload your UQFF to Hugging Face, run:");
    info!(
        "  hf upload {repo} {} --repo-type model --private",
        output_dir.display()
    );
}

/// The `ModelSelected` to load for UQFF output: quantize's arguments go through `serve`'s conversion.
fn convert_to_model_selected(
    model_type: &QuantizeModelType,
    write_uqff: UqffWriteConfig,
) -> Result<(ModelSelected, bool, Option<Vec<String>>)> {
    let mut model_type = as_model_type(model_type);
    serve::normalize_quant_flags(&mut model_type)?;
    let (cpu, device_layers) = extract_device_settings(&model_type);
    let mut selected =
        serve::convert_to_model_selected(&model_type, &MatformerSelection::default())?;
    *selected
        .write_uqff_mut()
        .expect("every kind quantize accepts can write a UQFF") = Some(write_uqff);
    Ok((selected, cpu, device_layers))
}

fn as_model_type(model_type: &QuantizeModelType) -> ModelType {
    let source =
        |model: &QuantizeModelSourceOptions, arch: Option<NormalLoaderType>| ModelSourceOptions {
            model_id: model
                .model_id
                .clone()
                .expect("quantize model source was normalized"),
            tokenizer: model.tokenizer.clone(),
            arch,
            dtype: model.dtype,
            hf_overrides: None,
            max_model_len: None,
        };
    let quantization =
        |options: &QuantizeQuantizationOptions, quant: Option<String>| QuantizationOptions {
            quant,
            in_situ_quant: None,
            from_uqff: None,
            isq_organization: options.isq_organization,
            imatrix: options.imatrix.clone(),
            calibration_file: options.calibration_file.clone(),
        };
    let multimodal = |options: &QuantizeMultimodalOptions| MultimodalOptions {
        encoder_cache_memory_mb: None,
        max_edge: options.max_edge,
        max_num_images: options.max_num_images,
        max_image_length: options.max_image_length,
    };
    match model_type {
        QuantizeModelType::Auto {
            model,
            quantization: options,
            device,
            multimodal: multimodal_options,
            ..
        } => ModelType::Auto {
            model: source(model, None),
            format: model.format.to_format_options(),
            adapter: AdapterOptions::default(),
            quantization: quantization(options, model.quant.clone()),
            device: device_options(device),
            cache: CacheOptions::default(),
            multimodal: multimodal(multimodal_options),
        },
        QuantizeModelType::Text {
            model,
            arch,
            quantization: options,
            device,
            ..
        } => ModelType::Text {
            model: source(model, arch.clone()),
            format: model.format.to_format_options(),
            adapter: AdapterOptions::default(),
            quantization: quantization(options, model.quant.clone()),
            device: device_options(device),
            cache: CacheOptions::default(),
        },
        QuantizeModelType::Multimodal {
            model,
            quantization: options,
            device,
            multimodal: multimodal_options,
            ..
        } => ModelType::Multimodal {
            model: source(model, None),
            format: model.format.to_format_options(),
            adapter: MultimodalAdapterOptions::default(),
            quantization: quantization(options, model.quant.clone()),
            device: device_options(device),
            cache: CacheOptions::default(),
            multimodal: multimodal(multimodal_options),
        },
        QuantizeModelType::Embedding {
            model,
            quantization: options,
            device,
            ..
        } => ModelType::Embedding {
            model: ModelSourceOptions {
                model_id: model.model_id.clone(),
                tokenizer: model.tokenizer.clone(),
                arch: None,
                dtype: model.dtype,
                hf_overrides: None,
                max_model_len: None,
            },
            format: FormatOptions::default(),
            quantization: quantization(options, None),
            device: device_options(device),
            cache: CacheOptions::default(),
        },
    }
}

fn device_options(device: &QuantizeDeviceOptions) -> DeviceOptions {
    DeviceOptions {
        cpu: device.cpu,
        vulkan: false,
        device_layers: device.device_layers.clone(),
        topology: device.topology.clone(),
        hf_cache: device.hf_cache.clone(),
        max_seq_len: device.max_seq_len,
        max_batch_size: device.max_batch_size,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use clap::Parser;

    use super::*;
    use crate::args::{Cli, Command, ModelType, QuantizeCommand, resolve_quantize_command};

    fn parse(args: &[&str]) -> QuantizeModelType {
        let cli = Cli::try_parse_from(
            ["inference", "quantize"]
                .into_iter()
                .chain(args.iter().copied()),
        )
        .unwrap();
        let Command::Quantize {
            model_type,
            default_quantize,
        } = cli.command
        else {
            unreachable!()
        };
        match resolve_quantize_command(model_type, default_quantize).unwrap() {
            QuantizeCommand::Uqff(model_type) => *model_type,
            QuantizeCommand::Asr(_)
            | QuantizeCommand::HiDreamO1(_)
            | QuantizeCommand::Worldgen(_)
            | QuantizeCommand::Gguf(_)
            | QuantizeCommand::Imatrix(_)
            | QuantizeCommand::NemotronTranscribeAudio(_)
            | QuantizeCommand::Mmproj(_)
            | QuantizeCommand::Exl3(_)
            | QuantizeCommand::Hessians(_)
            | QuantizeCommand::Kld(_) => {
                unreachable!("these tests quantize UQFF")
            }
        }
    }

    #[test]
    fn multimodal_model_card_commands_parse() {
        let root = std::env::temp_dir().join(format!("inference-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("q4k.uqff"), []).unwrap();

        generate_model_card(&root, "org/base", Some("org/quantized"), true).unwrap();
        let readme = fs::read_to_string(root.join("README.md")).unwrap();
        let command = readme
            .lines()
            .find(|line| line.starts_with("|Q4K|"))
            .and_then(|line| line.split('`').nth(1))
            .expect("Q4K command in generated model card");
        assert_eq!(
            command,
            "inference run multimodal -m org/quantized --from-uqff q4k.uqff"
        );

        let cli = Cli::try_parse_from(command.split_whitespace()).unwrap();
        let Command::Run {
            model_type:
                Some(ModelType::Multimodal {
                    model,
                    quantization,
                    ..
                }),
            ..
        } = cli.command
        else {
            panic!("expected a multimodal run command")
        };
        assert_eq!(model.model_id, "org/quantized");
        assert_eq!(quantization.from_uqff.as_deref(), Some("q4k.uqff"));

        fs::remove_dir_all(root).unwrap();
    }

    async fn resolved(
        model_type: &QuantizeModelType,
        output: &Path,
        isq: IsqType,
    ) -> Result<ModelSelected> {
        let write_uqff = UqffWriteConfig::with_types(output.to_path_buf(), vec![isq]);
        let (selected, _, _) = convert_to_model_selected(model_type, write_uqff)?;
        let resolved = resolve_model_source(
            selected,
            &inference_api::engine::TokenSource::None,
            true,
            QuantPolicy::GgufInput,
        )
        .await?;
        Ok(resolved.model)
    }

    #[tokio::test]
    async fn gguf_artifact_and_projector_are_selected_for_uqff_output() {
        let root = std::env::temp_dir().join(format!("inference-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        for file in [
            "model-Q4_K_S.gguf",
            "model-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "mmproj-BF16.gguf",
        ] {
            fs::write(root.join(file), []).unwrap();
        }
        let output = root.join("output.uqff");
        let model_type = parse(&[
            "-m",
            &root.to_string_lossy(),
            "--quant",
            "4",
            "--isq",
            "q8_0",
            "--dtype",
            "bf16",
            "-o",
            &output.to_string_lossy(),
        ]);
        assert_eq!(get_isq_values(&model_type), ["q8_0"]);

        let selected = resolved(&model_type, &output, IsqType::Q8_0).await.unwrap();
        let ModelSelected::GGUF {
            quantized_filename,
            mmproj_filename,
            write_uqff,
            ..
        } = selected
        else {
            panic!("expected GGUF source")
        };
        assert_eq!(quantized_filename, "model-Q4_K_M.gguf");
        assert_eq!(mmproj_filename.as_deref(), Some("mmproj-BF16.gguf"));
        let write_uqff = write_uqff.expect("GGUF source should write UQFF");
        assert_eq!(write_uqff.output, output);
        assert_eq!(write_uqff.types, [IsqType::Q8_0]);

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn exact_gguf_preserves_asset_and_projector_overrides() {
        let root = std::env::temp_dir().join(format!("inference-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model-Q4_K_M.gguf"), []).unwrap();
        fs::write(root.join("custom-mmproj.gguf"), []).unwrap();
        let output = root.join("output.uqff");
        let model_type = parse(&[
            "-m",
            &root.to_string_lossy(),
            "-f",
            "model-Q4_K_M.gguf",
            "--mmproj",
            "custom-mmproj.gguf",
            "--tok-model-id",
            "org/base",
            "--isq",
            "q5k",
            "-o",
            &output.to_string_lossy(),
        ]);

        let selected = resolved(&model_type, &output, IsqType::Q5K).await.unwrap();
        let ModelSelected::GGUF {
            tok_model_id,
            quantized_filename,
            mmproj_filename,
            ..
        } = selected
        else {
            panic!("expected GGUF source")
        };
        assert_eq!(tok_model_id.as_deref(), Some("org/base"));
        assert_eq!(quantized_filename, "model-Q4_K_M.gguf");
        assert_eq!(mmproj_filename.as_deref(), Some("custom-mmproj.gguf"));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn direct_local_gguf_discovers_only_a_sibling_projector() {
        let root = std::env::temp_dir().join(format!("inference-{}", uuid::Uuid::new_v4()));
        let unrelated = root.join("unrelated");
        fs::create_dir_all(&unrelated).unwrap();
        let model_path = root.join("model.gguf");
        fs::write(&model_path, []).unwrap();
        fs::write(root.join("mmproj-BF16.gguf"), []).unwrap();
        fs::write(root.join("model.safetensors"), []).unwrap();
        fs::write(unrelated.join("mmproj-BF16.gguf"), []).unwrap();
        let output = root.join("output.uqff");
        let model_type = parse(&[
            "-f",
            &model_path.to_string_lossy(),
            "--isq",
            "q4k",
            "-o",
            &output.to_string_lossy(),
        ]);

        let selected = resolved(&model_type, &output, IsqType::Q4K).await.unwrap();
        let ModelSelected::GGUF {
            quantized_filename,
            mmproj_filename,
            ..
        } = selected
        else {
            panic!("expected GGUF source")
        };
        assert_eq!(quantized_filename, "model.gguf");
        assert_eq!(mmproj_filename.as_deref(), Some("mmproj-BF16.gguf"));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn multimodal_gguf_requires_a_projector() {
        let root = std::env::temp_dir().join(format!("inference-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model-Q4_K_M.gguf"), []).unwrap();
        let output = root.join("output.uqff");
        let model_type = parse(&[
            "multimodal",
            "-m",
            &root.to_string_lossy(),
            "-f",
            "model-Q4_K_M.gguf",
            "--isq",
            "q4k",
            "-o",
            &output.to_string_lossy(),
        ]);

        let error = resolved(&model_type, &output, IsqType::Q4K)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("projector"), "{error}");

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn mixed_source_directory_requires_explicit_gguf_format() {
        let root = std::env::temp_dir().join(format!("inference-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model-Q4_K_M.gguf"), []).unwrap();
        fs::write(root.join("model.safetensors"), []).unwrap();
        let output = root.join("output.uqff");
        let root_arg = root.to_string_lossy().into_owned();
        let output_arg = output.to_string_lossy().into_owned();
        let args = [
            "-m",
            root_arg.as_str(),
            "--quant",
            "4",
            "--isq",
            "q8_0",
            "-o",
            output_arg.as_str(),
        ];

        let ambiguous = parse(&args);
        let error = resolved(&ambiguous, &output, IsqType::Q8_0)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("--format gguf"), "{error}");

        let mut explicit_args = args.to_vec();
        explicit_args.extend(["--format", "gguf"]);
        let explicit = parse(&explicit_args);
        let selected = resolved(&explicit, &output, IsqType::Q8_0).await.unwrap();
        assert!(matches!(
            selected,
            ModelSelected::GGUF { ref quantized_filename, .. } if quantized_filename == "model-Q4_K_M.gguf"
        ));

        fs::remove_dir_all(root).unwrap();
    }
}
