//! inference-cli - Clean command-line interface for inference.rs
//!
//! A new CLI design with:
//! - Orthogonal flags (format, adapter, modality are independent)
//! - Unified PagedAttention configuration
//! - Logical argument grouping
//! - Config-file-first support

mod args;
mod commands;
mod config;
#[cfg(test)]
mod docgen;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use clap_complete::generate;

use args::{
    CacheCommand, Cli, Command, KernelsCommand, QuantizeCommand, resolve_model_type,
    resolve_quantize_command,
};
use commands::{
    AgentModel, BenchRunConfig, LaunchRequest, list_agent_sessions, run_agent, run_asr_quantize,
    run_bench, run_cache_delete, run_cache_list, run_doctor, run_from_config, run_hidream_quantize,
    run_interactive, run_kernels_fetch, run_kernels_prune, run_launch, run_login, run_quantize,
    run_server, run_tune, run_uninstall, run_update, run_uqff, run_worldgen_quantize,
};
use inference_api::{LogVerbosity, initialize_inference_logging};

// Tensor ops allocate fresh output buffers constantly; mimalloc removes the page-fault
// churn that dominates small-model CPU inference with the system allocator.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> Result<()> {
    inference_tensor::utils::init_global_threadpool();
    let cli = Cli::parse();
    init_tracing(cli.global.verbose);

    match cli.command {
        Command::Serve {
            model_type,
            default_model,
            server,
            runtime,
            agent_options,
            sandbox,
        } => {
            // without a model the server starts empty and loads models on request
            let empty = model_type.is_none() && !default_model.names_model();
            let model_type = if empty {
                default_model.into_settings_only()?
            } else {
                resolve_model_type(model_type, default_model)?
            };
            run_server(
                model_type,
                empty,
                server,
                runtime,
                agent_options,
                sandbox,
                cli.global,
            )
            .await?;
        }

        Command::Run {
            model_type,
            default_model,
            runtime,
            agent_options,
            sandbox,
            thinking,
            reasoning_effort,
            input,
            image,
            video,
            audio,
            adapter,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_interactive(
                model_type,
                runtime,
                agent_options,
                sandbox,
                cli.global,
                thinking,
                reasoning_effort,
                input,
                image,
                video,
                audio,
                adapter,
            )
            .await?;
        }

        Command::Launch {
            agent,
            url,
            config,
            api_key_file,
            model,
            agent_args,
        } => {
            let code = run_launch(LaunchRequest {
                agent,
                url,
                config,
                api_key_file,
                model,
                agent_args,
            })
            .await?;
            std::process::exit(code);
        }

        Command::Agent { coder, .. } if coder.sessions => list_agent_sessions(&coder)?,

        Command::Agent {
            model_type,
            default_model,
            runtime,
            sandbox,
            coder,
            thinking,
            reasoning_effort,
        } => {
            let model = match &coder.url {
                Some(url) => AgentModel::Server {
                    url: url.clone(),
                    model: default_model
                        .model_id
                        .unwrap_or_else(|| "default".to_string()),
                },
                None => AgentModel::Local {
                    model_type: Box::new(resolve_model_type(model_type, default_model)?),
                    runtime: Box::new(runtime),
                    sandbox,
                },
            };
            run_agent(model, coder, cli.global, thinking, reasoning_effort).await?;
        }

        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            generate(shell, &mut cmd, name, &mut std::io::stdout());
        }

        Command::Quantize {
            model_type,
            default_quantize,
        } => match resolve_quantize_command(model_type, default_quantize)? {
            QuantizeCommand::Uqff(model_type) => run_quantize(*model_type, cli.global).await?,
            QuantizeCommand::Asr(options) => run_asr_quantize(options, cli.global)?,
            QuantizeCommand::HiDreamO1(options) => run_hidream_quantize(options, cli.global)?,
            QuantizeCommand::Worldgen(options) => run_worldgen_quantize(options)?,
        },

        Command::Uqff { command } => {
            run_uqff(command, cli.global).await?;
        }

        Command::FromConfig { file } => {
            run_from_config(file).await?;
        }

        Command::Update { tag } => {
            run_update(tag)?;
        }

        Command::Uninstall { yes } => {
            run_uninstall(yes)?;
        }

        Command::Doctor { json } => {
            run_doctor(json)?;
        }

        Command::Tune {
            model_type,
            default_model,
            profile,
            json,
            emit_config,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_tune(model_type, cli.global, profile, json, emit_config).await?;
        }

        Command::Login { token } => {
            run_login(token)?;
        }

        Command::Cache { cmd } => match cmd {
            CacheCommand::List => run_cache_list()?,
            CacheCommand::Delete { model_id } => run_cache_delete(&model_id)?,
        },

        Command::Reconstruct {
            model,
            input,
            out,
            target_size,
            fps,
            max_frames,
            no_edge_mask,
            no_sky_mask,
            prior_cameras,
            prior_depth,
            cpu,
        } => {
            use inference_api::reconstruction::{
                OutputSettings, ReconstructOptions, VideoOptions, reconstruct,
            };
            let options = ReconstructOptions {
                target_size,
                output: OutputSettings {
                    edge_mask: !no_edge_mask,
                    sky_mask: !no_sky_mask,
                    ..OutputSettings::default()
                },
                prior_cameras,
                prior_depth,
            };
            let video = VideoOptions { fps, max_frames };
            let report = reconstruct(&model, &input, &out, &options, video, cpu).await?;
            println!(
                "{} frames at {}x{}: {} points, {} Gaussians in {} (forward {:.1} s, total {:.1} s)",
                report.frames,
                report.width,
                report.height,
                report.written.points,
                report.written.gaussians,
                out.display(),
                report.forward_seconds,
                report.total_seconds
            );
        }

        Command::Splat {
            data,
            out,
            steps,
            lpips,
        } => {
            #[cfg(feature = "cuda")]
            {
                use inference_api::reconstruction::{NoProgress, TrainOptions, train_splats};
                let options = TrainOptions { steps, lpips };
                let report = train_splats(&data, &out, &options, &NoProgress)?;
                println!(
                    "{} gaussians ({} exported) after {} steps, validation PSNR {:.3}, {:.1} s; {}",
                    report.gaussians,
                    report.exported,
                    report.steps,
                    report.val_psnr,
                    report.seconds,
                    out.join("splats.ply").display()
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (data, out, steps, lpips);
                anyhow::bail!("splat training needs a CUDA build");
            }
        }

        Command::Worldgen {
            panorama,
            out,
            worldstereo,
            wan,
            moge,
            worldmirror,
            prompt,
            scene,
            seed,
            reconstruct_size,
            attention,
            weights,
        } => {
            use inference_api::reconstruction::{
                AttentionKind, GgmlDType, NoProgress, WorldGenModels, WorldGenOptions,
                generate_world,
            };
            let models = WorldGenModels {
                worldstereo,
                wan,
                moge,
                worldmirror,
            };
            let options = WorldGenOptions {
                prompt,
                outdoor: scene.map(|s| s == "outdoor"),
                seed,
                reconstruct_size,
                attention: if attention == "bf16" {
                    AttentionKind::Bf16
                } else {
                    AttentionKind::Int8Qk
                },
                block_weights: (weights == "q8_0").then_some(GgmlDType::Q8_0),
            };
            let report = generate_world(&models, &panorama, &out, &options, &NoProgress)?;
            for stage in &report.stages {
                println!(
                    "{:<36} {:8.1} s  {:6} MiB allocated at peak, {:6} MiB in use at its end",
                    stage.stage, stage.seconds, stage.allocated_peak_mib, stage.device_mib_at_end
                );
            }
            println!(
                "{} clips, {} frames; reconstruction from {} frames in {}",
                report.clips.len(),
                report.frames,
                report.reconstructed_frames,
                out.display()
            );
        }

        Command::Kernels { cmd } => match cmd {
            KernelsCommand::Fetch { types, all } => run_kernels_fetch(types, all)?,
            KernelsCommand::Prune { dry_run } => run_kernels_prune(dry_run)?,
        },

        Command::Bench {
            model_type,
            default_model,
            runtime,
            adapter,
            prompt_len,
            gen_len,
            depth,
            iterations,
            warmup,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_bench(
                model_type,
                runtime,
                cli.global,
                BenchRunConfig {
                    prompt_lens: prompt_len,
                    gen_len,
                    depths: depth,
                    iterations,
                    warmup,
                    adapter,
                },
            )
            .await?;
        }
    }

    Ok(())
}

fn init_tracing(verbose: u8) {
    initialize_inference_logging(LogVerbosity::from_count(verbose));
}
