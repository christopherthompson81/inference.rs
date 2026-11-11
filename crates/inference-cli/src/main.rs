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
    run_bench, run_cache_delete, run_cache_list, run_doctor, run_from_config, run_gguf_quantize,
    run_hidream_quantize, run_imatrix_convert, run_interactive, run_kernels_fetch,
    run_kernels_prune, run_launch, run_login, run_mmproj_quantize, run_nemotron_transcribe_audio,
    run_quantize, run_server, run_tune, run_uninstall, run_update, run_uqff, run_worldgen_quantize,
};
use inference_api::{LogVerbosity, initialize_inference_logging};

// Tensor ops allocate fresh output buffers constantly; mimalloc removes the page-fault
// churn that dominates small-model CPU inference with the system allocator.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// an idle engine hands back the pages mimalloc keeps per thread (the weights' f32 copies after a model halves them)
fn release_idle_memory() {
    // SAFETY: mi_collect takes no pointers and is safe on any thread
    unsafe { libmimalloc_sys::mi_collect(true) };
}

#[tokio::main]
async fn main() -> Result<()> {
    inference_tensor::utils::set_idle_release(release_idle_memory);
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
            QuantizeCommand::Gguf(options) => run_gguf_quantize(options)?,
            QuantizeCommand::Imatrix(options) => run_imatrix_convert(options)?,
            QuantizeCommand::NemotronTranscribeAudio(options) => {
                run_nemotron_transcribe_audio(options)?
            }
            QuantizeCommand::Mmproj(options) => run_mmproj_quantize(options)?,
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
            use inference_jobs::reconstruction::{
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

        Command::ImageTo3d {
            image,
            out,
            variant,
            seed,
            vertex_colors,
            texture_size,
            remesh,
            faces,
            samples,
            structure_steps,
            structure_guidance,
            structure_guidance_rescale,
            structure_rescale_t,
            shape_steps,
            shape_guidance,
            shape_guidance_rescale,
            shape_rescale_t,
            texture_steps,
            texture_guidance,
            texture_guidance_rescale,
            texture_rescale_t,
            trellis2,
            trellis_large,
            dinov3,
            birefnet,
            remove_background,
            cpu,
        } => {
            use inference_jobs::image_to_3d::Variant;
            use inference_jobs::image_to_3d::{
                BakeOptions, FACES_RANGE, ImageTo3dOptions, MAX_SAMPLES, NoProgress, SamplerChoice,
                Settings, fetch_models, generate_3d_on, sample_dir,
            };
            use inference_jobs::voxels::needs_background_removal;
            let variant = Variant::parse(&variant).ok_or_else(|| {
                anyhow::anyhow!(
                    "--variant must be 1024-cascade, 1536-cascade, 1024 or 512, got `{variant}`"
                )
            })?;
            let settings = Settings {
                structure: SamplerChoice {
                    steps: structure_steps,
                    guidance_strength: structure_guidance,
                    guidance_rescale: structure_guidance_rescale,
                    rescale_t: structure_rescale_t,
                },
                shape: SamplerChoice {
                    steps: shape_steps,
                    guidance_strength: shape_guidance,
                    guidance_rescale: shape_guidance_rescale,
                    rescale_t: shape_rescale_t,
                },
                texture: SamplerChoice {
                    steps: texture_steps,
                    guidance_strength: texture_guidance,
                    guidance_rescale: texture_guidance_rescale,
                    rescale_t: texture_rescale_t,
                },
                samples,
                ..Settings::new(variant, seed)
            };
            if !(1..=MAX_SAMPLES).contains(&samples) {
                anyhow::bail!("--samples must be 1 to {MAX_SAMPLES}");
            }
            for (stage, choice) in [
                ("structure", settings.structure),
                ("shape", settings.shape),
                ("texture", settings.texture),
            ] {
                if let Some(problem) = choice.problem() {
                    anyhow::bail!("--{stage}-*: {problem}");
                }
            }
            if !FACES_RANGE.contains(&faces) {
                anyhow::bail!(
                    "--faces must be in {}..={}",
                    FACES_RANGE.start(),
                    FACES_RANGE.end()
                );
            }
            let options = ImageTo3dOptions {
                settings,
                faces,
                remove_background,
                bake: (!vertex_colors).then(|| BakeOptions {
                    texture_size,
                    remesh,
                    ..BakeOptions::default()
                }),
            };
            let birefnet = needs_background_removal(&image, remove_background)?.then_some(birefnet);
            let models = fetch_models(trellis2, trellis_large, dinov3, birefnet, &NoProgress)?;
            let device = if cpu {
                inference_tensor::Device::Cpu
            } else {
                inference_tensor::Device::cuda_if_available(0)?
            };
            let reports = generate_3d_on(&models, &image, &out, &options, &NoProgress, &device)?;
            for (i, report) in reports.iter().enumerate() {
                println!(
                    "{} vertices, {} faces ({} extracted) at {}^3{}, seed {}, in {} ({:.1} s)",
                    report.vertices,
                    report.faces,
                    report.raw_faces,
                    report.resolution,
                    if report.background_removed {
                        ", background removed"
                    } else {
                        ""
                    },
                    report.seed,
                    sample_dir(&out, i, reports.len()).display(),
                    report.total_seconds
                );
            }
        }

        Command::TextureMesh {
            mesh,
            image,
            out,
            resolution,
            seed,
            texture_size,
            unwrap,
            texture_steps,
            texture_guidance,
            texture_guidance_rescale,
            texture_rescale_t,
            trellis2,
            dinov3,
            birefnet,
            remove_background,
            cpu,
        } => {
            use inference_jobs::image_to_3d::SamplerChoice;
            use inference_jobs::texture_mesh::{
                NoProgress, RESOLUTIONS, TextureMeshOptions, TexturingSettings, fetch_models,
                texture_mesh_on,
            };
            use inference_jobs::voxels::needs_background_removal;
            if !RESOLUTIONS.contains(&resolution) {
                anyhow::bail!("--resolution must be 512 or 1024");
            }
            let texture = SamplerChoice {
                steps: texture_steps,
                guidance_strength: texture_guidance,
                guidance_rescale: texture_guidance_rescale,
                rescale_t: texture_rescale_t,
            };
            if let Some(problem) = texture.problem() {
                anyhow::bail!("--texture-*: {problem}");
            }
            let options = TextureMeshOptions {
                settings: TexturingSettings {
                    resolution,
                    seed,
                    texture,
                },
                texture_size,
                unwrap,
                remove_background,
            };
            let birefnet = needs_background_removal(&image, remove_background)?.then_some(birefnet);
            let models = fetch_models(trellis2, dinov3, birefnet, &NoProgress)?;
            let device = if cpu {
                inference_tensor::Device::Cpu
            } else {
                inference_tensor::Device::cuda_if_available(0)?
            };
            let report =
                texture_mesh_on(&models, &mesh, &image, &out, &options, &NoProgress, &device)?;
            println!(
                "{} vertices, {} faces painted at {}^3 ({}{}) in {} ({:.1} s)",
                report.vertices,
                report.faces,
                report.resolution,
                if report.own_uvs {
                    "its own uvs"
                } else {
                    "unwrapped"
                },
                if report.background_removed {
                    ", background removed"
                } else {
                    ""
                },
                out.display(),
                report.total_seconds
            );
        }

        Command::Voxels {
            image,
            out,
            jovox,
            dinov3,
            birefnet,
            trellis2,
            no_mesh,
            seed,
            steps,
            cfg,
            temperature,
            fill,
            remove_background,
            cpu,
        } => {
            use inference_jobs::voxels::{
                NoProgress, VoxelOptions, fetch_models, generate_voxels_on,
                needs_background_removal,
            };
            let mut options = VoxelOptions {
                fill,
                remove_background,
                mesh: !no_mesh,
                ..Default::default()
            };
            options.sample.seed = seed;
            options.sample.steps = steps;
            options.sample.cfg_strength = cfg;
            options.sample.temperature = temperature;
            let birefnet = needs_background_removal(&image, remove_background)?.then_some(birefnet);
            let models = fetch_models(
                jovox,
                dinov3,
                birefnet,
                (!no_mesh).then_some(trellis2),
                &NoProgress,
            )?;
            let device = if cpu {
                inference_tensor::Device::Cpu
            } else {
                inference_tensor::Device::cuda_if_available(0)?
            };
            let report = generate_voxels_on(&models, &image, &out, &options, &NoProgress, &device)?;
            println!(
                "{} voxels at {}^3{} in {} (sampling {:.1} s, total {:.1} s)",
                report.voxels,
                report.resolution,
                if report.background_removed {
                    ", background removed"
                } else {
                    ""
                },
                out.display(),
                report.sample_seconds,
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
                use inference_jobs::reconstruction::{NoProgress, TrainOptions, train_splats};
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
            use inference_jobs::reconstruction::{
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
