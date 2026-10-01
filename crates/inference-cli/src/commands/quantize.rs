//! Quantize command implementation for UQFF generation

use std::collections::{BTreeMap, HashSet};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use tracing::{info, warn};

use inference_api::{
    Engine, EngineSpec,
    engine::{
        IsqType, ModelSelected, NormalLoaderType, RuntimeSpec, UqffWriteConfig, expand_isq_value,
    },
    initialize_logging,
    uqff::{QuantPolicy, resolve_model_source},
};

use super::serve::{self, extract_device_settings};
use crate::args::{
    AdapterOptions, CacheOptions, DeviceOptions, FormatOptions, GlobalOptions, MatformerSelection,
    ModelSourceOptions, ModelType, MultimodalAdapterOptions, MultimodalOptions,
    QuantizationOptions, QuantizeDeviceOptions, QuantizeModelSourceOptions, QuantizeModelType,
    QuantizeMultimodalOptions, QuantizeQuantizationOptions,
};

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
    use crate::args::{Cli, Command, ModelType, resolve_quantize_model_type};

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
        resolve_quantize_model_type(model_type, default_quantize).unwrap()
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
