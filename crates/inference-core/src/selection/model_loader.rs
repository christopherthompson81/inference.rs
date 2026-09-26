use std::{fs::File, path::PathBuf};

use anyhow::Context;

use crate::{
    pipeline::{
        AutoLoaderBuilder, DiffusionLoaderBuilder, GGMLLoaderBuilder, GGMLSpecificConfig,
        GGUFLoaderBuilder, GGUFSpecificConfig, HfConfigOverrides, IsqOrganization,
        MultimodalLoaderBuilder, MultimodalSpecificConfig, NormalLoaderBuilder,
        NormalSpecificConfig, UqffWriteConfig,
    },
    AutoDeviceMapParams, EmbeddingLoaderBuilder, EmbeddingSpecificConfig, Loader, ModelDType,
    ModelSelected, Ordering, SpeechLoader, Topology, GGUF_MULTI_FILE_DELIMITER,
    UQFF_MULTI_FILE_DELIMITER,
};

/// A builder for a loader using the selected model.
pub struct LoaderBuilder {
    model: ModelSelected,
    no_kv_cache: bool,
    chat_template: Option<String>,
    jinja_explicit: Option<String>,
    max_model_len: Option<usize>,
    hf_config_overrides: Option<HfConfigOverrides>,
    mtp: bool,
    encoder_cache_memory_bytes: Option<usize>,
}

impl LoaderBuilder {
    pub fn new(model: ModelSelected) -> Self {
        Self {
            model,
            no_kv_cache: false,
            chat_template: None,
            jinja_explicit: None,
            max_model_len: None,
            hf_config_overrides: None,
            mtp: false,
            encoder_cache_memory_bytes: None,
        }
    }

    /// Load the MTP head built into the checkpoint so it can drive speculative decoding.
    pub fn with_mtp(mut self, mtp: bool) -> Self {
        self.mtp = mtp;
        self
    }

    pub fn with_encoder_cache_memory_bytes(mut self, max_bytes: Option<usize>) -> Self {
        if let Some(max_bytes) = max_bytes {
            assert!(max_bytes > 0, "encoder cache memory must be nonzero");
        }
        self.encoder_cache_memory_bytes = max_bytes;
        self
    }

    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = no_kv_cache;
        self
    }
    pub fn with_chat_template(mut self, chat_template: Option<String>) -> Self {
        self.chat_template = chat_template;
        self
    }
    pub fn with_jinja_explicit(mut self, jinja_explicit: Option<String>) -> Self {
        self.jinja_explicit = jinja_explicit;
        self
    }
    pub fn with_max_model_len(mut self, max_model_len: Option<usize>) -> Self {
        self.max_model_len = max_model_len;
        self
    }
    pub fn with_hf_config_overrides(
        mut self,
        hf_config_overrides: Option<HfConfigOverrides>,
    ) -> Self {
        self.hf_config_overrides = hf_config_overrides;
        self
    }

    pub fn build(self) -> anyhow::Result<Box<dyn Loader>> {
        loader_from_model_selected(self)
    }
}

fn uqff_paths(from_uqff: Option<String>) -> Option<Vec<PathBuf>> {
    from_uqff.map(|paths| {
        paths
            .split(UQFF_MULTI_FILE_DELIMITER)
            .map(PathBuf::from)
            .collect()
    })
}

fn gguf_files(names: &str) -> Vec<String> {
    names
        .split(GGUF_MULTI_FILE_DELIMITER)
        .map(ToOwned::to_owned)
        .collect()
}

fn load_ordering(path: &str) -> anyhow::Result<Ordering> {
    let file =
        File::open(path).with_context(|| format!("Could not load ordering file at {path}"))?;
    Ok(serde_json::from_reader(file)?)
}

/// The options every safetensors model kind takes; the per-kind configs are all built from one of these.
#[derive(Clone, Default)]
struct SafetensorsOptions {
    topology: Option<Topology>,
    organization: IsqOrganization,
    write_uqff: Option<UqffWriteConfig>,
    from_uqff: Option<Vec<PathBuf>>,
    imatrix: Option<PathBuf>,
    calibration_file: Option<PathBuf>,
    hf_cache_path: Option<PathBuf>,
    matformer_config_path: Option<PathBuf>,
    matformer_slice_name: Option<String>,
    hf_config_overrides: Option<HfConfigOverrides>,
    max_model_len: Option<usize>,
}

impl SafetensorsOptions {
    fn from_args(args: &LoaderBuilder) -> Self {
        Self {
            hf_config_overrides: args.hf_config_overrides.clone(),
            max_model_len: args.max_model_len,
            ..Default::default()
        }
    }

    fn normal(&self) -> NormalSpecificConfig {
        NormalSpecificConfig {
            topology: self.topology.clone(),
            organization: self.organization,
            write_uqff: self.write_uqff.clone(),
            from_uqff: self.from_uqff.clone(),
            imatrix: self.imatrix.clone(),
            calibration_file: self.calibration_file.clone(),
            hf_cache_path: self.hf_cache_path.clone(),
            hf_config_overrides: self.hf_config_overrides.clone(),
            max_model_len: self.max_model_len,
            matformer_config_path: self.matformer_config_path.clone(),
            matformer_slice_name: self.matformer_slice_name.clone(),
        }
    }

    fn multimodal(&self, max_edge: Option<u32>) -> MultimodalSpecificConfig {
        MultimodalSpecificConfig {
            topology: self.topology.clone(),
            write_uqff: self.write_uqff.clone(),
            from_uqff: self.from_uqff.clone(),
            max_edge,
            max_model_len: self.max_model_len,
            hf_config_overrides: self.hf_config_overrides.clone(),
            imatrix: self.imatrix.clone(),
            calibration_file: self.calibration_file.clone(),
            hf_cache_path: self.hf_cache_path.clone(),
            matformer_config_path: self.matformer_config_path.clone(),
            matformer_slice_name: self.matformer_slice_name.clone(),
            organization: self.organization,
        }
    }

    fn embedding(&self) -> EmbeddingSpecificConfig {
        EmbeddingSpecificConfig {
            topology: self.topology.clone(),
            write_uqff: self.write_uqff.clone(),
            from_uqff: self.from_uqff.clone(),
            imatrix: self.imatrix.clone(),
            calibration_file: self.calibration_file.clone(),
            hf_cache_path: self.hf_cache_path.clone(),
        }
    }
}

pub fn get_tgt_non_granular_index(model: &ModelSelected) -> Option<usize> {
    match model {
        ModelSelected::Plain { .. }
        | ModelSelected::Run { .. }
        | ModelSelected::Lora { .. }
        | ModelSelected::GGUF { .. }
        | ModelSelected::LoraGGUF { .. }
        | ModelSelected::GGML { .. }
        | ModelSelected::LoraGGML { .. }
        | ModelSelected::MultimodalPlain { .. }
        | ModelSelected::DiffusionPlain { .. }
        | ModelSelected::Speech { .. }
        | ModelSelected::Embedding { .. } => None,
        ModelSelected::XLora {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraGGUF {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraGGML {
            tgt_non_granular_index,
            ..
        } => *tgt_non_granular_index,
        ModelSelected::MultiModel { .. } => {
            panic!("MultiModel variant should not be used in model loading functions")
        }
    }
}

pub fn get_model_dtype(model: &ModelSelected) -> anyhow::Result<ModelDType> {
    match model {
        ModelSelected::Plain { dtype, .. }
        | ModelSelected::Lora { dtype, .. }
        | ModelSelected::XLora { dtype, .. }
        | ModelSelected::MultimodalPlain { dtype, .. }
        | ModelSelected::DiffusionPlain { dtype, .. }
        | ModelSelected::GGML { dtype, .. }
        | ModelSelected::GGUF { dtype, .. }
        | ModelSelected::XLoraGGUF { dtype, .. }
        | ModelSelected::XLoraGGML { dtype, .. }
        | ModelSelected::LoraGGUF { dtype, .. }
        | ModelSelected::LoraGGML { dtype, .. }
        | ModelSelected::Run { dtype, .. }
        | ModelSelected::Speech { dtype, .. }
        | ModelSelected::Embedding { dtype, .. } => Ok(*dtype),
        ModelSelected::MultiModel { .. } => {
            anyhow::bail!("MultiModel variant should not be used in model loading functions")
        }
    }
}

pub fn get_auto_device_map_params(model: &ModelSelected) -> anyhow::Result<AutoDeviceMapParams> {
    match model {
        ModelSelected::Plain {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::XLora {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::GGML {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::XLoraGGUF {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::XLoraGGML {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::LoraGGUF {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::LoraGGML {
            max_seq_len,
            max_batch_size,
            ..
        } => Ok(AutoDeviceMapParams::Text {
            max_seq_len: *max_seq_len,
            max_batch_size: *max_batch_size,
        }),
        ModelSelected::GGUF {
            mmproj_filename,
            max_seq_len,
            max_batch_size,
            max_image_length,
            max_num_images,
            ..
        } => {
            if mmproj_filename.is_some() {
                let max_image_length =
                    max_image_length.unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH);
                Ok(AutoDeviceMapParams::Multimodal {
                    max_seq_len: *max_seq_len,
                    max_batch_size: *max_batch_size,
                    max_image_shape: (max_image_length, max_image_length),
                    max_num_images: max_num_images
                        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES),
                })
            } else {
                Ok(AutoDeviceMapParams::Text {
                    max_seq_len: *max_seq_len,
                    max_batch_size: *max_batch_size,
                })
            }
        }
        ModelSelected::Lora {
            arch,
            max_seq_len,
            max_batch_size,
            max_image_length,
            max_num_images,
            ..
        } => {
            if arch.is_none() && (max_num_images.is_some() || max_image_length.is_some()) {
                let max_image_length =
                    max_image_length.unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH);
                Ok(AutoDeviceMapParams::Multimodal {
                    max_seq_len: *max_seq_len,
                    max_batch_size: *max_batch_size,
                    max_image_shape: (max_image_length, max_image_length),
                    max_num_images: max_num_images
                        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES),
                })
            } else {
                Ok(AutoDeviceMapParams::Text {
                    max_seq_len: *max_seq_len,
                    max_batch_size: *max_batch_size,
                })
            }
        }
        ModelSelected::Run {
            max_seq_len,
            max_batch_size,
            max_image_length,
            max_num_images,
            ..
        } => {
            if max_num_images.is_some() || max_image_length.is_some() {
                let max_image_length =
                    max_image_length.unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH);
                Ok(AutoDeviceMapParams::Multimodal {
                    max_seq_len: *max_seq_len,
                    max_batch_size: *max_batch_size,
                    max_image_shape: (max_image_length, max_image_length),
                    max_num_images: max_num_images
                        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES),
                })
            } else {
                Ok(AutoDeviceMapParams::Text {
                    max_seq_len: *max_seq_len,
                    max_batch_size: *max_batch_size,
                })
            }
        }
        ModelSelected::MultimodalPlain {
            max_seq_len,
            max_batch_size,
            max_image_length,
            max_num_images,
            ..
        } => Ok(AutoDeviceMapParams::Multimodal {
            max_seq_len: *max_seq_len,
            max_batch_size: *max_batch_size,
            max_image_shape: (*max_image_length, *max_image_length),
            max_num_images: *max_num_images,
        }),
        ModelSelected::DiffusionPlain { .. }
        | ModelSelected::Speech { .. }
        | ModelSelected::Embedding { .. } => Ok(AutoDeviceMapParams::default_text()),
        ModelSelected::MultiModel { .. } => {
            anyhow::bail!("MultiModel variant should not be used in model loading functions")
        }
    }
}

fn loader_from_model_selected(args: LoaderBuilder) -> anyhow::Result<Box<dyn Loader>> {
    if args.max_model_len == Some(0) {
        anyhow::bail!("max_model_len must be greater than zero");
    }
    let supports_hf_config_overrides = matches!(
        &args.model,
        ModelSelected::Plain { .. }
            | ModelSelected::Run { .. }
            | ModelSelected::Lora { .. }
            | ModelSelected::XLora { .. }
            | ModelSelected::MultimodalPlain { .. }
    );
    if args.hf_config_overrides.is_some() && !supports_hf_config_overrides {
        anyhow::bail!("HF config overrides are supported only for text and multimodal models");
    }
    // the legacy X-LoRA / LoRA GGUF pipelines take their length from the model and cannot cap it
    let supports_max_model_len =
        supports_hf_config_overrides || matches!(&args.model, ModelSelected::GGUF { .. });
    if args.max_model_len.is_some() && !supports_max_model_len {
        anyhow::bail!("max_model_len is not supported by this model format");
    }

    let base = SafetensorsOptions::from_args(&args);
    let loader: Box<dyn Loader> = match args.model {
        ModelSelected::Plain {
            model_id,
            tokenizer_json,
            arch,
            dtype: _,
            topology,
            organization,
            write_uqff,
            from_uqff,
            imatrix,
            calibration_file,
            max_seq_len: _,
            max_batch_size: _,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
        } => {
            let options = SafetensorsOptions {
                topology: Topology::from_option_path(topology)?,
                organization: organization.unwrap_or_default(),
                write_uqff,
                from_uqff: uqff_paths(from_uqff),
                imatrix,
                calibration_file,
                hf_cache_path,
                matformer_config_path,
                matformer_slice_name,
                ..base.clone()
            };
            NormalLoaderBuilder::new(
                options.normal(),
                args.chat_template,
                tokenizer_json,
                Some(model_id),
                args.no_kv_cache,
                args.jinja_explicit,
            )
            .with_mtp(args.mtp)
            .build(arch)?
        }
        ModelSelected::Run {
            model_id,
            tokenizer_json,
            dtype: _,
            topology,
            organization,
            write_uqff,
            from_uqff,
            imatrix,
            calibration_file,
            max_edge,
            max_seq_len: _,
            max_batch_size: _,
            max_num_images: _,
            max_image_length: _,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
        } => {
            let options = SafetensorsOptions {
                topology: Topology::from_option_path(topology)?,
                organization: organization.unwrap_or_default(),
                write_uqff,
                from_uqff: uqff_paths(from_uqff),
                imatrix,
                calibration_file,
                hf_cache_path: hf_cache_path.clone(),
                matformer_config_path,
                matformer_slice_name,
                ..base.clone()
            };
            let builder = AutoLoaderBuilder::new(
                options.normal(),
                options.multimodal(max_edge),
                options.embedding(),
                args.chat_template,
                tokenizer_json,
                model_id,
                args.no_kv_cache,
                args.jinja_explicit,
            );
            let builder = if let Some(ref path) = hf_cache_path {
                builder.hf_cache_path(path.clone())
            } else {
                builder
            };
            builder
                .with_mtp(args.mtp)
                .with_encoder_cache_memory_bytes(args.encoder_cache_memory_bytes)
                .build()
        }
        ModelSelected::MultimodalPlain {
            model_id,
            tokenizer_json,
            arch,
            dtype: _,
            topology,
            write_uqff,
            from_uqff,
            max_edge,
            calibration_file,
            max_seq_len: _,
            max_batch_size: _,
            max_num_images: _,
            max_image_length: _,
            hf_cache_path,
            imatrix,
            matformer_config_path,
            matformer_slice_name,
            organization,
        } => {
            let options = SafetensorsOptions {
                topology: Topology::from_option_path(topology)?,
                organization: organization.unwrap_or_default(),
                write_uqff,
                from_uqff: uqff_paths(from_uqff),
                imatrix,
                calibration_file,
                hf_cache_path,
                matformer_config_path,
                matformer_slice_name,
                ..base.clone()
            };
            MultimodalLoaderBuilder::new(
                options.multimodal(max_edge),
                args.chat_template,
                tokenizer_json,
                Some(model_id),
                args.jinja_explicit,
            )
            .with_mtp(args.mtp)
            .with_encoder_cache_memory_bytes(args.encoder_cache_memory_bytes)
            .build(arch)?
        }
        ModelSelected::DiffusionPlain {
            model_id,
            arch,
            dtype: _,
        } => DiffusionLoaderBuilder::new(Some(model_id)).build(arch),
        ModelSelected::Speech {
            model_id,
            dac_model_id,
            arch,
            ..
        } => Box::new(SpeechLoader {
            model_id,
            dac_model_id,
            arch,
            cfg: None,
        }),
        ModelSelected::XLora {
            model_id,
            xlora_model_id,
            order,
            tokenizer_json,
            tgt_non_granular_index,
            arch,
            dtype: _,
            topology,
            write_uqff,
            from_uqff,
            max_seq_len: _,
            max_batch_size: _,
            hf_cache_path,
        } => {
            let options = SafetensorsOptions {
                topology: Topology::from_option_path(topology)?,
                write_uqff,
                from_uqff: uqff_paths(from_uqff),
                hf_cache_path,
                ..base.clone()
            };
            NormalLoaderBuilder::new(
                options.normal(),
                args.chat_template,
                tokenizer_json,
                model_id,
                args.no_kv_cache,
                args.jinja_explicit,
            )
            .with_xlora(
                xlora_model_id,
                load_ordering(&order)?,
                args.no_kv_cache,
                tgt_non_granular_index,
            )
            .build(arch)?
        }
        ModelSelected::Lora {
            model_id,
            tokenizer_json,
            adapters,
            runtime_config,
            arch,
            dtype: _,
            topology,
            organization,
            write_uqff,
            from_uqff,
            imatrix,
            calibration_file,
            max_edge,
            max_seq_len: _,
            max_batch_size: _,
            max_num_images: _,
            max_image_length: _,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
        } => {
            let options = SafetensorsOptions {
                topology: Topology::from_option_path(topology)?,
                organization: organization.unwrap_or_default(),
                write_uqff,
                from_uqff: uqff_paths(from_uqff),
                imatrix,
                calibration_file,
                hf_cache_path: hf_cache_path.clone(),
                matformer_config_path,
                matformer_slice_name,
                ..base.clone()
            };
            if let Some(arch) = arch {
                NormalLoaderBuilder::new(
                    options.normal(),
                    args.chat_template,
                    tokenizer_json,
                    Some(model_id),
                    args.no_kv_cache,
                    args.jinja_explicit,
                )
                .with_lora(adapters, runtime_config)
                .build(Some(arch))?
            } else {
                let builder = AutoLoaderBuilder::new(
                    options.normal(),
                    options.multimodal(max_edge),
                    options.embedding(),
                    args.chat_template,
                    tokenizer_json,
                    model_id,
                    args.no_kv_cache,
                    args.jinja_explicit,
                )
                .with_lora(adapters, runtime_config);
                if let Some(path) = hf_cache_path {
                    builder.hf_cache_path(path).build()
                } else {
                    builder.build()
                }
            }
        }
        ModelSelected::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            tokenizer_json,
            mmproj_filename,
            lora_adapters,
            lora_runtime_config,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_edge,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            ..
        } => {
            if lora_runtime_config.is_none() && !lora_adapters.is_empty() {
                anyhow::bail!("GGUF LoRA adapters require a dynamic LoRA runtime configuration");
            }
            let mut builder = GGUFLoaderBuilder::new(
                args.chat_template,
                tok_model_id,
                quantized_model_id,
                gguf_files(&quantized_filename),
                GGUFSpecificConfig {
                    topology: Topology::from_option_path(topology)?,
                    organization: organization.unwrap_or_default(),
                    write_uqff,
                    imatrix,
                    calibration_file,
                    max_edge,
                    max_model_len: args.max_model_len,
                    hf_cache_path,
                    matformer_config_path,
                    matformer_slice_name,
                },
                args.no_kv_cache,
                args.jinja_explicit,
            )
            .with_encoder_cache_memory_bytes(args.encoder_cache_memory_bytes);
            if let Some(mmproj_filename) = mmproj_filename {
                builder = builder.with_mmproj_files(gguf_files(&mmproj_filename));
            }
            if let Some(tokenizer_json) = tokenizer_json {
                builder = builder.with_tokenizer_json(tokenizer_json);
            }
            if let Some(runtime_config) = lora_runtime_config {
                builder = builder.with_dynamic_lora(lora_adapters, runtime_config);
            }
            builder.build()
        }
        ModelSelected::XLoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            topology,
            ..
        } => GGUFLoaderBuilder::new(
            args.chat_template,
            tok_model_id,
            quantized_model_id,
            gguf_files(&quantized_filename),
            GGUFSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                ..Default::default()
            },
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_encoder_cache_memory_bytes(args.encoder_cache_memory_bytes)
        .with_xlora(
            xlora_model_id,
            load_ordering(&order)?,
            args.no_kv_cache,
            tgt_non_granular_index,
        )
        .build(),
        ModelSelected::LoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            order,
            topology,
            ..
        } => GGUFLoaderBuilder::new(
            args.chat_template,
            tok_model_id,
            quantized_model_id,
            gguf_files(&quantized_filename),
            GGUFSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                ..Default::default()
            },
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_encoder_cache_memory_bytes(args.encoder_cache_memory_bytes)
        .with_lora(adapters_model_id, load_ordering(&order)?)
        .build(),
        ModelSelected::GGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            gqa,
            topology,
            ..
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(),
        ModelSelected::XLoraGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            gqa,
            topology,
            ..
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_xlora(
            xlora_model_id,
            load_ordering(&order)?,
            args.no_kv_cache,
            tgt_non_granular_index,
        )
        .build(),
        ModelSelected::LoraGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            order,
            gqa,
            topology,
            ..
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_lora(adapters_model_id, load_ordering(&order)?)
        .build(),
        ModelSelected::Embedding {
            model_id,
            tokenizer_json,
            arch,
            dtype: _,
            topology,
            write_uqff,
            from_uqff,
            imatrix,
            calibration_file,
            hf_cache_path,
        } => {
            let options = SafetensorsOptions {
                topology: Topology::from_option_path(topology)?,
                write_uqff,
                from_uqff: uqff_paths(from_uqff),
                imatrix,
                calibration_file,
                hf_cache_path,
                ..base.clone()
            };
            EmbeddingLoaderBuilder::new(options.embedding(), tokenizer_json, Some(model_id))
                .build(arch)
        }
        ModelSelected::MultiModel { .. } => {
            anyhow::bail!("MultiModel variant should not be used in model loading functions")
        }
    };
    Ok(loader)
}

#[cfg(test)]
mod tests {
    use super::*;

    const XLORA_ORDERING: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/orderings/xlora-paper-ordering.json"
    );

    fn selected(json: serde_json::Value) -> ModelSelected {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn multi_file_arguments_split_on_their_delimiters() {
        let uqff = format!("a.uqff{UQFF_MULTI_FILE_DELIMITER}b.uqff");
        assert_eq!(
            uqff_paths(Some(uqff)),
            Some(vec![PathBuf::from("a.uqff"), PathBuf::from("b.uqff")])
        );
        assert_eq!(uqff_paths(None), None);
        let gguf = format!("x-00001.gguf{GGUF_MULTI_FILE_DELIMITER}x-00002.gguf");
        assert_eq!(gguf_files(&gguf), vec!["x-00001.gguf", "x-00002.gguf"]);
    }

    #[test]
    fn a_missing_ordering_file_is_an_error_not_a_panic() {
        let err = load_ordering("/nonexistent/ordering.json").unwrap_err();
        assert!(
            err.to_string().contains("/nonexistent/ordering.json"),
            "{err}"
        );
        assert!(load_ordering(XLORA_ORDERING).is_ok());
    }

    #[test]
    fn plain_selection_builds_a_loader_for_its_model() -> anyhow::Result<()> {
        let loader = LoaderBuilder::new(selected(
            serde_json::json!({"Plain": {"model_id": "org/model"}}),
        ))
        .build()?;
        assert_eq!(loader.get_id(), "org/model");
        Ok(())
    }

    fn loader_config(
        model: ModelSelected,
        max_model_len: Option<usize>,
    ) -> crate::ModelLoaderConfig {
        crate::ModelLoaderConfig {
            model_selected: model,
            token_source: crate::TokenSource::None,
            hf_revision: None,
            dtype: ModelDType::Auto,
            device: candle_core::Device::Cpu,
            device_map_setting: crate::DeviceMapSetting::dummy(),
            isq: None,
            paged_attn_config: None,
            silent: true,
            chat_template: None,
            jinja_explicit: None,
            max_model_len,
            hf_config_overrides: None,
            mtp_config: None,
            encoder_cache_memory_bytes: None,
        }
    }

    #[test]
    fn a_stored_loader_config_rebuilds_its_loader() -> anyhow::Result<()> {
        let plain = || selected(serde_json::json!({"Plain": {"model_id": "org/model"}}));
        assert_eq!(
            loader_config(plain(), None).build_loader(false)?.get_id(),
            "org/model"
        );
        // the config's options reach the builder, so reload validates exactly like the first load did
        let err = loader_config(plain(), Some(0))
            .build_loader(false)
            .err()
            .unwrap();
        assert!(err.to_string().contains("greater than zero"), "{err}");
        Ok(())
    }

    #[test]
    fn max_model_len_is_validated_per_format() {
        let plain = || selected(serde_json::json!({"Plain": {"model_id": "m"}}));
        let err = LoaderBuilder::new(plain())
            .with_max_model_len(Some(0))
            .build()
            .err()
            .unwrap();
        assert!(err.to_string().contains("greater than zero"), "{err}");
        let embedding =
            selected(serde_json::json!({"Embedding": {"model_id": "m", "dtype": "auto"}}));
        let err = LoaderBuilder::new(embedding)
            .with_max_model_len(Some(1024))
            .build()
            .err()
            .unwrap();
        assert!(err.to_string().contains("not supported"), "{err}");
        // the legacy X-LoRA GGUF pipeline cannot cap its length, so the flag is refused rather than ignored
        let xlora_gguf = selected(serde_json::json!({"XLoraGGUF": {
            "quantized_model_id": "q",
            "quantized_filename": "q.gguf",
            "xlora_model_id": "x",
            "order": XLORA_ORDERING,
            "dtype": "auto",
            "max_seq_len": 4096,
            "max_batch_size": 1,
        }}));
        let err = LoaderBuilder::new(xlora_gguf)
            .with_max_model_len(Some(1024))
            .build()
            .err()
            .unwrap();
        assert!(err.to_string().contains("not supported"), "{err}");
    }
}
