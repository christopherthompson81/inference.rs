use std::path::PathBuf;

use inference_core::{
    AutoDeviceMapParams, DiffusionLoaderType, EmbeddingLoaderType, IsqOrganization,
    LoraAdapterSpec, LoraRuntimeConfig, ModelDType, MultimodalLoaderType, NormalLoaderType,
    SpeechGenerationConfig, SpeechLoaderType, TranscriptionLoaderType, UqffWriteConfig,
};

/// Speech sampling for every generation of the loaded model; an unset field keeps the architecture's default.
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechGenerationSpec {
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Classifier-free guidance strength.
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Kokoro's default speaking rate, which a request's `speed` overrides.
    #[serde(default)]
    pub speed: Option<f32>,
}

impl SpeechGenerationSpec {
    pub fn into_config(self, arch: SpeechLoaderType) -> SpeechGenerationConfig {
        match (
            arch,
            SpeechGenerationConfig::dia_default(),
            SpeechGenerationConfig::kokoro_default(),
        ) {
            (
                SpeechLoaderType::Dia,
                SpeechGenerationConfig::Dia {
                    max_tokens,
                    cfg_scale,
                    temperature,
                    top_p,
                    top_k,
                },
                _,
            ) => SpeechGenerationConfig::Dia {
                max_tokens: self.max_tokens.or(max_tokens),
                cfg_scale: self.cfg_scale.unwrap_or(cfg_scale),
                temperature: self.temperature.unwrap_or(temperature),
                top_p: self.top_p.unwrap_or(top_p),
                top_k: self.top_k.or(top_k),
            },
            (SpeechLoaderType::Kokoro, _, SpeechGenerationConfig::Kokoro { speed }) => {
                SpeechGenerationConfig::Kokoro {
                    speed: self.speed.unwrap_or(speed),
                }
            }
            (_, dia, kokoro) => unreachable!("defaults are {dia:?} and {kokoro:?}"),
        }
    }
}

// Default value functions for serde deserialization
fn default_model_dtype() -> ModelDType {
    ModelDType::Auto
}

fn default_max_seq_len() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN
}

fn default_max_batch_size() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE
}

fn default_max_num_images() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES
}

fn default_max_image_length() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH
}

/// How a GGUF spec without `mmproj_filename` gets its multimodal projector.
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MmprojSelection {
    /// Only `mmproj_filename`, except that a `quant` resolved in a GGUF artifact repo also picks its projector.
    #[default]
    Given,
    /// The projector a GGUF artifact repo publishes; a repo that also holds other weights gets none.
    ArtifactRepo,
    /// Any projector published beside the model file.
    Any,
    /// Any projector beside the model file, failing when there is none: the model is multimodal.
    Required,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum ModelSelected {
    /// Select a model for running via auto loader
    Run {
        /// Model ID to load from. May be a HF hub repo or a local path.
        model_id: String,

        /// A quantization level (`4`, `q4k`, `auto`) resolved against what the repository publishes: one of its
        /// GGUF files, else a prebuilt UQFF, else ISQ at that level. `Engine::load` resolves it before loading.
        #[serde(default)]
        quant: Option<String>,

        /// Path to local tokenizer.json file. If specified, it is used over any remote file.
        tokenizer_json: Option<String>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe`.
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<inference_core::UqffWriteSpec>))]
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
        /// Only supported on specific multimodal models.
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_seq_len))]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_batch_size))]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for this model. This affects automatic device mapping but is not a hard limit.
        /// Only supported on specific multimodal models.
        max_num_images: Option<usize>,

        /// Maximum expected image size will have this edge length on both edges.
        /// This affects automatic device mapping but is not a hard limit.
        /// Only supported on specific multimodal models.
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        matformer_slice_name: Option<String>,
    },

    /// Select a plain model, without quantization or adapters
    Plain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// A quantization level (`4`, `q4k`, `auto`) resolved against what the repository publishes, as for `Run`.
        #[serde(default)]
        quant: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[serde(default)]
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        #[serde(default)]
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[serde(default)]
        topology: Option<String>,

        #[allow(rustdoc::bare_urls)]
        /// ISQ organization: `default` or `moqe` (Mixture of Quantized Experts: <https://arxiv.org/abs/2310.02410>).
        #[serde(default)]
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<inference_core::UqffWriteSpec>))]
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;)
        #[serde(default)]
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        /// Incompatible with `--calibration-file/-c`
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        /// Incompatible with `--imatrix/-i`
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        calibration_file: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_seq_len))]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_batch_size))]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        #[serde(default)]
        matformer_slice_name: Option<String>,
    },

    /// Select a LoRA architecture
    Lora {
        /// Base model ID. This may be a Hugging Face repository or a local path.
        model_id: String,

        /// A quantization level (`4`, `q4k`, `auto`) resolved against what the repository publishes, as for `Run`.
        #[serde(default)]
        quant: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        tokenizer_json: Option<String>,

        /// LoRA adapters to preload, each formatted as ALIAS=SOURCE.
        #[serde(default)]
        adapters: Vec<LoraAdapterSpec>,

        /// Dynamic LoRA runtime capacity and rank limits.
        #[serde(default)]
        runtime_config: LoraRuntimeConfig,

        /// How a GGUF that `quant` resolves to gets its projector.
        #[serde(default)]
        mmproj_selection: MmprojSelection,

        /// The architecture of the model.
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe`.
        #[serde(default)]
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<inference_core::UqffWriteSpec>))]
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length.
        #[serde(default)]
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_seq_len))]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_batch_size))]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for automatic device mapping.
        #[serde(default)]
        max_num_images: Option<usize>,

        /// Maximum expected image edge length for automatic device mapping.
        #[serde(default)]
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file.
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use.
        #[serde(default)]
        matformer_slice_name: Option<String>,
    },

    /// Select a GGUF model.
    GGUF {
        /// `tok_model_id` optionally overrides embedded configuration and tokenizer assets.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use semicolons to separate multiple files. Leave empty for `quant` to pick one.
        #[serde(default)]
        quantized_filename: String,

        /// A GGUF quantization level (`4`, `q4k`) to pick `quantized_filename` from the repository's GGUF files.
        #[serde(default)]
        quant: Option<String>,

        /// Path to a tokenizer JSON file.
        tokenizer_json: Option<String>,

        /// Multimodal projector filename(s), separated by semicolons.
        mmproj_filename: Option<String>,

        /// How to pick a projector when `mmproj_filename` is unset.
        #[serde(default)]
        mmproj_selection: MmprojSelection,

        /// Dynamic LoRA adapters to preload.
        #[serde(default)]
        lora_adapters: Vec<LoraAdapterSpec>,

        /// Dynamic LoRA runtime limits. `None` disables dynamic LoRA.
        #[serde(default)]
        lora_runtime_config: Option<LoraRuntimeConfig>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe`.
        organization: Option<IsqOrganization>,

        /// UQFF path and quantization types to write.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<inference_core::UqffWriteSpec>))]
        write_uqff: Option<UqffWriteConfig>,

        /// Imatrix file to use while requantizing the GGUF weights.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        imatrix: Option<PathBuf>,

        /// Calibration file used to generate an imatrix while requantizing the GGUF weights.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length.
        #[serde(default)]
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_seq_len))]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_batch_size))]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for automatic device mapping.
        #[serde(default)]
        max_num_images: Option<usize>,

        /// Maximum expected image edge length for automatic device mapping.
        #[serde(default)]
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally.
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        hf_cache_path: Option<PathBuf>,

        /// Path to a local Matryoshka Transformer configuration CSV file.
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use.
        #[serde(default)]
        matformer_slice_name: Option<String>,
    },

    /// Select a GGML model.
    GGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename.
        quantized_filename: String,

        /// GQA value
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_seq_len))]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_batch_size))]
        max_batch_size: usize,
    },

    /// Select a multimodal plain model, without quantization or adapters
    MultimodalPlain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// A quantization level (`4`, `q4k`, `auto`) resolved against what the repository publishes, as for `Run`.
        #[serde(default)]
        quant: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        arch: Option<MultimodalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// UQFF path to write to.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<inference_core::UqffWriteSpec>))]
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        from_uqff: Option<String>,

        /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
        /// This is only supported on the Qwen2-VL and Idefics models. Others handle this internally.
        max_edge: Option<u32>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        calibration_file: Option<PathBuf>,

        /// .cimatrix file to enhance GGUF quantizations with. This must be a .cimatrix file.
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        imatrix: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_seq_len))]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_batch_size))]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_num_images")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_num_images))]
        max_num_images: usize,

        /// Maximum expected image size will have this edge length on both edges.
        /// This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_image_length")]
        #[cfg_attr(feature = "utoipa", schema(default = default_max_image_length))]
        max_image_length: usize,

        /// Cache path for Hugging Face models downloaded locally
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        matformer_slice_name: Option<String>,

        /// ISQ organization: `default` or `moqe` (Mixture of Quantized Experts: <https://arxiv.org/abs/2310.02410>).
        organization: Option<IsqOrganization>,
    },

    /// Select a diffusion model, without quantization or adapters
    DiffusionPlain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// The architecture of the model.
        arch: DiffusionLoaderType,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,
    },

    Speech {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// DAC Model ID to load from. If not provided, this is automatically downloaded from the default path for the model.
        /// This may be a HF hub repo or a local path.
        dac_model_id: Option<String>,

        /// The architecture of the model; unset reads it from the model's `config.json`.
        #[serde(default)]
        arch: Option<SpeechLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,

        /// Load-time sampling for every request; it is specific to an architecture, so it needs `arch` set.
        #[serde(default)]
        generation: Option<SpeechGenerationSpec>,
    },

    /// Select a speech recognition model
    Transcription {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// The architecture of the model; unset reads it from the model's `config.json`.
        #[serde(default)]
        arch: Option<TranscriptionLoaderType>,

        /// A Silero VAD GGUF (file, directory or HF repo) that cuts long recordings at their silences; without it
        /// Parakeet transcribes at most 24 minutes.
        #[serde(default)]
        vad_model_id: Option<String>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,
    },

    /// Select a speaker diarization model (Nemotron-3 Diarization, or a Streaming Sortformer `.nemo`)
    Diarization {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// Model data type. Defaults to `auto`, which is F32 here.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,
    },

    /// Select a voice activity detection model: a Silero VAD GGUF file, a directory or HF repo holding one
    VoiceActivity {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,
    },

    /// Select an embedding model, without quantization or adapters
    Embedding {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// A quantization level (`4`, `q4k`, `auto`) resolved against what the repository publishes, as for `Run`.
        #[serde(default)]
        quant: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[serde(default)]
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        #[serde(default)]
        arch: Option<EmbeddingLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        #[cfg_attr(feature = "utoipa", schema(default = default_model_dtype))]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[serde(default)]
        topology: Option<String>,

        /// UQFF path to write to.
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<inference_core::UqffWriteSpec>))]
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;)
        #[serde(default)]
        from_uqff: Option<String>,

        /// imatrix file for enhanced quantization.
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        imatrix: Option<PathBuf>,

        /// Calibration file for imatrix generation.
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        calibration_file: Option<PathBuf>,

        /// Cache path for Hugging Face models downloaded locally
        #[serde(default)]
        #[cfg_attr(feature = "utoipa", schema(value_type = Option<String>))]
        hf_cache_path: Option<PathBuf>,
    },
}

impl ModelSelected {
    pub fn dtype(&self) -> ModelDType {
        match self {
            Self::Run { dtype, .. }
            | Self::Plain { dtype, .. }
            | Self::Lora { dtype, .. }
            | Self::GGUF { dtype, .. }
            | Self::GGML { dtype, .. }
            | Self::MultimodalPlain { dtype, .. }
            | Self::DiffusionPlain { dtype, .. }
            | Self::Speech { dtype, .. }
            | Self::Transcription { dtype, .. }
            | Self::Embedding { dtype, .. } => *dtype,
            // an F32 checkpoint whose speaker decisions move in BF16, for no speedup
            Self::Diarization {
                dtype: ModelDType::Auto,
                ..
            } => ModelDType::F32,
            Self::Diarization { dtype, .. } => *dtype,
            // the VAD runs in F32 whatever is asked
            Self::VoiceActivity { .. } => ModelDType::F32,
        }
    }

    /// Where Hugging Face downloads go; `None` for the default cache or kinds that take no cache path.
    pub fn hf_cache_path(&self) -> Option<&PathBuf> {
        match self {
            Self::Run { hf_cache_path, .. }
            | Self::Plain { hf_cache_path, .. }
            | Self::Lora { hf_cache_path, .. }
            | Self::MultimodalPlain { hf_cache_path, .. }
            | Self::Embedding { hf_cache_path, .. }
            | Self::GGUF { hf_cache_path, .. } => hf_cache_path.as_ref(),
            Self::GGML { .. }
            | Self::DiffusionPlain { .. }
            | Self::Speech { .. }
            | Self::Transcription { .. }
            | Self::VoiceActivity { .. }
            | Self::Diarization { .. } => None,
        }
    }

    /// The (max_seq_len, max_batch_size) device mapping sizes for; `None` for kinds sized by their defaults.
    pub fn sequence_limits(&self) -> Option<(usize, usize)> {
        match self {
            Self::Run {
                max_seq_len,
                max_batch_size,
                ..
            }
            | Self::Plain {
                max_seq_len,
                max_batch_size,
                ..
            }
            | Self::Lora {
                max_seq_len,
                max_batch_size,
                ..
            }
            | Self::GGUF {
                max_seq_len,
                max_batch_size,
                ..
            }
            | Self::GGML {
                max_seq_len,
                max_batch_size,
                ..
            }
            | Self::MultimodalPlain {
                max_seq_len,
                max_batch_size,
                ..
            } => Some((*max_seq_len, *max_batch_size)),
            Self::DiffusionPlain { .. }
            | Self::Speech { .. }
            | Self::Transcription { .. }
            | Self::VoiceActivity { .. }
            | Self::Diarization { .. }
            | Self::Embedding { .. } => None,
        }
    }

    /// Where the spec writes a UQFF of the weights it loads; `None` for kinds that cannot write one.
    pub fn write_uqff_mut(&mut self) -> Option<&mut Option<UqffWriteConfig>> {
        match self {
            Self::Run { write_uqff, .. }
            | Self::Plain { write_uqff, .. }
            | Self::Lora { write_uqff, .. }
            | Self::GGUF { write_uqff, .. }
            | Self::MultimodalPlain { write_uqff, .. }
            | Self::Embedding { write_uqff, .. } => Some(write_uqff),
            Self::GGML { .. }
            | Self::DiffusionPlain { .. }
            | Self::Speech { .. }
            | Self::Transcription { .. }
            | Self::VoiceActivity { .. }
            | Self::Diarization { .. } => None,
        }
    }

    /// The quantization level the spec asks resolution to pick weights for.
    pub fn quant(&self) -> Option<&str> {
        match self {
            Self::Run { quant, .. }
            | Self::Plain { quant, .. }
            | Self::Lora { quant, .. }
            | Self::MultimodalPlain { quant, .. }
            | Self::Embedding { quant, .. }
            | Self::GGUF { quant, .. } => quant.as_deref(),
            _ => None,
        }
    }

    /// Whether `selection::quant::resolve_model_source` still has files or a projector to pick before loading.
    pub fn needs_source_resolution(&self) -> bool {
        self.quant().is_some()
            || matches!(
                self,
                Self::GGUF { quantized_filename, mmproj_filename, mmproj_selection, .. }
                    if quantized_filename.is_empty()
                        || (mmproj_filename.is_none() && *mmproj_selection != MmprojSelection::Given)
            )
    }
}

#[cfg(test)]
mod speech_tests {
    use super::*;

    #[test]
    fn a_speech_generation_spec_overrides_only_what_it_sets() {
        let spec: SpeechGenerationSpec =
            serde_json::from_value(serde_json::json!({"temperature": 0.5, "max_tokens": 64}))
                .unwrap();
        let SpeechGenerationConfig::Dia {
            max_tokens,
            cfg_scale,
            temperature,
            top_k,
            ..
        } = spec.clone().into_config(SpeechLoaderType::Dia)
        else {
            panic!("Dia gets a Dia config")
        };
        let SpeechGenerationConfig::Dia {
            cfg_scale: default_scale,
            top_k: default_top_k,
            ..
        } = SpeechGenerationConfig::dia_default()
        else {
            panic!("Dia's default is a Dia config")
        };
        assert_eq!((max_tokens, temperature), (Some(64), 0.5));
        assert_eq!((cfg_scale, top_k), (default_scale, default_top_k));
        assert!(matches!(
            spec.into_config(SpeechLoaderType::Kokoro),
            SpeechGenerationConfig::Kokoro { speed } if speed == 1.
        ));
        let fast: SpeechGenerationSpec =
            serde_json::from_value(serde_json::json!({"speed": 1.25})).unwrap();
        assert!(matches!(
            fast.into_config(SpeechLoaderType::Kokoro),
            SpeechGenerationConfig::Kokoro { speed } if speed == 1.25
        ));
    }
}

#[cfg(all(test, feature = "utoipa"))]
mod tests {
    use serde::{Serialize, de::DeserializeOwned};
    use strum::IntoEnumIterator;

    use inference_core::{
        DiarizationLoaderType, DiffusionLoaderType, EmbeddingLoaderType, ModelDType,
        MultimodalLoaderType, NormalLoaderType, SpeechLoaderType, TranscriptionLoaderType,
    };

    // The names a schema publishes must be the ones serde writes, and each must parse back.
    fn assert_schema_names_are_serde_names<T>()
    where
        T: utoipa::PartialSchema + IntoEnumIterator + Serialize + DeserializeOwned,
    {
        let schema = serde_json::to_value(T::schema()).unwrap();
        let published: Vec<serde_json::Value> = schema["enum"].as_array().unwrap().clone();
        let serde: Vec<serde_json::Value> = T::iter()
            .map(|v| serde_json::to_value(v).unwrap())
            .collect();
        assert_eq!(published, serde, "{}", std::any::type_name::<T>());
        for name in published {
            serde_json::from_value::<T>(name).unwrap();
        }
    }

    #[test]
    fn loader_and_dtype_schemas_list_the_names_serde_accepts() {
        assert_schema_names_are_serde_names::<NormalLoaderType>();
        assert_schema_names_are_serde_names::<MultimodalLoaderType>();
        assert_schema_names_are_serde_names::<EmbeddingLoaderType>();
        assert_schema_names_are_serde_names::<DiffusionLoaderType>();
        assert_schema_names_are_serde_names::<SpeechLoaderType>();
        assert_schema_names_are_serde_names::<TranscriptionLoaderType>();
        assert_schema_names_are_serde_names::<DiarizationLoaderType>();
        let dtype = serde_json::to_value(<ModelDType as utoipa::PartialSchema>::schema()).unwrap();
        for name in dtype["enum"].as_array().unwrap() {
            serde_json::from_value::<ModelDType>(name.clone()).unwrap();
        }
    }
}
