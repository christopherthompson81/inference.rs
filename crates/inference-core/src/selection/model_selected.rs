use std::path::PathBuf;

use crate::{
    pipeline::{
        AutoDeviceMapParams, EmbeddingLoaderType, IsqOrganization, MultimodalLoaderType,
        NormalLoaderType, UqffWriteConfig,
    },
    DiffusionLoaderType, LoraAdapterSpec, LoraRuntimeConfig, ModelDType, SpeechLoaderType,
};

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

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum ModelSelected {
    /// Select a model for running via auto loader
    Run {
        /// Model ID to load from. May be a HF hub repo or a local path.
        model_id: String,

        /// Path to local tokenizer.json file. If specified, it is used over any remote file.
        tokenizer_json: Option<String>,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe`.
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
        /// Only supported on specific multimodal models.
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for this model. This affects automatic device mapping but is not a hard limit.
        /// Only supported on specific multimodal models.
        max_num_images: Option<usize>,

        /// Maximum expected image size will have this edge length on both edges.
        /// This affects automatic device mapping but is not a hard limit.
        /// Only supported on specific multimodal models.
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally.
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        matformer_slice_name: Option<String>,
    },

    /// Select a plain model, without quantization or adapters
    Plain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[serde(default)]
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        #[serde(default)]
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
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
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;)
        #[serde(default)]
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        /// Incompatible with `--calibration-file/-c`
        #[serde(default)]
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        /// Incompatible with `--imatrix/-i`
        #[serde(default)]
        calibration_file: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        #[serde(default)]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        #[serde(default)]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        #[serde(default)]
        matformer_slice_name: Option<String>,
    },

    /// Select an X-LoRA architecture
    XLora {
        /// Force a base model ID to load from instead of using the ordering file. This may be a HF hub repo or a local path.
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
        xlora_model_id: String,

        /// Ordering JSON file
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        tgt_non_granular_index: Option<usize>,

        /// The architecture of the model.
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// UQFF path to write to.
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        from_uqff: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        hf_cache_path: Option<PathBuf>,
    },

    /// Select a LoRA architecture
    Lora {
        /// Base model ID. This may be a Hugging Face repository or a local path.
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        tokenizer_json: Option<String>,

        /// LoRA adapters to preload, each formatted as ALIAS=SOURCE.
        adapters: Vec<LoraAdapterSpec>,

        /// Dynamic LoRA runtime capacity and rank limits.
        runtime_config: LoraRuntimeConfig,

        /// The architecture of the model.
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe`.
        #[serde(default)]
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        #[serde(default)]
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        #[serde(default)]
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length.
        #[serde(default)]
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for automatic device mapping.
        #[serde(default)]
        max_num_images: Option<usize>,

        /// Maximum expected image edge length for automatic device mapping.
        #[serde(default)]
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file.
        #[serde(default)]
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
        /// May be a single filename, or use semicolons to separate multiple files.
        quantized_filename: String,

        /// Path to a tokenizer JSON file.
        tokenizer_json: Option<String>,

        /// Multimodal projector filename(s), separated by semicolons.
        mmproj_filename: Option<String>,

        /// Dynamic LoRA adapters to preload.
        #[serde(default)]
        lora_adapters: Vec<LoraAdapterSpec>,

        /// Dynamic LoRA runtime limits. `None` disables dynamic LoRA.
        #[serde(default)]
        lora_runtime_config: Option<LoraRuntimeConfig>,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe`.
        organization: Option<IsqOrganization>,

        /// UQFF path and quantization types to write.
        write_uqff: Option<UqffWriteConfig>,

        /// Imatrix file to use while requantizing the GGUF weights.
        imatrix: Option<PathBuf>,

        /// Calibration file used to generate an imatrix while requantizing the GGUF weights.
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length.
        #[serde(default)]
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for automatic device mapping.
        #[serde(default)]
        max_num_images: Option<usize>,

        /// Maximum expected image edge length for automatic device mapping.
        #[serde(default)]
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally.
        #[serde(default)]
        hf_cache_path: Option<PathBuf>,

        /// Path to a local Matryoshka Transformer configuration CSV file.
        #[serde(default)]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use.
        #[serde(default)]
        matformer_slice_name: Option<String>,
    },

    /// Select a GGUF model with X-LoRA.
    XLoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use semicolons to separate multiple files.
        quantized_filename: String,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
        xlora_model_id: String,

        /// Ordering JSON file
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        tgt_non_granular_index: Option<usize>,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,
    },

    /// Select a GGUF model with LoRA.
    LoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use semicolons to separate multiple files.
        quantized_filename: String,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        adapters_model_id: String,

        /// Ordering JSON file
        order: String,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,
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
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,
    },

    /// Select a GGML model with X-LoRA.
    XLoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename.
        quantized_filename: String,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
        xlora_model_id: String,

        /// Ordering JSON file
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        tgt_non_granular_index: Option<usize>,

        /// GQA value
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,
    },

    /// Select a GGML model with LoRA.
    LoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename.
        quantized_filename: String,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        adapters_model_id: String,

        /// Ordering JSON file
        order: String,

        /// GQA value
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,
    },

    /// Select a multimodal plain model, without quantization or adapters
    MultimodalPlain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        arch: Option<MultimodalLoaderType>,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// UQFF path to write to.
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        from_uqff: Option<String>,

        /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
        /// This is only supported on the Qwen2-VL and Idefics models. Others handle this internally.
        max_edge: Option<u32>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        calibration_file: Option<PathBuf>,

        /// .cimatrix file to enhance GGUF quantizations with. This must be a .cimatrix file.
        imatrix: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for this model. This affects automatic device mapping but is not a hard limit.
        max_num_images: usize,

        /// Maximum expected image size will have this edge length on both edges.
        /// This affects automatic device mapping but is not a hard limit.
        max_image_length: usize,

        /// Cache path for Hugging Face models downloaded locally
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
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
        dtype: ModelDType,
    },

    Speech {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// DAC Model ID to load from. If not provided, this is automatically downloaded from the default path for the model.
        /// This may be a HF hub repo or a local path.
        dac_model_id: Option<String>,

        /// The architecture of the model.
        arch: SpeechLoaderType,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,
    },

    /// Select multi-model mode with configuration file
    MultiModel {
        /// Multi-model configuration file path (JSON format)
        config: String,

        /// Default model ID to use when no model is specified in requests
        default_model_id: Option<String>,
    },

    /// Select an embedding model, without quantization or adapters
    Embedding {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[serde(default)]
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        #[serde(default)]
        arch: Option<EmbeddingLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_model_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[serde(default)]
        topology: Option<String>,

        /// UQFF path to write to.
        #[serde(default)]
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;)
        #[serde(default)]
        from_uqff: Option<String>,

        /// imatrix file for enhanced quantization.
        #[serde(default)]
        imatrix: Option<PathBuf>,

        /// Calibration file for imatrix generation.
        #[serde(default)]
        calibration_file: Option<PathBuf>,

        /// Cache path for Hugging Face models downloaded locally
        #[serde(default)]
        hf_cache_path: Option<PathBuf>,
    },
}
