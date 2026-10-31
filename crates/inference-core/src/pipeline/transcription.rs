use super::{
    AdapterPaths, AnyMoePipelineMixin, Cache, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, InputProcessorOutput, InputsProcessor, InputsProcessorType, IsqPipelineMixin,
    Loader, MessagesAction, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, Processor, TokenSource, TranscriptionOutcome,
};
use crate::device_map::DeviceMapper;
use crate::paged_attention::PagedAttentionMeta;
use crate::pipeline::LoadOptions;
use crate::pipeline::{ChatTemplate, EmbeddingModulePaths, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::progress::ProgressScopeGuard;
use crate::{
    MessageContent, Pipeline, TranscriptionStreamOp, TranscriptionStreamRequest,
    TranscriptionStreamUpdate,
};
use anyhow::Result;
use futures::future::BoxFuture;
use hf_hub::{Repo, RepoType};
use indexmap::IndexMap;
use inference_audio::AudioInput;
use inference_audio::mel::StreamResampler;
use inference_models_speech::diarization::{
    DiarizationOptions, Diarizer, Nemotron3Diarizer, Nemotron3Files, SortformerDiarizer,
    is_sortformer,
};
use inference_models_speech::nemo;
use inference_models_speech::parakeet::{
    self, MODEL_TYPES, Parakeet, ParakeetFiles, ParakeetStream, QuantPolicy, is_asr, is_asr_gguf,
};
use inference_models_speech::silero::{SegmentOptions, SileroVad, is_silero_gguf, read_gguf};
use inference_models_speech::whisper::{self, Whisper, WhisperFiles};
use inference_quant::IsqType;
use inference_tensor::{Device, Tensor};
use rand_isaac::Isaac64Rng;
use serde::Serialize;
use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

const CONFIG: &str = "config.json";
const PROCESSOR_CONFIG: &str = "processor_config.json";
const TOKENIZER: &str = "tokenizer.json";
const WEIGHTS: &str = "model.safetensors";
const WEIGHTS_INDEX: &str = "model.safetensors.index.json";
const GENERATION_CONFIG: &str = "generation_config.json";
const GGUF_EXTENSION: &str = "gguf";
const NEMOTRON3_DIARIZATION_TYPE: &str = "nemotron3_diarization";
// the engine's sequence bookkeeping wants a length; one-shot audio requests never reach it
const METADATA_MAX_SEQ_LEN: usize = 1024;
// an open session holds its caches (about 15 MB for a 0.6B model) until it finishes, closes or idles this long
const STREAM_SESSION_IDLE: Duration = Duration::from_secs(10 * 60);
const MAX_STREAM_SESSIONS: usize = 64;

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, inference_json::FromJson, Serialize, PartialEq, strum::EnumIter)]
pub enum TranscriptionLoaderType {
    #[serde(rename = "parakeet")]
    Parakeet,
    #[serde(rename = "whisper")]
    Whisper,
}

impl FromStr for TranscriptionLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "parakeet" => Ok(Self::Parakeet),
            "whisper" => Ok(Self::Whisper),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `parakeet`, `whisper`."
            )),
        }
    }
}

impl TranscriptionLoaderType {
    /// Detects the architecture from a `config.json`'s `model_type`.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        if whisper::is_whisper_config(config) {
            return Some(Self::Whisper);
        }
        let value: serde_json::Value = serde_json::from_str(config).ok()?;
        let model_type = value.get("model_type")?.as_str()?;
        MODEL_TYPES
            .iter()
            .any(|(name, _)| *name == model_type)
            .then_some(Self::Parakeet)
    }
}

// a recogniser in the transformers layout, one `.nemo`, or one NVIDIA `asr` GGUF
#[derive(Clone, Debug)]
enum AsrCheckpoint {
    Transformers(ParakeetFiles),
    Whisper(WhisperFiles),
    Nemo(PathBuf),
    Gguf(PathBuf),
}

#[derive(Clone, Debug)]
pub struct TranscriptionModelPaths {
    checkpoint: AsrCheckpoint,
    // the VAD's GGUF, resolved with the loader's token when the recogniser came from the Hub
    vad: Option<PathBuf>,
}

impl ModelPaths for TranscriptionModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        match &self.checkpoint {
            AsrCheckpoint::Transformers(files) => &files.config,
            AsrCheckpoint::Whisper(files) => &files.config,
            AsrCheckpoint::Nemo(file) | AsrCheckpoint::Gguf(file) => file,
        }
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        match &self.checkpoint {
            AsrCheckpoint::Transformers(files) => &files.tokenizer,
            AsrCheckpoint::Whisper(files) => &files.tokenizer,
            AsrCheckpoint::Nemo(file) | AsrCheckpoint::Gguf(file) => file,
        }
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        match &self.checkpoint {
            AsrCheckpoint::Transformers(files) => &files.weights,
            AsrCheckpoint::Whisper(files) => &files.weights,
            AsrCheckpoint::Nemo(file) | AsrCheckpoint::Gguf(file) => std::slice::from_ref(file),
        }
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        unreachable!("Use `std::any::Any`.")
    }
}

pub struct TranscriptionProcessor;

impl Processor for TranscriptionProcessor {
    fn process(
        &self,
        _pipeline: &dyn Pipeline,
        _messages: Vec<IndexMap<String, MessageContent>>,
        _add_generation_prompt: bool,
        _add_special_tokens: bool,
        _enable_thinking: Option<bool>,
        _reasoning_effort: Option<crate::request::ReasoningEffort>,
        _tools: Vec<crate::Tool>,
    ) -> Result<(Vec<u32>, String)> {
        anyhow::bail!("a transcription model takes audio, not chat messages")
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(TranscriptionInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

pub struct TranscriptionInputsProcessor;

// a sequence's audio, or a streamed session's step, whose samples ride in its request
enum AudioRequest {
    Audio(AudioInput),
    Stream(TranscriptionStreamRequest),
}

impl AudioRequest {
    fn audio(&self) -> &AudioInput {
        match self {
            Self::Audio(audio) => audio,
            Self::Stream(_) => {
                unreachable!("the category check sends streams only to a recogniser")
            }
        }
    }
}

struct ModelInputs {
    audios: Vec<AudioRequest>,
    options: Vec<SegmentOptions>,
    diarization: Vec<DiarizationOptions>,
    transcription: Vec<inference_models_speech::TranscriptionOptions>,
}

impl InputsProcessor for TranscriptionInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Text
    }

    fn process_inputs(
        &self,
        _tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _is_prompt: bool,
        _device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _sliding_window: Option<usize>,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        _mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let audios = input_seqs
            .iter()
            .map(|seq| {
                match (
                    seq.audios().and_then(|a| a.first().cloned()),
                    seq.transcription_stream(),
                ) {
                    (Some(audio), _) => Ok(AudioRequest::Audio(audio)),
                    (None, Some(stream)) => Ok(AudioRequest::Stream(stream.clone())),
                    (None, None) => {
                        Err(anyhow::anyhow!("a transcription request carries no audio"))
                    }
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let options = input_seqs
            .iter()
            .map(|seq| seq.segment_options().cloned().unwrap_or_default())
            .collect();
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                audios,
                options,
                diarization: input_seqs
                    .iter()
                    .map(|seq| seq.diarization_options().cloned().unwrap_or_default())
                    .collect(),
                transcription: input_seqs
                    .iter()
                    .map(|seq| seq.transcription_options().cloned().unwrap_or_default())
                    .collect(),
            }),
            seq_indices: (0..input_seqs.len()).collect::<Vec<_>>(),
        })
    }
}

// the speech analysis models one pipeline hosts: a recogniser (with a VAD for long audio), or a VAD alone
enum AudioModel {
    Parakeet {
        asr: Box<Parakeet>,
        vad: Option<SileroVad>,
    },
    Whisper(Box<Whisper>),
    Silero(SileroVad),
    Diarizer(Diarizer),
}

impl AudioModel {
    fn device(&self) -> &Device {
        match self {
            Self::Parakeet { asr, .. } => asr.device(),
            Self::Whisper(asr) => asr.device(),
            Self::Silero(vad) => vad.device(),
            Self::Diarizer(diarizer) => diarizer.device(),
        }
    }
}

pub struct TranscriptionPipeline {
    model_id: String,
    model: AudioModel,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: EitherCache,
    sessions: HashMap<u64, StreamSession>,
}

// a streamed transcription between its client's requests
struct StreamSession {
    stream: ParakeetStream,
    resampler: StreamResampler,
    last_used: Instant,
}

pub struct TranscriptionLoader {
    pub model_id: String,
    /// Unset reads the architecture from the model's `config.json`.
    pub arch: Option<TranscriptionLoaderType>,
    /// A Silero VAD to cut long recordings at their silences; without one Parakeet takes at most 24 minutes.
    pub vad_model_id: Option<String>,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, inference_json::FromJson, Serialize, PartialEq, strum::EnumIter)]
pub enum DiarizationLoaderType {
    #[serde(rename = "nemotron3")]
    Nemotron3,
    #[serde(rename = "sortformer")]
    Sortformer,
}

impl DiarizationLoaderType {
    /// Detects the architecture from a `config.json`'s `model_type`.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        let value: serde_json::Value = serde_json::from_str(config).ok()?;
        (value.get("model_type")?.as_str()? == NEMOTRON3_DIARIZATION_TYPE)
            .then_some(Self::Nemotron3)
    }
}

/// Loads a speaker diarization model: Nemotron-3 from the transformers layout, or a Streaming Sortformer `.nemo`.
pub struct DiarizationLoader {
    pub model_id: String,
}

impl Loader for DiarizationLoader {
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(options.silent);
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let api = crate::pipeline::hf::build_api(&token_source, !options.silent)?.repo(
            Repo::with_revision(self.model_id.clone(), RepoType::Model, revision.clone()),
        );
        let id = std::path::Path::new(&self.model_id);
        let dtype = options.dtype.try_into_dtype(&[options.device])?;
        let get = |file: &str| crate::pipeline::hf::get_file(&api, id, file, &revision);
        // Nemotron-3's repo carries a `.nemo` beside its transformers files; those win where a config resolves
        let transformers_config = (!id.is_file()).then(|| get(CONFIG).ok()).flatten();
        let diarizer = if let Some(config) = transformers_config {
            let files = Nemotron3Files {
                config,
                processor_config: get(PROCESSOR_CONFIG)?,
                weights: vec![get(WEIGHTS)?],
            };
            let config = std::fs::read_to_string(&files.config)?;
            if DiarizationLoaderType::auto_detect_from_config(&config).is_none() {
                anyhow::bail!(
                    "`{}` is not a Nemotron-3 Diarization config",
                    files.config.display()
                )
            }
            Diarizer::Nemotron3(Box::new(Nemotron3Diarizer::load(
                &files,
                options.device,
                dtype,
            )?))
        } else {
            let file = sortformer_nemo(
                &self.model_id,
                Some(revision),
                &token_source,
                options.silent,
            )?;
            Diarizer::Sortformer(Box::new(SortformerDiarizer::load(
                &file,
                options.device,
                dtype,
            )?))
        };
        Ok(Arc::new(Mutex::new(TranscriptionPipeline::new(
            self.model_id.clone(),
            AudioModel::Diarizer(diarizer),
            dtype,
        ))))
    }

    fn load_model_from_path(
        &self,
        _paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        crate::pipeline::loading::ensure_cuda_kernels(std::slice::from_ref(options.device))?;
        self.load_model_from_hf(None, TokenSource::None, options)
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

/// Loads a Silero VAD GGUF: a file, a directory holding one, or a Hugging Face repo holding one.
pub struct VoiceActivityLoader {
    pub model_id: String,
}

// where a model id's files come from on the Hub
struct HubSource<'a> {
    revision: Option<String>,
    token_source: &'a TokenSource,
    silent: bool,
}

/// The one file `model_id` names with `extension`: the file itself, the one in a local directory `counts` keeps,
/// or the one in a Hugging Face repo.
fn single_file(
    model_id: &str,
    source: HubSource<'_>,
    extension: &str,
    what: &str,
    counts: impl Fn(&std::path::Path) -> bool,
) -> Result<PathBuf> {
    let HubSource {
        revision,
        token_source,
        silent,
    } = source;
    let path = std::path::Path::new(model_id);
    let has_extension = |p: &std::path::Path| p.extension().is_some_and(|e| e == extension);
    let only = |files: Vec<String>| -> Result<String> {
        let mut matching = files
            .into_iter()
            .filter(|f| has_extension(std::path::Path::new(f)));
        match (matching.next(), matching.next()) {
            (Some(only), None) => Ok(only),
            _ => anyhow::bail!("`{model_id}` should hold exactly one {what}"),
        }
    };
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        // only files `counts` keeps, so other models' files beside one do not make the choice ambiguous
        let names = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| counts(p))
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        return Ok(PathBuf::from(only(names)?));
    }
    let revision = revision.unwrap_or_else(|| "main".to_string());
    let api = crate::pipeline::hf::build_api(token_source, !silent)?.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.clone(),
    ));
    let name = only(crate::pipeline::hf::list_repo_files(
        &api, path, true, &revision,
    )?)?;
    crate::pipeline::hf::get_file(&api, path, &name, &revision)
}

/// The Silero GGUF `model_id` names: the file itself, the one in a local directory, or the one in a Hugging Face repo.
pub(crate) fn silero_gguf(
    model_id: &str,
    revision: Option<String>,
    token_source: &TokenSource,
    silent: bool,
) -> Result<PathBuf> {
    let file = single_file(
        model_id,
        HubSource {
            revision,
            token_source,
            silent,
        },
        GGUF_EXTENSION,
        "Silero VAD `.gguf`",
        is_silero_gguf,
    )?;
    if !is_silero_gguf(&file) {
        anyhow::bail!("`{}` is not a Silero VAD GGUF", file.display())
    }
    Ok(file)
}

// the one `.nemo` `model_id` names, found as `silero_gguf` finds its file, which `is` must accept
fn nemo_of_kind(
    model_id: &str,
    source: HubSource<'_>,
    is: fn(&nemo::NemoArchive) -> inference_tensor::Result<bool>,
    what: &str,
) -> Result<PathBuf> {
    let is_nemo = |p: &std::path::Path| p.extension().is_some_and(|e| e == nemo::EXTENSION);
    let file = single_file(model_id, source, nemo::EXTENSION, "`.nemo`", is_nemo)?;
    if !is(&nemo::NemoArchive::open(&file)?)? {
        anyhow::bail!("`{}` is not a {what} `.nemo`", file.display())
    }
    Ok(file)
}

/// The recogniser `.nemo` `model_id` names (Parakeet, hybrid, streaming).
pub(crate) fn asr_nemo(
    model_id: &str,
    revision: Option<String>,
    token_source: &TokenSource,
    silent: bool,
) -> Result<PathBuf> {
    let source = HubSource {
        revision,
        token_source,
        silent,
    };
    nemo_of_kind(model_id, source, is_asr, "speech recognition")
}

/// Writes the recogniser `.nemo` `model_id` names to `out` as an `asr` GGUF under `policy`.
pub fn write_asr_gguf(
    model_id: &str,
    revision: Option<String>,
    token_source: &TokenSource,
    out: &std::path::Path,
    policy: &QuantPolicy,
) -> Result<crate::GgufWriteSummary> {
    let nemo = asr_nemo(model_id, revision, token_source, false)?;
    let chosen = parakeet::write_asr_gguf(&nemo, out, policy)?;
    crate::GgufWriteSummary::of(out, chosen.into_iter().map(|(_, dtype)| dtype))
}

/// The NVIDIA `asr` GGUF `model_id` names (Parakeet, streaming), found as `silero_gguf` finds its file.
pub(crate) fn asr_gguf(
    model_id: &str,
    revision: Option<String>,
    token_source: &TokenSource,
    silent: bool,
) -> Result<PathBuf> {
    let source = HubSource {
        revision,
        token_source,
        silent,
    };
    let file = single_file(model_id, source, GGUF_EXTENSION, "`asr` GGUF", is_asr_gguf)?;
    if !is_asr_gguf(&file) {
        anyhow::bail!("`{}` is not an `asr` GGUF", file.display())
    }
    Ok(file)
}

/// The Streaming Sortformer `.nemo` `model_id` names.
pub(crate) fn sortformer_nemo(
    model_id: &str,
    revision: Option<String>,
    token_source: &TokenSource,
    silent: bool,
) -> Result<PathBuf> {
    let source = HubSource {
        revision,
        token_source,
        silent,
    };
    nemo_of_kind(model_id, source, is_sortformer, "Streaming Sortformer")
}

fn read_silero(file: &std::path::Path, device: &Device) -> Result<SileroVad> {
    let (config, vb) = read_gguf(
        &mut std::io::BufReader::new(std::fs::File::open(file)?),
        device,
    )?;
    Ok(SileroVad::new(config, vb)?)
}

impl Loader for VoiceActivityLoader {
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(options.silent);
        let file = silero_gguf(&self.model_id, revision, &token_source, options.silent)?;
        let vad = read_silero(&file, options.device)?;
        Ok(Arc::new(Mutex::new(TranscriptionPipeline::new(
            self.model_id.clone(),
            AudioModel::Silero(vad),
            inference_tensor::DType::F32,
        ))))
    }

    fn load_model_from_path(
        &self,
        _paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        crate::pipeline::loading::ensure_cuda_kernels(std::slice::from_ref(options.device))?;
        self.load_model_from_hf(None, TokenSource::None, options)
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

impl Loader for TranscriptionLoader {
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(options.silent);
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let api = crate::pipeline::hf::build_api(&token_source, !options.silent)?.repo(
            Repo::with_revision(self.model_id.clone(), RepoType::Model, revision.clone()),
        );
        let id = std::path::Path::new(&self.model_id);
        let get = |file: &str| crate::pipeline::hf::get_file(&api, id, file, &revision);
        let whisper = matches!(self.arch, Some(TranscriptionLoaderType::Whisper))
            || (self.arch.is_none()
                && !id.is_file()
                && get(CONFIG)
                    .ok()
                    .and_then(|config| std::fs::read_to_string(config).ok())
                    .is_some_and(|config| whisper::is_whisper_config(&config)));
        if whisper {
            let weights = match get(WEIGHTS) {
                Ok(weights) => vec![weights],
                Err(_) => {
                    let index: serde_json::Value =
                        serde_json::from_str(&std::fs::read_to_string(get(WEIGHTS_INDEX)?)?)?;
                    let mut shards = index["weight_map"]
                        .as_object()
                        .ok_or_else(|| anyhow::anyhow!("`{WEIGHTS_INDEX}` has no `weight_map`"))?
                        .values()
                        .filter_map(|shard| shard.as_str().map(str::to_string))
                        .collect::<Vec<_>>();
                    shards.sort();
                    shards.dedup();
                    shards
                        .iter()
                        .map(|shard| get(shard))
                        .collect::<Result<Vec<_>>>()?
                }
            };
            let paths = TranscriptionModelPaths {
                checkpoint: AsrCheckpoint::Whisper(WhisperFiles {
                    config: get(CONFIG)?,
                    generation_config: get(GENERATION_CONFIG)?,
                    tokenizer: get(TOKENIZER)?,
                    weights,
                }),
                vad: None,
            };
            return self.load_model_from_path(&paths, options);
        }
        // a repo's transformers Parakeet files win over a `.nemo` beside them; neither loading reports both causes
        let transformers = (!id.is_file())
            .then(|| -> Result<ParakeetFiles> {
                let config = get(CONFIG)?;
                if self.arch.is_none()
                    && TranscriptionLoaderType::auto_detect_from_config(&std::fs::read_to_string(
                        &config,
                    )?)
                    .is_none()
                {
                    anyhow::bail!("`{}` is not a Parakeet config", config.display())
                }
                Ok(ParakeetFiles {
                    config,
                    processor_config: get(PROCESSOR_CONFIG)?,
                    tokenizer: get(TOKENIZER)?,
                    weights: vec![get(WEIGHTS)?],
                })
            })
            .transpose();
        // past the transformers files, the `.nemo` (its F32 weights) before the 8-bit `asr` GGUF
        let packed = || -> Result<AsrCheckpoint> {
            let nemo = asr_nemo(
                &self.model_id,
                Some(revision.clone()),
                &token_source,
                options.silent,
            );
            nemo.map(AsrCheckpoint::Nemo).or_else(|nemo| {
                asr_gguf(
                    &self.model_id,
                    Some(revision.clone()),
                    &token_source,
                    options.silent,
                )
                .map(AsrCheckpoint::Gguf)
                .map_err(|gguf| anyhow::anyhow!("no `.nemo` ({nemo}) and no `asr` GGUF ({gguf})"))
            })
        };
        let checkpoint = match transformers {
            // a GGUF named outright loads as one, its own errors reported
            Ok(None) if is_asr_gguf(id) => AsrCheckpoint::Gguf(id.to_path_buf()),
            Ok(Some(files)) => AsrCheckpoint::Transformers(files),
            Ok(None) => packed()?,
            Err(layout) => packed()
                .map_err(|packed| anyhow::anyhow!("no transformers layout ({layout}), {packed}"))?,
        };
        let paths = TranscriptionModelPaths {
            checkpoint,
            vad: self
                .vad_model_id
                .as_deref()
                .map(|id| silero_gguf(id, None, &token_source, options.silent))
                .transpose()?,
        };
        self.load_model_from_path(&paths, options)
    }

    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        crate::pipeline::loading::ensure_cuda_kernels(std::slice::from_ref(options.device))?;
        let LoadOptions {
            dtype,
            device,
            silent,
            in_situ_quant,
            ..
        } = options;
        let _progress_guard = ProgressScopeGuard::new(silent);
        if in_situ_quant.is_some() {
            anyhow::bail!("Transcription models do not support in-situ quantization.")
        }
        let paths = paths
            .as_any()
            .downcast_ref::<TranscriptionModelPaths>()
            .expect("Path downcast failed.");
        if let AsrCheckpoint::Transformers(files) = &paths.checkpoint
            && self.arch.is_none()
            && TranscriptionLoaderType::auto_detect_from_config(&std::fs::read_to_string(
                &files.config,
            )?)
            .is_none()
        {
            anyhow::bail!(
                "`{}` is not a Parakeet config; pass the architecture explicitly",
                files.config.display()
            )
        }
        let dtype = dtype.try_into_dtype(&[device])?;
        let vad = match (&paths.vad, self.vad_model_id.as_deref()) {
            (Some(file), _) => Some(read_silero(file, device)?),
            (None, Some(id)) => Some(read_silero(
                &silero_gguf(id, None, &TokenSource::CacheToken, silent)?,
                device,
            )?),
            (None, None) => None,
        };
        if let AsrCheckpoint::Whisper(files) = &paths.checkpoint {
            let model = AudioModel::Whisper(Box::new(Whisper::load(files, device, dtype)?));
            return Ok(Arc::new(Mutex::new(TranscriptionPipeline::new(
                self.model_id.clone(),
                model,
                dtype,
            ))));
        }
        let asr = match &paths.checkpoint {
            AsrCheckpoint::Whisper(_) => unreachable!("a Whisper checkpoint loaded above"),
            AsrCheckpoint::Transformers(files) => Parakeet::load(files, device, dtype)?,
            AsrCheckpoint::Nemo(nemo) => Parakeet::load_nemo(nemo, device, dtype)?,
            AsrCheckpoint::Gguf(gguf) => Parakeet::load_gguf(gguf, device, dtype)?,
        };
        let model = AudioModel::Parakeet {
            asr: Box::new(asr),
            vad,
        };
        Ok(Arc::new(Mutex::new(TranscriptionPipeline::new(
            self.model_id.clone(),
            model,
            dtype,
        ))))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

impl TranscriptionPipeline {
    fn new(model_id: String, model: AudioModel, activation_dtype: inference_tensor::DType) -> Self {
        // a VAD answers with times, not text
        let output = match model {
            AudioModel::Parakeet { .. } | AudioModel::Whisper(_) => vec![SupportedModality::Text],
            AudioModel::Silero(_) | AudioModel::Diarizer(_) => Vec::new(),
        };
        Self {
            model_id,
            model,
            metadata: Arc::new(GeneralMetadata {
                prompt_chunk_tokens: None,
                max_seq_len: METADATA_MAX_SEQ_LEN,
                llg_factory: None,
                no_prefix_cache: true,
                num_hidden_layers: 1,
                eos_tok: vec![],
                kind: ModelKind::Normal,
                no_kv_cache: true,
                activation_dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Audio],
                    output,
                },
                loaded_for_uqff_write: false,
                label_decision: None,
                audio_transcriber: None,
                prompt_activation_bytes_per_token: None,
            }),
            dummy_cache: EitherCache::Full(Cache::new(0)),
            sessions: HashMap::new(),
        }
    }
}

// one op of a streamed session on recogniser `asr`; a session whose push fails is dropped, as its state may have moved
fn stream_step(
    asr: &Parakeet,
    sessions: &mut HashMap<u64, StreamSession>,
    request: &TranscriptionStreamRequest,
) -> std::result::Result<TranscriptionStreamUpdate, String> {
    // a client that vanished without closing its session releases it after a while
    sessions.retain(|_, s| s.last_used.elapsed() < STREAM_SESSION_IDLE);
    let update = |delta, transcription| TranscriptionStreamUpdate {
        delta,
        transcription,
    };
    let id = request.session;
    match &request.op {
        TranscriptionStreamOp::Open {
            options,
            sample_rate,
        } => {
            if !asr.can_stream() {
                return Err(
                    "this checkpoint transcribes whole recordings; only a cache-aware streaming one (Nemotron streaming ASR) streams"
                        .into(),
                );
            }
            if sessions.len() >= MAX_STREAM_SESSIONS {
                return Err(format!(
                    "{MAX_STREAM_SESSIONS} streamed transcriptions are open on this model; finish or close one first"
                ));
            }
            let session = StreamSession {
                stream: asr.stream(options).map_err(|e| e.to_string())?,
                resampler: StreamResampler::new(*sample_rate, asr.sample_rate())
                    .map_err(|e| e.to_string())?,
                last_used: Instant::now(),
            };
            sessions.insert(id, session);
            Ok(update(String::new(), None))
        }
        TranscriptionStreamOp::Push(pcm) => {
            let session = sessions.get_mut(&id).ok_or_else(|| unknown_session(id))?;
            session.last_used = Instant::now();
            let delta = session
                .resampler
                .push(pcm)
                .and_then(|pcm| Ok(asr.stream_push(&mut session.stream, &pcm)?));
            if delta.is_err() {
                sessions.remove(&id);
            }
            Ok(update(delta.map_err(|e| e.to_string())?, None))
        }
        TranscriptionStreamOp::Finish => {
            let mut session = sessions.remove(&id).ok_or_else(|| unknown_session(id))?;
            let tail = session.resampler.finish().map_err(|e| e.to_string())?;
            let mut delta = asr
                .stream_push(&mut session.stream, &tail)
                .map_err(|e| e.to_string())?;
            let (last, transcription) = asr
                .stream_finish(session.stream)
                .map_err(|e| e.to_string())?;
            delta.push_str(&last);
            Ok(update(delta, Some(transcription)))
        }
        TranscriptionStreamOp::Close => {
            sessions.remove(&id);
            Ok(update(String::new(), None))
        }
    }
}

fn unknown_session(id: u64) -> String {
    format!("streamed transcription {id} is not open: it finished, closed or sat idle too long")
}

impl PreProcessingMixin for TranscriptionPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(TranscriptionProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for TranscriptionPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("Transcription models do not support ISQ.")
    }
}

impl CacheManagerMixin for TranscriptionPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) -> inference_tensor::Result<()> {
        Ok(())
    }
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn set_none_cache(
        &self,
        _seqs: &mut [&mut Sequence],
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) -> inference_tensor::Result<()> {
        Ok(())
    }
    fn cache(&self) -> &EitherCache {
        &self.dummy_cache
    }
}

impl MetadataMixin for TranscriptionPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        None
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

impl Pipeline for TranscriptionPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> inference_tensor::Result<ForwardInputsResult> {
        assert!(!return_raw_logits);
        let ModelInputs {
            audios,
            options,
            diarization,
            transcription,
        } = *inputs.downcast().expect("Downcast failed.");
        let Self {
            model, sessions, ..
        } = self;
        Ok(match model {
            AudioModel::Parakeet { asr, vad } => ForwardInputsResult::Transcription {
                transcripts: audios
                    .iter()
                    .zip(&transcription)
                    .map(|(input, request)| match input {
                        AudioRequest::Stream(stream) => {
                            stream_step(asr, sessions, stream).map(TranscriptionOutcome::Stream)
                        }
                        AudioRequest::Audio(audio) => {
                            let pcm = audio.to_mono();
                            match vad {
                                Some(vad) => {
                                    asr.transcribe_with_vad(&pcm, audio.sample_rate, vad, request)
                                }
                                None => asr.transcribe(&pcm, audio.sample_rate, request),
                            }
                            .map(TranscriptionOutcome::Whole)
                            .map_err(|e| e.to_string())
                        }
                    })
                    .collect(),
            },
            AudioModel::Whisper(asr) => ForwardInputsResult::Transcription {
                transcripts: audios
                    .iter()
                    .zip(&transcription)
                    .map(|(input, request)| match input {
                        AudioRequest::Stream(_) => {
                            Err("Whisper transcribes whole recordings; it does not stream"
                                .to_string())
                        }
                        AudioRequest::Audio(audio) => asr
                            .transcribe(&audio.to_mono(), audio.sample_rate, request)
                            .map(TranscriptionOutcome::Whole)
                            .map_err(|e| e.to_string()),
                    })
                    .collect(),
            },
            AudioModel::Silero(vad) => ForwardInputsResult::VoiceActivity {
                results: audios
                    .iter()
                    .zip(&options)
                    .map(|(input, options)| {
                        let audio = input.audio();
                        vad.detect(&audio.to_mono(), audio.sample_rate, options)
                            .map_err(|e| e.to_string())
                    })
                    .collect(),
            },
            AudioModel::Diarizer(diarizer) => ForwardInputsResult::Diarization {
                results: audios
                    .iter()
                    .zip(&diarization)
                    .map(|(input, options)| {
                        let audio = input.audio();
                        diarizer
                            .diarize(&audio.to_mono(), audio.sample_rate, options.threshold)
                            .map_err(|e| e.to_string())
                    })
                    .collect(),
            },
        })
    }

    fn sample_causal_gen<'a>(
        &'a self,
        _seqs: &'a mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &'a mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _srng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<(), inference_tensor::Error>> {
        Box::pin(std::future::ready(Err(inference_tensor::Error::Msg(
            "`sample_causal_gen` is incompatible with `TranscriptionPipeline`".to_string(),
        )
        .bt())))
    }

    fn category(&self) -> ModelCategory {
        match self.model {
            AudioModel::Parakeet { .. } | AudioModel::Whisper(_) => ModelCategory::Transcription,
            AudioModel::Silero(_) => ModelCategory::VoiceActivity,
            AudioModel::Diarizer(_) => ModelCategory::Diarization,
        }
    }
}

impl AnyMoePipelineMixin for TranscriptionPipeline {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parakeet_model_types_are_detected() {
        for (model_type, _) in MODEL_TYPES {
            let config = format!(r#"{{"model_type": "{model_type}"}}"#);
            assert_eq!(
                TranscriptionLoaderType::auto_detect_from_config(&config),
                Some(TranscriptionLoaderType::Parakeet)
            );
        }
        assert_eq!(
            TranscriptionLoaderType::auto_detect_from_config(r#"{"model_type": "llama"}"#),
            None
        );
    }
}
