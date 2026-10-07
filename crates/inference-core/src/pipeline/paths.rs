use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use either::Either;
use hf_hub::{Repo, RepoType, api::sync::ApiRepo};
use regex_automata::meta::Regex;
use serde_json::Value;
use tracing::{debug, info, trace, warn};

use inference_protocol::chat_template::{BeginEndUnkPadTok, ChatTemplate, ChatTemplateValue};

use crate::{
    LoraAdapterSpec, ModelPaths, TokenSource,
    pipeline::{hf::build_api, isq::UQFF_RESIDUAL_SAFETENSORS},
};

// Match files against these
const SAFETENSOR_MATCH: &str = r"model-\d+-of-\d+\.safetensors\b";
const QUANT_SAFETENSOR_MATCH: &str = r"model\.safetensors\b";
const CONSOLIDATED_SAFETENSOR_MATCH: &str = r"consolidated\.safetensors\b";
const SAFETENSOR_INDEX: &str = "model.safetensors.index.json";
const PICKLE_MATCH: &str = r"pytorch_model-\d{5}-of-\d{5}.((pth)|(pt)|(bin))\b";

#[derive(Clone, Debug)]
pub struct ResolvedLoraAdapter {
    pub alias: String,
    pub source: String,
    pub revision: Option<String>,
    pub config_path: PathBuf,
    pub weights_path: PathBuf,
}

#[derive(Clone, Debug)]
pub enum AdapterPaths {
    Lora(Vec<ResolvedLoraAdapter>),
    None,
}

fn get_adapter_paths(
    lora_adapters: Option<&[LoraAdapterSpec]>,
    token_source: &TokenSource,
) -> Result<AdapterPaths> {
    let Some(adapters) = lora_adapters else {
        return Ok(AdapterPaths::None);
    };
    let mut lora_adapter_paths = Vec::new();
    let mut aliases = HashSet::new();
    for adapter in adapters {
        let alias = adapter.alias.trim();
        let source = adapter.source.trim();
        if alias.is_empty() {
            anyhow::bail!("LoRA adapter alias must not be empty");
        }
        if source.is_empty() {
            anyhow::bail!(
                "LoRA adapter source for alias `{}` must not be empty",
                adapter.alias
            );
        }
        if adapter.revision().is_empty() {
            anyhow::bail!(
                "LoRA adapter revision for alias `{}` must not be empty",
                adapter.alias
            );
        }
        if let Some(expected) = adapter.base_model_name.as_deref().map(str::trim)
            && expected.is_empty()
        {
            anyhow::bail!(
                "LoRA adapter `{}` has an empty base_model_name",
                adapter.alias
            );
        }
        if !aliases.insert(alias) {
            anyhow::bail!(
                "LoRA adapter alias `{}` is specified more than once",
                adapter.alias
            );
        }
        info!(
            "Loading LoRA adapter `{}` from `{}` at revision `{}`",
            alias,
            source,
            adapter.revision()
        );

        let api = build_api(token_source, true).map_err(inference_tensor::Error::msg)?;
        let api = api.repo(Repo::with_revision(
            source.to_string(),
            RepoType::Model,
            adapter.revision().to_string(),
        ));

        let adapter_path_buf = std::path::Path::new(source);
        let config_path = crate::pipeline::hf::get_file(
            &api,
            adapter_path_buf,
            "adapter_config.json",
            adapter.revision(),
        )?;
        let weights_path = crate::pipeline::hf::get_file(
            &api,
            adapter_path_buf,
            "adapter_model.safetensors",
            adapter.revision(),
        )?;
        lora_adapter_paths.push(ResolvedLoraAdapter {
            alias: alias.to_string(),
            source: source.to_string(),
            revision: (!adapter_path_buf.exists()).then(|| adapter.revision().to_string()),
            config_path,
            weights_path,
        });
    }

    Ok(AdapterPaths::Lora(lora_adapter_paths))
}

pub fn get_model_paths(
    revision: String,
    token_source: &TokenSource,
    quantized_model_id: Option<&String>,
    quantized_filename: Option<&Vec<String>>,
    api: &ApiRepo,
    model_id: &Path,
    loading_from_uqff: bool,
) -> Result<Vec<PathBuf>> {
    match quantized_filename {
        Some(names) => {
            let id = quantized_model_id.unwrap();
            let mut files = Vec::new();

            for name in names {
                let qapi = build_api(token_source, true).map_err(inference_tensor::Error::msg)?;
                let qapi = qapi.repo(Repo::with_revision(
                    id.to_string(),
                    RepoType::Model,
                    revision.clone(),
                ));
                let model_id = Path::new(&id);
                files.push(crate::pipeline::hf::get_file(
                    &qapi, model_id, name, &revision,
                )?);
            }
            Ok(files)
        }
        None => {
            let safetensor_match = Regex::new(SAFETENSOR_MATCH)?;
            let quant_safetensor_match = Regex::new(QUANT_SAFETENSOR_MATCH)?;
            let consolidated_safetensor_match = Regex::new(CONSOLIDATED_SAFETENSOR_MATCH)?;
            let pickle_match = Regex::new(PICKLE_MATCH)?;

            let mut filenames = vec![];
            let repo_files = crate::pipeline::hf::list_repo_files(api, model_id, true, &revision)?;
            let safetensors = if repo_files.iter().any(|file| file == SAFETENSOR_INDEX) {
                let index_path =
                    crate::pipeline::hf::get_file(api, model_id, SAFETENSOR_INDEX, &revision)?;
                parse_safetensor_index(&fs::read_to_string(index_path)?)?
            } else {
                repo_files
                    .iter()
                    .filter(|file| {
                        safetensor_match.is_match(file)
                            || quant_safetensor_match.is_match(file)
                            || consolidated_safetensor_match.is_match(file)
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            };
            let pickles = repo_files
                .iter()
                .filter(|file| pickle_match.is_match(file))
                .filter(|x| x.ends_with(".pth") || x.ends_with(".pt") || x.ends_with(".bin"))
                .cloned()
                .collect::<Vec<_>>();
            let uqff_residual = repo_files
                .iter()
                .filter(|file| file.as_str() == UQFF_RESIDUAL_SAFETENSORS)
                .cloned()
                .collect::<Vec<_>>();
            let files = if !safetensors.is_empty() {
                // Always prefer safetensors
                safetensors
            } else if !pickles.is_empty() {
                // Fall back to pickle
                pickles
            } else if !uqff_residual.is_empty() && loading_from_uqff {
                uqff_residual
            } else {
                anyhow::bail!("Expected file with extension one of .safetensors, .pth, .pt, .bin.");
            };
            trace!(
                "Found model weight filenames {:?}",
                files
                    .iter()
                    .map(|x| x.split('/').next_back().unwrap())
                    .collect::<Vec<_>>()
            );
            for rfilename in files {
                filenames.push(crate::pipeline::hf::get_file(
                    api, model_id, &rfilename, &revision,
                )?);
            }
            Ok(filenames)
        }
    }
}

fn parse_safetensor_index(contents: &str) -> Result<Vec<String>> {
    let index: Value = serde_json::from_str(contents)?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("Safetensors index has no object `weight_map`"))?;
    let mut files = HashSet::new();
    for filename in weight_map.values() {
        let filename = filename
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Safetensors index contains a non-string filename"))?;
        if !filename.ends_with(".safetensors") {
            anyhow::bail!("Safetensors index references non-safetensors file `{filename}`")
        }
        files.insert(filename.to_string());
    }
    if files.is_empty() {
        anyhow::bail!("Safetensors index has an empty `weight_map`")
    }
    let mut files = files.into_iter().collect::<Vec<_>>();
    files.sort_unstable();
    Ok(files)
}

/// Find and parse the appropriate [`ChatTemplate`], and ensure is has a valid [`ChatTemplate.chat_template`].
/// If the provided `tokenizer_config.json` from [`ModelPaths.get_template_filename`] does not
/// have a `chat_template`, use the provided one.
///
/// - Uses `chat_template_fallback` if `paths` does not contain a chat template file. This may be a literal or .json file.
/// - `chat_template_ovrd` (GGUF chat template content) causes the usage of that string chat template initially.
///   Falls back to `chat_template_file` if it is invalid. *The user must add the bos/unk/eos tokens manually if this
///   is used.*
///
/// THE FOLLOWING IS IGNORED:
/// After this, if the `chat_template_explicit` filename is specified (a json with one field: "chat_template" OR a jinja file),
///  the chat template is overwritten with this chat template.
#[allow(clippy::borrowed_box)]
pub(crate) fn get_chat_template(
    paths: &dyn ModelPaths,
    jinja_explicit: Option<&String>,
    chat_template_explicit: Option<&String>,
    chat_template_fallback: Option<&String>,
    chat_template_ovrd: Option<String>,
) -> ChatTemplate {
    // Get template content, this may be overridden.
    let template_content = if let Some(template_filename) = paths.get_template_filename() {
        if !["jinja", "json"].contains(
            &template_filename
                .extension()
                .expect("Template filename must be a file")
                .to_string_lossy()
                .to_string()
                .as_str(),
        ) {
            panic!("Template filename {template_filename:?} must end with `.json` or `.jinja`.");
        }
        Some(fs::read_to_string(template_filename).expect("Loading chat template failed."))
    } else if chat_template_fallback.is_some_and(|f| f.ends_with(".json")) {
        // User specified a file
        let template_filename = chat_template_fallback
            .expect("A tokenizer config or chat template file path must be specified.");
        Some(fs::read_to_string(template_filename).expect("Loading chat template failed."))
    } else if chat_template_ovrd.is_some() {
        None
    } else {
        debug!(
            "No chat template file found. Chat template may be set via `chat_template.json` or processor config."
        );
        None
    };
    let mut template: ChatTemplate = match chat_template_ovrd {
        Some(chat_template) => {
            // In this case the override chat template is being used. The user must add the bos/eos/unk toks themselves.
            debug!("Using literal chat template.");
            let mut template = ChatTemplate::default();
            template.chat_template = Some(ChatTemplateValue(Either::Left(chat_template)));
            template
        }
        None => {
            if let Some(ref content) = template_content {
                // Check if template_filename is a .jinja file
                if let Some(template_filename) = paths.get_template_filename() {
                    if template_filename.extension().map(|e| e.to_str()) == Some(Some("jinja")) {
                        debug!("Using chat template from .jinja file.");
                        // Load special tokens (bos/eos/unk) from tokenizer_config.json
                        // in the same directory, matching HF's behavior where
                        // apply_chat_template passes self.special_tokens_map to the template.
                        let mut template = template_filename
                            .parent()
                            .map(|dir| dir.join("tokenizer_config.json"))
                            .filter(|p| p.exists())
                            .and_then(|p| fs::read_to_string(p).ok())
                            .and_then(|s| serde_json::from_str::<ChatTemplate>(&s).ok())
                            .unwrap_or_else(|| {
                                // Fallback: older UQFF repos may not have tokenizer_config.json.
                                // Try to extract bos/eos tokens from the tokenizer.json's
                                // added_tokens list to avoid rendering "none" in the template.
                                let mut ct = ChatTemplate::default();
                                if let Some(tok_path) = paths
                                    .get_tokenizer_filename()
                                    .parent()
                                    .map(|d| d.join("tokenizer.json"))
                                    .filter(|p| p.exists())
                                    .or_else(|| {
                                        template_filename
                                            .parent()
                                            .map(|d| d.join("tokenizer.json"))
                                            .filter(|p| p.exists())
                                    })
                                    && let Some(tok_json) =
                                        fs::read_to_string(&tok_path).ok().and_then(|s| {
                                            serde_json::from_str::<serde_json::Value>(&s).ok()
                                        })
                                {
                                    let added = tok_json
                                        .get("added_tokens")
                                        .and_then(serde_json::Value::as_array);
                                    for token in added.into_iter().flatten() {
                                        let content = token
                                            .get("content")
                                            .and_then(serde_json::Value::as_str)
                                            .unwrap_or("");
                                        let special = token
                                            .get("special")
                                            .and_then(serde_json::Value::as_bool)
                                            .unwrap_or(false);
                                        if special {
                                            if content == "<bos>" {
                                                ct.bos_token = Some(BeginEndUnkPadTok(
                                                    Either::Left(content.to_string()),
                                                ));
                                            } else if content == "<eos>" {
                                                ct.eos_token = Some(BeginEndUnkPadTok(
                                                    Either::Left(content.to_string()),
                                                ));
                                            } else if content == "<unk>" {
                                                ct.unk_token = Some(BeginEndUnkPadTok(
                                                    Either::Left(content.to_string()),
                                                ));
                                            }
                                        }
                                    }
                                }
                                ct
                            });
                        template.chat_template =
                            Some(ChatTemplateValue(Either::Left(content.clone())));
                        template
                    } else {
                        serde_json::from_str(content).unwrap()
                    }
                } else {
                    serde_json::from_str(content).unwrap()
                }
            } else {
                // No template content available; downstream code may fill in from
                // chat_template.json, processor_config, or jinja_explicit.
                ChatTemplate::default()
            }
        }
    };
    // Overwrite to use any present `chat_template.json`, only if there is not one present already.
    if template.chat_template.is_none()
        && let Some(chat_template_explicit) = chat_template_explicit
    {
        let ct = fs::read_to_string(chat_template_explicit).expect("Loading chat template failed.");

        let new_chat_template = if chat_template_explicit.ends_with(".jinja") {
            ct
        } else {
            #[derive(Debug, serde::Deserialize)]
            struct AutomaticTemplate {
                chat_template: String,
            }
            let deser: AutomaticTemplate = serde_json::from_str(&ct).unwrap();
            deser.chat_template
        };

        template.chat_template = Some(ChatTemplateValue(Either::Left(new_chat_template)));
    }

    let processor_conf: Option<crate::vision_models::processor_config::ProcessorConfig> = paths
        .get_processor_config()
        .as_ref()
        .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
    if let Some(processor_conf) = processor_conf
        && processor_conf.chat_template.is_some()
    {
        template.chat_template = processor_conf
            .chat_template
            .map(|x| ChatTemplateValue(Either::Left(x)));
    }

    if let Some(jinja_explicit) = jinja_explicit {
        if !jinja_explicit.ends_with(".jinja") {
            panic!("jinja_explicit must end with .jinja!");
        }

        let ct = fs::read_to_string(jinja_explicit).expect("Loading chat template failed.");
        template.chat_template = Some(ChatTemplateValue(Either::Left(ct)));
    }

    #[derive(Debug, serde::Deserialize)]
    struct SpecifiedTemplate {
        chat_template: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
        unk_token: Option<String>,
    }

    if template.chat_template.is_some() {
        return template;
    };

    match &template.chat_template {
        Some(_) => template,
        None => {
            if let Some(template_content) = template_content {
                info!(
                    "`tokenizer_config.json` does not contain a chat template, attempting to use specified JINJA chat template."
                );
                let mut deser: HashMap<String, Value> =
                    serde_json::from_str(&template_content).unwrap();

                match chat_template_fallback.cloned() {
                    Some(t) => {
                        info!("Loading specified loading chat template file at `{t}`.");
                        let templ: SpecifiedTemplate =
                            serde_json::from_str(&fs::read_to_string(t.clone()).unwrap()).unwrap();
                        deser.insert(
                            "chat_template".to_string(),
                            Value::String(templ.chat_template),
                        );
                        if let Some(bos_token) = templ.bos_token {
                            deser.insert("bos_token".to_string(), Value::String(bos_token));
                        }
                        if let Some(eos_token) = templ.eos_token {
                            deser.insert("eos_token".to_string(), Value::String(eos_token));
                        }
                        if let Some(unk_token) = templ.unk_token {
                            deser.insert("unk_token".to_string(), Value::String(unk_token));
                        }
                    }
                    None => {
                        warn!(
                            "No specified chat template. No chat template will be used. Only prompts will be accepted, not messages."
                        );
                        deser.insert("chat_template".to_string(), Value::Null);
                    }
                }

                let ser = serde_json::to_string_pretty(&deser)
                    .expect("Serialization of modified chat template failed.");
                serde_json::from_str(&ser).unwrap()
            } else {
                warn!(
                    "No chat template source found. No chat template will be used. Only prompts will be accepted, not messages."
                );
                template
            }
        }
    }
}

/// One repository (a hub repo or a local directory) at one revision: what it lists, and a fetch for any file.
pub(crate) struct RepoFiles<'a> {
    api: ApiRepo,
    model_id: &'a Path,
    revision: String,
    listed: Vec<String>,
}

impl<'a> RepoFiles<'a> {
    pub fn open(
        model_id: &'a str,
        token_source: &TokenSource,
        revision: Option<String>,
        silent: bool,
    ) -> Result<Self> {
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let api = build_api(token_source, !silent)?.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.clone(),
        ));
        let model_id = Path::new(model_id);
        let listed = super::hf::list_repo_files(&api, model_id, false, &revision)?;
        Ok(Self {
            api,
            model_id,
            revision,
            listed,
        })
    }

    pub fn has(&self, file: &str) -> bool {
        self.listed.iter().any(|listed| listed == file)
    }

    pub fn get(&self, file: &str) -> Result<PathBuf> {
        trace!("Loading `{file}` at `{}`", self.model_id.display());
        super::hf::get_file(&self.api, self.model_id, file, &self.revision)
    }

    /// `file` when the repository lists it.
    pub fn get_listed(&self, file: &str) -> Result<Option<PathBuf>> {
        self.has(file).then(|| self.get(file)).transpose()
    }

    /// The weight files: `quantized_filenames` from `quantized_model_id`, or the repository's own safetensors.
    fn weights(
        &self,
        token_source: &TokenSource,
        quantized_model_id: Option<&String>,
        quantized_filenames: Option<&Vec<String>>,
        loading_uqff: bool,
    ) -> Result<Vec<PathBuf>> {
        get_model_paths(
            self.revision.clone(),
            token_source,
            quantized_model_id,
            quantized_filenames,
            &self.api,
            self.model_id,
            loading_uqff,
        )
    }
}

/// What a safetensors (or GGML) model's file lookup takes from its loader.
pub(crate) struct PathsRequest<'a> {
    pub model_id: &'a str,
    pub tokenizer_json: Option<&'a str>,
    pub chat_template: Option<&'a str>,
    pub token_source: &'a TokenSource,
    pub revision: Option<String>,
    pub quantized_model_id: Option<&'a String>,
    pub quantized_filenames: Option<&'a Vec<String>>,
    pub silent: bool,
    pub loading_uqff: bool,
}

/// The tokenizer, config, weights, adapters, templates and processor configs a model loads from.
pub(crate) fn get_paths(
    request: PathsRequest<'_>,
    lora_adapters: Option<&[LoraAdapterSpec]>,
) -> Result<crate::pipeline::LocalModelPaths<PathBuf>> {
    let repo = RepoFiles::open(
        request.model_id,
        request.token_source,
        request.revision,
        request.silent,
    )?;
    let tokenizer_filename = match request.tokenizer_json {
        Some(path) => {
            trace!("Using tokenizer.json at `{path}`");
            PathBuf::from(path)
        }
        // Mistral checkpoints ship `tekken.json` in place of a HF tokenizer
        None if !repo.has("tokenizer.json") && repo.has("tekken.json") => {
            repo.get("tekken.json")?
        }
        None => repo.get("tokenizer.json")?,
    };
    // Mistral's native `params.json` takes precedence over `config.json`
    let config_filename = if repo.has("params.json") {
        repo.get("params.json")?
    } else {
        repo.get("config.json")?
    };
    repo.get_listed("hf_quant_config.json")?;
    let filenames = repo.weights(
        request.token_source,
        request.quantized_model_id,
        request.quantized_filenames,
        request.loading_uqff,
    )?;
    let adapter_paths = get_adapter_paths(lora_adapters, request.token_source)?;
    let processor_configs = ProcessorConfigs::fetch(&repo)?;
    let template_filename = match request.chat_template {
        Some(path) => {
            debug!("Using chat template file at `{path}`");
            Some(PathBuf::from(path))
        }
        None => listed_chat_template(&repo)?,
    };
    let chat_template_json_filename = repo.get_listed("chat_template.json")?;
    Ok(
        processor_configs.into_paths(crate::pipeline::LocalModelPaths {
            tokenizer_filename,
            config_filename,
            filenames,
            adapter_paths,
            template_filename,
            gen_conf: None,
            preprocessor_config: None,
            video_preprocessor_config: None,
            processor_config: None,
            chat_template_json_filename,
        }),
    )
}

/// The generation and processor configs a repository lists, fetched together.
struct ProcessorConfigs {
    gen_conf: Option<PathBuf>,
    preprocessor_config: Option<PathBuf>,
    video_preprocessor_config: Option<PathBuf>,
    processor_config: Option<PathBuf>,
}

impl ProcessorConfigs {
    fn fetch(repo: &RepoFiles<'_>) -> Result<Self> {
        Ok(Self {
            gen_conf: repo.get_listed("generation_config.json")?,
            preprocessor_config: repo.get_listed("preprocessor_config.json")?,
            video_preprocessor_config: repo.get_listed("video_preprocessor_config.json")?,
            processor_config: repo.get_listed("processor_config.json")?,
        })
    }

    fn into_paths(
        self,
        paths: crate::pipeline::LocalModelPaths<PathBuf>,
    ) -> crate::pipeline::LocalModelPaths<PathBuf> {
        crate::pipeline::LocalModelPaths {
            gen_conf: self.gen_conf,
            preprocessor_config: self.preprocessor_config,
            video_preprocessor_config: self.video_preprocessor_config,
            processor_config: self.processor_config,
            ..paths
        }
    }
}

// A `.jinja` template renders the bos/eos tokens `tokenizer_config.json` holds, so that is fetched beside it.
fn listed_chat_template(repo: &RepoFiles<'_>) -> Result<Option<PathBuf>> {
    if repo.has("chat_template.jinja") {
        repo.get_listed("tokenizer_config.json")?;
        return Ok(Some(repo.get("chat_template.jinja")?));
    }
    let template = repo.get_listed("tokenizer_config.json")?;
    if template.is_none() {
        debug!(
            "No chat template or `tokenizer_config.json` found at `{}`",
            repo.model_id.display()
        );
    }
    Ok(template)
}

/// An embedding model's files: as [`get_paths`] has them, plus its sentence-transformers modules.
pub(crate) fn get_embedding_paths(
    request: PathsRequest<'_>,
) -> Result<crate::pipeline::EmbeddingModelPaths<PathBuf>> {
    let repo = RepoFiles::open(
        request.model_id,
        request.token_source,
        request.revision,
        request.silent,
    )?;
    let tokenizer_filename = match request.tokenizer_json {
        Some(path) => PathBuf::from(path),
        None if !repo.has("tokenizer.json") && repo.has("tekken.json") => {
            repo.get("tekken.json")?
        }
        None => repo.get("tokenizer.json")?,
    };
    let config_filename = if repo.has("params.json") {
        repo.get("params.json")?
    } else {
        repo.get("config.json")?
    };
    repo.get_listed("hf_quant_config.json")?;
    let filenames = repo.weights(
        request.token_source,
        request.quantized_model_id,
        request.quantized_filenames,
        request.loading_uqff,
    )?;
    let modules_path = if repo.model_id.exists() {
        repo.model_id.join("modules.json")
    } else {
        repo.get("modules.json")?
    };
    let mut modules = Vec::new();
    if modules_path.exists() {
        let listed: Vec<crate::pipeline::EmbeddingModule> =
            serde_json::from_str(&fs::read_to_string(&modules_path)?)?;
        for module in listed {
            use crate::pipeline::{EmbeddingModulePaths as Paths, EmbeddingModuleType as Type};
            let path = module.path.clone();
            modules.push(match module.ty {
                Type::Transformer => Paths::Transformer { path },
                Type::Pooling => Paths::Pooling {
                    config: repo.get(&format!("{path}/config.json"))?,
                    path,
                },
                Type::Dense => Paths::Dense {
                    config: repo.get(&format!("{path}/config.json"))?,
                    model: repo.get(&format!("{path}/model.safetensors"))?,
                    path,
                },
                Type::Normalize => Paths::Normalize { path },
            });
        }
    }
    Ok(crate::pipeline::EmbeddingModelPaths {
        tokenizer_filename,
        config_filename,
        filenames,
        adapter_paths: AdapterPaths::None,
        modules,
    })
}

/// What a GGUF model's file lookup takes from its loader.
pub(crate) struct GgufPathsRequest<'a> {
    /// The repository with the tokenizer and configs; `None` loads everything from the GGUF repository.
    pub model_id: Option<&'a str>,
    pub quantized_model_id: &'a String,
    pub quantized_filenames: &'a Vec<String>,
    pub chat_template: Option<&'a str>,
    pub token_source: &'a TokenSource,
    pub revision: Option<String>,
    pub silent: bool,
}

/// A GGUF model's files, with its LoRA adapters' when given.
pub(crate) fn get_paths_gguf(
    request: GgufPathsRequest<'_>,
    lora_adapters: Option<&[LoraAdapterSpec]>,
) -> Result<crate::pipeline::LocalModelPaths<PathBuf>> {
    let this_model_id = request.model_id.unwrap_or(request.quantized_model_id);
    let repo = RepoFiles::open(
        this_model_id,
        request.token_source,
        request.revision,
        request.silent,
    )?;
    let template_filename = match request.chat_template {
        Some(path) if path.ends_with(".json") || path.ends_with(".jinja") => {
            debug!("Using chat template file at `{path}`");
            Some(PathBuf::from(path))
        }
        Some(_) => panic!("Specified chat template file must end with .json or .jinja"),
        None if request.model_id.is_none() => None,
        None => listed_chat_template(&repo)?,
    };
    let filenames = repo.weights(
        request.token_source,
        Some(request.quantized_model_id),
        Some(request.quantized_filenames),
        false,
    )?;
    debug!("GGUF file(s) {:?}", filenames);
    let adapter_paths = get_adapter_paths(lora_adapters, request.token_source)?;
    let processor_configs = ProcessorConfigs::fetch(&repo)?;
    // empty when the repository has none, and the GGUF file's own tokenizer and config are used
    let tokenizer_filename = repo.get_listed("tokenizer.json")?.unwrap_or_default();
    let config_filename = match repo.get_listed("config.json")? {
        Some(path) => path,
        None => repo.get_listed("params.json")?.unwrap_or_default(),
    };
    let chat_template_json_filename = repo.get_listed("chat_template.json")?;
    Ok(
        processor_configs.into_paths(crate::pipeline::LocalModelPaths {
            tokenizer_filename,
            config_filename,
            filenames,
            adapter_paths,
            template_filename,
            gen_conf: None,
            preprocessor_config: None,
            video_preprocessor_config: None,
            processor_config: None,
            chat_template_json_filename,
        }),
    )
}

/// The UQFF files `from_uqff` names, with the shard siblings and report-resolved names the repository lists.
pub(crate) fn get_uqff_paths(
    from_uqff: &[PathBuf],
    model_id: &str,
    token_source: &TokenSource,
    revision: Option<String>,
    silent: bool,
) -> Result<Vec<PathBuf>> {
    let revision = revision.unwrap_or_else(|| "main".to_string());
    let api = build_api(token_source, !silent)?.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.clone(),
    ));
    let model_path = Path::new(model_id);
    let available_files =
        super::hf::list_repo_files(&api, model_path, false, &revision).unwrap_or_default();
    let uqff_report = if available_files
        .iter()
        .any(|file| file == inference_quant::UQFF_REPORT_JSON)
    {
        let report_path = super::hf::get_file(
            &api,
            model_path,
            inference_quant::UQFF_REPORT_JSON,
            &revision,
        )?;
        Some(super::isq::read_uqff_report_file(&report_path)?)
    } else {
        None
    };

    let input_files: Vec<String> = from_uqff.iter().map(|f| f.display().to_string()).collect();
    let mut expanded_files: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for input in &input_files {
        let resolved =
            super::isq::resolve_uqff_input_files(input, &available_files, uqff_report.as_ref())?;
        if resolved.len() != 1 || resolved.first() != Some(input) {
            debug!("Resolved UQFF input `{}` to {:?}", input, resolved);
        } else if input.parse::<u32>().is_ok() {
            let available_uqff: Vec<_> = available_files
                .iter()
                .filter(|file| file.ends_with(".uqff"))
                .collect();
            warn!(
                "No UQFF file found for shorthand `{}`. Available UQFF files: {:?}",
                input, available_uqff,
            );
        }
        for file in resolved {
            if seen.insert(file.clone()) {
                expanded_files.push(file);
            }
        }
    }
    if expanded_files.len() > input_files.len() {
        debug!(
            "Auto-discovered {} UQFF shard files (from {} specified)",
            expanded_files.len(),
            input_files.len()
        );
    }
    expanded_files
        .iter()
        .map(|file| super::hf::get_file(&api, model_path, file, &revision))
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::pipeline::loaders::LocalModelPaths;

    use super::{AdapterPaths, get_chat_template, parse_safetensor_index};

    #[test]
    fn explicit_jinja_overrides_processor_template() {
        let dir = tempfile::tempdir().unwrap();
        let processor_path = dir.path().join("processor_config.json");
        let jinja_path = dir.path().join("explicit.jinja");
        std::fs::write(&processor_path, r#"{"chat_template":"processor"}"#).unwrap();
        std::fs::write(&jinja_path, "explicit").unwrap();
        let paths = LocalModelPaths {
            tokenizer_filename: dir.path().join("tokenizer.json"),
            config_filename: dir.path().join("config.json"),
            template_filename: None,
            filenames: Vec::new(),
            adapter_paths: AdapterPaths::None,
            gen_conf: None,
            preprocessor_config: None,
            video_preprocessor_config: None,
            processor_config: Some(processor_path),
            chat_template_json_filename: None,
        };
        let jinja_path = jinja_path.to_string_lossy().into_owned();

        let template = get_chat_template(&paths, Some(&jinja_path), None, None, None);

        assert_eq!(template.get_template_contents(), ["explicit"]);
    }

    #[test]
    fn match_safetensors() -> anyhow::Result<()> {
        use regex_automata::meta::Regex;

        use super::SAFETENSOR_MATCH;
        let safetensor_match = Regex::new(SAFETENSOR_MATCH)?;

        let positive_ids = [
            "model-00001-of-00001.safetensors",
            "model-00002-of-00002.safetensors",
            "model-00003-of-00003.safetensors",
            "model-00004-of-00004.safetensors",
            "model-00005-of-00005.safetensors",
            "model-00006-of-00006.safetensors",
        ];
        let negative_ids = [
            "model-0000a-of-00002.safetensors",
            "consolidated.safetensors",
        ];
        for id in positive_ids {
            assert!(safetensor_match.is_match(id));
        }
        for id in negative_ids {
            assert!(!safetensor_match.is_match(id));
        }
        Ok(())
    }

    #[test]
    fn safetensor_index_drives_arbitrary_shard_names() -> anyhow::Result<()> {
        let files = parse_safetensor_index(
            r#"{
                "metadata": {"total_size": 42},
                "weight_map": {
                    "model.embed_tokens.weight": "embeddings.safetensors",
                    "model.layers.0.weight": "layers-0.safetensors",
                    "model.layers.0.weight_scale_inv": "layers-0.safetensors",
                    "model.visual.weight": "vision.safetensors"
                }
            }"#,
        )?;
        assert_eq!(
            files,
            [
                "embeddings.safetensors",
                "layers-0.safetensors",
                "vision.safetensors"
            ]
        );
        Ok(())
    }

    #[test]
    fn safetensor_index_rejects_non_safetensor_shards() {
        let err = parse_safetensor_index(r#"{"weight_map":{"model.weight":"pytorch_model.bin"}}"#)
            .unwrap_err();
        assert!(err.to_string().contains("non-safetensors"));
    }

    #[test]
    fn match_pickle() -> anyhow::Result<()> {
        use regex_automata::meta::Regex;

        use super::PICKLE_MATCH;
        let pickle_match = Regex::new(PICKLE_MATCH)?;

        let positive_ids = [
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ];
        let negative_ids = [
            "pytorch_model-000001-of-00001.bin",
            "pytorch_model-0000a-of-00002.bin",
            "pytorch_model-000-of-00003.bin",
            "pytorch_consolidated.bin",
        ];
        for id in positive_ids {
            assert!(pickle_match.is_match(id));
        }
        for id in negative_ids {
            assert!(!pickle_match.is_match(id));
        }
        Ok(())
    }
}
