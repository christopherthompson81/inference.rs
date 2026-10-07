//! Resolves a quantization level (`--quant`, or `quant` in a spec) against what a repository publishes.

mod gguf_discovery;

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use futures::future::BoxFuture;
use tracing::{debug, info, warn};

pub use gguf_discovery::{
    has_gguf_model_files, list_local_files_recursive, list_local_gguf_companions,
    resolve_gguf_projector, resolve_gguf_quant,
};

use inference_core::{
    TokenSource, parse_isq_value, parse_uqff_shard, probe_hf_repo_files,
    resolve_uqff_report_output, resolve_uqff_shorthand, try_get_model_file,
};
use inference_quant::UqffReport;

use crate::{AutoTuneRequest, MmprojSelection, ModelSelected, TuneProfile, auto_tune};

const DEFAULT_REVISION: &str = "main";
const UQFF_REPO_ORG: &str = "inference-community";
const UQFF_REPO_SUFFIX: &str = "-UQFF";
const UQFF_REPO_SUFFIX_LOWER: &str = "-uqff";
const UQFF_RESIDUAL_SAFETENSORS: &str = "residual.safetensors";

#[derive(Default, Debug, Clone)]
pub struct ResolvedQuant {
    pub model_id_swap: Option<String>,
    pub from_uqff: Option<String>,
    pub in_situ_quant: Option<String>,
}

const QUANT_WITH_FROM_UQFF: &str = "`quant` and `from_uqff` both pick the weights; give one";
const QUANT_WITH_FILENAME: &str =
    "`quant` (`--quant`) and `quantized_filename` (`-f`) both pick the GGUF file; give one";
const GGUF_WITHOUT_FILE: &str = "a GGUF model needs a file: give `quantized_filename` (`-f`), or `quant` to pick one \
                                 from the repository";

/// What `quant` may pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantPolicy {
    /// The weights to load: a published GGUF, else a published UQFF, else ISQ at that level.
    Weights,
    /// Only an input GGUF, for a run that requantizes it.
    GgufInput,
}

/// A spec with its `quant` and GGUF projector resolved: what to load, and the ISQ level to apply to it.
pub struct ResolvedModelQuant {
    pub model: ModelSelected,
    pub isq: Option<String>,
    /// The id given, when resolution moved to another repository (a published UQFF).
    pub requested_model_id: Option<String>,
}

impl ResolvedModelQuant {
    fn unchanged(model: ModelSelected) -> Self {
        Self {
            model,
            isq: None,
            requested_model_id: None,
        }
    }
}

/// Resolves a spec's `quant` against what its repository publishes, and picks the GGUF projector it asks for.
pub fn resolve_model_source<'a>(
    model: ModelSelected,
    token_source: &'a TokenSource,
    force_cpu: bool,
    policy: QuantPolicy,
) -> BoxFuture<'a, Result<ResolvedModelQuant>> {
    Box::pin(resolve_model_source_inner(
        model,
        token_source,
        force_cpu,
        policy,
    ))
}

async fn resolve_model_source_inner(
    model: ModelSelected,
    token_source: &TokenSource,
    force_cpu: bool,
    policy: QuantPolicy,
) -> Result<ResolvedModelQuant> {
    if !model.needs_source_resolution() {
        return Ok(ResolvedModelQuant::unchanged(model));
    }
    if let Some(path) = hf_cache_path(&model) {
        inference_core::set_hf_cache_path(path.clone());
    }
    if matches!(model, ModelSelected::GGUF { .. }) {
        return resolve_gguf_spec(model, token_source).map(ResolvedModelQuant::unchanged);
    }
    let quant = model
        .quant()
        .expect("only `quant` leaves a non-GGUF spec unresolved")
        .to_string();
    let (model_id, from_uqff) = source_of(&model)?;
    let model_id = model_id.to_string();
    if from_uqff {
        bail!(QUANT_WITH_FROM_UQFF);
    }
    let files = selected_model_files(&model_id, None, token_source)?;
    if policy == QuantPolicy::GgufInput {
        require_gguf_input(&model_id, &quant, files.as_deref())?;
    }
    if let Some(files) = files.as_ref().filter(|files| {
        has_gguf_model_files(files) && is_confident_gguf_artifact_repo(&model_id, files)
    }) {
        let artifact = resolve_gguf_quant(files, &quant)?;
        info!(
            "quant: {quant} -> GGUF {} from `{model_id}`",
            artifact.label
        );
        let mut model = into_gguf(model, artifact.file_spec())?;
        if matches!(model, ModelSelected::GGUF { .. }) {
            pick_projector(&mut model, files, true, true)?;
        }
        return Ok(ResolvedModelQuant::unchanged(model));
    }
    if model_name_looks_gguf(&model_id) {
        bail!(
            "Model `{model_id}` appears to be a GGUF artifact repo, but its files could not be inspected or no \
             model GGUF was found; name the file explicitly or check repository access."
        );
    }
    let resolved = resolve_quant(&quant, &model_id, token_source, &model, force_cpu).await?;
    let mut model = model;
    let (id, quant, from_uqff) = weights_fields_mut(&mut model);
    *quant = None;
    *from_uqff = resolved.from_uqff;
    let requested_model_id = resolved
        .model_id_swap
        .map(|swap| std::mem::replace(id, swap));
    Ok(ResolvedModelQuant {
        model,
        isq: resolved.in_situ_quant,
        requested_model_id,
    })
}

fn hf_cache_path(model: &ModelSelected) -> Option<&PathBuf> {
    match model {
        ModelSelected::Run { hf_cache_path, .. }
        | ModelSelected::Plain { hf_cache_path, .. }
        | ModelSelected::Lora { hf_cache_path, .. }
        | ModelSelected::MultimodalPlain { hf_cache_path, .. }
        | ModelSelected::Embedding { hf_cache_path, .. }
        | ModelSelected::GGUF { hf_cache_path, .. } => hf_cache_path.as_ref(),
        _ => None,
    }
}

/// The repository a safetensors spec loads from, and whether it already names a UQFF.
fn source_of(model: &ModelSelected) -> Result<(&str, bool)> {
    match model {
        ModelSelected::Run {
            model_id,
            from_uqff,
            ..
        }
        | ModelSelected::Plain {
            model_id,
            from_uqff,
            ..
        }
        | ModelSelected::Lora {
            model_id,
            from_uqff,
            ..
        }
        | ModelSelected::MultimodalPlain {
            model_id,
            from_uqff,
            ..
        }
        | ModelSelected::Embedding {
            model_id,
            from_uqff,
            ..
        } => Ok((model_id, from_uqff.is_some())),
        _ => unreachable!("`ModelSelected::quant` covers only these kinds and the GGUF ones"),
    }
}

fn weights_fields_mut(
    model: &mut ModelSelected,
) -> (&mut String, &mut Option<String>, &mut Option<String>) {
    match model {
        ModelSelected::Run {
            model_id,
            quant,
            from_uqff,
            ..
        }
        | ModelSelected::Plain {
            model_id,
            quant,
            from_uqff,
            ..
        }
        | ModelSelected::Lora {
            model_id,
            quant,
            from_uqff,
            ..
        }
        | ModelSelected::MultimodalPlain {
            model_id,
            quant,
            from_uqff,
            ..
        }
        | ModelSelected::Embedding {
            model_id,
            quant,
            from_uqff,
            ..
        } => (model_id, quant, from_uqff),
        _ => unreachable!("`source_of` accepted the spec"),
    }
}

fn uninspectable(model_id: &str) -> anyhow::Error {
    anyhow!(
        "Could not inspect GGUF artifacts for `{model_id}`. Name the file (`-f`, or `quantized_filename` in a \
         spec) or check repository access."
    )
}

fn require_gguf_input(model_id: &str, quant: &str, files: Option<&[String]>) -> Result<()> {
    let files = files.ok_or_else(|| uninspectable(model_id))?;
    if !has_gguf_model_files(files) {
        bail!(
            "`quant = {quant}` selects an input GGUF artifact, but `{model_id}` has no model GGUF files"
        );
    }
    if !is_confident_gguf_artifact_repo(model_id, files) {
        bail!(
            "`{model_id}` contains GGUF files alongside another model format. Give the GGUF format \
             explicitly (`--format gguf`, or a `GGUF` spec) for `quant = {quant}` to pick the input artifact."
        );
    }
    Ok(())
}

fn gguf_file_fields_mut(model: &mut ModelSelected) -> (&str, &mut String, &mut Option<String>) {
    match model {
        ModelSelected::GGUF {
            quantized_model_id,
            quantized_filename,
            quant,
            ..
        } => (quantized_model_id, quantized_filename, quant),
        _ => unreachable!("called for a GGUF spec"),
    }
}

fn resolve_gguf_spec(
    mut model: ModelSelected,
    token_source: &TokenSource,
) -> Result<ModelSelected> {
    let wants_projector = matches!(
        &model,
        ModelSelected::GGUF { mmproj_filename: None, mmproj_selection, quant, .. }
            if quant.is_some() || *mmproj_selection != MmprojSelection::Given
    );
    let (model_id, quantized_filename, quant) = gguf_file_fields_mut(&mut model);
    let model_id = model_id.to_string();
    if quant.is_some() && !quantized_filename.is_empty() {
        bail!(QUANT_WITH_FILENAME);
    }
    if quant.is_none() && quantized_filename.is_empty() {
        bail!(GGUF_WITHOUT_FILE);
    }
    let files = if quant.is_some() || wants_projector {
        let exact = (!quantized_filename.is_empty()).then_some(quantized_filename.as_str());
        selected_model_files(&model_id, exact, token_source)?
    } else {
        None
    };
    let picked_by_quant = quant.is_some();
    if let Some(requested) = quant.take() {
        let files = files.as_ref().ok_or_else(|| uninspectable(&model_id))?;
        if !has_gguf_model_files(files) {
            bail!(
                "`quant = {requested}` picks a GGUF file, but `{model_id}` has no model GGUF files"
            );
        }
        let artifact = resolve_gguf_quant(files, &requested)?;
        info!(
            "quant: {requested} -> GGUF {} from `{model_id}`",
            artifact.label
        );
        *quantized_filename = artifact.file_spec();
    }
    if matches!(model, ModelSelected::GGUF { .. }) {
        let files = files.unwrap_or_default();
        let artifact_repo = is_confident_gguf_artifact_repo(&model_id, &files);
        pick_projector(&mut model, &files, artifact_repo, picked_by_quant)?;
    }
    Ok(model)
}

/// Picks the projector a GGUF spec asks for, then clears the resolution inputs so the spec loads as given.
fn pick_projector(
    model: &mut ModelSelected,
    files: &[String],
    artifact_repo: bool,
    picked_by_quant: bool,
) -> Result<()> {
    let ModelSelected::GGUF {
        quantized_model_id,
        mmproj_filename,
        mmproj_selection,
        dtype,
        ..
    } = model
    else {
        unreachable!("called for a GGUF spec")
    };
    let wanted = match mmproj_selection {
        MmprojSelection::Given => picked_by_quant && artifact_repo,
        MmprojSelection::ArtifactRepo => artifact_repo,
        MmprojSelection::Any | MmprojSelection::Required => true,
    };
    if mmproj_filename.is_none()
        && wanted
        && let Some(projector) = resolve_gguf_projector(files, *dtype)?
    {
        info!(
            "GGUF: selected {} projector `{}`",
            projector.label,
            projector.file_spec()
        );
        *mmproj_filename = Some(projector.file_spec());
    }
    if *mmproj_selection == MmprojSelection::Required && mmproj_filename.is_none() {
        bail!(
            "No companion projector was found for the multimodal GGUF `{quantized_model_id}`; name one \
             (`--mmproj`, or a GGUF spec's `mmproj_filename`)"
        );
    }
    *mmproj_selection = MmprojSelection::Given;
    Ok(())
}

/// The GGUF spec for a safetensors spec whose `quant` picked a published GGUF.
fn into_gguf(model: ModelSelected, quantized_filename: String) -> Result<ModelSelected> {
    Ok(match model {
        ModelSelected::Run {
            model_id,
            tokenizer_json,
            dtype,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_edge,
            max_seq_len,
            max_batch_size,
            max_num_images,
            max_image_length,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            ..
        } => GgufBase {
            quantized_model_id: model_id,
            quantized_filename,
            tokenizer_json,
            dtype,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_edge,
            max_seq_len,
            max_batch_size,
            max_num_images,
            max_image_length,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            ..GgufBase::default()
        }
        .into_spec(),
        ModelSelected::Plain {
            model_id,
            tokenizer_json,
            dtype,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_seq_len,
            max_batch_size,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            ..
        } => GgufBase {
            quantized_model_id: model_id,
            quantized_filename,
            tokenizer_json,
            dtype,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_seq_len,
            max_batch_size,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            ..GgufBase::default()
        }
        .into_spec(),
        ModelSelected::Lora {
            model_id,
            tokenizer_json,
            adapters,
            runtime_config,
            mmproj_selection,
            dtype,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_edge,
            max_seq_len,
            max_batch_size,
            max_num_images,
            max_image_length,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            ..
        } => GgufBase {
            quantized_model_id: model_id,
            quantized_filename,
            tokenizer_json,
            lora_adapters: adapters,
            lora_runtime_config: Some(runtime_config),
            mmproj_selection,
            dtype,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_edge,
            max_seq_len,
            max_batch_size,
            max_num_images,
            max_image_length,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
        }
        .into_spec(),
        ModelSelected::MultimodalPlain {
            model_id,
            tokenizer_json,
            dtype,
            topology,
            write_uqff,
            max_edge,
            calibration_file,
            imatrix,
            max_seq_len,
            max_batch_size,
            max_num_images,
            max_image_length,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            organization,
            ..
        } => GgufBase {
            quantized_model_id: model_id,
            quantized_filename,
            tokenizer_json,
            mmproj_selection: MmprojSelection::Required,
            dtype,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_edge,
            max_seq_len,
            max_batch_size,
            max_num_images: Some(max_num_images),
            max_image_length: Some(max_image_length),
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            ..GgufBase::default()
        }
        .into_spec(),
        ModelSelected::Embedding { model_id, .. } => bail!(
            "`quant` picked a GGUF file from `{model_id}`, but embedding models do not load from GGUF"
        ),
        _ => unreachable!("`source_of` accepted the spec"),
    })
}

/// The fields of a `ModelSelected::GGUF` a safetensors spec carries over.
#[derive(Default)]
struct GgufBase {
    quantized_model_id: String,
    quantized_filename: String,
    tokenizer_json: Option<String>,
    lora_adapters: Vec<inference_core::LoraAdapterSpec>,
    lora_runtime_config: Option<inference_core::LoraRuntimeConfig>,
    mmproj_selection: MmprojSelection,
    dtype: inference_core::ModelDType,
    topology: Option<String>,
    organization: Option<inference_core::IsqOrganization>,
    write_uqff: Option<inference_core::UqffWriteConfig>,
    imatrix: Option<PathBuf>,
    calibration_file: Option<PathBuf>,
    max_edge: Option<u32>,
    max_seq_len: usize,
    max_batch_size: usize,
    max_num_images: Option<usize>,
    max_image_length: Option<usize>,
    hf_cache_path: Option<PathBuf>,
    matformer_config_path: Option<PathBuf>,
    matformer_slice_name: Option<String>,
}

impl GgufBase {
    fn into_spec(self) -> ModelSelected {
        ModelSelected::GGUF {
            tok_model_id: None,
            quantized_model_id: self.quantized_model_id,
            quantized_filename: self.quantized_filename,
            quant: None,
            tokenizer_json: self.tokenizer_json,
            mmproj_filename: None,
            mmproj_selection: self.mmproj_selection,
            lora_adapters: self.lora_adapters,
            lora_runtime_config: self.lora_runtime_config,
            dtype: self.dtype,
            topology: self.topology,
            organization: self.organization,
            write_uqff: self.write_uqff,
            imatrix: self.imatrix,
            calibration_file: self.calibration_file,
            max_edge: self.max_edge,
            max_seq_len: self.max_seq_len,
            max_batch_size: self.max_batch_size,
            max_num_images: self.max_num_images,
            max_image_length: self.max_image_length,
            hf_cache_path: self.hf_cache_path,
            matformer_config_path: self.matformer_config_path,
            matformer_slice_name: self.matformer_slice_name,
        }
    }
}

pub fn resolve_quant<'a>(
    raw: &'a str,
    model_id: &'a str,
    token_source: &'a TokenSource,
    model_selected: &'a ModelSelected,
    force_cpu: bool,
) -> BoxFuture<'a, Result<ResolvedQuant>> {
    Box::pin(resolve_quant_inner(
        raw,
        model_id,
        token_source,
        model_selected,
        force_cpu,
    ))
}

async fn resolve_quant_inner(
    raw: &str,
    model_id: &str,
    token_source: &TokenSource,
    model_selected: &ModelSelected,
    force_cpu: bool,
) -> Result<ResolvedQuant> {
    let lowered = raw.trim().to_lowercase();
    if lowered == "auto" {
        return resolve_auto(model_id, token_source, model_selected, force_cpu).await;
    }
    resolve_explicit(&lowered, model_id, token_source).await
}

async fn resolve_auto(
    model_id: &str,
    token_source: &TokenSource,
    model_selected: &ModelSelected,
    force_cpu: bool,
) -> Result<ResolvedQuant> {
    debug!("quant: auto, probing hardware via `tune`");
    let result = auto_tune(AutoTuneRequest {
        model: model_selected.clone(),
        token_source: token_source.clone(),
        hf_revision: None,
        force_cpu,
        profile: TuneProfile::Balanced,
        requested_isq: None,
    })
    .map_err(|e| anyhow!("`quant = auto` failed during tune analysis: {e}"))?;

    let Some(isq) = result.recommended_isq else {
        info!("quant: auto -> full precision (model fits)");
        return Ok(ResolvedQuant::default());
    };
    let isq_name = format!("{isq:?}").to_lowercase();
    // Logged to a tenth of a GB, so the cast's lost precision never shows.
    #[allow(clippy::cast_precision_loss)]
    let vram_gb = result.total_vram_bytes as f64 / 1e9;
    info!(
        "quant: auto -> {isq_name} (backend={}, vram={vram_gb:.1} GB)",
        result.backend,
    );
    resolve_explicit(&isq_name, model_id, token_source).await
}

async fn resolve_explicit(
    raw: &str,
    model_id: &str,
    token_source: &TokenSource,
) -> Result<ResolvedQuant> {
    parse_isq_value(raw, None)
        .map_err(|e| anyhow!("`quant = {raw}` is not a recognized quant level: {e}"))?;

    let selected_files = selected_repo_files(model_id, token_source)?;
    if let Some(files) = &selected_files {
        let report =
            read_existing_uqff_report(model_id, DEFAULT_REVISION, files, token_source).await?;
        if let Some(resolved) = resolve_selected_uqff(raw, model_id, files, report.as_ref())? {
            return Ok(resolved);
        }
    } else if model_name_looks_uqff(model_id) {
        anyhow::bail!(
            "Model `{model_id}` appears to be a UQFF artifact repo, but its file listing could not \
             be inspected. Use `from_uqff = {raw}` to load a known local/cached artifact, or check \
             repository access."
        );
    }

    if Path::new(model_id).exists() {
        info!("quant: {raw} -> ISQ {raw} (local model)");
        return Ok(fallback_isq(raw));
    }

    let uqff_repo = sibling_uqff_repo(model_id);
    debug!("quant: probing prebuilt UQFF at `{uqff_repo}`");
    let Some(files) = probe_hf_repo_files(&uqff_repo, DEFAULT_REVISION, token_source) else {
        debug!("quant: no UQFF repo at `{uqff_repo}` (or unreachable)");
        info!("quant: {raw} -> ISQ {raw}");
        return Ok(fallback_isq(raw));
    };
    let report =
        read_existing_uqff_report(&uqff_repo, DEFAULT_REVISION, &files, token_source).await?;

    let Some(shorthand) = resolve_uqff_quant(raw, &files, report.as_ref())? else {
        let available: Vec<&String> = files.iter().filter(|f| f.ends_with(".uqff")).collect();
        warn!(
            "quant: `{uqff_repo}` has no shard matching `{raw}` (available: {available:?}); falling back to ISQ {raw}"
        );
        return Ok(fallback_isq(raw));
    };

    info!(
        "quant: {raw} -> UQFF {shorthand} from `{uqff_repo}`; use `isq = {raw}` to \
         quantize the selected model source instead"
    );
    Ok(ResolvedQuant {
        model_id_swap: Some(uqff_repo),
        from_uqff: Some(shorthand),
        in_situ_quant: None,
    })
}

/// The UQFF report a repo publishes, when its file listing has one.
pub fn read_existing_uqff_report<'a>(
    model_id: &'a str,
    revision: &'a str,
    files: &'a [String],
    token_source: &'a TokenSource,
) -> BoxFuture<'a, Result<Option<UqffReport>>> {
    Box::pin(read_existing_uqff_report_inner(
        model_id,
        revision,
        files,
        token_source,
    ))
}

async fn read_existing_uqff_report_inner(
    model_id: &str,
    revision: &str,
    files: &[String],
    token_source: &TokenSource,
) -> Result<Option<UqffReport>> {
    if !files
        .iter()
        .any(|file| file == inference_quant::UQFF_REPORT_JSON)
    {
        return Ok(None);
    }
    let Some(path) = try_get_model_file(
        model_id,
        revision,
        inference_quant::UQFF_REPORT_JSON,
        token_source,
    )
    .await?
    else {
        return Ok(None);
    };
    let data = tokio::fs::read_to_string(&path).await?;
    serde_json::from_str(&data)
        .map(Some)
        .map_err(|e| anyhow!("{}: {e}", path.display()))
}

fn sibling_uqff_repo(model_id: &str) -> String {
    let base = model_id.rsplit_once('/').map_or(model_id, |(_, n)| n);
    format!("{UQFF_REPO_ORG}/{base}{UQFF_REPO_SUFFIX}")
}

fn selected_repo_files(model_id: &str, token_source: &TokenSource) -> Result<Option<Vec<String>>> {
    let path = Path::new(model_id);
    if path.exists() {
        return Ok(Some(local_model_files(path)?));
    }
    Ok(probe_hf_repo_files(
        model_id,
        DEFAULT_REVISION,
        token_source,
    ))
}

fn local_model_files(model_path: &Path) -> Result<Vec<String>> {
    list_local_files_recursive(model_path)
}

fn resolve_selected_uqff(
    raw: &str,
    model_id: &str,
    files: &[String],
    report: Option<&UqffReport>,
) -> Result<Option<ResolvedQuant>> {
    if let Some(shorthand) = resolve_uqff_quant(raw, files, report)? {
        info!("quant: {raw} -> UQFF {shorthand} from selected model `{model_id}`");
        return Ok(Some(ResolvedQuant {
            model_id_swap: None,
            from_uqff: Some(shorthand),
            in_situ_quant: None,
        }));
    }

    if is_uqff_artifact_repo(model_id, files) {
        anyhow::bail!(
            "Model `{model_id}` appears to be a UQFF artifact repo, but no UQFF shard matched \
             `quant = {raw}`. Available UQFF files: {}. Use `from_uqff` to choose an \
             available artifact, or select the base model and use `isq = {raw}` to quantize it at \
             load time.",
            format_available_uqff(files)
        );
    }

    Ok(None)
}

fn resolve_uqff_quant(
    raw: &str,
    files: &[String],
    report: Option<&UqffReport>,
) -> Result<Option<String>> {
    if let Some(report) = report
        && let Some(output) = resolve_uqff_report_output(raw, files, report)?
    {
        return Ok(Some(output.quant.clone()));
    }
    Ok(resolve_uqff_shorthand(raw, files).map(|matched| uqff_shorthand_from_match(&matched)))
}

fn uqff_shorthand_from_match(matched: &str) -> String {
    parse_uqff_shard(matched)
        .map(|(name, _)| name)
        .unwrap_or_else(|| matched.to_string())
}

fn model_name_looks_uqff(model_id: &str) -> bool {
    model_id
        .rsplit_once('/')
        .map_or(model_id, |(_, name)| name)
        .to_ascii_lowercase()
        .ends_with(UQFF_REPO_SUFFIX_LOWER)
}

fn is_uqff_artifact_repo(model_id: &str, files: &[String]) -> bool {
    if model_name_looks_uqff(model_id) {
        return true;
    }

    let has_uqff = files.iter().any(|file| file.ends_with(".uqff"));
    let has_uqff_metadata = files
        .iter()
        .any(|file| file == inference_quant::UQFF_REPORT_JSON || file == UQFF_RESIDUAL_SAFETENSORS);
    let has_source_weights = files.iter().any(|file| {
        let lower = file.to_ascii_lowercase();
        (lower.ends_with(".safetensors") && lower != UQFF_RESIDUAL_SAFETENSORS)
            || lower.ends_with(".pth")
            || lower.ends_with(".pt")
            || lower.ends_with(".bin")
    });

    has_uqff_metadata || (has_uqff && !has_source_weights)
}

fn format_available_uqff(files: &[String]) -> String {
    let available = files
        .iter()
        .filter(|file| file.ends_with(".uqff"))
        .cloned()
        .collect::<Vec<_>>();
    if available.is_empty() {
        "none".to_string()
    } else {
        available.join(", ")
    }
}

fn fallback_isq(raw: &str) -> ResolvedQuant {
    ResolvedQuant {
        in_situ_quant: Some(raw.to_string()),
        ..Default::default()
    }
}

pub fn selected_model_files(
    model_id: &str,
    exact_file: Option<&str>,
    token_source: &TokenSource,
) -> Result<Option<Vec<String>>> {
    let path = Path::new(model_id);
    if path.exists() {
        if let Some(exact_file) = exact_file {
            return list_local_gguf_companions(path, exact_file).map(Some);
        }
        return list_local_files_recursive(path).map(Some);
    }
    Ok(inference_core::probe_hf_repo_files(
        model_id,
        DEFAULT_REVISION,
        token_source,
    ))
}

pub fn model_name_looks_gguf(model_id: &str) -> bool {
    model_id
        .rsplit_once('/')
        .map_or(model_id, |(_, name)| name)
        .to_ascii_lowercase()
        .ends_with("-gguf")
}

pub fn is_confident_gguf_artifact_repo(model_id: &str, files: &[String]) -> bool {
    if model_name_looks_gguf(model_id) {
        return true;
    }
    !files.iter().any(|file| {
        let lower = file.to_ascii_lowercase();
        lower.ends_with(".uqff")
            || lower.ends_with(".safetensors")
            || lower.ends_with(".pth")
            || lower.ends_with(".pt")
            || lower.ends_with(".bin")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_with_quant(dir: &Path, quant: &str) -> ModelSelected {
        serde_json::from_value(serde_json::json!({
            "Run": {"model_id": dir.to_string_lossy(), "quant": quant, "dtype": "f16"}
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn run_quant_picks_a_local_gguf_and_its_projector() {
        let dir = tempfile::tempdir().unwrap();
        for file in [
            "model-Q4_K_S.gguf",
            "model-Q4_K_M.gguf",
            "mmproj-BF16.gguf",
            "mmproj-F16.gguf",
        ] {
            std::fs::write(dir.path().join(file), []).unwrap();
        }
        let resolved = resolve_model_source(
            run_with_quant(dir.path(), "4"),
            &TokenSource::None,
            true,
            QuantPolicy::Weights,
        )
        .await
        .unwrap();
        let ModelSelected::GGUF {
            quantized_model_id,
            quantized_filename,
            mmproj_filename,
            ..
        } = resolved.model
        else {
            panic!("expected a GGUF model");
        };
        assert_eq!(quantized_model_id, dir.path().to_string_lossy());
        assert_eq!(quantized_filename, "model-Q4_K_M.gguf");
        assert_eq!(mmproj_filename.as_deref(), Some("mmproj-F16.gguf"));
        assert!(resolved.isq.is_none());
        assert!(resolved.requested_model_id.is_none());
    }

    fn dir_with(files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for file in files {
            let path = dir.path().join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, []).unwrap();
        }
        dir
    }

    fn spec(kind: &str, fields: serde_json::Value) -> ModelSelected {
        serde_json::from_value(serde_json::json!({ kind: fields })).unwrap()
    }

    fn gguf_spec(dir: &Path, fields: serde_json::Value) -> ModelSelected {
        let mut fields = fields;
        fields["quantized_model_id"] = dir.to_string_lossy().into();
        spec("GGUF", fields)
    }

    async fn resolve(model: ModelSelected, policy: QuantPolicy) -> Result<ResolvedModelQuant> {
        resolve_model_source(model, &TokenSource::None, true, policy).await
    }

    fn gguf_files(model: &ModelSelected) -> (&str, Option<&str>) {
        let ModelSelected::GGUF {
            quantized_filename,
            mmproj_filename,
            quant,
            mmproj_selection,
            ..
        } = model
        else {
            panic!("expected a GGUF model, got {model:?}");
        };
        assert!(quant.is_none());
        assert_eq!(*mmproj_selection, MmprojSelection::Given);
        assert!(!model.needs_source_resolution());
        (quantized_filename, mmproj_filename.as_deref())
    }

    #[tokio::test]
    async fn run_quant_picks_vision_and_audio_projectors() {
        let dir = dir_with(&[
            "model-Q4_K_M.gguf",
            "model-vision-mmproj-BF16.gguf",
            "model-audio-mmproj-BF16.gguf",
        ]);
        let resolved = resolve(run_with_quant(dir.path(), "4"), QuantPolicy::Weights)
            .await
            .unwrap();
        assert_eq!(
            gguf_files(&resolved.model).1,
            Some("model-vision-mmproj-BF16.gguf;model-audio-mmproj-BF16.gguf")
        );
    }

    #[tokio::test]
    async fn lora_quant_keeps_the_dynamic_runtime_and_picks_a_projector() {
        let dir = dir_with(&["mmproj-BF16.gguf", "model-Q4_K_M.gguf"]);
        let model = spec(
            "Lora",
            serde_json::json!({
                "model_id": dir.path().to_string_lossy(), "quant": "4", "tokenizer_json": null,
                "arch": null, "topology": null, "write_uqff": null, "from_uqff": null, "hf_cache_path": null,
                "max_num_images": 3,
            }),
        );
        let resolved = resolve(model, QuantPolicy::Weights).await.unwrap();
        assert_eq!(
            gguf_files(&resolved.model),
            ("model-Q4_K_M.gguf", Some("mmproj-BF16.gguf"))
        );
        let ModelSelected::GGUF {
            lora_runtime_config,
            max_num_images,
            ..
        } = &resolved.model
        else {
            unreachable!()
        };
        assert!(lora_runtime_config.is_some());
        assert_eq!(*max_num_images, Some(3));
    }

    #[tokio::test]
    async fn multimodal_lora_quant_can_require_a_projector() {
        let dir = dir_with(&["model-Q4_K_M.gguf"]);
        let model = spec(
            "Lora",
            serde_json::json!({
                "model_id": dir.path().to_string_lossy(), "quant": "4", "mmproj_selection": "required",
                "tokenizer_json": null, "arch": null, "topology": null, "write_uqff": null, "from_uqff": null,
                "hf_cache_path": null,
            }),
        );
        let error = resolve(model, QuantPolicy::Weights).await.err().unwrap();
        assert!(
            error.to_string().contains("No companion projector"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn plain_quant_in_a_mixed_source_directory_falls_back_to_isq() {
        let dir = dir_with(&["model-Q4_K_M.gguf", "model.safetensors", "config.json"]);
        let model = spec(
            "Plain",
            serde_json::json!({"model_id": dir.path().to_string_lossy(), "quant": "q4k"}),
        );
        let resolved = resolve(model, QuantPolicy::Weights).await.unwrap();
        assert!(matches!(
            resolved.model,
            ModelSelected::Plain {
                quant: None,
                from_uqff: None,
                ..
            }
        ));
        assert_eq!(resolved.isq.as_deref(), Some("q4k"));
    }

    #[tokio::test]
    async fn multimodal_quant_requires_a_projector() {
        let dir = dir_with(&["model-Q4_K_M.gguf"]);
        let model = spec(
            "MultimodalPlain",
            serde_json::json!({
                "model_id": dir.path().to_string_lossy(), "quant": "4", "tokenizer_json": null, "arch": null,
                "topology": null, "write_uqff": null, "from_uqff": null, "max_edge": null,
                "calibration_file": null, "imatrix": null, "hf_cache_path": null,
                "matformer_config_path": null, "matformer_slice_name": null, "organization": null,
            }),
        );
        let error = resolve(model, QuantPolicy::Weights).await.err().unwrap();
        assert!(
            error.to_string().contains("No companion projector"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn embedding_quant_refuses_a_gguf_repository() {
        let dir = dir_with(&["model-Q4_K_M.gguf"]);
        let model = spec(
            "Embedding",
            serde_json::json!({"model_id": dir.path().to_string_lossy(), "quant": "4"}),
        );
        let error = resolve(model, QuantPolicy::Weights).await.err().unwrap();
        assert!(error.to_string().contains("embedding models"), "{error}");
    }

    #[tokio::test]
    async fn gguf_quant_picks_a_file_in_a_mixed_repository_without_guessing_a_projector() {
        let dir = dir_with(&["model-Q4_K_M.gguf", "mmproj-BF16.gguf", "model.safetensors"]);
        let resolved = resolve(
            gguf_spec(dir.path(), serde_json::json!({"quant": "4"})),
            QuantPolicy::Weights,
        )
        .await
        .unwrap();
        assert_eq!(gguf_files(&resolved.model), ("model-Q4_K_M.gguf", None));
    }

    #[tokio::test]
    async fn gguf_quant_does_not_fall_back_to_isq() {
        let dir = dir_with(&["model.safetensors"]);
        let error = resolve(
            gguf_spec(dir.path(), serde_json::json!({"quant": "4"})),
            QuantPolicy::Weights,
        )
        .await
        .err()
        .unwrap();
        assert!(error.to_string().contains("no model GGUF files"), "{error}");
    }

    #[tokio::test]
    async fn gguf_quant_and_filename_conflict_before_repository_access() {
        let model = spec(
            "GGUF",
            serde_json::json!({
                "quantized_model_id": "org/unreachable-GGUF", "quantized_filename": "model.gguf", "quant": "4",
            }),
        );
        let error = resolve(model, QuantPolicy::Weights).await.err().unwrap();
        assert_eq!(error.to_string(), QUANT_WITH_FILENAME);
    }

    #[tokio::test]
    async fn gguf_without_a_file_or_quant_is_refused() {
        let model = spec(
            "GGUF",
            serde_json::json!({"quantized_model_id": "org/unreachable-GGUF"}),
        );
        let error = resolve(model, QuantPolicy::Weights).await.err().unwrap();
        assert_eq!(error.to_string(), GGUF_WITHOUT_FILE);
    }

    #[tokio::test]
    async fn artifact_repo_selection_only_looks_beside_the_named_file() {
        let dir = dir_with(&[
            "selected/model.gguf",
            "selected/mmproj-BF16.gguf",
            "unrelated/model-Q4_K_M.gguf",
            "unrelated/mmproj-BF16.gguf",
        ]);
        let model = gguf_spec(
            dir.path(),
            serde_json::json!({"quantized_filename": "selected/model.gguf", "mmproj_selection": "artifact_repo"}),
        );
        let resolved = resolve(model, QuantPolicy::Weights).await.unwrap();
        assert_eq!(
            gguf_files(&resolved.model),
            ("selected/model.gguf", Some("selected/mmproj-BF16.gguf"))
        );
    }

    #[tokio::test]
    async fn artifact_repo_selection_does_not_guess_in_a_source_repository() {
        let dir = dir_with(&["model.gguf", "mmproj-BF16.gguf", "model.safetensors"]);
        let model = gguf_spec(
            dir.path(),
            serde_json::json!({"quantized_filename": "model.gguf", "mmproj_selection": "artifact_repo"}),
        );
        let resolved = resolve(model, QuantPolicy::Weights).await.unwrap();
        assert_eq!(gguf_files(&resolved.model), ("model.gguf", None));
    }

    #[tokio::test]
    async fn any_selection_takes_a_sibling_projector_in_a_source_directory() {
        let dir = dir_with(&[
            "model.gguf",
            "mmproj-BF16.gguf",
            "model.safetensors",
            "unrelated/mmproj-F16.gguf",
        ]);
        let model = gguf_spec(
            dir.path(),
            serde_json::json!({"quantized_filename": "model.gguf", "mmproj_selection": "any"}),
        );
        let resolved = resolve(model, QuantPolicy::Weights).await.unwrap();
        assert_eq!(
            gguf_files(&resolved.model),
            ("model.gguf", Some("mmproj-BF16.gguf"))
        );
    }

    #[tokio::test]
    async fn required_selection_fails_without_a_projector_and_yields_to_a_named_one() {
        let dir = dir_with(&["model.gguf", "model.safetensors"]);
        let required =
            serde_json::json!({"quantized_filename": "model.gguf", "mmproj_selection": "required"});
        let error = resolve(
            gguf_spec(dir.path(), required.clone()),
            QuantPolicy::Weights,
        )
        .await
        .err()
        .unwrap();
        assert!(
            error.to_string().contains("No companion projector"),
            "{error}"
        );

        let mut named = required;
        named["mmproj_filename"] = "chosen-mmproj-F16.gguf".into();
        let model = gguf_spec(dir.path(), named);
        assert!(!model.needs_source_resolution());
        let resolved = resolve(model, QuantPolicy::Weights).await.unwrap();
        let ModelSelected::GGUF {
            mmproj_filename, ..
        } = resolved.model
        else {
            unreachable!()
        };
        assert_eq!(mmproj_filename.as_deref(), Some("chosen-mmproj-F16.gguf"));
    }

    #[tokio::test]
    async fn gguf_input_policy_needs_a_gguf_artifact_repository() {
        let mixed = dir_with(&["model-Q4_K_M.gguf", "model.safetensors"]);
        let model = spec(
            "Plain",
            serde_json::json!({"model_id": mixed.path().to_string_lossy(), "quant": "4"}),
        );
        let error = resolve(model, QuantPolicy::GgufInput).await.err().unwrap();
        assert!(
            error.to_string().contains("alongside another model format"),
            "{error}"
        );

        let source = dir_with(&["model.safetensors"]);
        let model = spec(
            "Plain",
            serde_json::json!({"model_id": source.path().to_string_lossy(), "quant": "4"}),
        );
        let error = resolve(model, QuantPolicy::GgufInput).await.err().unwrap();
        assert!(error.to_string().contains("no model GGUF files"), "{error}");

        let artifacts = dir_with(&["model-Q4_K_M.gguf", "model-Q8_0.gguf"]);
        let model = spec(
            "Plain",
            serde_json::json!({"model_id": artifacts.path().to_string_lossy(), "quant": "8"}),
        );
        let resolved = resolve(model, QuantPolicy::GgufInput).await.unwrap();
        assert_eq!(gguf_files(&resolved.model), ("model-Q8_0.gguf", None));
    }

    #[test]
    fn unresolved_specs_are_refused_by_the_loader() {
        let model = spec(
            "GGUF",
            serde_json::json!({"quantized_model_id": "org/model-GGUF", "quant": "4"}),
        );
        let error = crate::LoaderBuilder::new(model).build().err().unwrap();
        assert!(
            error.to_string().contains("resolve_model_source"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn run_quant_falls_back_to_isq_for_a_local_source_model() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), []).unwrap();
        let resolved = resolve_model_source(
            run_with_quant(dir.path(), "q4k"),
            &TokenSource::None,
            true,
            QuantPolicy::Weights,
        )
        .await
        .unwrap();
        let ModelSelected::Run {
            quant, from_uqff, ..
        } = &resolved.model
        else {
            panic!("expected the Run model");
        };
        assert!(quant.is_none() && from_uqff.is_none());
        assert_eq!(resolved.isq.as_deref(), Some("q4k"));
        assert!(resolved.requested_model_id.is_none());
    }

    #[tokio::test]
    async fn run_quant_refuses_from_uqff() {
        let dir = tempfile::tempdir().unwrap();
        let mut model = run_with_quant(dir.path(), "4");
        let ModelSelected::Run { from_uqff, .. } = &mut model else {
            unreachable!()
        };
        *from_uqff = Some("q4k".to_string());
        let error = resolve_model_source(model, &TokenSource::None, true, QuantPolicy::Weights)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("from_uqff"), "{error}");
    }

    #[test]
    fn selected_uqff_repo_resolves_matching_quant() {
        let files = vec![
            "q8_0-0.uqff".to_string(),
            UQFF_RESIDUAL_SAFETENSORS.to_string(),
            "config.json".to_string(),
        ];

        let resolved = resolve_selected_uqff("8", "inference-community/foo-UQFF", &files, None)
            .unwrap()
            .unwrap();

        assert_eq!(resolved.model_id_swap, None);
        assert_eq!(resolved.from_uqff, Some("q8_0".to_string()));
        assert_eq!(resolved.in_situ_quant, None);
    }

    #[test]
    fn selected_uqff_repo_errors_on_unmatched_quant() {
        let files = vec![
            "q4k-0.uqff".to_string(),
            UQFF_RESIDUAL_SAFETENSORS.to_string(),
            "config.json".to_string(),
        ];

        let err = resolve_selected_uqff("8", "inference-community/foo-UQFF", &files, None)
            .unwrap_err()
            .to_string();

        assert!(err.contains("appears to be a UQFF artifact repo"));
        assert!(err.contains("q4k-0.uqff"));
    }

    #[test]
    fn selected_mixed_source_repo_uses_matching_uqff() {
        let files = vec![
            "model.safetensors".to_string(),
            "q8_0-0.uqff".to_string(),
            "config.json".to_string(),
        ];

        let resolved = resolve_selected_uqff("8", "org/foo", &files, None)
            .unwrap()
            .unwrap();

        assert_eq!(resolved.from_uqff, Some("q8_0".to_string()));
        assert_eq!(resolved.in_situ_quant, None);
    }

    #[test]
    fn selected_mixed_source_repo_allows_isq_fallback_without_match() {
        let files = vec![
            "model.safetensors".to_string(),
            "q4k-0.uqff".to_string(),
            "config.json".to_string(),
        ];

        assert!(
            resolve_selected_uqff("8", "org/foo", &files, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn selected_uqff_repo_resolves_report_quant_with_custom_shards() {
        let files = vec![
            "release-part-a.uqff".to_string(),
            "release-part-b.uqff".to_string(),
            inference_quant::UQFF_REPORT_JSON.to_string(),
            UQFF_RESIDUAL_SAFETENSORS.to_string(),
        ];
        let report = serde_json::from_value(serde_json::json!({
            "schema": 1,
            "generated_by": { "tool": "test" },
            "uqff_version": "0.1.0",
            "outputs": [{
                "quant": "q8_0",
                "shards": ["release-part-a.uqff", "release-part-b.uqff"],
                "layers": 0,
                "actual_counts": {},
                "fallback_count": 0
            }]
        }))
        .unwrap();

        for quant in ["q8_0", "8"] {
            let resolved =
                resolve_selected_uqff(quant, "inference-community/foo-UQFF", &files, Some(&report))
                    .unwrap()
                    .unwrap();
            assert_eq!(resolved.from_uqff, Some("q8_0".to_string()));
            assert_eq!(resolved.in_situ_quant, None);
        }
    }

    #[test]
    fn selected_uqff_metadata_marks_artifact_repo() {
        let files = vec![
            "q4k-0.uqff".to_string(),
            inference_quant::UQFF_REPORT_JSON.to_string(),
            "config.json".to_string(),
        ];

        assert!(is_uqff_artifact_repo("org/foo", &files));
    }
}
