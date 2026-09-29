//! Resolves a quantization level (`--quant`, or `quant` in a spec) against what a repository publishes.

mod gguf_discovery;

use std::path::Path;

use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};

pub use gguf_discovery::{
    has_gguf_model_files, list_local_files_recursive, list_local_gguf_companions,
    resolve_gguf_projector, resolve_gguf_quant,
};

use crate::{
    AutoTuneRequest, ModelSelected, TokenSource, TuneProfile, auto_tune, parse_isq_value,
    parse_uqff_shard, probe_hf_repo_files, resolve_uqff_report_output, resolve_uqff_shorthand,
    try_get_model_file,
};
use inference_quant::UqffReport;

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

/// An auto-detected model with its `quant` resolved: what to load, and the ISQ level to apply to it.
pub struct ResolvedModelQuant {
    pub model: ModelSelected,
    pub isq: Option<String>,
    /// The id given, when resolution moved to another repository (a published UQFF).
    pub requested_model_id: Option<String>,
}

/// Resolves `Run.quant` to a published GGUF variant, else a published UQFF (own or `-UQFF` sibling), else ISQ.
pub async fn resolve_model_quant(
    model: ModelSelected,
    token_source: &TokenSource,
    force_cpu: bool,
) -> Result<ResolvedModelQuant> {
    let (model_id, quant) = match &model {
        ModelSelected::Run {
            model_id,
            quant: Some(quant),
            from_uqff,
            ..
        } => {
            if from_uqff.is_some() {
                anyhow::bail!("`quant` and `from_uqff` both pick the weights; give one");
            }
            (model_id.clone(), quant.clone())
        }
        _ => {
            return Ok(ResolvedModelQuant {
                model,
                isq: None,
                requested_model_id: None,
            });
        }
    };
    if let ModelSelected::Run {
        hf_cache_path: Some(path),
        ..
    } = &model
    {
        crate::set_hf_cache_path(path.clone());
    }
    let files = selected_model_files(&model_id, None, token_source)?;
    if let Some(files) = files.as_ref().filter(|files| {
        has_gguf_model_files(files) && is_confident_gguf_artifact_repo(&model_id, files)
    }) {
        let artifact = resolve_gguf_quant(files, &quant)?;
        info!(
            "quant: {quant} -> GGUF {} from `{model_id}`",
            artifact.label
        );
        return Ok(ResolvedModelQuant {
            model: run_as_gguf(model, files, artifact.file_spec())?,
            isq: None,
            requested_model_id: None,
        });
    }
    if model_name_looks_gguf(&model_id) {
        anyhow::bail!(
            "Model `{model_id}` appears to be a GGUF artifact repo, but its files could not be inspected or no \
             model GGUF was found; name the file explicitly or check repository access."
        );
    }
    let resolved = resolve_quant(&quant, &model_id, token_source, &model, force_cpu).await?;
    let mut model = model;
    let ModelSelected::Run {
        model_id: run_id,
        quant: run_quant,
        from_uqff,
        ..
    } = &mut model
    else {
        unreachable!("matched as Run above")
    };
    *run_quant = None;
    *from_uqff = resolved.from_uqff;
    let requested_model_id = resolved
        .model_id_swap
        .map(|swap| std::mem::replace(run_id, swap));
    Ok(ResolvedModelQuant {
        model,
        isq: resolved.in_situ_quant,
        requested_model_id,
    })
}

fn run_as_gguf(
    model: ModelSelected,
    files: &[String],
    quantized_filename: String,
) -> Result<ModelSelected> {
    let ModelSelected::Run {
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
    } = model
    else {
        unreachable!("only a Run carries `quant`")
    };
    let mmproj_filename = resolve_gguf_projector(files, dtype)?.map(|projector| {
        info!(
            "GGUF: selected {} projector `{}`",
            projector.label,
            projector.file_spec()
        );
        projector.file_spec()
    });
    Ok(ModelSelected::GGUF {
        tok_model_id: None,
        quantized_model_id: model_id,
        quantized_filename,
        tokenizer_json,
        mmproj_filename,
        lora_adapters: Vec::new(),
        lora_runtime_config: None,
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
    })
}

pub async fn resolve_quant(
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
pub async fn read_existing_uqff_report(
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
    Ok(crate::probe_hf_repo_files(
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
        let resolved =
            resolve_model_quant(run_with_quant(dir.path(), "4"), &TokenSource::None, true)
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

    #[tokio::test]
    async fn run_quant_falls_back_to_isq_for_a_local_source_model() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), []).unwrap();
        let resolved =
            resolve_model_quant(run_with_quant(dir.path(), "q4k"), &TokenSource::None, true)
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
        let error = resolve_model_quant(model, &TokenSource::None, true)
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
