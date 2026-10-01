//! Shared post-load ISQ orchestration. [`plan`] resolves flags into a capture plan,
//! [`drive`] runs offline calibration text through a model, [`online`] is the live
//! calibration lifecycle; the requantize-and-swap primitives here serve all of them
//! plus runtime re-ISQ.

mod drive;
mod online;
mod plan;

pub(crate) use drive::{
    CalibrationCtx, CalibrationDrive, EmbeddingCalibrationDrive, MultimodalCalibrationDrive,
    NormalCalibrationDrive, resolve_imatrix_map,
};
pub use online::CalibrationStatus;
pub(crate) use online::{apply_calibration, begin_calibration, calibration_status};
pub(crate) use plan::{
    AutoDeviceMapSizes, AutoDeviceMapSizingInputs, IsqLoadPlan, IsqPlanInputs,
    auto_device_map_sizes, resolve_and_install_isq_plan, resolve_auto_device_map_sizing,
    resolve_weight_load_dtype,
};

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use candle_core::Tensor;
use inference_quant::{IsqType, QuantMethod, TrackedModule};
use tracing::info;

use super::isq::{UqffFullSer, UqffWriteConfig, UqffWriteRequest, write_uqff_artifacts};

/// A UQFF to write once the model is loaded: where, and the tensors and files that are not quantized layers.
pub(crate) struct UqffArtifact<'a> {
    pub config: &'a UqffWriteConfig,
    pub residual: Vec<(String, Tensor)>,
    pub full_ser: UqffFullSer<'a>,
}

pub(crate) struct FinishIsqLoad<'a> {
    pub plan: &'a IsqLoadPlan,
    pub modules: Vec<TrackedModule>,
    pub drive: &'a dyn CalibrationDrive,
    pub in_situ_quant: Option<IsqType>,
    pub imatrix: Option<&'a PathBuf>,
    pub calibration_file: Option<&'a PathBuf>,
    pub calibration: CalibrationCtx<'a>,
    pub uqff: Option<UqffArtifact<'a>>,
}

/// After the weights are read: validate the ISQ selection, calibrate, capture, write the UQFF, then quantize.
pub(crate) fn finish_isq_load(inputs: FinishIsqLoad<'_>) -> Result<()> {
    let FinishIsqLoad {
        plan,
        modules,
        drive,
        in_situ_quant,
        imatrix,
        calibration_file,
        calibration,
        uqff,
    } = inputs;
    plan.validate_tracked_selection(&modules)?;
    let imatrix_map = if plan.wants_imatrix {
        Some(resolve_imatrix_map(
            drive,
            &modules,
            imatrix,
            calibration_file,
            &calibration,
        )?)
    } else {
        None
    };
    if plan.capture == inference_quant::IsqCaptureMode::CaptureMatches {
        let ty = in_situ_quant.context("imatrix quantization requires an ISQ type")?;
        complete_isq_capture(
            &modules,
            ty,
            imatrix_map
                .as_ref()
                .expect("CaptureMatches requires imatrix data"),
        )?;
    }
    if let Some(UqffArtifact {
        config,
        residual,
        full_ser,
    }) = uqff
    {
        let types = plan
            .write_types
            .clone()
            .filter(|types| !types.is_empty())
            .context("UQFF serialization requires at least one ISQ type.")?;
        write_uqff_artifacts(UqffWriteRequest {
            output: config.output.clone(),
            types,
            base_model: config.base_model.clone(),
            repo_id: config.repo_id.clone(),
            layers: modules.clone(),
            quantize_predicates: plan.uqff_quantize_predicates.clone(),
            residual,
            full_ser,
            imatrix: imatrix_map.unwrap_or_default(),
        })?;
    }
    if plan.immediate_isq_installed {
        for module in modules {
            module.ct.resolve()?;
        }
    }
    Ok(())
}

/// Runtime re-ISQ: requantizes a model's tracked layers to `dtype`; one loaded without ISQ has none to requantize.
pub(crate) fn requantize_tracked_modules(modules: &[TrackedModule], dtype: IsqType) -> Result<()> {
    if modules.is_empty() {
        anyhow::bail!("Runtime re-ISQ requires the model to have been loaded with ISQ.");
    }
    info!("Re-quantizing {} layers to {dtype}.", modules.len());
    requantize_and_swap(modules, dtype, |module| module.default_type(dtype), &|_| {
        None
    })
}

pub(crate) fn requantize_and_swap(
    modules: &[TrackedModule],
    pool_ty: IsqType,
    ty_for: impl Fn(&TrackedModule) -> IsqType,
    imatrix_for: &dyn Fn(&str) -> Option<Vec<f32>>,
) -> Result<()> {
    let handles = inference_quant::requantize_tracked(
        modules,
        pool_ty,
        ty_for,
        imatrix_for,
        inference_quant::IsqConsumer::RuntimeSwap,
        0,
        None,
    )?;
    // drain everything; failed layers keep their prior resident, so a partial swap stays consistent
    let mut errors: Vec<String> = Vec::new();
    for (module, rx) in modules.iter().zip(handles.receivers) {
        match rx.recv() {
            Ok(Ok(output)) => module.ct.replace(output.value),
            Ok(Err(e)) => errors.push(format!("{}: {e}", module.key)),
            Err(e) => errors.push(format!("{}: channel error: {e}", module.key)),
        }
    }
    if !errors.is_empty() {
        anyhow::bail!(
            "{} of {} layers failed to requantize; first: {}",
            errors.len(),
            modules.len(),
            errors[0]
        );
    }
    Ok(())
}

/// Finish a CaptureMatches load: quantize the deferred layers with imatrix data and swap them in.
pub(crate) fn complete_isq_capture(
    modules: &[TrackedModule],
    ty: IsqType,
    imatrix_map: &HashMap<String, Vec<f32>>,
) -> Result<()> {
    let missing = modules
        .iter()
        .filter(|module| !imatrix_map.contains_key(&module.key))
        .count();
    if missing > 0 {
        tracing::warn!(
            "{missing} of {} layers have no imatrix data; quantizing those without weights.",
            modules.len()
        );
    }
    info!("Quantizing {} layers to {ty} with imatrix.", modules.len());
    requantize_and_swap(modules, ty, |m| m.resolve_type(ty), &|key| {
        imatrix_map.get(key).cloned()
    })
}

/// Drain collected statistics into a key -> imatrix map; layers without data are absent.
fn harvest_imatrix(modules: &[TrackedModule]) -> Result<HashMap<String, Vec<f32>>> {
    let mut map = HashMap::new();
    for module in modules {
        if let Ok(stats) = module.ct.end_track_stats() {
            map.insert(module.key.clone(), stats.flatten_all()?.to_vec1::<f32>()?);
        }
    }
    Ok(map)
}

fn module_imatrix(
    module: &TrackedModule,
    pool_ty: IsqType,
    imatrix_map: &HashMap<String, Vec<f32>>,
) -> (IsqType, Option<Vec<f32>>) {
    let ty = module.resolve_type(pool_ty);
    let imatrix = ty
        .supports_imatrix()
        .then(|| imatrix_map.get(&module.key).cloned())
        .flatten();
    (ty, imatrix)
}
