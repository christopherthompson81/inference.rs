//! The offline calibration driver: tokenize a calibration file and drive text chunks
//! through any pipeline's model via [`CalibrationDrive`], collecting per-layer statistics.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use inference_quant::{QuantMethod, TrackedModule};
use inference_tensor::Device;
use tokenizers::Tokenizer;
use tracing::info;

use super::super::isq::load_imatrix_map;
use super::super::text_models_inputs_processor::{InputMetadata, make_prompt_chunk};
use super::super::{
    EitherCache, EmbeddingModel, ModelForwardContext, MultimodalModel, NormalModel,
};
use super::harvest_imatrix;

pub(crate) trait CalibrationDrive {
    fn calibration_forward(&self, inputs: &InputMetadata) -> inference_tensor::Result<()>;
    fn reset_cache(&self) -> inference_tensor::Result<()> {
        Ok(())
    }
    fn sliding_window(&self) -> Option<usize> {
        None
    }
}

pub(crate) struct NormalCalibrationDrive<'a>(pub &'a dyn NormalModel);

impl CalibrationDrive for NormalCalibrationDrive<'_> {
    fn calibration_forward(&self, inputs: &InputMetadata) -> inference_tensor::Result<()> {
        let input = inputs.input.to_device(self.0.device())?;
        let mut ctx = ModelForwardContext::new(
            &inputs.positions,
            &inputs.context_lens,
            &inputs.position_ids,
            None,
            &inputs.flash_meta,
        );
        self.0.forward(&input, &mut ctx)?;
        Ok(())
    }

    fn reset_cache(&self) -> inference_tensor::Result<()> {
        reset_either_cache(self.0.cache())
    }

    fn sliding_window(&self) -> Option<usize> {
        self.0.config().sliding_window
    }
}

pub(crate) struct MultimodalCalibrationDrive<'a>(pub &'a dyn MultimodalModel);

impl CalibrationDrive for MultimodalCalibrationDrive<'_> {
    fn calibration_forward(&self, inputs: &InputMetadata) -> inference_tensor::Result<()> {
        let input = inputs.input.to_device(self.0.device())?;
        let mut ctx = ModelForwardContext::new(
            &inputs.positions,
            &inputs.context_lens,
            &inputs.position_ids,
            None,
            &inputs.flash_meta,
        );
        // Text-only drive: the vision tower sees no calibration data, so its
        // layers quantize without imatrix weights.
        let args = self.0.default_model_specific_args(&input);
        self.0.forward(&input, None, args, &mut ctx)?;
        Ok(())
    }

    fn reset_cache(&self) -> inference_tensor::Result<()> {
        reset_either_cache(self.0.cache())
    }

    fn sliding_window(&self) -> Option<usize> {
        self.0.config().sliding_window
    }
}

pub(crate) struct EmbeddingCalibrationDrive<'a>(pub &'a dyn EmbeddingModel);

impl CalibrationDrive for EmbeddingCalibrationDrive<'_> {
    fn calibration_forward(&self, inputs: &InputMetadata) -> inference_tensor::Result<()> {
        let input = inputs.input.to_device(self.0.device())?;
        self.0
            .forward(&input, &Default::default(), &inputs.flash_meta)?;
        Ok(())
    }
}

fn reset_either_cache(cache: &EitherCache) -> inference_tensor::Result<()> {
    match cache {
        EitherCache::Full(full) => {
            for layer in &mut *full.lock() {
                *layer = None
            }
        }
        EitherCache::Normal(normal) => {
            for layer in &mut *normal.lock().unwrap().0 {
                layer.reset();
            }
        }
        EitherCache::Hybrid(hybrid) => {
            hybrid.lock().unwrap().reset()?;
        }
    }
    Ok(())
}

pub(crate) struct CalibrationCtx<'a> {
    pub tokenizer: &'a Tokenizer,
    pub bos_tok_id: Option<u32>,
    pub load_device: &'a Device,
    pub mapper: Option<&'a dyn crate::device_map::DeviceMapper>,
}

const CALIBRATION_CHUNK_SIZE: usize = 1024;
const TIED_EMBEDDING: &str = "embed_tokens";
const TIED_HEAD: &str = "lm_head";
// the Hessians one calibration pass collects at once (f32 `[in, in]` per layer); more layers take more passes
const HESSIAN_PASS_BYTES: usize = 4 << 30;

pub(super) fn calibration_tokens(
    calibration_file: &PathBuf,
    ctx: &CalibrationCtx<'_>,
) -> Result<Vec<u32>> {
    let calibration_data = std::fs::read_to_string(calibration_file)?;
    // Tokenize without bos; it is inserted per chunk
    Ok(ctx
        .tokenizer
        .encode_fast(calibration_data, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec())
}

// every chunk of `tokens` through the model once, the cache reset after each
pub(super) fn run_calibration(
    drive: &dyn CalibrationDrive,
    tokens: &[u32],
    ctx: &CalibrationCtx<'_>,
) -> Result<()> {
    let n_chunks = tokens.len().div_ceil(CALIBRATION_CHUNK_SIZE);
    let collect_start = std::time::Instant::now();
    for (i, chunk) in tokens.chunks(CALIBRATION_CHUNK_SIZE).enumerate() {
        let mut chunk = chunk.to_vec();
        if let Some(bos_tok_id) = ctx.bos_tok_id {
            chunk.insert(0, bos_tok_id);
        }
        let chunk_len = chunk.len();

        let chunk_start = std::time::Instant::now();
        let inputs = make_prompt_chunk(
            0,
            vec![&chunk],
            &[0],
            ctx.load_device,
            None,
            false,
            None,
            ctx.mapper,
            None,
            drive.sliding_window(),
            false,
        )?;
        drive.calibration_forward(&inputs)?;
        drive.reset_cache()?;

        info!(
            "Processed chunk {}/{n_chunks} ({chunk_len} tokens), {:.2}s",
            i + 1,
            chunk_start.elapsed().as_secs_f32()
        );
    }
    ctx.load_device.synchronize()?;
    info!(
        "Finished the calibration pass in {:.2}s",
        collect_start.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Each tracked layer's input Hessian (`x^T x` over the calibration file's rows, over their count) written as
/// `<key>.safetensors` holding `hin` (f32 `[in, in]`) under `dir`: the layout exllamav3's converter takes as
/// `--hessians`. Layers collect in passes whose Hessians fit `HESSIAN_PASS_BYTES`.
pub(crate) fn write_hessians(
    drive: &dyn CalibrationDrive,
    modules: &[TrackedModule],
    calibration_file: &PathBuf,
    dir: &PathBuf,
    ctx: &CalibrationCtx<'_>,
) -> Result<()> {
    let tokens = calibration_tokens(calibration_file, ctx)?;
    std::fs::create_dir_all(dir)?;
    let mut sized = Vec::with_capacity(modules.len());
    for module in modules {
        let layer = module.ct.resolve()?;
        let Some((weight, _)) = layer.unquant_weight_bias() else {
            anyhow::bail!(
                "`{}` holds no full-precision weight to collect a Hessian for",
                module.key
            );
        };
        // stacked experts have no single input Hessian
        if weight.rank() != 2 {
            continue;
        }
        let in_dim = weight.dim(1)?;
        sized.push((module, in_dim * in_dim * 4));
    }
    let mut passes: Vec<Vec<&TrackedModule>> = Vec::new();
    let mut bytes = 0;
    for (module, size) in sized {
        if passes.is_empty() || bytes + size > HESSIAN_PASS_BYTES {
            passes.push(Vec::new());
            bytes = 0;
        }
        bytes += size;
        passes.last_mut().unwrap().push(module);
    }
    info!(
        "Collecting Hessians for {} layers from `{}` ({} tokens) in {} pass(es).",
        passes.iter().map(Vec::len).sum::<usize>(),
        calibration_file.display(),
        tokens.len(),
        passes.len()
    );
    for pass in passes {
        for module in &pass {
            module.ct.begin_track_hessian()?;
        }
        run_calibration(drive, &tokens, ctx)?;
        for module in pass {
            // an embedding is read by lookup, never multiplied: no rows, and no Hessian to write
            if module
                .ct
                .stats_snapshot()
                .is_some_and(|(_, rows)| rows == 0)
            {
                let _ = module.ct.end_track_stats();
                continue;
            }
            let hessian = module
                .ct
                .end_track_stats()
                .map_err(|e| anyhow::anyhow!("`{}`: {e}", module.key))?
                .to_device(&Device::Cpu)?;
            // an embedding that recorded rows was multiplied as the tied lm_head, which is the key a converter uses
            let key = if module.key.ends_with(TIED_EMBEDDING) {
                TIED_HEAD
            } else {
                &module.key
            };
            let path = dir.join(format!("{key}.safetensors"));
            inference_tensor::safetensors::save(
                &HashMap::from([("hin".to_string(), hessian)]),
                &path,
            )?;
        }
    }
    Ok(())
}

/// Produce the per-layer imatrix map: collect via calibration forwards, or load from a file.
pub(crate) fn resolve_imatrix_map(
    drive: &dyn CalibrationDrive,
    modules: &[TrackedModule],
    imatrix_path: Option<&PathBuf>,
    calibration_file: Option<&PathBuf>,
    ctx: &CalibrationCtx<'_>,
) -> Result<HashMap<String, Vec<f32>>> {
    if let Some(calibration_file) = calibration_file {
        let tokens = calibration_tokens(calibration_file, ctx)?;
        info!(
            "Collecting imatrix from calibration file `{}` of {} tokens.",
            calibration_file.display(),
            tokens.len()
        );
        for module in modules {
            module.ct.begin_track_stats()?;
        }
        run_calibration(drive, &tokens, ctx)?;
        harvest_imatrix(modules)
    } else if let Some(imatrix_path) = imatrix_path {
        load_imatrix_map(imatrix_path, modules)
    } else {
        unreachable!("wants_imatrix requires imatrix or calibration_file")
    }
}
