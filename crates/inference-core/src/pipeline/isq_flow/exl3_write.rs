//! An EXL3 model written from a calibration file, a layer at a time as exllamav3's converter goes: each layer's
//! input Hessians are collected with every layer before it already swapped for its encoding, so its LDLQ sees (and
//! compensates) the error those leave; then the layer is encoded and swapped too. The output is exllamav3's layout:
//! each linear's `trellis`, `suh`, `svh` (and codebook marker) in place of its `weight`, every other tensor and file
//! copied, the config's `quantization_config` naming the rates.
#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use inference_quant::TrackedModule;
use inference_tensor::{Device, Tensor};
use tracing::info;

use super::drive::{CalibrationCtx, CalibrationDrive, calibration_tokens, run_calibration};
use crate::pipeline::isq::Exl3WriteConfig;

const DEFAULT_HEAD_BITS: f64 = 6.0;
// the encoder rotates both sides in blocks this wide; a tensor in other sizes stays dense (the loader reads either)
const HAD_BLOCK: usize = 128;
const DEFAULT_CODEBOOK: &str = "mul1";
const HEAD: &str = "lm_head";
const TIED_EMBEDDING: &str = "embed_tokens";
const LAYERS: &str = ".layers.";
const INDEX_FILE: &str = "model.safetensors.index.json";
const CONFIG_FILE: &str = "config.json";
// exllamav3's version the layout follows
const LAYOUT_VERSION: &str = "1.6.6";
// the markers' stored multipliers (only their presence is read)
const MCG_MULT: u32 = 0xCBAC1FED;
const MUL1_MULT: u32 = 0x83DCD12D;

/// The checkpoint a write copies its unquantized tensors and files from.
pub(crate) struct Exl3Source<'a> {
    pub weights: &'a [PathBuf],
    pub config: &'a PathBuf,
}

// the layer a module belongs to, so layers run in order and the rest (the head) after them
fn layer_of(key: &str) -> Option<usize> {
    let rest = &key[key.find(LAYERS)? + LAYERS.len()..];
    rest.split('.').next()?.parse().ok()
}

fn fnv(key: &str) -> u64 {
    key.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(feature = "cuda")]
fn bits_of(rate: f64) -> Result<inference_quant::exl3::Exl3Bits> {
    use inference_quant::exl3::Exl3Bits;
    (1..=8u8)
        .map(|k| (f64::from(k), Exl3Bits::Whole(k.into())))
        .chain((1..=3u8).map(|k| (f64::from(k) + 0.5, Exl3Bits::Half(k.into()))))
        .find(|&(r, _)| r == rate)
        .map(|(_, bits)| bits)
        .with_context(|| format!("EXL3 rates are whole 1..8 or 1.5, 2.5, 3.5 bits, not {rate}"))
}

#[cfg(feature = "cuda")]
fn codebook_of(
    name: &str,
) -> Result<(
    inference_quant::exl3::Exl3Codebook,
    Option<(&'static str, u32)>,
)> {
    use inference_quant::exl3::Exl3Codebook;
    Ok(match name {
        "3inst" => (Exl3Codebook::ThreeInst, None),
        "mcg" => (Exl3Codebook::Mcg, Some(("mcg", MCG_MULT))),
        "mul1" => (Exl3Codebook::Mul1, Some(("mul1", MUL1_MULT))),
        _ => anyhow::bail!("EXL3 codebooks are 3inst, mcg and mul1, not `{name}`"),
    })
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn write_exl3(
    _: &dyn CalibrationDrive,
    _: &[TrackedModule],
    _: &PathBuf,
    _: &Exl3WriteConfig,
    _: &Exl3Source<'_>,
    _: &CalibrationCtx<'_>,
) -> Result<()> {
    anyhow::bail!("writing EXL3 runs on CUDA")
}

/// The EXL3 model `config` names, from `modules` (every linear, captured unquantized) and the calibration file.
#[cfg(feature = "cuda")]
pub(crate) fn write_exl3(
    drive: &dyn CalibrationDrive,
    modules: &[TrackedModule],
    calibration_file: &PathBuf,
    config: &Exl3WriteConfig,
    source: &Exl3Source<'_>,
    ctx: &CalibrationCtx<'_>,
) -> Result<()> {
    use std::sync::Arc;

    use inference_quant::exl3::{dense_weight, encode::encode};
    use inference_quant::{QuantMethod, QuantMethodConfig, UnquantLinear};

    let bits = bits_of(config.bits)?;
    let head_rate = config.head_bits.unwrap_or(DEFAULT_HEAD_BITS);
    let head_bits = bits_of(head_rate)?;
    let codebook_name = config.codebook.as_deref().unwrap_or(DEFAULT_CODEBOOK);
    let (codebook, marker) = codebook_of(codebook_name)?;
    let tokens = calibration_tokens(calibration_file, ctx)?;
    let out = &config.output;
    std::fs::create_dir_all(out)?;

    let mut groups: BTreeMap<(bool, usize), Vec<&TrackedModule>> = BTreeMap::new();
    for module in modules {
        let order = layer_of(&module.key).map_or((true, 0), |l| (false, l));
        groups.entry(order).or_default().push(module);
    }
    info!(
        "Writing EXL3 at {} bits (head {head_rate}, {codebook_name}) from `{}` ({} tokens): {} groups of layers, one \
         calibration pass each.",
        config.bits,
        calibration_file.display(),
        tokens.len(),
        groups.len()
    );
    let mut weight_map = BTreeMap::new();
    let mut replaced = std::collections::HashSet::new();
    for (shard, group) in groups.values().enumerate() {
        for module in group {
            module.ct.begin_track_hessian()?;
        }
        run_calibration(drive, &tokens, ctx)?;
        let mut tensors = HashMap::new();
        for module in group {
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
                .map_err(|e| anyhow::anyhow!("`{}`: {e}", module.key))?;
            let layer = module.ct.resolve()?;
            let (weight, bias) = layer
                .unquant_weight_bias()
                .with_context(|| format!("`{}` holds no full-precision weight", module.key))?;
            let (rows, cols) = weight.dims2()?;
            if rows % HAD_BLOCK != 0 || cols % HAD_BLOCK != 0 {
                info!(
                    "{}: {rows} x {cols} is not in {HAD_BLOCK}-wide blocks, kept dense",
                    module.key
                );
                continue;
            }
            // a tied embedding that recorded rows was multiplied as the head; the lookups keep the embedding
            let tied = module.key.ends_with(TIED_EMBEDDING);
            let name = if tied { HEAD } else { module.key.as_str() };
            let rate = if name == HEAD { head_bits } else { bits };
            let encoding = encode(&weight, &hessian, rate, codebook, fnv(name))?;
            info!("{name}: {rate:?}, proxy error {:.6}", encoding.proxy_error);
            if !tied {
                let dense = dense_weight(
                    &encoding.trellis,
                    &encoding.suh,
                    &encoding.svh,
                    codebook,
                    weight.dtype(),
                    weight.device(),
                )?;
                module
                    .ct
                    .replace(Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                        inference_tensor::nn::Linear::new(dense, bias),
                    ))?) as Arc<dyn QuantMethod>);
            }
            // a checkpoint may store a tied head's weight as well; the encoding replaces it either way
            replaced.insert(format!("{name}.weight"));
            tensors.insert(format!("{name}.trellis"), encoding.trellis);
            tensors.insert(format!("{name}.suh"), encoding.suh.to_device(&Device::Cpu)?);
            tensors.insert(format!("{name}.svh"), encoding.svh.to_device(&Device::Cpu)?);
            if let Some((marker, mult)) = marker {
                tensors.insert(
                    format!("{name}.{marker}"),
                    Tensor::new(&[mult as i32], &Device::Cpu)?,
                );
            }
        }
        write_shard(
            out,
            &format!("model-q{shard:05}.safetensors"),
            tensors,
            &mut weight_map,
        )?;
    }
    copy_unquantized(source, out, &replaced, &mut weight_map)?;
    std::fs::write(
        out.join(INDEX_FILE),
        serde_json::to_string_pretty(
            &serde_json::json!({ "metadata": {}, "weight_map": weight_map }),
        )?,
    )?;
    let mut model_config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(source.config)?)?;
    model_config["quantization_config"] = serde_json::json!({
        "quant_method": "exl3",
        "version": LAYOUT_VERSION,
        "bits": config.bits,
        "head_bits": head_rate,
        "calibration": { "tokens": tokens.len() },
        "out_scales": "always",
        "codebook": codebook_name,
    });
    std::fs::write(
        out.join(CONFIG_FILE),
        serde_json::to_string_pretty(&model_config)?,
    )?;
    copy_side_files(source.config, out)?;
    info!("Wrote the EXL3 model to `{}`.", out.display());
    Ok(())
}

fn write_shard(
    out: &Path,
    file: &str,
    tensors: HashMap<String, Tensor>,
    weight_map: &mut BTreeMap<String, String>,
) -> Result<()> {
    if tensors.is_empty() {
        return Ok(());
    }
    for name in tensors.keys() {
        weight_map.insert(name.clone(), file.to_string());
    }
    inference_tensor::safetensors::save(&tensors, out.join(file))?;
    Ok(())
}

// every source tensor but the replaced weights, a shard for each source file
fn copy_unquantized(
    source: &Exl3Source<'_>,
    out: &Path,
    replaced: &std::collections::HashSet<String>,
    weight_map: &mut BTreeMap<String, String>,
) -> Result<()> {
    for (i, file) in source.weights.iter().enumerate() {
        let tensors = inference_tensor::safetensors::load(file, &Device::Cpu)?
            .into_iter()
            .filter(|(name, _)| !replaced.contains(name))
            .collect::<HashMap<_, _>>();
        write_shard(
            out,
            &format!("model-u{i:05}.safetensors"),
            tensors,
            weight_map,
        )?;
    }
    Ok(())
}

// the tokenizer, generation config, chat template and the like beside the source config
fn copy_side_files(config: &Path, out: &Path) -> Result<()> {
    let Some(dir) = config.parent() else {
        return Ok(());
    };
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if path.is_file()
            && !name.ends_with(".safetensors")
            && name != INDEX_FILE
            && name != CONFIG_FILE
        {
            std::fs::copy(&path, out.join(name))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::layer_of;

    #[test]
    fn modules_order_by_their_layer() {
        assert_eq!(layer_of("model.layers.12.self_attn.q_proj"), Some(12));
        assert_eq!(
            layer_of("model.language_model.layers.3.mlp.down_proj"),
            Some(3)
        );
        assert_eq!(layer_of("lm_head"), None);
    }
}
