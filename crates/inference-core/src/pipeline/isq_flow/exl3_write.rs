//! An EXL3 model written from a calibration file, a layer at a time as exllamav3's converter goes: each layer's
//! input Hessians are collected with every layer before it already swapped for its encoding, so its LDLQ sees (and
//! compensates) the error those leave; then the layer is encoded and swapped too. The output is exllamav3's layout:
//! each linear's `trellis`, `suh`, `svh` (and codebook marker) in place of its `weight`, every other tensor and file
//! copied, the config's `quantization_config` naming the rates.
#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use inference_quant::{QuantMethod, TrackedModule};
use inference_tensor::{Device, Tensor};
use tracing::info;

use super::drive::{
    CALIBRATION_CHUNK_SIZE, CalibrationCtx, CalibrationDrive, calibration_tokens, run_calibration,
};
use crate::pipeline::isq::Exl3WriteConfig;

const DEFAULT_HEAD_BITS: f64 = 6.0;
const DEFAULT_CODEBOOK: &str = "mul1";
const HEAD: &str = "lm_head";
// a MoE layer's router, kept in full precision as exllamav3 keeps it
const ROUTER: &str = ".gate";
const TIED_EMBEDDING: &str = "embed_tokens";
const LAYERS: &str = ".layers.";
const SHARED_INPUTS: &[(&str, &[&str])] = &[
    ("qkv", &["q_proj", "k_proj", "v_proj"]),
    ("gate_up", &["gate_proj", "up_proj"]),
];
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

#[cfg(feature = "cuda")]
struct Encoder {
    bits: inference_quant::exl3::Exl3Bits,
    head_bits: inference_quant::exl3::Exl3Bits,
    codebook: inference_quant::exl3::Exl3Codebook,
    marker: Option<(&'static str, u32)>,
    replaced: std::collections::HashSet<String>,
    replaced_prefixes: std::collections::HashSet<String>,
}

#[cfg(feature = "cuda")]
impl Encoder {
    // one linear encoded into `tensors` under `name`, and its reconstruction at its own size for the swap
    fn linear(
        &self,
        name: &str,
        weight: &Tensor,
        hessian: &Tensor,
        rate: inference_quant::exl3::Exl3Bits,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let encoding =
            inference_quant::exl3::encode::encode(weight, hessian, rate, self.codebook, fnv(name))?;
        self.store(name, weight, encoding, rate, tensors)
    }

    // `encoding` of `weight` stored under `name`, and its reconstruction at the weight's own size
    fn store(
        &self,
        name: &str,
        weight: &Tensor,
        encoding: inference_quant::exl3::encode::Exl3Encoding,
        rate: inference_quant::exl3::Exl3Bits,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let (rows, cols) = weight.dims2()?;
        info!("{name}: {rate:?}, proxy error {:.6}", encoding.proxy_error);
        // a side not in 128 blocks was padded for the encoding
        let dense = encoding
            .decoded
            .narrow(0, 0, rows)?
            .narrow(1, 0, cols)?
            .to_dtype(weight.dtype())?
            .contiguous()?
            .to_device(weight.device())?;
        tensors.insert(
            format!("{name}.trellis"),
            encoding.trellis.to_device(&Device::Cpu)?,
        );
        tensors.insert(format!("{name}.suh"), encoding.suh.to_device(&Device::Cpu)?);
        tensors.insert(format!("{name}.svh"), encoding.svh.to_device(&Device::Cpu)?);
        if let Some((marker, mult)) = self.marker {
            tensors.insert(
                format!("{name}.{marker}"),
                Tensor::new(&[mult as i32], &Device::Cpu)?,
            );
        }
        Ok(dense)
    }

    // every module of `group` (tracking Hessians since its pass began) encoded and swapped for its reconstruction
    fn modules(
        &mut self,
        group: &[&TrackedModule],
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        for module in group {
            if let Some(hessian) = take_hessian(module)? {
                self.module(module, &hessian, tensors)?;
            }
        }
        Ok(())
    }

    // `module` encoded against `hessian` and swapped for its reconstruction
    fn module(
        &mut self,
        module: &TrackedModule,
        hessian: &Tensor,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        use std::sync::Arc;

        use inference_quant::{QuantMethodConfig, UnquantLinear};

        let swap = |weight: Tensor, bias: Option<Tensor>| -> Result<()> {
            module
                .ct
                .replace(Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                    inference_tensor::nn::Linear::new(weight, bias),
                ))?) as Arc<dyn QuantMethod>);
            Ok(())
        };
        let layer = module.ct.resolve()?;
        let (weight, bias) = layer
            .unquant_weight_bias()
            .with_context(|| format!("`{}` holds no full-precision weight", module.key))?;
        // a 2D weight is a linear; a 3D one a projection of every expert of a MoE layer, stacked
        if weight.rank() == 3 {
            let (experts, proj) = module
                .key
                .rsplit_once('.')
                .with_context(|| format!("`{}` names no expert projection", module.key))?;
            // the experts encoded together, a row of tiles of each per quantizer launch
            let count = weight.dim(0)?;
            let names = (0..count)
                .map(|e| format!("{experts}.{e}.{proj}"))
                .collect::<Vec<_>>();
            let weights = (0..count)
                .map(|e| weight.get(e))
                .collect::<inference_tensor::Result<Vec<_>>>()?;
            // routed Hessians stay on the host; each moves to the device as its batch encodes
            let hessians = (0..count)
                .map(|e| hessian.get(e))
                .collect::<inference_tensor::Result<Vec<_>>>()?;
            let linears = (0..count)
                .map(|e| (&weights[e], &hessians[e], fnv(&names[e])))
                .collect::<Vec<_>>();
            let encodings =
                inference_quant::exl3::encode::encode_experts(&linears, self.bits, self.codebook)?;
            let mut denses = Vec::with_capacity(count);
            for ((name, weight), encoding) in names.iter().zip(&weights).zip(encodings) {
                denses.push(self.store(name, weight, encoding, self.bits, tensors)?);
            }
            swap(Tensor::stack(&denses, 0)?, bias)?;
            self.replaced_prefixes.insert(format!("{experts}."));
            return Ok(());
        }
        // a tied embedding that recorded rows was multiplied as the head; the lookups keep the embedding
        let tied = module.key.ends_with(TIED_EMBEDDING);
        let name = if tied { HEAD } else { module.key.as_str() };
        let rate = if name == HEAD {
            self.head_bits
        } else {
            self.bits
        };
        let dense = self.linear(name, &weight, hessian, rate, tensors)?;
        if !tied {
            swap(dense, bias)?;
        }
        // a checkpoint may store a tied head's weight as well; the encoding replaces it either way
        self.replaced.insert(format!("{name}.weight"));
        Ok(())
    }
}

// the Hessian `module` collected, or `None` when no row reached it (an embedding read only by lookup)
#[cfg(feature = "cuda")]
fn take_hessian(module: &TrackedModule) -> Result<Option<Tensor>> {
    if module
        .ct
        .stats_snapshot()
        .is_some_and(|(_, rows)| rows == 0)
    {
        let _ = module.ct.end_track_stats();
        return Ok(None);
    }
    Ok(Some(
        module
            .ct
            .end_track_stats()
            .map_err(|e| anyhow::anyhow!("`{}`: {e}", module.key))?,
    ))
}

// the linears of a standard decoder layer that read one input, so collect one Hessian: q/k/v, and gate/up (of the
// dense MLP, the shared expert and the stacked experts alike, whose rows route the same)
fn shared_input(key: &str) -> Option<String> {
    let (parent, proj) = key.rsplit_once('.')?;
    let group = SHARED_INPUTS
        .iter()
        .find(|(_, projs)| projs.contains(&proj))?
        .0;
    Some(format!("{parent}.{group}"))
}

/// The EXL3 model `config` names, from `modules` (every linear, captured unquantized) and the calibration file. A
/// model loaded on the host whose layers rebuild streams through the GPU a layer at a time; any other runs resident.
#[cfg(feature = "cuda")]
pub(crate) fn write_exl3(
    drive: &dyn CalibrationDrive,
    modules: &[TrackedModule],
    calibration_file: &PathBuf,
    config: &Exl3WriteConfig,
    source: &Exl3Source<'_>,
    ctx: &CalibrationCtx<'_>,
) -> Result<()> {
    let head_rate = config.head_bits.unwrap_or(DEFAULT_HEAD_BITS);
    let codebook_name = config.codebook.as_deref().unwrap_or(DEFAULT_CODEBOOK);
    let (codebook, marker) = codebook_of(codebook_name)?;
    let mut encoder = Encoder {
        bits: bits_of(config.bits)?,
        head_bits: bits_of(head_rate)?,
        codebook,
        marker,
        replaced: Default::default(),
        replaced_prefixes: Default::default(),
    };
    let tokens = calibration_tokens(calibration_file, ctx)?;
    let out = &config.output;
    std::fs::create_dir_all(out)?;
    info!(
        "Writing EXL3 at {} bits (head {head_rate}, {codebook_name}) from `{}` ({} tokens).",
        config.bits,
        calibration_file.display(),
        tokens.len(),
    );
    let mut weight_map = BTreeMap::new();
    match drive.layer_stream() {
        Some(stream) if ctx.load_device.is_cpu() => {
            write_streamed(stream, &tokens, ctx, &mut encoder, out, &mut weight_map)?
        }
        _ if ctx.load_device.is_cpu() => {
            anyhow::bail!(
                "this model does not rebuild its layers to stream them; load it on the GPU to write EXL3"
            )
        }
        _ => write_resident(
            drive,
            modules,
            &tokens,
            ctx,
            &mut encoder,
            out,
            &mut weight_map,
        )?,
    }
    copy_unquantized(
        source,
        out,
        &encoder.replaced,
        &encoder.replaced_prefixes,
        &mut weight_map,
    )?;
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

// the model resident on its devices: one calibration pass per layer, run up to it, its layers before it encoded
#[cfg(feature = "cuda")]
fn write_resident(
    drive: &dyn CalibrationDrive,
    modules: &[TrackedModule],
    tokens: &[u32],
    ctx: &CalibrationCtx<'_>,
    encoder: &mut Encoder,
    out: &Path,
    weight_map: &mut BTreeMap<String, String>,
) -> Result<()> {
    let mut groups: BTreeMap<(bool, usize), Vec<&TrackedModule>> = BTreeMap::new();
    for module in modules {
        let order = layer_of(&module.key).map_or((true, 0), |l| (false, l));
        groups.entry(order).or_default().push(module);
    }
    info!(
        "{} groups of layers, one calibration pass each.",
        groups.len()
    );
    for (shard, (&(rest, layer), group)) in groups.iter().enumerate() {
        let group = group
            .iter()
            .filter(|m| !m.key.ends_with(ROUTER))
            .copied()
            .collect::<Vec<_>>();
        for module in &group {
            module.ct.begin_track_hessian()?;
        }
        // a layer's pass needs the model only up to it (on the shared decoder stack; other models run whole)
        let _stop = (!rest).then(|| inference_nn::decoder::stop_after_layer(layer));
        run_calibration(drive, tokens, ctx)?;
        let mut tensors = HashMap::new();
        encoder.modules(&group, &mut tensors)?;
        write_shard(
            out,
            &format!("model-q{shard:05}.safetensors"),
            tensors,
            weight_map,
        )?;
    }
    Ok(())
}

// the model on the host, its hidden states over the calibration chunks kept there: each layer built on the GPU,
// its Hessians collected over the states, encoded, the states run on through the encoded layer, the layer dropped;
// then the head over the normed states
#[cfg(feature = "cuda")]
fn write_streamed(
    stream: &dyn inference_nn::model::LayerStream,
    tokens: &[u32],
    ctx: &CalibrationCtx<'_>,
    encoder: &mut Encoder,
    out: &Path,
    weight_map: &mut BTreeMap<String, String>,
) -> Result<()> {
    let gpu = Device::new_cuda(0)?;
    let mut states = Vec::new();
    for chunk in tokens.chunks(CALIBRATION_CHUNK_SIZE) {
        let ids = ctx
            .bos_tok_id
            .into_iter()
            .chain(chunk.iter().copied())
            .collect::<Vec<_>>();
        states.push(stream.embed(&Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?)?);
    }
    let layers = stream.num_layers();
    info!(
        "Streaming {layers} layers through {gpu:?}, {} chunks of hidden states on the host.",
        states.len()
    );
    for layer_idx in 0..layers {
        let start = std::time::Instant::now();
        let layer = stream.build_layer(layer_idx, &gpu)?;
        let group = layer
            .modules()
            .iter()
            .filter(|m| !m.key.ends_with(ROUTER))
            .collect::<Vec<_>>();
        // each module's Hessian source: itself, or the first module reading the same input
        let mut first = HashMap::new();
        let sources = group
            .iter()
            .enumerate()
            .map(|(i, m)| match shared_input(&m.key) {
                Some(input) => *first.entry(input).or_insert(i),
                None => i,
            })
            .collect::<Vec<_>>();
        for (i, module) in group.iter().enumerate() {
            if sources[i] == i {
                module.ct.begin_track_hessian()?;
            }
        }
        for state in &states {
            layer.forward(&state.to_device(&gpu)?)?;
        }
        let mut hessians = vec![None; group.len()];
        for (i, module) in group.iter().enumerate() {
            if sources[i] == i {
                // held on the host while the layer encodes, each moved to the device for its own linears
                hessians[i] = take_hessian(module)?
                    .map(|h| h.to_device(&Device::Cpu))
                    .transpose()?;
            }
        }
        let mut tensors = HashMap::new();
        for (i, module) in group.iter().enumerate() {
            if let Some(hessian) = &hessians[sources[i]] {
                encoder.module(module, hessian, &mut tensors)?;
            }
        }
        for state in &mut states {
            *state = layer
                .forward(&state.to_device(&gpu)?)?
                .to_device(&Device::Cpu)?;
        }
        drop(group);
        drop(layer);
        write_shard(
            out,
            &format!("model-q{layer_idx:05}.safetensors"),
            tensors,
            weight_map,
        )?;
        info!(
            "Layer {}/{layers} in {:.1}s.",
            layer_idx + 1,
            start.elapsed().as_secs_f32()
        );
    }
    let head = stream.head();
    let (weight, _) = head
        .unquant_weight_bias()
        .context("the head holds no full-precision weight")?;
    let weight = weight.to_device(&gpu)?;
    let in_dim = weight.dim(1)?;
    let mut hessian = Tensor::zeros((in_dim, in_dim), inference_tensor::DType::F32, &gpu)?;
    let mut rows = 0;
    for state in &states {
        let xs = stream
            .final_norm(state)?
            .to_device(&gpu)?
            .to_dtype(inference_tensor::DType::F32)?
            .reshape(((), in_dim))?;
        rows += xs.dim(0)?;
        hessian = (hessian + xs.t()?.matmul(&xs)?)?;
    }
    let hessian = (hessian / f64::from(u32::try_from(rows)?))?;
    let mut tensors = HashMap::new();
    encoder.linear(HEAD, &weight, &hessian, encoder.head_bits, &mut tensors)?;
    encoder.replaced.insert(format!("{HEAD}.weight"));
    write_shard(
        out,
        &format!("model-q{layers:05}.safetensors"),
        tensors,
        weight_map,
    )?;
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

// every source tensor but the replaced weights (and every tensor of the replaced experts), a shard for each source
// file
fn copy_unquantized(
    source: &Exl3Source<'_>,
    out: &Path,
    replaced: &std::collections::HashSet<String>,
    replaced_prefixes: &std::collections::HashSet<String>,
    weight_map: &mut BTreeMap<String, String>,
) -> Result<()> {
    for (i, file) in source.weights.iter().enumerate() {
        let tensors = inference_tensor::safetensors::load(file, &Device::Cpu)?
            .into_iter()
            .filter(|(name, _)| {
                !replaced.contains(name) && !replaced_prefixes.iter().any(|p| name.starts_with(p))
            })
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
