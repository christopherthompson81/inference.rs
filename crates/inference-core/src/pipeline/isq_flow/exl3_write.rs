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
// linears exllamav3 keeps in full precision: a MoE layer's router and the GDN's b and a
const KEPT_DENSE: &[&str] = &[".gate", ".in_proj_b", ".in_proj_a", ".in_proj_ba"];
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
// read-ahead of the next layer's weights: the read size, and the gap below which neighbouring tensors read as one
const PREFETCH_CHUNK: usize = 8 << 20;
const PREFETCH_MERGE_GAP: u64 = 1 << 20;
// stacked experts whose per-expert Hessians would take more than this are written as one, their mean
const SPLIT_EXPERT_HESSIANS_BYTES: usize = 1 << 30;
const GATE_UP_PROJS: &[&str] = &["gate_proj", "up_proj", "gate_up_proj"];
const SHARED_EXPERT: &str = "shared_expert";
const GGUF_MAGIC: &[u8] = b"GGUF";
const GGUF_BLOCK: &str = "blk.";
// set, stacked gate/up experts collect each expert's own routed Hessian (stored bf16) rather than the shared expert's
const PER_EXPERT_HESSIANS_ENV: &str = "INFERENCE_PER_EXPERT_HESSIANS";
// the hidden states Hessian collection keeps on the GPU at most (beyond, on the host)
const RESIDENT_STATE_BYTES: usize = 4 << 30;
// diagnostics: a streamed run records its final log-probabilities to this file, or compares its own with them
#[cfg(feature = "cuda")]
const STREAM_TWIN_FREE_ENV: &str = "INFERENCE_STREAM_TWIN_FREE";
// set to a directory (with a twin), each layer's MoE input in the model and the twin is written there, row for row
#[cfg(feature = "cuda")]
const STREAM_ROUTER_DUMP_ENV: &str = "INFERENCE_STREAM_ROUTER_DUMP";
#[cfg(feature = "cuda")]
const ROUTER_INPUT_MODULE: &str = "mlp.shared_expert.gate_proj";
// chunks a plain streamed pass runs through a layer in one call
const STREAM_BATCH_CHUNKS: usize = 8;
// set to a path, a streamed scoring run writes each run's scores there as JSON, with each chunk's mean KLD
#[cfg(feature = "cuda")]
const STREAM_SCORES_OUT_ENV: &str = "INFERENCE_STREAM_SCORES_OUT";
#[cfg(feature = "cuda")]
const STREAM_SCORE_ONLY_ENV: &str = "INFERENCE_STREAM_SCORE_ONLY";
#[cfg(feature = "cuda")]
const STREAM_LOGITS_OUT_ENV: &str = "INFERENCE_STREAM_LOGITS_OUT";
#[cfg(feature = "cuda")]
const STREAM_LOGITS_REF_ENV: &str = "INFERENCE_STREAM_LOGITS_REF";
#[cfg(feature = "cuda")]
const LOGP_TENSOR: &str = "logp";

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

// a shard's bytes `[start, end)`
type Run = (PathBuf, u64, u64);

// a GGUF tensor's layer, from its `blk.<n>.` name
fn gguf_layer_of(name: &str) -> Option<usize> {
    name.strip_prefix(GGUF_BLOCK)?
        .split('.')
        .next()?
        .parse()
        .ok()
}

// each layer's tensor bytes in the source shards (safetensors or GGUF), as runs with near neighbours merged
fn layer_ranges(weights: &[PathBuf]) -> Result<BTreeMap<usize, Vec<Run>>> {
    use std::io::Read;
    let mut layers: BTreeMap<usize, Vec<Run>> = BTreeMap::new();
    for path in weights {
        let mut file = std::fs::File::open(path)?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        if len.starts_with(GGUF_MAGIC) {
            let archive = inference_quant::GgufArchive::open_file(path)?;
            for (name, info) in archive.tensors() {
                if let (Some(layer), Some(range)) = (gguf_layer_of(name), info.data_range()) {
                    layers.entry(layer).or_default().push((
                        path.clone(),
                        range.start as u64,
                        range.end as u64,
                    ));
                }
            }
            continue;
        }
        let len = u64::from_le_bytes(len);
        let mut header = vec![0u8; usize::try_from(len)?];
        file.read_exact(&mut header)?;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)?;
        let base = 8 + len;
        for (name, info) in &header {
            let (Some(layer), Some([start, end])) = (
                layer_of(name),
                info.get("data_offsets")
                    .and_then(|o| serde_json::from_value::<[u64; 2]>(o.clone()).ok()),
            ) else {
                continue;
            };
            layers
                .entry(layer)
                .or_default()
                .push((path.clone(), base + start, base + end));
        }
    }
    for runs in layers.values_mut() {
        runs.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        let mut merged: Vec<Run> = Vec::with_capacity(runs.len());
        for (path, start, end) in runs.drain(..) {
            match merged.last_mut() {
                Some(last) if last.0 == path && start <= last.2 + PREFETCH_MERGE_GAP => {
                    last.2 = last.2.max(end);
                }
                _ => merged.push((path, start, end)),
            }
        }
        *runs = merged;
    }
    Ok(layers)
}

// reads `runs` so their pages are cached when the layer builds from the mapped shards; a failure only loses the hint
fn prefetch(runs: Vec<Run>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = vec![0u8; PREFETCH_CHUNK];
        for (path, start, end) in runs {
            if let Err(e) = read_run(&path, start, end, &mut buf) {
                tracing::debug!("reading ahead `{}` failed: {e}", path.display());
                return;
            }
        }
    })
}

fn read_run(path: &Path, start: u64, end: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    let mut left = end - start;
    while left > 0 {
        let n = usize::try_from(left).map_or(buf.len(), |l| l.min(buf.len()));
        file.read_exact(&mut buf[..n])?;
        left -= n as u64;
    }
    Ok(())
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
        Some(stream) if ctx.load_device.is_cpu() => write_streamed(
            stream,
            &tokens,
            ctx,
            &mut encoder,
            out,
            &mut weight_map,
            source.weights,
        )?,
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
            .copied()
            .filter(|m| encoded(m))
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
    weights: &[PathBuf],
) -> Result<()> {
    let gpu = Device::new_cuda(0)?;
    let mut ranges = layer_ranges(weights)?;
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
    let mut ahead: Option<std::thread::JoinHandle<()>> = None;
    for layer_idx in 0..layers {
        let start = std::time::Instant::now();
        // the read-ahead of this layer finishes before the next one's starts, so reads stay sequential
        if let Some(reading) = ahead.take() {
            let _ = reading.join();
        }
        ahead = ranges.remove(&(layer_idx + 1)).map(prefetch);
        let layer = stream.build_layer(layer_idx, &gpu)?;
        let group = layer
            .modules()
            .iter()
            .filter(|m| encoded(m))
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

// a module's weight rank (3 for stacked experts `[E, out, in]`)
fn weight_rank(module: &TrackedModule) -> Result<usize> {
    let layer = module.ct.resolve()?;
    // a quantized GGUF source's layers hold their blocks (collecting from a quantized model's activations)
    layer
        .unquant_weight_bias()
        .map(|(w, _)| w.rank())
        .or_else(|| layer.get_qtensor().map(|q| q.shape().rank()))
        .or_else(|| layer.weight_dims().map(|d| d.len()))
        .with_context(|| format!("`{}` holds no weight", module.key))
}

// a Hessian written as `<key>.safetensors` holding `hin`, exllamav3's `--hessians` layout
fn save_hessian(dir: &Path, key: &str, h: Tensor) -> Result<()> {
    inference_tensor::safetensors::save(
        &HashMap::from([(crate::gguf_imatrix::HESSIAN_TENSOR.to_string(), h)]),
        dir.join(format!("{key}.safetensors")),
    )?;
    Ok(())
}

// GPTAQ's statistics as `<key>.safetensors`: the twin's Hessian `hin` and the cross moment `dxx`
#[cfg(feature = "cuda")]
fn save_hessian_pair(dir: &Path, key: &str, h: Tensor, cross: Tensor) -> Result<()> {
    inference_tensor::safetensors::save(
        &HashMap::from([
            (crate::gguf_imatrix::HESSIAN_TENSOR.to_string(), h),
            (crate::gguf_imatrix::CROSS_TENSOR.to_string(), cross),
        ]),
        dir.join(format!("{key}.safetensors")),
    )?;
    Ok(())
}

// one dense input's GPTAQ statistics over the chunks: `X_q^T X_q` and `(X_fp - X_q)^T X_q`, `X_fp` the model's
// inputs to the linear and `X_q` the twin's, row for row (the twin routed as the model)
#[cfg(feature = "cuda")]
struct Gptaq {
    index: usize,
    model: TrackedModule,
    twin: TrackedModule,
    h: Option<Tensor>,
    cross: Option<Tensor>,
    rows: usize,
}

#[cfg(feature = "cuda")]
impl Gptaq {
    // this chunk's inputs of both, taken and folded in
    fn add_chunk(&mut self) -> Result<()> {
        use inference_tensor::DType;
        let model = self.model.ct.take_captured_inputs()?;
        let twin = self.twin.ct.take_captured_inputs()?;
        if model.len() != twin.len() {
            anyhow::bail!(
                "`{}`: {} inputs in the model against {} in the twin",
                self.model.key,
                model.len(),
                twin.len()
            );
        }
        for (xf, xq) in model.iter().zip(&twin) {
            let xq = xq.to_dtype(DType::F32)?;
            let drift = (xf.to_device(xq.device())?.to_dtype(DType::F32)? - &xq)?;
            let h = xq.t()?.matmul(&xq)?;
            let cross = drift.t()?.matmul(&xq)?;
            self.h = Some(match self.h.take() {
                Some(acc) => (acc + h)?,
                None => h,
            });
            self.cross = Some(match self.cross.take() {
                Some(acc) => (acc + cross)?,
                None => cross,
            });
            self.rows += xq.dim(0)?;
        }
        Ok(())
    }

    // both over the rows, on the host
    fn finish(self) -> Result<Option<(Tensor, Tensor)>> {
        let (Some(h), Some(cross)) = (self.h, self.cross) else {
            return Ok(None);
        };
        let rows = f64::from(u32::try_from(self.rows)?);
        Ok(Some((
            (h / rows)?.to_device(&Device::Cpu)?,
            (cross / rows)?.to_device(&Device::Cpu)?,
        )))
    }
}

// a Hessian as its packed upper triangle (row-major, `i <= j`): exact at half the bytes, where bf16's rounding left
// rank-deficient expert Hessians indefinite past GPTQ's damping
fn save_upper_hessian(dir: &Path, key: &str, h: &Tensor) -> Result<()> {
    let n = h.dim(0)?;
    let full = h.flatten_all()?.to_vec1::<f32>()?;
    let upper = (0..n)
        .flat_map(|i| full[i * n + i..(i + 1) * n].iter().copied())
        .collect::<Vec<_>>();
    let len = upper.len();
    inference_tensor::safetensors::save(
        &HashMap::from([(
            crate::gguf_imatrix::HESSIAN_UPPER_TENSOR.to_string(),
            Tensor::from_vec(upper, len, &Device::Cpu)?,
        )]),
        dir.join(format!("{key}.safetensors")),
    )?;
    Ok(())
}

// a stacked projection's per-expert Hessians `[E, in, in]`: each as `<experts>.<e>.<proj>` (packed upper triangles
// when `each` keeps stacks past SPLIT_EXPERT_HESSIANS_BYTES whole), or, past that, their mean under the stack's own key
fn save_expert_hessians(dir: &Path, key: &str, h: Tensor, each: bool) -> Result<()> {
    let large = h.elem_count() * 4 > SPLIT_EXPERT_HESSIANS_BYTES;
    if large && !each {
        return save_hessian(dir, key, h.mean(0)?);
    }
    let (experts, proj) = key
        .rsplit_once('.')
        .with_context(|| format!("`{key}` names no projection"))?;
    for e in 0..h.dim(0)? {
        let key = format!("{experts}.{e}.{proj}");
        if large {
            save_upper_hessian(dir, &key, &h.get(e)?.to_device(&Device::Cpu)?)?;
        } else {
            save_hessian(dir, &key, h.get(e)?)?;
        }
    }
    Ok(())
}

/// Every linear's input Hessian over the calibration tokens, with the model on the host streaming through the GPU a
/// layer at a time (its states run on through the layer as loaded): `<key>.safetensors` holding `hin` under `dir`, the
/// head's as `lm_head`. Stacked gate/up experts read the layer's shared input, so they take the shared expert's
/// Hessian where the layer has one (their rows' pooled second moment); other stacks write one per expert.
#[cfg(feature = "cuda")]
pub(crate) fn write_hessians_streamed(
    stream: &dyn inference_nn::model::LayerStream,
    tokens: &[u32],
    ctx: &CalibrationCtx<'_>,
    dir: &Path,
    weights: &[PathBuf],
) -> Result<()> {
    let chunks = tokens
        .chunks(CALIBRATION_CHUNK_SIZE)
        .map(|chunk| {
            ctx.bos_tok_id
                .into_iter()
                .chain(chunk.iter().copied())
                .collect()
        })
        .collect();
    run_streamed(stream, chunks, StreamedRun::Collect(dir), weights)
}

/// The model's KL divergence from a llama-perplexity base (`--kl-divergence-base`), streamed a layer at a time
/// through the GPU over the base's own chunks (each chunk's first token the BOS where encoding adds one, as
/// llama-perplexity runs it): mean, p99 and median KLD, the top token's agreement, and both perplexities.
#[cfg(feature = "cuda")]
pub(crate) fn score_kld_streamed(
    stream: &dyn inference_nn::model::LayerStream,
    base: &Path,
    ctx: &CalibrationCtx<'_>,
    weights: &[PathBuf],
) -> Result<()> {
    let mut base = inference_nn::logits_reference::KldBase::open(base)?;
    // llama-perplexity swaps a chunk's first token for BOS only when the vocab adds one when encoding
    let bos = ctx.bos_tok_id;
    let chunks = base
        .tokens
        .chunks(base.n_ctx)
        .map(|chunk| {
            let mut chunk = chunk.to_vec();
            if let Some(bos) = bos {
                chunk[0] = bos;
            }
            chunk
        })
        .collect();
    run_streamed(stream, chunks, StreamedRun::Score(&mut base), weights)
}

/// A new llama-perplexity KL-divergence base at `out` from the model over `job`'s text, streamed a layer at a time
/// through the GPU: the text tokenized with its special tokens (and the BOS before it where encoding adds one), its
/// first `job.chunks` chunks of `job.n_ctx` tokens, each chunk's second half recorded.
#[cfg(feature = "cuda")]
pub(crate) fn write_kld_base_streamed(
    stream: &dyn inference_nn::model::LayerStream,
    job: &inference_nn::logits_reference::KldBaseJob,
    out: &Path,
    ctx: &CalibrationCtx<'_>,
    weights: &[PathBuf],
) -> Result<()> {
    let text = std::fs::read_to_string(&job.text)?;
    // added tokens (a chat template's) match in the text either way; `false` keeps the template's own BOS out
    let encoded = ctx
        .tokenizer
        .encode_fast(text, false)
        .map_err(anyhow::Error::msg)?;
    let tokens = ctx
        .bos_tok_id
        .into_iter()
        .chain(encoded.get_ids().iter().copied())
        .collect::<Vec<_>>();
    let n_chunk = job.chunks.min(tokens.len() / job.n_ctx);
    anyhow::ensure!(
        n_chunk > 0,
        "{} holds {} tokens, less than one chunk of {}",
        job.text.display(),
        tokens.len(),
        job.n_ctx
    );
    let tokens = &tokens[..n_chunk * job.n_ctx];
    let chunks = tokens
        .chunks(job.n_ctx)
        .map(|chunk| {
            let mut chunk = chunk.to_vec();
            if let Some(bos) = ctx.bos_tok_id {
                chunk[0] = bos;
            }
            chunk
        })
        .collect();
    let vocab = stream.head().dequantize_w()?.dim(0)?;
    let mut writer =
        inference_nn::logits_reference::KldBaseWriter::create(out, job.n_ctx, vocab, tokens)?;
    run_streamed(stream, chunks, StreamedRun::WriteBase(&mut writer), weights)?;
    writer.finish()?;
    info!(
        "Wrote a base of {n_chunk} chunks of {} tokens to `{}`.",
        job.n_ctx,
        out.display()
    );
    Ok(())
}

// what a streamed run does with the states: collect each linear's Hessian into a directory, score the final logits
// against a KL-divergence base, or write them as a new one
#[cfg(feature = "cuda")]
enum StreamedRun<'a> {
    Collect(&'a Path),
    Score(&'a mut inference_nn::logits_reference::KldBase),
    WriteBase(&'a mut inference_nn::logits_reference::KldBaseWriter),
}

#[cfg(feature = "cuda")]
fn run_streamed(
    stream: &dyn inference_nn::model::LayerStream,
    chunks: Vec<Vec<u32>>,
    mut run: StreamedRun<'_>,
    weights: &[PathBuf],
) -> Result<()> {
    let gpu = Device::new_cuda(0)?;
    let dir = match &run {
        StreamedRun::Collect(dir) => {
            std::fs::create_dir_all(dir)?;
            Some(*dir)
        }
        StreamedRun::Score(_) | StreamedRun::WriteBase(_) => None,
    };
    let each = std::env::var_os(PER_EXPERT_HESSIANS_ENV).is_some();
    // a scoring run collects nothing, so the experts take the engine's kernels instead of the collecting gather path
    let score_only = dir.is_none() || std::env::var_os(STREAM_SCORE_ONLY_ENV).is_some();
    let mut ranges = layer_ranges(weights)?;
    let mut states = Vec::new();
    for ids in &chunks {
        states.push(stream.embed(&Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?)?);
    }
    let layers = stream.num_layers();
    // variants of the model a scoring run scores beside it (INFERENCE_STREAM_CANDIDATES): each shares the model's
    // states until its embedding or a layer it replaces sets its own apart
    let candidates = match &run {
        StreamedRun::Score(_) => inference_nn::decoder::take_stream_candidates(),
        StreamedRun::Collect(_) | StreamedRun::WriteBase(_) => Vec::new(),
    };
    // the states stay on the GPU when they fit beside a layer, so they never cross the bus
    let state_bytes: usize = states
        .iter()
        .map(|s| s.elem_count() * s.dtype().size_in_bytes())
        .sum();
    let resident = state_bytes * (1 + candidates.len()) <= RESIDENT_STATE_BYTES;
    if resident {
        states = states
            .iter()
            .map(|s| s.to_device(&gpu))
            .collect::<inference_tensor::Result<_>>()?;
    }
    info!(
        "Collecting Hessians: {layers} layers streamed through {gpu:?}, {} chunks of states on {}.",
        states.len(),
        if resident { "it" } else { "the host" }
    );
    // a second checkpoint (INFERENCE_STREAM_TWIN) runs its own states beside: each of its layers on the states its
    // layers before gave, routed as the model's own layer routed (INFERENCE_STREAM_TWIN_FREE leaves it its own routes)
    let twin = inference_nn::decoder::take_stream_twin();
    let free_routes = std::env::var_os(STREAM_TWIN_FREE_ENV).is_some();
    let mut twin_states = twin.as_ref().map(|_| states.clone());
    if twin.is_some() && !candidates.is_empty() {
        anyhow::bail!("a streamed run takes a twin or candidates, not both");
    }
    let mut candidate_states = candidates
        .iter()
        .map(|candidate| -> Result<Option<Vec<Tensor>>> {
            if !candidate.embed {
                return Ok(None);
            }
            let ids = chunks
                .iter()
                .map(|ids| Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0))
                .collect::<inference_tensor::Result<Vec<_>>>()?;
            stream
                .embed_from(&ids, &candidate.weights)?
                .into_iter()
                .map(|state| {
                    Ok(if resident {
                        state.to_device(&gpu)?
                    } else {
                        state
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(Some)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut ahead: Option<std::thread::JoinHandle<()>> = None;
    for layer_idx in 0..layers {
        let start = std::time::Instant::now();
        if let Some(reading) = ahead.take() {
            let _ = reading.join();
        }
        ahead = ranges.remove(&(layer_idx + 1)).map(prefetch);
        let layer = stream.build_layer(layer_idx, &gpu)?;
        let twin_layer = twin
            .as_ref()
            .map(|weights| stream.build_layer_from(layer_idx, &gpu, weights))
            .transpose()?;
        let group = layer
            .modules()
            .iter()
            .filter(|m| encoded(m))
            .collect::<Vec<_>>();
        let ranks = group
            .iter()
            .map(|m| weight_rank(m))
            .collect::<Result<Vec<_>>>()?;
        let proj = |m: &TrackedModule| m.key.rsplit_once('.').map_or("", |(_, p)| p).to_string();
        let shared = group
            .iter()
            .zip(&ranks)
            .position(|(m, &r)| {
                r == 2 && m.key.contains(SHARED_EXPERT) && GATE_UP_PROJS.contains(&proj(m).as_str())
            })
            .filter(|_| !each);
        // each module's Hessian source: itself, the first reading the same input, or (stacked gate/up) the shared expert
        let mut first = HashMap::new();
        let sources = group
            .iter()
            .zip(&ranks)
            .enumerate()
            .map(|(i, (m, &rank))| match (rank, shared) {
                (3, Some(s)) if GATE_UP_PROJS.contains(&proj(m).as_str()) => s,
                _ => match shared_input(&m.key) {
                    Some(input) => *first.entry(input).or_insert(i),
                    None => i,
                },
            })
            .collect::<Vec<_>>();
        // with a twin a collection pairs each dense input of the two (GPTAQ's statistics): the twin's own Hessian,
        // and its inputs' cross moment with the drift from the model's
        let dump_dir = std::env::var_os(STREAM_ROUTER_DUMP_ENV);
        let gptaq = twin_layer.is_some() && dir.is_some() && dump_dir.is_none();
        let mut pairs = Vec::new();
        if let (true, Some(twin_layer)) = (gptaq, &twin_layer) {
            for (i, module) in group.iter().enumerate() {
                if sources[i] != i || ranks[i] != 2 {
                    continue;
                }
                let Some(twin) = twin_layer.modules().iter().find(|m| m.key == module.key) else {
                    continue;
                };
                module.ct.begin_capture_inputs()?;
                twin.ct.begin_capture_inputs()?;
                pairs.push(Gptaq {
                    index: i,
                    model: (*module).clone(),
                    twin: twin.clone(),
                    h: None,
                    cross: None,
                    rows: 0,
                });
            }
        }
        let mut router_dump = dump_dir
            .zip(twin_layer.as_ref())
            .and_then(|(dir, twin_layer)| {
                let model = group
                    .iter()
                    .find(|m| m.key.ends_with(ROUTER_INPUT_MODULE))?;
                let twin = twin_layer.modules().iter().find(|m| m.key == model.key)?;
                Some((
                    PathBuf::from(dir),
                    (*model).clone(),
                    twin.clone(),
                    Vec::new(),
                    Vec::new(),
                ))
            });
        if let Some((_, model, twin, _, _)) = &router_dump {
            model.ct.begin_capture_inputs()?;
            twin.ct.begin_capture_inputs()?;
        }
        for (i, module) in group.iter().enumerate() {
            if sources[i] == i && !score_only && !gptaq && router_dump.is_none() {
                module.ct.begin_track_hessian()?;
            }
        }
        let before = (!candidates.is_empty()).then(|| states.clone());
        // one pass collects the Hessians and runs the states on (the layer as loaded, so no second pass is needed);
        // a plain pass batches chunks of one length, which the MoE's per-call costs (a launch per expert) dominate
        let batched =
            !score_only && twin_layer.is_none() && pairs.is_empty() && router_dump.is_none();
        if batched {
            let mut i = 0;
            while i < states.len() {
                let len = states[i].dims().to_vec();
                let mut j = i + 1;
                while j < states.len()
                    && j - i < STREAM_BATCH_CHUNKS
                    && states[j].dims() == len.as_slice()
                {
                    j += 1;
                }
                let xs = Tensor::cat(
                    &states[i..j]
                        .iter()
                        .map(|s| s.to_device(&gpu))
                        .collect::<inference_tensor::Result<Vec<_>>>()?,
                    0,
                )?;
                let ys = layer.forward(&xs)?;
                for (k, state) in states[i..j].iter_mut().enumerate() {
                    let y = ys.narrow(0, k, 1)?;
                    *state = if resident {
                        y
                    } else {
                        y.to_device(&Device::Cpu)?
                    };
                }
                i = j;
            }
        }
        for (i, state) in states.iter_mut().enumerate().filter(|_| !batched) {
            if twin_layer.is_some() && !free_routes {
                inference_nn::ops::record_routes();
            }
            let next = layer.forward(&state.to_device(&gpu)?)?;
            *state = if resident {
                next
            } else {
                next.to_device(&Device::Cpu)?
            };
            if let (Some(twin_layer), Some(twin_states)) = (&twin_layer, &mut twin_states) {
                if !free_routes {
                    inference_nn::ops::replay_routes(inference_nn::ops::take_recorded_routes());
                }
                let twin_state = &mut twin_states[i];
                let next = twin_layer.forward(&twin_state.to_device(&gpu)?);
                inference_nn::ops::stop_replaying_routes();
                *twin_state = if resident {
                    next?
                } else {
                    next?.to_device(&Device::Cpu)?
                };
            }
            for pair in &mut pairs {
                pair.add_chunk()?;
            }
            if let Some((_, model, twin, xf, xq)) = &mut router_dump {
                for x in model.ct.take_captured_inputs()? {
                    xf.push(
                        x.to_dtype(inference_tensor::DType::BF16)?
                            .to_device(&Device::Cpu)?,
                    );
                }
                for x in twin.ct.take_captured_inputs()? {
                    xq.push(
                        x.to_dtype(inference_tensor::DType::BF16)?
                            .to_device(&Device::Cpu)?,
                    );
                }
            }
        }
        let dumping = router_dump.is_some();
        if let Some((dir, _, _, xf, xq)) = router_dump {
            std::fs::create_dir_all(&dir)?;
            inference_tensor::safetensors::save(
                &HashMap::from([
                    ("xf".to_string(), Tensor::cat(&xf, 0)?),
                    ("xq".to_string(), Tensor::cat(&xq, 0)?),
                ]),
                dir.join(format!("layer{layer_idx}.safetensors")),
            )?;
        }
        for (candidate, own_states) in candidates.iter().zip(&mut candidate_states) {
            let replaced = candidate.layers.contains(&layer_idx);
            let input = match own_states.take() {
                Some(own) => own,
                None if replaced => before.clone().context("the states before the layer")?,
                None => continue,
            };
            let own_layer = replaced
                .then(|| stream.build_layer_from(layer_idx, &gpu, &candidate.weights))
                .transpose()?;
            let run_layer = own_layer.as_deref().unwrap_or(&*layer);
            let mut next = Vec::with_capacity(input.len());
            for state in &input {
                let out = run_layer.forward(&state.to_device(&gpu)?)?;
                next.push(if resident {
                    out
                } else {
                    out.to_device(&Device::Cpu)?
                });
            }
            *own_states = Some(next);
        }
        let mut hessians = vec![None; group.len()];
        let mut crosses = vec![None; group.len()];
        for (i, module) in group.iter().enumerate() {
            if sources[i] == i && !score_only && !gptaq && !dumping {
                hessians[i] = take_hessian(module)?
                    .map(|h| h.to_device(&Device::Cpu))
                    .transpose()?;
            }
        }
        for pair in pairs {
            let index = pair.index;
            if let Some((h, cross)) = pair.finish()? {
                hessians[index] = Some(h);
                crosses[index] = Some(cross);
            }
        }
        for (i, module) in group.iter().enumerate() {
            let (Some(h), Some(dir)) = (&hessians[sources[i]], dir) else {
                continue;
            };
            match h.rank() {
                // a stack reading another's input (up beside gate) is found through it, not written again
                3 if each && sources[i] != i => {}
                3 => save_expert_hessians(dir, &module.key, h.clone(), each)?,
                _ => match &crosses[sources[i]] {
                    Some(cross) => save_hessian_pair(dir, &module.key, h.clone(), cross.clone())?,
                    None => save_hessian(dir, &module.key, h.clone())?,
                },
            }
        }
        drop(group);
        drop(layer);
        info!(
            "Layer {}/{layers} in {:.1}s.",
            layer_idx + 1,
            start.elapsed().as_secs_f32()
        );
    }
    match &mut run {
        StreamedRun::Collect(dir) => {
            let in_dim = states[0].dim(states[0].rank() - 1)?;
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
            save_hessian(dir, HEAD, hessian.to_device(&Device::Cpu)?)?;
            info!("Wrote the Hessians to `{}`.", dir.display());
        }
        StreamedRun::Score(base) => {
            let model_states = twin_states.as_deref().unwrap_or(&states);
            let mut runs = vec![ScoredRun {
                name: "model".to_string(),
                states: model_states,
                head: None,
            }];
            for (candidate, own) in candidates.iter().zip(&candidate_states) {
                runs.push(ScoredRun {
                    name: candidate.name.clone(),
                    states: own.as_deref().unwrap_or(model_states),
                    head: candidate
                        .head
                        .then(|| stream.head_from(&gpu, &candidate.weights))
                        .transpose()?,
                });
            }
            score_states(stream, &runs, &chunks, base, &gpu)?
        }
        StreamedRun::WriteBase(writer) => {
            write_base_states(stream, &states, &chunks, writer, &gpu)?
        }
    }
    if let Some(path) = std::env::var_os(STREAM_LOGITS_OUT_ENV) {
        stream_logits(stream, &states, &gpu, LogitsUse::Record(Path::new(&path)))?;
    }
    if let Some(path) = std::env::var_os(STREAM_LOGITS_REF_ENV) {
        stream_logits(stream, &states, &gpu, LogitsUse::Compare(Path::new(&path)))?;
    }
    Ok(())
}

// each chunk's final logits over its second half (llama-perplexity's scored rows) into the base
#[cfg(feature = "cuda")]
fn write_base_states(
    stream: &dyn inference_nn::model::LayerStream,
    states: &[Tensor],
    chunks: &[Vec<u32>],
    writer: &mut inference_nn::logits_reference::KldBaseWriter,
    gpu: &Device,
) -> Result<()> {
    use inference_tensor::DType;
    let head = stream
        .head()
        .dequantize_w()?
        .to_dtype(DType::BF16)?
        .to_device(gpu)?;
    for (state, tokens) in states.iter().zip(chunks) {
        let first = tokens.len() / 2;
        let rows = tokens.len() - 1 - first;
        let xs = stream
            .final_norm(state)?
            .to_device(gpu)?
            .to_dtype(DType::BF16)?
            .squeeze(0)?
            .narrow(0, first, rows)?;
        let logits = xs.matmul(&head.t()?)?.to_dtype(DType::F32)?;
        writer.write_chunk(&logits.flatten_all()?.to_vec1::<f32>()?)?;
    }
    Ok(())
}

// each run's final logits against the base's records chunk by chunk, reported as llama-perplexity's KLD summary
#[cfg(feature = "cuda")]
fn score_states(
    stream: &dyn inference_nn::model::LayerStream,
    runs: &[ScoredRun<'_>],
    chunks: &[Vec<u32>],
    base: &mut inference_nn::logits_reference::KldBase,
    gpu: &Device,
) -> Result<()> {
    use inference_tensor::{D, DType};
    let dense = |head: &std::sync::Arc<dyn QuantMethod>| -> Result<Tensor> {
        Ok(head.dequantize_w()?.to_dtype(DType::BF16)?.to_device(gpu)?)
    };
    let model_head = dense(&stream.head())?;
    let heads = runs
        .iter()
        .map(|run| run.head.as_ref().map(dense).transpose())
        .collect::<Result<Vec<_>>>()?;
    let mut scores = vec![RunScore::default(); runs.len()];
    let first = base.first_row();
    for (c, tokens) in chunks.iter().enumerate() {
        // one read and decode of the chunk's base records serves every run
        let reference = base.next_chunk()?;
        let reference = inference_nn::logits_reference::decode_reference(
            &reference,
            tokens,
            model_head.dim(0)?,
            gpu,
        )?;
        for ((run, head), score) in runs.iter().zip(&heads).zip(&mut scores) {
            let xs = stream
                .final_norm(&run.states[c])?
                .to_device(gpu)?
                .to_dtype(DType::BF16)?
                .squeeze(0)?;
            let head = head.as_ref().unwrap_or(&model_head);
            let logits = xs.matmul(&head.t()?)?.to_dtype(DType::F32)?;
            let divergence =
                inference_nn::logits_reference::divergence_decoded(&logits, &reference)?;
            let rows = divergence.kld.len();
            let next = Tensor::from_slice(&tokens[first + 1..first + 1 + rows], (rows, 1), gpu)?;
            let logp =
                inference_tensor::nn::ops::log_softmax(&logits.narrow(0, first, rows)?, D::Minus1)?;
            score.nll -= logp
                .gather(&next, 1)?
                .to_dtype(DType::F64)?
                .sum_all()?
                .to_scalar::<f64>()?;
            score.nll_base -= divergence
                .reference_logprobs
                .iter()
                .map(|&l| f64::from(l))
                .sum::<f64>();
            score.same += divergence.same_top.iter().filter(|&&s| s).count();
            score.chunk_means.push(
                divergence.kld.iter().map(|&k| f64::from(k)).sum::<f64>()
                    / f64::from(u32::try_from(rows)?),
            );
            score.klds.extend(divergence.kld);
        }
    }
    let mut report = Vec::with_capacity(runs.len());
    for (run, score) in runs.iter().zip(&mut scores) {
        let n = f64::from(u32::try_from(score.klds.len())?);
        let mean = score.klds.iter().map(|&k| f64::from(k)).sum::<f64>() / n;
        score.klds.sort_by(f32::total_cmp);
        // the q-quantile by index (q in hundredths, as the report names them)
        let pick = |hundredths: usize| score.klds[(score.klds.len() - 1) * hundredths / 100];
        let same_top = 100.0 * f64::from(u32::try_from(score.same)?) / n;
        info!(
            "`{}`: KLD over {} positions ({} chunks of {}): mean {mean:.5}, median {:.5}, p99 {:.4}, same top \
             {same_top:.2}%, PPL {:.4} (base {:.4})",
            run.name,
            score.klds.len(),
            chunks.len(),
            base.n_ctx,
            pick(50),
            pick(99),
            (score.nll / n).exp(),
            (score.nll_base / n).exp()
        );
        report.push(serde_json::json!({
            "name": run.name,
            "kld": mean,
            "p99": pick(99),
            "same_top": same_top,
            "ppl": (score.nll / n).exp(),
            "chunk_klds": score.chunk_means,
        }));
    }
    if let Some(path) = std::env::var_os(STREAM_SCORES_OUT_ENV) {
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

// one model a streamed scoring run scores: its final states and, where it has its own, its head
#[cfg(feature = "cuda")]
struct ScoredRun<'a> {
    name: String,
    states: &'a [Tensor],
    head: Option<std::sync::Arc<dyn QuantMethod>>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Default)]
struct RunScore {
    klds: Vec<f32>,
    chunk_means: Vec<f64>,
    same: usize,
    nll: f64,
    nll_base: f64,
}

// a diagnostic over a streamed run's final states: record its log-probabilities, or compare them with a recording
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum LogitsUse<'a> {
    Record(&'a Path),
    Compare(&'a Path),
}

// each chunk's log-probabilities (f16, every position) saved as `<dir>/<chunk>.safetensors`, or read back and the KL
// divergence of this run from the recorded one reported per position (mean, p99) with the top token's agreement
#[cfg(feature = "cuda")]
fn stream_logits(
    stream: &dyn inference_nn::model::LayerStream,
    states: &[Tensor],
    gpu: &Device,
    use_: LogitsUse<'_>,
) -> Result<()> {
    use inference_tensor::{D, DType};
    let head = stream
        .head()
        .dequantize_w()?
        .to_dtype(DType::BF16)?
        .to_device(gpu)?;
    if let LogitsUse::Record(dir) = use_ {
        std::fs::create_dir_all(dir)?;
    }
    let (mut klds, mut same) = (Vec::new(), 0usize);
    // the positions past half a chunk, as llama-perplexity scores them
    let mut late = Vec::new();
    for (i, state) in states.iter().enumerate() {
        let xs = stream
            .final_norm(state)?
            .to_device(gpu)?
            .to_dtype(DType::BF16)?
            .squeeze(0)?;
        let logp = inference_tensor::nn::ops::log_softmax(
            &xs.matmul(&head.t()?)?.to_dtype(DType::F32)?,
            D::Minus1,
        )?;
        match use_ {
            LogitsUse::Record(dir) => inference_tensor::safetensors::save(
                &HashMap::from([(
                    LOGP_TENSOR.to_string(),
                    logp.to_dtype(DType::F16)?.to_device(&Device::Cpu)?,
                )]),
                dir.join(format!("{i}.safetensors")),
            )?,
            LogitsUse::Compare(dir) => {
                let ref_logp =
                    inference_tensor::safetensors::load(dir.join(format!("{i}.safetensors")), gpu)?
                        .remove(LOGP_TENSOR)
                        .context("a recorded chunk holds no log-probabilities")?
                        .to_dtype(DType::F32)?;
                let kld = (ref_logp.exp()? * (&ref_logp - &logp)?)?
                    .sum(D::Minus1)?
                    .to_vec1::<f32>()?;
                late.extend_from_slice(&kld[kld.len() / 2..]);
                klds.extend(kld);
                let agree = ref_logp.argmax(D::Minus1)?.eq(&logp.argmax(D::Minus1)?)?;
                same +=
                    usize::try_from(agree.to_dtype(DType::U32)?.sum_all()?.to_scalar::<u32>()?)?;
            }
        }
    }
    if let LogitsUse::Compare(_) = use_ {
        let n = klds.len();
        let count = f64::from(u32::try_from(n)?);
        let mean = klds.iter().map(|&k| f64::from(k)).sum::<f64>() / count;
        klds.sort_by(f32::total_cmp);
        let late_mean = late.iter().map(|&k| f64::from(k)).sum::<f64>()
            / f64::from(u32::try_from(late.len().max(1))?);
        info!(
            "Streamed KLD over {n} positions: mean {mean:.5} (second halves {late_mean:.5}), p99 {:.4}, same top {:.2}%",
            klds[n * 99 / 100],
            100.0 * f64::from(u32::try_from(same)?) / count
        );
    }
    Ok(())
}

// every source tensor but the replaced weights (and every tensor of the replaced experts), a shard for each source
// file
fn encoded(module: &TrackedModule) -> bool {
    !KEPT_DENSE.iter().any(|k| module.key.ends_with(k))
}

fn copy_unquantized(
    source: &Exl3Source<'_>,
    out: &Path,
    replaced: &std::collections::HashSet<String>,
    replaced_prefixes: &std::collections::HashSet<String>,
    weight_map: &mut BTreeMap<String, String>,
) -> Result<()> {
    for (i, file) in source.weights.iter().enumerate() {
        // mapped, so only the kept tensors are read: the encoded weights are most of a shard
        // SAFETY: the source checkpoint is not written while the converter reads it
        let mapped = unsafe { inference_tensor::safetensors::MmapedSafetensors::new(file)? };
        let tensors = mapped
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .filter(|name| {
                !replaced.contains(name) && !replaced_prefixes.iter().any(|p| name.starts_with(p))
            })
            .map(|name| Ok((name.clone(), mapped.load(&name, &Device::Cpu)?)))
            .collect::<Result<HashMap<_, _>>>()?;
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
    use super::{layer_of, layer_ranges, prefetch};

    #[test]
    fn modules_order_by_their_layer() {
        assert_eq!(layer_of("model.layers.12.self_attn.q_proj"), Some(12));
        assert_eq!(
            layer_of("model.language_model.layers.3.mlp.down_proj"),
            Some(3)
        );
        assert_eq!(layer_of("lm_head"), None);
    }

    #[test]
    fn a_layers_tensors_read_ahead_as_merged_runs() -> anyhow::Result<()> {
        use safetensors::{Dtype, serialize_to_file, tensor::TensorView};
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.safetensors");
        let data = vec![0u8; 64];
        let tensors = [
            "model.layers.0.a.weight",
            "model.layers.0.b.weight",
            "model.layers.1.a.weight",
            "model.embed_tokens.weight",
        ]
        .map(|name| Ok::<_, anyhow::Error>((name, TensorView::new(Dtype::U8, vec![64], &data)?)));
        serialize_to_file(
            tensors.into_iter().collect::<anyhow::Result<Vec<_>>>()?,
            None,
            &path,
        )?;
        let mut ranges = layer_ranges(std::slice::from_ref(&path))?;
        assert_eq!(ranges.keys().copied().collect::<Vec<_>>(), [0, 1]);
        let layer0 = &ranges[&0];
        assert_eq!(layer0.len(), 1, "neighbouring tensors read as one run");
        assert_eq!(layer0[0].2 - layer0[0].1, 128);
        let file = std::fs::metadata(&path)?.len();
        assert!(ranges.values().flatten().all(|&(_, _, end)| end <= file));
        prefetch(ranges.remove(&1).unwrap()).join().unwrap();
        Ok(())
    }
}
