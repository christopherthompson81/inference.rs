//! NVIDIA's `asr` GGUF layout written from a `.nemo`, with a per-tensor precision policy: the matrices in one
//! quantized type, the ones a policy names kept finer.

use std::path::Path;

use inference_tensor::quantized::gguf_file::{self, Value};
use inference_tensor::quantized::{GgmlDType, QTensor, WRITABLE_TYPES};
use inference_tensor::{Error, Result, Tensor};

use super::config::HeadKind;
use super::nemo_checkpoint::{NemoRecogniser, read_recogniser};
use crate::nemo::NemoArchive;

const LSTM_PREFIX: &str = "decoder.prediction.dec_rnn.lstm.";
const EMBEDDING: &str = "decoder.prediction.embed.weight";
// `pos_bias_u`/`pos_bias_v` are (heads, head_dim) biases, kept at full precision like the other biases
const POSITION_BIASES: &str = ".pos_bias_";
// what the loader never reads: the frontend, batch-norm counters, a hybrid's CTC head
const SKIPPED: [&str; 3] = ["preprocessor.", "num_batches_tracked", "ctc_decoder."];
// GGUF stores the conformer's 1x1 convs as matrices, as NVIDIA's converter does
const POINTWISE_CONV: &str = ".conv.pointwise_conv";
const FALLBACK: GgmlDType = GgmlDType::Q8_0;
// the predictor (LSTM, projector) and joint: a few MiB whose 8-bit error lands straight in the transcript
const DECODER: [&str; 2] = ["decoder.prediction.", "joint."];
// the joint's width is its own setting (`joint_hidden`), read off the projector the predictor feeds it through
const JOINT_PROJECTOR: &str = "joint.pred.weight";
const FEED_FORWARD: &str = ".feed_forward";
const NO_CONTEXT: i32 = -1;
// the keys NVIDIA's converter writes for its own runtime, kept so a file written here carries them too
const POSITIONS: u32 = 5000;
const DITHER: f32 = 1e-5;
const Q8_LAYOUT: &str = "block_q8_0";
const OVERRIDE_SEPARATOR: char = '=';

type Preset = (&'static str, fn() -> QuantPolicy);

/// The named policies [`QuantPolicy::parse`] takes besides a type name.
pub const PRESETS: [Preset; 2] = [
    ("q8_0-decoder", QuantPolicy::q8_0_decoder),
    ("q6_k-mix", QuantPolicy::q6_k_mix),
];

fn type_named(name: &str) -> Result<GgmlDType> {
    GgmlDType::from_name(name)
}

// llama.cpp's `general.file_type`: the "mostly" type the large matrices take
fn file_type(matrices: GgmlDType) -> u32 {
    match matrices {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q5_0 => 8,
        GgmlDType::Q3K => 12,
        GgmlDType::Q4K => 15,
        GgmlDType::Q5K => 17,
        GgmlDType::Q6K => 18,
        GgmlDType::Q8_0 => 7,
        GgmlDType::BF16 => 32,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q5_1 => 9,
        GgmlDType::Q2K => 10,
        // no llama.cpp file type is mostly one of these; Q8_0's is the nearest
        GgmlDType::Q8_1 | GgmlDType::Q8K => 7,
    }
}

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

/// Which precision each tensor takes: norms and biases F32, convs and the embedding F16, as NVIDIA's converter
/// writes them, and the matrices `matrices` unless an override's name fragment matches first.
#[derive(Debug, Clone)]
pub struct QuantPolicy {
    pub matrices: GgmlDType,
    pub overrides: Vec<(String, GgmlDType)>,
}

impl QuantPolicy {
    /// Every matrix in `matrices`, as NVIDIA's `q8_0.gguf` is written with Q8_0.
    pub fn uniform(matrices: GgmlDType) -> Self {
        Self {
            matrices,
            overrides: Vec::new(),
        }
    }

    /// A preset or uniform type, `policy`, with `keep`'s `fragment=type` overrides ahead of its own.
    pub fn parse(policy: &str, keep: &[String]) -> Result<Self> {
        let mut policy = match PRESETS.iter().find(|(name, _)| *name == policy) {
            Some((_, preset)) => preset(),
            None => Self::uniform(type_named(policy).map_err(|_| {
                msg(format!(
                    "`{policy}` is not a preset {:?} or a type {:?}",
                    PRESETS.map(|(n, _)| n),
                    WRITABLE_TYPES.map(|(n, _)| n)
                ))
            })?),
        };
        let keep = keep
            .iter()
            .map(|k| match k.split_once(OVERRIDE_SEPARATOR) {
                Some((fragment, d)) => Ok((fragment.to_string(), type_named(d)?)),
                None => Err(msg(format!(
                    "`{k}` is not fragment{OVERRIDE_SEPARATOR}type"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;
        policy.overrides.splice(0..0, keep);
        Ok(policy)
    }

    /// Q8_0 with the decoder and joint at F16: NVIDIA's size plus 2%, about a third of its word errors.
    pub fn q8_0_decoder() -> Self {
        Self::uniform(GgmlDType::Q8_0).with_decoder_f16()
    }

    /// Q6_K with the feed-forward at Q8_0 and the decoder at F16: smaller than NVIDIA's Q8_0 and more accurate.
    pub fn q6_k_mix() -> Self {
        let mut policy = Self::uniform(GgmlDType::Q6K).with_decoder_f16();
        policy
            .overrides
            .push((FEED_FORWARD.to_string(), GgmlDType::Q8_0));
        policy
    }

    fn with_decoder_f16(mut self) -> Self {
        self.overrides
            .extend(DECODER.map(|d| (d.to_string(), GgmlDType::F16)));
        self
    }

    fn dtype(&self, name: &str, t: &Tensor) -> GgmlDType {
        if t.rank() <= 1 || name.contains(POSITION_BIASES) {
            return GgmlDType::F32;
        }
        if t.rank() > 2 || name == EMBEDDING {
            return GgmlDType::F16;
        }
        let wanted = self
            .overrides
            .iter()
            .find(|(fragment, _)| name.contains(fragment.as_str()))
            .map_or(self.matrices, |(_, dtype)| *dtype);
        // a row that does not fill whole blocks of the wanted type takes the next one down that it does
        let row = t.dims()[t.rank() - 1];
        [wanted, FALLBACK]
            .into_iter()
            .find(|d| row.is_multiple_of(d.block_size()))
            .unwrap_or(GgmlDType::F16)
    }
}

// a NeMo name and tensor in the converter's forms: `lstm.ih_l0.weight`, pointwise convs as matrices
fn converter_form(name: &str, t: Tensor) -> Result<(String, Tensor)> {
    let name = match name.strip_prefix(LSTM_PREFIX) {
        Some(rest) => {
            let (param, gate_layer) = rest
                .split_once('_')
                .ok_or_else(|| msg(format!("`{name}` is not an LSTM parameter")))?;
            format!("{LSTM_PREFIX}{gate_layer}.{param}")
        }
        None => name.to_string(),
    };
    let t = if name.contains(POINTWISE_CONV) && t.rank() == 3 {
        t.squeeze(2)?
    } else {
        t
    };
    Ok((name, t))
}

// the facts the metadata needs beyond the parsed config
struct Written<'a> {
    name: &'a str,
    matrices: GgmlDType,
    joint_dim: usize,
}

fn metadata(r: &NemoRecogniser, w: &Written) -> Result<Vec<(String, Value)>> {
    let c = &r.config;
    let (Some(pred_hidden), Some(pred_layers)) = (c.decoder_hidden_size, c.num_decoder_layers)
    else {
        return Err(msg("an RNN-T head without its predictor sizes"));
    };
    let enc = &c.encoder_config;
    let head = match c.head() {
        Some(HeadKind::Tdt) => "tdt",
        Some(HeadKind::Rnnt) => "rnnt",
        _ => return Err(msg("only RNN-T and TDT heads are written as `asr` GGUFs")),
    };
    let (left, right) = enc
        .chunked_context()
        .map_or((NO_CONTEXT, NO_CONTEXT), |ctx| {
            (ctx.left as i32, ctx.right as i32)
        });
    let mel = &r.mel;
    let rate = mel.sample_rate as f32;
    let u = |v: usize| Value::U32(v as u32);
    let s = |v: &str| Value::String(v.to_string());
    let mut out: Vec<(&str, Value)> = vec![
        ("general.architecture", s("asr")),
        ("general.name", s(w.name)),
        ("asr.head_type", s(head)),
        ("general.file_type", Value::U32(file_type(w.matrices))),
        ("asr.encoder.d_model", u(enc.hidden_size)),
        ("asr.encoder.n_layers", u(enc.num_hidden_layers)),
        ("asr.encoder.n_heads", u(enc.num_attention_heads)),
        ("asr.encoder.d_ff", u(enc.intermediate_size)),
        ("asr.encoder.conv_kernel_size", u(enc.conv_kernel_size)),
        ("asr.encoder.subsampling_factor", u(enc.subsampling_factor)),
        (
            "asr.encoder.subsampling_conv_channels",
            u(enc.subsampling_conv_channels),
        ),
        ("asr.encoder.feat_in", u(enc.num_mel_bins)),
        ("asr.encoder.xscaling", Value::Bool(enc.scale_input)),
        ("asr.encoder.use_bias", Value::Bool(enc.attention_bias)),
        ("asr.encoder.pos_emb_max_len", Value::U32(POSITIONS)),
        ("asr.encoder.q8_layout", s(Q8_LAYOUT)),
        ("asr.encoder.qkv_q8_layout", s(Q8_LAYOUT)),
        ("asr.encoder.cache_supported", Value::Bool(r.streaming)),
        ("asr.encoder.train_left_ctx", Value::U32(left.max(0) as u32)),
        (
            "asr.encoder.train_right_ctx",
            Value::U32(right.max(0) as u32),
        ),
        ("asr.encoder.offline_left_ctx", Value::I32(left)),
        ("asr.encoder.offline_right_ctx", Value::I32(right)),
        (
            "asr.encoder.conv_norm",
            s(if r.streaming {
                "layer_norm"
            } else {
                "batch_norm"
            }),
        ),
        (
            "asr.encoder.conv_context",
            s(if r.streaming { "causal" } else { "None" }),
        ),
        (
            "asr.encoder.att_context_style",
            s(if r.streaming {
                "chunked_limited"
            } else {
                "regular"
            }),
        ),
        ("asr.preprocessor.sample_rate", Value::U32(mel.sample_rate)),
        (
            "asr.preprocessor.window_size",
            Value::F32(mel.win_length as f32 / rate),
        ),
        (
            "asr.preprocessor.window_stride",
            Value::F32(mel.hop_length as f32 / rate),
        ),
        ("asr.preprocessor.n_fft", u(mel.n_fft)),
        ("asr.preprocessor.features", u(mel.n_mels)),
        (
            "asr.preprocessor.normalize",
            s(if mel.normalize { "per_feature" } else { "NA" }),
        ),
        (
            "asr.preprocessor.preemph",
            Value::F32(mel.preemphasis.unwrap_or(0.0)),
        ),
        ("asr.preprocessor.dither", Value::F32(DITHER)),
        ("asr.preprocessor.stft_center_window", Value::Bool(true)),
        ("asr.preprocessor.hann_periodic", Value::Bool(false)),
        ("asr.preprocessor.mask_invalid_frames", Value::Bool(true)),
        ("asr.rnnt.vocab_size", u(c.vocab_size)),
        ("asr.rnnt.blank_id", u(c.vocab_size - 1)),
        ("asr.rnnt.pred_hidden", u(pred_hidden)),
        ("asr.rnnt.pred_num_layers", u(pred_layers)),
        ("asr.rnnt.pred_embed_dim", u(pred_hidden)),
        ("asr.rnnt.joint_dim", u(w.joint_dim)),
        ("asr.rnnt.max_symbols_per_step", u(c.max_symbols_per_step)),
        ("asr.rnnt.num_prompts", Value::U32(0)),
        ("asr.tokenizer.type", s("sentencepiece_bpe")),
        (
            "asr.tokenizer.vocab",
            Value::Array(r.pieces.iter().map(|p| s(&p.text)).collect()),
        ),
    ];
    if !c.durations.is_empty() {
        out.push((
            "asr.tdt.durations",
            Value::Array(c.durations.iter().map(|d| Value::I32(*d as i32)).collect()),
        ));
    }
    Ok(out.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// Writes `nemo`'s recogniser to `out` as an `asr` GGUF under `policy`, returning each tensor's chosen type.
pub fn write_asr_gguf(
    nemo: &Path,
    out: &Path,
    policy: &QuantPolicy,
) -> Result<Vec<(String, GgmlDType)>> {
    let archive = NemoArchive::open(nemo)?;
    let recogniser = read_recogniser(&archive)?;
    let name = nemo
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let weights = archive.tensors()?;
    let joint_dim = weights
        .get(JOINT_PROJECTOR)
        .ok_or_else(|| msg(format!("the `.nemo` has no `{JOINT_PROJECTOR}`")))?
        .dims()[0];
    let written = Written {
        name: &name,
        matrices: policy.matrices,
        joint_dim,
    };
    let metadata = metadata(&recogniser, &written)?;
    let mut tensors = Vec::new();
    for (name, t) in weights {
        if SKIPPED.iter().any(|s| name.contains(s)) {
            continue;
        }
        let (name, t) = converter_form(&name, t)?;
        let dtype = policy.dtype(&name, &t);
        tensors.push((name, QTensor::quantize(&t, dtype)?, dtype));
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    let metadata: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, q, _)| (n.as_str(), q)).collect();
    gguf_file::write(&mut file, &metadata, &refs)?;
    // dropping a BufWriter would lose a failed final write, as on a full disk
    std::io::Write::flush(&mut file)?;
    Ok(tensors.into_iter().map(|(n, _, d)| (n, d)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::{DType, Device};

    fn zeros(dims: &[usize]) -> Tensor {
        Tensor::zeros(dims, DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn a_parsed_policy_puts_its_overrides_ahead_of_the_presets() -> Result<()> {
        let q = "encoder.layers.0.self_attn.linear_q.weight";
        let ff = "encoder.layers.0.feed_forward1.linear1.weight";
        let matrix = zeros(&[1024, 1024]);
        let keep = vec!["self_attn=q4_k".to_string(), "feed_forward=f16".to_string()];
        let policy = QuantPolicy::parse("q6_k-mix", &keep)?;
        assert_eq!(policy.dtype(q, &matrix), GgmlDType::Q4K);
        assert_eq!(policy.dtype(ff, &matrix), GgmlDType::F16);
        assert_eq!(
            QuantPolicy::parse("q5_k", &[])?.dtype(q, &matrix),
            GgmlDType::Q5K
        );
        assert!(QuantPolicy::parse("q7_x", &[]).is_err());
        assert!(QuantPolicy::parse("q8_0", &["self_attn".to_string()]).is_err());
        assert!(QuantPolicy::parse("q8_0", &["self_attn=q7_x".to_string()]).is_err());
        Ok(())
    }

    #[test]
    fn each_tensor_takes_its_kind_of_precision() {
        let mix = QuantPolicy::q6_k_mix();
        for (name, dims, want) in [
            (
                "encoder.layers.0.norm_out.weight",
                vec![1024],
                GgmlDType::F32,
            ),
            (
                "encoder.layers.0.self_attn.pos_bias_u",
                vec![8, 128],
                GgmlDType::F32,
            ),
            (
                "encoder.layers.0.conv.depthwise_conv.weight",
                vec![1024, 1, 9],
                GgmlDType::F16,
            ),
            (EMBEDDING, vec![8193, 640], GgmlDType::F16),
            (
                "encoder.layers.0.self_attn.linear_q.weight",
                vec![1024, 1024],
                GgmlDType::Q6K,
            ),
            (
                "encoder.layers.0.feed_forward1.linear1.weight",
                vec![4096, 1024],
                GgmlDType::Q8_0,
            ),
            (
                "decoder.prediction.dec_rnn.lstm.ih_l0.weight",
                vec![2560, 640],
                GgmlDType::F16,
            ),
            ("joint.joint_net.2.weight", vec![8198, 640], GgmlDType::F16),
            ("joint.pred.weight", vec![640, 640], GgmlDType::F16),
            (
                "encoder.pre_encode.out.weight",
                vec![1024, 4096],
                GgmlDType::Q6K,
            ),
        ] {
            assert_eq!(mix.dtype(name, &zeros(&dims)), want, "{name}");
        }
        // a 640-wide row fills no 256-element Q6_K block, so it drops to Q8_0
        let q6 = QuantPolicy::uniform(GgmlDType::Q6K);
        assert_eq!(
            q6.dtype("joint.pred.weight", &zeros(&[640, 640])),
            GgmlDType::Q8_0
        );
    }
}
