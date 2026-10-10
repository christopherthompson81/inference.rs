//! Parakeet-family recognisers from NVIDIA's own GGUF (`general.architecture = asr`): the NeMo config as `asr.*`
//! keys, NeMo's tensor names, the tokenizer's pieces, and the weights dequantized into the model's dtype.

use std::collections::HashMap;
use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::quantized::gguf_file::{self, Content, Value};
use inference_tensor::{DType, Device, Error, Result, Tensor};

use super::Parakeet;
use super::config::{FeatureConfig, ParakeetConfig, ProcessorConfig};
use crate::nemo::{
    AttentionContext, ConvContext, NemoEncoderConfig, NemoPreprocessorConfig, PAD_PIECE, Piece,
    parakeet_name, piece_tokenizer,
};

const ARCHITECTURE: &str = "asr";
const GGUF_EXTENSION: &str = "gguf";
const SENTENCEPIECE_BPE: &str = "sentencepiece_bpe";
const NO_CONTEXT: &str = "None";
const DW_STRIDING: &str = "dw_striding";
const REL_POS: &str = "rel_pos";
const PRE_ENCODE_OUT: &str = "encoder.pre_encode.out.weight";
// NeMo's defaults where an older converter leaves a key out
const DEFAULT_POSITIONS: usize = 5000;
const DEFAULT_PREEMPHASIS: f64 = 0.97;
const DEFAULT_MAX_SYMBOLS: usize = 10;
// dw_striding's stride-2 convs, padded (kernel - 1, stride - 1) in frequency when causal: f -> f / 2 + 1 bins
const SUBSAMPLING_STRIDE: usize = 2;
// tensors the converter keeps for its own runtime: the sinusoid table and mel filterbank, both recomputed here
const UNUSED: [&str; 2] = ["encoder.pos_enc.pe", "preprocessor.fb"];
// the converter's LSTM names, `lstm.ih_l0.weight`, against PyTorch's `lstm.weight_ih_l0`
const LSTM_PREFIX: &str = "decoder.prediction.dec_rnn.lstm.";
// the converter stores the conformer's 1x1 convs as matrices; PyTorch's carry a kernel axis of 1
const POINTWISE_CONVS: [&str; 2] = [
    ".conv.pointwise_conv1.weight",
    ".conv.pointwise_conv2.weight",
];
// the pieces transformers' conversion makes special; the GGUF keeps no piece types to tell control pieces by
const UNKNOWN_PIECE: &str = "<unk>";

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

/// Whether `path` is a GGUF of NVIDIA's `asr` architecture.
pub fn is_asr_gguf(path: &Path) -> bool {
    let read = || -> Result<bool> {
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
        Ok(gguf_file::peek_architecture(&mut file)?.is_some_and(|a| a == ARCHITECTURE))
    };
    path.extension().is_some_and(|e| e == GGUF_EXTENSION)
        && path.is_file()
        && read().unwrap_or(false)
}

struct Metadata<'a>(&'a HashMap<String, Value>);

impl Metadata<'_> {
    fn get(&self, key: &str) -> Result<&Value> {
        self.0
            .get(key)
            .ok_or_else(|| msg(format!("the GGUF has no `{key}`")))
    }

    fn usize(&self, key: &str) -> Result<usize> {
        Ok(self.get(key)?.to_u64()? as usize)
    }

    fn i64(&self, key: &str) -> Result<i64> {
        let v = self.get(key)?;
        v.to_i32()
            .map(i64::from)
            .or_else(|_| v.to_i64())
            .or_else(|_| v.to_u64().map(|u| u as i64))
    }

    fn f64(&self, key: &str) -> Result<f64> {
        let v = self.get(key)?;
        v.to_f32().map(f64::from).or_else(|_| v.to_f64())
    }

    fn string(&self, key: &str) -> Result<&str> {
        Ok(self.get(key)?.to_string()?.as_str())
    }

    fn bool(&self, key: &str) -> Result<bool> {
        self.get(key)?.to_bool()
    }
}

// the converter's `lstm.ih_l0.weight` as PyTorch's `lstm.weight_ih_l0`; other names pass through
fn pytorch_lstm_name(name: &str) -> String {
    let Some(rest) = name.strip_prefix(LSTM_PREFIX) else {
        return name.to_string();
    };
    match rest.split_once('.') {
        Some((gate_layer, param)) => format!("{LSTM_PREFIX}{param}_{gate_layer}"),
        None => name.to_string(),
    }
}

// the pre-encode's output linear takes channels x frequency bins: the causal padding leaves one bin more per stage
fn causal_subsampling(m: &Metadata, content: &Content) -> Result<bool> {
    let info = content
        .tensor_infos
        .get(PRE_ENCODE_OUT)
        .ok_or_else(|| msg(format!("the GGUF has no `{PRE_ENCODE_OUT}`")))?;
    let channels = m.usize("asr.encoder.subsampling_conv_channels")?;
    let stages = m.usize("asr.encoder.subsampling_factor")?.ilog2();
    let causal_bins = (0..stages).fold(m.usize("asr.encoder.feat_in")?, |f, _| {
        f / SUBSAMPLING_STRIDE + 1
    });
    Ok(info.shape.dims().last() == Some(&(channels * causal_bins)))
}

fn encoder_config(m: &Metadata, content: &Content) -> Result<NemoEncoderConfig> {
    let style = m.string("asr.encoder.att_context_style")?.to_string();
    let conv_context = match m.string("asr.encoder.conv_context")? {
        NO_CONTEXT => None,
        named => Some(ConvContext::Named(named.to_string())),
    };
    let d_model = m.usize("asr.encoder.d_model")?;
    Ok(NemoEncoderConfig {
        feat_in: m.usize("asr.encoder.feat_in")?,
        n_layers: m.usize("asr.encoder.n_layers")?,
        d_model,
        n_heads: m.usize("asr.encoder.n_heads")?,
        ff_expansion_factor: m.usize("asr.encoder.d_ff")? / d_model,
        subsampling: DW_STRIDING.to_string(),
        subsampling_factor: m.usize("asr.encoder.subsampling_factor")?,
        subsampling_conv_channels: m.usize("asr.encoder.subsampling_conv_channels")?,
        self_attention_model: REL_POS.to_string(),
        conv_kernel_size: m.usize("asr.encoder.conv_kernel_size")?,
        xscaling: m.bool("asr.encoder.xscaling")?,
        pos_emb_max_len: Some(
            m.usize("asr.encoder.pos_emb_max_len")
                .unwrap_or(DEFAULT_POSITIONS),
        ),
        use_bias: m.bool("asr.encoder.use_bias")?,
        // the converter writes no flag for it; the pre-encode's output width tells
        causal_downsampling: causal_subsampling(m, content)?,
        att_context_style: style,
        att_context_size: Some(AttentionContext::One(vec![
            m.i64("asr.encoder.offline_left_ctx")?,
            m.i64("asr.encoder.offline_right_ctx")?,
        ])),
        conv_norm_type: Some(m.string("asr.encoder.conv_norm")?.to_string()),
        conv_context_size: conv_context,
        reduction: None,
    })
}

fn durations(m: &Metadata) -> Result<Vec<usize>> {
    m.get("asr.tdt.durations")?
        .to_vec()?
        .iter()
        .map(|v| {
            v.to_u32()
                .map(|v| v as usize)
                .or_else(|_| v.to_i32().map(|v| v as usize))
        })
        .collect()
}

impl Parakeet {
    /// A Parakeet or streaming recogniser from NVIDIA's `asr` GGUF, its weights dequantized into `dtype`.
    pub fn load_gguf(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
        let content = Content::read(&mut file)?;
        let m = Metadata(&content.metadata);
        if m.string("general.architecture")? != ARCHITECTURE {
            return Err(msg(format!("`{}` is not an `asr` GGUF", path.display())));
        }
        if m.usize("asr.rnnt.num_prompts").unwrap_or(0) > 0 {
            return Err(msg(
                "a prompted GGUF (Nemotron-3.5) loads from its transformers layout instead",
            ));
        }
        let (model_type, durations) = match m.string("asr.head_type")? {
            "tdt" => ("parakeet_tdt", durations(&m)?),
            "rnnt" => ("parakeet_rnnt", Vec::new()),
            other => return Err(msg(format!("an `{other}` head is not one this loads"))),
        };
        let encoder_config = encoder_config(&m, &content)?.parakeet()?;
        let streaming = encoder_config.is_streaming();
        let mel = NemoPreprocessorConfig {
            sample_rate: m.usize("asr.preprocessor.sample_rate")? as u32,
            window_size: m.f64("asr.preprocessor.window_size")?,
            window_stride: m.f64("asr.preprocessor.window_stride")?,
            features: m.usize("asr.preprocessor.features")?,
            n_fft: m.usize("asr.preprocessor.n_fft")?,
            normalize: Some(m.string("asr.preprocessor.normalize")?.to_string()),
            preemph: m
                .f64("asr.preprocessor.preemph")
                .unwrap_or(DEFAULT_PREEMPHASIS) as f32,
        }
        .mel()?;
        if mel.normalize == streaming {
            return Err(msg(
                "a streaming encoder takes raw log-mel features and an offline one normalized ones",
            ));
        }
        let config = ParakeetConfig {
            model_type: model_type.to_string(),
            encoder_config,
            vocab_size: m.usize("asr.rnnt.vocab_size")?,
            pad_token_id: None,
            blank_token_id: Some(m.usize("asr.rnnt.blank_id")? as u32),
            decoder_hidden_size: Some(m.usize("asr.rnnt.pred_hidden")?),
            num_decoder_layers: Some(m.usize("asr.rnnt.pred_num_layers")?),
            durations,
            max_symbols_per_step: m
                .usize("asr.rnnt.max_symbols_per_step")
                .unwrap_or(DEFAULT_MAX_SYMBOLS),
            num_prompts: None,
            prompt_intermediate_size: None,
            default_prompt_id: None,
        };
        let processor = ProcessorConfig {
            feature_extractor: FeatureConfig {
                feature_size: mel.n_mels,
                sampling_rate: mel.sample_rate,
                n_fft: mel.n_fft,
                win_length: mel.win_length,
                hop_length: mel.hop_length,
                preemphasis: mel.preemphasis,
            },
            prompt_dictionary: HashMap::new(),
        };
        if m.string("asr.tokenizer.type")? != SENTENCEPIECE_BPE {
            return Err(msg(
                "a tokenizer other than SentencePiece BPE is not one this decodes",
            ));
        }
        let pieces = m
            .get("asr.tokenizer.vocab")?
            .to_vec()?
            .iter()
            .map(|v| {
                let text = v.to_string()?.clone();
                Ok(Piece {
                    special: text == UNKNOWN_PIECE || text == PAD_PIECE,
                    text,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tokenizer = piece_tokenizer(&pieces)?;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for name in content.tensor_infos.keys() {
            if UNUSED.contains(&name.as_str()) {
                continue;
            }
            let mut t = content
                .tensor(&mut file, name, &Device::Cpu)?
                .dequantize(&Device::Cpu)?;
            if t.rank() == 2 && POINTWISE_CONVS.iter().any(|c| name.ends_with(c)) {
                t = t.unsqueeze(2)?;
            }
            tensors.insert(parakeet_name(&pytorch_lstm_name(name), streaming), t);
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        Self::new(config, processor, tokenizer, vb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_converters_lstm_names_become_pytorchs() {
        assert_eq!(
            pytorch_lstm_name("decoder.prediction.dec_rnn.lstm.ih_l0.weight"),
            "decoder.prediction.dec_rnn.lstm.weight_ih_l0"
        );
        assert_eq!(
            pytorch_lstm_name("decoder.prediction.dec_rnn.lstm.hh_l1.bias"),
            "decoder.prediction.dec_rnn.lstm.bias_hh_l1"
        );
        assert_eq!(pytorch_lstm_name("joint.pred.weight"), "joint.pred.weight");
    }
}
