//! NeMo's `.nemo` checkpoints: an uncompressed tar of `model_config.yaml` and a torch `model_weights.ckpt`, the
//! weights read in place from the archive.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inference_tensor::nn::VarBuilder;
use inference_tensor::pickle::PthTensors;
use inference_tensor::{DType, Device, Error, Result, Tensor};
use serde::Deserialize;
use serde::de::DeserializeOwned;

use inference_audio::nemo::NemoMelConfig;

use crate::parakeet::{EncoderConfig, STREAMING_ENCODER_TYPE};

pub const EXTENSION: &str = "nemo";
const CONFIG: &str = "model_config.yaml";
const WEIGHTS: &str = "model_weights.ckpt";
// older NeMo saved `.nemo` as a gzipped tar
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
// NeMo's dw_striding pre-encode hard-codes its convs rather than configuring them
const SUBSAMPLING_KERNEL: usize = 3;
const SUBSAMPLING_STRIDE: usize = 2;
// a `.nemo` config names its other members with this prefix
const MEMBER_PREFIX: &str = "nemo:";
// NeMo's preprocessor default when the config leaves it out
const DEFAULT_PREEMPHASIS: f32 = 0.97;
const NO_NORMALIZATION: &str = "NA";
const PER_FEATURE: &str = "per_feature";
const REGULAR: &str = "regular";
const CHUNKED_LIMITED: &str = "chunked_limited";
const CAUSAL: &str = "causal";
const LAYER_NORM: &str = "layer_norm";

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

/// An opened `.nemo`: its config text, and where each member sits in the archive.
pub struct NemoArchive {
    path: PathBuf,
    config: String,
    weights: (u64, u64),
    members: HashMap<String, (u64, u64)>,
}

#[derive(Deserialize)]
struct Target {
    target: Option<String>,
}

impl NemoArchive {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut magic = [0u8; 2];
        std::io::Read::read_exact(&mut file, &mut magic)?;
        if magic == GZIP_MAGIC {
            return Err(msg(format!(
                "`{}` is a gzipped `.nemo`; unpack it to a plain tar (`gunzip -c`) to load it",
                path.display()
            )));
        }
        std::io::Seek::rewind(&mut file)?;
        let mut archive = tar::Archive::new(file);
        let (mut config, mut weights) = (None, None);
        let mut members = HashMap::new();
        for entry in archive.entries_with_seek()? {
            let mut entry = entry?;
            let name = entry
                .path()?
                .file_name()
                .map(|n| n.to_string_lossy().into_owned());
            match name.as_deref() {
                Some(CONFIG) => {
                    let mut text = String::new();
                    std::io::Read::read_to_string(&mut entry, &mut text)?;
                    config = Some(text);
                }
                Some(WEIGHTS) => weights = Some((entry.raw_file_position(), entry.size())),
                Some(other) => {
                    members.insert(other.to_string(), (entry.raw_file_position(), entry.size()));
                }
                None => {}
            }
        }
        let missing = |file: &str| msg(format!("`{}` holds no `{file}`", path.display()));
        Ok(Self {
            path: path.to_owned(),
            config: config.ok_or_else(|| missing(CONFIG))?,
            weights: weights.ok_or_else(|| missing(WEIGHTS))?,
            members,
        })
    }

    /// The bytes of a member the config names, as `nemo:<hash>_tokenizer.model`.
    pub fn member(&self, name: &str) -> Result<Vec<u8>> {
        let name = name.strip_prefix(MEMBER_PREFIX).unwrap_or(name);
        let &(offset, len) = self
            .members
            .get(name)
            .ok_or_else(|| msg(format!("`{}` holds no `{name}`", self.path.display())))?;
        let mut file = std::fs::File::open(&self.path)?;
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(offset))?;
        let mut bytes = vec![0u8; len as usize];
        std::io::Read::read_exact(&mut file, &mut bytes)?;
        Ok(bytes)
    }

    pub fn config<T: DeserializeOwned>(&self) -> Result<T> {
        serde_saphyr::from_str(&self.config).map_err(msg)
    }

    /// The NeMo class the checkpoint restores to, as `nemo.collections.asr.models.SortformerEncLabelModel`.
    pub fn target(&self) -> Result<Option<String>> {
        Ok(self.config::<Target>()?.target)
    }

    /// Every tensor, each name passed through `rename`, as a builder on `device` in `dtype`.
    pub fn var_builder(
        &self,
        rename: impl Fn(&str) -> String,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        let (offset, len) = self.weights;
        let tensors = PthTensors::in_range(&self.path, offset, len, None)?;
        let mut renamed: HashMap<String, Tensor> = HashMap::new();
        for name in tensors.tensor_infos().keys() {
            if let Some(t) = tensors.get(name)? {
                renamed.insert(rename(name), t);
            }
        }
        Ok(VarBuilder::from_tensors(renamed, dtype, device))
    }
}

/// Writes a `.nemo` of `config`, `tensors` and other `members` (a tokenizer model), as tests build tiny checkpoints.
pub fn write_nemo(
    path: &Path,
    config: &str,
    tensors: &[(&str, &Tensor)],
    members: &[(&str, &[u8])],
) -> Result<()> {
    let mut weights = std::io::Cursor::new(Vec::new());
    inference_tensor::pickle::write_pth(&mut weights, tensors)?;
    let mut builder = tar::Builder::new(std::fs::File::create(path)?);
    let parts = [
        (CONFIG, config.as_bytes()),
        (WEIGHTS, weights.get_ref().as_slice()),
    ];
    for (name, data) in members.iter().copied().chain(parts) {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, name, data)?;
    }
    builder.finish()?;
    Ok(())
}

// SentencePiece's ModelProto: pieces are field 1, each a message of its text (1) and its type (3)
const PROTO_PIECES: u64 = 1;
const PROTO_PIECE_TEXT: u64 = 1;
const PROTO_PIECE_TYPE: u64 = 3;
const WIRE_VARINT: u64 = 0;
const WIRE_FIXED64: u64 = 1;
const WIRE_BYTES: u64 = 2;
const WIRE_FIXED32: u64 = 5;
// SentencePiece piece types: transformers makes the unknown and control pieces special, and a byte piece means
// byte fallback, which a decode-only tokenizer here does not reassemble
const PIECE_UNKNOWN: u64 = 2;
const PIECE_CONTROL: u64 = 3;
const PIECE_BYTE: u64 = 6;
const METASPACE: char = '\u{2581}';

fn varint(bytes: &[u8], at: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes
            .get(*at)
            .ok_or_else(|| msg("a SentencePiece model ends inside a varint"))?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(msg("a SentencePiece model holds an over-long varint"))
}

enum Field<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

// one protobuf message's varint and length-delimited fields with their numbers; fixed-width ones are skipped
fn fields(bytes: &[u8]) -> Result<Vec<(u64, Field<'_>)>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let key = varint(bytes, &mut at)?;
        let (number, wire) = (key >> 3, key & 7);
        let width = match wire {
            WIRE_VARINT => {
                out.push((number, Field::Varint(varint(bytes, &mut at)?)));
                continue;
            }
            WIRE_FIXED64 => 8,
            WIRE_FIXED32 => 4,
            WIRE_BYTES => varint(bytes, &mut at)? as usize,
            wire => return Err(msg(format!("a SentencePiece model uses wire type {wire}"))),
        };
        let payload = at
            .checked_add(width)
            .and_then(|end| bytes.get(at..end))
            .ok_or_else(|| msg("a SentencePiece model ends inside a field"))?;
        if wire == WIRE_BYTES {
            out.push((number, Field::Bytes(payload)));
        }
        at += width;
    }
    Ok(out)
}

/// A SentencePiece piece, and whether a transcript skips it.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    pub text: String,
    pub special: bool,
}

/// The pieces of a serialized SentencePiece model (`tokenizer.model`), in id order.
pub fn sentencepiece_pieces(model: &[u8]) -> Result<Vec<Piece>> {
    let mut pieces = Vec::new();
    for (number, field) in fields(model)? {
        let (PROTO_PIECES, Field::Bytes(piece)) = (number, field) else {
            continue;
        };
        let (mut text, mut kind) = (None, None);
        for (number, field) in fields(piece)? {
            match (number, field) {
                (PROTO_PIECE_TEXT, Field::Bytes(t)) => {
                    text = Some(String::from_utf8_lossy(t).into_owned())
                }
                (PROTO_PIECE_TYPE, Field::Varint(t)) => kind = Some(t),
                _ => {}
            }
        }
        if kind == Some(PIECE_BYTE) {
            return Err(msg(
                "a byte-fallback SentencePiece model is not one this decodes",
            ));
        }
        pieces.push(Piece {
            text: text.ok_or_else(|| msg("a SentencePiece piece without its text"))?,
            special: matches!(kind, Some(PIECE_UNKNOWN | PIECE_CONTROL)),
        });
    }
    Ok(pieces)
}

/// A decoding tokenizer over SentencePiece pieces, as transformers converts NeMo's: Metaspace and specials skipped.
pub fn piece_tokenizer(pieces: &[Piece]) -> Result<tokenizers::Tokenizer> {
    use tokenizers::decoders::metaspace::{Metaspace, PrependScheme};
    let vocab: tokenizers::models::bpe::Vocab = pieces
        .iter()
        .enumerate()
        .map(|(id, piece)| (piece.text.clone(), id as u32))
        .collect();
    let mut bpe = tokenizers::models::bpe::BPE::builder().vocab_and_merges(vocab, Vec::new());
    if let Some(unknown) = pieces.iter().find(|p| p.special) {
        bpe = bpe.unk_token(unknown.text.clone());
    }
    let mut tokenizer = tokenizers::Tokenizer::new(bpe.build().map_err(msg)?);
    tokenizer.with_decoder(Some(Metaspace::new(METASPACE, PrependScheme::Always, true)));
    let special: Vec<tokenizers::AddedToken> = pieces
        .iter()
        .filter(|p| p.special)
        .map(|p| tokenizers::AddedToken::from(p.text.clone(), true))
        .collect();
    tokenizer.add_special_tokens(special).map_err(msg)?;
    Ok(tokenizer)
}

fn default_preemphasis() -> f32 {
    DEFAULT_PREEMPHASIS
}

/// The `preprocessor:` section: NeMo's log-mel frontend.
#[derive(Debug, Clone, Deserialize)]
pub struct NemoPreprocessorConfig {
    pub sample_rate: u32,
    pub window_size: f64,
    pub window_stride: f64,
    pub features: usize,
    pub n_fft: usize,
    #[serde(default)]
    pub normalize: Option<String>,
    #[serde(default = "default_preemphasis")]
    pub preemph: f32,
}

impl NemoPreprocessorConfig {
    pub fn hop_length(&self) -> usize {
        (self.window_stride * f64::from(self.sample_rate)).round() as usize
    }

    /// The mel frontend, for the two normalizations NeMo's FastConformers use: per feature, or none.
    pub fn mel(&self) -> Result<NemoMelConfig> {
        let normalize = match self.normalize.as_deref() {
            Some(PER_FEATURE) => true,
            None | Some(NO_NORMALIZATION) => false,
            Some(other) => {
                return Err(msg(format!(
                    "a `{other}` normalization is not one this computes"
                )));
            }
        };
        Ok(NemoMelConfig {
            sample_rate: self.sample_rate,
            n_fft: self.n_fft,
            win_length: (self.window_size * f64::from(self.sample_rate)).round() as usize,
            hop_length: self.hop_length(),
            n_mels: self.features,
            preemphasis: Some(self.preemph),
            normalize,
        })
    }
}

fn default_true() -> bool {
    true
}

fn default_style() -> String {
    REGULAR.to_string()
}

/// `att_context_size`: one `[left, right]`, or a multi-lookahead model's several, the first the default.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AttentionContext {
    One(Vec<i64>),
    Several(Vec<Vec<i64>>),
}

impl AttentionContext {
    fn first(&self) -> Option<(i64, i64)> {
        let pair = match self {
            Self::One(pair) => pair.as_slice(),
            Self::Several(pairs) => pairs.first()?.as_slice(),
        };
        match pair {
            [left, right] => Some((*left, *right)),
            _ => None,
        }
    }
}

/// `conv_context_size`: `causal`, or an explicit `[left, right]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ConvContext {
    Named(String),
    Pair(Vec<i64>),
}

/// The `encoder:` section of a FastConformer checkpoint's config.
#[derive(Debug, Clone, Deserialize)]
pub struct NemoEncoderConfig {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub subsampling: String,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub self_attention_model: String,
    pub conv_kernel_size: usize,
    pub xscaling: bool,
    #[serde(default)]
    pub pos_emb_max_len: Option<usize>,
    #[serde(default = "default_true")]
    pub use_bias: bool,
    #[serde(default)]
    pub causal_downsampling: bool,
    #[serde(default = "default_style")]
    pub att_context_style: String,
    #[serde(default)]
    pub att_context_size: Option<AttentionContext>,
    #[serde(default)]
    pub conv_norm_type: Option<String>,
    #[serde(default)]
    pub conv_context_size: Option<ConvContext>,
    /// A time reduction inside the encoder, which this does not run.
    #[serde(default)]
    pub reduction: Option<String>,
}

impl NemoEncoderConfig {
    /// Cache-aware streaming: causal subsampling and convs, chunked-limited attention, layer-normed convs.
    pub fn is_streaming(&self) -> bool {
        self.att_context_style == CHUNKED_LIMITED
    }

    // Some(true) causal, Some(false) centred, None any other context
    fn causal_convs(&self) -> Option<bool> {
        let k = self.conv_kernel_size as i64 - 1;
        match &self.conv_context_size {
            None => Some(false),
            Some(ConvContext::Named(name)) => (name == CAUSAL).then_some(true),
            Some(ConvContext::Pair(pair)) if pair == &[k, 0] => Some(true),
            Some(ConvContext::Pair(pair)) if pair == &[k / 2, k / 2] => Some(false),
            Some(ConvContext::Pair(_)) => None,
        }
    }

    /// The same encoder as transformers configures it: Parakeet's, or the streaming one.
    pub fn parakeet(&self) -> Result<EncoderConfig> {
        if self.subsampling != "dw_striding" || self.self_attention_model != "rel_pos" {
            return Err(msg(format!(
                "a `{}` pre-encode with `{}` attention is not a FastConformer this loads",
                self.subsampling, self.self_attention_model
            )));
        }
        if self.reduction.is_some() {
            return Err(msg(
                "an encoder with a time reduction is not a FastConformer this loads",
            ));
        }
        let causal = self.causal_convs().ok_or_else(|| {
            msg("a conv context neither causal nor centred is not one this loads")
        })?;
        let layer_norm = self.conv_norm_type.as_deref() == Some(LAYER_NORM);
        let context = self
            .att_context_size
            .as_ref()
            .and_then(AttentionContext::first);
        let (model_type, sliding_window, lookahead) = if self.is_streaming() {
            let Some((left, right)) = context.filter(|(l, r)| *l >= 0 && *r >= 0) else {
                return Err(msg(
                    "a chunked-limited encoder needs a finite `att_context_size`",
                ));
            };
            if !(self.causal_downsampling && causal && layer_norm) {
                return Err(msg(
                    "a chunked-limited encoder this loads has causal subsampling and convs, layer-normed",
                ));
            }
            (
                Some(STREAMING_ENCODER_TYPE.to_string()),
                Some(left as usize + 1),
                Some(right as usize),
            )
        } else {
            // NeMo bands a regular encoder's attention to a finite context; this runs only full attention there
            if context.is_some_and(|c| c != (-1, -1)) || self.att_context_style != REGULAR {
                return Err(msg(
                    "limited-context attention outside a chunked-limited encoder is not one this loads",
                ));
            }
            if self.causal_downsampling || causal || layer_norm {
                return Err(msg(
                    "causal or layer-normed convs without chunked-limited attention are not a FastConformer this loads",
                ));
            }
            (None, None, None)
        };
        Ok(EncoderConfig {
            hidden_size: self.d_model,
            intermediate_size: self.d_model * self.ff_expansion_factor,
            num_hidden_layers: self.n_layers,
            num_attention_heads: self.n_heads,
            num_mel_bins: self.feat_in,
            conv_kernel_size: self.conv_kernel_size,
            subsampling_conv_channels: self.subsampling_conv_channels,
            subsampling_conv_kernel_size: SUBSAMPLING_KERNEL,
            subsampling_conv_stride: SUBSAMPLING_STRIDE,
            subsampling_factor: self.subsampling_factor,
            attention_bias: self.use_bias,
            convolution_bias: self.use_bias,
            scale_input: self.xscaling,
            model_type,
            sliding_window,
            default_num_lookahead_tokens: lookahead,
        })
    }
}

// NeMo's FastConformer names to the transformers names the Parakeet encoder loads
const ENCODER_RENAMES: [(&str, &str); 10] = [
    (".pre_encode.conv.", ".subsampling.layers."),
    (".pre_encode.out.", ".subsampling.linear."),
    (".self_attn.linear_q.", ".self_attn.q_proj."),
    (".self_attn.linear_k.", ".self_attn.k_proj."),
    (".self_attn.linear_v.", ".self_attn.v_proj."),
    (".self_attn.linear_out.", ".self_attn.o_proj."),
    (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
    (".self_attn.pos_bias_u", ".self_attn.bias_u"),
    (".self_attn.pos_bias_v", ".self_attn.bias_v"),
    (".conv.batch_norm.", ".conv.norm."),
];

// NeMo's ASR heads to transformers' Parakeet names, by prefix
const HEAD_RENAMES: [(&str, &str); 7] = [
    ("decoder.prediction.embed.", "decoder.embedding."),
    ("decoder.prediction.dec_rnn.lstm.", "decoder.lstm."),
    ("joint.pred.", "decoder.decoder_projector."),
    ("joint.enc.", "encoder_projector."),
    ("joint.joint_net.2.", "joint.head."),
    ("decoder.decoder_layers.0.", "ctc_head."),
    ("ctc_decoder.decoder_layers.0.", "ctc_head."),
];
// a streaming encoder's causal pre-encode keeps NeMo's dw_striding indices; transformers names them
const STREAMING_SUBSAMPLING: [(&str, &str); 5] = [
    (
        "encoder.subsampling.layers.0.",
        "encoder.subsampling.conv_in.",
    ),
    (
        "encoder.subsampling.layers.2.",
        "encoder.subsampling.layers.0.depthwise_conv.",
    ),
    (
        "encoder.subsampling.layers.3.",
        "encoder.subsampling.layers.0.pointwise_conv.",
    ),
    (
        "encoder.subsampling.layers.5.",
        "encoder.subsampling.layers.1.depthwise_conv.",
    ),
    (
        "encoder.subsampling.layers.6.",
        "encoder.subsampling.layers.1.pointwise_conv.",
    ),
];

/// A NeMo ASR checkpoint's name as the Parakeet model names it: the encoder's, then the head's.
pub fn parakeet_name(name: &str, streaming: bool) -> String {
    let name = parakeet_encoder_name(name);
    let renames = HEAD_RENAMES.iter().chain(
        streaming
            .then_some(STREAMING_SUBSAMPLING.iter())
            .into_iter()
            .flatten(),
    );
    for (from, to) in renames {
        if let Some(rest) = name.strip_prefix(from) {
            return format!("{to}{rest}");
        }
    }
    name
}

/// A NeMo checkpoint name under `encoder.` as the Parakeet encoder names it; others pass through.
pub fn parakeet_encoder_name(name: &str) -> String {
    if !name.starts_with("encoder.") {
        return name.to_string();
    }
    ENCODER_RENAMES
        .iter()
        .fold(name.to_string(), |n, (from, to)| n.replace(from, to))
}

/// The NeMo name of a Parakeet model name: the inverse of [`parakeet_name`].
pub fn nemo_name(name: &str, streaming: bool) -> String {
    let renames = streaming
        .then_some(STREAMING_SUBSAMPLING.iter())
        .into_iter()
        .flatten()
        .chain(HEAD_RENAMES.iter());
    for (nemo, ours) in renames {
        if let Some(rest) = name.strip_prefix(ours) {
            return nemo_encoder_name(&format!("{nemo}{rest}"));
        }
    }
    nemo_encoder_name(name)
}

/// The NeMo name of a Parakeet encoder name: the inverse of [`parakeet_encoder_name`].
pub fn nemo_encoder_name(name: &str) -> String {
    if !name.starts_with("encoder.") {
        return name.to_string();
    }
    ENCODER_RENAMES
        .iter()
        .fold(name.to_string(), |n, (nemo, ours)| n.replace(ours, nemo))
}

#[cfg(test)]
mod tests {
    use super::*;

    // a ModelProto of `(piece, type)`s, each with a score field the reader skips, after a long trainer spec
    fn proto(pieces: &[(&str, u8)]) -> Vec<u8> {
        let mut out = vec![0x12, 0x96, 0x01];
        out.extend(std::iter::repeat_n(0u8, 150));
        for (piece, kind) in pieces {
            let mut inner = vec![0x0a, piece.len() as u8];
            inner.extend_from_slice(piece.as_bytes());
            inner.extend_from_slice(&[0x15, 0, 0, 0x80, 0xbf, 0x18, *kind]);
            out.extend_from_slice(&[0x0a, inner.len() as u8]);
            out.extend(inner);
        }
        out
    }

    #[test]
    fn pieces_decode_with_metaspace_and_skip_their_specials() -> anyhow::Result<()> {
        // a 150-byte trainer spec needs a two-byte length; the user-defined `<|cue|>` stays text, as transformers
        // converts it
        let pieces = sentencepiece_pieces(&proto(&[
            ("<unk>", 2),
            ("<de-DE>", 3),
            ("\u{2581}he", 1),
            ("llo", 1),
            ("\u{2581}wor", 1),
            ("ld", 1),
            ("7", 4),
            ("<|cue|>", 4),
        ]))?;
        assert_eq!(pieces.len(), 8);
        let tokenizer = piece_tokenizer(&pieces)?;
        let text = tokenizer
            .decode(&[1, 2, 3, 4, 5, 6], true)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(text, "hello world7");
        assert_eq!(
            tokenizer
                .decode(&[7, 2], true)
                .map_err(anyhow::Error::msg)?,
            "<|cue|> he"
        );
        assert!(sentencepiece_pieces(&proto(&[("<0x0A>", 6)])).is_err());
        Ok(())
    }

    #[test]
    fn fastconformer_names_become_the_parakeet_encoders() {
        for (nemo, ours) in [
            (
                "encoder.pre_encode.conv.5.weight",
                "encoder.subsampling.layers.5.weight",
            ),
            (
                "encoder.pre_encode.out.bias",
                "encoder.subsampling.linear.bias",
            ),
            (
                "encoder.layers.3.self_attn.linear_pos.weight",
                "encoder.layers.3.self_attn.relative_k_proj.weight",
            ),
            (
                "encoder.layers.0.self_attn.pos_bias_u",
                "encoder.layers.0.self_attn.bias_u",
            ),
            (
                "encoder.layers.16.conv.batch_norm.running_var",
                "encoder.layers.16.conv.norm.running_var",
            ),
            (
                "encoder.layers.2.conv.depthwise_conv.weight",
                "encoder.layers.2.conv.depthwise_conv.weight",
            ),
            (
                "transformer_encoder.layers.0.first_sub_layer.query_net.weight",
                "transformer_encoder.layers.0.first_sub_layer.query_net.weight",
            ),
        ] {
            assert_eq!(parakeet_encoder_name(nemo), ours);
            assert_eq!(nemo_encoder_name(ours), nemo);
            assert_eq!(nemo_name(&parakeet_name(nemo, false), false), nemo);
        }
    }

    #[test]
    fn asr_head_and_streaming_names_round_trip() {
        for (nemo, ours, streaming) in [
            (
                "decoder.prediction.embed.weight",
                "decoder.embedding.weight",
                false,
            ),
            (
                "decoder.prediction.dec_rnn.lstm.weight_hh_l1",
                "decoder.lstm.weight_hh_l1",
                false,
            ),
            ("joint.pred.bias", "decoder.decoder_projector.bias", false),
            ("joint.enc.weight", "encoder_projector.weight", false),
            ("joint.joint_net.2.weight", "joint.head.weight", false),
            ("decoder.decoder_layers.0.weight", "ctc_head.weight", false),
            (
                "encoder.pre_encode.conv.0.weight",
                "encoder.subsampling.conv_in.weight",
                true,
            ),
            (
                "encoder.pre_encode.conv.5.bias",
                "encoder.subsampling.layers.1.depthwise_conv.bias",
                true,
            ),
            (
                "encoder.pre_encode.conv.5.bias",
                "encoder.subsampling.layers.5.bias",
                false,
            ),
        ] {
            assert_eq!(parakeet_name(nemo, streaming), ours);
            assert_eq!(nemo_name(ours, streaming), nemo);
        }
        // a hybrid's CTC head lands on the same name its RNN-T loader leaves unread
        assert_eq!(
            parakeet_name("ctc_decoder.decoder_layers.0.bias", false),
            "ctc_head.bias"
        );
    }

    #[test]
    fn an_archive_yields_its_config_and_in_place_weights() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tiny.nemo");
        let bias = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        let yaml =
            "target: nemo.collections.asr.models.SortformerEncLabelModel\nsample_rate: 16000\n";
        write_nemo(&path, yaml, &[("encoder.pre_encode.out.bias", &bias)], &[])?;

        let archive = NemoArchive::open(&path)?;
        assert_eq!(
            archive.target()?.as_deref(),
            Some("nemo.collections.asr.models.SortformerEncLabelModel")
        );
        let vb = archive.var_builder(parakeet_encoder_name, DType::F32, &Device::Cpu)?;
        let read = vb.get(3, "encoder.subsampling.linear.bias")?;
        assert_eq!(read.to_vec1::<f32>()?, [1., 2., 3.]);
        Ok(())
    }
}
