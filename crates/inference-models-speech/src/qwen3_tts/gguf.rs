//! llama.cpp's Qwen3-TTS GGUFs: the talker (`qwen3tts`, the text projection folded into its token table and the codec
//! embedding appended to it) and an mmproj with the speaker encoder, the code predictor and the codec decoder
//! (`qwen3tts_spkenc`, `qwen3tts_gen`). The config is rebuilt from their metadata and token names; the codec decoder's
//! attention shape, which llama.cpp hardcodes, is the 12 Hz release's. They carry no codec encoder, so a Base GGUF
//! clones from the x-vector alone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use inference_quant::{
    GgufArchive, GgufBindingMap, GgufTensorBinding, GgufWeightSource, ShardedVarBuilder,
};
use inference_tensor::quantized::gguf_file::{Value, peek_architecture};
use inference_tensor::{DType, Device};
use tokenizers::Tokenizer;

use super::config::{
    CodePredictorConfig, CodecConfig, CodecDecoderConfig, Qwen3TtsConfig, SpeakerEncoderConfig,
    TalkerConfig,
};
use crate::layers::Activation;

pub const ARCHITECTURE: &str = "qwen3tts";
const GGUF_EXTENSION: &str = "gguf";
const MMPROJ_PREFIX: &str = "mmproj-";
const TOKENS: &str = "tokenizer.ggml.tokens";
const TOKEN_TYPES: &str = "tokenizer.ggml.token_type";
const MERGES: &str = "tokenizer.ggml.merges";
// llama.cpp's control and user-defined token types: added tokens, matched whole before BPE (special or not)
const CONTROL: u32 = 3;
const USER_DEFINED: u32 = 4;
// the codec rows llama.cpp appends to the text vocabulary, named by their config keys
const FIRST_CODEC_TOKEN: &str = "<|codec_0|>";
const LANGUAGE_TOKEN: &str = "<|codec_language_";
const SPEAKER_TOKEN: &str = "<|spk_";
const TTS_PAD: &str = "<tts_pad>";
const TTS_BOS: &str = "<tts_text_bos>";
const TTS_EOS: &str = "<tts_text_eod>";
const SAMPLE_RATE: u32 = 24_000;
// the 12 Hz codec decoder's settings that llama.cpp hardcodes rather than stores; its heads follow from the head size
const CODEC_HEAD_DIM: usize = 64;
const CODEC_ROPE_THETA: f64 = 10_000.0;
const CODEC_WINDOW: usize = 72;
const CODEC_EPS: f64 = 1e-5;
const CODEC_POSITIONS: usize = 8000;
const PREDICTOR_POSITIONS: usize = 65_536;
// the talker width the 0.6B checkpoints have, which take no instructions
const SMALL_HIDDEN: usize = 1024;

/// Whether `path` is a Qwen3-TTS talker GGUF.
pub fn is_qwen3_tts_gguf(path: &Path) -> bool {
    let read = || -> inference_tensor::Result<bool> {
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
        Ok(peek_architecture(&mut file)?.is_some_and(|a| a == ARCHITECTURE))
    };
    path.extension().is_some_and(|e| e == GGUF_EXTENSION)
        && path.is_file()
        && read().unwrap_or(false)
}

/// The talker GGUF `model_id` names (the file, or a directory holding one) and the mmproj beside it: the one llama.cpp
/// names after it (`mmproj-<talker file>`), else the directory's only mmproj.
pub fn files(model_id: &Path) -> Option<(PathBuf, PathBuf)> {
    let ggufs = |dir: &Path| -> Vec<PathBuf> {
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == GGUF_EXTENSION))
            .collect();
        found.sort();
        found
    };
    let is_mmproj = |p: &Path| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(MMPROJ_PREFIX))
    };
    let talker = if model_id.is_dir() {
        let mut talkers = ggufs(model_id)
            .into_iter()
            .filter(|p| !is_mmproj(p) && is_qwen3_tts_gguf(p));
        match (talkers.next(), talkers.next()) {
            (Some(only), None) => only,
            _ => return None,
        }
    } else if is_qwen3_tts_gguf(model_id) {
        model_id.to_path_buf()
    } else {
        return None;
    };
    let dir = talker.parent()?;
    let named = dir.join(format!("{MMPROJ_PREFIX}{}", talker.file_name()?.to_str()?));
    if named.is_file() {
        return Some((talker, named));
    }
    let mut mmprojs = ggufs(dir).into_iter().filter(|p| is_mmproj(p));
    match (mmprojs.next(), mmprojs.next()) {
        (Some(only), None) => Some((talker, only)),
        _ => None,
    }
}

fn value<'a>(archive: &'a GgufArchive, key: &str) -> anyhow::Result<&'a Value> {
    archive
        .metadata_value(key)
        .ok_or_else(|| anyhow::anyhow!("the Qwen3-TTS GGUF has no `{key}`"))
}

fn usize_of(archive: &GgufArchive, key: &str) -> anyhow::Result<usize> {
    Ok(usize::try_from(value(archive, key)?.to_u32()?)?)
}

fn f64_of(archive: &GgufArchive, key: &str) -> anyhow::Result<f64> {
    Ok(f64::from(value(archive, key)?.to_f32()?))
}

// a tensor's shape, outermost first as PyTorch has it
fn shape(archive: &GgufArchive, name: &str) -> anyhow::Result<Vec<usize>> {
    Ok(archive.tensor_info(name)?.shape().to_vec())
}

fn count(archive: &GgufArchive, prefix: &str, suffix: &str) -> usize {
    (0..)
        .take_while(|i| archive.contains_tensor(&format!("{prefix}{i}{suffix}")))
        .count()
}

const LAYER_NAMES: [(&str, &str); 11] = [
    ("attn_norm", "input_layernorm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
];
// the mmproj's code predictor and codec transformer layers name their norms and output otherwise
const GEN_LAYER_NAMES: [(&str, &str); 11] = [
    ("ln1", "input_layernorm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_out", "self_attn.o_proj"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("ln2", "post_attention_layernorm"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
];

struct Bindings<'a> {
    archive: &'a GgufArchive,
    map: GgufBindingMap,
}

impl Bindings<'_> {
    fn bind(&mut self, native: impl Into<String>, binding: GgufTensorBinding) {
        self.map.insert(native.into(), binding);
    }

    // `gguf` under `native`, when the file has it
    fn direct(&mut self, native: impl Into<String>, gguf: &str) {
        if self.archive.contains_tensor(gguf) {
            self.bind(native, GgufTensorBinding::tensor(gguf));
        }
    }

    // a conv or linear's weight and bias
    fn module(&mut self, native: &str, gguf: &str) {
        self.direct(format!("{native}.weight"), &format!("{gguf}.weight"));
        self.direct(format!("{native}.bias"), &format!("{gguf}.bias"));
    }

    // llama.cpp stores SnakeBeta as `e^alpha` and `1 / (e^beta + 1e-9)`; undone to alpha and (to within 1e-9) beta
    fn snake(&mut self, native: &str, gguf: &str) {
        self.bind(
            format!("{native}.alpha"),
            GgufTensorBinding::tensor(format!("{gguf}.alpha")).log(),
        );
        self.bind(
            format!("{native}.beta"),
            GgufTensorBinding::tensor(format!("{gguf}.beta"))
                .log()
                .affine(-1.0, 0.0),
        );
    }

    // a normalized codebook as `embedding_sum` with a usage of ones
    fn codebook(&mut self, native: &str, gguf: GgufTensorBinding, rows: usize, dim: usize) {
        let gguf = gguf.reshape(vec![rows, dim]);
        self.bind(
            format!("{native}.cluster_usage"),
            gguf.clone()
                .slice(1, 0, 1)
                .reshape(vec![rows])
                .affine(0.0, 1.0),
        );
        self.bind(format!("{native}.embedding_sum"), gguf);
    }
}

// the speaker encoder's widths and kernels from its tensors; its dilations, which no shape shows, are the reference's
fn speaker_config(archive: &GgufArchive) -> anyhow::Result<SpeakerEncoderConfig> {
    let blocks: Vec<usize> = (1..)
        .take_while(|i| archive.contains_tensor(&format!("a.blk.{i}.conv_pw1.weight")))
        .collect();
    let first = shape(archive, "a.conv1d.0.weight")?;
    let out = shape(archive, "a.conv_out.weight")?;
    let mut channels = vec![first[0]];
    let mut kernels = vec![first[2]];
    for i in &blocks {
        channels.push(shape(archive, &format!("a.blk.{i}.conv_pw1.weight"))?[0]);
        kernels.push(shape(archive, &format!("a.blk.{i}.res2.0.weight"))?[2]);
    }
    channels.push(out[0]);
    kernels.push(out[2]);
    let first_block = blocks.first().copied().unwrap_or(1);
    let defaults = SpeakerEncoderConfig::ecapa(
        usize_of(archive, "clip.audio.num_mel_bins")?,
        usize_of(archive, "clip.audio.projection_dim")?,
        SAMPLE_RATE,
    );
    Ok(SpeakerEncoderConfig {
        enc_attention_channels: shape(archive, "a.asp_tdnn.weight")?[0],
        enc_res2net_scale: count(archive, &format!("a.blk.{first_block}.res2."), ".weight") + 1,
        enc_se_channels: shape(archive, &format!("a.blk.{first_block}.se_conv1.weight"))?[0],
        enc_channels: channels,
        enc_kernel_sizes: kernels,
        ..defaults
    })
}

/// What a talker GGUF and its mmproj load as: the config, the weights under the release's names, the tokenizer.
pub struct GgufCheckpoint {
    pub config: Qwen3TtsConfig,
    pub codec: CodecConfig,
    pub weights: ShardedVarBuilder,
    pub codec_weights: ShardedVarBuilder,
    pub tokenizer: Tokenizer,
}

/// Reads `talker` and `mmproj`; the talker's linears keep their GGUF types, the codec runs in f32.
pub fn read(
    talker: &Path,
    mmproj: &Path,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<GgufCheckpoint> {
    let mut archive = GgufArchive::open_file(talker)?;
    archive.merge_component(GgufArchive::open_file(mmproj)?)?;
    let a = |key: &str| format!("{ARCHITECTURE}.{key}");

    let tokens: Vec<String> = value(&archive, TOKENS)?
        .to_vec()?
        .iter()
        .map(|t| t.to_string().cloned())
        .collect::<inference_tensor::Result<_>>()?;
    let types: Vec<u32> = value(&archive, TOKEN_TYPES)?
        .to_vec()?
        .iter()
        .map(|t| t.to_i32().map(|t| u32::try_from(t).unwrap_or(0)))
        .collect::<inference_tensor::Result<_>>()?;
    let text_vocab = tokens
        .iter()
        .position(|t| t == FIRST_CODEC_TOKEN)
        .ok_or_else(|| anyhow::anyhow!("the talker GGUF has no codec tokens"))?;
    let id_of = |name: &str| -> anyhow::Result<u32> {
        let at = tokens
            .iter()
            .position(|t| t == name)
            .ok_or_else(|| anyhow::anyhow!("the talker GGUF has no `{name}` token"))?;
        Ok(u32::try_from(at)?)
    };
    let codec_id = |name: &str| -> anyhow::Result<u32> {
        Ok(id_of(&format!("<|{name}|>"))? - u32::try_from(text_vocab)?)
    };
    let named_codec = |prefix: &str| -> anyhow::Result<BTreeMap<String, u32>> {
        tokens[text_vocab..]
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                let name = t.strip_prefix(prefix)?.strip_suffix("|>")?;
                Some(u32::try_from(i).map(|i| (name.to_string(), i)))
            })
            .collect::<Result<_, _>>()
            .map_err(Into::into)
    };
    // a vocabulary of bytes alone has no merges, and llama.cpp then writes none
    let no_merges = Vec::new();
    let merges = match archive.metadata_value(MERGES) {
        Some(merges) => merges.to_vec()?,
        None => &no_merges,
    }
    .iter()
    .map(|m| {
        let m = m.to_string()?;
        m.split_once(' ')
            .map(|(l, r)| (l.to_string(), r.to_string()))
            .ok_or_else(|| inference_tensor::Error::Msg(format!("bad merge `{m}`")))
    })
    .collect::<inference_tensor::Result<_>>()?;
    let vocab = tokens[..text_vocab]
        .iter()
        .enumerate()
        .map(|(i, t)| Ok((t.clone(), u32::try_from(i)?)))
        .collect::<anyhow::Result<_>>()?;
    let added = tokens[..text_vocab]
        .iter()
        .zip(&types)
        .enumerate()
        .filter(|(_, (_, ty))| **ty == CONTROL || **ty == USER_DEFINED)
        .map(|(i, (t, ty))| Ok((u32::try_from(i)?, t.clone(), *ty == CONTROL)))
        .collect::<anyhow::Result<_>>()?;
    let tokenizer = super::tokenizer::bpe(vocab, merges, added)?;

    let hidden = usize_of(&archive, &a("embedding_length"))?;
    let layers = usize_of(&archive, &a("block_count"))?;
    let theta = f64_of(&archive, &a("rope.freq_base"))?;
    let codec_vocab = shape(&archive, "output.weight")?[0];
    let code_heads = shape(&archive, "a.gen.code.head.weight")?;
    let predictor = CodePredictorConfig {
        hidden_size: usize_of(&archive, "clip.gen.audio.embedding_length")?,
        intermediate_size: usize_of(&archive, "clip.gen.audio.feed_forward_length")?,
        num_hidden_layers: usize_of(&archive, "clip.gen.audio.block_count")?,
        num_attention_heads: usize_of(&archive, "clip.gen.audio.attention.head_count")?,
        num_key_value_heads: usize_of(&archive, "clip.gen.audio.attention.head_count_kv")?,
        head_dim: shape(&archive, "a.gen.code.blk.0.attn_q_norm.weight")?[0],
        hidden_act: Activation::Silu,
        rms_norm_eps: f64_of(&archive, "clip.gen.audio.attention.layer_norm_epsilon")?,
        rope_theta: theta,
        max_position_embeddings: PREDICTOR_POSITIONS,
        vocab_size: code_heads[1],
    };
    let speakers = named_codec(SPEAKER_TOKEN)?;
    let talker_config = TalkerConfig {
        hidden_size: hidden,
        intermediate_size: usize_of(&archive, &a("feed_forward_length"))?,
        num_hidden_layers: layers,
        num_attention_heads: usize_of(&archive, &a("attention.head_count"))?,
        num_key_value_heads: usize_of(&archive, &a("attention.head_count_kv"))?,
        head_dim: usize_of(&archive, &a("attention.key_length"))?,
        hidden_act: Activation::Silu,
        rms_norm_eps: f64_of(&archive, &a("attention.layer_norm_rms_epsilon"))?,
        rope_theta: theta,
        max_position_embeddings: usize_of(&archive, &a("context_length"))?,
        vocab_size: codec_vocab,
        text_hidden_size: hidden,
        text_vocab_size: text_vocab,
        num_code_groups: code_heads[0] + 1,
        codec_bos_id: codec_id("codec_bos")?,
        codec_eos_token_id: codec_id("codec_eos_token")?,
        codec_pad_id: codec_id("codec_pad")?,
        codec_think_id: codec_id("codec_think")?,
        codec_nothink_id: codec_id("codec_nothink")?,
        codec_think_bos_id: codec_id("codec_think_bos")?,
        codec_think_eos_id: codec_id("codec_think_eos")?,
        codec_language_id: named_codec(LANGUAGE_TOKEN)?,
        // llama.cpp keeps the voices' ids but not which speak a dialect
        spk_is_dialect: BTreeMap::new(),
        spk_id: speakers.clone(),
        code_predictor_config: predictor,
    };
    let has_speaker_encoder = archive.contains_tensor("mm.a.fc.weight");
    let config = Qwen3TtsConfig {
        model_type: "qwen3_tts".to_string(),
        tokenizer_type: "qwen3_tts_tokenizer_12hz".to_string(),
        tts_model_type: if !speakers.is_empty() {
            "custom_voice"
        } else if has_speaker_encoder {
            "base"
        } else {
            "voice_design"
        }
        .to_string(),
        tts_model_size: if hidden == SMALL_HIDDEN { "0b6" } else { "1b7" }.to_string(),
        tts_bos_token_id: id_of(TTS_BOS)?,
        tts_eos_token_id: id_of(TTS_EOS)?,
        tts_pad_token_id: id_of(TTS_PAD)?,
        talker_config,
        speaker_encoder_config: has_speaker_encoder
            .then(|| speaker_config(&archive))
            .transpose()?,
    };

    let w = "a.gen.wav.";
    let rest = shape(&archive, &format!("{w}quant.rest.codebook.weight"))?;
    let dac_blocks = count(&archive, &format!("{w}dac.blk."), ".conv.weight");
    let up_blocks = count(&archive, &format!("{w}up.blk."), ".conv.weight");
    let decoder = CodecDecoderConfig {
        codebook_size: rest[1],
        codebook_dim: 2 * rest[2],
        latent_dim: shape(&archive, &format!("{w}pre_conv.weight"))?[0],
        decoder_dim: shape(&archive, &format!("{w}dac.entry.weight"))?[0],
        hidden_size: shape(&archive, &format!("{w}tfm.in_proj.weight"))?[0],
        intermediate_size: shape(&archive, &format!("{w}tfm.blk.0.ffn_gate.weight"))?[0],
        num_hidden_layers: count(&archive, &format!("{w}tfm.blk."), ".ln1.weight"),
        num_attention_heads: shape(&archive, &format!("{w}tfm.blk.0.attn_q.weight"))?[0]
            / CODEC_HEAD_DIM,
        num_key_value_heads: shape(&archive, &format!("{w}tfm.blk.0.attn_k.weight"))?[0]
            / CODEC_HEAD_DIM,
        head_dim: CODEC_HEAD_DIM,
        hidden_act: Activation::Silu,
        rms_norm_eps: CODEC_EPS,
        rope_theta: CODEC_ROPE_THETA,
        max_position_embeddings: CODEC_POSITIONS,
        sliding_window: CODEC_WINDOW,
        num_quantizers: rest[0] + 1,
        num_semantic_quantizers: 1,
        // each block's transposed conv spans twice its stride
        upsample_rates: (0..dac_blocks)
            .map(|i| Ok(shape(&archive, &format!("{w}dac.blk.{i}.conv.weight"))?[2] / 2))
            .collect::<anyhow::Result<_>>()?,
        upsampling_ratios: (0..up_blocks)
            .map(|i| Ok(shape(&archive, &format!("{w}up.blk.{i}.conv.weight"))?[2]))
            .collect::<anyhow::Result<_>>()?,
    };

    let mut b = Bindings {
        archive: &archive,
        map: GgufBindingMap::new(),
    };
    // the talker: the token table splits into the folded text rows and the codec rows
    let embd = || GgufTensorBinding::tensor("token_embd.weight");
    b.bind(
        "talker.model.text_embedding.weight",
        embd().slice(0, 0, text_vocab),
    );
    b.bind(
        "talker.model.codec_embedding.weight",
        embd().slice(0, text_vocab, codec_vocab),
    );
    b.direct("talker.codec_head.weight", "output.weight");
    b.direct("talker.model.norm.weight", "output_norm.weight");
    for i in 0..layers {
        for (gguf, native) in LAYER_NAMES {
            b.direct(
                format!("talker.model.layers.{i}.{native}.weight"),
                &format!("blk.{i}.{gguf}.weight"),
            );
        }
    }
    // the code predictor: its tables and heads are stacked, one slice per codebook
    let cp = "talker.code_predictor.";
    let predictor = &config.talker_config.code_predictor_config;
    for i in 0..predictor.num_hidden_layers {
        for (gguf, native) in GEN_LAYER_NAMES {
            b.direct(
                format!("{cp}model.layers.{i}.{native}.weight"),
                &format!("a.gen.code.blk.{i}.{gguf}.weight"),
            );
        }
    }
    b.direct(
        format!("{cp}model.norm.weight"),
        "a.gen.code.output_norm.weight",
    );
    b.module(
        &format!("{cp}small_to_mtp_projection"),
        "a.gen.code.proj_in",
    );
    let embd = shape(&archive, "a.gen.code.embd.weight")?;
    for i in 0..code_heads[0] {
        b.bind(
            format!("{cp}model.codec_embedding.{i}.weight"),
            GgufTensorBinding::tensor("a.gen.code.embd.weight")
                .slice(0, i, 1)
                .reshape(vec![embd[1], embd[2]]),
        );
        b.bind(
            format!("{cp}lm_head.{i}.weight"),
            GgufTensorBinding::tensor("a.gen.code.head.weight")
                .slice(0, i, 1)
                .reshape(vec![code_heads[1], code_heads[2]]),
        );
    }
    // the speaker encoder
    if has_speaker_encoder {
        let se = "speaker_encoder.";
        b.module(&format!("{se}blocks.0.conv"), "a.conv1d.0");
        // the SE-Res2Net blocks follow the first conv, numbered from 1
        for i in
            (1..).take_while(|i| archive.contains_tensor(&format!("a.blk.{i}.conv_pw1.weight")))
        {
            let blk = format!("{se}blocks.{i}.");
            b.module(&format!("{blk}tdnn1.conv"), &format!("a.blk.{i}.conv_pw1"));
            b.module(&format!("{blk}tdnn2.conv"), &format!("a.blk.{i}.conv_pw2"));
            b.module(
                &format!("{blk}se_block.conv1"),
                &format!("a.blk.{i}.se_conv1"),
            );
            b.module(
                &format!("{blk}se_block.conv2"),
                &format!("a.blk.{i}.se_conv2"),
            );
            for j in 0..count(&archive, &format!("a.blk.{i}.res2."), ".weight") {
                b.module(
                    &format!("{blk}res2net_block.blocks.{j}.conv"),
                    &format!("a.blk.{i}.res2.{j}"),
                );
            }
        }
        b.module(&format!("{se}mfa.conv"), "a.conv_out");
        b.module(&format!("{se}asp.tdnn.conv"), "a.asp_tdnn");
        b.module(&format!("{se}asp.conv"), "a.asp_attn");
        b.module(&format!("{se}fc"), "mm.a.fc");
    }
    let model_map = std::mem::replace(&mut b.map, GgufBindingMap::new());

    // the codec decoder
    let d = "decoder.";
    let dim = decoder.codebook_dim / 2;
    let q = format!("{d}quantizer.");
    b.bind(
        format!("{q}rvq_first.output_proj.weight"),
        GgufTensorBinding::tensor(format!("{w}quant.first.out_proj.weight")).reshape(vec![
            decoder.codebook_dim,
            dim,
            1,
        ]),
    );
    b.bind(
        format!("{q}rvq_rest.output_proj.weight"),
        GgufTensorBinding::tensor(format!("{w}quant.rest.out_proj.weight")).reshape(vec![
            decoder.codebook_dim,
            dim,
            1,
        ]),
    );
    b.codebook(
        &format!("{q}rvq_first.vq.layers.0._codebook"),
        GgufTensorBinding::tensor(format!("{w}quant.first.codebook.weight")),
        decoder.codebook_size,
        dim,
    );
    for i in 0..rest[0] {
        b.codebook(
            &format!("{q}rvq_rest.vq.layers.{i}._codebook"),
            GgufTensorBinding::tensor(format!("{w}quant.rest.codebook.weight")).slice(0, i, 1),
            decoder.codebook_size,
            dim,
        );
    }
    b.module(&format!("{d}pre_conv.conv"), &format!("{w}pre_conv"));
    let t = format!("{d}pre_transformer.");
    b.module(&format!("{t}input_proj"), &format!("{w}tfm.in_proj"));
    b.module(&format!("{t}output_proj"), &format!("{w}tfm.out_proj"));
    b.direct(
        format!("{t}norm.weight"),
        &format!("{w}tfm.output_norm.weight"),
    );
    for i in 0..decoder.num_hidden_layers {
        let blk = format!("{w}tfm.blk.{i}.");
        for (gguf, native) in GEN_LAYER_NAMES {
            b.direct(
                format!("{t}layers.{i}.{native}.weight"),
                &format!("{blk}{gguf}.weight"),
            );
        }
        b.direct(
            format!("{t}layers.{i}.self_attn_layer_scale.scale"),
            &format!("{blk}ls1.weight"),
        );
        b.direct(
            format!("{t}layers.{i}.mlp_layer_scale.scale"),
            &format!("{blk}ls2.weight"),
        );
    }
    for i in 0..up_blocks {
        let (n, g) = (format!("{d}upsample.{i}."), format!("{w}up.blk.{i}."));
        b.module(&format!("{n}0.conv"), &format!("{g}conv"));
        b.module(&format!("{n}1.dwconv.conv"), &format!("{g}dwconv"));
        b.module(&format!("{n}1.norm"), &format!("{g}norm"));
        b.module(&format!("{n}1.pwconv1"), &format!("{g}pw1"));
        b.module(&format!("{n}1.pwconv2"), &format!("{g}pw2"));
        b.direct(format!("{n}1.gamma"), &format!("{g}gamma"));
    }
    b.module(&format!("{d}decoder.0.conv"), &format!("{w}dac.entry"));
    for i in 0..dac_blocks {
        let (n, g) = (
            format!("{d}decoder.{}.block.", i + 1),
            format!("{w}dac.blk.{i}."),
        );
        b.snake(&format!("{n}0"), &format!("{g}snake"));
        b.module(&format!("{n}1.conv"), &format!("{g}conv"));
        for x in 0..3 {
            let (unit, res) = (format!("{n}{}.", x + 2), format!("{g}res.{x}."));
            b.snake(&format!("{unit}act1"), &format!("{res}act1"));
            b.module(&format!("{unit}conv1.conv"), &format!("{res}conv1"));
            b.snake(&format!("{unit}act2"), &format!("{res}act2"));
            b.module(&format!("{unit}conv2.conv"), &format!("{res}conv2"));
        }
    }
    b.snake(
        &format!("{d}decoder.{}", dac_blocks + 1),
        &format!("{w}dac.post_snake"),
    );
    b.module(
        &format!("{d}decoder.{}.conv", dac_blocks + 2),
        &format!("{w}dac.post_conv"),
    );
    let codec_map = b.map;

    let archive = Arc::new(archive);
    let weights = Arc::new(GgufWeightSource::new(archive.clone(), &model_map, dtype)?);
    let codec_weights = Arc::new(GgufWeightSource::new(archive, &codec_map, DType::F32)?);
    Ok(GgufCheckpoint {
        config,
        codec: CodecConfig {
            decoder_config: decoder,
            encoder_config: None,
            encoder_valid_num_quantizers: None,
            output_sample_rate: usize::try_from(SAMPLE_RATE)?,
        },
        weights: weights.sharded_var_builder(device.clone()),
        codec_weights: codec_weights.sharded_var_builder(device.clone()),
        tokenizer,
    })
}
