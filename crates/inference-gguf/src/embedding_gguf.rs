//! GGUF encoders, as llama.cpp's converter writes them: the HF config their embedding loaders read, the native
//! names of their tensors, and the pooling their metadata asks for.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

const TOKEN_EMBEDDINGS: &str = "token_embd.weight";
const TOKEN_TYPE_COUNT_KEY: &str = "tokenizer.ggml.token_type_count";
// BERT's own default: a pair's two segments
const DEFAULT_TOKEN_TYPES: u32 = 2;
// llama.cpp's LLAMA_POOLING_TYPE_*
const POOLING_MEAN: u32 = 1;
const POOLING_CLS: u32 = 2;
const POOLING_LAST: u32 = 3;

/// An encoder architecture a GGUF of `general.architecture` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingGgufArchitecture {
    /// BERT, RoBERTa and XLM-R (whose position table the converter trims to start at 0).
    Bert,
    NomicBert,
}

/// How the encoder's last hidden states become one vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingGgufPooling {
    Mean,
    Cls,
    Last,
}

impl EmbeddingGgufArchitecture {
    pub fn from_architecture(architecture: &str) -> Option<Self> {
        match architecture {
            "bert" => Some(Self::Bert),
            "nomic-bert" => Some(Self::NomicBert),
            _ => None,
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::Bert => "bert",
            Self::NomicBert => "nomic-bert",
        }
    }
}

struct Metadata<'a> {
    values: &'a HashMap<String, Value>,
    prefix: &'static str,
}

impl Metadata<'_> {
    fn value(&self, key: &str) -> Result<&Value> {
        let key = format!("{}.{key}", self.prefix);
        self.values
            .get(&key)
            .with_context(|| format!("GGUF metadata is missing `{key}`"))
    }

    fn u32(&self, key: &str) -> Result<u32> {
        Ok(self.value(key)?.to_u32()?)
    }

    fn f32(&self, key: &str) -> Result<f32> {
        Ok(self.value(key)?.to_f32()?)
    }
}

/// The HF `config.json` the architecture's embedding loader reads, from the GGUF's metadata.
pub fn synthesize_embedding_config(
    archive: &GgufArchive,
    architecture: EmbeddingGgufArchitecture,
) -> Result<String> {
    let values = archive.metadata();
    let meta = Metadata {
        values,
        prefix: architecture.prefix(),
    };
    // the table's rows, which a converter may pad past the tokenizer's tokens
    let vocab_size = archive
        .tensor_info(TOKEN_EMBEDDINGS)?
        .shape()
        .first()
        .copied()
        .context("GGUF `token_embd.weight` has no rows")?;
    let token_types = values
        .get(TOKEN_TYPE_COUNT_KEY)
        .map(|v| v.to_u32())
        .transpose()?
        .unwrap_or(DEFAULT_TOKEN_TYPES);
    let config = match architecture {
        EmbeddingGgufArchitecture::Bert => json!({
            "architectures": ["BertModel"],
            "model_type": "bert",
            "vocab_size": vocab_size,
            "hidden_size": meta.u32("embedding_length")?,
            "num_hidden_layers": meta.u32("block_count")?,
            "num_attention_heads": meta.u32("attention.head_count")?,
            "intermediate_size": meta.u32("feed_forward_length")?,
            "hidden_act": "gelu",
            "max_position_embeddings": meta.u32("context_length")?,
            "type_vocab_size": token_types,
            "layer_norm_eps": meta.f32("attention.layer_norm_epsilon")?,
        }),
        EmbeddingGgufArchitecture::NomicBert => json!({
            "architectures": ["NomicBertModel"],
            "model_type": "nomic_bert",
            "vocab_size": vocab_size,
            "n_embd": meta.u32("embedding_length")?,
            "n_layer": meta.u32("block_count")?,
            "n_head": meta.u32("attention.head_count")?,
            "n_inner": meta.u32("feed_forward_length")?,
            "n_positions": meta.u32("context_length")?,
            "type_vocab_size": token_types,
            "layer_norm_epsilon": meta.f32("attention.layer_norm_epsilon")?,
            "rotary_emb_base": meta.f32("rope.freq_base")?,
            "activation_function": "swiglu",
            "qkv_proj_bias": archive.contains_tensor("blk.0.attn_qkv.bias"),
            "mlp_fc1_bias": archive.contains_tensor("blk.0.ffn_up.bias"),
            "mlp_fc2_bias": archive.contains_tensor("blk.0.ffn_down.bias"),
        }),
    };
    Ok(serde_json::to_string(&config)?)
}

/// The pooling the GGUF's `<arch>.pooling_type` asks for.
pub fn embedding_pooling(
    archive: &GgufArchive,
    architecture: EmbeddingGgufArchitecture,
) -> Result<EmbeddingGgufPooling> {
    let meta = Metadata {
        values: archive.metadata(),
        prefix: architecture.prefix(),
    };
    Ok(match meta.u32("pooling_type")? {
        POOLING_MEAN => EmbeddingGgufPooling::Mean,
        POOLING_CLS => EmbeddingGgufPooling::Cls,
        POOLING_LAST => EmbeddingGgufPooling::Last,
        other => bail!("GGUF pooling_type {other} is not supported for embeddings"),
    })
}

/// `(native name, GGUF name)` of the tensors outside the blocks.
fn root_tensors(
    architecture: EmbeddingGgufArchitecture,
) -> &'static [(&'static str, &'static str)] {
    match architecture {
        EmbeddingGgufArchitecture::Bert => &[
            ("embeddings.word_embeddings.weight", "token_embd.weight"),
            (
                "embeddings.position_embeddings.weight",
                "position_embd.weight",
            ),
            (
                "embeddings.token_type_embeddings.weight",
                "token_types.weight",
            ),
            ("embeddings.LayerNorm.weight", "token_embd_norm.weight"),
            ("embeddings.LayerNorm.bias", "token_embd_norm.bias"),
        ],
        EmbeddingGgufArchitecture::NomicBert => &[
            ("embeddings.word_embeddings.weight", "token_embd.weight"),
            (
                "embeddings.token_type_embeddings.weight",
                "token_types.weight",
            ),
            ("emb_ln.weight", "token_embd_norm.weight"),
            ("emb_ln.bias", "token_embd_norm.bias"),
        ],
    }
}

/// `(native module under the layer, GGUF role under `blk.N`)`; each binds its weight and, when present, its bias.
fn block_tensors(
    architecture: EmbeddingGgufArchitecture,
) -> &'static [(&'static str, &'static str)] {
    match architecture {
        EmbeddingGgufArchitecture::Bert => &[
            ("attention.self.query", "attn_q"),
            ("attention.self.key", "attn_k"),
            ("attention.self.value", "attn_v"),
            ("attention.output.dense", "attn_output"),
            ("attention.output.LayerNorm", "attn_output_norm"),
            ("intermediate.dense", "ffn_up"),
            ("output.dense", "ffn_down"),
            ("output.LayerNorm", "layer_output_norm"),
        ],
        EmbeddingGgufArchitecture::NomicBert => &[
            ("attn.Wqkv", "attn_qkv"),
            ("attn.out_proj", "attn_output"),
            ("norm1", "attn_output_norm"),
            // the converter maps Nomic's up (fc11) and gate (fc12) halves to ffn_up and ffn_gate
            ("mlp.fc11", "ffn_up"),
            ("mlp.fc12", "ffn_gate"),
            ("mlp.fc2", "ffn_down"),
            ("norm2", "layer_output_norm"),
        ],
    }
}

fn layers_prefix(architecture: EmbeddingGgufArchitecture) -> &'static str {
    match architecture {
        EmbeddingGgufArchitecture::Bert => "encoder.layer",
        EmbeddingGgufArchitecture::NomicBert => "encoder.layers",
    }
}

/// Native tensor names to the GGUF's own.
pub fn build_embedding_bindings(
    archive: &GgufArchive,
    architecture: EmbeddingGgufArchitecture,
) -> Result<GgufBindingMap> {
    let meta = Metadata {
        values: archive.metadata(),
        prefix: architecture.prefix(),
    };
    let blocks = meta.u32("block_count")?;
    let mut bindings = GgufBindingMap::new();
    let mut bind = |native: String, gguf: String| {
        if archive.contains_tensor(&gguf) {
            bindings.insert(&native, GgufTensorBinding::tensor(gguf));
        }
    };
    for (native, gguf) in root_tensors(architecture) {
        bind(native.to_string(), gguf.to_string());
    }
    let layers = layers_prefix(architecture);
    for block in 0..blocks {
        for (native, role) in block_tensors(architecture) {
            for part in ["weight", "bias"] {
                bind(
                    format!("{layers}.{block}.{native}.{part}"),
                    format!("blk.{block}.{role}.{part}"),
                );
            }
        }
    }
    Ok(bindings)
}
