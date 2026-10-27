//! GGUF encoders, as llama.cpp's converter writes them: the HF config their embedding loaders read, the native
//! names of their tensors, and the pooling their metadata asks for.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::{Value as Json, json};
use tokenizers::Tokenizer;

const TOKEN_EMBEDDINGS: &str = "token_embd.weight";
const TOKEN_TYPES: &str = "token_types.weight";
const TOKEN_TYPE_COUNT_KEY: &str = "tokenizer.ggml.token_type_count";
const TOKENIZER_MODEL_KEY: &str = "tokenizer.ggml.model";
const BOS_KEY: &str = "tokenizer.ggml.bos_token_id";
const EOS_KEY: &str = "tokenizer.ggml.eos_token_id";
const ADD_BOS_KEY: &str = "tokenizer.ggml.add_bos_token";
const ADD_EOS_KEY: &str = "tokenizer.ggml.add_eos_token";
// the BPE vocabulary, whose conversion leaves an encoder's [CLS] / [SEP] to the caller
const BPE_TOKENIZER_MODEL: &str = "gpt2";
// a reranker's head: an optional dense layer (`cls`) and the label projection (`cls.output`)
const CLASSIFIER_DENSE: &str = "cls.weight";
const CLASSIFIER_OUTPUT: &str = "cls.output.weight";
// BERT's own default: a pair's two segments
const DEFAULT_TOKEN_TYPES: u32 = 2;
// llama.cpp's LLAMA_POOLING_TYPE_*
const POOLING_MEAN: u32 = 1;
const POOLING_CLS: u32 = 2;
const POOLING_LAST: u32 = 3;
const POOLING_RANK: u32 = 4;

/// An encoder architecture a GGUF of `general.architecture` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingGgufArchitecture {
    /// BERT, RoBERTa and XLM-R (whose position table the converter trims to start at 0).
    Bert,
    NomicBert,
    NomicBertMoe,
    JinaBertV2,
    ModernBert,
    NeoBert,
}

/// How the encoder's last hidden states become one vector; `Rank` is a reranker's classifier head instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingGgufPooling {
    Mean,
    Cls,
    Last,
    Rank,
}

impl EmbeddingGgufArchitecture {
    pub fn from_architecture(architecture: &str) -> Option<Self> {
        match architecture {
            "bert" => Some(Self::Bert),
            "nomic-bert" => Some(Self::NomicBert),
            "nomic-bert-moe" => Some(Self::NomicBertMoe),
            "jina-bert-v2" => Some(Self::JinaBertV2),
            "modern-bert" => Some(Self::ModernBert),
            "neo-bert" => Some(Self::NeoBert),
            _ => None,
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::Bert => "bert",
            Self::NomicBert => "nomic-bert",
            Self::NomicBertMoe => "nomic-bert-moe",
            Self::JinaBertV2 => "jina-bert-v2",
            Self::ModernBert => "modern-bert",
            Self::NeoBert => "neo-bert",
        }
    }

    /// Where the native layers sit.
    fn layers(self) -> &'static str {
        match self {
            Self::Bert | Self::JinaBertV2 => "encoder.layer",
            Self::NomicBert | Self::NomicBertMoe => "encoder.layers",
            Self::ModernBert => "layers",
            Self::NeoBert => "transformer_encoder",
        }
    }
}

struct Metadata<'a> {
    archive: &'a GgufArchive,
    prefix: &'static str,
}

impl Metadata<'_> {
    fn value(&self, key: &str) -> Result<&Value> {
        let key = format!("{}.{key}", self.prefix);
        self.archive
            .metadata()
            .get(&key)
            .with_context(|| format!("GGUF metadata is missing `{key}`"))
    }

    fn optional(&self, key: &str) -> Option<&Value> {
        self.archive
            .metadata()
            .get(&format!("{}.{key}", self.prefix))
    }

    fn u32(&self, key: &str) -> Result<u32> {
        Ok(self.value(key)?.to_u32()?)
    }

    fn f32(&self, key: &str) -> Result<f32> {
        Ok(self.value(key)?.to_f32()?)
    }

    /// The labels of a reranker's head, from `classifier.output_labels`.
    fn labels(&self) -> Result<Vec<String>> {
        match self.optional("classifier.output_labels") {
            Some(labels) => labels
                .to_vec()?
                .iter()
                .map(|label| Ok(label.to_string()?.clone()))
                .collect(),
            None => Ok(Vec::new()),
        }
    }
}

fn metadata(archive: &GgufArchive, architecture: EmbeddingGgufArchitecture) -> Metadata<'_> {
    Metadata {
        archive,
        prefix: architecture.prefix(),
    }
}

/// A reranker: the converter wrote a `cls.output` label projection.
pub fn is_classifier(archive: &GgufArchive) -> bool {
    archive.contains_tensor(CLASSIFIER_OUTPUT)
}

/// A BERT reranker GGUF without its pooler: llama.cpp's converter drops `pooler.dense` from BERT checkpoints, and
/// `BertForSequenceClassification` scores through it, so the file no longer holds the model's scoring head.
fn check_reranker(archive: &GgufArchive) -> Result<()> {
    if is_classifier(archive) && !archive.contains_tensor(CLASSIFIER_DENSE) {
        bail!(
            "this reranker GGUF has no pooler: llama.cpp's converter drops BERT's `pooler.dense`, which \
             BertForSequenceClassification scores through, so its scores would not be the model's; load the \
             safetensors checkpoint instead"
        );
    }
    Ok(())
}

/// The HF `config.json` the architecture's embedding loader reads, from the GGUF's metadata.
pub fn synthesize_embedding_config(
    archive: &GgufArchive,
    architecture: EmbeddingGgufArchitecture,
) -> Result<String> {
    check_reranker(archive)?;
    let meta = metadata(archive, architecture);
    // the table's rows, which a converter may pad past the tokenizer's tokens
    let vocab_size = archive
        .tensor_info(TOKEN_EMBEDDINGS)?
        .shape()
        .first()
        .copied()
        .context("GGUF `token_embd.weight` has no rows")?;
    // the table's rows over the metadata count, which some converters write as the default
    let token_types = if archive.contains_tensor(TOKEN_TYPES) {
        archive
            .tensor_info(TOKEN_TYPES)?
            .shape()
            .first()
            .copied()
            .unwrap_or_default() as u32
    } else {
        archive
            .metadata()
            .get(TOKEN_TYPE_COUNT_KEY)
            .map(|v| v.to_u32())
            .transpose()?
            .unwrap_or(DEFAULT_TOKEN_TYPES)
    };
    let has = |name: &str| archive.contains_tensor(name);
    let (hidden, layers, heads, ffn, context) = (
        meta.u32("embedding_length")?,
        meta.u32("block_count")?,
        meta.u32("attention.head_count")?,
        meta.u32("feed_forward_length")?,
        meta.u32("context_length")?,
    );
    let mut config = match architecture {
        EmbeddingGgufArchitecture::Bert | EmbeddingGgufArchitecture::JinaBertV2 => {
            let mut config = json!({
                "architectures": ["BertModel"],
                "model_type": "bert",
                "vocab_size": vocab_size,
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "intermediate_size": ffn,
                "hidden_act": "gelu",
                "max_position_embeddings": context,
                "type_vocab_size": token_types,
                "layer_norm_eps": meta.f32("attention.layer_norm_epsilon")?,
            });
            if architecture == EmbeddingGgufArchitecture::JinaBertV2 {
                // the converter writes Jina v2's fused gated_layers, which are GeGLU
                config["position_embedding_type"] = "alibi".into();
                config["feed_forward_type"] = "geglu".into();
            }
            config
        }
        EmbeddingGgufArchitecture::NomicBert | EmbeddingGgufArchitecture::NomicBertMoe => {
            let mut config = json!({
                "architectures": ["NomicBertModel"],
                "model_type": "nomic_bert",
                "vocab_size": vocab_size,
                "n_embd": hidden,
                "n_layer": layers,
                "n_head": heads,
                "n_inner": ffn,
                "n_positions": context,
                "type_vocab_size": token_types,
                "layer_norm_epsilon": meta.f32("attention.layer_norm_epsilon")?,
                "rotary_emb_base": meta.f32("rope.freq_base")?,
                "activation_function": "swiglu",
                "qkv_proj_bias": has("blk.0.attn_qkv.bias"),
                "mlp_fc1_bias": has("blk.0.ffn_up.bias"),
                "mlp_fc2_bias": has("blk.0.ffn_down.bias"),
            });
            if architecture == EmbeddingGgufArchitecture::NomicBertMoe {
                config["activation_function"] = "gelu".into();
                config["moe_every_n_layers"] = meta.u32("moe_every_n_layers")?.into();
                config["num_experts"] = meta.u32("expert_count")?.into();
                config["moe_top_k"] = meta.u32("expert_used_count")?.into();
            }
            config
        }
        EmbeddingGgufArchitecture::ModernBert => json!({
            "architectures": ["ModernBertModel"],
            "model_type": "modernbert",
            "vocab_size": vocab_size,
            "hidden_size": hidden,
            "intermediate_size": ffn,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "max_position_embeddings": context,
            "norm_eps": meta.f32("attention.layer_norm_epsilon")?,
            "norm_bias": has("token_embd_norm.bias"),
            "attention_bias": has("blk.0.attn_qkv.bias"),
            "mlp_bias": has("blk.0.ffn_up.bias"),
            "hidden_activation": match meta.optional("hidden_activation") {
                Some(act) => act.to_string()?.clone(),
                None => "gelu".to_string(),
            },
            "global_attn_every_n_layers": meta.u32("attention.sliding_window_pattern")?,
            "local_attention": meta.u32("attention.sliding_window")?,
            "global_rope_theta": meta.f32("rope.freq_base")?,
            "local_rope_theta": meta.f32("rope.freq_base_swa")?,
        }),
        // the converter writes NeoBERT's SwiGLU width (2/3 of intermediate_size), which the config states as 3/2 of it
        EmbeddingGgufArchitecture::NeoBert => json!({
            "architectures": ["NeoBERT"],
            "model_type": "neobert",
            "vocab_size": vocab_size,
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "intermediate_size": ffn * 3 / 2,
            "norm_eps": meta.f32("attention.layer_norm_rms_epsilon")?,
            "max_length": context,
        }),
    };
    if is_classifier(archive) {
        let labels = meta.labels()?;
        config["architectures"] = json!(["BertForSequenceClassification"]);
        config["id2label"] = labels
            .iter()
            .enumerate()
            .map(|(i, label)| (i.to_string(), Json::from(label.clone())))
            .collect::<serde_json::Map<_, _>>()
            .into();
    }
    Ok(serde_json::to_string(&config)?)
}

/// The pooling the GGUF's `<arch>.pooling_type` asks for; a reranker without one ranks.
pub fn embedding_pooling(
    archive: &GgufArchive,
    architecture: EmbeddingGgufArchitecture,
) -> Result<EmbeddingGgufPooling> {
    let meta = metadata(archive, architecture);
    let pooling = match meta.optional("pooling_type") {
        Some(pooling) => pooling.to_u32()?,
        None if is_classifier(archive) => POOLING_RANK,
        None => bail!(
            "GGUF metadata is missing `{}.pooling_type`",
            architecture.prefix()
        ),
    };
    Ok(match pooling {
        POOLING_MEAN => EmbeddingGgufPooling::Mean,
        POOLING_CLS => EmbeddingGgufPooling::Cls,
        POOLING_LAST => EmbeddingGgufPooling::Last,
        POOLING_RANK => EmbeddingGgufPooling::Rank,
        other => bail!("GGUF pooling_type {other} is not supported for embeddings"),
    })
}

/// `(native name, GGUF name)` of the tensors outside the blocks.
fn root_tensors(
    architecture: EmbeddingGgufArchitecture,
) -> &'static [(&'static str, &'static str)] {
    use EmbeddingGgufArchitecture as A;
    match architecture {
        A::Bert | A::JinaBertV2 => &[
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
        A::NomicBert | A::NomicBertMoe => &[
            ("embeddings.word_embeddings.weight", "token_embd.weight"),
            (
                "embeddings.token_type_embeddings.weight",
                "token_types.weight",
            ),
            ("emb_ln.weight", "token_embd_norm.weight"),
            ("emb_ln.bias", "token_embd_norm.bias"),
        ],
        A::ModernBert => &[
            ("embeddings.tok_embeddings.weight", "token_embd.weight"),
            ("embeddings.norm.weight", "token_embd_norm.weight"),
            ("embeddings.norm.bias", "token_embd_norm.bias"),
            ("final_norm.weight", "output_norm.weight"),
            ("final_norm.bias", "output_norm.bias"),
        ],
        A::NeoBert => &[
            ("encoder.weight", "token_embd.weight"),
            ("layer_norm.weight", "enc.output_norm.weight"),
        ],
    }
}

/// `(native module under the layer, GGUF role under `blk.N`)`; each binds its weight and, when present, its bias.
fn block_tensors(
    architecture: EmbeddingGgufArchitecture,
) -> &'static [(&'static str, &'static str)] {
    use EmbeddingGgufArchitecture as A;
    match architecture {
        A::Bert => &[
            ("attention.self.query", "attn_q"),
            ("attention.self.key", "attn_k"),
            ("attention.self.value", "attn_v"),
            ("attention.output.dense", "attn_output"),
            ("attention.output.LayerNorm", "attn_output_norm"),
            ("intermediate.dense", "ffn_up"),
            ("output.dense", "ffn_down"),
            ("output.LayerNorm", "layer_output_norm"),
        ],
        A::JinaBertV2 => &[
            ("attention.self.query", "attn_q"),
            ("attention.self.key", "attn_k"),
            ("attention.self.value", "attn_v"),
            ("attention.output.dense", "attn_output"),
            ("attention.output.LayerNorm", "attn_output_norm"),
            ("mlp.gated_layers", "ffn_up"),
            ("mlp.wo", "ffn_down"),
            ("mlp.layernorm", "layer_output_norm"),
        ],
        A::NomicBert => &[
            ("attn.Wqkv", "attn_qkv"),
            ("attn.out_proj", "attn_output"),
            ("norm1", "attn_output_norm"),
            // the converter maps Nomic's up (fc11) and gate (fc12) halves to ffn_up and ffn_gate
            ("mlp.fc11", "ffn_up"),
            ("mlp.fc12", "ffn_gate"),
            ("mlp.fc2", "ffn_down"),
            ("norm2", "layer_output_norm"),
        ],
        // the MoE layers' expert stacks bind on their own
        A::NomicBertMoe => &[
            ("attn.Wqkv", "attn_qkv"),
            ("attn.out_proj", "attn_output"),
            ("norm1", "attn_output_norm"),
            ("mlp.fc1", "ffn_up"),
            ("mlp.fc2", "ffn_down"),
            ("norm2", "layer_output_norm"),
            ("mlp.router.layer", "ffn_gate_inp"),
        ],
        A::ModernBert => &[
            ("attn_norm", "attn_norm"),
            ("attn.Wqkv", "attn_qkv"),
            ("attn.Wo", "attn_output"),
            ("mlp_norm", "ffn_norm"),
            ("mlp.Wi", "ffn_up"),
            ("mlp.Wo", "ffn_down"),
        ],
        // the qkv rows stay in NeoBERT's per-head order: the converter copies them as they are
        A::NeoBert => &[
            ("attention_norm", "attn_norm"),
            ("qkv", "attn_qkv"),
            ("wo", "attn_output"),
            ("ffn_norm", "ffn_norm"),
            ("ffn.w12", "ffn_up"),
            ("ffn.w3", "ffn_down"),
        ],
    }
}

fn bind(archive: &GgufArchive, bindings: &mut GgufBindingMap, native: String, gguf: String) {
    if archive.contains_tensor(&gguf) {
        bindings.insert(native, GgufTensorBinding::tensor(gguf));
    }
}

/// Native tensor names to the GGUF's own.
pub fn build_embedding_bindings(
    archive: &GgufArchive,
    architecture: EmbeddingGgufArchitecture,
) -> Result<GgufBindingMap> {
    let blocks = metadata(archive, architecture).u32("block_count")?;
    let mut bindings = GgufBindingMap::new();
    for (native, gguf) in root_tensors(architecture) {
        bind(archive, &mut bindings, native.to_string(), gguf.to_string());
    }
    let layers = architecture.layers();
    for block in 0..blocks {
        for (native, role) in block_tensors(architecture) {
            for part in ["weight", "bias"] {
                bind(
                    archive,
                    &mut bindings,
                    format!("{layers}.{block}.{native}.{part}"),
                    format!("blk.{block}.{role}.{part}"),
                );
            }
        }
        if architecture == EmbeddingGgufArchitecture::NomicBertMoe {
            bind_nomic_experts(archive, block, &mut bindings)?;
        }
    }
    // a reranker's head (XLM-R's): the dense layer, then the label projection
    for (native, gguf) in [
        ("classifier.dense", "cls"),
        ("classifier.out_proj", "cls.output"),
    ] {
        for part in ["weight", "bias"] {
            bind(
                archive,
                &mut bindings,
                format!("{native}.{part}"),
                format!("{gguf}.{part}"),
            );
        }
    }
    Ok(bindings)
}

/// A MoE layer's experts as megablocks stacks them: `w1` and `w2` both `[experts * intermediate, hidden]`.
fn bind_nomic_experts(
    archive: &GgufArchive,
    block: u32,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let up = format!("blk.{block}.ffn_up_exps.weight");
    if !archive.contains_tensor(&up) {
        return Ok(());
    }
    let down = format!("blk.{block}.ffn_down_exps.weight");
    // [experts, intermediate, hidden] and [experts, hidden, intermediate]
    let shape = archive.tensor_info(&up)?.shape().to_vec();
    let [experts, intermediate, hidden] = shape[..] else {
        bail!("GGUF `{up}` is not a 3-D expert stack");
    };
    let prefix = format!("encoder.layers.{block}.mlp.experts");
    bindings.insert(
        format!("{prefix}.mlp.w1"),
        GgufTensorBinding::tensor(up).reshape(vec![experts * intermediate, hidden]),
    );
    bindings.insert(
        format!("{prefix}.mlp.w2"),
        GgufTensorBinding::tensor(down)
            .transpose(1, 2)
            .reshape(vec![experts * intermediate, hidden]),
    );
    // the converter drops the experts' shared bias, which llama.cpp does without; a hidden-wide zero stands in
    bindings.insert(
        format!("{prefix}.bias"),
        GgufTensorBinding::tensor(format!("blk.{block}.layer_output_norm.bias")).affine(0.0, 0.0),
    );
    Ok(())
}

/// Wraps a BPE-vocabulary encoder's inputs in the `[CLS]` / `[SEP]` its `add_bos` / `add_eos` flags name; the
/// WordPiece and Unigram conversions template their own.
pub fn add_encoder_specials(
    tokenizer: &mut Tokenizer,
    values: &HashMap<String, Value>,
) -> Result<()> {
    use tokenizers::processors::{PostProcessorWrapper, template::TemplateProcessing};

    let model = values
        .get(TOKENIZER_MODEL_KEY)
        .map(|v| v.to_string())
        .transpose()?;
    if model.is_none_or(|model| model != BPE_TOKENIZER_MODEL) {
        return Ok(());
    }
    let flag = |key: &str| -> Result<bool> {
        Ok(values
            .get(key)
            .map(|v| v.to_bool())
            .transpose()?
            .unwrap_or(false))
    };
    let special = |key: &str, add: bool| -> Result<Option<(String, u32)>> {
        if !add {
            return Ok(None);
        }
        let id = values
            .get(key)
            .context("GGUF metadata is missing a special token id")?
            .to_u32()?;
        let token = tokenizer
            .id_to_token(id)
            .with_context(|| format!("GGUF special token id {id} is not in the vocabulary"))?;
        Ok(Some((token, id)))
    };
    let bos = special(BOS_KEY, flag(ADD_BOS_KEY)?)?;
    let eos = special(EOS_KEY, flag(ADD_EOS_KEY)?)?;
    if bos.is_none() && eos.is_none() {
        return Ok(());
    }
    let wrap = |input: &str, type_id: u32| {
        bos.iter()
            .map(|(token, _)| token.as_str())
            .chain(std::iter::once(input))
            .chain(eos.iter().map(|(token, _)| token.as_str()))
            .map(|piece| format!("{piece}:{type_id}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let post = TemplateProcessing::builder()
        .try_single(wrap("$A", 0))
        .map_err(anyhow::Error::msg)?
        .try_pair(format!("{} {}", wrap("$A", 0), wrap("$B", 1)))
        .map_err(anyhow::Error::msg)?
        .special_tokens(bos.iter().chain(eos.iter()).cloned().collect::<Vec<_>>())
        .build()
        .map_err(anyhow::Error::msg)?;
    tokenizer.with_post_processor(Some(PostProcessorWrapper::Template(post)));
    Ok(())
}
