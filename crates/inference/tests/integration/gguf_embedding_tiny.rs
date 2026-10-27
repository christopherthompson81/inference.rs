//! Tiny encoders written as GGUFs under llama.cpp's tensor names and metadata (BERT, Nomic BERT and its MoE,
//! Jina BERT v2, ModernBERT, NeoBERT), against the safetensors checkpoints they came from.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use inference::{EmbeddingModelBuilder, GgufModelBuilder, Model, ModelDType};
use inference_tensor::quantized::{GgmlDType, QTensor, gguf_file};
use inference_tensor::{Device, Tensor};

#[path = "../support/bert_embedding_tiny.rs"]
mod bert;
#[path = "../support/rope_encoders_tiny.rs"]
mod rope;

const PROMPT: &str = "hello there, how are you";
const GGUF_FILE: &str = "tiny.gguf";
// the GGUF path L2-normalizes, as llama.cpp does, so F32 is compared by direction
const F32_MIN_COSINE: f32 = 0.99999;
// Q8_0 on 32-wide layers keeps the embedding close, not equal
const Q8_MIN_COSINE: f32 = 0.99;
// llama.cpp's LLAMA_POOLING_TYPE_MEAN / _CLS
const POOLING_MEAN: u32 = 1;
const POOLING_CLS: u32 = 2;
// From the committed tiny configs
const HIDDEN: u32 = 32;
const HEADS: u32 = 2;
const EPS: f32 = 1e-12;
const NORM_EPS: f32 = 1e-5;
const NEO_EPS: f32 = 1e-6;
const NOMIC_EXPERTS: usize = 4;

#[derive(Clone, Copy, PartialEq)]
enum Arch {
    Bert,
    JinaBertV2,
    NomicBert,
    NomicBertMoe,
    ModernBert,
    NeoBert,
}

impl Arch {
    fn name(self) -> &'static str {
        match self {
            Self::Bert => "bert",
            Self::JinaBertV2 => "jina-bert-v2",
            Self::NomicBert => "nomic-bert",
            Self::NomicBertMoe => "nomic-bert-moe",
            Self::ModernBert => "modern-bert",
            Self::NeoBert => "neo-bert",
        }
    }

    fn layers(self) -> &'static str {
        match self {
            Self::Bert | Self::JinaBertV2 => "encoder.layer.",
            Self::NomicBert | Self::NomicBertMoe => "encoder.layers.",
            Self::ModernBert => "layers.",
            Self::NeoBert => "transformer_encoder.",
        }
    }

    /// The GGUF name llama.cpp's converter gives a native tensor; `None` for what it leaves out.
    fn gguf_name(self, native: &str) -> Option<String> {
        let root = match native {
            "embeddings.word_embeddings.weight"
            | "embeddings.tok_embeddings.weight"
            | "encoder.weight" => Some("token_embd.weight"),
            "embeddings.position_embeddings.weight" => Some("position_embd.weight"),
            "embeddings.token_type_embeddings.weight" => Some("token_types.weight"),
            "embeddings.LayerNorm.weight" | "emb_ln.weight" | "embeddings.norm.weight" => {
                Some("token_embd_norm.weight")
            }
            "embeddings.LayerNorm.bias" | "emb_ln.bias" => Some("token_embd_norm.bias"),
            "final_norm.weight" => Some("output_norm.weight"),
            "layer_norm.weight" => Some("enc.output_norm.weight"),
            _ => None,
        };
        if let Some(root) = root {
            return Some(root.to_string());
        }
        let (block, rest) = native.strip_prefix(self.layers())?.split_once('.')?;
        let (module, part) = rest.rsplit_once('.')?;
        let role = match (self, module) {
            (_, "attention.self.query") => "attn_q",
            (_, "attention.self.key") => "attn_k",
            (_, "attention.self.value") => "attn_v",
            (_, "attention.output.dense" | "attn.out_proj" | "attn.Wo" | "wo") => "attn_output",
            (_, "attention.output.LayerNorm" | "norm1") => "attn_output_norm",
            (_, "intermediate.dense" | "mlp.fc11" | "mlp.fc1" | "mlp.gated_layers") => "ffn_up",
            (_, "mlp.Wi" | "ffn.w12") => "ffn_up",
            (_, "mlp.fc12") => "ffn_gate",
            (_, "output.dense" | "mlp.fc2" | "mlp.wo" | "mlp.Wo" | "ffn.w3") => "ffn_down",
            (_, "output.LayerNorm" | "norm2" | "mlp.layernorm") => "layer_output_norm",
            (_, "attn.Wqkv" | "qkv") => "attn_qkv",
            (_, "attn_norm" | "attention_norm") => "attn_norm",
            (_, "mlp_norm" | "ffn_norm") => "ffn_norm",
            (_, "mlp.router.layer") => "ffn_gate_inp",
            _ => return None,
        };
        Some(format!("blk.{block}.{role}.{part}"))
    }

    fn metadata(self, cfg: &serde_json::Value) -> Vec<(String, gguf_file::Value)> {
        use gguf_file::Value::{F32, U32};
        let key = |k: &str| format!("{}.{k}", self.name());
        let num = |k: &str| U32(cfg[k].as_u64().unwrap_or_default() as u32);
        let mut metadata = vec![
            (key("embedding_length"), U32(HIDDEN)),
            (key("attention.head_count"), U32(HEADS)),
        ];
        match self {
            Self::Bert | Self::JinaBertV2 => metadata.extend([
                (key("block_count"), num("num_hidden_layers")),
                (key("feed_forward_length"), num("intermediate_size")),
                (key("context_length"), num("max_position_embeddings")),
                (key("attention.layer_norm_epsilon"), F32(EPS)),
                (key("pooling_type"), U32(POOLING_CLS)),
            ]),
            Self::NomicBert | Self::NomicBertMoe => {
                metadata.extend([
                    (key("block_count"), num("n_layer")),
                    (key("feed_forward_length"), num("n_inner")),
                    (key("context_length"), num("n_positions")),
                    (key("attention.layer_norm_epsilon"), F32(EPS)),
                    (
                        key("rope.freq_base"),
                        F32(cfg["rotary_emb_base"].as_f64().unwrap_or_default() as f32),
                    ),
                    (key("pooling_type"), U32(POOLING_MEAN)),
                ]);
                if self == Self::NomicBertMoe {
                    metadata.extend([
                        (key("expert_count"), num("num_experts")),
                        (key("expert_used_count"), num("moe_top_k")),
                        (key("moe_every_n_layers"), num("moe_every_n_layers")),
                    ]);
                }
            }
            Self::ModernBert => metadata.extend([
                (key("block_count"), num("num_hidden_layers")),
                (key("feed_forward_length"), num("intermediate_size")),
                (key("context_length"), num("max_position_embeddings")),
                (key("attention.layer_norm_epsilon"), F32(NORM_EPS)),
                (
                    key("rope.freq_base"),
                    F32(cfg["global_rope_theta"].as_f64().unwrap_or_default() as f32),
                ),
                (
                    key("rope.freq_base_swa"),
                    F32(cfg["local_rope_theta"].as_f64().unwrap_or_default() as f32),
                ),
                (key("attention.sliding_window"), num("local_attention")),
                (
                    key("attention.sliding_window_pattern"),
                    num("global_attn_every_n_layers"),
                ),
                (key("pooling_type"), U32(POOLING_MEAN)),
            ]),
            Self::NeoBert => {
                // the converter writes the SwiGLU width, 2/3 of intermediate_size
                let ffn = cfg["intermediate_size"].as_u64().unwrap_or_default() * 2 / 3;
                metadata.extend([
                    (key("block_count"), num("num_hidden_layers")),
                    (key("feed_forward_length"), U32(ffn as u32)),
                    (key("context_length"), num("max_length")),
                    (key("attention.layer_norm_rms_epsilon"), F32(NEO_EPS)),
                    (key("rope.freq_base"), F32(10_000.0)),
                    (key("pooling_type"), U32(POOLING_MEAN)),
                ]);
            }
        }
        metadata
    }
}

fn tokens_by_id(tokenizer_json: &Path) -> anyhow::Result<Vec<gguf_file::Value>> {
    let tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tokenizer_json)?)?;
    let mut by_id = BTreeMap::new();
    for (token, id) in tokenizer["model"]["vocab"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no model.vocab"))?
    {
        by_id.insert(id.as_u64().unwrap_or_default(), token.clone());
    }
    for added in tokenizer["added_tokens"].as_array().into_iter().flatten() {
        by_id.insert(
            added["id"].as_u64().unwrap_or_default(),
            added["content"].as_str().unwrap_or_default().to_string(),
        );
    }
    Ok(by_id.into_values().map(gguf_file::Value::String).collect())
}

/// llama.cpp's expert stacks from megablocks' `w1` / `w2` (`[experts * intermediate, hidden]`).
fn nomic_experts(weights: &HashMap<String, Tensor>) -> anyhow::Result<Vec<(String, Tensor)>> {
    let mut stacks = Vec::new();
    for (name, w) in weights {
        let Some(prefix) = name.strip_suffix(".mlp.experts.mlp.w1") else {
            continue;
        };
        let block = prefix
            .strip_prefix("encoder.layers.")
            .ok_or_else(|| anyhow::anyhow!("unexpected expert tensor `{name}`"))?;
        let (rows, hidden) = w.dims2()?;
        let stack = (NOMIC_EXPERTS, rows / NOMIC_EXPERTS, hidden);
        stacks.push((format!("blk.{block}.ffn_up_exps.weight"), w.reshape(stack)?));
        let w2 = &weights[&format!("{prefix}.mlp.experts.mlp.w2")];
        stacks.push((
            format!("blk.{block}.ffn_down_exps.weight"),
            w2.reshape(stack)?.transpose(1, 2)?.contiguous()?,
        ));
    }
    Ok(stacks)
}

/// The checkpoint's weights as a GGUF, matrices in `matrix_dtype`, with the metadata llama.cpp writes.
fn write_gguf(
    checkpoint: &Path,
    arch: Arch,
    matrix_dtype: GgmlDType,
    out: &Path,
) -> anyhow::Result<()> {
    let weights =
        inference_tensor::safetensors::load(checkpoint.join("model.safetensors"), &Device::Cpu)?;
    let mut tensors = Vec::new();
    let mut named = weights
        .iter()
        .filter_map(|(name, t)| arch.gguf_name(name).map(|gguf| (gguf, t.clone())))
        .collect::<Vec<_>>();
    if arch == Arch::NomicBertMoe {
        named.extend(nomic_experts(&weights)?);
    }
    for (gguf, tensor) in named {
        let quantizable = tensor.rank() >= 2
            && !gguf.starts_with("token_types")
            && !gguf.contains("gate_inp")
            && tensor.dim(inference_tensor::D::Minus1)? % 32 == 0;
        let dtype = if quantizable {
            matrix_dtype
        } else {
            GgmlDType::F32
        };
        tensors.push((gguf, QTensor::quantize(&tensor, dtype)?));
    }
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let mut metadata = arch.metadata(&cfg);
    metadata.push((
        "general.architecture".to_string(),
        gguf_file::Value::String(arch.name().into()),
    ));
    metadata.push((
        "tokenizer.ggml.tokens".to_string(),
        gguf_file::Value::Array(tokens_by_id(&checkpoint.join("tokenizer.json"))?),
    ));
    let metadata = metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let tensors = tensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    gguf_file::write(&mut std::fs::File::create(out)?, &metadata, &tensors)?;
    Ok(())
}

async fn safetensors_embedding(checkpoint: &Path) -> anyhow::Result<Vec<f32>> {
    let model = EmbeddingModelBuilder::new(checkpoint.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    Ok(model.generate_embedding(PROMPT).await?)
}

async fn gguf_model(checkpoint: &Path, dir: &Path) -> anyhow::Result<Model> {
    // the tokenizer comes from the checkpoint; the GGUF vocabularies have their own conversion tests
    Ok(
        GgufModelBuilder::new(dir.to_string_lossy(), vec![GGUF_FILE])
            .with_tok_model_id(checkpoint.to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?,
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

async fn assert_gguf_embeds_as_its_source(checkpoint: &Path, arch: Arch) -> anyhow::Result<()> {
    let expected = safetensors_embedding(checkpoint).await?;
    for (dtype, min_cosine) in [
        (GgmlDType::F32, F32_MIN_COSINE),
        (GgmlDType::Q8_0, Q8_MIN_COSINE),
    ] {
        let dir = tempfile::tempdir()?;
        write_gguf(checkpoint, arch, dtype, &dir.path().join(GGUF_FILE))?;
        let embedding = gguf_model(checkpoint, dir.path())
            .await?
            .generate_embedding(PROMPT)
            .await?;
        let cos = cosine(&expected, &embedding);
        anyhow::ensure!(
            cos > min_cosine,
            "{} {dtype:?} GGUF cosine {cos}",
            arch.name()
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_bert_gguf_embeds_as_its_safetensors_source() -> anyhow::Result<()> {
    assert_gguf_embeds_as_its_source(bert::tiny_bert_checkpoint(|_| {})?.path(), Arch::Bert).await
}

#[tokio::test]
async fn a_jina_bert_v2_gguf_embeds_as_its_safetensors_source() -> anyhow::Result<()> {
    let checkpoint = bert::tiny_bert_checkpoint(|config| {
        config["architectures"] = serde_json::json!(["JinaBertModel"]);
        config["position_embedding_type"] = "alibi".into();
        config["feed_forward_type"] = "geglu".into();
    })?;
    assert_gguf_embeds_as_its_source(checkpoint.path(), Arch::JinaBertV2).await
}

#[tokio::test]
async fn a_nomic_bert_gguf_embeds_as_its_safetensors_source() -> anyhow::Result<()> {
    let checkpoint = rope::tiny_nomic_bert_checkpoint(|_| {})?;
    assert_gguf_embeds_as_its_source(checkpoint.path(), Arch::NomicBert).await
}

// llama.cpp's converter drops the experts' shared bias; zeroed in the source, the two match
#[tokio::test]
async fn a_nomic_bert_moe_gguf_embeds_as_its_source_without_the_expert_bias() -> anyhow::Result<()>
{
    let checkpoint = rope::tiny_nomic_bert_checkpoint(rope::nomic_moe_config)?;
    let path = checkpoint.path().join("model.safetensors");
    let mut weights = inference_tensor::safetensors::load(&path, &Device::Cpu)?;
    for (name, tensor) in weights.iter_mut() {
        if name.ends_with("mlp.experts.bias") {
            *tensor = tensor.zeros_like()?;
        }
    }
    inference_tensor::safetensors::save(&weights, &path)?;
    assert_gguf_embeds_as_its_source(checkpoint.path(), Arch::NomicBertMoe).await
}

#[tokio::test]
async fn a_modern_bert_gguf_embeds_as_its_safetensors_source() -> anyhow::Result<()> {
    let checkpoint = rope::tiny_modern_bert_checkpoint(|_| {})?;
    assert_gguf_embeds_as_its_source(checkpoint.path(), Arch::ModernBert).await
}

#[tokio::test]
async fn a_neo_bert_gguf_embeds_as_its_safetensors_source() -> anyhow::Result<()> {
    let checkpoint = rope::tiny_neo_bert_checkpoint()?;
    assert_gguf_embeds_as_its_source(checkpoint.path(), Arch::NeoBert).await
}

// the converter keeps a BERT classifier's label projection but drops the pooler it scores through
#[tokio::test]
async fn a_bert_reranker_gguf_without_its_pooler_is_refused() -> anyhow::Result<()> {
    let checkpoint = bert::tiny_bert_checkpoint(|config| {
        config["architectures"] = serde_json::json!(["BertForSequenceClassification"]);
        config["id2label"] = serde_json::json!({"0": "LABEL_0"});
    })?;
    let dir = tempfile::tempdir()?;
    write_gguf(
        checkpoint.path(),
        Arch::Bert,
        GgmlDType::F32,
        &dir.path().join(GGUF_FILE),
    )?;
    // add the label projection llama.cpp writes as `cls.output`
    let path = dir.path().join(GGUF_FILE);
    let mut file = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut tensors = content
        .tensor_infos
        .keys()
        .map(|name| Ok((name.clone(), content.tensor(&mut file, name, &Device::Cpu)?)))
        .collect::<anyhow::Result<Vec<_>>>()?;
    tensors.push((
        "cls.output.weight".to_string(),
        QTensor::quantize(
            &Tensor::zeros(
                (1, HIDDEN as usize),
                inference_tensor::DType::F32,
                &Device::Cpu,
            )?,
            GgmlDType::F32,
        )?,
    ));
    let metadata = content
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let tensors = tensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    gguf_file::write(&mut std::fs::File::create(&path)?, &metadata, &tensors)?;

    let Err(error) = gguf_model(checkpoint.path(), dir.path()).await else {
        anyhow::bail!("a reranker GGUF without a pooler loaded");
    };
    anyhow::ensure!(error.to_string().contains("no pooler"), "{error}");
    Ok(())
}
