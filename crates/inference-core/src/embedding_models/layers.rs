#![allow(clippy::cast_precision_loss)]

use inference_gguf::embedding_gguf::EmbeddingGgufPooling;
use inference_tensor::nn::Module;
use inference_tensor::{D, IndexOp, Result, Tensor};
use serde::Deserialize;

fn include_prompt() -> bool {
    true
}

/// Pooling layer
#[derive(Deserialize, Debug, Clone)]
pub struct Pooling {
    pub word_embedding_dimension: usize,
    pub pooling_mode_cls_token: bool,
    pub pooling_mode_mean_tokens: bool,
    pub pooling_mode_max_tokens: bool,
    pub pooling_mode_mean_sqrt_len_tokens: bool,
    // older sentence-transformers configs predate these fields
    #[serde(default)]
    pub pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pub pooling_mode_lasttoken: bool,
    #[serde(default = "include_prompt")]
    pub include_prompt: bool,
}

impl Pooling {
    /// Mean pooling over every token, the module sentence-transformers adds to a model that ships none.
    pub fn mean(word_embedding_dimension: usize) -> Self {
        Self::from_gguf(EmbeddingGgufPooling::Mean, word_embedding_dimension)
    }

    /// The one pooling a GGUF's `pooling_type` names.
    pub fn from_gguf(pooling: EmbeddingGgufPooling, word_embedding_dimension: usize) -> Self {
        Self {
            word_embedding_dimension,
            pooling_mode_cls_token: pooling == EmbeddingGgufPooling::Cls,
            pooling_mode_mean_tokens: pooling == EmbeddingGgufPooling::Mean,
            pooling_mode_max_tokens: false,
            pooling_mode_mean_sqrt_len_tokens: false,
            pooling_mode_weightedmean_tokens: false,
            pooling_mode_lasttoken: pooling == EmbeddingGgufPooling::Last,
            include_prompt: true,
        }
    }
}

impl Module for Pooling {
    // https://github.com/huggingface/sentence-transformers/blob/85ec64559f4414aa536eca4bf53538291e0a333f/sentence_transformers/models/Pooling.py#L26
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if !self.include_prompt {
            inference_tensor::bail!("Only support include_prompt==true");
        }
        if xs.dim(D::Minus1)? != self.word_embedding_dimension {
            inference_tensor::bail!("xs does not match the expected embedding dimension.");
        }

        // batches hold equal-length prompts, so every token is real and the mask is all ones
        let seq_len = xs.dim(1)?;
        let mut outputs = Vec::new();
        if self.pooling_mode_cls_token {
            outputs.push(xs.i((.., 0, ..))?);
        }
        if self.pooling_mode_mean_tokens {
            outputs.push((xs.sum(1)? / seq_len as f64)?);
        }
        if self.pooling_mode_max_tokens {
            outputs.push(xs.max(1)?);
        }
        if self.pooling_mode_mean_sqrt_len_tokens {
            outputs.push((xs.sum(1)? / (seq_len as f64).sqrt())?);
        }
        if self.pooling_mode_weightedmean_tokens {
            inference_tensor::bail!("weighted-mean pooling is not supported");
        }
        if self.pooling_mode_lasttoken {
            outputs.push(xs.i((.., seq_len - 1, ..))?);
        }

        Tensor::cat(&outputs, 1)
    }
}

/// Normalize layer
#[derive(Deserialize, Debug, Clone)]
pub struct Normalize;

impl Module for Normalize {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let norm = (xs.sqr()?.sum(1)? + 1e-12)?.sqrt()?;

        xs.broadcast_div(&norm.unsqueeze(D::Minus1)?)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub enum DenseActivation {
    #[serde(alias = "torch.nn.modules.linear.Identity")]
    Identity,
}

/// Dense layer
#[derive(Deserialize, Debug, Clone)]
pub struct Dense {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub activation_function: DenseActivation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::Device;

    // one sequence of three 2-wide token states
    const STATES: [f32; 6] = [1.0, -4.0, 3.0, 2.0, 5.0, 8.0];
    const SEQ_LEN: usize = 3;
    const WIDTH: usize = 2;

    fn pooled(config: &str) -> Vec<f32> {
        let pooling: Pooling = serde_json::from_str(config).unwrap();
        let xs = Tensor::from_slice(&STATES, (1, SEQ_LEN, WIDTH), &Device::Cpu).unwrap();
        pooling
            .forward(&xs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn config(mode: &str) -> String {
        let mut config = serde_json::json!({
            "word_embedding_dimension": WIDTH,
            "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": false,
            "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false,
        });
        config[mode] = true.into();
        config.to_string()
    }

    #[test]
    fn each_pooling_mode_reduces_the_tokens_as_sentence_transformers_does() {
        assert_eq!(pooled(&config("pooling_mode_cls_token")), [1.0, -4.0]);
        assert_eq!(pooled(&config("pooling_mode_mean_tokens")), [3.0, 2.0]);
        assert_eq!(pooled(&config("pooling_mode_max_tokens")), [5.0, 8.0]);
        let sqrt_len = (SEQ_LEN as f32).sqrt();
        let sums = [9.0 / sqrt_len, 6.0 / sqrt_len];
        let got = pooled(&config("pooling_mode_mean_sqrt_len_tokens"));
        assert!(
            got.iter().zip(sums).all(|(g, s)| (g - s).abs() < 1e-6),
            "{got:?}"
        );
        assert_eq!(pooled(&config("pooling_mode_lasttoken")), [5.0, 8.0]);
    }
}
