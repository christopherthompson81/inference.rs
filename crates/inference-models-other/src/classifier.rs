//! The sequence-classification head cross-encoders put over an encoder: pool, a dense layer and its activation, an
//! optional LayerNorm, then the label projection. Each encoder names its own weights.

use std::collections::HashMap;
use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{LayerNorm, LayerNormConfig, Module};
use inference_tensor::{IndexOp, Result, Tensor};

use crate::{
    layers::{Activation, layer_norm},
    utils::unvarbuilder::UnVarBuilder,
};

const CLASSIFIER_SUFFIX: &str = "ForSequenceClassification";
// transformers' PretrainedConfig default when a config lists no labels
const DEFAULT_NUM_LABELS: usize = 2;

/// A cross-encoder: a classification head over the encoder, scoring rather than embedding.
pub fn is_classifier(architectures: &[String]) -> bool {
    architectures
        .iter()
        .any(|arch| arch.ends_with(CLASSIFIER_SUFFIX))
}

pub fn num_labels(id2label: Option<&HashMap<String, String>>) -> usize {
    id2label.map_or(DEFAULT_NUM_LABELS, HashMap::len)
}

/// A head's weights: the hidden-to-hidden layer with its bias or norm, and the biased label projection.
pub fn head_elems(hidden: usize, labels: usize) -> usize {
    hidden * hidden + hidden + hidden * labels + labels
}

/// What the head reads from the encoder's last hidden states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadPooling {
    First,
    // batches hold equal-length prompts, so every token counts
    Mean,
}

#[derive(Clone, Copy, Debug)]
pub enum HeadActivation {
    Tanh,
    Act(Activation),
}

/// A head's shape and where its weights sit, as paths from the checkpoint's root.
pub struct HeadSpec<'a> {
    pub hidden: usize,
    pub labels: usize,
    pub pooling: HeadPooling,
    pub dense: &'a str,
    pub dense_bias: bool,
    pub activation: HeadActivation,
    pub norm: Option<(&'a str, LayerNormConfig)>,
    pub out: &'a str,
}

pub struct ClassifierHead {
    pooling: HeadPooling,
    dense: Arc<dyn QuantMethod>,
    activation: HeadActivation,
    norm: Option<LayerNorm>,
    out: Arc<dyn QuantMethod>,
    // the paths again, for the UQFF residual
    names: (String, Option<String>, String),
}

impl ClassifierHead {
    pub fn new(spec: &HeadSpec<'_>, root: &ShardedVarBuilder) -> Result<Self> {
        let HeadSpec { hidden, labels, .. } = *spec;
        Ok(Self {
            pooling: spec.pooling,
            dense: inference_quant::linear_b(
                hidden,
                hidden,
                spec.dense_bias,
                &None,
                root.pp(spec.dense),
            )?,
            activation: spec.activation,
            norm: spec
                .norm
                .map(|(name, config)| layer_norm(hidden, config, root.pp(name)))
                .transpose()?,
            out: inference_quant::linear(hidden, labels, &None, root.pp(spec.out))?,
            names: (
                spec.dense.to_string(),
                spec.norm.map(|(name, _)| name.to_string()),
                spec.out.to_string(),
            ),
        })
    }

    /// `[batch, labels]` logits from `[batch, seq, hidden]` states.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let pooled = match self.pooling {
            HeadPooling::First => xs.i((.., 0, ..))?.contiguous()?,
            HeadPooling::Mean => xs.mean(1)?,
        };
        let xs = self.dense.forward(&pooled)?;
        let xs = match self.activation {
            HeadActivation::Tanh => xs.tanh()?,
            HeadActivation::Act(act) => act.forward(&xs)?,
        };
        let xs = match &self.norm {
            Some(norm) => norm.forward(&xs)?,
            None => xs,
        };
        self.out.forward(&xs)
    }

    pub fn add_residual(&self, root: &UnVarBuilder) {
        let (dense, norm, out) = &self.names;
        root.pp(dense).add(&self.dense);
        if let (Some(name), Some(layer)) = (norm, &self.norm) {
            root.pp(name).add(layer);
        }
        root.pp(out).add(&self.out);
    }
}
