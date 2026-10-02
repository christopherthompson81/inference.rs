//! A text or GGUF model whose MLPs are mixed with expert models' through a trained gate.

use inference_api::{
    engine::{AnyMoeSpec, EngineCallbacks, ModelSelected},
    sdk::AnyMoeConfig,
};

use crate::{GgufModelBuilder, Model, TextModelBuilder, error::Result, load::LoadOptions};

/// Loads a base model with an AnyMoE layer over the MLPs of `model_ids`.
pub struct AnyMoeModelBuilder {
    model: ModelSelected,
    options: LoadOptions,
    anymoe: AnyMoeSpec,
}

impl AnyMoeModelBuilder {
    /// `path` holds the gate's training data (or trained weights), `prefix` names the layer modules (e.g.
    /// `model.layers`), `mlp` the MLP within each, and `layers` the layers to mix (empty mixes all).
    pub fn from_text_builder(
        base: TextModelBuilder,
        config: AnyMoeConfig,
        path: impl ToString,
        prefix: impl ToString,
        mlp: impl ToString,
        model_ids: Vec<impl ToString>,
        layers: Vec<usize>,
    ) -> Self {
        let model = base.model_selected();
        let anymoe = anymoe_spec(config, path, prefix, mlp, model_ids, layers);
        Self {
            model,
            options: base.options,
            anymoe,
        }
    }

    /// As [`Self::from_text_builder`], over a GGUF base.
    pub fn from_gguf_builder(
        base: GgufModelBuilder,
        config: AnyMoeConfig,
        path: impl ToString,
        prefix: impl ToString,
        mlp: impl ToString,
        model_ids: Vec<impl ToString>,
        layers: Vec<usize>,
    ) -> Self {
        let model = base.model_selected();
        let anymoe = anymoe_spec(config, path, prefix, mlp, model_ids, layers);
        Self {
            model,
            options: base.options,
            anymoe,
        }
    }

    pub fn into_spec(self) -> (inference_api::EngineSpec, EngineCallbacks) {
        let (mut spec, callbacks) = self.options.spec(self.model);
        spec.anymoe = Some(self.anymoe);
        (spec, callbacks)
    }

    pub async fn build(self) -> Result<Model> {
        let with_logging = self.options.with_logging;
        let (spec, callbacks) = self.into_spec();
        crate::load::load_engine(spec, callbacks, with_logging).await
    }
}

fn anymoe_spec(
    config: AnyMoeConfig,
    path: impl ToString,
    prefix: impl ToString,
    mlp: impl ToString,
    model_ids: Vec<impl ToString>,
    layers: Vec<usize>,
) -> AnyMoeSpec {
    AnyMoeSpec {
        config,
        path: path.to_string(),
        prefix: prefix.to_string(),
        mlp: mlp.to_string(),
        model_ids: model_ids.into_iter().map(|id| id.to_string()).collect(),
        layers,
    }
}
