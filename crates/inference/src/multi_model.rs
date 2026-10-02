//! Several models served by one engine, each request picking one by id.

use inference_api::engine::{EngineCallbacks, EngineSpec, ModelSelected, ModelSpec};

use crate::{
    DiffusionModelBuilder, EmbeddingModelBuilder, EngineLoadError, GgufLoraModelBuilder,
    GgufModelBuilder, GgufXLoraModelBuilder, LoraModelBuilder, Model, ModelBuilder,
    MultimodalModelBuilder, SpeechModelBuilder, TextModelBuilder, UqffEmbeddingModelBuilder,
    UqffMultimodalModelBuilder, UqffTextModelBuilder, XLoraModelBuilder, error::Result,
    load::LoadOptions,
};

const NO_MODELS: &str = "MultiModelBuilder needs at least one model";

/// A builder one model of a [`MultiModelBuilder`] comes from.
pub trait IntoModelSpec {
    /// The model and its own overrides (ISQ, template, device layers, revision); the rest is the multi-builder's.
    fn model_spec(self) -> Result<ModelSpec>;
}

// The settings `ModelSpec` keeps per model; the rest of `options` is the engine's, set on the multi-model builder.
fn per_model(model: ModelSelected, options: LoadOptions) -> ModelSpec {
    let runtime = options.runtime;
    ModelSpec {
        model,
        model_id: None,
        chat_template: runtime.chat_template,
        jinja_explicit: runtime.jinja_explicit,
        max_model_len: runtime.max_model_len,
        hf_config_overrides: runtime.hf_config_overrides,
        device_layers: runtime.device_layers,
        isq: runtime.isq,
        encoder_cache_memory_bytes: runtime.encoder_cache_memory_bytes,
        hf_revision: runtime.hf_revision,
    }
}

macro_rules! into_model_spec {
    ($($builder:ty => $($options:ident).+;)*) => {$(
        impl IntoModelSpec for $builder {
            fn model_spec(self) -> Result<ModelSpec> {
                let model = self.model_selected();
                Ok(per_model(model, self.$($options).+))
            }
        }
    )*};
}

into_model_spec! {
    TextModelBuilder => options;
    MultimodalModelBuilder => options;
    ModelBuilder => options;
    GgufModelBuilder => options;
    DiffusionModelBuilder => options;
    SpeechModelBuilder => options;
    EmbeddingModelBuilder => options;
    LoraModelBuilder => text_model.options;
    XLoraModelBuilder => text_model.options;
}

impl IntoModelSpec for GgufLoraModelBuilder {
    fn model_spec(self) -> Result<ModelSpec> {
        let model = self.checked_model_selected()?;
        Ok(per_model(model, self.gguf_model.options))
    }
}

impl IntoModelSpec for GgufXLoraModelBuilder {
    fn model_spec(self) -> Result<ModelSpec> {
        let model = self.checked_model_selected()?;
        Ok(per_model(model, self.gguf_model.options))
    }
}

impl IntoModelSpec for UqffTextModelBuilder {
    fn model_spec(self) -> Result<ModelSpec> {
        self.into_inner().model_spec()
    }
}

impl IntoModelSpec for UqffMultimodalModelBuilder {
    fn model_spec(self) -> Result<ModelSpec> {
        self.into_inner().model_spec()
    }
}

impl IntoModelSpec for UqffEmbeddingModelBuilder {
    fn model_spec(self) -> Result<ModelSpec> {
        self.into_inner().model_spec()
    }
}

/// Several models in one engine, the first the default; engine-wide settings (device, caches, tools) are set here.
pub struct MultiModelBuilder {
    models: Vec<Result<ModelSpec>>,
    default_model_id: Option<String>,
    options: LoadOptions,
}

impl Default for MultiModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiModelBuilder {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            default_model_id: None,
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    /// Adds a model that requests name by its own id.
    pub fn add_model(mut self, builder: impl IntoModelSpec) -> Self {
        self.models.push(builder.model_spec());
        self
    }

    /// Adds a model that requests name by `alias`.
    pub fn add_model_with_alias(
        mut self,
        alias: impl Into<String>,
        builder: impl IntoModelSpec,
    ) -> Self {
        let alias = alias.into();
        self.models.push(builder.model_spec().map(|mut spec| {
            spec.model_id = Some(alias);
            spec
        }));
        self
    }

    /// The model, by id or alias, a request without a `model` goes to.
    pub fn with_default_model(mut self, model_id: impl ToString) -> Self {
        self.default_model_id = Some(model_id.to_string());
        self
    }

    pub fn into_spec(self) -> Result<(EngineSpec, EngineCallbacks)> {
        let models = self.models.into_iter().collect::<Result<Vec<_>>>()?;
        if models.is_empty() {
            return Err(EngineLoadError::InvalidSpec(NO_MODELS.to_string()).into());
        }
        let default_model_id = self.default_model_id;
        Ok(self.options.spec_with(|spec| {
            spec.models = models;
            spec.default_model_id = default_model_id;
        }))
    }

    pub async fn build(self) -> Result<Model> {
        let with_logging = self.options.with_logging;
        let (spec, callbacks) = self.into_spec()?;
        crate::load::load_engine(spec, callbacks, with_logging).await
    }
}
