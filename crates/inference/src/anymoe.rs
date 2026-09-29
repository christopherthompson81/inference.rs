use inference_core::{AnyMoeConfig, AnyMoeSpec, LoadOverrides};

use crate::{
    GgufModelBuilder, Model, TextModelBuilder,
    model_builder_trait::{
        build_gguf_pipeline_as, build_model_from_pipeline, build_text_pipeline_as, gguf_selection,
        plain_text_selection,
    },
};

enum AnyMoeBase {
    Text(TextModelBuilder),
    Gguf(GgufModelBuilder),
}

/// Configure and build an AnyMoE (Mixture of Experts) model on top of a text model.
pub struct AnyMoeModelBuilder {
    base: AnyMoeBase,
    config: AnyMoeConfig,
    path: String,
    prefix: String,
    mlp: String,
    model_ids: Vec<String>,
    layers: Vec<usize>,
}

impl AnyMoeModelBuilder {
    /// Create from a base [`TextModelBuilder`] with AnyMoE config, gating model path, prefix,
    /// MLP name, expert model IDs, and target layers.
    pub fn from_text_builder(
        base: TextModelBuilder,
        config: AnyMoeConfig,
        path: impl ToString,
        prefix: impl ToString,
        mlp: impl ToString,
        model_ids: Vec<impl ToString>,
        layers: Vec<usize>,
    ) -> Self {
        Self {
            base: AnyMoeBase::Text(base),
            config,
            path: path.to_string(),
            prefix: prefix.to_string(),
            mlp: mlp.to_string(),
            model_ids: model_ids
                .into_iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>(),
            layers,
        }
    }

    /// Create an AnyMoE model from a GGUF base.
    pub fn from_gguf_builder(
        base: GgufModelBuilder,
        config: AnyMoeConfig,
        path: impl ToString,
        prefix: impl ToString,
        mlp: impl ToString,
        model_ids: Vec<impl ToString>,
        layers: Vec<usize>,
    ) -> Self {
        Self {
            base: AnyMoeBase::Gguf(base),
            config,
            path: path.to_string(),
            prefix: prefix.to_string(),
            mlp: mlp.to_string(),
            model_ids: model_ids
                .into_iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>(),
            layers,
        }
    }

    /// Load the AnyMoE model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let overrides = LoadOverrides {
            anymoe: Some(AnyMoeSpec {
                config: self.config,
                path: self.path,
                prefix: self.prefix,
                mlp: self.mlp,
                model_ids: self.model_ids,
                layers: self.layers,
            }),
            ..Default::default()
        };
        let (pipeline, scheduler_config, add_model_config) = match self.base {
            AnyMoeBase::Text(base) => {
                let model_selected = plain_text_selection(&base);
                build_text_pipeline_as(base, model_selected, overrides).await?
            }
            AnyMoeBase::Gguf(base) => {
                build_gguf_pipeline_as(base, gguf_selection, overrides).await?
            }
        };

        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}

#[cfg(test)]
mod tests {
    use inference_core::{AnyMoeConfig, AnyMoeExpertType};

    use super::{AnyMoeBase, AnyMoeModelBuilder};
    use crate::GgufModelBuilder;

    #[test]
    fn accepts_a_gguf_base() {
        let builder = AnyMoeModelBuilder::from_gguf_builder(
            GgufModelBuilder::new("repo", vec!["model.gguf"]),
            AnyMoeConfig {
                hidden_size: 128,
                lr: 1e-3,
                epochs: 1,
                batch_size: 1,
                expert_type: AnyMoeExpertType::FineTuned,
                gate_model_id: None,
                training: false,
                loss_csv_path: None,
            },
            "train.json",
            "model.layers",
            "mlp",
            vec!["expert"],
            vec![0],
        );

        assert!(matches!(builder.base, AnyMoeBase::Gguf(_)));
    }
}
