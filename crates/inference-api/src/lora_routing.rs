//! Routing a request's model name to a loaded LoRA adapter, for every API that takes a model id.

use std::collections::HashSet;

use inference_core::{InferenceRs, InferenceRsError, LoraAdapterInfo, LoraAdapterRoute};

use crate::api_error::{ApiError, ApiErrorKind};

// The shape a malformed generation had when the wire type rejected it during deserialization.
const INVALID_REQUEST_BODY_CODE: &str = "invalid_request_body";
const ADAPTER_PARAM: &str = "adapter";

pub const DEFAULT_MODEL_ID: &str = "default";

#[derive(Clone, Debug)]
pub struct LoraAdapterModel {
    pub id: String,
    pub parent: String,
    pub adapter: LoraAdapterInfo,
}

pub fn list_lora_adapter_models(
    state: &InferenceRs,
) -> Result<Vec<LoraAdapterModel>, InferenceRsError> {
    let routes = state.list_lora_adapter_routes()?;
    adapter_models_from_routes(routes, |model_id| state.model_exists(model_id))
}

fn adapter_model_id_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '%' => encoded.push_str("%25"),
            ':' => encoded.push_str("%3A"),
            '#' => encoded.push_str("%23"),
            character => encoded.push(character),
        }
    }
    encoded
}

fn adapter_models_from_routes(
    routes: Vec<LoraAdapterRoute>,
    mut model_exists: impl FnMut(&str) -> Result<bool, InferenceRsError>,
) -> Result<Vec<LoraAdapterModel>, InferenceRsError> {
    let mut used = HashSet::new();
    let mut models = Vec::with_capacity(routes.len());
    for route in routes {
        let alias = &route.adapter.alias;
        let mut id = format!(
            "{}::{}",
            adapter_model_id_component(&route.model_id),
            adapter_model_id_component(alias)
        );
        let base_id = id.clone();
        let mut suffix = 2usize;
        while used.contains(&id) || model_exists(&id)? || id == DEFAULT_MODEL_ID {
            id = format!("{base_id}#{suffix}");
            suffix += 1;
        }
        used.insert(id.clone());
        models.push(LoraAdapterModel {
            id,
            parent: route.model_id,
            adapter: route.adapter,
        });
    }
    Ok(models)
}

fn select_lora_adapter_model(
    selected: &LoraAdapterModel,
    model: &mut String,
    adapter: &mut Option<crate::openai::AdapterSelection>,
) -> Result<(), String> {
    if adapter.is_some() {
        return Err(format!(
            "model `{}` already selects LoRA adapter `{}`; omit `adapter` or use base model `{}`",
            selected.id, selected.adapter.alias, selected.parent
        ));
    }
    *model = selected.parent.clone();
    *adapter = Some(crate::openai::AdapterSelection::Alias(
        selected.adapter.alias.clone(),
    ));
    Ok(())
}

fn resolve_lora_adapter_model_from_models(
    models: &[LoraAdapterModel],
    model: &mut String,
    adapter: &mut Option<crate::openai::AdapterSelection>,
) -> Result<(), String> {
    if let Some(selected) = models.iter().find(|candidate| candidate.id == *model) {
        return select_lora_adapter_model(selected, model, adapter);
    }

    let alias_matches = models
        .iter()
        .filter(|candidate| candidate.adapter.alias == *model)
        .collect::<Vec<_>>();
    match alias_matches.as_slice() {
        [] => Ok(()),
        [selected] => select_lora_adapter_model(selected, model, adapter),
        _ => Err(format!(
            "LoRA adapter model `{model}` is ambiguous; use one of: {}",
            alias_matches
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

pub fn is_resolvable_lora_adapter_model(models: &[LoraAdapterModel], model: &str) -> bool {
    models.iter().any(|candidate| candidate.id == model)
        || models
            .iter()
            .filter(|candidate| candidate.adapter.alias == model)
            .take(2)
            .count()
            == 1
}

pub fn resolve_lora_adapter_model(
    state: &InferenceRs,
    model: &mut String,
    adapter: &mut Option<crate::openai::AdapterSelection>,
) -> Result<(), ApiError> {
    if model == DEFAULT_MODEL_ID
        || state
            .model_exists(model)
            .map_err(|error| ApiError::from_error(&error, ApiErrorKind::Internal))?
    {
        return Ok(());
    }

    let models = list_lora_adapter_models(state)
        .map_err(|error| ApiError::from_error(&error, ApiErrorKind::Internal))?;
    resolve_lora_adapter_model_from_models(&models, model, adapter)
        .map_err(ApiError::invalid_request)
}

/// The engine's adapter selection for a request's wire-level one; a malformed generation ID is a bad request.
pub(crate) fn core_adapter_selection(
    selection: crate::openai::AdapterSelection,
) -> anyhow::Result<inference_core::AdapterSelection> {
    Ok(match selection {
        crate::openai::AdapterSelection::Alias(alias) => {
            inference_core::AdapterSelection::alias(alias)
        }
        crate::openai::AdapterSelection::Generation(selection) => {
            let generation = selection.generation.parse().map_err(|_| {
                ApiError::new(
                    ApiErrorKind::InvalidRequest,
                    format!(
                        "`{}` is not a LoRA adapter generation ID",
                        selection.generation
                    ),
                    Some(INVALID_REQUEST_BODY_CODE),
                    Some(ADAPTER_PARAM),
                )
            })?;
            inference_core::AdapterSelection::generation(generation)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_selection_accepts_alias_and_exact_generation() {
        let alias: crate::openai::AdapterSelection =
            serde_json::from_value(serde_json::json!("production")).unwrap();
        let alias = core_adapter_selection(alias).unwrap();
        assert_eq!(
            serde_json::to_value(alias).unwrap(),
            serde_json::json!("production")
        );

        let generation = inference_core::AdapterGenerationId::from_bytes([0x5a; 32]);
        let wire = serde_json::json!({"generation": generation.to_string()});
        let exact: crate::openai::AdapterSelection = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(&exact).unwrap(), wire);
        let exact = core_adapter_selection(exact).unwrap();
        assert_eq!(exact.resolved_generation(), Some(generation));

        let malformed =
            serde_json::from_value(serde_json::json!({"generation": "not-a-generation"}));
        let err = core_adapter_selection(malformed.unwrap()).unwrap_err();
        let api_error = err.downcast_ref::<ApiError>().expect("an ApiError");
        assert_eq!(api_error.kind, ApiErrorKind::InvalidRequest);
        assert_eq!(api_error.code.as_deref(), Some(INVALID_REQUEST_BODY_CODE));
        assert_eq!(api_error.param.as_deref(), Some(ADAPTER_PARAM));
    }

    fn route(model_id: &str, alias: &str, generation: u8) -> LoraAdapterRoute {
        LoraAdapterRoute {
            model_id: model_id.to_string(),
            adapter: LoraAdapterInfo {
                alias: alias.to_string(),
                source: "source".to_string(),
                revision: None,
                generation: inference_core::AdapterGenerationId::from_bytes([generation; 32]),
                rank: 8,
                bytes: 16,
            },
        }
    }

    #[test]
    fn adapter_model_ids_are_stably_qualified() {
        let models = adapter_models_from_routes(
            vec![
                route("base-a", "code", 1),
                route("base-b", "code", 2),
                route("base-a", "math", 3),
            ],
            |_| Ok(false),
        )
        .unwrap();
        assert_eq!(
            models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            vec!["base-a::code", "base-b::code", "base-a::math"]
        );

        let models = adapter_models_from_routes(vec![route("base-a", "math", 3)], |model_id| {
            Ok(model_id == "math")
        })
        .unwrap();
        assert_eq!(models[0].id, "base-a::math");

        let before =
            adapter_models_from_routes(vec![route("base-a", "code", 1)], |_| Ok(false)).unwrap();
        let after = adapter_models_from_routes(
            vec![route("base-a", "code", 1), route("base-b", "code", 2)],
            |_| Ok(false),
        )
        .unwrap();
        assert_eq!(before[0].id, after[0].id);

        let models =
            adapter_models_from_routes(vec![route("a::b", "c", 1), route("a", "b::c", 2)], |_| {
                Ok(false)
            })
            .unwrap();
        assert_eq!(models[0].id, "a%3A%3Ab::c");
        assert_eq!(models[1].id, "a::b%3A%3Ac");
    }

    #[test]
    fn adapter_model_resolution_accepts_stable_ids_and_unique_aliases() {
        let models = adapter_models_from_routes(
            vec![
                route("base-a", "code", 1),
                route("base-b", "code", 2),
                route("base-a", "math", 3),
            ],
            |_| Ok(false),
        )
        .unwrap();

        let mut model = "base-a::code".to_string();
        let mut adapter = None;
        resolve_lora_adapter_model_from_models(&models, &mut model, &mut adapter).unwrap();
        assert_eq!(model, "base-a");
        assert!(matches!(
            adapter,
            Some(crate::openai::AdapterSelection::Alias(alias)) if alias == "code"
        ));

        let mut model = "math".to_string();
        let mut adapter = None;
        resolve_lora_adapter_model_from_models(&models, &mut model, &mut adapter).unwrap();
        assert_eq!(model, "base-a");
        assert!(is_resolvable_lora_adapter_model(&models, "math"));
        assert!(is_resolvable_lora_adapter_model(&models, "base-a::code"));
        assert!(!is_resolvable_lora_adapter_model(&models, "code"));
        assert!(!is_resolvable_lora_adapter_model(
            &models,
            "unbounded-user-value"
        ));

        let mut model = "code".to_string();
        let mut adapter = None;
        let error =
            resolve_lora_adapter_model_from_models(&models, &mut model, &mut adapter).unwrap_err();
        assert!(error.contains("base-a::code"));
        assert!(error.contains("base-b::code"));
    }
}
