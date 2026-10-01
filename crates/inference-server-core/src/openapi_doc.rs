//! The OpenAPI document: served from the committed `docs/openapi.json`, which tests regenerate from the handlers.

const OPENAPI_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/openapi.json"
));

/// The routes' OpenAPI document with every path under `base_path`; serve it with `SwaggerUi::external_url_unchecked`.
pub fn get_openapi_doc(base_path: Option<&str>) -> serde_json::Value {
    let mut doc: serde_json::Value =
        serde_json::from_str(OPENAPI_JSON).expect("docs/openapi.json is valid JSON");
    if let Some(prefix) = base_path.filter(|prefix| !prefix.is_empty())
        && let Some(paths) = doc
            .get_mut("paths")
            .and_then(serde_json::Value::as_object_mut)
    {
        *paths = std::mem::take(paths)
            .into_iter()
            .map(|(path, item)| (format!("{prefix}{path}"), item))
            .collect();
    }
    doc
}

#[cfg(test)]
mod generated {
    use utoipa::OpenApi;

    use crate::{
        anthropic::{
            __path_anthropic_count_tokens, __path_anthropic_messages, AnthropicContainer,
            AnthropicContentBlock, AnthropicCountTokensResponse, AnthropicError,
            AnthropicErrorBody, AnthropicImageSource, AnthropicJsonOutputFormat, AnthropicMessage,
            AnthropicMessageContent, AnthropicMessageResponse, AnthropicMessagesRequest,
            AnthropicOutputConfig, AnthropicResponseContentBlock, AnthropicSkillReference,
            AnthropicSystem, AnthropicThinking, AnthropicTool, AnthropicToolChoice, AnthropicUsage,
            AnthropicWebSearchUserLocation,
        },
        approvals::{
            __path_resolve_agent_approval, ApprovalDecision, ApprovalDecisionRequest,
            ApprovalDecisionResponse,
        },
        auth::{__path_sign_in, __path_sign_out, SignInRequest},
        chat_completion::__path_chatcompletions,
        completions::__path_completions,
        embeddings::__path_embeddings,
        files::{
            __path_delete_file, __path_get_container_file, __path_get_container_file_content,
            __path_get_file, __path_get_file_content, __path_list_container_files,
            __path_list_files, __path_upload_file, ContainerFileMetadata, FileMetadata, SourceMeta,
        },
        handlers::{
            __path_add_model, __path_add_model_alias, __path_calibration_apply,
            __path_calibration_start, __path_calibration_status, __path_delete_session,
            __path_get_model_status, __path_get_session, __path_health, __path_model_cache_stats,
            __path_models, __path_put_session, __path_re_isq, __path_reload_model,
            __path_remove_model, __path_set_default_model, __path_system_doctor,
            __path_system_info, __path_tune_model, __path_unload_model, ModelOperationRequest,
            ModelStatus, ModelStatusResponse,
        },
        image_generation::__path_image_generation,
        lora_adapters::{
            __path_list_lora_adapters, __path_load_lora_adapter, __path_unload_lora_adapter,
            LoadLoraAdapterRequest, LoraAdapterListResponse, LoraAdapterObject,
            UnloadLoraAdapterRequest,
        },
        metrics::__path_metrics,
        openai::{
            AdapterGenerationSelection, AdapterSelection, AudioResponseFormat,
            ChatCompletionRequest, CompletionRequest, EmbeddingData, EmbeddingEncodingFormat,
            EmbeddingInput, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, EmbeddingVector,
            FunctionCalled, Grammar, ImageGenerationRequest, JsonSchemaResponseFormat, Message,
            MessageContent, MessageInnerContent, ModelObject, ModelObjects,
            OpenAiCodeInterpreterAutoContainer, OpenAiCodeInterpreterContainer,
            OpenAiCodeInterpreterContainerType, OpenAiCodeInterpreterTool,
            OpenAiCodeInterpreterToolType, OpenAiFunctionToolType, OpenAiResponsesFunctionTool,
            OpenAiShellEnvironment, OpenAiShellSkill, OpenAiShellTool, OpenAiShellToolType,
            OpenAiTool, OpenAiWebSearchTool, OpenAiWebSearchToolType, OpenAiWebSearchUserLocation,
            ResponseFormat, ResponsesAnnotation, ResponsesChunk, ResponsesContent,
            ResponsesCreateRequest, ResponsesDelta, ResponsesDeltaContent, ResponsesDeltaOutput,
            ResponsesError, ResponsesIncompleteDetails, ResponsesInputTokensDetails,
            ResponsesMessages, ResponsesObject, ResponsesOutput, ResponsesOutputTokensDetails,
            ResponsesUsage, SpeechGenerationRequest, StopTokens, ToolCall,
        },
        responses::{
            __path_cancel_response, __path_create_response, __path_delete_response,
            __path_get_response,
        },
        responses_types::content::{FileCitation, FilePathInfo, UrlCitation},
        skills::{
            __path_list_skill_versions, __path_list_skills, __path_upload_skill,
            __path_upload_skill_version, AnthropicSkillListObject, AnthropicSkillObject,
            AnthropicSkillVersionListObject, AnthropicSkillVersionObject, SkillListObject,
            SkillListQuery, SkillObject, SkillVersionObject,
        },
        speech_generation::__path_speech_generation,
        system::{TuneModelRequest, TuneProfileRequest},
    };
    use inference_api::operations::{CalibrationApplyRequest, ReIsqRequest, ReIsqResponse};
    use inference_core::{
        ApproximateUserLocation, CalibrationStatus, Function, ImageGenerationResponseFormat,
        NamedFunctionToolChoice, SearchContextSize, SerializedSession, Tool, ToolChoice, ToolType,
        WebSearchContentType, WebSearchFilters, WebSearchImageSettings, WebSearchOptions,
        WebSearchReturnTokenBudget, WebSearchUserLocation,
    };

    pub(super) fn doc() -> utoipa::openapi::OpenApi {
        #[derive(OpenApi)]
        #[openapi(
            paths(models, model_cache_stats, health, chatcompletions, anthropic_messages, anthropic_count_tokens, completions, embeddings, re_isq, calibration_start, calibration_status, calibration_apply, image_generation, speech_generation, create_response, get_response, delete_response, cancel_response, upload_skill, list_skills, upload_skill_version, list_skill_versions, load_lora_adapter, unload_lora_adapter, list_lora_adapters, unload_model, reload_model, get_model_status, add_model, remove_model, set_default_model, add_model_alias, tune_model, system_info, system_doctor, get_session, put_session, delete_session, list_files, upload_file, get_file, get_file_content, delete_file, list_container_files, get_container_file, get_container_file_content, resolve_agent_approval, sign_in, sign_out, metrics),
            components(schemas(
                // Not a route's body: the engine spec the C ABI and bindings load from, typed from this document.
                inference_api::EngineSpec,
                // Engine operations the C ABI offers without an HTTP route.
                inference_api::operations::SessionList,
                inference_api::operations::SessionDeleted,
                inference_api::operations::SessionStored,
                inference_api::operations::SessionForkRequest,
                inference_api::operations::McpToolList,
                inference_api::operations::McpToolObject,
                inference_api::models::ModelServed,
                inference_api::models::ModelRemoved,
                inference_api::models::DefaultModel,
                inference_api::models::ModelAlias,
                SignInRequest,
                inference_api::operations::TokenizeRequest,
                inference_api::operations::TokenizeResponse,
                inference_api::operations::DetokenizeRequest,
                inference_api::operations::DetokenizeResponse,
                ApprovalDecision,
                ApprovalDecisionRequest,
                ApprovalDecisionResponse,
                AdapterGenerationSelection,
                AdapterSelection,
                ApproximateUserLocation,
                AnthropicContainer,
                AnthropicContentBlock,
                AnthropicCountTokensResponse,
                AnthropicError,
                AnthropicErrorBody,
                AnthropicImageSource,
                AnthropicJsonOutputFormat,
                AnthropicMessage,
                AnthropicMessageContent,
                AnthropicMessageResponse,
                AnthropicMessagesRequest,
                AnthropicOutputConfig,
                AnthropicResponseContentBlock,
                AnthropicSkillReference,
                AnthropicSkillListObject,
                AnthropicSkillObject,
                AnthropicSkillVersionListObject,
                AnthropicSkillVersionObject,
                AnthropicSystem,
                AnthropicThinking,
                AnthropicTool,
                AnthropicToolChoice,
                AnthropicUsage,
                AnthropicWebSearchUserLocation,
                AudioResponseFormat,
                CalibrationStatus,
                ChatCompletionRequest,
                CompletionRequest,
                EmbeddingData,
                EmbeddingEncodingFormat,
                EmbeddingInput,
                EmbeddingRequest,
                EmbeddingResponse,
                EmbeddingUsage,
                EmbeddingVector,
                ContainerFileMetadata,
                FileMetadata,
                FileCitation,
                FilePathInfo,
                Function,
                FunctionCalled,
                Grammar,
                ImageGenerationRequest,
                LoadLoraAdapterRequest,
                LoraAdapterListResponse,
                LoraAdapterObject,
                ImageGenerationResponseFormat,
                JsonSchemaResponseFormat,
                Message,
                MessageContent,
                MessageInnerContent,
                ModelObject,
                ModelObjects,
                inference_api::models::CacheStats,
                inference_api::models::ModelCacheStats,
                inference_api::models::EncoderCacheStats,
                NamedFunctionToolChoice,
                OpenAiCodeInterpreterAutoContainer,
                OpenAiCodeInterpreterContainer,
                OpenAiCodeInterpreterContainerType,
                OpenAiCodeInterpreterTool,
                OpenAiCodeInterpreterToolType,
                OpenAiFunctionToolType,
                OpenAiResponsesFunctionTool,
                OpenAiShellEnvironment,
                OpenAiShellSkill,
                OpenAiShellTool,
                OpenAiShellToolType,
                OpenAiTool,
                OpenAiWebSearchTool,
                OpenAiWebSearchToolType,
                OpenAiWebSearchUserLocation,
                ModelOperationRequest,
                ModelStatus,
                ModelStatusResponse,
                ReIsqRequest, ReIsqResponse, CalibrationApplyRequest,
                ResponseFormat,
                ResponsesAnnotation,
                ResponsesChunk,
                ResponsesContent,
                ResponsesCreateRequest,
                ResponsesDelta,
                ResponsesDeltaContent,
                ResponsesDeltaOutput,
                ResponsesError,
                ResponsesIncompleteDetails,
                ResponsesInputTokensDetails,
                ResponsesMessages,
                ResponsesObject,
                ResponsesOutput,
                ResponsesOutputTokensDetails,
                ResponsesUsage,
                SearchContextSize,
                SerializedSession,
                SkillListObject,
                SkillListQuery,
                SkillObject,
                SkillVersionObject,
                SourceMeta,
                SpeechGenerationRequest,
                StopTokens,
                Tool,
                ToolCall,
                ToolChoice,
                ToolType,
                TuneModelRequest,
                TuneProfileRequest,
                UnloadLoraAdapterRequest,
                UrlCitation,
                WebSearchContentType,
                WebSearchFilters,
                WebSearchImageSettings,
                WebSearchOptions,
                WebSearchReturnTokenBudget,
                WebSearchUserLocation
            )),
            tags(
                (name = "inference.rs", description = "inference.rs API"),
                (name = "LoRA adapters", description = "Dynamic LoRA discovery and lifecycle operations")
            ),
            info(
                title = "inference.rs",
                license(
                name = "MIT",
            )
            )
        )]
        struct ApiDoc;

        ApiDoc::openapi()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMITTED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/openapi.json");

    fn render() -> String {
        serde_json::to_string_pretty(&generated::doc()).expect("openapi doc serializes") + "\n"
    }

    #[test]
    fn inference_responses_have_schemas_with_adapter_generation() {
        let value = serde_json::to_value(generated::doc()).unwrap();
        for schema in [
            "ChatCompletionResponse",
            "ChatCompletionChunkResponse",
            "CompletionResponse",
            "CompletionChunkResponse",
            "ResponseResource",
        ] {
            assert!(
                value["components"]["schemas"][schema]["properties"]
                    .get("adapter_generation")
                    .is_some(),
                "missing adapter_generation from {schema}"
            );
        }

        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            assert!(
                value["paths"][path]["post"]["responses"]["200"]["content"]
                    .as_object()
                    .is_some_and(|content| !content.is_empty()),
                "missing response content schema for {path}"
            );
        }

        for path in ["/v1/load_lora_adapter", "/v1/unload_lora_adapter"] {
            assert!(
                value["paths"][path]["post"]["description"]
                    .as_str()
                    .is_some_and(
                        |description| description.contains("only when runtime LoRA mutation")
                    )
            );
        }

        let list_model = &value["paths"]["/v1/lora_adapters"]["get"]["parameters"][0];
        assert_eq!(list_model["name"], "model");
        assert_eq!(list_model["in"], "query");
        assert_eq!(list_model["required"], false);

        let load_example = &value["paths"]["/v1/load_lora_adapter"]["post"]["requestBody"]["content"]
            ["application/json"]["example"];
        assert_eq!(
            load_example,
            &serde_json::json!({
                "lora_name": "production",
                "lora_path": "/srv/adapters/production"
            })
        );

        let unload_example = &value["paths"]["/v1/unload_lora_adapter"]["post"]["requestBody"]["content"]
            ["application/json"]["example"];
        assert_eq!(
            unload_example,
            &serde_json::json!({"lora_name": "production"})
        );

        assert!(
            value["paths"]["/v1/lora_adapters"]["get"]["description"]
                .as_str()
                .is_some_and(|description| description.contains("Always registered"))
        );
    }

    #[test]
    fn the_served_document_is_the_committed_one_under_the_base_path() {
        let committed: serde_json::Value = serde_json::from_str(OPENAPI_JSON).unwrap();
        assert_eq!(get_openapi_doc(None), committed);
        let nested = get_openapi_doc(Some("/api/inference"));
        assert!(
            nested["paths"]
                .get("/api/inference/v1/chat/completions")
                .is_some()
        );
        assert!(nested["paths"].get("/v1/chat/completions").is_none());
        assert_eq!(nested["components"], committed["components"]);
    }

    // docs/openapi.json is a committed artifact consumed by the docs site.
    #[test]
    fn openapi_matches_committed() {
        let committed = std::fs::read_to_string(COMMITTED).unwrap_or_default();
        assert_eq!(
            render(),
            committed,
            "docs/openapi.json is stale; regenerate with: cargo test -p inference-server-core regenerate_openapi -- --ignored"
        );
    }

    #[test]
    #[ignore = "writes docs/openapi.json"]
    fn regenerate_openapi() {
        std::fs::write(COMMITTED, render()).expect("write openapi dump");
    }
}
