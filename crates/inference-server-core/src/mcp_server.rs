//! MCP server support: exposes the loaded model as an MCP tool over HTTP JSON-RPC.

use std::sync::Arc;

use axum::{
    Extension, Router,
    extract::{State, rejection::JsonRejection},
    response::Json,
    routing::post,
};
use inference_api::Engine;
use inference_core::{AgentPermission, SupportedModality};
use serde_json::{Value, json};

use crate::{
    handler_core::{ApiError, ApiErrorKind},
    openai::ChatCompletionRequest,
};

pub const MCP_ROUTE: &str = "/mcp";
pub const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

const JSONRPC_VERSION: &str = "2.0";
const CHAT_TOOL_NAME: &str = "chat";

const PARSE_ERROR: i32 = -32700;
const INVALID_REQUEST: i32 = -32600;
const INVALID_PARAMS: i32 = -32602;
const METHOD_NOT_FOUND: i32 = -32601;
const INTERNAL_ERROR: i32 = -32603;
const INTERNAL_ERROR_MESSAGE: &str = "Internal error";

const MCP_INSTRUCTIONS: &str = r#"
This server provides LLM text and multimodal model inference. You can use the following tools:
- `chat` for sending a chat completion request with a model message history
"#;

#[derive(Debug, serde::Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(serde::Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(serde::Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

fn ok_response(id: Option<Value>, result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: JSONRPC_VERSION,
        id,
        result: Some(result),
        error: None,
    }
}

fn error_response(id: Option<Value>, code: i32, message: String) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: JSONRPC_VERSION,
        id,
        result: None,
        error: Some(JsonRpcError { code, message }),
    }
}

fn initialize_result() -> Value {
    json!({
        "capabilities": { "tools": {} },
        "instructions": MCP_INSTRUCTIONS,
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": { "name": "inference", "version": env!("CARGO_PKG_VERSION") },
    })
}

fn chat_tool() -> Value {
    json!({
        "name": CHAT_TOOL_NAME,
        "description": "Send a chat completion request with messages and other hyperparameters.",
        "inputSchema": {
            "type": "object",
            "required": ["messages"],
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Conversation messages so far",
                    "items": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {
                            "role": { "type": "string", "enum": ["user", "assistant", "system"] },
                            "content": { "type": "string" }
                        }
                    }
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum number of tokens to generate"
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature between 0 and 2",
                    "minimum": 0.0,
                    "maximum": 2.0
                }
            }
        }
    })
}

fn list_tools_result(chat_enabled: bool) -> Value {
    let tools = if chat_enabled {
        vec![chat_tool()]
    } else {
        Vec::new()
    };
    json!({ "tools": tools })
}

// Handles every method that does not need model access; tools/call returns None for the caller.
fn dispatch_stateless(
    method: &str,
    id: Option<Value>,
    chat_enabled: bool,
) -> Option<JsonRpcResponse> {
    match method {
        "initialize" => Some(ok_response(id, initialize_result())),
        "ping" => Some(ok_response(id, json!({}))),
        "tools/list" => Some(ok_response(id, list_tools_result(chat_enabled))),
        "tools/call" => None,
        other => Some(error_response(
            id,
            METHOD_NOT_FOUND,
            format!("Method not found: {other}"),
        )),
    }
}

struct McpState {
    engine: Engine,
    chat_enabled: bool,
}

/// Build the MCP router (`POST /mcp`, JSON-RPC 2.0), requiring `auth`'s keys when given. Mount on its own port or
/// into an existing app.
pub fn create_mcp_router(engine: &Engine, auth: Option<Arc<crate::auth::Auth>>) -> Router {
    let text_model = engine
        .state()
        .config(None)
        .map(|c| {
            c.modalities.input.contains(&SupportedModality::Text)
                && c.modalities.output.contains(&SupportedModality::Text)
        })
        .unwrap_or(false);
    // `ask` needs a stream to carry approvals and a tool call is one blocking chat, so the tool could never succeed.
    let asks = engine.chat_engine().agentic.agent_permission == Some(AgentPermission::Ask);
    let chat_enabled = text_model && !asks;
    let state = Arc::new(McpState {
        engine: engine.clone(),
        chat_enabled,
    });
    let router = Router::new()
        .route(MCP_ROUTE, post(handle_jsonrpc))
        .with_state(state);
    crate::auth::require(router, auth, crate::auth::KEY_ONLY_GUARD)
}

async fn handle_jsonrpc(
    State(state): State<Arc<McpState>>,
    Extension(owner): Extension<crate::auth::Owner>,
    payload: Result<Json<JsonRpcRequest>, JsonRejection>,
) -> Json<JsonRpcResponse> {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => return Json(json_rejection_response(error)),
    };
    Json(handle_request(&state, owner.as_deref(), request).await)
}

fn json_rejection_response(error: JsonRejection) -> JsonRpcResponse {
    let (code, message) = match error {
        JsonRejection::JsonSyntaxError(_) => (PARSE_ERROR, "Parse error"),
        _ => (INVALID_REQUEST, "Invalid Request"),
    };
    error_response(None, code, message.to_string())
}

async fn handle_request(
    state: &McpState,
    owner: Option<&str>,
    request: JsonRpcRequest,
) -> JsonRpcResponse {
    if request.jsonrpc != JSONRPC_VERSION {
        return error_response(
            request.id,
            INVALID_REQUEST,
            "Expected jsonrpc to be 2.0".to_string(),
        );
    }

    if let Some(response) =
        dispatch_stateless(&request.method, request.id.clone(), state.chat_enabled)
    {
        return response;
    }

    let params = request.params.unwrap_or(json!({}));
    let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    if tool_name != CHAT_TOOL_NAME || !state.chat_enabled {
        return error_response(
            request.id,
            INVALID_PARAMS,
            format!("Unknown tool: {tool_name}"),
        );
    }

    let args = params.get("arguments").cloned().unwrap_or(json!({}));
    let scoped;
    let engine = match owner {
        Some(owner) => {
            scoped = state.engine.for_owner(owner);
            &scoped
        }
        None => &state.engine,
    };
    match call_chat_tool(engine, args).await {
        Ok(result) => ok_response(request.id, result),
        Err(McpCallError::InvalidParams(message)) => {
            error_response(request.id, INVALID_PARAMS, message)
        }
        Err(McpCallError::Internal) => error_response(
            request.id,
            INTERNAL_ERROR,
            INTERNAL_ERROR_MESSAGE.to_string(),
        ),
    }
}

enum McpCallError {
    InvalidParams(String),
    Internal,
}

impl McpCallError {
    fn from_api_error(error: ApiError) -> Self {
        match error.kind {
            ApiErrorKind::InvalidRequest
            | ApiErrorKind::NotFound
            | ApiErrorKind::Gone
            | ApiErrorKind::Unauthorized
            | ApiErrorKind::Forbidden
            | ApiErrorKind::Conflict
            | ApiErrorKind::PayloadTooLarge
            | ApiErrorKind::UnsupportedMediaType => Self::InvalidParams(error.message),
            ApiErrorKind::RateLimited
            | ApiErrorKind::Unavailable
            | ApiErrorKind::Overloaded
            | ApiErrorKind::Internal => Self::Internal,
        }
    }
}

// Through the engine's chat operation, so the spec's agent defaults and approval policy apply as they do over HTTP.
async fn call_chat_tool(engine: &Engine, args: Value) -> Result<Value, McpCallError> {
    let chat_req: ChatCompletionRequest = serde_json::from_value(args)
        .map_err(|error| McpCallError::InvalidParams(error.to_string()))?;
    let response = engine
        .chat(chat_req, Default::default())
        .await
        .map_err(McpCallError::from_api_error)?;
    let content = response
        .choices
        .iter()
        .filter_map(|c| c.message.content.clone())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(json!({ "content": [{ "type": "text", "text": content }] }))
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, extract::FromRequest, http::Request};

    use super::*;

    fn dispatch(method: &str, chat_enabled: bool) -> JsonRpcResponse {
        dispatch_stateless(method, Some(json!(1)), chat_enabled).unwrap()
    }

    #[test]
    fn initialize_reports_protocol_and_tools_capability() {
        let resp = dispatch("initialize", true);
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert!(result["capabilities"]["tools"].is_object());
        assert_eq!(result["serverInfo"]["name"], "inference");
        assert!(resp.error.is_none());
    }

    #[test]
    fn tools_list_exposes_chat_for_text_models() {
        let result = dispatch("tools/list", true).result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], CHAT_TOOL_NAME);
        assert_eq!(tools[0]["inputSchema"]["required"], json!(["messages"]));
    }

    #[test]
    fn tools_list_is_empty_without_text_modality() {
        let result = dispatch("tools/list", false).result.unwrap();
        assert!(result["tools"].as_array().unwrap().is_empty());
    }

    #[test]
    fn ping_returns_empty_object() {
        assert_eq!(dispatch("ping", true).result.unwrap(), json!({}));
    }

    #[test]
    fn unknown_method_is_a_jsonrpc_error() {
        let resp = dispatch("bogus", true);
        assert!(resp.result.is_none());
        assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn tools_call_is_deferred_to_the_stateful_path() {
        assert!(dispatch_stateless("tools/call", None, true).is_none());
    }

    #[test]
    fn error_envelope_omits_result() {
        let resp = error_response(Some(json!(7)), INVALID_REQUEST, "bad".to_string());
        let wire = serde_json::to_value(&resp).unwrap();
        assert_eq!(wire["jsonrpc"], "2.0");
        assert_eq!(wire["error"]["code"], INVALID_REQUEST);
        assert!(wire.get("result").is_none());
    }

    #[test]
    fn call_errors_distinguish_invalid_params_from_internal_failures() {
        let missing_model = inference_core::InferenceRsError::ModelNotFound("missing".to_string());
        assert!(matches!(
            McpCallError::from_api_error(ApiError::from_error(
                &missing_model,
                ApiErrorKind::Internal
            )),
            McpCallError::InvalidParams(_)
        ));

        let internal = std::io::Error::other("private detail");
        assert!(matches!(
            McpCallError::from_api_error(ApiError::from_error(&internal, ApiErrorKind::Internal)),
            McpCallError::Internal
        ));
    }

    #[tokio::test]
    async fn json_rejections_use_jsonrpc_errors() {
        let malformed = Request::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from("{"))
            .unwrap();
        let rejection = Json::<JsonRpcRequest>::from_request(malformed, &())
            .await
            .unwrap_err();
        assert_eq!(
            json_rejection_response(rejection).error.unwrap().code,
            PARSE_ERROR
        );

        let invalid = Request::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let rejection = Json::<JsonRpcRequest>::from_request(invalid, &())
            .await
            .unwrap_err();
        assert_eq!(
            json_rejection_response(rejection).error.unwrap().code,
            INVALID_REQUEST
        );
    }
}
