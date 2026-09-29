//! Text output configuration for the OpenResponses API.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

/// Text output configuration
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct TextConfig {
    /// Format for text output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
}

/// Text format configuration
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum TextFormat {
    /// Plain text output
    #[serde(rename = "text")]
    Text,
    /// JSON output with optional schema
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// Name for the schema
        name: String,
        /// JSON Schema definition
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<Value>,
        /// Whether to use strict schema validation
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// JSON object output
    #[serde(rename = "json_object")]
    JsonObject,
}
