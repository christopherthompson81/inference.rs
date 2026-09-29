use either::Either;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type LlguidanceGrammar = llguidance::api::TopLevelGrammar;

#[derive(Clone, Serialize, Deserialize)]
/// Control the constraint with llguidance.
pub enum Constraint {
    Regex(String),
    Lark(String),
    JsonSchema(serde_json::Value),
    Llguidance(LlguidanceGrammar),
    None,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
/// Image generation response format
#[serde(rename_all = "snake_case")]
pub enum ImageGenerationResponseFormat {
    #[serde(alias = "Url")]
    Url,
    #[serde(alias = "B64Json")]
    B64Json,
}

pub type MessageContent = Either<String, Vec<IndexMap<String, Value>>>;

/// Reasoning effort passed to chat templates. `none` is a parse-only alias for `off`.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disable reasoning.
    Off,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Maximum reasoning effort.
    XHigh,
}

impl ReasoningEffort {
    /// Return the canonical wire value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }

    /// Return whether this effort disables reasoning.
    pub const fn is_off(self) -> bool {
        matches!(self, Self::Off)
    }
}

#[cfg(feature = "utoipa")]
impl utoipa::PartialSchema for ReasoningEffort {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{schema::SchemaType, ObjectBuilder, RefOr, Schema, Type};

        RefOr::T(Schema::Object(
            ObjectBuilder::new()
                .schema_type(SchemaType::Type(Type::String))
                .description(Some(
                    "Reasoning effort. `none` aliases `off`; `max` aliases `xhigh`.",
                ))
                .enum_values(Some(
                    ["off", "none", "low", "medium", "high", "xhigh", "max"]
                        .into_iter()
                        .map(|value| Value::String(value.to_string()))
                        .collect::<Vec<_>>(),
                ))
                .build(),
        ))
    }
}

#[cfg(feature = "utoipa")]
impl utoipa::ToSchema for ReasoningEffort {}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "invalid reasoning effort `{value}`; expected one of: off, none, low, medium, high, xhigh, max"
)]
/// Error returned when a reasoning effort string is invalid.
pub struct ReasoningEffortParseError {
    value: String,
}

impl std::str::FromStr for ReasoningEffort {
    type Err = ReasoningEffortParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => Ok(Self::Off),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" | "max" => Ok(Self::XHigh),
            _ => Err(ReasoningEffortParseError {
                value: value.to_string(),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for ReasoningEffort {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Default thinking toggle when neither reasoning control is provided.
pub const DEFAULT_ENABLE_THINKING: bool = true;

/// Effective reasoning controls after validating their relationship.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedReasoningControls {
    /// Effective thinking toggle.
    pub enable_thinking: bool,
    /// Explicit effort, if the caller selected one.
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// Contradictory reasoning controls supplied by a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReasoningControlError {
    #[error("reasoning effort `off` conflicts with enable_thinking=true")]
    OffWithThinkingEnabled,
    #[error("reasoning effort `{0}` conflicts with enable_thinking=false")]
    EffortWithThinkingDisabled(ReasoningEffort),
}

/// Validate reasoning controls and derive the effective thinking toggle.
pub fn resolve_reasoning_controls(
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<ResolvedReasoningControls, ReasoningControlError> {
    let enable_thinking = match (enable_thinking, reasoning_effort) {
        (Some(true), Some(ReasoningEffort::Off)) => {
            return Err(ReasoningControlError::OffWithThinkingEnabled)
        }
        (Some(false), Some(effort)) if !effort.is_off() => {
            return Err(ReasoningControlError::EffortWithThinkingDisabled(effort))
        }
        (_, Some(effort)) => !effort.is_off(),
        (enable_thinking, None) => enable_thinking.unwrap_or(DEFAULT_ENABLE_THINKING),
    };

    Ok(ResolvedReasoningControls {
        enable_thinking,
        reasoning_effort,
    })
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum SearchContextSize {
    #[serde(rename = "low")]
    Low,
    #[default]
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ApproximateUserLocation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum WebSearchUserLocation {
    #[serde(rename = "approximate")]
    Approximate {
        approximate: ApproximateUserLocation,
    },
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct WebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<SearchContextSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<WebSearchUserLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<WebSearchFilters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_web_access: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_token_budget: Option<WebSearchReturnTokenBudget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_content_types: Option<Vec<WebSearchContentType>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_settings: Option<WebSearchImageSettings>,
    /// Override the description for the search tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_description: Option<String>,
    /// Override the description for the extraction tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extract_description: Option<String>,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct WebSearchFilters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchContentType {
    Text,
    Image,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchReturnTokenBudget {
    Default,
    Unlimited,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct WebSearchImageSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_response_formats_use_openai_spellings_and_accept_the_old_ones() {
        let cases = [
            ("url", ImageGenerationResponseFormat::Url),
            ("b64_json", ImageGenerationResponseFormat::B64Json),
            ("Url", ImageGenerationResponseFormat::Url),
            ("B64Json", ImageGenerationResponseFormat::B64Json),
        ];
        for (spelling, format) in cases {
            let parsed: ImageGenerationResponseFormat =
                serde_json::from_value(serde_json::json!(spelling)).unwrap();
            assert_eq!(parsed, format, "{spelling}");
        }
        assert_eq!(
            serde_json::to_value(ImageGenerationResponseFormat::B64Json).unwrap(),
            serde_json::json!("b64_json")
        );
    }

    #[test]
    fn reasoning_effort_parsing_is_canonical() {
        let cases = [
            ("off", ReasoningEffort::Off),
            (" none ", ReasoningEffort::Off),
            ("LOW", ReasoningEffort::Low),
            ("Medium", ReasoningEffort::Medium),
            ("high", ReasoningEffort::High),
            (" XHIGH\n", ReasoningEffort::XHigh),
            ("max", ReasoningEffort::XHigh),
        ];

        for (input, expected) in cases {
            assert_eq!(input.parse::<ReasoningEffort>().unwrap(), expected);
        }
        assert_eq!(
            "extreme"
                .parse::<ReasoningEffort>()
                .unwrap_err()
                .to_string(),
            "invalid reasoning effort `extreme`; expected one of: off, none, low, medium, high, xhigh, max"
        );
        assert_eq!(
            serde_json::from_str::<ReasoningEffort>(r#"" NoNe ""#).unwrap(),
            ReasoningEffort::Off
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::XHigh).unwrap(),
            r#""xhigh""#
        );
    }

    #[test]
    fn reasoning_controls_resolve_consistently() {
        let cases = [
            (None, None, true, None),
            (Some(true), None, true, None),
            (Some(false), None, false, None),
            (
                None,
                Some(ReasoningEffort::Off),
                false,
                Some(ReasoningEffort::Off),
            ),
            (
                Some(false),
                Some(ReasoningEffort::Off),
                false,
                Some(ReasoningEffort::Off),
            ),
            (
                None,
                Some(ReasoningEffort::Low),
                true,
                Some(ReasoningEffort::Low),
            ),
            (
                Some(true),
                Some(ReasoningEffort::XHigh),
                true,
                Some(ReasoningEffort::XHigh),
            ),
        ];

        for (enable_thinking, reasoning_effort, expected_enabled, expected_effort) in cases {
            let resolved = resolve_reasoning_controls(enable_thinking, reasoning_effort).unwrap();
            assert_eq!(resolved.enable_thinking, expected_enabled);
            assert_eq!(resolved.reasoning_effort, expected_effort);
        }

        assert_eq!(
            resolve_reasoning_controls(Some(true), Some(ReasoningEffort::Off)),
            Err(ReasoningControlError::OffWithThinkingEnabled)
        );
        assert_eq!(
            resolve_reasoning_controls(Some(false), Some(ReasoningEffort::High)),
            Err(ReasoningControlError::EffortWithThinkingDisabled(
                ReasoningEffort::High
            ))
        );
    }
}
