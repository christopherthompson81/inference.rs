mod embedder;

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub use embedder::SearchEmbedder;

pub const SEARCH_TOOL_NAME: &str = "inference_search_the_web";
pub const EXTRACT_TOOL_NAME: &str = "inference_website_content_extractor";

/// Callback used to override how search results are gathered. The returned
/// vector must be sorted in decreasing order of relevance.
pub type SearchCallback =
    dyn Fn(&SearchFunctionParameters) -> Result<Vec<SearchResult>> + Send + Sync;

pub fn search_tool_called(name: &str) -> bool {
    name == SEARCH_TOOL_NAME || name == EXTRACT_TOOL_NAME
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct SearchResult {
    pub title: String,
    pub description: String,
    pub url: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchFunctionParameters {
    pub query: String,
}
