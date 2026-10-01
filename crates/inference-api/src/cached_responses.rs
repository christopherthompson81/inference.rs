//! ## Response caching functionality for the Responses API.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::{Arc, RwLock};

use crate::openai::Message;
use crate::responses_types::ResponseResource;

/// A stored reply's conversation, which `previous_response_id` continues.
#[derive(Debug, Clone, Default)]
pub struct StoredConversation {
    pub messages: Vec<Message>,
    /// The agent session the run used, so a follow-up continues it by id rather than by matching messages.
    pub session_id: Option<String>,
}

/// Trait for caching responses
pub trait ResponseCache: Send + Sync {
    /// Store a response object with the given ID
    fn store_response(&self, id: String, response: ResponseResource) -> Result<()>;

    /// Retrieve a response object by ID
    fn get_response(&self, id: &str) -> Result<Option<ResponseResource>>;

    /// Delete a response object by ID
    fn delete_response(&self, id: &str) -> Result<bool>;

    /// Store conversation history for a response
    fn store_conversation(&self, id: String, conversation: StoredConversation) -> Result<()>;

    /// Retrieve conversation history for a response
    fn get_conversation(&self, id: &str) -> Result<Option<StoredConversation>>;

    /// The response last stored with `session_id`, the only one a follow-up continues that session from.
    fn session_head(&self, session_id: &str) -> Result<Option<String>>;
}

/// In-memory implementation of ResponseCache
pub struct InMemoryResponseCache {
    responses: Arc<RwLock<HashMap<String, ResponseResource>>>,
    conversation_histories: Arc<RwLock<HashMap<String, StoredConversation>>>,
    session_heads: Arc<RwLock<HashMap<String, String>>>,
}

impl InMemoryResponseCache {
    /// Create a new in-memory cache
    pub fn new() -> Self {
        Self {
            responses: Arc::new(RwLock::new(HashMap::new())),
            conversation_histories: Arc::new(RwLock::new(HashMap::new())),
            session_heads: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryResponseCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseCache for InMemoryResponseCache {
    fn store_response(&self, id: String, response: ResponseResource) -> Result<()> {
        let mut responses = self.responses.write().unwrap();
        responses.insert(id, response);
        Ok(())
    }

    fn get_response(&self, id: &str) -> Result<Option<ResponseResource>> {
        let responses = self.responses.read().unwrap();
        Ok(responses.get(id).cloned())
    }

    fn delete_response(&self, id: &str) -> Result<bool> {
        // IMPORTANT: Lock ordering must be maintained to prevent deadlocks.
        // Order: responses -> conversation_histories
        // All methods that acquire multiple locks must follow this order.
        //
        // We acquire all locks before any modifications to ensure atomicity.
        // The locks are released in reverse order when dropped at end of scope.
        let mut responses = self.responses.write().unwrap();
        let mut histories = self.conversation_histories.write().unwrap();

        let response_removed = responses.remove(id).is_some();
        let history_removed = histories.remove(id).is_some();

        Ok(response_removed || history_removed)
    }

    fn store_conversation(&self, id: String, conversation: StoredConversation) -> Result<()> {
        if let Some(session_id) = &conversation.session_id {
            let mut heads = self.session_heads.write().unwrap();
            heads.insert(session_id.clone(), id.clone());
        }
        let mut histories = self.conversation_histories.write().unwrap();
        histories.insert(id, conversation);
        Ok(())
    }

    fn get_conversation(&self, id: &str) -> Result<Option<StoredConversation>> {
        let histories = self.conversation_histories.read().unwrap();
        Ok(histories.get(id).cloned())
    }

    fn session_head(&self, session_id: &str) -> Result<Option<String>> {
        let heads = self.session_heads.read().unwrap();
        Ok(heads.get(session_id).cloned())
    }
}

/// Global response cache instance
pub static RESPONSE_CACHE: LazyLock<Arc<dyn ResponseCache>> =
    LazyLock::new(|| Arc::new(InMemoryResponseCache::new()));

/// Helper function to get the global cache instance
pub fn get_response_cache() -> Arc<dyn ResponseCache> {
    RESPONSE_CACHE.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses_types::{ItemStatus, OutputContent, OutputItem, ResponseStatus};

    #[test]
    fn test_in_memory_cache() {
        let cache = InMemoryResponseCache::new();

        // Create a test response
        let response =
            ResponseResource::new("test-id".to_string(), "test-model".to_string(), 1234567890)
                .with_status(ResponseStatus::Completed)
                .with_output(vec![OutputItem::message(
                    "msg-1".to_string(),
                    vec![OutputContent::text("Hello".to_string())],
                    ItemStatus::Completed,
                )]);

        // Store and retrieve
        cache
            .store_response("test-id".to_string(), response.clone())
            .unwrap();
        let retrieved = cache.get_response("test-id").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test-id");

        // Delete
        let deleted = cache.delete_response("test-id").unwrap();
        assert!(deleted);
        let retrieved = cache.get_response("test-id").unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_conversation_history() {
        let cache = InMemoryResponseCache::new();

        let messages = vec![Message {
            content: Some(crate::openai::MessageContent::from_text(
                "Hello".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let conversation = StoredConversation {
            messages,
            session_id: Some("session-a".to_string()),
        };
        cache
            .store_conversation("test-id".to_string(), conversation)
            .unwrap();

        let retrieved = cache.get_conversation("test-id").unwrap().unwrap();
        assert_eq!(retrieved.messages.len(), 1);
        assert_eq!(retrieved.session_id.as_deref(), Some("session-a"));
    }
}
