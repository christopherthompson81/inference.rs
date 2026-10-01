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

/// Trait for caching responses. Each entry belongs to the owner that stored it, and only that owner reaches it.
pub trait ResponseCache: Send + Sync {
    /// Store `owner`'s response object with the given ID
    fn store_response(
        &self,
        id: String,
        response: ResponseResource,
        owner: Option<&str>,
    ) -> Result<()>;

    /// Retrieve `owner`'s response object by ID
    fn get_response(&self, id: &str, owner: Option<&str>) -> Result<Option<ResponseResource>>;

    /// Delete `owner`'s response object, and its conversation, by ID
    fn delete_response(&self, id: &str, owner: Option<&str>) -> Result<bool>;

    /// Store `owner`'s conversation history for a response
    fn store_conversation(
        &self,
        id: String,
        conversation: StoredConversation,
        owner: Option<&str>,
    ) -> Result<()>;

    /// Retrieve `owner`'s conversation history for a response
    fn get_conversation(&self, id: &str, owner: Option<&str>)
    -> Result<Option<StoredConversation>>;

    /// The response `owner` last stored with `session_id`, the only one a follow-up continues that session from.
    fn session_head(&self, session_id: &str, owner: Option<&str>) -> Result<Option<String>>;
}

/// An entry and the owner that stored it.
struct Owned<T> {
    owner: Option<String>,
    value: T,
}

impl<T: Clone> Owned<T> {
    fn new(value: T, owner: Option<&str>) -> Self {
        Self {
            owner: owner.map(str::to_string),
            value,
        }
    }

    fn is_owned_by(&self, owner: Option<&str>) -> bool {
        self.owner.as_deref() == owner
    }

    fn visible_to(&self, owner: Option<&str>) -> Option<T> {
        self.is_owned_by(owner).then(|| self.value.clone())
    }
}

/// In-memory implementation of ResponseCache
pub struct InMemoryResponseCache {
    responses: Arc<RwLock<HashMap<String, Owned<ResponseResource>>>>,
    conversation_histories: Arc<RwLock<HashMap<String, Owned<StoredConversation>>>>,
    /// Keyed by `sandbox_key(owner, session_id)`, so another owner reusing a session id moves no one else's head.
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
    fn store_response(
        &self,
        id: String,
        response: ResponseResource,
        owner: Option<&str>,
    ) -> Result<()> {
        let mut responses = self.responses.write().unwrap();
        responses.insert(id, Owned::new(response, owner));
        Ok(())
    }

    fn get_response(&self, id: &str, owner: Option<&str>) -> Result<Option<ResponseResource>> {
        let responses = self.responses.read().unwrap();
        Ok(responses.get(id).and_then(|entry| entry.visible_to(owner)))
    }

    fn delete_response(&self, id: &str, owner: Option<&str>) -> Result<bool> {
        // Lock order: responses, then conversation_histories, in every method that takes both.
        let mut responses = self.responses.write().unwrap();
        let mut histories = self.conversation_histories.write().unwrap();
        let response_removed = responses.get(id).is_some_and(|e| e.is_owned_by(owner))
            && responses.remove(id).is_some();
        let history_removed = histories.get(id).is_some_and(|e| e.is_owned_by(owner))
            && histories.remove(id).is_some();
        Ok(response_removed || history_removed)
    }

    fn store_conversation(
        &self,
        id: String,
        conversation: StoredConversation,
        owner: Option<&str>,
    ) -> Result<()> {
        if let Some(session_id) = &conversation.session_id {
            let mut heads = self.session_heads.write().unwrap();
            heads.insert(inference_core::sandbox_key(owner, session_id), id.clone());
        }
        let mut histories = self.conversation_histories.write().unwrap();
        histories.insert(id, Owned::new(conversation, owner));
        Ok(())
    }

    fn get_conversation(
        &self,
        id: &str,
        owner: Option<&str>,
    ) -> Result<Option<StoredConversation>> {
        let histories = self.conversation_histories.read().unwrap();
        Ok(histories.get(id).and_then(|entry| entry.visible_to(owner)))
    }

    fn session_head(&self, session_id: &str, owner: Option<&str>) -> Result<Option<String>> {
        let heads = self.session_heads.read().unwrap();
        Ok(heads
            .get(&inference_core::sandbox_key(owner, session_id))
            .cloned())
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

    const OWNER: Option<&str> = Some("team-a");
    const OTHER: Option<&str> = Some("team-b");

    #[test]
    fn a_response_is_reached_and_deleted_only_by_its_owner() {
        let cache = InMemoryResponseCache::new();
        let response =
            ResponseResource::new("test-id".to_string(), "test-model".to_string(), 1234567890)
                .with_status(ResponseStatus::Completed)
                .with_output(vec![OutputItem::message(
                    "msg-1".to_string(),
                    vec![OutputContent::text("Hello".to_string())],
                    ItemStatus::Completed,
                )]);
        cache
            .store_response("test-id".to_string(), response, OWNER)
            .unwrap();

        for other in [OTHER, None] {
            assert!(cache.get_response("test-id", other).unwrap().is_none());
            assert!(!cache.delete_response("test-id", other).unwrap());
        }
        assert_eq!(
            cache.get_response("test-id", OWNER).unwrap().unwrap().id,
            "test-id"
        );
        assert!(cache.delete_response("test-id", OWNER).unwrap());
        assert!(cache.get_response("test-id", OWNER).unwrap().is_none());
    }

    #[test]
    fn a_conversation_is_continued_only_by_its_owner() {
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
            .store_conversation("test-id".to_string(), conversation, OWNER)
            .unwrap();

        assert!(cache.get_conversation("test-id", OTHER).unwrap().is_none());
        let retrieved = cache.get_conversation("test-id", OWNER).unwrap().unwrap();
        assert_eq!(retrieved.messages.len(), 1);
        assert_eq!(retrieved.session_id.as_deref(), Some("session-a"));
    }
}
