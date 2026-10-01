use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{MessageContent, NormalRequest, RequestMessage, VideoInput};

const MAX_SESSIONS: usize = 128;
const SESSION_TTL: Duration = Duration::from_secs(30 * 60);

/// A stored agentic conversation, tool call/response messages included.
#[derive(Clone)]
pub struct AgenticSessionEntry {
    pub messages: Vec<IndexMap<String, MessageContent>>,
    /// Positional with `messages`.
    pub images: Vec<DynamicImage>,
    pub videos: Vec<VideoInput>,
    last_accessed: Instant,
    /// Who the session belongs to; only the same owner (or, with none, only an unscoped caller) reaches it.
    owner: Option<String>,
}

/// Agentic conversation state, keyed by session ID. Also supports content-based matching for clients that don't pass an ID.
pub struct AgenticSessionStore {
    sessions: HashMap<String, AgenticSessionEntry>,
    /// Keyed by `sandbox_key(owner, session_id)`, so an id another owner reuses after expiry starts unapproved.
    approved_agent_sessions: HashMap<String, Instant>,
}

impl Default for AgenticSessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AgenticSessionStore {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            approved_agent_sessions: HashMap::new(),
        }
    }

    pub fn approve_agent_actions(&mut self, sandbox_key: impl Into<String>) {
        self.evict();
        self.approved_agent_sessions
            .insert(sandbox_key.into(), Instant::now());
    }

    pub fn agent_actions_approved(&mut self, sandbox_key: &str) -> bool {
        self.evict();
        let Some(last_accessed) = self.approved_agent_sessions.get_mut(sandbox_key) else {
            return false;
        };
        *last_accessed = Instant::now();
        true
    }

    /// `owner`'s session under `session_id`. Updates `last_accessed`.
    pub fn get(&mut self, session_id: &str, owner: Option<&str>) -> Option<AgenticSessionEntry> {
        let entry = self
            .sessions
            .get_mut(session_id)
            .filter(|entry| entry.owner.as_deref() == owner)?;
        entry.last_accessed = Instant::now();
        Some(entry.clone())
    }

    /// Whether `session_id` names a session that isn't `owner`'s, which `owner` may then neither use nor replace.
    pub fn held_by_other(&self, session_id: &str, owner: Option<&str>) -> bool {
        self.sessions
            .get(session_id)
            .is_some_and(|entry| entry.owner.as_deref() != owner)
    }

    /// Find one of `owner`'s sessions whose user-visible messages (no tool turns) are a prefix of `incoming`.
    pub fn find_by_messages(
        &mut self,
        incoming: &[IndexMap<String, MessageContent>],
        owner: Option<&str>,
    ) -> Option<(String, AgenticSessionEntry)> {
        // Need at least 2 messages (system/user + assistant) to match meaningfully.
        if incoming.len() < 2 {
            return None;
        }

        for (id, entry) in &mut self.sessions {
            if entry.owner.as_deref() != owner {
                continue;
            }
            let stored_visible = user_visible_messages(&entry.messages);
            if stored_visible.len() > incoming.len() {
                continue;
            }

            let matches = stored_visible
                .iter()
                .zip(incoming.iter())
                .all(|(stored, inc)| messages_match(stored, inc));

            if matches && !stored_visible.is_empty() {
                entry.last_accessed = Instant::now();
                return Some((id.clone(), entry.clone()));
            }
        }

        None
    }

    /// Save or update as `owner`'s. Evicts stale entries if needed.
    pub fn save(
        &mut self,
        session_id: String,
        mut entry: AgenticSessionEntry,
        owner: Option<&str>,
    ) {
        self.evict();
        entry.owner = owner.map(str::to_string);
        self.sessions.insert(session_id, entry);
    }

    /// Returns whether `owner`'s session existed.
    pub fn delete(&mut self, session_id: &str, owner: Option<&str>) -> bool {
        if self.held_by_other(session_id, owner) || !self.sessions.contains_key(session_id) {
            return false;
        }
        self.approved_agent_sessions
            .remove(&crate::sandbox_key(owner, session_id));
        self.sessions.remove(session_id).is_some()
    }

    pub fn list_ids(&self, owner: Option<&str>) -> Vec<String> {
        self.sessions
            .iter()
            .filter(|(_, entry)| entry.owner.as_deref() == owner)
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn export(
        &mut self,
        session_id: &str,
        owner: Option<&str>,
    ) -> Result<Option<SerializedSession>> {
        let Some(entry) = self.get(session_id, owner) else {
            return Ok(None);
        };
        Ok(Some(SerializedSession::from_entry(&entry)?))
    }

    /// Replaces any existing entry of `owner`'s with the same ID.
    pub fn import(
        &mut self,
        session_id: String,
        serialized: SerializedSession,
        owner: Option<&str>,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.held_by_other(&session_id, owner),
            "session id `{session_id}` is in use"
        );
        let entry = serialized.into_entry()?;
        self.save(session_id, entry, owner);
        Ok(())
    }

    /// Clone the first `num_turns` complete turns of `src` into `dest`. A turn ends at the first
    /// `role: assistant` message that has no `tool_calls` field. Images and videos are copied as-is.
    pub fn fork(
        &mut self,
        src: &str,
        dest: String,
        num_turns: usize,
        owner: Option<&str>,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.held_by_other(&dest, owner),
            "session id `{dest}` is in use"
        );
        let entry = self
            .get(src, owner)
            .ok_or_else(|| anyhow::anyhow!("source session {src} not found"))?;

        let mut turns_seen = 0;
        let mut cutoff: Option<usize> = None;
        for (i, m) in entry.messages.iter().enumerate() {
            let role = m
                .get("role")
                .and_then(|r| match r {
                    Either::Left(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("");
            if role == "assistant" && !m.contains_key("tool_calls") {
                turns_seen += 1;
                if turns_seen == num_turns {
                    cutoff = Some(i);
                    break;
                }
            }
        }
        let messages = match cutoff {
            Some(i) => entry.messages[..=i].to_vec(),
            None => entry.messages.clone(),
        };
        let forked = AgenticSessionEntry::new(messages, entry.images.clone(), entry.videos.clone());
        self.save(dest, forked, owner);
        Ok(())
    }

    /// Drop expired and over-limit entries.
    fn evict(&mut self) {
        let now = Instant::now();

        self.sessions
            .retain(|_, entry| now.duration_since(entry.last_accessed) < SESSION_TTL);
        self.approved_agent_sessions
            .retain(|_, last_accessed| now.duration_since(*last_accessed) < SESSION_TTL);

        while self.sessions.len() >= MAX_SESSIONS {
            let oldest = self
                .sessions
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest {
                self.sessions.remove(&key);
            } else {
                break;
            }
        }
    }
}

impl AgenticSessionEntry {
    pub fn new(
        messages: Vec<IndexMap<String, MessageContent>>,
        images: Vec<DynamicImage>,
        videos: Vec<VideoInput>,
    ) -> Self {
        Self {
            messages,
            images,
            videos,
            last_accessed: Instant::now(),
            owner: None,
        }
    }
}

/// User-visible messages only, skipping tool call/response messages.
fn user_visible_messages(
    messages: &[IndexMap<String, MessageContent>],
) -> Vec<&IndexMap<String, MessageContent>> {
    messages
        .iter()
        .filter(|msg| !is_tool_message(msg))
        .collect()
}

/// True for tool call / tool response messages.
fn is_tool_message(msg: &IndexMap<String, MessageContent>) -> bool {
    let role = msg
        .get("role")
        .and_then(|r| match r {
            Either::Left(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("");

    if role == "tool" {
        return true;
    }

    // Assistant messages with tool_calls.
    if msg.contains_key("tool_calls") {
        return true;
    }

    false
}

fn messages_match(
    a: &IndexMap<String, MessageContent>,
    b: &IndexMap<String, MessageContent>,
) -> bool {
    a.get("role") == b.get("role")
        && a.get("content") == b.get("content")
        && a.get("tool_calls") == b.get("tool_calls")
}

/// Splice stored tool call/response messages back into an incoming request between matched user-visible messages.
pub fn splice_session_into_request(request: &mut NormalRequest, entry: &AgenticSessionEntry) {
    let incoming = match &mut request.messages {
        RequestMessage::Chat { messages, .. } | RequestMessage::MultimodalChat { messages, .. } => {
            messages
        }
        _ => return,
    };

    let stored = &entry.messages;

    let mut result: Vec<IndexMap<String, MessageContent>> = Vec::new();
    let mut incoming_idx = 0;
    let mut stored_idx = 0;

    while stored_idx < stored.len() && incoming_idx < incoming.len() {
        let stored_msg = &stored[stored_idx];

        if is_tool_message(stored_msg) {
            result.push(stored_msg.clone());
            stored_idx += 1;
        } else {
            let incoming_msg = &incoming[incoming_idx];
            if messages_match(stored_msg, incoming_msg) {
                result.push(stored_msg.clone());
                stored_idx += 1;
                incoming_idx += 1;
            } else {
                // Conversation diverged. Stop splicing.
                break;
            }
        }
    }

    // Drain trailing tool messages after the last matched user-visible message.
    while stored_idx < stored.len() && is_tool_message(&stored[stored_idx]) {
        result.push(stored[stored_idx].clone());
        stored_idx += 1;
    }

    while incoming_idx < incoming.len() {
        result.push(incoming[incoming_idx].clone());
        incoming_idx += 1;
    }

    *incoming = result;

    if !entry.images.is_empty() || !entry.videos.is_empty() {
        request.upgrade_to_multimodal();
        if !entry.images.is_empty() {
            let req_images = request.images_mut();
            *req_images = entry.images.clone();
        }
        if !entry.videos.is_empty() {
            let req_videos = request.videos_mut();
            *req_videos = entry.videos.clone();
        }
    }
}

/// Wire format. Images and video frames are base64 PNGs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SerializedSession {
    #[cfg_attr(feature = "utoipa", schema(value_type = Vec<serde_json::Value>))]
    pub messages: Vec<IndexMap<String, MessageContent>>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub videos: Vec<SerializedVideo>,
    #[serde(default)]
    #[cfg_attr(feature = "utoipa", schema(value_type = Vec<serde_json::Value>))]
    pub files: Vec<crate::files::File>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SerializedVideo {
    pub fps: f64,
    pub frames: Vec<String>,
    pub total_num_frames: usize,
    pub sampled_indices: Vec<usize>,
}

impl SerializedSession {
    pub fn from_entry(entry: &AgenticSessionEntry) -> Result<Self> {
        let images = entry
            .images
            .iter()
            .map(encode_png_base64)
            .collect::<Result<Vec<_>>>()?;

        let videos = entry
            .videos
            .iter()
            .map(SerializedVideo::from_video)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            messages: entry.messages.clone(),
            images,
            videos,
            files: Vec::new(),
        })
    }

    pub fn into_entry(self) -> Result<AgenticSessionEntry> {
        let images = self
            .images
            .iter()
            .map(|s| decode_png_base64(s))
            .collect::<Result<Vec<_>>>()?;

        let videos = self
            .videos
            .into_iter()
            .map(SerializedVideo::into_video)
            .collect::<Result<Vec<_>>>()?;

        Ok(AgenticSessionEntry {
            messages: self.messages,
            images,
            videos,
            last_accessed: Instant::now(),
            owner: None,
        })
    }
}

impl SerializedVideo {
    fn from_video(video: &VideoInput) -> Result<Self> {
        let frames = video
            .frames
            .iter()
            .map(encode_png_base64)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            fps: video.fps,
            frames,
            total_num_frames: video.total_num_frames,
            sampled_indices: video.sampled_indices.clone(),
        })
    }

    fn into_video(self) -> Result<VideoInput> {
        let frames = self
            .frames
            .iter()
            .map(|s| decode_png_base64(s))
            .collect::<Result<Vec<_>>>()?;
        Ok(VideoInput {
            frames,
            fps: self.fps,
            total_num_frames: self.total_num_frames,
            sampled_indices: self.sampled_indices,
        })
    }
}

fn encode_png_base64(img: &DynamicImage) -> Result<String> {
    let buf = crate::images::encode_png(img).context("encoding image as PNG")?;
    Ok(BASE64.encode(&buf))
}

fn decode_png_base64(s: &str) -> Result<DynamicImage> {
    let bytes = BASE64.decode(s).context("base64 decoding image")?;
    image::load_from_memory(&bytes).context("loading image bytes")
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER: Option<&str> = Some("team-a");
    const OTHER: Option<&str> = Some("team-b");

    fn message(role: &str, text: &str) -> IndexMap<String, MessageContent> {
        IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ])
    }

    fn conversation() -> Vec<IndexMap<String, MessageContent>> {
        vec![message("user", "hello"), message("assistant", "hi")]
    }

    #[test]
    fn a_session_is_found_used_and_removed_only_by_its_owner() {
        let mut store = AgenticSessionStore::new();
        let entry = AgenticSessionEntry::new(conversation(), Vec::new(), Vec::new());
        store.save("session_a".to_string(), entry, OWNER);

        let mut incoming = conversation();
        incoming.push(message("user", "and then?"));
        for other in [OTHER, None] {
            assert!(store.find_by_messages(&incoming, other).is_none());
            assert!(store.get("session_a", other).is_none());
            assert!(store.held_by_other("session_a", other));
            assert!(store.list_ids(other).is_empty());
            assert!(!store.delete("session_a", other));
            assert!(
                store
                    .fork("session_a", "session_b".to_string(), 1, other)
                    .is_err()
            );
        }
        assert_eq!(
            store.find_by_messages(&incoming, OWNER).unwrap().0,
            "session_a"
        );
        assert!(!store.held_by_other("session_a", OWNER));
        assert!(store.delete("session_a", OWNER));
    }
}
