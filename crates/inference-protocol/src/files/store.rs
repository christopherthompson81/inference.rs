//! In-process file store keyed by id, with per-entry TTL.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use super::{File, FileContent};

/// Per-entry TTL. Matches the agentic session default.
pub const DEFAULT_FILE_TTL: Duration = Duration::from_secs(30 * 60);

/// Hard entry cap. Oldest evicted on insert.
pub const MAX_FILES: usize = 4096;

/// Cap on the bytes the store holds (base64 bodies and text, as kept). Oldest evicted on insert; the newest stays.
pub const MAX_STORE_BYTES: usize = 1 << 30;

const CLEANUP_INTERVAL: Duration = Duration::from_secs(120);

struct StoredFile {
    file: Arc<File>,
    expires_at: Instant,
    session_ids: HashSet<String>,
    /// Insertion order. `list_for_session` returns oldest first.
    seq: u64,
}

#[derive(Clone)]
pub struct FileStore {
    inner: Arc<RwLock<Inner>>,
    ttl: Duration,
    max_bytes: usize,
}

struct Inner {
    by_id: HashMap<String, StoredFile>,
    next_seq: u64,
    /// What the stored bodies take, kept in step with `by_id`.
    resident_bytes: usize,
}

impl Inner {
    fn remove(&mut self, id: &str) -> Option<StoredFile> {
        let removed = self.by_id.remove(id)?;
        self.resident_bytes -= resident_bytes(&removed.file);
        Some(removed)
    }

    fn remove_expired(&mut self, now: Instant) -> usize {
        let expired: Vec<String> = self
            .by_id
            .iter()
            .filter(|(_, entry)| entry.expires_at < now)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            self.remove(id);
        }
        expired.len()
    }
}

fn resident_bytes(file: &File) -> usize {
    match &file.content {
        FileContent::Text { text, preview } => {
            text.as_ref().map_or(0, String::len) + preview.as_ref().map_or(0, String::len)
        }
        FileContent::Binary { data_base64 } => data_base64.as_ref().map_or(0, String::len),
        FileContent::Error { code, message } => code.len() + message.len(),
    }
}

impl FileStore {
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_FILE_TTL)
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self::with_limits(ttl, MAX_STORE_BYTES)
    }

    pub fn with_limits(ttl: Duration, max_bytes: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                by_id: HashMap::new(),
                next_seq: 0,
                resident_bytes: 0,
            })),
            ttl,
            max_bytes,
        }
    }

    /// Evicts expired entries, then the oldest, to stay under `MAX_FILES` and the byte cap.
    pub fn insert(&self, file: File, session_id: Option<String>) {
        let id = file.id.clone();
        let size = resident_bytes(&file);
        let mut guard = self.inner.write().unwrap();
        let (seq, mut session_ids) = match guard.remove(&id) {
            Some(existing) => (existing.seq, existing.session_ids),
            None => {
                let seq = guard.next_seq;
                guard.next_seq += 1;
                (seq, HashSet::new())
            }
        };
        session_ids.extend(session_id);
        guard.resident_bytes += size;
        guard.by_id.insert(
            id.clone(),
            StoredFile {
                file: Arc::new(file),
                expires_at: Instant::now() + self.ttl,
                session_ids,
                seq,
            },
        );
        let over =
            |inner: &Inner| inner.by_id.len() > MAX_FILES || inner.resident_bytes > self.max_bytes;
        if over(&guard) {
            guard.remove_expired(Instant::now());
        }
        while over(&guard) && guard.by_id.len() > 1 {
            let Some(oldest_id) = guard
                .by_id
                .iter()
                .filter(|(other, _)| **other != id)
                .min_by_key(|(_, e)| e.seq)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            guard.remove(&oldest_id);
            tracing::debug!("FileStore evicted `{oldest_id}` to stay under its file and byte caps");
        }
    }

    /// `None` if missing or expired.
    pub fn get(&self, id: &str) -> Option<Arc<File>> {
        let now = Instant::now();
        let guard = self.inner.read().unwrap();
        let entry = guard.by_id.get(id)?;
        if entry.expires_at < now {
            None
        } else {
            Some(Arc::clone(&entry.file))
        }
    }

    /// Returns true if an entry existed.
    pub fn remove(&self, id: &str) -> bool {
        self.inner.write().unwrap().remove(id).is_some()
    }

    /// Refresh the TTL on every file tagged with `session_id`. Call when the session is touched.
    pub fn touch_session(&self, session_id: &str) {
        let new_expiry = std::time::Instant::now() + self.ttl;
        let mut guard = self.inner.write().unwrap();
        for entry in guard.by_id.values_mut() {
            if entry.session_ids.contains(session_id) {
                entry.expires_at = new_expiry;
            }
        }
    }

    pub fn attach_to_session(&self, id: &str, session_id: impl Into<String>) -> bool {
        let mut guard = self.inner.write().unwrap();
        let Some(entry) = guard.by_id.get_mut(id) else {
            return false;
        };
        entry.session_ids.insert(session_id.into());
        entry.expires_at = Instant::now() + self.ttl;
        true
    }

    /// Non-expired files tagged with `session_id`, oldest first.
    pub fn list_for_session(&self, session_id: &str) -> Vec<Arc<File>> {
        let now = Instant::now();
        let guard = self.inner.read().unwrap();
        let mut hits: Vec<&StoredFile> = guard
            .by_id
            .values()
            .filter(|s| s.expires_at >= now && s.session_ids.contains(session_id))
            .collect();
        hits.sort_by_key(|s| s.seq);
        hits.into_iter().map(|s| Arc::clone(&s.file)).collect()
    }

    /// Every non-expired file regardless of session, oldest first.
    pub fn list_all(&self) -> Vec<Arc<File>> {
        let now = Instant::now();
        let guard = self.inner.read().unwrap();
        let mut hits: Vec<&StoredFile> = guard
            .by_id
            .values()
            .filter(|s| s.expires_at >= now)
            .collect();
        hits.sort_by_key(|s| s.seq);
        hits.into_iter().map(|s| Arc::clone(&s.file)).collect()
    }

    pub fn cleanup_expired(&self) -> usize {
        self.inner.write().unwrap().remove_expired(Instant::now())
    }

    /// What the stored bodies take now, as `MAX_STORE_BYTES` counts it.
    pub fn resident_bytes(&self) -> usize {
        self.inner.read().unwrap().resident_bytes
    }

    /// Periodic reaper bound to the store's lifetime via `Weak`. Dies with the last `Arc`.
    pub fn spawn_cleanup_task(&self) {
        let weak = Arc::downgrade(&self.inner);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(CLEANUP_INTERVAL).await;
                let Some(inner) = weak.upgrade() else { break };
                let reaped = inner.write().unwrap().remove_expired(Instant::now());
                if reaped > 0 {
                    tracing::debug!("FileStore reaped {reaped} expired file(s)");
                }
            }
        });
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().by_id.is_empty()
    }
}

impl Default for FileStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{FileContent, FileSource};

    fn make(id: &str) -> File {
        File {
            id: id.into(),
            name: format!("{id}.txt"),
            format: Some("txt".into()),
            mime_type: Some("text/plain".into()),
            bytes: 2,
            created_at: 0,
            purpose: crate::files::FILE_PURPOSE_AGENT_OUTPUT.to_string(),
            source: FileSource {
                tool: "execute_python".into(),
                round: 0,
                turn: 0,
            },
            content: FileContent::Text {
                text: Some("hi".into()),
                preview: None,
            },
        }
    }

    #[test]
    fn insert_and_get() {
        let s = FileStore::new();
        s.insert(make("file_a"), None);
        assert_eq!(s.get("file_a").unwrap().as_text(), Some("hi"));
        assert!(s.get("missing").is_none());
    }

    #[test]
    fn list_by_session_oldest_first() {
        let s = FileStore::new();
        s.insert(make("file_a"), Some("sess1".into()));
        s.insert(make("file_b"), Some("sess1".into()));
        s.insert(make("file_c"), Some("sess2".into()));
        let list: Vec<_> = s
            .list_for_session("sess1")
            .iter()
            .map(|f| f.id.clone())
            .collect();
        assert_eq!(list, vec!["file_a".to_string(), "file_b".to_string()]);
        let list2: Vec<_> = s
            .list_for_session("sess2")
            .iter()
            .map(|f| f.id.clone())
            .collect();
        assert_eq!(list2, vec!["file_c".to_string()]);
    }

    #[test]
    fn attach_existing_file_to_multiple_sessions() {
        let s = FileStore::new();
        s.insert(make("file_a"), None);
        assert!(s.attach_to_session("file_a", "sess1"));
        assert!(s.attach_to_session("file_a", "sess2"));
        assert_eq!(s.list_for_session("sess1")[0].id, "file_a");
        assert_eq!(s.list_for_session("sess2")[0].id, "file_a");
    }

    fn sized(id: &str, len: usize) -> File {
        File {
            content: FileContent::Text {
                text: Some("x".repeat(len)),
                preview: None,
            },
            ..make(id)
        }
    }

    #[test]
    fn the_byte_cap_evicts_the_oldest_and_keeps_the_newest() {
        let s = FileStore::with_limits(DEFAULT_FILE_TTL, 10);
        s.insert(sized("file_a", 4), None);
        s.insert(sized("file_b", 4), None);
        assert_eq!(s.resident_bytes(), 8);
        s.insert(sized("file_c", 4), None);
        assert!(s.get("file_a").is_none(), "the oldest goes first");
        assert!(s.get("file_b").is_some() && s.get("file_c").is_some());
        assert_eq!(s.resident_bytes(), 8);

        // replacing an entry counts its new size, not both
        s.insert(sized("file_c", 6), None);
        assert_eq!(s.resident_bytes(), 10);

        // one file over the cap on its own is still kept, alone
        s.insert(sized("file_big", 20), None);
        assert_eq!(s.len(), 1);
        assert!(s.get("file_big").is_some());
        assert!(s.remove("file_big"));
        assert_eq!(s.resident_bytes(), 0);
    }

    #[test]
    fn ttl_eviction() {
        let s = FileStore::with_ttl(Duration::from_millis(1));
        s.insert(make("file_a"), None);
        std::thread::sleep(Duration::from_millis(5));
        assert!(s.get("file_a").is_none());
        let swept = s.cleanup_expired();
        assert_eq!(swept, 1);
        assert!(s.is_empty());
    }
}
