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
    /// Who stored it; only the same owner (or, with none, only an unscoped caller) sees it.
    owner: Option<String>,
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

impl StoredFile {
    fn visible(&self, now: Instant, owner: Option<&str>) -> bool {
        self.expires_at >= now && self.owner.as_deref() == owner
    }
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

    /// Stores `file` for `owner` unless another owner's live file holds its id; evicts to stay under both caps.
    pub fn insert(&self, file: File, session_id: Option<String>, owner: Option<&str>) {
        let id = file.id.clone();
        let size = resident_bytes(&file);
        let mut guard = self.inner.write().unwrap();
        let now = Instant::now();
        let held_by_other = guard
            .by_id
            .get(&id)
            .is_some_and(|e| e.expires_at >= now && e.owner.as_deref() != owner);
        if held_by_other {
            return;
        }
        // an expired entry is gone: its tags don't pass to a new body
        let (seq, mut session_ids) = match guard.remove(&id).filter(|e| e.visible(now, owner)) {
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
                owner: owner.map(str::to_string),
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

    /// `None` if missing, expired or another owner's.
    pub fn get(&self, id: &str, owner: Option<&str>) -> Option<Arc<File>> {
        let guard = self.inner.read().unwrap();
        let entry = guard.by_id.get(id)?;
        entry
            .visible(Instant::now(), owner)
            .then(|| Arc::clone(&entry.file))
    }

    /// Whether any live entry holds `id`, whoever owns it.
    pub fn holds(&self, id: &str) -> bool {
        let guard = self.inner.read().unwrap();
        guard
            .by_id
            .get(id)
            .is_some_and(|entry| entry.expires_at >= Instant::now())
    }

    /// Returns true if `owner`'s entry existed.
    pub fn remove(&self, id: &str, owner: Option<&str>) -> bool {
        let mut guard = self.inner.write().unwrap();
        let visible = guard
            .by_id
            .get(id)
            .is_some_and(|entry| entry.visible(Instant::now(), owner));
        visible && guard.remove(id).is_some()
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

    pub fn attach_to_session(
        &self,
        id: &str,
        session_id: impl Into<String>,
        owner: Option<&str>,
    ) -> bool {
        let mut guard = self.inner.write().unwrap();
        let now = Instant::now();
        let Some(entry) = guard.by_id.get_mut(id).filter(|e| e.visible(now, owner)) else {
            return false;
        };
        entry.session_ids.insert(session_id.into());
        entry.expires_at = Instant::now() + self.ttl;
        true
    }

    /// `owner`'s non-expired files tagged with `session_id`, oldest first.
    pub fn list_for_session(&self, session_id: &str, owner: Option<&str>) -> Vec<Arc<File>> {
        let now = Instant::now();
        let guard = self.inner.read().unwrap();
        let mut hits: Vec<&StoredFile> = guard
            .by_id
            .values()
            .filter(|s| s.visible(now, owner) && s.session_ids.contains(session_id))
            .collect();
        hits.sort_by_key(|s| s.seq);
        hits.into_iter().map(|s| Arc::clone(&s.file)).collect()
    }

    /// Every non-expired file of `owner`'s regardless of session, oldest first.
    pub fn list_all(&self, owner: Option<&str>) -> Vec<Arc<File>> {
        let now = Instant::now();
        let guard = self.inner.read().unwrap();
        let mut hits: Vec<&StoredFile> = guard
            .by_id
            .values()
            .filter(|s| s.visible(now, owner))
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
                tool_call_id: None,
            },
            content: FileContent::Text {
                text: Some("hi".into()),
                preview: None,
            },
        }
    }

    #[test]
    fn a_body_stored_over_an_expired_entry_leaves_its_tags_behind() {
        const SHORT_TTL: Duration = Duration::from_millis(5);
        let s = FileStore::with_ttl(SHORT_TTL);
        s.insert(make("file_a"), Some("sess_old".into()), None);
        std::thread::sleep(SHORT_TTL * 2);
        assert!(!s.holds("file_a"));
        s.insert(make("file_a"), Some("sess_new".into()), None);
        assert!(s.list_for_session("sess_old", None).is_empty());
        assert_eq!(s.list_for_session("sess_new", None).len(), 1);
    }

    #[test]
    fn a_file_is_reached_only_by_its_owner() {
        const OWNER: Option<&str> = Some("team-a");
        let s = FileStore::new();
        s.insert(make("file_a"), Some("sess1".into()), OWNER);
        for other in [Some("team-b"), None] {
            assert!(s.get("file_a", other).is_none());
            assert!(s.list_all(other).is_empty());
            assert!(s.list_for_session("sess1", other).is_empty());
            assert!(!s.attach_to_session("file_a", "sess2", other));
            assert!(!s.remove("file_a", other));
        }
        assert!(s.holds("file_a"));
        assert_eq!(s.get("file_a", OWNER).unwrap().as_text(), Some("hi"));
        // another owner storing under the id leaves the first owner's file as it was
        let mut forged = make("file_a");
        forged.content = FileContent::Text {
            text: Some("forged".into()),
            preview: None,
        };
        s.insert(forged, Some("sess2".into()), Some("team-b"));
        assert!(s.get("file_a", Some("team-b")).is_none());
        assert_eq!(s.get("file_a", OWNER).unwrap().as_text(), Some("hi"));
    }

    #[test]
    fn insert_and_get() {
        let s = FileStore::new();
        s.insert(make("file_a"), None, None);
        assert_eq!(s.get("file_a", None).unwrap().as_text(), Some("hi"));
        assert!(s.get("missing", None).is_none());
    }

    #[test]
    fn list_by_session_oldest_first() {
        let s = FileStore::new();
        s.insert(make("file_a"), Some("sess1".into()), None);
        s.insert(make("file_b"), Some("sess1".into()), None);
        s.insert(make("file_c"), Some("sess2".into()), None);
        let list: Vec<_> = s
            .list_for_session("sess1", None)
            .iter()
            .map(|f| f.id.clone())
            .collect();
        assert_eq!(list, vec!["file_a".to_string(), "file_b".to_string()]);
        let list2: Vec<_> = s
            .list_for_session("sess2", None)
            .iter()
            .map(|f| f.id.clone())
            .collect();
        assert_eq!(list2, vec!["file_c".to_string()]);
    }

    #[test]
    fn attach_existing_file_to_multiple_sessions() {
        let s = FileStore::new();
        s.insert(make("file_a"), None, None);
        assert!(s.attach_to_session("file_a", "sess1", None));
        assert!(s.attach_to_session("file_a", "sess2", None));
        assert_eq!(s.list_for_session("sess1", None)[0].id, "file_a");
        assert_eq!(s.list_for_session("sess2", None)[0].id, "file_a");
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
        s.insert(sized("file_a", 4), None, None);
        s.insert(sized("file_b", 4), None, None);
        assert_eq!(s.resident_bytes(), 8);
        s.insert(sized("file_c", 4), None, None);
        assert!(s.get("file_a", None).is_none(), "the oldest goes first");
        assert!(s.get("file_b", None).is_some() && s.get("file_c", None).is_some());
        assert_eq!(s.resident_bytes(), 8);

        // replacing an entry counts its new size, not both
        s.insert(sized("file_c", 6), None, None);
        assert_eq!(s.resident_bytes(), 10);

        // one file over the cap on its own is still kept, alone
        s.insert(sized("file_big", 20), None, None);
        assert_eq!(s.len(), 1);
        assert!(s.get("file_big", None).is_some());
        assert!(s.remove("file_big", None));
        assert_eq!(s.resident_bytes(), 0);
    }

    #[test]
    fn ttl_eviction() {
        let s = FileStore::with_ttl(Duration::from_millis(1));
        s.insert(make("file_a"), None, None);
        std::thread::sleep(Duration::from_millis(5));
        assert!(s.get("file_a", None).is_none());
        let swept = s.cleanup_expired();
        assert_eq!(swept, 1);
        assert!(s.is_empty());
    }
}
