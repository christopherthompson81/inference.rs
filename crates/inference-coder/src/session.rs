//! Sessions on disk: one JSONL file each, a header line then one line per message, only ever appended to.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

const SESSION_EXT: &str = "jsonl";
const HEADER_TYPE: &str = "session";
const MESSAGE_TYPE: &str = "message";
const LABELS_TYPE: &str = "labels";
const FORMAT_VERSION: u64 = 1;
// how much of the first prompt a listing shows
const TITLE_CHARS: usize = 60;

/// Where sessions live unless the caller picks a directory.
pub fn default_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("inference-rs")
        .join("agent")
        .join("sessions")
}

/// A conversation: its messages (system prompt first) and the file they are appended to.
pub struct Session {
    pub id: String,
    pub path: PathBuf,
    pub root: PathBuf,
    pub messages: Vec<Value>,
    /// What each run of tool calls was for, by the id of its first call ([`crate::activity`]).
    pub labels: BTreeMap<String, String>,
    file: File,
}

/// A session as a listing shows it.
#[derive(Debug)]
pub struct SessionInfo {
    pub id: String,
    pub path: PathBuf,
    pub root: PathBuf,
    pub modified: SystemTime,
    pub title: String,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Session {
    /// A new session in `dir` for the project at `root`, starting from `system`.
    pub fn create(dir: &Path, root: &Path, model: &str, system: &str) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let id = format!(
            "{}-{}",
            now_secs(),
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let path = dir.join(format!("{id}.{SESSION_EXT}"));
        let file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)?;
        let mut session = Self {
            id: id.clone(),
            path,
            root: root.to_path_buf(),
            messages: Vec::new(),
            labels: BTreeMap::new(),
            file,
        };
        session.write_line(
            &json!({"type": HEADER_TYPE, "version": FORMAT_VERSION, "id": id,
            "root": root, "model": model, "created": now_secs()}),
        )?;
        session.append(json!({"role": "system", "content": system}))?;
        Ok(session)
    }

    pub fn open(path: &Path) -> Result<Self> {
        let reader = BufReader::new(
            File::open(path).with_context(|| format!("opening {}", path.display()))?,
        );
        let mut header = None;
        let mut messages = Vec::new();
        let mut labels = BTreeMap::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            // a line cut short by a crash is the last one; everything before it is whole
            let Ok(entry) = serde_json::from_str::<Value>(&line) else {
                break;
            };
            match entry.get("type").and_then(Value::as_str) {
                Some(HEADER_TYPE) => header = Some(entry),
                Some(MESSAGE_TYPE) => messages.extend(entry.get("message").cloned()),
                Some(LABELS_TYPE) => {
                    if let Some(entries) = entry.get("labels").and_then(Value::as_object) {
                        for (call, label) in entries {
                            if let Some(label) = label.as_str() {
                                labels.insert(call.clone(), label.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        let Some(header) = header else {
            bail!("{} is not a session file", path.display());
        };
        let field = |key: &str| {
            header
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        Ok(Self {
            id: field("id"),
            path: path.to_path_buf(),
            root: PathBuf::from(field("root")),
            messages,
            labels,
            file: OpenOptions::new().append(true).open(path)?,
        })
    }

    pub fn append(&mut self, message: Value) -> Result<()> {
        self.write_line(&json!({"type": MESSAGE_TYPE, "message": message}))?;
        self.messages.push(message);
        Ok(())
    }

    /// Adds activity labels to the session at `path`, through a handle of its own so a prompt running in the session
    /// keeps appending meanwhile; one short line, which an append writes whole.
    pub fn append_labels(path: &Path, labels: &BTreeMap<String, String>) -> Result<()> {
        let mut line = serde_json::to_string(&json!({"type": LABELS_TYPE, "labels": labels}))?;
        line.push('\n');
        OpenOptions::new()
            .append(true)
            .open(path)
            .and_then(|mut file| file.write_all(line.as_bytes()))
            .with_context(|| format!("writing {}", path.display()))
    }

    fn write_line(&mut self, entry: &Value) -> Result<()> {
        let mut line = serde_json::to_string(entry)?;
        line.push('\n');
        self.file
            .write_all(line.as_bytes())
            .with_context(|| format!("writing {}", self.path.display()))
    }

    /// The sessions in `dir`, newest first.
    pub fn list(dir: &Path) -> Result<Vec<SessionInfo>> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(Vec::new());
        };
        let mut sessions = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(SESSION_EXT) {
                continue;
            }
            let Ok(session) = Self::open(&path) else {
                continue;
            };
            let title = session
                .messages
                .iter()
                .find(|m| m["role"] == "user")
                .and_then(|m| m["content"].as_str())
                .map(|text| {
                    text.lines()
                        .next()
                        .unwrap_or_default()
                        .chars()
                        .take(TITLE_CHARS)
                        .collect()
                })
                .unwrap_or_default();
            sessions.push(SessionInfo {
                id: session.id,
                modified: entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(UNIX_EPOCH),
                path,
                root: session.root,
                title,
            });
        }
        sessions.sort_by_key(|s| std::cmp::Reverse(s.modified));
        Ok(sessions)
    }

    /// The newest session for the project at `root`.
    pub fn latest_for(dir: &Path, root: &Path) -> Result<Option<PathBuf>> {
        Ok(Self::list(dir)?
            .into_iter()
            .find(|s| s.root == root)
            .map(|s| s.path))
    }

    /// The session in `dir` whose id is or starts with `id`.
    pub fn find(dir: &Path, id: &str) -> Result<PathBuf> {
        let matches = Self::list(dir)?
            .into_iter()
            .filter(|s| s.id.starts_with(id))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [one] => Ok(one.path.clone()),
            [] => bail!("no session `{id}` in {}", dir.display()),
            _ => bail!(
                "`{id}` matches {} sessions; give more of the id",
                matches.len()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Session;
    use serde_json::json;

    #[test]
    fn sessions_round_trip_and_list_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("project");
        let mut session = Session::create(dir.path(), &root, "m", "be brief").unwrap();
        session
            .append(json!({"role": "user", "content": "fix the build\nplease"}))
            .unwrap();
        session
            .append(json!({"role": "assistant", "content": "done"}))
            .unwrap();
        let reopened = Session::open(&session.path).unwrap();
        assert_eq!(reopened.messages, session.messages);
        assert_eq!(reopened.root, root);
        let listed = Session::list(dir.path()).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].title, "fix the build");
        assert_eq!(
            Session::latest_for(dir.path(), &root).unwrap(),
            Some(session.path.clone())
        );
        assert_eq!(
            Session::find(dir.path(), &session.id[..12]).unwrap(),
            session.path
        );
    }

    #[test]
    fn a_torn_last_line_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = Session::create(dir.path(), dir.path(), "m", "s").unwrap();
        session
            .append(json!({"role": "user", "content": "hi"}))
            .unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&session.path)
            .and_then(|mut f| std::io::Write::write_all(&mut f, b"{\"type\":\"mess"))
            .unwrap();
        assert_eq!(Session::open(&session.path).unwrap().messages.len(), 2);
    }

    #[test]
    fn labels_written_beside_a_session_read_back_with_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = Session::create(dir.path(), dir.path(), "m", "s").unwrap();
        let labels = [("call_1".to_string(), "Found the failing test".to_string())].into();
        Session::append_labels(&session.path, &labels).unwrap();
        session
            .append(json!({"role": "user", "content": "next"}))
            .unwrap();
        let reopened = Session::open(&session.path).unwrap();
        assert_eq!(reopened.labels, labels);
        assert_eq!(reopened.messages, session.messages);
    }
}
