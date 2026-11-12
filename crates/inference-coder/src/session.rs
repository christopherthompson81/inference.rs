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
const COMPACTION_TYPE: &str = "compaction";
const REASONING: &str = "reasoning_content";
// how the folded history opens the model's view of the session, and the reply that closes it before a user turn
const SUMMARY_HEADER: &str =
    "[Summary of the earlier part of this session, which no longer fits the context]";
pub const SUMMARY_ACK: &str = "Understood; continuing from that summary.";
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

/// The earlier history folded into a summary: the model sees the summary, then `messages[start..]`, without the
/// reasoning of those before `through` (the session's length when it was folded).
#[derive(Clone, Debug, PartialEq)]
pub struct Compaction {
    pub summary: String,
    pub start: usize,
    pub through: usize,
}

/// A conversation: its messages (system prompt first) and the file they are appended to.
pub struct Session {
    pub id: String,
    pub path: PathBuf,
    pub root: PathBuf,
    /// The whole transcript, as shown; [`Session::context`] is what the model is sent.
    pub messages: Vec<Value>,
    pub compaction: Option<Compaction>,
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
            compaction: None,
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
        let mut compaction = None;
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
                Some(COMPACTION_TYPE) => {
                    let summary = entry.get("summary").and_then(Value::as_str);
                    let start = entry.get("start").and_then(Value::as_u64);
                    if let (Some(summary), Some(start)) = (summary, start) {
                        let start = usize::try_from(start)?;
                        let through = entry.get("through").and_then(Value::as_u64);
                        compaction = Some(Compaction {
                            summary: summary.to_string(),
                            start,
                            through: through.map_or(Ok(start), usize::try_from)?,
                        });
                    }
                }
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
            compaction,
            labels,
            file: OpenOptions::new().append(true).open(path)?,
        })
    }

    pub fn append(&mut self, message: Value) -> Result<()> {
        self.write_line(&json!({"type": MESSAGE_TYPE, "message": message}))?;
        self.messages.push(message);
        Ok(())
    }

    /// Folds `messages[1..start]` (and any earlier summary) into `summary`, and drops the reasoning of the kept
    /// messages from what the model sees; the transcript keeps every message whole.
    pub fn compact(&mut self, summary: String, start: usize) -> Result<()> {
        if start == 0 || start > self.messages.len() {
            bail!("a compaction starting at message {start} is outside the session");
        }
        let through = self.messages.len();
        self.write_line(
            &json!({"type": COMPACTION_TYPE, "summary": summary, "start": start,
            "through": through}),
        )?;
        self.compaction = Some(Compaction {
            summary,
            start,
            through,
        });
        Ok(())
    }

    /// Where the history the model still sees in full begins: past the system prompt and any compaction.
    pub fn context_start(&self) -> usize {
        self.compaction.as_ref().map_or(1, |c| c.start)
    }

    /// What the model is sent: the system prompt, the summary of what was folded, then the messages since.
    pub fn context(&self) -> Vec<Value> {
        let mut context = self.messages.iter().take(1).cloned().collect::<Vec<_>>();
        if let Some(compaction) = &self.compaction {
            context.push(summary_message(&compaction.summary));
            // a summary is a user turn, so a user message after it needs a reply between them
            if self
                .messages
                .get(compaction.start)
                .is_some_and(|m| m["role"] == "user")
            {
                context.push(json!({"role": "assistant", "content": SUMMARY_ACK}));
            }
        }
        let through = self.compaction.as_ref().map_or(0, |c| c.through);
        for (i, message) in self.messages.iter().enumerate().skip(self.context_start()) {
            let mut message = message.clone();
            // the summary stands for the thinking that led here; later turns keep theirs, so their prefix still caches
            if i < through
                && let Some(message) = message.as_object_mut()
            {
                message.remove(REASONING);
            }
            context.push(message);
        }
        context
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

/// The message a summary of folded history is sent as.
pub fn summary_message(summary: &str) -> Value {
    json!({"role": "user", "content": format!("{SUMMARY_HEADER}\n\n{summary}")})
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

    #[test]
    fn a_compaction_folds_the_context_but_keeps_the_transcript() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = Session::create(dir.path(), dir.path(), "m", "s").unwrap();
        for (role, content) in [("user", "a"), ("assistant", "b"), ("user", "c")] {
            session
                .append(json!({"role": role, "content": content}))
                .unwrap();
        }
        session
            .append(json!({"role": "assistant", "content": "c2", "reasoning_content": "thought"}))
            .unwrap();
        session.compact("asked a, answered b".into(), 3).unwrap();
        session
            .append(json!({"role": "assistant", "content": "d", "reasoning_content": "kept"}))
            .unwrap();
        let reopened = Session::open(&session.path).unwrap();
        assert_eq!(reopened.messages.len(), 6);
        let context = reopened.context();
        let roles = context
            .iter()
            .map(|m| m["role"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        // system, summary, its acknowledgement before the kept user turn, then c, c2 and d
        assert_eq!(
            roles,
            [
                "system",
                "user",
                "assistant",
                "user",
                "assistant",
                "assistant"
            ]
        );
        assert!(context[1]["content"].as_str().unwrap().contains("asked a"));
        assert_eq!(context[3]["content"], "c");
        // the thinking folded with the summary goes; the turn after the fold keeps its own
        assert!(context[4].get("reasoning_content").is_none());
        assert_eq!(context[5]["reasoning_content"], "kept");
        assert_eq!(reopened.messages[4]["reasoning_content"], "thought");
        assert!(session.compact("x".into(), 9).is_err());
    }
}
