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
// a content part naming an image file by path, kept in the session file instead of the image's bytes
const IMAGE_FILE_PART: &str = "image_file";
const IMAGE_URL_PART: &str = "image_url";
// what a request says in place of an image file that is gone by the time the request is built
const MISSING_IMAGE: &str = "(this image file no longer exists)";
// beside a session file, the copies of the images its tool calls returned
const IMAGES_EXT: &str = "images";
/// The file extensions a read shows the model as an image, with their media types.
pub const IMAGE_TYPES: [(&str, &str); 6] = [
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("gif", "image/gif"),
    ("webp", "image/webp"),
    ("bmp", "image/bmp"),
];
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
    /// When it was folded, in seconds since the epoch; unset in sessions from before it was kept.
    pub at: Option<u64>,
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
                            at: entry.get("at").and_then(Value::as_u64),
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
        let at = now_secs();
        self.write_line(
            &json!({"type": COMPACTION_TYPE, "summary": summary, "start": start,
            "through": through, "at": at}),
        )?;
        self.compaction = Some(Compaction {
            summary,
            start,
            through,
            at: Some(at),
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
                .and_then(|m| text_of(&m["content"]))
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

    /// Deletes the session at `path` with the copies it kept of the files it changed and of the images it was shown.
    pub fn delete(path: &Path) -> Result<()> {
        std::fs::remove_file(path).with_context(|| format!("deleting {}", path.display()))?;
        for kept in [
            path.with_extension(crate::changes::CHANGES_EXT),
            path.with_extension(IMAGES_EXT),
        ] {
            if kept.exists() {
                std::fs::remove_dir_all(&kept)
                    .with_context(|| format!("deleting {}", kept.display()))?;
            }
        }
        Ok(())
    }

    /// A copy of the image a tool call returned, kept beside the session as it was when read: a later request shows the
    /// model the same picture (and its prefix stays cached) though the file is overwritten or deleted.
    pub fn keep_image(&self, call_id: &str, index: usize, path: &Path) -> Result<PathBuf> {
        let dir = self.path.with_extension(IMAGES_EXT);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();
        // call ids are the model's, so only their safe characters name the file
        let call = call_id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect::<String>();
        let kept = dir.join(format!("{call}-{index}.{ext}"));
        std::fs::create_dir_all(&dir)
            .and_then(|()| std::fs::copy(path, &kept))
            .with_context(|| format!("keeping {} for the session", path.display()))?;
        Ok(kept)
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

/// A user turn's content: its text, then its images (data or http URLs) when it has any.
pub fn user_content(text: &str, images: &[String]) -> Value {
    if images.is_empty() {
        return Value::String(text.to_string());
    }
    let images = images
        .iter()
        .map(|url| json!({"type": "image_url", "image_url": {"url": url}}));
    std::iter::once(json!({"type": "text", "text": text}))
        .chain(images)
        .collect()
}

/// The media type of an image file by its extension, for those a read shows as images.
pub fn image_type(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    IMAGE_TYPES
        .iter()
        .find(|(known, _)| *known == ext)
        .map(|(_, media)| *media)
}

/// An image a tool call returned: the call, the file (the session's copy) and the name the model saw it by.
pub struct ToolImage {
    pub call_id: String,
    pub path: PathBuf,
    pub shown_as: String,
}

/// The user turn that shows the model the images tool calls returned. Each part names the file, not its bytes, so the
/// session stays small; [`with_image_data`] reads them in when a request is built.
pub fn tool_images_message(images: &[ToolImage]) -> Value {
    let names = images
        .iter()
        .map(|image| image.shown_as.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let text = json!({"type": "text", "text": format!("The images the tool calls above returned: {names}")});
    let parts = images.iter().map(
        |image| json!({"type": IMAGE_FILE_PART, "path": image.path, "tool_call_id": image.call_id}),
    );
    json!({"role": "user", "content": std::iter::once(text).chain(parts).collect::<Vec<_>>()})
}

/// Whether `message` is a turn of tool images rather than one the user wrote.
pub fn is_tool_images(message: &Value) -> bool {
    message["role"] == "user"
        && message["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(|part| part["type"] == IMAGE_FILE_PART))
}

/// An image file as a `data:` URL.
pub fn image_data_url(path: &Path) -> Option<String> {
    use base64::Engine;
    let media = image_type(path)?;
    let bytes = std::fs::read(path).ok()?;
    let data = base64::engine::general_purpose::STANDARD.encode(bytes);
    Some(format!("data:{media};base64,{data}"))
}

/// `message` with each image file part read in as an `image_url` part (one whose file is gone says so instead); with
/// `keep_ids` each keeps the call that returned it, for a page placing it under that call.
pub fn with_image_data(message: &Value, keep_ids: bool) -> Value {
    let Some(parts) = message["content"].as_array() else {
        return message.clone();
    };
    if !parts.iter().any(|part| part["type"] == IMAGE_FILE_PART) {
        return message.clone();
    }
    let parts = parts
        .iter()
        .map(|part| {
            if part["type"] != IMAGE_FILE_PART {
                return part.clone();
            }
            let path = Path::new(part["path"].as_str().unwrap_or_default());
            let mut image = match image_data_url(path) {
                Some(url) => json!({"type": IMAGE_URL_PART, IMAGE_URL_PART: {"url": url}}),
                None => {
                    json!({"type": "text", "text": format!("{}: {MISSING_IMAGE}", path.display())})
                }
            };
            if keep_ids {
                image["tool_call_id"] = part["tool_call_id"].clone();
            }
            image
        })
        .collect::<Vec<_>>();
    let mut message = message.clone();
    message["content"] = Value::Array(parts);
    message
}

/// `message` with each image file part as a line of text naming the file, for a request that needs no pixels.
pub fn without_image_files(message: &Value) -> Value {
    let Some(parts) = message["content"].as_array() else {
        return message.clone();
    };
    let parts = parts
        .iter()
        .map(|part| match part["type"] == IMAGE_FILE_PART {
            true => json!({"type": "text", "text": format!("[image {}]", part["path"].as_str().unwrap_or_default())}),
            false => part.clone(),
        })
        .collect::<Vec<_>>();
    let mut message = message.clone();
    message["content"] = Value::Array(parts);
    message
}

/// A message content's text: the string itself, or the first text part of a list of parts.
pub fn text_of(content: &Value) -> Option<&str> {
    match content {
        Value::Array(parts) => parts
            .iter()
            .find(|part| part["type"] == "text")
            .and_then(|part| part["text"].as_str()),
        content => content.as_str(),
    }
}

/// The message a summary of folded history is sent as.
pub fn summary_message(summary: &str) -> Value {
    json!({"role": "user", "content": format!("{SUMMARY_HEADER}\n\n{summary}")})
}

#[cfg(test)]
mod tests {
    use super::{
        Session, ToolImage, is_tool_images, tool_images_message, user_content, with_image_data,
        without_image_files,
    };
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
        let kept = session.path.with_extension(crate::changes::CHANGES_EXT);
        std::fs::create_dir_all(&kept).unwrap();
        let shot = dir.path().join("shot.png");
        std::fs::write(&shot, b"png").unwrap();
        let image = session.keep_image("call:1", 0, &shot).unwrap();
        assert!(image.ends_with("call1-0.png"), "{}", image.display());
        Session::delete(&session.path).unwrap();
        assert!(!session.path.exists() && !kept.exists() && !image.exists());
        assert!(Session::list(dir.path()).unwrap().is_empty());
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
        assert!(reopened.compaction.unwrap().at.is_some());
        assert!(session.compact("x".into(), 9).is_err());
    }

    #[test]
    fn a_prompt_with_images_lists_them_after_its_text_and_titles_by_the_text() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = Session::create(dir.path(), dir.path(), "m", "s").unwrap();
        let content = user_content("match this mockup", &["data:image/png;base64,AAAA".into()]);
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(user_content("plain", &[]), "plain");
        session
            .append(json!({"role": "user", "content": content}))
            .unwrap();
        assert_eq!(
            Session::list(dir.path()).unwrap()[0].title,
            "match this mockup"
        );
    }

    #[test]
    fn a_tool_image_is_read_in_when_sent_and_noted_once_its_file_is_gone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.png");
        std::fs::write(&path, b"png").unwrap();
        let image = ToolImage {
            call_id: "c1".into(),
            path: path.clone(),
            shown_as: "shot.png".into(),
        };
        let message = tool_images_message(&[image]);
        assert!(is_tool_images(&message));
        let sent = with_image_data(&message, false);
        assert_eq!(
            sent["content"][1]["image_url"]["url"],
            "data:image/png;base64,cG5n"
        );
        assert!(sent["content"][1].get("tool_call_id").is_none());
        assert_eq!(
            with_image_data(&message, true)["content"][1]["tool_call_id"],
            "c1"
        );
        std::fs::remove_file(&path).unwrap();
        let gone = with_image_data(&message, true);
        assert_eq!(gone["content"][1]["type"], "text");
        assert_eq!(gone["content"][1]["tool_call_id"], "c1");
        let named = without_image_files(&message);
        assert!(
            named["content"][1]["text"]
                .as_str()
                .unwrap()
                .contains("shot.png")
        );
    }
}
