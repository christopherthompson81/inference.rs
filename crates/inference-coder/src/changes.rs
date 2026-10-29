//! What a session changed in its project. Each file the agent writes is kept as it was before its first write; once a
//! command has run, other files modified since tracking began count too, compared with git's HEAD when tracked there.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde::Serialize;
use similar::{ChangeTag, TextDiff};

const CHANGES_EXT: &str = "changes";
const INDEX_FILE: &str = "index.json";
// its mtime is when tracking began, on the same clock as the project's file times
const STARTED_FILE: &str = "started";
// files past this size are neither kept nor diffed
const MAX_FILE_BYTES: u64 = 4 << 20;
// a scan gives up after this many entries, so a huge folder can't stall a listing
const MAX_SCAN_ENTRIES: usize = 200_000;
const CONTEXT_LINES: usize = 3;
// git's rule: a NUL in the first 8000 bytes makes a file binary
const BINARY_SNIFF_BYTES: usize = 8000;
// past this a diff settles for a coarser answer rather than stall on a huge rewrite
const DIFF_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Default, Serialize, inference_json::FromJson)]
struct Index {
    /// Path (relative when inside the project) to its kept copy's file name, or `None` when there was no file.
    originals: BTreeMap<String, Option<String>>,
    commands_ran: bool,
}

/// A session's changes, kept beside its session file.
pub struct Changes {
    root: PathBuf,
    dir: PathBuf,
    index: Index,
    started: Option<SystemTime>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeStatus {
    Added,
    Modified,
    Deleted,
    /// Modified by a command, with no earlier copy to compare against.
    NoOriginal,
}

#[derive(Debug, Serialize)]
pub struct FileChange {
    pub path: String,
    pub status: ChangeStatus,
    pub added: usize,
    pub removed: usize,
    pub binary: bool,
}

/// One file's diff; an unchanged file comes back whole as context, with no status.
#[derive(Debug, Serialize)]
pub struct FileDiff {
    pub path: String,
    pub status: Option<ChangeStatus>,
    pub added: usize,
    pub removed: usize,
    pub binary: bool,
    pub too_large: bool,
    pub hunks: Vec<Vec<DiffLine>>,
}

#[derive(Debug, Serialize)]
pub struct DiffLine {
    pub kind: LineKind,
    /// 1-based line numbers on each side.
    pub old: Option<usize>,
    pub new: Option<usize>,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LineKind {
    Context,
    Added,
    Removed,
}

/// A directory entry, as a file tree shows it.
#[derive(Debug, Serialize)]
pub struct Entry {
    pub name: String,
    pub dir: bool,
}

enum Content {
    Missing,
    TooLarge,
    Bytes(Vec<u8>),
}

enum Original {
    Kept(Content),
    Unknown,
}

fn read(path: &Path) -> Result<Content> {
    match fs::metadata(path) {
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(Content::Missing),
        Err(error) => Err(error).with_context(|| path.display().to_string()),
        Ok(meta) if !meta.is_file() => Ok(Content::Missing),
        Ok(meta) if meta.len() > MAX_FILE_BYTES => Ok(Content::TooLarge),
        Ok(_) => Ok(Content::Bytes(
            fs::read(path).with_context(|| path.display().to_string())?,
        )),
    }
}

fn is_binary(bytes: &[u8]) -> bool {
    bytes[..bytes.len().min(BINARY_SNIFF_BYTES)].contains(&0)
}

fn lines_of(text: &str) -> impl Iterator<Item = &str> {
    text.lines().map(|line| line.trim_end_matches('\r'))
}

impl Changes {
    /// The changes of the session stored at `session_path`, for the project at `root`.
    pub fn open(root: &Path, session_path: &Path) -> Self {
        let dir = session_path.with_extension(CHANGES_EXT);
        // a missing or unreadable index means nothing was tracked yet
        let index = fs::read(dir.join(INDEX_FILE))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        let started = fs::metadata(dir.join(STARTED_FILE))
            .and_then(|meta| meta.modified())
            .ok();
        Self {
            root: root.canonicalize().unwrap_or_else(|_| root.to_path_buf()),
            dir,
            index,
            started,
        }
    }

    fn start(&mut self) -> Result<SystemTime> {
        if let Some(started) = self.started {
            return Ok(started);
        }
        fs::create_dir_all(&self.dir).with_context(|| self.dir.display().to_string())?;
        let marker = self.dir.join(STARTED_FILE);
        fs::write(&marker, b"")?;
        let started = fs::metadata(&marker)?.modified()?;
        self.started = Some(started);
        Ok(started)
    }

    fn save(&self) -> Result<()> {
        fs::write(self.dir.join(INDEX_FILE), serde_json::to_vec(&self.index)?)
            .with_context(|| self.dir.display().to_string())
    }

    fn key(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned()
    }

    fn path(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }

    /// Keeps `path` as it is now, unless an earlier copy is already kept; call it before the agent writes there.
    pub fn record(&mut self, path: &Path) -> Result<()> {
        self.start()?;
        let key = self.key(path);
        if self.index.originals.contains_key(&key) {
            return Ok(());
        }
        let kept = match read(path)? {
            Content::Missing => None,
            Content::TooLarge => return Ok(()),
            Content::Bytes(bytes) => {
                let name = self.index.originals.len().to_string();
                fs::write(self.dir.join(&name), bytes)?;
                Some(name)
            }
        };
        self.index.originals.insert(key, kept);
        self.save()
    }

    /// From now on files modified since tracking began count as changed, since a command may have written them.
    pub fn note_command(&mut self) -> Result<()> {
        self.start()?;
        if !self.index.commands_ran {
            self.index.commands_ran = true;
            self.save()?;
        }
        Ok(())
    }

    fn modified_since_start(&self) -> Vec<String> {
        let Some(started) = self.started.filter(|_| self.index.commands_ran) else {
            return Vec::new();
        };
        self.walker(&self.root, None)
            .take(MAX_SCAN_ENTRIES)
            .flatten()
            .filter(|entry| entry.file_type().is_some_and(|t| t.is_file()))
            .filter(|entry| {
                entry
                    .metadata()
                    .ok()
                    .and_then(|meta| meta.modified().ok())
                    .is_some_and(|modified| modified >= started)
            })
            .map(|entry| self.key(entry.path()))
            .collect()
    }

    fn walker(&self, dir: &Path, depth: Option<usize>) -> ignore::Walk {
        ignore::WalkBuilder::new(dir)
            .hidden(false)
            .require_git(false)
            .max_depth(depth)
            .filter_entry(|entry| entry.file_name() != ".git")
            .build()
    }

    fn original(&self, key: &str) -> Result<Original> {
        match self.index.originals.get(key) {
            Some(None) => return Ok(Original::Kept(Content::Missing)),
            Some(Some(name)) => return Ok(Original::Kept(read(&self.dir.join(name))?)),
            None => {}
        }
        let born = fs::metadata(self.path(key)).and_then(|meta| meta.created());
        if let (Ok(born), Some(started)) = (born, self.started)
            && born > started
        {
            return Ok(Original::Kept(Content::Missing));
        }
        let head = Command::new("git")
            .arg("-C")
            .arg(&self.root)
            .args(["show", &format!("HEAD:./{key}")])
            .output();
        Ok(match head {
            Ok(output) if output.status.success() => Original::Kept(Content::Bytes(output.stdout)),
            _ => Original::Unknown,
        })
    }

    /// The changed files, by path.
    pub fn list(&self) -> Result<Vec<FileChange>> {
        let mut keys: BTreeSet<String> = self.index.originals.keys().cloned().collect();
        keys.extend(self.modified_since_start());
        let mut changes = Vec::new();
        for key in keys {
            let diff = self.compare(&key, false)?;
            if let Some(status) = diff.status {
                changes.push(FileChange {
                    path: diff.path,
                    status,
                    added: diff.added,
                    removed: diff.removed,
                    binary: diff.binary,
                });
            }
        }
        Ok(changes)
    }

    /// `path`'s diff, or `None` when it is neither a tracked change nor a file inside the project.
    pub fn diff(&self, path: &str) -> Result<Option<FileDiff>> {
        if !self.index.originals.contains_key(path) && self.inside(path).is_none() {
            return Ok(None);
        }
        self.compare(path, true).map(Some)
    }

    // canonical, so `..` and symlinks can't reach outside the project
    fn inside(&self, path: &str) -> Option<PathBuf> {
        self.path(path)
            .canonicalize()
            .ok()
            .filter(|path| path.starts_with(&self.root))
    }

    fn compare(&self, key: &str, with_hunks: bool) -> Result<FileDiff> {
        let new = read(&self.path(key))?;
        let old = self.original(key)?;
        let status = match (&old, &new) {
            (Original::Unknown, _) => Some(ChangeStatus::NoOriginal),
            (Original::Kept(Content::Missing), Content::Missing) => None,
            (Original::Kept(Content::Missing), _) => Some(ChangeStatus::Added),
            (Original::Kept(_), Content::Missing) => Some(ChangeStatus::Deleted),
            (Original::Kept(Content::Bytes(a)), Content::Bytes(b)) if a == b => None,
            (Original::Kept(_), _) => Some(ChangeStatus::Modified),
        };
        let mut diff = FileDiff {
            path: key.to_string(),
            status,
            added: 0,
            removed: 0,
            binary: false,
            too_large: false,
            hunks: Vec::new(),
        };
        let bytes = |content: &Content| match content {
            Content::Bytes(bytes) => Some(bytes.clone()),
            _ => None,
        };
        let old_bytes = match &old {
            Original::Kept(content) => bytes(content),
            Original::Unknown => None,
        };
        let new_bytes = bytes(&new);
        diff.too_large =
            matches!(new, Content::TooLarge) || matches!(old, Original::Kept(Content::TooLarge));
        diff.binary = [&old_bytes, &new_bytes]
            .into_iter()
            .flatten()
            .any(|bytes| is_binary(bytes));
        if diff.too_large || diff.binary {
            return Ok(diff);
        }
        let old_text = String::from_utf8_lossy(old_bytes.as_deref().unwrap_or_default());
        let new_text = String::from_utf8_lossy(new_bytes.as_deref().unwrap_or_default());
        if matches!(status, None | Some(ChangeStatus::NoOriginal)) {
            if with_hunks {
                diff.hunks = vec![
                    lines_of(&new_text)
                        .enumerate()
                        .map(|(i, text)| DiffLine {
                            kind: LineKind::Context,
                            old: Some(i + 1),
                            new: Some(i + 1),
                            text: text.to_string(),
                        })
                        .collect(),
                ];
            }
            return Ok(diff);
        }
        let text_diff: TextDiff<'_, '_, str> = TextDiff::configure()
            .timeout(DIFF_TIMEOUT)
            .diff_lines(&*old_text, &*new_text);
        for group in text_diff.grouped_ops(CONTEXT_LINES) {
            let mut hunk = Vec::new();
            for op in &group {
                for change in text_diff.iter_changes(op) {
                    let kind = match change.tag() {
                        ChangeTag::Equal => LineKind::Context,
                        ChangeTag::Insert => {
                            diff.added += 1;
                            LineKind::Added
                        }
                        ChangeTag::Delete => {
                            diff.removed += 1;
                            LineKind::Removed
                        }
                    };
                    if with_hunks {
                        hunk.push(DiffLine {
                            kind,
                            old: change.old_index().map(|i| i + 1),
                            new: change.new_index().map(|i| i + 1),
                            text: change.value().trim_end_matches(['\n', '\r']).to_string(),
                        });
                    }
                }
            }
            if with_hunks {
                diff.hunks.push(hunk);
            }
        }
        Ok(diff)
    }

    /// The entries of `dir` (relative to the project), directories first, skipping what `.gitignore` excludes; `None`
    /// when it isn't a directory inside the project.
    pub fn entries(&self, dir: &str) -> Option<Vec<Entry>> {
        let path = self.inside(dir).filter(|path| path.is_dir())?;
        let mut entries: Vec<Entry> = self
            .walker(&path, Some(1))
            .flatten()
            .filter(|entry| entry.depth() == 1)
            .map(|entry| Entry {
                name: entry.file_name().to_string_lossy().into_owned(),
                dir: entry.file_type().is_some_and(|t| t.is_dir()),
            })
            .collect();
        entries.sort_by(|a, b| b.dir.cmp(&a.dir).then_with(|| a.name.cmp(&b.name)));
        Some(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::{ChangeStatus, Changes, LineKind};
    use std::path::Path;
    use std::process::Command;
    use std::time::Duration;

    // past the kernel's coarse file-time tick, so a file made after this counts as born after tracking began
    const TICK: Duration = Duration::from_millis(30);

    fn git(root: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
            .args(args)
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn kept_copies_and_command_writes_show_as_changes() {
        let project = tempfile::tempdir().unwrap();
        let sessions = tempfile::tempdir().unwrap();
        let root = project.path().canonicalize().unwrap();
        std::fs::write(root.join("kept.txt"), "one\ntwo\nthree\n").unwrap();
        std::fs::write(root.join("same.txt"), "same\n").unwrap();
        std::fs::write(root.join("tracked.txt"), "a\nb\n").unwrap();
        std::fs::write(root.join(".gitignore"), "ignored.txt\n").unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-qm", "init"]);
        std::thread::sleep(TICK);

        let mut changes = Changes::open(&root, &sessions.path().join("s.jsonl"));
        changes.record(&root.join("kept.txt")).unwrap();
        std::fs::write(root.join("kept.txt"), "one\n2\nthree\nfour\n").unwrap();
        changes.record(&root.join("same.txt")).unwrap();
        changes.record(&root.join("new.txt")).unwrap();
        std::fs::write(root.join("new.txt"), "fresh\n").unwrap();
        assert!(
            changes
                .list()
                .unwrap()
                .iter()
                .all(|c| c.path != "tracked.txt")
        );

        changes.note_command().unwrap();
        std::thread::sleep(TICK);
        std::fs::write(root.join("tracked.txt"), "a\nB\n").unwrap();
        std::fs::write(root.join("made.txt"), "x\n").unwrap();
        std::fs::write(root.join("ignored.txt"), "x\n").unwrap();

        // reopened, as a resumed session would be
        let changes = Changes::open(&root, &sessions.path().join("s.jsonl"));
        let listed: Vec<_> = changes
            .list()
            .unwrap()
            .into_iter()
            .map(|c| (c.path, c.status, c.added, c.removed))
            .collect();
        assert_eq!(
            listed,
            [
                ("kept.txt".to_string(), ChangeStatus::Modified, 2, 1),
                ("made.txt".to_string(), ChangeStatus::Added, 1, 0),
                ("new.txt".to_string(), ChangeStatus::Added, 1, 0),
                ("tracked.txt".to_string(), ChangeStatus::Modified, 1, 1),
            ]
        );

        let diff = changes.diff("kept.txt").unwrap().unwrap();
        let lines: Vec<_> = diff.hunks[0]
            .iter()
            .map(|l| (l.kind, l.old, l.new, l.text.as_str()))
            .collect();
        assert_eq!(
            lines,
            [
                (LineKind::Context, Some(1), Some(1), "one"),
                (LineKind::Removed, Some(2), None, "two"),
                (LineKind::Added, None, Some(2), "2"),
                (LineKind::Context, Some(3), Some(3), "three"),
                (LineKind::Added, None, Some(4), "four"),
            ]
        );
        let same = changes.diff("same.txt").unwrap().unwrap();
        assert_eq!((same.status, same.hunks[0].len()), (None, 1));
        assert!(changes.diff("../outside.txt").unwrap().is_none());
        assert!(changes.diff("/etc/passwd").unwrap().is_none());

        let names: Vec<_> = changes
            .entries("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(
            names,
            [
                ".gitignore",
                "kept.txt",
                "made.txt",
                "new.txt",
                "same.txt",
                "tracked.txt"
            ]
        );
        assert!(changes.entries("..").is_none());
    }
}
