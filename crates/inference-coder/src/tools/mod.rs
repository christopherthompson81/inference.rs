//! The coding tools: read, write, edit, bash, grep and glob over the project directory.

mod bash;
mod edit;
mod read;
mod search;
mod write;

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::{Value, json};

pub use bash::BashTool;
pub use edit::EditTool;
pub use read::ReadTool;
pub use search::{GlobTool, GrepTool};
pub use write::WriteTool;

/// What a tool can do, which decides whether a call needs approval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolKind {
    Read,
    Edit,
    Execute,
}

/// What a tool call produced: text for the model, and whether it failed.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutcome {
    pub text: String,
    pub is_error: bool,
}

impl ToolOutcome {
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
        }
    }
}

/// Where tools act: the project directory, which relative paths resolve against.
#[derive(Clone, Debug)]
pub struct ToolContext {
    pub root: PathBuf,
}

impl ToolContext {
    /// `path` against the project root, with `.` and `..` folded so containment checks see the real target.
    pub fn resolve(&self, path: &str) -> PathBuf {
        let joined = self.root.join(path);
        let mut out = PathBuf::new();
        for component in joined.components() {
            match component {
                Component::ParentDir => {
                    out.pop();
                }
                Component::CurDir => {}
                other => out.push(other),
            }
        }
        out
    }

    pub fn is_inside(&self, path: &Path) -> bool {
        path.starts_with(&self.root)
    }

    /// `path` as shown to the model and the user: relative to the root when inside it.
    pub fn display(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    /// The JSON schema of the tool's arguments.
    fn parameters(&self) -> Value;
    fn kind(&self) -> ToolKind;
    /// The files a call reads or changes, for approval decisions on paths outside the project.
    fn paths(&self, args: &Value, ctx: &ToolContext) -> Vec<PathBuf> {
        let _ = (args, ctx);
        Vec::new()
    }
    fn run<'a>(&'a self, args: Value, ctx: &'a ToolContext) -> BoxFuture<'a, ToolOutcome>;

    /// The OpenAI function definition sent with each request.
    fn definition(&self) -> Value {
        json!({"type": "function", "function": {
            "name": self.name(), "description": self.description(), "parameters": self.parameters()}})
    }
}

/// The default toolset, in a fixed order so the request's tool list never changes and stays in the cached prefix.
pub fn default_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(ReadTool),
        Arc::new(WriteTool),
        Arc::new(EditTool),
        Arc::new(BashTool),
        Arc::new(GrepTool),
        Arc::new(GlobTool),
    ]
}

/// A string argument, or the error the model sees.
pub(crate) fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, ToolOutcome> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolOutcome::error(format!("missing string argument `{key}`")))
}

pub(crate) fn usize_arg(args: &Value, key: &str) -> Option<usize> {
    args.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

/// `text` cut to `max` bytes on a char boundary, keeping its head and tail and saying how much went.
pub(crate) fn truncate_middle(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let half = max / 2;
    let head_end = (0..=half)
        .rev()
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(0);
    let tail_start = (text.len() - half..text.len())
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(text.len());
    format!(
        "{}\n... [{} bytes omitted] ...\n{}",
        &text[..head_end],
        tail_start - head_end,
        &text[tail_start..]
    )
}

#[cfg(test)]
mod tests {
    use super::{ToolContext, truncate_middle};
    use std::path::PathBuf;

    #[test]
    fn paths_resolve_against_the_root_and_fold_parents() {
        let ctx = ToolContext {
            root: PathBuf::from("/work/repo"),
        };
        assert_eq!(
            ctx.resolve("src/../lib.rs"),
            PathBuf::from("/work/repo/lib.rs")
        );
        assert!(ctx.is_inside(&ctx.resolve("a/b.rs")));
        assert!(!ctx.is_inside(&ctx.resolve("../other/x")));
        assert_eq!(ctx.resolve("/etc/hosts"), PathBuf::from("/etc/hosts"));
        assert_eq!(ctx.display(&ctx.resolve("a/b.rs")), "a/b.rs");
    }

    #[test]
    fn long_output_keeps_its_head_and_tail() {
        let text = "a".repeat(100) + &"b".repeat(100);
        let cut = truncate_middle(&text, 50);
        assert!(
            cut.starts_with("aaaa") && cut.ends_with("bbbb") && cut.contains("150 bytes omitted")
        );
    }
}
