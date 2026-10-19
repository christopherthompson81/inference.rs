use std::path::PathBuf;

use futures::future::BoxFuture;
use serde_json::{Value, json};

use super::{Tool, ToolContext, ToolKind, ToolOutcome, str_arg, usize_arg};

const DEFAULT_LINES: usize = 2000;
const MAX_LINE_CHARS: usize = 2000;

pub struct ReadTool;

impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }

    fn description(&self) -> &'static str {
        "Read a text file, with line numbers. `offset` (1-based) and `limit` select a range of lines; long files are \
         cut at 2000 lines unless a range is given."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to the project or absolute"},
            "offset": {"type": "integer", "description": "First line to read, 1-based"},
            "limit": {"type": "integer", "description": "Number of lines to read"}},
            "required": ["path"]})
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    fn paths(&self, args: &Value, ctx: &ToolContext) -> Vec<PathBuf> {
        str_arg(args, "path")
            .map(|p| vec![ctx.resolve(p)])
            .unwrap_or_default()
    }

    fn run<'a>(&'a self, args: Value, ctx: &'a ToolContext) -> BoxFuture<'a, ToolOutcome> {
        Box::pin(async move {
            let path = match str_arg(&args, "path") {
                Ok(path) => ctx.resolve(path),
                Err(error) => return error,
            };
            let text = match tokio::fs::read(&path).await {
                Ok(bytes) => match String::from_utf8(bytes) {
                    Ok(text) => text,
                    Err(_) => {
                        return ToolOutcome::error(format!(
                            "{} is not UTF-8 text",
                            ctx.display(&path)
                        ));
                    }
                },
                Err(error) => {
                    return ToolOutcome::error(format!(
                        "cannot read {}: {error}",
                        ctx.display(&path)
                    ));
                }
            };
            let offset = usize_arg(&args, "offset").unwrap_or(1).max(1);
            let limit = usize_arg(&args, "limit").unwrap_or(DEFAULT_LINES);
            let total = text.lines().count();
            let mut out = String::new();
            for (n, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
                let line = match line.char_indices().nth(MAX_LINE_CHARS) {
                    Some((cut, _)) => format!("{}...", &line[..cut]),
                    None => line.to_string(),
                };
                out.push_str(&format!("{:>6}\t{line}\n", n + 1));
            }
            let last = (offset - 1 + limit).min(total);
            if total == 0 {
                out.push_str("(empty file)\n");
            } else if last < total {
                out.push_str(&format!(
                    "... {} more lines (read with offset {})\n",
                    total - last,
                    last + 1
                ));
            }
            ToolOutcome::ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ReadTool;
    use crate::tools::{Tool, ToolContext};
    use serde_json::json;

    #[tokio::test]
    async fn reads_a_numbered_range() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "one\ntwo\nthree\nfour\n").unwrap();
        let ctx = ToolContext {
            root: dir.path().to_path_buf(),
        };
        let out = ReadTool
            .run(json!({"path": "f.txt", "offset": 2, "limit": 2}), &ctx)
            .await;
        assert!(!out.is_error);
        assert_eq!(
            out.text,
            "     2\ttwo\n     3\tthree\n... 1 more lines (read with offset 4)\n"
        );
        assert!(
            ReadTool
                .run(json!({"path": "missing"}), &ctx)
                .await
                .is_error
        );
    }
}
