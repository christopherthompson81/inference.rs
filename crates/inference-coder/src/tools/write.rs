use std::path::PathBuf;

use futures::future::BoxFuture;
use serde_json::{Value, json};

use super::{Tool, ToolContext, ToolKind, ToolOutcome, str_arg};

pub struct WriteTool;

impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "write"
    }

    fn description(&self) -> &'static str {
        "Write a whole file, creating it and its directories or replacing it. Prefer `edit` for changes to an existing \
         file."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to the project or absolute"},
            "content": {"type": "string", "description": "The file's full new content"}},
            "required": ["path", "content"]})
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Edit
    }

    fn paths(&self, args: &Value, ctx: &ToolContext) -> Vec<PathBuf> {
        str_arg(args, "path")
            .map(|p| vec![ctx.resolve(p)])
            .unwrap_or_default()
    }

    fn run<'a>(&'a self, args: Value, ctx: &'a ToolContext) -> BoxFuture<'a, ToolOutcome> {
        Box::pin(async move {
            let (path, content) = match (str_arg(&args, "path"), str_arg(&args, "content")) {
                (Ok(path), Ok(content)) => (ctx.resolve(path), content),
                (Err(error), _) | (_, Err(error)) => return error,
            };
            if let Some(parent) = path.parent()
                && let Err(error) = tokio::fs::create_dir_all(parent).await
            {
                return ToolOutcome::error(format!(
                    "cannot create {}: {error}",
                    ctx.display(parent)
                ));
            }
            let existed = path.exists();
            match tokio::fs::write(&path, content).await {
                Ok(()) => ToolOutcome::ok(format!(
                    "{} {} ({} lines)",
                    if existed { "replaced" } else { "created" },
                    ctx.display(&path),
                    content.lines().count()
                )),
                Err(error) => {
                    ToolOutcome::error(format!("cannot write {}: {error}", ctx.display(&path)))
                }
            }
        })
    }
}
