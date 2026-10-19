use std::path::PathBuf;

use futures::future::BoxFuture;
use serde_json::{Value, json};

use super::{Tool, ToolContext, ToolKind, ToolOutcome, str_arg};

pub struct EditTool;

impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn description(&self) -> &'static str {
        "Replace exact text in a file. `old_string` must match the file exactly (whitespace included) and occur once, \
         unless `replace_all` is true. Read the file first."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to the project or absolute"},
            "old_string": {"type": "string", "description": "The exact text to replace"},
            "new_string": {"type": "string", "description": "The text to put in its place"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence"}},
            "required": ["path", "old_string", "new_string"]})
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
            let (path, old, new) = match (
                str_arg(&args, "path"),
                str_arg(&args, "old_string"),
                str_arg(&args, "new_string"),
            ) {
                (Ok(path), Ok(old), Ok(new)) => (ctx.resolve(path), old, new),
                (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => return error,
            };
            if old.is_empty() {
                return ToolOutcome::error("`old_string` is empty; use `write` to create a file");
            }
            if old == new {
                return ToolOutcome::error("`old_string` and `new_string` are the same");
            }
            let replace_all = args
                .get("replace_all")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let text = match tokio::fs::read_to_string(&path).await {
                Ok(text) => text,
                Err(error) => {
                    return ToolOutcome::error(format!(
                        "cannot read {}: {error}",
                        ctx.display(&path)
                    ));
                }
            };
            let count = text.matches(old).count();
            let updated = match (count, replace_all) {
                (0, _) => {
                    return ToolOutcome::error(format!(
                        "`old_string` does not occur in {}; read the file and copy the text exactly",
                        ctx.display(&path)
                    ));
                }
                (1, _) | (_, true) => text.replace(old, new),
                (n, false) => {
                    return ToolOutcome::error(format!(
                        "`old_string` occurs {n} times in {}; include more context or set `replace_all`",
                        ctx.display(&path)
                    ));
                }
            };
            match tokio::fs::write(&path, updated).await {
                Ok(()) => ToolOutcome::ok(format!(
                    "edited {} ({count} replacement{})",
                    ctx.display(&path),
                    if count == 1 { "" } else { "s" }
                )),
                Err(error) => {
                    ToolOutcome::error(format!("cannot write {}: {error}", ctx.display(&path)))
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::EditTool;
    use crate::tools::{Tool, ToolContext};
    use serde_json::json;

    #[tokio::test]
    async fn replaces_a_unique_match_and_refuses_ambiguity() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.rs");
        std::fs::write(&file, "let a = 1;\nlet b = 1;\n").unwrap();
        let ctx = ToolContext {
            root: dir.path().to_path_buf(),
        };
        let ambiguous = EditTool
            .run(
                json!({"path": "f.rs", "old_string": "= 1", "new_string": "= 2"}),
                &ctx,
            )
            .await;
        assert!(ambiguous.is_error && ambiguous.text.contains("2 times"));
        let unique = EditTool
            .run(
                json!({"path": "f.rs", "old_string": "let b = 1", "new_string": "let b = 3"}),
                &ctx,
            )
            .await;
        assert!(!unique.is_error);
        let all = EditTool
            .run(json!({"path": "f.rs", "old_string": "let", "new_string": "const", "replace_all": true}), &ctx)
            .await;
        assert!(!all.is_error);
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "const a = 1;\nconst b = 3;\n"
        );
        let missing = EditTool
            .run(
                json!({"path": "f.rs", "old_string": "nope", "new_string": "x"}),
                &ctx,
            )
            .await;
        assert!(missing.is_error);
    }
}
