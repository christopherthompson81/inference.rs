use std::path::{Path, PathBuf};

use futures::future::BoxFuture;
use ignore::{WalkBuilder, overrides::OverrideBuilder};
use serde_json::{Value, json};

use super::{Tool, ToolContext, ToolKind, ToolOutcome, str_arg, usize_arg};

const DEFAULT_MATCHES: usize = 200;
const DEFAULT_FILES: usize = 500;
const MAX_MATCH_LINE_CHARS: usize = 300;

// Files under `base` the project's ignore files keep, narrowed to `glob` when given
fn walk(base: &Path, glob: Option<&str>) -> Result<ignore::Walk, String> {
    let mut builder = WalkBuilder::new(base);
    builder
        .hidden(false)
        .git_global(false)
        .filter_entry(|entry| entry.file_name() != ".git");
    if let Some(glob) = glob {
        let mut overrides = OverrideBuilder::new(base);
        overrides
            .add(glob)
            .map_err(|error| format!("bad glob `{glob}`: {error}"))?;
        builder.overrides(
            overrides
                .build()
                .map_err(|error| format!("bad glob `{glob}`: {error}"))?,
        );
    }
    Ok(builder.build())
}

fn base_dir(args: &Value, ctx: &ToolContext) -> PathBuf {
    args.get("path")
        .and_then(Value::as_str)
        .map_or_else(|| ctx.root.clone(), |p| ctx.resolve(p))
}

pub struct GrepTool;

impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn description(&self) -> &'static str {
        "Search file contents for a regular expression, skipping files the project ignores. Returns `path:line: text` \
         matches. `glob` narrows the files (e.g. `*.rs`); `path` narrows the directory."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regular expression (Rust regex syntax)"},
            "path": {"type": "string", "description": "Directory or file to search, default the project"},
            "glob": {"type": "string", "description": "Only files matching this glob"},
            "case_insensitive": {"type": "boolean"},
            "max_results": {"type": "integer", "description": "Default 200"}},
            "required": ["pattern"]})
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    fn paths(&self, args: &Value, ctx: &ToolContext) -> Vec<PathBuf> {
        vec![base_dir(args, ctx)]
    }

    fn run<'a>(&'a self, args: Value, ctx: &'a ToolContext) -> BoxFuture<'a, ToolOutcome> {
        Box::pin(async move {
            let pattern = match str_arg(&args, "pattern") {
                Ok(pattern) => pattern.to_string(),
                Err(error) => return error,
            };
            let regex = match regex::RegexBuilder::new(&pattern)
                .case_insensitive(
                    args.get("case_insensitive")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                )
                .build()
            {
                Ok(regex) => regex,
                Err(error) => return ToolOutcome::error(format!("bad pattern: {error}")),
            };
            let base = base_dir(&args, ctx);
            let glob = args.get("glob").and_then(Value::as_str).map(str::to_string);
            let max = usize_arg(&args, "max_results").unwrap_or(DEFAULT_MATCHES);
            let ctx = ctx.clone();
            let search = tokio::task::spawn_blocking(move || -> Result<String, String> {
                let mut out = String::new();
                let mut found = 0;
                for entry in walk(&base, glob.as_deref())?.flatten() {
                    if !entry.file_type().is_some_and(|t| t.is_file()) {
                        continue;
                    }
                    let Ok(text) = std::fs::read_to_string(entry.path()) else {
                        continue;
                    };
                    for (n, line) in text.lines().enumerate() {
                        if !regex.is_match(line) {
                            continue;
                        }
                        if found == max {
                            out.push_str(
                                "... more matches (raise `max_results` or narrow the search)\n",
                            );
                            return Ok(out);
                        }
                        let line = match line.char_indices().nth(MAX_MATCH_LINE_CHARS) {
                            Some((cut, _)) => format!("{}...", &line[..cut]),
                            None => line.to_string(),
                        };
                        out.push_str(&format!(
                            "{}:{}: {}\n",
                            ctx.display(entry.path()),
                            n + 1,
                            line.trim_end()
                        ));
                        found += 1;
                    }
                }
                Ok(if found == 0 {
                    "no matches".to_string()
                } else {
                    out
                })
            });
            match search.await {
                Ok(Ok(text)) => ToolOutcome::ok(text),
                Ok(Err(error)) => ToolOutcome::error(error),
                Err(error) => ToolOutcome::error(format!("search failed: {error}")),
            }
        })
    }
}

pub struct GlobTool;

impl Tool for GlobTool {
    fn name(&self) -> &'static str {
        "glob"
    }

    fn description(&self) -> &'static str {
        "List files matching a glob (e.g. `**/*.rs`, `src/**/mod.rs`), skipping files the project ignores. `path` \
         narrows the directory."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Directory to search, default the project"}},
            "required": ["pattern"]})
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    fn paths(&self, args: &Value, ctx: &ToolContext) -> Vec<PathBuf> {
        vec![base_dir(args, ctx)]
    }

    fn run<'a>(&'a self, args: Value, ctx: &'a ToolContext) -> BoxFuture<'a, ToolOutcome> {
        Box::pin(async move {
            let pattern = match str_arg(&args, "pattern") {
                Ok(pattern) => pattern.to_string(),
                Err(error) => return error,
            };
            let base = base_dir(&args, ctx);
            let ctx = ctx.clone();
            let listing = tokio::task::spawn_blocking(move || -> Result<String, String> {
                let mut files = walk(&base, Some(&pattern))?
                    .flatten()
                    .filter(|entry| entry.file_type().is_some_and(|t| t.is_file()))
                    .map(|entry| ctx.display(entry.path()))
                    .collect::<Vec<_>>();
                files.sort();
                let total = files.len();
                files.truncate(DEFAULT_FILES);
                let mut out = files.join("\n");
                if total == 0 {
                    out = "no files match".to_string();
                } else if total > DEFAULT_FILES {
                    out.push_str(&format!(
                        "\n... {} more files (narrow the pattern)",
                        total - DEFAULT_FILES
                    ));
                }
                Ok(out)
            });
            match listing.await {
                Ok(Ok(text)) => ToolOutcome::ok(text),
                Ok(Err(error)) => ToolOutcome::error(error),
                Err(error) => ToolOutcome::error(format!("listing failed: {error}")),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{GlobTool, GrepTool};
    use crate::tools::{Tool, ToolContext};
    use serde_json::json;

    fn project() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join("target")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "target/\n").unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "fn alpha() {}\nfn beta() {}\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() { alpha(); }\n").unwrap();
        std::fs::write(dir.path().join("target/out.rs"), "fn alpha() {}\n").unwrap();
        dir
    }

    #[tokio::test]
    async fn grep_finds_matches_and_skips_ignored_files() {
        let dir = project();
        // the ignore crate honors .gitignore inside a git repository
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        let ctx = ToolContext {
            root: dir.path().to_path_buf(),
        };
        let out = GrepTool.run(json!({"pattern": "alpha"}), &ctx).await;
        assert!(!out.is_error, "{}", out.text);
        assert!(out.text.contains("src/lib.rs:1: fn alpha() {}"));
        assert!(out.text.contains("src/main.rs:1:"));
        assert!(!out.text.contains("target/"));
        let narrowed = GrepTool
            .run(json!({"pattern": "fn", "glob": "lib.rs"}), &ctx)
            .await;
        assert_eq!(narrowed.text.lines().count(), 2);
    }

    #[tokio::test]
    async fn glob_lists_matching_files() {
        let dir = project();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        let ctx = ToolContext {
            root: dir.path().to_path_buf(),
        };
        let out = GlobTool.run(json!({"pattern": "**/*.rs"}), &ctx).await;
        assert_eq!(out.text, "src/lib.rs\nsrc/main.rs");
    }
}
