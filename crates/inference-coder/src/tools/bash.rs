use std::process::Stdio;
use std::time::Duration;

use futures::future::BoxFuture;
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;

use super::{Tool, ToolContext, ToolKind, ToolOutcome, str_arg, truncate_middle, usize_arg};

const SHELL: &str = "/bin/bash";
const DEFAULT_TIMEOUT_MS: usize = 120_000;
const MAX_TIMEOUT_MS: usize = 600_000;
// what the model sees of a command's output; the middle goes, the start and the end stay
const MAX_OUTPUT_BYTES: usize = 30_000;

pub struct BashTool;

impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Run a bash command in the project directory and return its output (stdout and stderr) and exit code. Each \
         call is a fresh shell. `timeout_ms` defaults to 120000 (max 600000)."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout_ms": {"type": "integer", "description": "How long to let it run, in milliseconds"}},
            "required": ["command"]})
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Execute
    }

    fn run<'a>(&'a self, args: Value, ctx: &'a ToolContext) -> BoxFuture<'a, ToolOutcome> {
        Box::pin(async move {
            let command = match str_arg(&args, "command") {
                Ok(command) => command,
                Err(error) => return error,
            };
            let timeout = usize_arg(&args, "timeout_ms")
                .unwrap_or(DEFAULT_TIMEOUT_MS)
                .min(MAX_TIMEOUT_MS);
            let child = tokio::process::Command::new(SHELL)
                .arg("-c")
                // stderr joins stdout so the model sees them interleaved as a terminal would show them
                .arg(format!("exec 2>&1\n{command}"))
                .current_dir(&ctx.root)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .kill_on_drop(true)
                .spawn();
            let mut child = match child {
                Ok(child) => child,
                Err(error) => return ToolOutcome::error(format!("cannot start {SHELL}: {error}")),
            };
            let mut stdout = child.stdout.take().expect("stdout is piped");
            let mut output = Vec::new();
            let run = async {
                let _ = stdout.read_to_end(&mut output).await;
                child.wait().await
            };
            let status = match tokio::time::timeout(Duration::from_millis(timeout as u64), run)
                .await
            {
                Ok(Ok(status)) => status,
                Ok(Err(error)) => {
                    return ToolOutcome::error(format!("waiting on the command: {error}"));
                }
                Err(_) => {
                    let text = truncate_middle(&String::from_utf8_lossy(&output), MAX_OUTPUT_BYTES);
                    return ToolOutcome::error(format!("{text}\n[timed out after {timeout} ms]"));
                }
            };
            let text = truncate_middle(&String::from_utf8_lossy(&output), MAX_OUTPUT_BYTES);
            match status.code() {
                Some(0) => ToolOutcome::ok(if text.is_empty() {
                    "(no output)".to_string()
                } else {
                    text
                }),
                Some(code) => ToolOutcome::error(format!("{text}\n[exit code {code}]")),
                None => ToolOutcome::error(format!("{text}\n[killed by a signal]")),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::BashTool;
    use crate::tools::{Tool, ToolContext};
    use serde_json::json;

    #[tokio::test]
    async fn runs_in_the_project_and_reports_failures() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext {
            root: dir.path().to_path_buf(),
        };
        let out = BashTool
            .run(json!({"command": "pwd; echo err >&2"}), &ctx)
            .await;
        assert!(!out.is_error);
        assert!(out.text.contains(&dir.path().display().to_string()) && out.text.contains("err"));
        let failed = BashTool.run(json!({"command": "exit 3"}), &ctx).await;
        assert!(failed.is_error && failed.text.contains("exit code 3"));
        let slow = BashTool
            .run(json!({"command": "sleep 5", "timeout_ms": 100}), &ctx)
            .await;
        assert!(slow.is_error && slow.text.contains("timed out"));
    }
}
