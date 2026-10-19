//! The system prompt: short and fixed for the session, so every request shares its cached prefix.

use std::path::Path;
use std::process::Command;

// project instruction files, read from the root in this order; the first ones a project has all go in
const PROJECT_FILES: [&str; 2] = ["AGENTS.md", "CLAUDE.md"];
// a project file past this is cut, so a huge one doesn't eat the context
const MAX_PROJECT_FILE_BYTES: usize = 32 * 1024;

const BASE_PROMPT: &str = "You are a coding agent working in the user's project through tools. Read files before \
changing them, make the smallest change that does the job, and check it (build, tests) when the project makes that \
possible. Use `edit` for changes to existing files and `write` only for new files or full rewrites. Prefer `grep` and \
`glob` over `bash` for searching. Paths are relative to the project directory. Keep replies short: say what you did \
and anything the user must know, without restating code they can read.";

/// The system prompt for a session rooted at `root`.
pub fn system_prompt(root: &Path) -> String {
    let mut prompt = String::from(BASE_PROMPT);
    prompt.push_str("\n\n# Environment\n");
    prompt.push_str(&format!("Project directory: {}\n", root.display()));
    prompt.push_str(&format!(
        "Platform: {} {}\n",
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    prompt.push_str(&format!(
        "Date: {}\n",
        chrono::Local::now().format("%Y-%m-%d")
    ));
    if let Some(branch) = git_branch(root) {
        prompt.push_str(&format!("Git branch: {branch}\n"));
    }
    for name in PROJECT_FILES {
        let Ok(text) = std::fs::read_to_string(root.join(name)) else {
            continue;
        };
        let text = match (0..=MAX_PROJECT_FILE_BYTES.min(text.len()))
            .rev()
            .find(|&i| text.is_char_boundary(i))
        {
            Some(end) if end < text.len() => format!("{}\n... (cut)", &text[..end]),
            _ => text,
        };
        prompt.push_str(&format!(
            "\n# Project instructions ({name})\n{}\n",
            text.trim_end()
        ));
    }
    prompt
}

fn git_branch(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(root)
        .output()
        .ok()?;
    let branch = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (output.status.success() && !branch.is_empty()).then_some(branch)
}

#[cfg(test)]
mod tests {
    use super::system_prompt;

    #[test]
    fn includes_the_environment_and_project_instructions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "Run the linter first.\n").unwrap();
        let prompt = system_prompt(dir.path());
        assert!(prompt.contains(&format!("Project directory: {}", dir.path().display())));
        assert!(prompt.contains("# Project instructions (AGENTS.md)\nRun the linter first."));
        assert!(!prompt.contains("CLAUDE.md"));
    }
}
