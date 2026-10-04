fn main() {
    set_git_revision();
}

fn git(args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git").args(args).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (output.status.success() && !text.is_empty()).then(|| text.to_string())
}

fn set_git_revision() {
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=INFERENCE_RS_GIT_REVISION={commit}");

    // Paths come from git so worktrees resolve; a missing watched file would rerun this script on every build.
    let git_path = |name: &str| {
        git(&["rev-parse", "--path-format=absolute", "--git-path", name])
            .map(std::path::PathBuf::from)
    };
    let mut watched: Vec<std::path::PathBuf> = git_path("HEAD").into_iter().collect();
    if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]).and_then(|r| git_path(&r)) {
        if branch.exists() {
            watched.push(branch);
        } else {
            // A packed ref: the first commit writes a loose file, so watch the nearest existing ref directory.
            watched.extend(
                branch
                    .ancestors()
                    .skip(1)
                    .find(|dir| dir.is_dir())
                    .map(|dir| dir.to_path_buf()),
            );
            watched.extend(git_path("packed-refs"));
        }
    }
    for path in watched.iter().filter(|path| path.exists()) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
