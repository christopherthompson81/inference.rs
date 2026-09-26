fn main() {
    set_git_revision();

    #[cfg(feature = "cudnn")]
    add_cudnn_link_search();
}

#[cfg(feature = "cudnn")]
fn add_cudnn_link_search() {
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=CUDNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("msvc") {
        return;
    }

    if let Ok(dir) = std::env::var("CUDNN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        return;
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        candidates.push(PathBuf::from(&cuda_path).join("lib").join("x64"));
    }
    let cudnn_root = PathBuf::from(r"C:\Program Files\NVIDIA\CUDNN");
    if let Ok(versions) = std::fs::read_dir(&cudnn_root) {
        for version in versions.flatten() {
            let lib = version.path().join("lib");
            candidates.push(lib.join("x64"));
            if let Ok(cuda_vers) = std::fs::read_dir(&lib) {
                for cuda_ver in cuda_vers.flatten() {
                    candidates.push(cuda_ver.path().join("x64"));
                }
            }
        }
    }

    for dir in candidates {
        if dir.join("cudnn.lib").is_file() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            return;
        }
    }

    println!(
        "cargo:warning=cudnn feature enabled but cudnn.lib not found; set CUDNN_LIB_DIR to its directory"
    );
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
