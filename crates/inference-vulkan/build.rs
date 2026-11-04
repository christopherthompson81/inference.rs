//! Compiles `shaders/*.comp` to SPIR-V with `glslc` (or `$GLSLC`) and writes `shaders.rs`, one embedded module each.

use std::path::{Path, PathBuf};
use std::process::Command;

const SHADER_DIR: &str = "shaders";
const GLSLC_ENV: &str = "GLSLC";
const TARGET_ENV: &str = "--target-env=vulkan1.2";

fn main() {
    println!("cargo:rerun-if-changed={SHADER_DIR}");
    println!("cargo:rerun-if-env-changed={GLSLC_ENV}");
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let glslc = std::env::var(GLSLC_ENV).unwrap_or_else(|_| "glslc".into());
    let mut sources: Vec<PathBuf> = std::fs::read_dir(SHADER_DIR)
        .expect("the crate ships its shaders")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "comp"))
        .collect();
    sources.sort();
    let mut table = String::new();
    for source in &sources {
        let name = source
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("shader names are UTF-8");
        let spv = out.join(format!("{name}.spv"));
        compile(&glslc, source, &spv);
        table.push_str(&format!(
            "pub(crate) const {}: &[u8] = include_bytes!({:?});\n",
            name.to_uppercase(),
            spv
        ));
    }
    std::fs::write(out.join("shaders.rs"), table).expect("OUT_DIR is writable");
}

fn compile(glslc: &str, source: &Path, spv: &Path) {
    let status = Command::new(glslc)
        .args([TARGET_ENV, "-O", "-Werror"])
        .arg(source)
        .arg("-o")
        .arg(spv)
        .status()
        .unwrap_or_else(|err| panic!("the `vulkan` feature compiles its shaders with {glslc} ({err}); install shaderc or set {GLSLC_ENV}"));
    assert!(status.success(), "{glslc} failed on {}", source.display());
}
