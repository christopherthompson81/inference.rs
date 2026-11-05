//! Compiles `shaders/*.comp` to SPIR-V with `glslc` (or `$GLSLC`) and writes `shaders.rs`, one embedded module each.

use std::path::{Path, PathBuf};
use std::process::Command;

const SHADER_DIR: &str = "shaders";
const GLSLC_ENV: &str = "GLSLC";
const TARGET_ENV: &str = "--target-env=vulkan1.2";
// NV_cooperative_matrix2 shaders take SPIR-V 1.6; a glslc too old for them (the probe fails) leaves them empty, and the
// device without
const CM2_MARK: &str = "_cm2";
const CM2_TARGET_ENV: &str = "--target-env=vulkan1.3";

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
    let cm2_glslc = cm2_supported(&glslc, &out);
    let mut table = String::new();
    for source in &sources {
        let name = source
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("shader names are UTF-8");
        let spv = out.join(format!("{name}.spv"));
        let cm2 = name.contains(CM2_MARK);
        let target = if cm2 { CM2_TARGET_ENV } else { TARGET_ENV };
        if compile(&glslc, target, source, &spv) {
            table.push_str(&format!(
                "pub(crate) const {}: &[u8] = include_bytes!({:?});\n",
                name.to_uppercase(),
                spv
            ));
        } else if cm2 && !cm2_glslc {
            println!(
                "cargo:warning={glslc} cannot compile {name} (NV_cooperative_matrix2); it stays empty"
            );
            table.push_str(&format!(
                "pub(crate) const {}: &[u8] = &[];\n",
                name.to_uppercase()
            ));
        } else {
            panic!("{glslc} failed on {}", source.display());
        }
    }
    std::fs::write(out.join("shaders.rs"), table).expect("OUT_DIR is writable");
}

fn cm2_supported(glslc: &str, out: &Path) -> bool {
    let probe = out.join("cm2_probe.comp");
    std::fs::write(&probe, CM2_PROBE).expect("OUT_DIR is writable");
    compile(glslc, CM2_TARGET_ENV, &probe, &out.join("cm2_probe.spv"))
}

const CM2_PROBE: &str =
    "#version 450\n#extension GL_NV_cooperative_matrix2 : require\nvoid main() {}\n";

fn compile(glslc: &str, target: &str, source: &Path, spv: &Path) -> bool {
    let status = Command::new(glslc)
        .args([target, "-O", "-Werror"])
        .arg(source)
        .arg("-o")
        .arg(spv)
        .status()
        .unwrap_or_else(|err| panic!("the `vulkan` feature compiles its shaders with {glslc} ({err}); install shaderc or set {GLSLC_ENV}"));
    status.success()
}
