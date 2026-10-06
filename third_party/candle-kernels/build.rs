use cudaforge::{KernelBuilder, Result};
use std::env;
use std::io::Write;
use std::path::PathBuf;

// cuobjdump -symbols: an `arch = sm_86` line opens each cubin, and a kernel entry carries this binding
const CUBIN_ARCH_PREFIX: &str = "arch = sm_";
const ENTRY_SYMBOL: &str = "STO_ENTRY";
// a PTX section carries its own arch line but no symbols, so only an ELF section opens a cubin
const CUBIN_SECTION: &str = "Fatbin elf code";
const CUOBJDUMP: &str = "cuobjdump";

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // fatbins build from the sources for every listed arch; an entry some arch's SASS lacks is optional
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let fatbins = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .compress_fatbin()
        .build_fatbin()?;
    let cuobjdump = cudaforge::CudaToolkit::detect()?.nvcc_path.with_file_name(CUOBJDUMP);
    let images_path = out_dir.join("images.rs");
    fatbins.write(&images_path)?;
    let mut images = std::fs::OpenOptions::new().append(true).open(&images_path)?;
    let mut archs = Vec::new();
    for path in fatbins.images() {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap().to_uppercase();
        let per_arch = entries_by_arch(&cuobjdump, &path)?;
        let module_archs: Vec<String> = per_arch.iter().map(|(arch, _)| arch.clone()).collect();
        assert!(archs.is_empty() || archs == module_archs, "{name} has SASS for {module_archs:?}, not {archs:?}");
        archs = module_archs;
        let mut every: Vec<&String> = Vec::new();
        for entry in per_arch.iter().flat_map(|(_, entries)| entries) {
            if !every.contains(&entry) {
                every.push(entry);
            }
        }
        let optional = every
            .iter()
            .filter(|entry| per_arch.iter().any(|(_, entries)| !entries.contains(entry)));
        writeln!(images, "pub const {name}_ENTRIES: &[&str] = &[{}];", quoted(every.iter().copied()))?;
        writeln!(images, "pub const {name}_OPTIONAL_ENTRIES: &[&str] = &[{}];", quoted(optional.copied()))?;
    }
    assert!(!archs.is_empty(), "cuobjdump listed no cubins in the candle fatbins");
    writeln!(images, "pub const ARCHS: &[&str] = &[{}];", quoted(archs.iter()))?;
    Ok(())
}

// Each cubin's arch and kernel entries, in the order cuobjdump lists them.
fn entries_by_arch(cuobjdump: &std::path::Path, fatbin: &std::path::Path) -> Result<Vec<(String, Vec<String>)>> {
    let output = std::process::Command::new(cuobjdump)
        .arg("-symbols")
        .arg(fatbin)
        .output()?;
    if !output.status.success() {
        panic!("{} -symbols {} failed: {}", cuobjdump.display(), fatbin.display(), String::from_utf8_lossy(&output.stderr));
    }
    let mut per_arch: Vec<(String, Vec<String>)> = Vec::new();
    let mut in_cubin = false;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.starts_with("Fatbin ") {
            in_cubin = line.starts_with(CUBIN_SECTION);
        } else if let Some(arch) = line.trim().strip_prefix(CUBIN_ARCH_PREFIX).filter(|_| in_cubin) {
            per_arch.push((arch.to_string(), Vec::new()));
        } else if line.contains(ENTRY_SYMBOL) {
            let entry = line.split_whitespace().last().unwrap_or_default().to_string();
            per_arch.last_mut().expect("symbols follow an arch line").1.push(entry);
        }
    }
    Ok(per_arch)
}

fn quoted<'a>(entries: impl Iterator<Item = &'a String>) -> String {
    entries.map(|entry| format!("{entry:?}")).collect::<Vec<_>>().join(", ")
}
