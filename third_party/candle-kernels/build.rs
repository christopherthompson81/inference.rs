use cudaforge::{KernelBuilder, Result};
use std::env;
use std::io::Write;
use std::path::PathBuf;

const CUTILE_FEATURE: &str = "CARGO_FEATURE_CUTILE";
const PTX_ENTRY_PREFIX: &str = ".visible .entry ";

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // PTX is an intermediate: its entry names feed the preloader, and the SASS fatbins built from it ship.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;
    let fatbins = KernelBuilder::new()
        .source_files(ptx.images())
        .compress_fatbin()
        .build_fatbin()?;
    let images_path = out_dir.join("images.rs");
    fatbins.write(&images_path)?;
    let mut images = std::fs::OpenOptions::new().append(true).open(&images_path)?;
    for path in ptx.images() {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap();
        let name = name.to_uppercase();
        let entries = std::fs::read_to_string(&path)?
            .lines()
            .filter_map(|line| {
                let entry = line.trim_start().strip_prefix(PTX_ENTRY_PREFIX)?;
                entry.split_once('(')
            })
            .map(|(entry, _)| format!("{:?}", entry.trim()))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(images, "pub const {name}_ENTRIES: &[&str] = &[{entries}];")?;
    }

    let mut moe_sources = vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
        "src/mmvq_gguf.cu",
        "src/mmq_gguf/mmq_quantize.cu",
        "src/mmq_gguf/mmq_instance_q4_0.cu",
        "src/mmq_gguf/mmq_instance_q4_1.cu",
        "src/mmq_gguf/mmq_instance_q5_0.cu",
        "src/mmq_gguf/mmq_instance_q5_1.cu",
        "src/mmq_gguf/mmq_instance_q8_0.cu",
        "src/mmq_gguf/mmq_instance_q2_k.cu",
        "src/mmq_gguf/mmq_instance_q3_k.cu",
        "src/mmq_gguf/mmq_instance_q4_k.cu",
        "src/mmq_gguf/mmq_instance_q5_k.cu",
        "src/mmq_gguf/mmq_instance_q6_k.cu",
    ];
    if env::var_os(CUTILE_FEATURE).is_some() {
        moe_sources.push("src/moe/moe_align.cu");
    }

    let mut moe_builder = KernelBuilder::default()
        .source_files(moe_sources)
        .compress_fatbin()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
