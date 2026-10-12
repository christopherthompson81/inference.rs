//! CUDA kernel library commands

use anyhow::Result;

/// Download the kernel libraries this GPU needs that this install has no copy of.
#[cfg(feature = "cuda")]
pub fn run_kernels_fetch(types: Vec<String>, all: bool) -> Result<()> {
    use inference_api::system::{KernelSet, fetch_kernels, kernel_types};
    let base_only = !all && types.is_empty();
    let set = if all {
        KernelSet::All
    } else if base_only {
        KernelSet::Base
    } else {
        KernelSet::Types(types)
    };
    let report = fetch_kernels(set)?;
    println!(
        "Kernel libraries for {} are installed; downloads go to {}",
        report.arch_dir,
        report.download_dir.join(&report.arch_dir).display()
    );
    if base_only {
        println!(
            "GGUF types whose kernels --types fetches too: {}",
            kernel_types().join(", ")
        );
    }
    Ok(())
}

/// Remove downloaded kernels of other versions and unused kernel build dirs.
#[cfg(feature = "cuda")]
pub fn run_kernels_prune(dry_run: bool) -> Result<()> {
    let pruned = inference_api::system::prune_kernels(dry_run)?;
    if pruned.is_empty() {
        println!("Nothing to prune.");
        return Ok(());
    }
    let verb = if dry_run { "Would remove" } else { "Removed" };
    let mut total = 0;
    for dir in &pruned {
        total += dir.bytes;
        println!(
            "{verb} {} ({})",
            dir.path.display(),
            super::cache::format_size(dir.bytes)
        );
    }
    println!("{verb} {} in all", super::cache::format_size(total));
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run_kernels_fetch(_types: Vec<String>, _all: bool) -> Result<()> {
    anyhow::bail!("this `inference` was built without CUDA, so it loads no kernel libraries")
}

#[cfg(not(feature = "cuda"))]
pub fn run_kernels_prune(_dry_run: bool) -> Result<()> {
    anyhow::bail!("this `inference` was built without CUDA, so it keeps no kernel libraries")
}
