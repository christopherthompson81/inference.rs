fn main() {
    // without it a CPU build reruns this script whenever any file in the package changes
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "cuda")]
    cuda::build();
}

#[cfg(feature = "cuda")]
mod cuda {
    pub fn build() {
        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        inference_kernel_build::KernelBuilder::new()
            .source_files([
                "kernels/cuda/mesh.cu",
                "kernels/cuda/dit.cu",
                "kernels/cuda/sparse.cu",
            ])
            .out_dir(&out_dir)
            // products round as on the host, so the GPU passes give the host port's results
            .arg("--fmad=false")
            .compress_fatbin()
            .build_fatbin()
            .expect("recon kernels failed to build");
    }
}
