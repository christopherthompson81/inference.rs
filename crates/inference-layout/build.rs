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
        cudaforge::KernelBuilder::new()
            .source_files(["kernels/cuda/layout.cu"])
            .out_dir(&out_dir)
            .compress_fatbin()
            .build_fatbin()
            .expect("layout kernels failed to build");
    }
}
