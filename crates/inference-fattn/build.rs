// K/V pairs with a vector-kernel instance (llama.cpp's GGML_CUDA_FA_QUANTS); other pairs convert to f16 first.
#[cfg(feature = "cuda")]
const VEC_KV_TYPES: [&str; 7] = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "BF16", "F16"];
#[cfg(feature = "cuda")]
const VEC_INSTANCES: [(&str, &str); 2] = [("F16", "F16"), ("BF16", "BF16")];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "cuda")]
    cuda::build();
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::{VEC_INSTANCES, VEC_KV_TYPES};

    pub fn build() {
        let mut sources = vec![
            "kernels/cuda/fattn.cu".to_string(),
            "kernels/cuda/fattn-tile.cu".to_string(),
            "kernels/cuda/ggml_compat.cu".to_string(),
            "kernels/cuda/entry.cu".to_string(),
        ];
        let mut instances = std::fs::read_dir("kernels/cuda/instances")
            .expect("kernels/cuda/instances")
            .map(|entry| entry.unwrap().path().display().to_string())
            .collect::<Vec<_>>();
        instances.sort();
        sources.extend(instances);
        let mut builder = cudaforge::KernelBuilder::new()
            .source_files(sources)
            .watch(["kernels/cuda"])
            .compress_fatbin()
            .include_path("kernels/cuda")
            .include_path("kernels/cuda/ggml")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-DNDEBUG")
            .arg("-use_fast_math")
            .arg("-extended-lambda");
        for k in VEC_KV_TYPES {
            for v in VEC_KV_TYPES {
                let on = VEC_INSTANCES.contains(&(k, v)) as u8;
                builder = builder.arg(&format!("-DGGML_CUDA_FA_{k}_{v}={on}"));
            }
        }
        if !std::env::var("TARGET").unwrap_or_default().contains("msvc") {
            builder = builder.arg("-Xcompiler").arg("-fPIC");
        }
        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        builder
            .build_and_link("inferencefattn", out_dir.join("libinferencefattn.a"))
            .expect("fattn kernels failed to build");
        println!("cargo:rustc-link-lib=dylib=cudart");
        if !std::env::var("TARGET").unwrap_or_default().contains("msvc") {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }
}
