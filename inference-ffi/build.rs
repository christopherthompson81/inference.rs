fn main() {
    // export only the inference_* C ABI from the shared library (dependencies' symbols stay hidden)
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    if target_os == "linux" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,--version-script={dir}/inference.map");
    }
    println!("cargo:rerun-if-changed=inference.map");
    println!("cargo:rerun-if-changed=include/inference.h");
}
