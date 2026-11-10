fn main() {
    println!("cargo:rerun-if-changed=third_party/xatlas");
    cc::Build::new()
        .cpp(true)
        .file("third_party/xatlas/xatlas.cpp")
        .define("XATLAS_C_API", "1")
        // without NDEBUG xatlas builds its debug checks (XA_DEBUG)
        .define("NDEBUG", None)
        .flag_if_supported("-std=c++11")
        .warnings(false)
        .compile("xatlas");
}
