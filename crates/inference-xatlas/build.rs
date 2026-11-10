fn main() {
    println!("cargo:rerun-if-changed=third_party/xatlas");
    cc::Build::new()
        .cpp(true)
        .file("third_party/xatlas/xatlas.cpp")
        .define("XATLAS_C_API", "1")
        .flag_if_supported("-std=c++11")
        .warnings(false)
        .compile("xatlas");
}
