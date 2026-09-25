fn main() {
    // rustc already limits a cdylib's exports to its #[no_mangle] items; this only gives installed copies a soname
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libinference_ffi.so");
    }
}
