//! CUDA kernel libraries opened at runtime, so a package can leave out the kernels its users do not need.

/// Where a packaged build's kernel libraries sit, beside the library or binary that loads them.
pub const KERNEL_SUBDIR: &str = "kernels";
/// Overrides every other location: the directory holding `lib<name>.so`.
pub const KERNEL_DIR_ENV: &str = "INFERENCE_KERNEL_DIR";
/// The string every kernel library exports with its build's id; `inference_kernel_build::BUILD_ID_SYMBOL` matches.
pub const BUILD_ID_SYMBOL: &std::ffi::CStr = c"inference_kernel_build_id";

/// A kernel library kept open for the life of the process.
#[cfg(unix)]
pub struct Library {
    handle: *mut libc::c_void,
    path: std::path::PathBuf,
}

// SAFETY: a dlopen handle may be used from any thread; it is never closed.
#[cfg(unix)]
unsafe impl Send for Library {}
#[cfg(unix)]
unsafe impl Sync for Library {}

#[cfg(unix)]
impl Library {
    /// The address of `symbol`.
    pub fn symbol(&self, symbol: &std::ffi::CStr) -> Result<*mut libc::c_void, String> {
        // SAFETY: the handle is open, and `symbol` is NUL-terminated
        let address = unsafe { libc::dlsym(self.handle, symbol.as_ptr()) };
        if address.is_null() {
            return Err(format!(
                "kernel library `{}` has no `{}`: it is from another build",
                self.path.display(),
                symbol.to_string_lossy()
            ));
        }
        Ok(address)
    }
}

// the directory of the binary or library this code was linked into
#[cfg(unix)]
fn own_dir() -> Option<std::path::PathBuf> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: any address inside this object identifies it
    if unsafe { libc::dladdr(own_dir as *const libc::c_void, &mut info) } == 0
        || info.dli_fname.is_null()
    {
        return None;
    }
    // SAFETY: dladdr filled a NUL-terminated path
    let path = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) };
    let path: &std::ffi::OsStr = std::os::unix::ffi::OsStrExt::from_bytes(path.to_bytes());
    let path = std::path::Path::new(path);
    // the main executable reports argv[0], which for a binary found on PATH is relative to nothing useful: a
    // relative directory would load libraries from the working directory
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_exe().ok()?
    };
    path.parent().map(std::path::Path::to_path_buf)
}

/// The places `lib<name>.so` is looked for, in order: the override, beside this code, its `kernels/`, the build's.
#[cfg(unix)]
pub fn candidates(name: &str, built: &str) -> Vec<std::path::PathBuf> {
    let file = format!("lib{name}.so");
    let mut paths = Vec::new();
    if let Some(dir) = std::env::var_os(KERNEL_DIR_ENV) {
        paths.push(std::path::PathBuf::from(dir).join(&file));
    }
    if let Some(dir) = own_dir() {
        paths.push(dir.join(&file));
        paths.push(dir.join(KERNEL_SUBDIR).join(&file));
    }
    paths.push(std::path::PathBuf::from(built));
    paths
}

/// Opens kernel library `name`, from the first of [`candidates`] that exists, refusing one whose build id is not
/// `id` (empty when this build made no kernels).
#[cfg(unix)]
pub fn open(name: &str, built: &str, id: &str) -> Result<Library, String> {
    let tried = candidates(name, built);
    let Some(path) = tried.iter().find(|p| p.is_file()) else {
        let tried: Vec<_> = tried.iter().map(|p| p.display().to_string()).collect();
        return Err(format!(
            "CUDA kernel library `lib{name}.so` is not installed (looked in {}); set {KERNEL_DIR_ENV} to its directory",
            tried.join(", ")
        ));
    };
    let c_path = std::ffi::CString::new(std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str()))
        .map_err(|e| e.to_string())?;
    // SAFETY: a NUL-terminated path; the handle is kept for the process's life
    let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        // SAFETY: dlerror describes the dlopen that just failed
        let error = unsafe { std::ffi::CStr::from_ptr(libc::dlerror()) };
        return Err(format!(
            "CUDA kernel library `{}` failed to load: {}",
            path.display(),
            error.to_string_lossy()
        ));
    }
    let library = Library {
        handle,
        path: path.clone(),
    };
    let found = library.symbol(BUILD_ID_SYMBOL)?;
    // SAFETY: the symbol is the NUL-terminated id string the build linked into the library
    let found = unsafe { std::ffi::CStr::from_ptr(found.cast()) }.to_string_lossy();
    if found != id {
        return Err(format!(
            "CUDA kernel library `{}` is from another build (id {found}, this binary expects {id}); install the \
             kernels that came with it, or set {KERNEL_DIR_ENV}",
            path.display()
        ));
    }
    Ok(library)
}

/// An `extern "C"` block whose functions live in kernel library `library` (built at `path`), called through a table.
#[cfg(unix)]
#[macro_export]
macro_rules! kernel_library {
    (
        library = $lib:expr;
        path = $path:expr;
        id = $id:expr;
        unsafe extern "C" {
            $(
                $(#[$attr:meta])*
                $vis:vis fn $name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ) $( -> $ret:ty )?;
            )*
        }
    ) => {
        #[allow(non_snake_case)]
        struct KernelTable {
            $( $(#[$attr])* $name: unsafe extern "C" fn( $( $ty ),* ) $( -> $ret )?, )*
        }

        static KERNEL_TABLE: ::std::sync::OnceLock<::std::result::Result<KernelTable, ::std::string::String>> =
            ::std::sync::OnceLock::new();

        #[allow(unused_doc_comments)]
        fn kernel_table() -> ::std::result::Result<&'static KernelTable, ::std::string::String> {
            KERNEL_TABLE
                .get_or_init(|| {
                    let library = $crate::kernel_library::open($lib, $path, $id)?;
                    ::std::result::Result::Ok(KernelTable {
                        $(
                            $(#[$attr])*
                            // SAFETY: `open` checked the library is this binary's own build, from the same header
                            $name: unsafe {
                                ::std::mem::transmute::<
                                    *mut ::std::ffi::c_void,
                                    unsafe extern "C" fn( $( $ty ),* ) $( -> $ret )?,
                                >(library.symbol(
                                    ::std::ffi::CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes())
                                        .map_err(|error| error.to_string())?,
                                )?)
                            },
                        )*
                    })
                })
                .as_ref()
                .map_err(::std::clone::Clone::clone)
        }

        /// Loads this kernel library now, so a missing one fails at model load rather than at the first launch.
        #[allow(dead_code)]
        pub fn ensure_loaded() -> ::std::result::Result<(), ::std::string::String> {
            kernel_table().map(|_| ())
        }

        $(
            $(#[$attr])*
            // the C ABI, so callers can still pass these as `extern "C"` function pointers
            #[allow(clippy::too_many_arguments, non_snake_case)]
            $vis unsafe extern "C" fn $name( $( $arg : $ty ),* ) $( -> $ret )? {
                let table = kernel_table().unwrap_or_else(|error| panic!("{error}"));
                // SAFETY: the caller upholds the C function's contract, as with a direct extern call
                unsafe { (table.$name)( $( $arg ),* ) }
            }
        )*
    };
}

/// On targets without `dlopen` the block links as written.
#[cfg(not(unix))]
#[macro_export]
macro_rules! kernel_library {
    (
        library = $lib:expr;
        path = $path:expr;
        id = $id:expr;
        unsafe extern "C" { $($body:tt)* }
    ) => {
        unsafe extern "C" { $($body)* }

        /// Linked statically here, so always loaded.
        pub fn ensure_loaded() -> ::std::result::Result<(), ::std::string::String> {
            ::std::result::Result::Ok(())
        }
    };
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_names_itself_and_the_override() {
        let Err(error) = open(
            "nonexistentkernels",
            "/nonexistent/libnonexistentkernels.so",
            "",
        ) else {
            panic!("a library that is nowhere opened");
        };
        assert!(
            error.contains("libnonexistentkernels.so") && error.contains(KERNEL_DIR_ENV),
            "{error}"
        );
    }

    #[test]
    fn the_build_path_is_looked_in_last() {
        let paths = candidates("somekernels", "/built/libsomekernels.so");
        assert_eq!(
            paths
                .last()
                .map(|p| p.to_string_lossy().into_owned())
                .as_deref(),
            Some("/built/libsomekernels.so")
        );
        let beside = paths
            .iter()
            .filter(|p| p.ends_with("libsomekernels.so"))
            .count();
        assert!(beside >= 2, "{paths:?}");
    }
}

#[cfg(all(test, unix))]
mod location_tests {
    #[test]
    fn libraries_are_only_looked_for_at_absolute_paths() {
        let paths = super::candidates("somekernels", "/built/libsomekernels.so");
        let relative: Vec<_> = paths.iter().filter(|p| !p.is_absolute()).collect();
        assert!(
            relative.is_empty() || std::env::var_os(super::KERNEL_DIR_ENV).is_some(),
            "{relative:?}"
        );
    }
}
