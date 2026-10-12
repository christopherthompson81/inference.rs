//! CUDA kernel libraries opened at runtime, so a package can leave out the kernels its users do not need.

/// Where a packaged build's kernel libraries sit, beside the library or binary that loads them.
pub const KERNEL_SUBDIR: &str = "kernels";
/// Overrides every other location: the directory holding `lib<name>.so`.
pub const KERNEL_DIR_ENV: &str = "INFERENCE_KERNEL_DIR";
/// The string every kernel library exports with its build's id; `inference_kernel_build::BUILD_ID_SYMBOL` matches.
pub const BUILD_ID_SYMBOL: &std::ffi::CStr = c"inference_kernel_build_id";
/// Overrides where downloaded kernel libraries are kept: `<dir>/<version>/sm<arch>/lib<name>.so`.
pub const DOWNLOAD_CACHE_ENV: &str = "INFERENCE_KERNEL_CACHE";
/// The build dir file a loader refreshes on open, so a build can prune dirs no binary has used for a while.
pub const BUILD_DIR_LOCK: &str = ".lock";
/// inference-kernel-build's `LIB_CACHE_ENV`: where release-type builds keep the kernel libraries they build.
pub const BUILD_CACHE_ENV: &str = "INFERENCE_KERNEL_BUILD_CACHE";
const CACHE_SUBDIR: &str = "inference-rs/kernels";
const BUILT_SUBDIR: &str = "built";
/// Each GPU arch's libraries sit in `sm<arch>/` (`sm86`, `sm90a`); `inference_kernel_build::ARCH_DIR_PREFIX` matches.
pub const ARCH_DIR_PREFIX: &str = "sm";

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
    /// Where this library was opened from.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

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

// compute capabilities of the visible GPUs (86 for 8.6), none without CUDA
#[cfg(unix)]
fn device_caps() -> &'static [usize] {
    static CAPS: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    CAPS.get_or_init(|| {
        #[cfg(feature = "cuda")]
        {
            let count = crate::cuda_backend::device_count().unwrap_or(0);
            (0..count)
                .filter_map(|ordinal| crate::cuda_backend::device_compute_cap(ordinal).ok())
                .collect()
        }
        #[cfg(not(feature = "cuda"))]
        Vec::new()
    })
}

// the archs this build compiled its kernels for (`86`, `90a`), which the kernel libraries were built with too
#[cfg(unix)]
fn built_archs() -> &'static [&'static str] {
    #[cfg(feature = "cuda")]
    {
        crate::cuda_backend::built_archs()
    }
    #[cfg(not(feature = "cuda"))]
    {
        &[]
    }
}

fn arch_number(arch: &str) -> usize {
    arch.trim_end_matches(['a', 'f']).parse().unwrap_or(0)
}

/// The arch among `archs` each of `caps` runs best, which all must agree on; the highest when there is no GPU.
// code gated on a device's compute capability assumes its kernels were compiled for exactly this arch
pub fn pick_arch<'a>(archs: &[&'a str], caps: &[usize]) -> Result<&'a str, String> {
    let best = |cc: usize| {
        archs
            .iter()
            .copied()
            .filter(|&arch| runs_on(arch, cc))
            .max_by_key(|arch| arch_number(arch))
    };
    let Some(&first) = caps.first() else {
        return archs
            .iter()
            .copied()
            .max_by_key(|arch| arch_number(arch))
            .ok_or_else(|| "this build has no CUDA kernels".to_string());
    };
    let wanted = best(first);
    for &cc in caps {
        match (best(cc), wanted) {
            (Some(arch), Some(want)) if arch == want => {}
            (None, _) | (_, None) => {
                return Err(format!(
                    "a GPU of compute capability {cc} has no kernels in this build (built for sm_{})",
                    archs.join(", sm_")
                ));
            }
            _ => {
                return Err(format!(
                    "the visible GPUs (compute capabilities {caps:?}) need kernels for different archs, and a \
                     process loads one; set CUDA_VISIBLE_DEVICES to GPUs of one arch"
                ));
            }
        }
    }
    wanted.ok_or_else(|| "no GPU".to_string())
}

/// Whether SASS built for `arch` (`86`, `90a`, `121f`) runs on a device of compute capability `cc`.
// plain and family SASS run on later minors of their major; arch-specific only on its own
pub(crate) fn runs_on(arch: &str, cc: usize) -> bool {
    let number = arch_number(arch);
    if arch.ends_with('a') {
        number == cc
    } else {
        number / 10 == cc / 10 && number <= cc
    }
}

/// The arch dir (`sm86`) this process opens kernel libraries from, or why there is none.
#[cfg(unix)]
pub fn process_arch_dir() -> Result<&'static str, String> {
    static DIR: std::sync::OnceLock<Result<String, String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        pick_arch(built_archs(), device_caps()).map(|arch| format!("{ARCH_DIR_PREFIX}{arch}"))
    })
    .as_deref()
    .map_err(Clone::clone)
}

/// Where `lib<name>.so` is looked for, under `arch_dir` (`sm86`) of: the override, this code's dir, its `kernels/`, the build's.
#[cfg(unix)]
pub fn candidates(name: &str, built: &str, arch_dir: &str) -> Vec<std::path::PathBuf> {
    let file = format!("lib{name}.so");
    let mut dirs = Vec::new();
    if let Some(dir) = std::env::var_os(KERNEL_DIR_ENV) {
        dirs.push(std::path::PathBuf::from(dir));
    }
    if let Some(dir) = own_dir() {
        let kernels = dir.join(KERNEL_SUBDIR);
        dirs.extend([dir, kernels]);
    }
    if let Some(dir) = std::path::Path::new(built).parent() {
        dirs.push(dir.to_path_buf());
    }
    dirs.extend(download_dir());
    dirs.into_iter()
        .map(|dir| dir.join(arch_dir).join(&file))
        .collect()
}

/// A kernel library to download: `lib{library}.so` for `arch_dir`, carrying `build_id`, into `dir/arch_dir/`.
pub struct Fetch<'a> {
    pub library: &'a str,
    pub arch_dir: &'a str,
    pub build_id: &'a str,
    pub dir: &'a std::path::Path,
}

/// Downloads a kernel library this build does not have locally.
pub type Fetcher = fn(&Fetch<'_>) -> Result<(), String>;

static FETCHER: std::sync::OnceLock<Fetcher> = std::sync::OnceLock::new();

/// Installs the fetcher [`open`] calls when a library is in none of its [`candidates`]; the first one stays.
pub fn set_fetcher(fetcher: Fetcher) {
    let _ = FETCHER.set(fetcher);
}

/// Where this version's downloaded kernel libraries are kept: `$INFERENCE_KERNEL_CACHE/<version>`, else the user cache.
pub fn download_dir() -> Option<std::path::PathBuf> {
    Some(download_root()?.join(env!("CARGO_PKG_VERSION")))
}

/// Where every version's downloaded kernel libraries are kept, one dir per version.
pub fn download_root() -> Option<std::path::PathBuf> {
    std::env::var_os(DOWNLOAD_CACHE_ENV)
        .map(std::path::PathBuf::from)
        .or_else(|| Some(user_cache()?.join(CACHE_SUBDIR)))
}

/// Where release-type builds keep the kernel libraries they build (inference-kernel-build's `persistent_lib_root`).
pub fn build_cache_root() -> Option<std::path::PathBuf> {
    std::env::var_os(BUILD_CACHE_ENV)
        .map(std::path::PathBuf::from)
        .or_else(|| Some(user_cache()?.join(CACHE_SUBDIR).join(BUILT_SUBDIR)))
}

fn user_cache() -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".cache"))
        })
}

/// Opens kernel library `name`, from the first of [`candidates`] that exists, refusing one whose build id is not
/// `id` (empty when this build made no kernels).
#[cfg(unix)]
pub fn open(name: &str, built: &str, id: &str) -> Result<Library, String> {
    let arch_dir = process_arch_dir()
        .map_err(|error| format!("CUDA kernel library `lib{name}.so`: {error}"))?;
    let tried = candidates(name, built, arch_dir);
    let mut fetched = None;
    if !tried.iter().any(|p| p.is_file())
        && let (Some(fetcher), Some(dir)) = (FETCHER.get(), download_dir())
    {
        fetched = Some(fetcher(&Fetch {
            library: name,
            arch_dir,
            build_id: id,
            dir: &dir,
        }));
    }
    let Some(path) = tried.iter().find(|p| p.is_file()) else {
        let tried: Vec<_> = tried.iter().map(|p| p.display().to_string()).collect();
        let fetch = match fetched {
            Some(Err(error)) => format!("; not downloaded: {error}"),
            _ => String::new(),
        };
        return Err(format!(
            "CUDA kernel library `lib{name}.so` is not installed for this GPU's arch ({arch_dir}; looked in {}){fetch}; \
             set {KERNEL_DIR_ENV} to a directory holding `{arch_dir}/lib{name}.so`",
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
    // a shared lock held for the process's life, and a fresh mtime, keep pruning off a build dir in use
    if let Some(build_dir) = path.parent().and_then(std::path::Path::parent)
        && let Ok(lock) = std::fs::File::options()
            .write(true)
            .open(build_dir.join(BUILD_DIR_LOCK))
    {
        let _ = lock.set_modified(std::time::SystemTime::now());
        if lock.try_lock_shared().is_ok() {
            static HELD: std::sync::Mutex<Vec<std::fs::File>> = std::sync::Mutex::new(Vec::new());
            HELD.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(lock);
        }
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
        // a kernel only some archs build is missing from the other archs' copies of the library
        #[allow(non_snake_case)]
        struct KernelTable {
            kernel_library_path: ::std::path::PathBuf,
            $( $(#[$attr])* $name: ::std::option::Option<unsafe extern "C" fn( $( $ty ),* ) $( -> $ret )?>, )*
        }

        static KERNEL_TABLE: ::std::sync::OnceLock<::std::result::Result<KernelTable, ::std::string::String>> =
            ::std::sync::OnceLock::new();

        #[allow(unused_doc_comments)]
        fn kernel_table() -> ::std::result::Result<&'static KernelTable, ::std::string::String> {
            KERNEL_TABLE
                .get_or_init(|| {
                    let library = $crate::kernel_library::open($lib, $path, $id)?;
                    ::std::result::Result::Ok(KernelTable {
                        kernel_library_path: library.path().to_path_buf(),
                        $(
                            $(#[$attr])*
                            // SAFETY: `open` checked the library is this binary's own build, from the same header
                            $name: library
                                .symbol(
                                    ::std::ffi::CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes())
                                        .map_err(|error| error.to_string())?,
                                )
                                .ok()
                                .map(|address| unsafe {
                                    ::std::mem::transmute::<
                                        *mut ::std::ffi::c_void,
                                        unsafe extern "C" fn( $( $ty ),* ) $( -> $ret )?,
                                    >(address)
                                }),
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
                let ::std::option::Option::Some(function) = table.$name else {
                    panic!(
                        "CUDA kernel `{}` is not in `{}`: it is not built for this GPU's architecture",
                        stringify!($name),
                        table.kernel_library_path.display()
                    );
                };
                // SAFETY: the caller upholds the C function's contract, as with a direct extern call
                unsafe { function( $( $arg ),* ) }
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
        assert!(error.contains("libnonexistentkernels.so"), "{error}");
    }

    #[test]
    fn the_arch_picked_is_the_one_each_gpu_runs_best() {
        let archs = ["80", "86", "90a", "121f"];
        assert_eq!(pick_arch(&archs, &[86]), Ok("86"));
        assert_eq!(pick_arch(&archs, &[89]), Ok("86"));
        assert_eq!(pick_arch(&archs, &[86, 89]), Ok("86"));
        assert_eq!(pick_arch(&archs, &[90]), Ok("90a"));
        assert_eq!(pick_arch(&archs, &[121]), Ok("121f"));
        assert_eq!(pick_arch(&archs, &[]), Ok("121f"));
        // an 8.0 copy would run on 8.9 too, but kernels gated on 8.6 features would find their bodies compiled out
        let mixed = pick_arch(&archs, &[80, 89]).unwrap_err();
        assert!(mixed.contains("CUDA_VISIBLE_DEVICES"), "{mixed}");
        assert!(pick_arch(&archs, &[86, 90]).is_err());
        assert!(pick_arch(&archs, &[75]).is_err());
        assert!(pick_arch(&["90a"], &[100]).is_err());
    }

    #[test]
    fn the_build_path_comes_after_the_packaged_ones_and_downloads_last() {
        let paths = candidates("somekernels", "/built/libsomekernels.so", "sm86");
        let built = paths
            .iter()
            .position(|p| p == std::path::Path::new("/built/sm86/libsomekernels.so"));
        assert_eq!(
            built,
            Some(paths.len() - 1 - usize::from(download_dir().is_some())),
            "{paths:?}"
        );
        if let Some(dir) = download_dir() {
            assert_eq!(paths.last(), Some(&dir.join("sm86/libsomekernels.so")));
        }
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
        let paths = super::candidates("somekernels", "/built/libsomekernels.so", "sm86");
        let relative: Vec<_> = paths.iter().filter(|p| !p.is_absolute()).collect();
        assert!(
            relative.is_empty() || std::env::var_os(super::KERNEL_DIR_ENV).is_some(),
            "{relative:?}"
        );
    }
}
