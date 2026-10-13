//! Main kernel builder implementation

use crate::compute_cap::{ComputeCapability, GpuArch, arch_key, gencode_args};
use crate::dependency::DependencyManager;
use crate::error::{Error, Result};
use crate::hash::{BuildCache, hash_args, hash_paths};
use crate::parallel::ParallelConfig;
use crate::source::SourceSelector;
use crate::toolkit::CudaToolkit;

use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

const SIZE_COMPRESSION_SINCE: (u32, u32) = (12, 8);

/// A build dir's lock: a build holds it exclusively, a binary using the dir's libraries shared for its life;
/// `inference_tensor::kernel_library::BUILD_DIR_LOCK` matches.
pub const BUILD_DIR_LOCK: &str = ".lock";
#[cfg(target_os = "linux")]
const PROC_LOCKS: &str = "/proc/locks";

/// Each GPU arch's copy of a loadable library sits in `sm<arch>/`; `inference_tensor::kernel_library` matches it.
pub const ARCH_DIR_PREFIX: &str = "sm";

/// The directory a per-arch build links `arch`'s libraries into: `sm86`, `sm90a`.
pub fn arch_dir(arch: &GpuArch) -> String {
    format!(
        "{ARCH_DIR_PREFIX}{}",
        arch.to_nvcc_arch().trim_start_matches("sm_")
    )
}

// Waits out another build, which ends. A running binary holds its shared lock for its life, so then the build goes on
// without the lock, and the error it returns is for the moment the build would replace a library.
fn lock_build_dir(lock: &std::fs::File, dir: &Path) -> Result<Option<Error>> {
    use std::fs::TryLockError;
    match lock.try_lock() {
        Ok(()) => return Ok(None),
        Err(TryLockError::WouldBlock) => {}
        Err(TryLockError::Error(error)) => return Err(error.into()),
    }
    match lock.try_lock_shared() {
        Ok(()) => {
            lock.unlock()?;
            Ok(Some(Error::KernelLibrariesInUse {
                dir: dir.to_path_buf(),
                holders: lock_holders(lock),
            }))
        }
        Err(TryLockError::WouldBlock) => {
            lock.lock()?;
            Ok(None)
        }
        Err(TryLockError::Error(error)) => Err(error.into()),
    }
}

// the processes holding `lock`, from the kernel's lock table
#[cfg(target_os = "linux")]
fn lock_holders(lock: &std::fs::File) -> String {
    use std::os::unix::fs::MetadataExt;
    let Ok(inode) = lock.metadata().map(|metadata| metadata.ino()) else {
        return "a running process".to_string();
    };
    let table = std::fs::read_to_string(PROC_LOCKS).unwrap_or_default();
    let mut holders: Vec<String> = table
        .lines()
        .filter_map(|line| {
            // `1: FLOCK  ADVISORY  READ 12614 103:05:22548678 0 EOF`
            let fields: Vec<&str> = line.split_whitespace().collect();
            let file = fields.get(5)?;
            (file.rsplit(':').next()?.parse::<u64>().ok()? == inode)
                .then(|| fields.get(4).copied())?
        })
        .map(|pid| {
            let command = std::fs::read(format!("/proc/{pid}/cmdline"))
                .map(|bytes| {
                    String::from_utf8_lossy(&bytes)
                        .replace('\0', " ")
                        .trim()
                        .to_string()
                })
                .unwrap_or_default();
            format!("pid {pid} ({command})")
        })
        .collect();
    holders.dedup();
    if holders.is_empty() {
        "a running process".to_string()
    } else {
        holders.join(", ")
    }
}

#[cfg(not(target_os = "linux"))]
fn lock_holders(_: &std::fs::File) -> String {
    "a running process built from this target dir".to_string()
}

// a kernel source and the object compiled from it, for one arch dir (`sm86`) in a per-arch build
struct Compiled {
    source: PathBuf,
    arch: Option<String>,
    object: PathBuf,
}

// one loadable library: the nominal `<dir>/lib<name>.so` the crate records, and its per-arch copies
struct LoadableLib {
    name: String,
    nominal: PathBuf,
    copies: Vec<PathBuf>,
}

/// Main builder for CUDA kernel compilation
#[derive(Debug)]
pub struct KernelBuilder {
    /// CUDA toolkit configuration
    toolkit: Option<CudaToolkit>,
    /// Compute capability configuration
    compute_cap: ComputeCapability,
    /// Source file selection
    sources: SourceSelector,
    /// External dependencies
    dependencies: DependencyManager,
    /// Parallel build configuration
    parallel: ParallelConfig,
    /// Output directory
    out_dir: PathBuf,
    /// Extra nvcc arguments
    extra_args: Vec<String>,
    /// Whether to use incremental builds
    incremental: bool,
    /// Libraries split off a loadable build: (library name, source file pattern)
    split_libs: Vec<(String, String)>,
}

impl Default for KernelBuilder {
    fn default() -> Self {
        let out_dir = std::env::var("OUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("target/debug"));

        Self {
            toolkit: None,
            compute_cap: ComputeCapability::default(),
            sources: SourceSelector::default(),
            dependencies: DependencyManager::default(),
            parallel: ParallelConfig::default(),
            out_dir,
            extra_args: Vec::new(),
            incremental: true,
            split_libs: Vec::new(),
        }
    }
}

impl KernelBuilder {
    /// Create a new kernel builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    // Source Selection

    /// Add a directory to search for .cu files (recursive)
    pub fn source_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.sources = self.sources.add_directory(dir);
        self
    }

    /// Add specific kernel files
    pub fn source_files<I, P>(mut self, files: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        self.sources = self.sources.add_files(files);
        self
    }

    /// Add kernel files matching a glob pattern
    pub fn source_glob(mut self, pattern: &str) -> Self {
        self.sources = self.sources.add_glob(pattern);
        self
    }

    /// Exclude files matching patterns
    pub fn exclude(mut self, patterns: &[&str]) -> Self {
        self.sources = self.sources.exclude(patterns);
        self
    }

    /// Add paths to watch for changes (headers, etc.)
    pub fn watch<I, P>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        self.sources = self.sources.watch(paths);
        self
    }

    // Compute Capability

    /// Set the default compute capability (numeric, auto-selects 'a' suffix for sm_90+)
    pub fn compute_cap(mut self, cap: usize) -> Self {
        self.compute_cap = self.compute_cap.with_default(cap);
        self
    }

    /// Set the default compute capability with explicit arch string (e.g., "90a", "100a", "80")
    ///
    /// Use this when you need explicit control over the architecture suffix.
    pub fn compute_cap_arch(mut self, arch: &str) -> Self {
        self.compute_cap = self.compute_cap.with_default_arch(arch);
        self
    }

    /// Set compute cap override for specific kernels (numeric)
    ///
    /// Pattern can use wildcards: "sm90_*.cu", "*_hopper.cu"
    pub fn with_compute_override(mut self, pattern: &str, cap: usize) -> Self {
        self.compute_cap = self.compute_cap.with_override(pattern, cap);
        self
    }

    /// Set compute cap override with explicit arch string (e.g., "90a", "100a")
    pub fn with_compute_override_arch(mut self, pattern: &str, arch: &str) -> Self {
        self.compute_cap = self.compute_cap.with_override_arch(pattern, arch);
        self
    }

    /// Get the lowest default compute capability (base number only), the one compile-time minimums must hold for
    pub fn get_compute_cap(&self) -> Option<usize> {
        self.compute_cap.get_default().ok().map(|a| a.base)
    }

    /// Get every default compute capability (base numbers), lowest first
    pub fn get_compute_caps(&self) -> Vec<usize> {
        self.compute_cap
            .get_defaults()
            .map(|archs| archs.iter().map(GpuArch::base).collect())
            .unwrap_or_default()
    }

    /// Set compute capability (mutable reference version)
    pub fn set_compute_cap(&mut self, cap: usize) {
        self.compute_cap = ComputeCapability::new().with_default(cap);
    }

    /// Require explicit compute capability (fail fast if not set)
    ///
    /// Use this for Docker builds or CI environments where nvidia-smi is unavailable.
    /// The build will fail immediately if CUDA_COMPUTE_CAP is not set and no
    /// compute capability was explicitly configured via `.compute_cap()`.
    ///
    /// # Example
    /// ```no_run
    /// use inference_kernel_build::KernelBuilder;
    ///
    /// // For Docker builds, require explicit compute cap
    /// KernelBuilder::new()
    ///     .require_explicit_compute_cap()?  // Fails if CUDA_COMPUTE_CAP not set
    ///     .source_dir("src/kernels")
    ///     .build_lib("libkernels.a")?;
    /// # Ok::<(), inference_kernel_build::Error>(())
    /// ```
    pub fn require_explicit_compute_cap(self) -> Result<Self> {
        // Check if compute cap is already set
        if self.compute_cap.get_default().is_ok() {
            return Ok(self);
        }

        // Check environment variable
        if std::env::var("CUDA_COMPUTE_CAP").is_ok() {
            return Ok(self);
        }

        Err(Error::ComputeCapDetectionFailed(
            "Explicit compute capability required but not set. \
            Either call .compute_cap(N) on the builder or set CUDA_COMPUTE_CAP environment variable. \
            This is required for Docker builds where nvidia-smi is unavailable.".to_string()
        ))
    }

    // External Dependencies

    /// Add CUTLASS dependency
    pub fn with_cutlass(mut self, commit: Option<&str>) -> Self {
        self.dependencies = self.dependencies.with_cutlass(commit);
        self
    }

    /// Add a custom git dependency
    /// If `recurse_submodules` is false, clone/fetch adds --no-recurse-submodules.
    pub fn with_git_dependency(
        mut self,
        name: &str,
        repo: &str,
        commit: &str,
        include_paths: Vec<&str>,
        extra_paths: Vec<&str>,
        recurse_submodules: bool,
    ) -> Self {
        self.dependencies = self.dependencies.with_git_dependency(
            name,
            repo,
            commit,
            include_paths,
            extra_paths,
            recurse_submodules,
        );
        self
    }

    /// Fetch a configured git dependency and return its checkout root.
    pub fn fetch_git_dependency(&self, name: &str) -> Result<PathBuf> {
        self.dependencies.fetch_dependency(name, &self.out_dir)
    }

    /// Add a local include path
    pub fn include_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.dependencies = self.dependencies.with_local_include(path);
        self
    }

    // Parallel Configuration

    /// Set the percentage of available threads to use (0.0 - 1.0)
    pub fn thread_percentage(mut self, percentage: f32) -> Self {
        self.parallel = self.parallel.with_percentage(percentage);
        self
    }

    /// Set the maximum number of threads
    pub fn max_threads(mut self, max: usize) -> Self {
        self.parallel = self.parallel.with_max_threads(max);
        self
    }

    /// Set patterns for files that should use nvcc threads
    ///
    /// This replaces the default patterns ("flash_api", "cutlass").
    /// `num_nvcc_threads` controls the `--threads=N` argument passed to nvcc for matching files.
    pub fn nvcc_thread_patterns<S: AsRef<str>>(
        mut self,
        patterns: &[S],
        num_nvcc_threads: usize,
    ) -> Self {
        self.parallel = self
            .parallel
            .with_nvcc_thread_patterns(patterns, num_nvcc_threads);
        self
    }

    // Build Configuration

    /// Set the output directory
    pub fn out_dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.out_dir = dir.into();
        self
    }

    /// Add an extra nvcc argument
    pub fn arg(mut self, arg: &str) -> Self {
        self.extra_args.push(arg.to_string());
        self
    }

    /// Add multiple extra nvcc arguments
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for arg in args {
            self.extra_args.push(arg.as_ref().to_string());
        }
        self
    }

    /// Compress the embedded fatbins: `-compress-mode=size` from CUDA 12.8 (as llama.cpp does), `-compress-all` before.
    pub fn compress_fatbin(mut self) -> Self {
        if self.toolkit.is_none() {
            self.toolkit = CudaToolkit::detect().ok();
        }
        let version = self.toolkit.as_ref().and_then(|t| t.version.as_deref());
        let parsed = version.and_then(parse_major_minor);
        if parsed.is_none() {
            println!(
                "cargo:warning=inference-kernel-build: unknown CUDA version {version:?}, falling back to -Xfatbin=-compress-all"
            );
        }
        let flag = if parsed >= Some(SIZE_COMPRESSION_SINCE) {
            "-compress-mode=size"
        } else {
            "-Xfatbin=-compress-all"
        };
        self.extra_args.push(flag.to_string());
        self
    }

    /// Link the sources matching `pattern` into their own loadable library `name`, so a package can ship them apart.
    pub fn split_lib(mut self, name: &str, pattern: &str) -> Self {
        self.split_libs
            .push((name.to_string(), pattern.to_string()));
        self
    }

    /// Disable incremental builds
    pub fn no_incremental(mut self) -> Self {
        self.incremental = false;
        self
    }

    /// Set explicit CUDA toolkit path
    pub fn cuda_root<P: AsRef<Path>>(mut self, path: P) -> Self {
        if let Ok(toolkit) = CudaToolkit::from_nvcc_path(path.as_ref().join("bin").join("nvcc")) {
            self.toolkit = Some(toolkit);
        }
        self
    }

    // Build Methods

    /// Build a static library from all kernel sources
    pub fn build_lib<P: Into<PathBuf>>(&self, out_file: P) -> Result<()> {
        let out_file = out_file.into();
        self.emit_cuda_link_search()?;
        let Some((toolkit, objects)) =
            self.compile_objects(std::slice::from_ref(&out_file), None)?
        else {
            return Ok(());
        };
        let mut command = Command::new(&toolkit.nvcc_path);
        command
            .arg("--lib")
            .arg("-o")
            .arg(&out_file)
            .args(objects.iter().map(|c| &c.object));
        run_link(command)
    }

    /// Build `lib{name}.so` and each [`Self::split_lib`] for `kernel_library!` to open, recorded in [`loadable_lib_env`].
    // targets without dlopen build and link one `archive` of every source instead
    pub fn build_loadable_lib(self, name: &str, archive: PathBuf) -> Result<()> {
        let dlopen = std::env::var("TARGET").is_ok_and(|t| !t.contains("windows"));
        if !dlopen {
            self.build_lib(&archive)?;
            if let Some(dir) = archive.parent() {
                println!("cargo:rustc-link-search={}", dir.display());
            }
            println!("cargo:rustc-link-lib={name}");
            return Ok(());
        }
        // a release-type build may be `cargo install`, whose target dir is deleted: its libraries live on beside it
        let root = if dev_shared_libs() {
            shared_lib_root()?
        } else {
            persistent_lib_root()?
        };
        let pointers = shared_lib_root()?;
        std::fs::create_dir_all(&pointers)?;
        for LoadableLib {
            name: lib,
            nominal: out_file,
            copies: variants,
        } in self.build_shared_objects(name, &root, !dev_shared_libs())?
        {
            println!(
                "cargo:rustc-env={}={}",
                loadable_lib_env(&lib),
                out_file.display()
            );
            println!(
                "cargo:rustc-env={}={}",
                loadable_id_env(&lib),
                Self::built_id(&out_file)?
            );
            // a deleted library must be rebuilt, or the path baked into the crate would point at nothing
            for variant in &variants {
                println!("cargo:rerun-if-changed={}", variant.display());
            }
            // packaging reads which library this build made, among the variants a target dir accumulates
            let pointer = pointers.join(format!("{lib}.{LIB_POINTER_EXTENSION}"));
            let tmp = pointers.join(format!("{lib}.{LIB_POINTER_EXTENSION}.tmp"));
            std::fs::write(&tmp, out_file.display().to_string())?;
            std::fs::rename(&tmp, &pointer)?;
        }
        Ok(())
    }

    // the library each source links into: the first split whose pattern matches it, else `name`
    fn library_of<'a>(&'a self, name: &'a str, source: &Path) -> &'a str {
        let filename = source.file_name().and_then(|n| n.to_str()).unwrap_or("");
        self.split_libs
            .iter()
            .find(|(_, pattern)| crate::compute_cap::matches_pattern(filename, pattern))
            .map_or(name, |(lib, _)| lib)
    }

    // Each library's nominal `<dir>/lib<name>.so` (the path the crate records), with the per-arch files linked for it.
    // `strip` drops the static symbol table; the dynamic one the loader resolves through stays
    fn build_shared_objects(
        mut self,
        name: &str,
        root: &Path,
        strip: bool,
    ) -> Result<Vec<LoadableLib>> {
        let toolkit = match self.toolkit.take() {
            Some(t) => t,
            None => CudaToolkit::detect()?,
        };
        println!(
            "cargo:rustc-link-search=native={}",
            toolkit.lib_dir.display()
        );
        let mut key = DefaultHasher::new();
        name.hash(&mut key);
        self.extra_args.hash(&mut key);
        self.split_libs.hash(&mut key);
        format!("{:?}", self.dependencies).hash(&mut key);
        // the object cache does not track the compiler, so a toolkit switch must land in a fresh dir
        toolkit.nvcc_path.hash(&mut key);
        toolkit.version.hash(&mut key);
        std::env::var("NVCC_CCBIN").ok().hash(&mut key);
        self.toolkit = Some(toolkit);
        let sources = self.sources.resolve()?;
        for file in &sources {
            let filename = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // absolute, so worktrees sharing a target dir never share a dir
            file.canonicalize()?.hash(&mut key);
            arch_key(&self.compute_cap.get_for_file(filename)?).hash(&mut key);
        }
        let dir = root.join(format!("{name}-{:016x}", key.finish()));
        std::fs::create_dir_all(&dir)?;
        let lock = std::fs::File::create(dir.join(BUILD_DIR_LOCK))?;
        // an up-to-date build changes nothing a running binary holds, so only a rebuild is refused
        let in_use = lock_build_dir(&lock, &dir)?;

        self.out_dir = dir.clone();
        let mut libs: Vec<String> = vec![name.to_string()];
        for source in &sources {
            let lib = self.library_of(name, source);
            if !libs.iter().any(|l| l == lib) {
                libs.push(lib.to_string());
            }
        }
        let archs: Vec<GpuArch> = self.compute_cap.get_defaults()?;
        let variant_path =
            |lib: &str, arch: &GpuArch| dir.join(arch_dir(arch)).join(format!("lib{lib}.so"));
        let nominal: Vec<PathBuf> = libs
            .iter()
            .map(|lib| dir.join(format!("lib{lib}.so")))
            .collect();
        let variants: Vec<Vec<PathBuf>> = libs
            .iter()
            .map(|lib| archs.iter().map(|arch| variant_path(lib, arch)).collect())
            .collect();
        for (out_file, lib_variants) in nominal.iter().zip(&variants).filter(|_| in_use.is_none()) {
            // a single-arch library from an older layout; the loader opens only the per-arch copies
            let _ = std::fs::remove_file(out_file);
            // a library from before ids were linked in is rebuilt
            if !PathBuf::from(format!("{}.{BUILD_ID_EXTENSION}", out_file.display())).is_file() {
                for variant in lib_variants {
                    let _ = std::fs::remove_file(variant);
                }
            }
        }
        let all_variants: Vec<PathBuf> = variants.iter().flatten().cloned().collect();
        if let Some((toolkit, objects)) = self.compile_objects(&all_variants, Some(&archs))? {
            if let Some(in_use) = in_use {
                return Err(in_use);
            }
            for (lib, out_file) in libs.iter().zip(&nominal) {
                let own: Vec<&Compiled> = objects
                    .iter()
                    .filter(|c| self.library_of(name, &c.source) == lib)
                    .collect();
                // one id over every arch's objects, so each arch's copy matches the one crate build
                let id = build_id(&own.iter().map(|c| c.object.clone()).collect::<Vec<_>>())?;
                let id_object = compile_build_id(&toolkit, &dir, lib, &id)?;
                for arch in &archs {
                    let target = variant_path(lib, arch);
                    std::fs::create_dir_all(target.parent().expect("variant has a dir"))?;
                    // Link aside and rename, so binaries that already map the old library keep working.
                    let tmp = PathBuf::from(format!("{}.tmp", target.display()));
                    let mut command = Command::new(&toolkit.nvcc_path);
                    command
                        .arg("-shared")
                        // nothing builds relocatable device code, and the device link would add an empty default-arch cubin
                        .arg("--no-device-link")
                        .args(["--cudart", "shared"])
                        .arg("-o")
                        .arg(&tmp)
                        .arg(format!("-Xlinker=-soname=lib{lib}.so"))
                        .arg("-Xlinker=--no-undefined")
                        .args(
                            own.iter()
                                .filter(|c| c.arch.as_deref() == Some(arch_dir(arch).as_str()))
                                .map(|c| &c.object),
                        )
                        .arg(&id_object);
                    if strip {
                        command.arg("-Xlinker=-s");
                    }
                    run_link(command)?;
                    std::fs::rename(&tmp, &target)?;
                }
                std::fs::write(format!("{}.{BUILD_ID_EXTENSION}", out_file.display()), &id)?;
            }
        }
        Ok(libs
            .into_iter()
            .zip(nominal)
            .zip(variants)
            .map(|((name, nominal), copies)| LoadableLib {
                name,
                nominal,
                copies,
            })
            .collect())
    }

    // the build id linked into the library at `out_file`
    fn built_id(out_file: &Path) -> Result<String> {
        let id_file = PathBuf::from(format!("{}.{BUILD_ID_EXTENSION}", out_file.display()));
        Ok(std::fs::read_to_string(id_file)?)
    }

    /// Kernel libraries link cudart, whose directory nothing else puts on the search path once cudarc loads at runtime.
    fn emit_cuda_link_search(&self) -> Result<()> {
        let lib_dir = match &self.toolkit {
            Some(t) => t.lib_dir.clone(),
            None => CudaToolkit::detect()?.lib_dir,
        };
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        Ok(())
    }

    /// Compile every out-of-date kernel into `out_dir`, once per arch in `per_arch`; `None` when `out_files` are current.
    fn compile_objects(
        &self,
        out_files: &[PathBuf],
        per_arch: Option<&[GpuArch]>,
    ) -> Result<Option<(CudaToolkit, Vec<Compiled>)>> {
        crate::jobserver::init();

        // Detect toolkit if not set
        let toolkit = match &self.toolkit {
            Some(t) => t.clone(),
            None => CudaToolkit::detect()?,
        };

        // Initialize thread pool
        let _ = self.parallel.init_thread_pool();

        println!(
            "cargo:warning=Using up to {} threads for compilation, within cargo's jobserver",
            self.parallel.thread_count()
        );

        // Create output directory
        std::fs::create_dir_all(&self.out_dir)?;

        // Resolve source files
        let kernel_files = self.sources.resolve()?;
        if kernel_files.is_empty() {
            println!("cargo:warning=No kernel files found");
            return Ok(None);
        }

        // Emit cargo:rerun-if-changed directives
        for file in &kernel_files {
            println!("cargo:rerun-if-changed={}", file.display());
        }
        for path in self.sources.watch_paths() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
        println!("cargo:rerun-if-env-changed=NVCC");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

        // Fetch external dependencies
        let dep_args = self.dependencies.fetch_all(&self.out_dir)?;

        // Load build cache
        let mut cache = if self.incremental {
            BuildCache::load(&self.out_dir)
        } else {
            BuildCache::default()
        };

        // Calculate args hash for cache
        let mut all_args = self.extra_args.clone();
        all_args.extend(dep_args.clone());
        let args_hash = hash_args(&all_args);

        // Calculate watch hash for cache
        let watch_hash = hash_paths(self.sources.watch_paths());

        // Determine which files need compilation
        let mut compile_jobs: Vec<(PathBuf, PathBuf, Vec<GpuArch>)> = Vec::new();
        let mut all_obj_files: Vec<Compiled> = Vec::new();
        let mut units: Vec<(PathBuf, Option<String>, Vec<GpuArch>)> = Vec::new();
        for kernel_file in &kernel_files {
            let filename = kernel_file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let gpu_arch = self.compute_cap.get_for_file(filename)?;
            let Some(variants) = per_arch else {
                units.push((kernel_file.clone(), None, gpu_arch));
                continue;
            };
            let before = units.len();
            for variant in variants {
                // the variant itself, or an override of the same arch (`90a` for a listed `90`)
                let chosen = gpu_arch
                    .iter()
                    .find(|a| *a == variant)
                    .or_else(|| gpu_arch.iter().find(|a| a.base == variant.base));
                if let Some(arch) = chosen {
                    units.push((
                        kernel_file.clone(),
                        Some(arch_dir(variant)),
                        vec![arch.clone()],
                    ));
                }
            }
            if units.len() == before {
                return Err(Error::CompilationFailed {
                    path: kernel_file.clone(),
                    message: format!(
                        "builds only for {}, none of the listed archs ({})",
                        arch_key(&gpu_arch),
                        arch_key(variants)
                    ),
                });
            }
        }

        for (kernel_file, arch, gpu_arch) in units {
            let kernel_file = &kernel_file;
            let obj_file = self.object_file_path(kernel_file, arch.as_deref());
            all_obj_files.push(Compiled {
                source: kernel_file.clone(),
                arch,
                object: obj_file.clone(),
            });

            if self.incremental
                && !cache.needs_rebuild(
                    kernel_file,
                    &obj_file,
                    &arch_key(&gpu_arch),
                    &args_hash,
                    &watch_hash,
                )
            {
                continue;
            }

            compile_jobs.push((kernel_file.clone(), obj_file, gpu_arch));
        }

        if per_arch.is_some() {
            remove_stale_objects(&self.out_dir, &all_obj_files)?;
        }
        if compile_jobs.is_empty() && out_files.iter().all(|f| f.exists()) {
            println!("cargo:warning=All library kernels up-to-date, skipping compilation");
            return Ok(None);
        }
        // the cache is saved before linking, so a failed link must not leave the old library looking current
        for out_file in out_files {
            match std::fs::remove_file(out_file) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
                _ => {}
            }
        }

        println!(
            "cargo:warning=Compiling {} of {} kernels",
            compile_jobs.len(),
            kernel_files.len()
        );

        // Get target info
        let target = std::env::var("TARGET").ok();
        let is_msvc = target.as_ref().is_some_and(|t| t.contains("msvc"));
        let ccbin_env = std::env::var("NVCC_CCBIN").ok();
        let nvcc_threads = self.parallel.nvcc_threads();

        // Compile in parallel
        let had_error = AtomicBool::new(false);

        compile_jobs.par_iter().try_for_each(
            |(kernel_file, obj_file, gpu_arch)| -> Result<()> {
                if had_error.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let _slot = crate::jobserver::acquire();
                if had_error.load(Ordering::Relaxed) {
                    return Ok(());
                }

                let gencode = gencode_args(gpu_arch);

                let mut command = Command::new(&toolkit.nvcc_path);
                command
                    .args(&gencode)
                    .arg("-c")
                    .arg("-o")
                    .arg(obj_file)
                    .args(["--default-stream", "per-thread"]);

                // Add extra args
                for arg in &self.extra_args {
                    command.arg(arg);
                }

                // Add dependency includes
                for arg in &dep_args {
                    command.arg(arg);
                }

                // Add CUTLASS define if using cutlass
                if self.dependencies.has_cutlass() {
                    command.arg("-DUSE_CUTLASS");
                }

                // Add ccbin if set
                if let Some(ccbin) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin]);
                }

                // Add PIC flag for non-MSVC
                if !is_msvc {
                    command.arg("-Xcompiler").arg("-fPIC");
                } else {
                    command.arg("-D_USE_MATH_DEFINES");
                }

                // Add nvcc threads for certain files
                if let Some(threads) = nvcc_threads {
                    let filename = kernel_file.to_string_lossy();
                    if self.parallel.should_use_nvcc_threads(&filename) {
                        command.arg(format!("--threads={}", threads));
                    }
                }

                command.arg(kernel_file);

                let output = command
                    .spawn()
                    .map_err(|e| Error::NvccNotFound(format!("Failed to spawn nvcc: {}", e)))?
                    .wait_with_output()
                    .map_err(|e| Error::CompilationFailed {
                        path: kernel_file.clone(),
                        message: e.to_string(),
                    })?;

                if !output.status.success() {
                    had_error.store(true, Ordering::Relaxed);
                    return Err(Error::CompilationFailed {
                        path: kernel_file.clone(),
                        message: format!(
                            "nvcc error:\n{}\n{}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        ),
                    });
                }

                Ok(())
            },
        )?;

        // Update cache for compiled files
        if self.incremental {
            for (kernel_file, obj_file, gpu_arch) in &compile_jobs {
                cache.update(
                    kernel_file,
                    obj_file,
                    &arch_key(gpu_arch),
                    &args_hash,
                    &watch_hash,
                )?;
            }
            cache.save(&self.out_dir)?;
        }

        Ok(Some((toolkit, all_obj_files)))
    }

    /// Build PTX files from all kernel sources
    pub fn build_ptx(&self) -> Result<PtxOutput> {
        self.build_images(ImageKind::Ptx)
    }

    /// Build one fatbin (SASS for the target arch) per source, to load without a JIT; overrides match the source name.
    pub fn build_fatbin(&self) -> Result<PtxOutput> {
        self.build_images(ImageKind::Fatbin)
    }

    fn build_images(&self, kind: ImageKind) -> Result<PtxOutput> {
        crate::jobserver::init();
        let toolkit = match &self.toolkit {
            Some(t) => t.clone(),
            None => CudaToolkit::detect()?,
        };

        let _ = self.parallel.init_thread_pool();
        std::fs::create_dir_all(&self.out_dir)?;

        let kernel_files = self.sources.resolve()?;

        // Set CUDA include dir
        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            toolkit.include_dir.display()
        );

        // Sources generated into out_dir (PTX feeding a fatbin build) are rewritten after cargo stamps the run
        for file in kernel_files
            .iter()
            .filter(|f| !f.starts_with(&self.out_dir))
        {
            println!("cargo:rerun-if-changed={}", file.display());
        }
        for path in self.sources.watch_paths() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

        let dep_args = self.dependencies.fetch_all(&self.out_dir)?;
        let ccbin_env = std::env::var("NVCC_CCBIN").ok();
        let nvcc_threads = self.parallel.nvcc_threads();
        let watch_hash = hash_paths(self.sources.watch_paths());
        let mut cache = BuildCache::load(&self.out_dir);

        let mut all_args = self.extra_args.clone();
        all_args.extend(dep_args.clone());
        let args_hash = hash_args(&all_args);

        let mut compile_jobs = Vec::new();
        for kernel_file in &kernel_files {
            let filename = kernel_file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let mut gpu_arch = self.compute_cap.get_for_file(filename)?;
            // nvcc emits PTX for one arch only; the lowest is the one every listed device can JIT
            if matches!(kind, ImageKind::Ptx) {
                gpu_arch.truncate(1);
            }

            let output_file = self.out_dir.join(
                kernel_file
                    .with_extension(kind.extension())
                    .file_name()
                    .unwrap(),
            );

            // Check if output is current using BuildCache
            if self.incremental
                && !cache.needs_rebuild(
                    kernel_file,
                    &output_file,
                    &arch_key(&gpu_arch),
                    &args_hash,
                    &watch_hash,
                )
            {
                continue;
            }

            compile_jobs.push((kernel_file, output_file, gpu_arch));
        }

        if compile_jobs.is_empty() {
            println!("cargo:warning=All {kind:?} kernels up-to-date, skipping compilation");
            return Ok(PtxOutput {
                paths: kernel_files,
                out_dir: self.out_dir.clone(),
                kind,
            });
        }

        println!(
            "cargo:warning=Compiling {} of {} {kind:?} kernels",
            compile_jobs.len(),
            kernel_files.len()
        );

        // Compile in parallel
        compile_jobs.par_iter().try_for_each(
            |(kernel_file, output_file, gpu_arch)| -> Result<()> {
                let _slot = crate::jobserver::acquire();
                let gencode = gencode_args(gpu_arch);

                let mut command = Command::new(&toolkit.nvcc_path);
                command
                    .args(&gencode)
                    .arg(kind.nvcc_flag())
                    .args(["--default-stream", "per-thread"])
                    .arg("-o")
                    .arg(output_file);

                for arg in &self.extra_args {
                    command.arg(arg);
                }
                for arg in &dep_args {
                    command.arg(arg);
                }
                if let Some(ccbin) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin]);
                }

                // Add nvcc threads for certain files
                if let Some(threads) = nvcc_threads {
                    let file_path = kernel_file.to_string_lossy();
                    if self.parallel.should_use_nvcc_threads(&file_path) {
                        command.arg(format!("--threads={}", threads));
                    }
                }

                command.arg(kernel_file);

                let output = command
                    .spawn()
                    .map_err(|e| Error::NvccNotFound(format!("Failed to spawn nvcc: {}", e)))?
                    .wait_with_output()
                    .map_err(|e| Error::CompilationFailed {
                        path: kernel_file.to_path_buf(),
                        message: e.to_string(),
                    })?;

                if !output.status.success() {
                    return Err(Error::CompilationFailed {
                        path: kernel_file.to_path_buf(),
                        message: format!(
                            "nvcc error:\n{}\n{}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        ),
                    });
                }

                Ok(())
            },
        )?;

        // Update cache for the built images
        if self.incremental {
            for kernel_file in &kernel_files {
                let filename = kernel_file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                let gpu_arch = self.compute_cap.get_for_file(filename)?;
                let output_file = self.out_dir.join(
                    kernel_file
                        .with_extension(kind.extension())
                        .file_name()
                        .unwrap(),
                );

                cache.update(
                    kernel_file,
                    &output_file,
                    &arch_key(&gpu_arch),
                    &args_hash,
                    &watch_hash,
                )?;
            }
            cache.save(&self.out_dir)?;
        }

        Ok(PtxOutput {
            paths: kernel_files,
            out_dir: self.out_dir.clone(),
            kind,
        })
    }

    /// Generate unique object file path for a kernel
    fn object_file_path(&self, kernel_file: &Path, arch: Option<&str>) -> PathBuf {
        let mut hasher = DefaultHasher::new();
        kernel_file.display().to_string().hash(&mut hasher);
        let hash = hasher.finish();

        let stem = kernel_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("kernel");

        match arch {
            Some(arch) => self.out_dir.join(format!("{stem}-{hash:x}-{arch}.o")),
            None => self.out_dir.join(format!("{}-{:x}.o", stem, hash)),
        }
    }
}

fn parse_major_minor(version: &str) -> Option<(u32, u32)> {
    let (major, minor) = version.split_once('.')?;
    Some((major.parse().ok()?, minor.split('.').next()?.parse().ok()?))
}

fn run_link(mut command: Command) -> Result<()> {
    let output = command
        .spawn()
        .map_err(|e| Error::NvccNotFound(format!("Failed to spawn nvcc for linking: {}", e)))?
        .wait_with_output()
        .map_err(|e| Error::LinkingFailed(e.to_string()))?;
    if !output.status.success() {
        return Err(Error::LinkingFailed(format!(
            "nvcc linking error:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(())
}

/// The environment variable a crate reads for the built path of loadable kernel library `name`.
pub fn loadable_lib_env(name: &str) -> String {
    format!("INFERENCE_KERNEL_LIB_{}", name.to_uppercase())
}

/// The environment variable a crate reads for the build id loadable kernel library `name` must report.
pub fn loadable_id_env(name: &str) -> String {
    format!("INFERENCE_KERNEL_ID_{}", name.to_uppercase())
}

/// The symbol a kernel library exports its build id under; `inference_tensor::kernel_library::BUILD_ID_SYMBOL` matches.
pub const BUILD_ID_SYMBOL: &str = "inference_kernel_build_id";
const BUILD_ID_EXTENSION: &str = "id";
const BUILD_ID_OBJECT_SUFFIX: &str = "_build_id.o";

// objects of an older layout or a removed source, which nothing links any more
fn remove_stale_objects(dir: &Path, objects: &[Compiled]) -> Result<()> {
    let current: std::collections::HashSet<&Path> =
        objects.iter().map(|c| c.object.as_path()).collect();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let id_object = path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().ends_with(BUILD_ID_OBJECT_SUFFIX));
        if path.extension().is_some_and(|e| e == "o")
            && !id_object
            && !current.contains(path.as_path())
        {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}

// a hash of the objects' contents: what changes when any kernel does
fn build_id(objects: &[PathBuf]) -> Result<String> {
    let mut key = DefaultHasher::new();
    let mut sorted = objects.to_vec();
    sorted.sort();
    for object in &sorted {
        std::fs::read(object)?.hash(&mut key);
    }
    Ok(format!("{:016x}", key.finish()))
}

// a one-line C object carrying `id`, so the loader can tell this build's library from another's
fn compile_build_id(toolkit: &CudaToolkit, dir: &Path, lib: &str, id: &str) -> Result<PathBuf> {
    let source = dir.join(format!("{lib}_build_id.c"));
    let object = dir.join(format!("{lib}{BUILD_ID_OBJECT_SUFFIX}"));
    std::fs::write(
        &source,
        format!(
            "__attribute__((visibility(\"default\"))) const char {BUILD_ID_SYMBOL}[] = \"{id}\";\n"
        ),
    )?;
    let mut command = Command::new(&toolkit.nvcc_path);
    command
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .args(["-Xcompiler", "-fPIC"]);
    run_link(command)?;
    Ok(object)
}

// dev builds rebuild kernels per cargo variant, so their libraries stay in the target dir with them
fn dev_shared_libs() -> bool {
    std::env::var("PROFILE").is_ok_and(|p| p == "debug")
        && std::env::var("TARGET").is_ok_and(|t| t.contains("linux"))
}

/// The extension of the file naming the library a build made, `cuda-kernels/<name>.lib`.
pub const LIB_POINTER_EXTENSION: &str = "lib";
/// Overrides where release-type builds keep their kernel libraries.
pub const LIB_CACHE_ENV: &str = "INFERENCE_KERNEL_BUILD_CACHE";

/// Where release-type builds keep kernel libraries: `$INFERENCE_KERNEL_BUILD_CACHE`, else the user cache.
fn persistent_lib_root() -> Result<PathBuf> {
    for var in [LIB_CACHE_ENV, "XDG_CACHE_HOME", "HOME"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if let Some(dir) = std::env::var_os(LIB_CACHE_ENV) {
        return Ok(PathBuf::from(dir));
    }
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .ok_or_else(|| {
            Error::LinkingFailed(format!(
                "no HOME or {LIB_CACHE_ENV} for the kernel library cache"
            ))
        })?;
    Ok(cache.join("inference-rs").join("kernels").join("built"))
}

/// `<target>/<profile>/cuda-kernels`, derived from `OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out`.
fn shared_lib_root() -> Result<PathBuf> {
    let out_dir =
        std::env::var("OUT_DIR").map_err(|_| Error::LinkingFailed("OUT_DIR not set".into()))?;
    let out_dir = PathBuf::from(out_dir);
    let profile_dir = out_dir.ancestors().nth(3).ok_or_else(|| {
        Error::LinkingFailed(format!("unexpected OUT_DIR layout: {}", out_dir.display()))
    })?;
    Ok(profile_dir.join("cuda-kernels"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageKind {
    Ptx,
    Fatbin,
}

impl ImageKind {
    fn extension(self) -> &'static str {
        match self {
            ImageKind::Ptx => "ptx",
            ImageKind::Fatbin => "fatbin",
        }
    }

    fn nvcc_flag(self) -> &'static str {
        match self {
            ImageKind::Ptx => "--ptx",
            ImageKind::Fatbin => "--fatbin",
        }
    }
}

/// Output from PTX or fatbin compilation
pub struct PtxOutput {
    paths: Vec<PathBuf>,
    out_dir: PathBuf,
    kind: ImageKind,
}

impl PtxOutput {
    /// Path of the built image for each kernel source
    pub fn images(&self) -> impl Iterator<Item = PathBuf> + '_ {
        self.paths.iter().map(|p| {
            self.out_dir
                .join(p.with_extension(self.kind.extension()).file_name().unwrap())
        })
    }

    /// Write a Rust source file with `const` declarations for each PTX (`&str`) or fatbin (`&[u8]`) file
    pub fn write<P: AsRef<Path>>(&self, out: P) -> Result<()> {
        let mut file = std::fs::File::create(out.as_ref())?;

        for kernel_path in &self.paths {
            let name = kernel_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("KERNEL");

            let ident = name.to_uppercase().replace(['.', '-'], "_");
            match self.kind {
                ImageKind::Ptx => writeln!(
                    file,
                    r#"pub const {ident}: &str = include_str!(concat!(env!("OUT_DIR"), "/{name}.ptx"));"#
                )?,
                ImageKind::Fatbin => writeln!(
                    file,
                    r#"pub const {ident}: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/{name}.fatbin"));"#
                )?,
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::Duration;

    #[test]
    fn a_split_takes_only_its_own_type_and_the_rest_stays_in_the_base() {
        let builder = KernelBuilder::new()
            .split_lib("kernels_q4_k", "*_instance_q4_k.cu")
            .split_lib("kernels_iq4_k", "*_instance_iq4_k.cu");
        let lib = |file: &str| builder.library_of("kernels", Path::new(file)).to_string();
        assert_eq!(lib("dir/mmq_instance_q4_k.cu"), "kernels_q4_k");
        assert_eq!(lib("dir/mmvq_instance_q4_k.cu"), "kernels_q4_k");
        assert_eq!(lib("dir/mmvq_instance_iq4_k.cu"), "kernels_iq4_k");
        assert_eq!(lib("dir/mmvq_instance_iq4_ks.cu"), "kernels");
        assert_eq!(lib("dir/mmvq_quantize.cu"), "kernels");
    }

    #[test]
    fn a_build_waits_out_another_build_and_reports_a_running_binary() {
        let dir = std::env::temp_dir().join(format!("kernel-build-lock-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(BUILD_DIR_LOCK);
        let open = || fs::File::create(&path).unwrap();
        // flock locks belong to the open file, so a second handle in this process contends like another process
        let build = open();
        assert!(lock_build_dir(&build, &dir).unwrap().is_none());
        build.unlock().unwrap();

        let binary = open();
        binary.lock_shared().unwrap();
        let error = lock_build_dir(&open(), &dir).unwrap().unwrap().to_string();
        assert!(error.contains("in use"), "{error}");
        #[cfg(target_os = "linux")]
        assert!(
            error.contains(&format!("pid {}", std::process::id())),
            "{error}"
        );
        drop(binary);

        let other = open();
        other.lock().unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            drop(other);
        });
        assert!(lock_build_dir(&open(), &dir).unwrap().is_none());
        release.join().unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn size_compression_starts_at_cuda_12_8() {
        let mode = |v| parse_major_minor(v) >= Some(SIZE_COMPRESSION_SINCE);
        assert!(!mode("12.6"));
        assert!(mode("12.8"));
        assert!(mode("13.0"));
        assert_eq!(parse_major_minor("12.10.1"), Some((12, 10)));
        assert_eq!(parse_major_minor("dev"), None);
    }

    #[test]
    fn test_incremental_rebuild_on_header_change() {
        // Needs nvcc and an arch to build for; a toolkit-only host without a GPU or CUDA_COMPUTE_CAP has no arch
        if crate::toolkit::CudaToolkit::detect().is_err()
            || crate::compute_cap::detect_compute_caps().is_err()
        {
            return;
        }

        // Use a semi-unique directory in the system temp folder
        let mut root = std::env::temp_dir();
        root.push(format!("cudaforge-test-{}", std::process::id()));

        // Ensure clean state
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        fs::create_dir_all(&root).unwrap();

        let out_dir = root.join("out");
        let src_dir = root.join("src");
        fs::create_dir_all(&out_dir).unwrap();
        fs::create_dir_all(&src_dir).unwrap();

        let cu_file = src_dir.join("kernel.cu");
        let cuh_file = src_dir.join("header.cuh");
        let lib_file = out_dir.join("libkernels.a");

        // Create initial version
        fs::write(
            &cu_file,
            "#include \"header.cuh\"\nextern \"C\" __global__ void add() {}",
        )
        .unwrap();
        fs::write(&cuh_file, "// version 1").unwrap();

        let builder = KernelBuilder::new()
            .source_files(vec![&cu_file])
            .watch(vec![&cuh_file])
            .out_dir(&out_dir);

        // First build
        builder.build_lib(&lib_file).expect("First build failed");
        let mtime1 = fs::metadata(&lib_file)
            .expect("Lib file should exist")
            .modified()
            .unwrap();

        std::thread::sleep(Duration::from_millis(1100));

        // Modify the watched header
        fs::write(&cuh_file, "// version 2").unwrap();

        // Second build - should trigger recompilation
        builder.build_lib(&lib_file).expect("Second build failed");
        let mtime2 = fs::metadata(&lib_file)
            .expect("Lib file should exist after second build")
            .modified()
            .unwrap();

        let recompiled = mtime2 > mtime1;

        // Cleanup BEFORE assertion so we don't leave mess on failure
        let _ = fs::remove_dir_all(&root);

        assert!(
            recompiled,
            "Recompilation was NOT triggered by header change!"
        );
    }
}
