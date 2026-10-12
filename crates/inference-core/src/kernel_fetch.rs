//! CUDA kernel libraries this build has no local copy of, downloaded from the release that built them.

use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime},
};

use anyhow::{Context, Result, bail};
use inference_tensor::kernel_library::{
    ARCH_DIR_PREFIX, BUILD_DIR_LOCK, Fetch, build_cache_root, download_dir, download_root,
    process_arch_dir, set_fetcher,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Overrides where kernel assets and their manifest download from: an https mirror, or a local http server for testing.
pub const KERNEL_URL_ENV: &str = "INFERENCE_KERNEL_URL";
/// `0` keeps kernel libraries from downloading; a missing one is then an error.
pub const KERNEL_DOWNLOAD_ENV: &str = "INFERENCE_KERNEL_DOWNLOAD";
/// How long a build dir no binary has opened a library from is kept.
pub const BUILD_DIR_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 60 * 60);
const RELEASE_URL: &str = concat!(
    env!("CARGO_PKG_REPOSITORY"),
    "/releases/download/v",
    env!("CARGO_PKG_VERSION")
);
const VERSION: &str = env!("CARGO_PKG_VERSION");
// package_kernels.py's MANIFEST_SCHEMA
const MANIFEST_SCHEMA: u64 = 1;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
// how long a download may go without a byte before it fails
const READ_STALL_TIMEOUT: Duration = Duration::from_secs(120);
// a partial download this old is from a process that died mid-download
const PARTIAL_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
const PARTIAL_SUFFIX: &str = ".part";
const BUILD_DIR_HASH_LEN: usize = 16;
const VERSION_PARTS: usize = 3;
const MIB: u64 = 1 << 20;
const COPY_BUFFER: usize = 1 << 20;
// plain http is only for a server on this machine
const LOOPBACK_URLS: [&str; 3] = ["http://127.0.0.1", "http://localhost", "http://[::1]"];

#[derive(Deserialize)]
struct Manifest {
    schema: u64,
    version: String,
    libraries: HashMap<String, LibraryEntry>,
    files: Vec<FileEntry>,
}

#[derive(Deserialize)]
struct LibraryEntry {
    build_id: String,
}

#[derive(Deserialize)]
struct FileEntry {
    library: String,
    arch: String,
    asset: String,
    sha256: String,
    size: u64,
}

/// Lets a missing kernel library download from this version's release.
pub(crate) fn install() {
    set_fetcher(fetch);
}

fn base_url() -> String {
    std::env::var(KERNEL_URL_ENV).unwrap_or_else(|_| RELEASE_URL.to_string())
}

// package_kernels.py names it by the CUDA runtime major the libraries link: kernels-cu12.json
fn manifest_name() -> Result<String> {
    let version = inference_nn::BUILD_CUDA_VERSION.context("this build records no CUDA version")?;
    let major = version.split('.').next().unwrap_or_default();
    Ok(format!("kernels-cu{major}.json"))
}

// a blocking client on its own thread, since model loading can run inside an async runtime
fn on_thread<T: Send>(work: impl FnOnce() -> Result<T> + Send) -> Result<T> {
    std::thread::scope(|scope| {
        scope
            .spawn(work)
            .join()
            .unwrap_or_else(|_| bail!("the download thread panicked"))
    })
}

fn get(url: &str) -> Result<reqwest::blocking::Response> {
    let loopback = LOOPBACK_URLS.iter().any(|prefix| url.starts_with(prefix));
    let client = crate::remote_fetch::blocking_http_client_builder()
        .https_only(!loopback)
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(READ_STALL_TIMEOUT)
        .build()?;
    let response = client
        .get(url)
        .send()
        .with_context(|| format!("fetching {url}"))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        bail!("{url} does not exist: no v{VERSION} release publishes this kernel file");
    }
    Ok(response.error_for_status()?)
}

// only a manifest that loaded is kept, so a failed fetch is retried by the next library that needs one
fn manifest() -> Result<Arc<Manifest>> {
    static MANIFEST: Mutex<Option<Arc<Manifest>>> = Mutex::new(None);
    let mut cached = MANIFEST.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(manifest) = cached.as_ref() {
        return Ok(manifest.clone());
    }
    let url = format!("{}/{}", base_url(), manifest_name()?);
    let manifest: Manifest = on_thread(|| Ok(get(&url)?.json()?))
        .with_context(|| format!("reading the kernel manifest {url}"))?;
    if manifest.schema != MANIFEST_SCHEMA || manifest.version != VERSION {
        bail!(
            "{url} is schema {} for version {}, this binary reads schema {MANIFEST_SCHEMA} for {VERSION}",
            manifest.schema,
            manifest.version
        );
    }
    let manifest = Arc::new(manifest);
    *cached = Some(manifest.clone());
    Ok(manifest)
}

fn fetch(request: &Fetch<'_>) -> std::result::Result<(), String> {
    fetch_inner(request).map_err(|error| format!("{error:#}"))
}

fn fetch_inner(request: &Fetch<'_>) -> Result<()> {
    if std::env::var(KERNEL_DOWNLOAD_ENV).is_ok_and(|value| value == "0") {
        bail!("{KERNEL_DOWNLOAD_ENV}=0 turns downloading off");
    }
    let manifest = manifest()?;
    let name = request.library;
    let Some(library) = manifest.libraries.get(name) else {
        bail!("the v{VERSION} release has no kernel library `{name}`");
    };
    // a binary built from source has kernels of its own build, which no release carries
    if library.build_id != request.build_id {
        bail!(
            "the v{VERSION} release's `lib{name}.so` is from another build (id {}, this binary expects {}). A binary \
             built from source loads its kernels from where its build put them; if they were pruned or deleted, \
             rebuild or reinstall it",
            library.build_id,
            request.build_id
        );
    }
    let arch = request.arch_dir.trim_start_matches(ARCH_DIR_PREFIX);
    let Some(file) = manifest
        .files
        .iter()
        .find(|file| file.library == name && file.arch == arch)
    else {
        let mut archs: Vec<&str> = manifest
            .files
            .iter()
            .filter(|file| file.library == name)
            .map(|file| file.arch.as_str())
            .collect();
        archs.sort_unstable();
        archs.dedup();
        bail!(
            "the v{VERSION} release has no `lib{name}.so` for {} (it has {ARCH_DIR_PREFIX}{})",
            request.arch_dir,
            archs.join(&format!(", {ARCH_DIR_PREFIX}"))
        );
    };
    let target = request
        .dir
        .join(request.arch_dir)
        .join(format!("lib{name}.so"));
    download(&format!("{}/{}", base_url(), file.asset), file, &target)
}

// to a file of its own beside the target, renamed into place once its size and hash match the manifest
fn download(url: &str, file: &FileEntry, target: &Path) -> Result<()> {
    static PARTIALS: AtomicUsize = AtomicUsize::new(0);
    let dir = target.parent().context("a kernel library path has a dir")?;
    let target_name = target
        .file_name()
        .context("a kernel library path has a file name")?
        .to_string_lossy();
    std::fs::create_dir_all(dir)?;
    let partial = dir.join(format!(
        ".{target_name}.{}.{}{PARTIAL_SUFFIX}",
        std::process::id(),
        PARTIALS.fetch_add(1, Ordering::Relaxed)
    ));
    tracing::info!(
        "Downloading CUDA kernel library {} ({} MiB) from {url}",
        file.asset,
        file.size.div_ceil(MIB)
    );
    let written = on_thread(|| {
        let mut response = get(url)?;
        let mut out = std::fs::File::create(&partial)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; COPY_BUFFER];
        let mut size = 0u64;
        loop {
            let read = response.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            size += read as u64;
            if size > file.size {
                bail!(
                    "{url} is larger than the {} bytes the manifest gives",
                    file.size
                );
            }
            hasher.update(&buffer[..read]);
            out.write_all(&buffer[..read])?;
        }
        out.sync_all()?;
        Ok((size, hex(&hasher.finalize())))
    })
    .and_then(|(size, digest)| {
        if size != file.size {
            bail!("{url} is {size} bytes, the manifest says {}", file.size);
        }
        if digest != file.sha256 {
            bail!(
                "{url} has SHA-256 {digest}, the manifest says {}",
                file.sha256
            );
        }
        Ok(())
    });
    if let Err(error) = written {
        let _ = std::fs::remove_file(&partial);
        return Err(error);
    }
    std::fs::rename(&partial, target)?;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The kernel libraries `inference kernels fetch` installs.
pub enum KernelSet {
    /// The libraries every CUDA model loads (attention, ops, the shared quantization kernels).
    Base,
    /// The base set plus these GGUF types' libraries (`q4_k`).
    Types(Vec<String>),
    /// The base set plus every GGUF type's library.
    All,
}

/// The arch dir the fetched libraries are for, and where downloads went.
pub struct FetchReport {
    pub arch_dir: String,
    pub download_dir: PathBuf,
}

/// Downloads `set` for this machine's GPUs wherever this binary has no local copy, checking each like a model load.
pub fn fetch_kernels(set: KernelSet) -> Result<FetchReport> {
    let known = kernel_types();
    let types: Vec<String> = match set {
        KernelSet::Base => Vec::new(),
        KernelSet::Types(types) => types.iter().map(|ty| ty.to_ascii_lowercase()).collect(),
        KernelSet::All => known.iter().map(|ty| ty.to_string()).collect(),
    };
    if let Some(unknown) = types.iter().find(|ty| !known.contains(&ty.as_str())) {
        bail!(
            "no GGUF type `{unknown}` has a kernel library (known: {})",
            known.join(", ")
        );
    }
    // the arch to fetch for is the one this machine's GPUs need
    if inference_tensor::cuda_backend::device_count().unwrap_or(0) == 0 {
        bail!(
            "no CUDA GPU is visible, so there is no arch to fetch kernels for; run this on the machine that uses them"
        );
    }
    install();
    let arch_dir = process_arch_dir().map_err(anyhow::Error::msg)?;
    let download_dir =
        download_dir().context("no HOME or INFERENCE_KERNEL_CACHE for the kernel cache")?;
    inference_nn::cuda::ensure_kernels_loaded().map_err(anyhow::Error::msg)?;
    for ty in &types {
        inference_quant::ensure_gguf_type_kernels(ty).map_err(anyhow::Error::msg)?;
    }
    Ok(FetchReport {
        arch_dir: arch_dir.to_string(),
        download_dir,
    })
}

/// The GGUF types with a kernel library of their own (`q4_k`), as `inference kernels fetch --types` takes them.
pub fn kernel_types() -> Vec<&'static str> {
    inference_quant::gguf_kernel_types().collect()
}

/// A kernel cache directory or file `inference kernels prune` removes, with its size.
pub struct Pruned {
    pub path: PathBuf,
    pub bytes: u64,
}

/// Removes other versions' downloads, dead partial downloads and unused build dirs; `dry_run` only lists them.
pub fn prune_kernels(dry_run: bool) -> Result<Vec<Pruned>> {
    let mut stale = Vec::new();
    if let Some(root) = download_root() {
        for dir in subdirs(&root)? {
            if is_version_dir(&dir) {
                if dir.file_name().is_some_and(|name| name != VERSION) {
                    stale.push(dir);
                } else {
                    stale.extend(dead_partials(&dir)?);
                }
            }
        }
    }
    if let Some(root) = build_cache_root() {
        let now = SystemTime::now();
        for dir in subdirs(&root)? {
            if is_build_dir(&dir) && unused_since(&dir, now) {
                stale.push(dir);
            }
        }
    }
    let mut pruned = Vec::new();
    for path in stale {
        let bytes = disk_size(&path);
        if !dry_run {
            let removed = if path.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
            removed.with_context(|| format!("removing {}", path.display()))?;
        }
        pruned.push(Pruned { path, bytes });
    }
    Ok(pruned)
}

// `0.9.3/` holding only `sm<arch>/` dirs: what downloads make, never another dir the cache root happens to hold
fn is_version_dir(dir: &Path) -> bool {
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let parts: Vec<&str> = name.split('.').collect();
    let version = parts.len() == VERSION_PARTS
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()));
    version
        && std::fs::read_dir(dir).is_ok_and(|entries| {
            entries.filter_map(|entry| entry.ok()).all(|entry| {
                entry.path().is_dir()
                    && entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with(ARCH_DIR_PREFIX)
            })
        })
}

// `.lib<name>.so.<pid>.<n>.part` files in `version/sm<arch>/` older than a day
fn dead_partials(version: &Path) -> Result<Vec<PathBuf>> {
    let now = SystemTime::now();
    let mut dead = Vec::new();
    for arch in subdirs(version)? {
        for entry in std::fs::read_dir(&arch)? {
            let path = entry?.path();
            let partial = path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with(PARTIAL_SUFFIX));
            let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
            if partial
                && modified
                    .is_some_and(|m| now.duration_since(m).unwrap_or_default() > PARTIAL_MAX_AGE)
            {
                dead.push(path);
            }
        }
    }
    Ok(dead)
}

// `<library>-<16 hex>/`, the dirs inference-kernel-build makes
fn is_build_dir(dir: &Path) -> bool {
    dir.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.rsplit_once('-'))
        .is_some_and(|(_, hash)| {
            hash.len() == BUILD_DIR_HASH_LEN && hash.chars().all(|c| c.is_ascii_hexdigit())
        })
}

// opened by no binary for BUILD_DIR_MAX_AGE, and locked by no running process or build
fn unused_since(dir: &Path, now: SystemTime) -> bool {
    let Ok(lock) = std::fs::File::open(dir.join(BUILD_DIR_LOCK)) else {
        return false;
    };
    let old = lock
        .metadata()
        .and_then(|m| m.modified())
        .is_ok_and(|used| now.duration_since(used).unwrap_or_default() > BUILD_DIR_MAX_AGE);
    old && lock.try_lock().is_ok()
}

fn subdirs(root: &Path) -> Result<Vec<PathBuf>> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Ok(Vec::new());
    };
    let mut dirs = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    Ok(dirs)
}

fn disk_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    };
    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| disk_size(&entry.path()))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_version_dirs_of_arch_dirs_count_as_downloads() {
        let root = tempfile::tempdir().unwrap();
        let make = |name: &str, inner: &str| {
            let dir = root.path().join(name);
            std::fs::create_dir_all(dir.join(inner)).unwrap();
            dir
        };
        assert!(is_version_dir(&make("0.9.2", "sm86")));
        assert!(!is_version_dir(&make("2024", "sm86")));
        assert!(!is_version_dir(&make("1.2", "sm86")));
        assert!(!is_version_dir(&make("0.9.1", "photos")));
        let mixed = make("0.9.0", "sm90a");
        std::fs::write(mixed.join("notes.txt"), "kept").unwrap();
        assert!(!is_version_dir(&mixed));
    }

    #[test]
    fn a_locked_build_dir_is_in_use_however_old() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("inferencequant-0123456789abcdef");
        std::fs::create_dir_all(&dir).unwrap();
        let lock = std::fs::File::create(dir.join(BUILD_DIR_LOCK)).unwrap();
        let later = SystemTime::now() + BUILD_DIR_MAX_AGE + Duration::from_secs(60);
        assert!(is_build_dir(&dir));
        assert!(unused_since(&dir, later));
        lock.lock_shared().unwrap();
        assert!(!unused_since(&dir, later));
        assert!(!is_build_dir(&root.path().join("inferencequant-notahash")));
    }
}
