//! Error types for inference-kernel-build

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for inference-kernel-build operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during CUDA kernel building
#[derive(Error)]
pub enum Error {
    /// nvcc binary not found
    #[error("nvcc not found: {0}. Set NVCC environment variable or ensure nvcc is in PATH")]
    NvccNotFound(String),

    /// CUDA toolkit not found
    #[error("CUDA toolkit not found at {0}")]
    CudaToolkitNotFound(PathBuf),

    /// Compute capability detection failed
    #[error("Failed to detect compute capability: {0}")]
    ComputeCapDetectionFailed(String),

    /// Kernel compilation failed
    #[error("Kernel compilation failed for {path}: {message}")]
    CompilationFailed {
        /// Path to the kernel file that failed
        path: PathBuf,
        /// Error message from nvcc
        message: String,
    },

    /// Linking failed
    #[error("Linking failed: {0}")]
    LinkingFailed(String),

    /// Invalid source path
    #[error("Source path does not exist: {0}")]
    SourcePathNotFound(PathBuf),

    /// Git operation failed
    #[error("Git operation failed: {0}")]
    GitOperationFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// A running binary loads its kernels from the build dir; rebuilding them would change what it opens next
    #[error(
        "the CUDA kernel libraries in {dir} are in use by {holders}; stop it and build again (a build would replace \
         libraries it may still open)"
    )]
    KernelLibrariesInUse {
        /// The build dir whose libraries are open.
        dir: PathBuf,
        /// The processes holding them, as far as the platform tells.
        holders: String,
    },
}

// build scripts `expect` these, which prints Debug: the message reads better than the variant
impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
