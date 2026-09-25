//! C ABI for inference.rs. The contract lives in `include/inference.h`; every `#[no_mangle]` item here mirrors it.
// pointer-validity requirements of the exported functions are the C contract, documented once in inference.h
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use candle_core::Device;

pub mod layout;

pub const ABI_VERSION_MAJOR: u32 = 0;
pub const ABI_VERSION_MINOR: u32 = 1;
pub const ABI_VERSION_PATCH: u32 = 0;

const BACKEND_TAG: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "metal") {
    "metal"
} else {
    "cpu"
};

static BUILD_VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();

/// Mirrors `inference_status` in `inference.h`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum inference_status {
    INFERENCE_OK = 0,
    INFERENCE_ERR_INVALID_ARGUMENT = 1,
    INFERENCE_ERR_LOAD_FAILED = 2,
    INFERENCE_ERR_RUNTIME = 3,
    INFERENCE_ERR_OUT_OF_RANGE = 4,
    INFERENCE_ERR_NOT_AVAILABLE = 5,
    INFERENCE_ERR_INTERNAL = 6,
}

use inference_status::*;

/// A failed call: the status to return and the detail for `inference_last_error`.
pub(crate) struct Failure {
    status: inference_status,
    message: String,
}

impl Failure {
    pub(crate) fn new(status: inference_status, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new(INFERENCE_ERR_INVALID_ARGUMENT, message)
    }
}

pub(crate) type FfiResult<T = ()> = std::result::Result<T, Failure>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(message: &str) {
    // interior NULs would truncate the C string anyway; drop them rather than fail
    let clean = CString::new(message.replace('\0', "")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = clean);
}

/// Runs an entry point: clears the thread's last error, maps failures to status codes and stops panics at the boundary.
pub(crate) fn guard(f: impl FnOnce() -> FfiResult) -> inference_status {
    set_last_error("");
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => INFERENCE_OK,
        Ok(Err(failure)) => {
            set_last_error(&failure.message);
            failure.status
        }
        Err(panic) => {
            let detail = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            set_last_error(&format!("internal error (panic): {detail}"));
            INFERENCE_ERR_INTERNAL
        }
    }
}

/// Non-failing entry points (counts, frees) still must not unwind into C.
pub(crate) fn guard_value<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(fallback)
}

/// `const char*` argument to `&str`. Safety: `ptr` is NULL or a NUL-terminated string valid for the call.
pub(crate) unsafe fn arg_str<'a>(ptr: *const c_char, name: &str) -> FfiResult<&'a str> {
    if ptr.is_null() {
        return Err(Failure::invalid(format!("{name} is NULL")));
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|_| Failure::invalid(format!("{name} is not valid UTF-8")))
}

/// Writes `value` through an optional out-pointer. Safety: `out` is NULL or valid for a write of `T`.
pub(crate) unsafe fn write_opt<T>(out: *mut T, value: T) {
    if !out.is_null() {
        out.write(value);
    }
}

/// Mirrors `inference_backend_config`.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct inference_backend_config {
    pub backend: *const c_char,
    pub device: i32,
    pub threads: i32,
}

pub(crate) struct Backend {
    pub device: Device,
    /// `None` = one thread per physical core.
    pub cpu_threads: Option<usize>,
}

/// Safety: `config` is NULL or a valid `inference_backend_config` whose `backend` is NULL or a C string.
pub(crate) unsafe fn backend_from(config: *const inference_backend_config) -> FfiResult<Backend> {
    let (name, ordinal, threads) = match config.as_ref() {
        None => ("cpu", 0, 0),
        Some(c) => {
            let name = if c.backend.is_null() {
                "cpu"
            } else {
                arg_str(c.backend, "backend")?
            };
            (name, c.device, c.threads)
        }
    };
    let ordinal = usize::try_from(ordinal)
        .map_err(|_| Failure::invalid(format!("device {ordinal} is negative")))?;
    let unavailable = |what: &str| {
        Failure::new(
            INFERENCE_ERR_NOT_AVAILABLE,
            format!("this build has no {what} support"),
        )
    };
    let device = match name {
        "cpu" => Device::Cpu,
        "cuda" if cfg!(feature = "cuda") => Device::new_cuda(ordinal).map_err(|e| {
            Failure::new(
                INFERENCE_ERR_NOT_AVAILABLE,
                format!("CUDA device {ordinal}: {e}"),
            )
        })?,
        "cuda" => return Err(unavailable("CUDA")),
        "metal" if cfg!(feature = "metal") => Device::new_metal(ordinal).map_err(|e| {
            Failure::new(
                INFERENCE_ERR_NOT_AVAILABLE,
                format!("Metal device {ordinal}: {e}"),
            )
        })?,
        "metal" => return Err(unavailable("Metal")),
        other => {
            return Err(Failure::invalid(format!(
                "unknown backend {other:?}; expected cpu, cuda or metal"
            )))
        }
    };
    Ok(Backend {
        device,
        cpu_threads: usize::try_from(threads).ok().filter(|&t| t > 0),
    })
}

#[no_mangle]
pub extern "C" fn inference_abi_version() -> u32 {
    (ABI_VERSION_MAJOR << 16) | (ABI_VERSION_MINOR << 8) | ABI_VERSION_PATCH
}

#[no_mangle]
pub extern "C" fn inference_build_version() -> *const c_char {
    BUILD_VERSION
        .get_or_init(|| {
            CString::new(format!(
                "inference.rs {} ({BACKEND_TAG})",
                env!("CARGO_PKG_VERSION")
            ))
            .unwrap_or_default()
        })
        .as_ptr()
}

#[no_mangle]
pub extern "C" fn inference_last_error() -> *const c_char {
    // the CString lives in the thread-local until the next call on this thread replaces it
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn inference_status_string(status: i32) -> *const c_char {
    // an i32, not the enum: C may pass any integer, and an out-of-range Rust enum value is undefined behaviour
    let name: &CStr = match status {
        0 => c"INFERENCE_OK",
        1 => c"INFERENCE_ERR_INVALID_ARGUMENT",
        2 => c"INFERENCE_ERR_LOAD_FAILED",
        3 => c"INFERENCE_ERR_RUNTIME",
        4 => c"INFERENCE_ERR_OUT_OF_RANGE",
        5 => c"INFERENCE_ERR_NOT_AVAILABLE",
        6 => c"INFERENCE_ERR_INTERNAL",
        _ => c"INFERENCE_UNKNOWN_STATUS",
    };
    name.as_ptr()
}
