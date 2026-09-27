//! Engine-independent reports about the host: devices and build, and environment diagnostics.

use inference_core::{collect_system_info, run_doctor, DoctorReport, SystemInfo};

use crate::api_error::ApiError;

pub fn system_info() -> SystemInfo {
    collect_system_info()
}

pub fn system_doctor() -> DoctorReport {
    run_doctor()
}

pub fn system_info_json() -> Result<String, ApiError> {
    serde_json::to_string(&system_info()).map_err(|_| ApiError::internal())
}

pub fn system_doctor_json() -> Result<String, ApiError> {
    serde_json::to_string(&system_doctor()).map_err(|_| ApiError::internal())
}
