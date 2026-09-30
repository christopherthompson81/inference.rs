//! Which model and quantization to load, and what fits this machine: model selection, the loader it builds, quant
//! discovery, auto-tuning, resource planning and the environment doctor. Built on inference-core's public surface.

mod diagnostics;
mod model_loader;
pub mod model_metadata;
mod model_selected;
pub mod quant;
pub mod resource_plan;
mod tuning;

pub use diagnostics::{
    BuildInfo, CpuInfo, DeviceInfo, DoctorCheck, DoctorReport, DoctorStatus, HfConnectivityInfo,
    MemoryInfo, SystemInfo, ToolchainInfo, check_hf_gated_access, collect_system_info,
    parse_nvidia_smi_cuda_version, run_doctor,
};
pub use model_loader::{
    LoaderBuilder, get_auto_device_map_params, get_model_dtype, get_tgt_non_granular_index,
};
pub use model_selected::{MmprojSelection, ModelSelected};
pub use resource_plan::{
    PagedKvModelRequest, PagedKvPlan, PagedKvPolicy, RuntimeResourcePlanOptions, plan_paged_kv,
};
pub use tuning::{
    AutoTuneRequest, AutoTuneResult, FitStatus, QualityTier, TuneCandidate, TuneProfile, auto_tune,
};
