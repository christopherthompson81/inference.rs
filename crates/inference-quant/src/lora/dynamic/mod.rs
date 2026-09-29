#[cfg(any(feature = "cuda", test))]
mod cuda;
#[cfg(feature = "cuda")]
mod cuda_ffi;
mod execution;
mod expert;
#[cfg(feature = "cuda")]
mod expert_cuda;
mod linear;
mod loader;
mod moe_cuda;
#[cfg(feature = "cuda")]
mod moe_cuda_ffi;
mod raw;
mod reference;
mod registry;

pub use execution::{
    LoraAdapterWeights, LoraExecution, LoraExecutionArena, LoraExecutionArenaStats, LoraSlotId,
    LoraWeights, has_active_lora_execution, with_lora_execution, with_lora_execution_repeated_row,
    with_lora_execution_row_range,
};
pub use expert::{
    DynamicLoraWeights, LoraExpertDelta, LoraExpertExecution, LoraExpertInputMode,
    LoraExpertProjection, LoraExpertProjectionNames, LoraExpertProjectionWeights,
    LoraExpertSiteHandle, LoraExpertSiteSpec, LoraExpertWeights, LoraGateUpOrder,
    add_expert_delta_reference,
};
pub use linear::maybe_wrap_dynamic_lora;
pub(crate) use linear::maybe_wrap_dynamic_lora_with_key;
pub use loader::{DynamicLoraLoadPlan, load_dynamic_lora_weights, plan_dynamic_lora_weights};
pub use moe_cuda::{
    ROUTED_LORA_BASE_SLOT, ROUTED_LORA_BLOCK_SIZE, ROUTED_LORA_MAX_RANK, ROUTED_LORA_WMMA_RANK_CAP,
    RoutedLoraAdapterWeight, RoutedLoraInputMode, RoutedLoraMetadataLayout,
    RoutedLoraProjectionLayout,
};
#[cfg(feature = "cuda")]
pub use moe_cuda::{
    RoutedLoraCudaMetadata, RoutedLoraCudaWeightTable, RoutedLoraDirectLaunch,
    RoutedLoraGroupedLaunch, launch_routed_lora_direct, launch_routed_lora_grouped,
};
pub use raw::{apply_dynamic_lora_delta, is_dynamic_lora_site_active, register_dynamic_lora_site};
pub(crate) use registry::LoraParallelism;
pub use registry::{
    LoraLayerRegistry, LoraLinearSpec, LoraRuntimeId, LoraSiteHandle, LoraSiteKey, LoraSiteSlice,
};

pub(crate) use execution::current_lora_execution;
pub(crate) use reference::add_delta;
