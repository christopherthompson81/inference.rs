//! MoE support that is independent of any specific quant scheme.

pub mod cuda;
#[cfg(has_cutlass_moe_kernels)]
pub mod cutlass;

/// CUTLASS grouped-GEMM MoE forward; errors when the kernels were not compiled in.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_fused_moe(
    xs: &inference_tensor::Tensor,
    gate_up: &inference_tensor::Tensor,
    down: &inference_tensor::Tensor,
    topk_ids: &inference_tensor::Tensor,
    topk_weights: &inference_tensor::Tensor,
    num_experts: usize,
    act: cuda::GatedAct,
    dev: &inference_tensor::CudaDevice,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    #[cfg(has_cutlass_moe_kernels)]
    {
        cutlass::cutlass_fused_moe(
            xs,
            gate_up,
            down,
            topk_ids,
            topk_weights,
            num_experts,
            act,
            dev,
        )
    }
    #[cfg(not(has_cutlass_moe_kernels))]
    {
        let _ = (
            xs,
            gate_up,
            down,
            topk_ids,
            topk_weights,
            num_experts,
            act,
            dev,
        );
        inference_tensor::bail!(
            "CUTLASS MoE kernels were not compiled in (requires sm_80+ at build)"
        )
    }
}

/// Whether the CUTLASS grouped-GEMM MoE kernels were compiled in and the device can run them.
pub fn cutlass_moe_available(dev: &inference_tensor::CudaDevice) -> bool {
    #[cfg(has_cutlass_moe_kernels)]
    {
        dev.compute_major() >= 8
    }
    #[cfg(not(has_cutlass_moe_kernels))]
    {
        let _ = dev;
        false
    }
}
