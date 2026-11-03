//! Vulkan kernels model code calls on tensors directly: Mamba's decode steps on its state pools. Every function errs
//! off a Vulkan build; callers branch on `Device::is_vulkan`.

use crate::{Result, Tensor};

/// One decode step of Mamba-2's selective scan: `x` `(batch, 1, heads, head_dim)`, `dt` `(batch, 1, heads)`, `b` and
/// `c` `(batch, 1, groups, state)`, `a`, `d` and `dt_bias` `(heads,)`, all f32; the pool `(rows, heads, head_dim,
/// state)` advances in place at the u32 `slots` rows.
pub struct ScanStep<'a> {
    pub x: &'a Tensor,
    pub dt: &'a Tensor,
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    pub c: &'a Tensor,
    pub d: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub dt_range: [f32; 2],
    pub pool: &'a Tensor,
    pub slots: &'a Tensor,
}

/// The scan step's `y`, `(batch, 1, heads, head_dim)`.
pub fn ssm_scan_step(step: &ScanStep<'_>) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::ssm_scan_step(step);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = step;
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// One decode step of Mamba's causal conv and SiLU: `x` `(batch, channels)`, `weight` `(channels, kernel)`, `bias`
/// `(channels,)`, f32; the pool `(rows, channels, kernel)` shifts in place at the u32 `slots` rows.
pub fn ssm_conv_step(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    pool: &Tensor,
    slots: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::ssm_conv_step(x, weight, bias, pool, slots);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (x, weight, bias, pool, slots);
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}
