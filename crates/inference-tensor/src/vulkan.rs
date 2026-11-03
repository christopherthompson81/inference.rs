//! Vulkan kernels model code calls on tensors directly: Mamba's decode steps on its state pools. Every function errs
//! off a Vulkan build; callers branch on `Device::is_vulkan`.

use crate::{Result, Tensor};

/// The states a Mamba head's row holds that [`ssm_scan_step`] takes (its lanes a row times states a lane).
pub const SCAN_STEP_STATE: usize = 128;

/// `seq` steps of Mamba-2's selective scan: `x` `(batch, seq, heads, head_dim)`, `dt` `(batch, seq, heads)`, `b` and
/// `c` `(batch, seq, groups, state)`, `a`, `d` and `dt_bias` `(heads,)`, all f32; the pool `(rows, heads, head_dim,
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

/// The scan's `y`, `(batch, seq, heads, head_dim)`.
pub fn ssm_scan_step(step: &ScanStep<'_>) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::ssm_scan_step(step);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = step;
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// `seq` steps of Mamba's causal conv and SiLU: `x` `(batch, seq, channels)`, `weight` `(channels, kernel)`, `bias`
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

/// How MoE routing turns logits into scores, or the chosen experts into weights (where `Raw` weighs each by its score).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RouterFn {
    Raw,
    Softmax,
    Sigmoid,
}

/// Top-k routing of `(tokens, experts)` f32 logits, as inference_nn's CPU router computes it.
pub struct Route<'a> {
    pub logits: &'a Tensor,
    /// added to the scores for choosing the experts, not for weighing them
    pub selection_bias: Option<&'a Tensor>,
    pub top_k: usize,
    pub score_fn: RouterFn,
    pub weight_fn: RouterFn,
    pub renormalize: bool,
    pub norm_min: f32,
    pub output_scale: f32,
    pub clip: Option<(f32, f32)>,
}

/// The routes' weights (f32) and expert ids (u32), each `(tokens, top_k)`.
pub fn moe_router(route: &Route<'_>) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::moe_router(route);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = route;
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// Writes `key` and `value` rows `(tokens, kv_heads, head_size)` into f32 vLLM-layout caches at the i64
/// `slot_mapping` (a negative slot skipped), in place.
pub fn paged_cache_write(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::paged_cache_write(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    );
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (key, value, key_cache, value_cache, slot_mapping);
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// One query a sequence (`q` `(seqs, heads, head_size)`) against f32 vLLM-layout caches through u32 `block_tables`
/// `(seqs, max_blocks)` over u32 `context_lens`, softmax-scaled by `scale`; or `None` past the kernel's longest
/// context.
pub struct PagedDecode<'a> {
    pub q: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub block_tables: &'a Tensor,
    pub context_lens: &'a Tensor,
    pub max_context: usize,
    pub scale: f32,
}

pub fn paged_decode(op: &PagedDecode<'_>) -> Result<Option<Tensor>> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::paged_decode(op);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = op;
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}
