//! Vulkan kernels model code calls on tensors directly: Mamba's decode steps on its state pools. Every function errs
//! off a Vulkan build; callers branch on `Device::is_vulkan`.

use crate::{Result, Tensor};

/// Set to keep a Vulkan device's prompt matmuls off its matrix cores, on the int8 kernels decode uses.
pub const NO_MATRIX_CORES_ENV: &str = "INFERENCE_VULKAN_NO_COOPMAT";

/// Set to put an integrated GPU's prompt matmuls on its matrix cores too; by default only its prompt attention takes them.
pub const IGPU_MATRIX_CORES_ENV: &str = "INFERENCE_VULKAN_IGPU_COOPMAT";

/// Whether `device` is a discrete Vulkan GPU, with memory of its own rather than an iGPU's share of the host's.
pub fn is_discrete(device: &crate::Device) -> bool {
    #[cfg(feature = "vulkan")]
    return device
        .as_vulkan_device()
        .is_ok_and(|d| !d.context().is_unified());
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = device;
        false
    }
}

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

/// Mamba-2's gated RMSNorm, as `mamba_gated_rmsnorm_cuda` computes it: `x` and `gate` f32 `(.., width)`, `weight`
/// `(width,)` f32, groups of `group_size` along the last dimension.
pub fn gated_rmsnorm(
    x: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f32,
    group_size: usize,
) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::gated_rmsnorm(x, gate, weight, eps, group_size);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (x, gate, weight, eps, group_size);
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

/// Fused attention: softmax(`scale` q k^T + `mask`) v over f32 `q` `(batch, heads, positions, head_size)` and `k`, `v`
/// `(batch, kv_heads, keys, head_size)`, `mask` additive and broadcast from `(positions, keys)`; `causal` lets the
/// kernel skip keys past each position's diagonal, which the mask must hide as well. `None` for a shape or mask the
/// kernel does not take.
pub struct FlashAttn<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub mask: Option<&'a Tensor>,
    pub scale: f32,
    pub causal: bool,
}

pub fn flash_attn(op: &FlashAttn<'_>) -> Result<Option<Tensor>> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::flash_attn(op);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = op;
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// Attention with a relative-position term (Transformer-XL's, as the FastConformer's), a query head a key head:
/// score(i, j) = `q_content`_i . `k`_j + `q_position`_i . `positions`[pos_base - i + j], both query sets scaled
/// beforehand, softmax over the keys, times `v`. `q_content` and `q_position` are `(batch, heads, rows, head_size)`,
/// `k` and `v` `(batch, heads, keys, head_size)`, `positions` `(batch, heads, p_rows, head_size)`; the output is laid
/// out as the queries. On matrix cores, its inputs rounded to f16; `None` where the device or shape has no kernel.
pub struct RelPosAttn<'a> {
    pub q_content: &'a Tensor,
    pub q_position: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub positions: &'a Tensor,
    pub pos_base: usize,
}

pub fn rel_pos_attn(op: &RelPosAttn<'_>) -> Result<Option<Tensor>> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::rel_pos_attn(op);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = op;
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// The packed top-[`RANKED_TOP_K`] of each row of f32 `logits` `(rows, vocab)`: `(rows, 2 K + 2)`, the largest values
/// sorted descending (ties to the lower index), their indices as floats, the softmax normalizer at `1 / temperature`
/// and the max over the temperature (CUDA's packed top-k layout).
pub fn ranked_top_k(logits: &Tensor, temperature: f64) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::ranked_top_k(logits, temperature);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (logits, temperature);
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// A copy of f32 `logits` `(rows, vocab)` with the sampler's penalties applied through `entries`: one a distinct token of
/// a row, `[row << 1 | seen, token, generated count, bias bits]` (see the Vulkan `Penalize` op).
pub fn penalize(
    logits: &Tensor,
    entries: &[[u32; 4]],
    frequency: f32,
    presence: f32,
    repetition: f32,
) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::penalize(logits, entries, frequency, presence, repetition);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (logits, entries, frequency, presence, repetition);
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// Words a row of [`ranked_draw`]'s params holds: inverse temperature, top-k (0 for none), top-p and min-p (f32
/// bits), and the uniform the draw lands at.
pub const RANKED_DRAW_PARAM_WORDS: usize = 5;

/// A token drawn on the device from each row of [`ranked_top_k`]'s packed `(rows, 2K + 2)` rows under the u32
/// `(rows, RANKED_DRAW_PARAM_WORDS)` params: the support's `(rows, K)` u32 ids and f32 normalized probabilities (the
/// draw's distribution; a nucleus past K is cut there), and the `(rows,)` u32 tokens.
pub fn ranked_draw(packed: &Tensor, params: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::ranked_draw(packed, params);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (packed, params);
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// The values a row of [`ranked_top_k`] holds.
pub const RANKED_TOP_K: usize = 128;

/// One query a sequence (`q` `(seqs, heads, head_size)`) against f32 vLLM-layout caches through u32 `block_tables`
/// `(seqs, max_blocks)` over u32 `context_lens`, softmax-scaled by `scale`.
/// One sequence's prompt rows `q` `(rows, heads, head_size)` over its `context` cached positions (the rows' own
/// included, last; row r causal over the first `context - rows + r + 1`) through `block_table` (its row, u32); the
/// output is laid out as `q`.
pub struct PagedPrefill<'a> {
    pub q: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub block_table: &'a Tensor,
    pub context: usize,
    pub scale: f32,
}

pub fn paged_prefill(op: &PagedPrefill<'_>) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::paged_prefill(op);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = op;
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

pub struct PagedDecode<'a> {
    pub q: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub block_tables: &'a Tensor,
    pub context_lens: &'a Tensor,
    pub max_context: usize,
    pub scale: f32,
}

pub fn paged_decode(op: &PagedDecode<'_>) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::paged_decode(op);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = op;
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}
