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

/// Whether `device` is a Vulkan GPU with matrix cores (cooperative matrices), which its f32 matmuls round to f16 on
/// and which take f16 weights in place (an iGPU's prompt matmuls of quantized weights stay off them).
pub fn has_matrix_cores(device: &crate::Device) -> bool {
    #[cfg(feature = "vulkan")]
    return device
        .as_vulkan_device()
        .is_ok_and(|d| d.context().has_matrix_cores());
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

/// The block types a [`BlockWeight`] takes, as ggml stores them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Q8_0,
    Q5_0,
    Q4_0,
    Q4_1,
    Q5_1,
    Q4K,
    Q5K,
    Q6K,
    Iq4Xs,
    Iq3S,
    Q3K,
}

/// A weight of ggml blocks in a Vulkan device's memory: `(experts, rows, cols)`, one expert for a dense matrix.
#[derive(Clone)]
pub struct BlockWeight {
    #[cfg(feature = "vulkan")]
    buffer: std::sync::Arc<inference_vulkan::Buffer>,
    kind: BlockKind,
    dims: (usize, usize, usize),
}

impl std::fmt::Debug for BlockWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockWeight")
            .field("kind", &self.kind)
            .field("dims", &self.dims)
            .finish()
    }
}

impl BlockWeight {
    /// `bytes` (row-major blocks, `experts * rows` rows of `cols`) copied to `device`'s memory.
    pub fn new(
        device: &crate::Device,
        kind: BlockKind,
        dims: (usize, usize, usize),
        bytes: &[u8],
    ) -> Result<Self> {
        #[cfg(feature = "vulkan")]
        return Ok(Self {
            buffer: crate::vulkan_backend::block_weight(device, kind, dims, bytes)?,
            kind,
            dims,
        });
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (device, kind, dims, bytes);
            Err(crate::Error::NotCompiledWithVulkanSupport)
        }
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn from_buffer(
        buffer: std::sync::Arc<inference_vulkan::Buffer>,
        ty: inference_vulkan::BlockType,
        dims: (usize, usize, usize),
    ) -> Self {
        Self {
            buffer,
            kind: crate::vulkan_backend::block_kind(ty),
            dims,
        }
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        self.dims
    }

    /// `act(x G^T) * (x U^T)` for this gate and `up`, dense weights of one shape and kind, as one matvec: `None` unless
    /// `x` is a decode's few contiguous f32 rows on Vulkan.
    pub fn glu_mat_vec(&self, up: &Self, x: &Tensor, act: GluAct) -> Result<Option<Tensor>> {
        if self.kind != up.kind || self.dims != up.dims || self.dims.0 != 1 {
            return Ok(None);
        }
        #[cfg(feature = "vulkan")]
        return crate::vulkan_backend::glu_mat_vec(
            x,
            (&self.buffer, &up.buffer),
            crate::vulkan_backend::block_type(self.kind),
            (self.dims.1, self.dims.2),
            act,
        );
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (x, act);
            Err(crate::Error::NotCompiledWithVulkanSupport)
        }
    }

    /// `y[slot] = W[ids[slot]] x[slot / x_div]`: `x` f32 rows of `cols`, `ids` u32 `(tokens, k)`; `(slots, rows)`.
    pub fn expert_mat_vec(&self, x: &Tensor, ids: &Tensor, x_div: usize) -> Result<Tensor> {
        #[cfg(feature = "vulkan")]
        return crate::vulkan_backend::expert_mat_vec(
            &self.buffer,
            crate::vulkan_backend::block_type(self.kind),
            self.dims,
            &x.to_dtype(crate::DType::F32)?.contiguous()?,
            &ids.to_dtype(crate::DType::U32)?.contiguous()?,
            x_div,
        );
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (x, ids, x_div);
            Err(crate::Error::NotCompiledWithVulkanSupport)
        }
    }

    /// The first expert's rows at the u32 `ids`, dequantized: `(ids, cols)` f32.
    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "vulkan")]
        return crate::vulkan_backend::block_embedding(
            &self.buffer,
            crate::vulkan_backend::block_type(self.kind),
            (self.dims.1, self.dims.2),
            ids,
        );
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = ids;
            Err(crate::Error::NotCompiledWithVulkanSupport)
        }
    }

    /// `x (.., cols)` times the first expert's matrix transposed: `(.., rows)` f32.
    pub fn matmul(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "vulkan")]
        return crate::vulkan_backend::block_matmul(
            &self.buffer,
            crate::vulkan_backend::block_type(self.kind),
            (self.dims.1, self.dims.2),
            x,
        );
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = x;
            Err(crate::Error::NotCompiledWithVulkanSupport)
        }
    }
}

/// RoPE as inference_quant's CPU kernel takes it: `x` `(batch, heads, seq, head_dim)`, `cos`/`sin` `(rows, rot)` rows
/// a position (`positions` `(batch seq,)` u32 picks them), a token (`batch seq` rows) or a sequence index (`seq` rows);
/// `None` where the device or dtypes have no kernel.
pub fn rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Option<Tensor>> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::rope(x, cos, sin, positions, is_neox);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (x, cos, sin, positions, is_neox);
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// Gated DeltaNet's recurrence from the convolved `mixed` `(batch, seq, conv_dim)` (q and k `k_heads * head_k` each,
/// then v), `b` and `a` `(batch, seq, v_heads)`, `a_log` and `dt_bias` `(v_heads,)`, all f32, over `state` `(batch,
/// v_heads, head_k, head_v)` f32 key-major; value head h reads key head h % k_heads when `tiled`, else h / (v_heads /
/// k_heads). `None` where the device has no kernel for the widths.
pub struct GdnRecurrence<'a> {
    pub mixed: &'a Tensor,
    pub b: &'a Tensor,
    pub a: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub state: &'a Tensor,
    pub k_heads: usize,
    pub tiled: bool,
    pub eps: f32,
    /// advance `state`'s own storage (a contiguous f32 view nothing else reads), returning it
    pub in_place: bool,
}

/// The recurrence's output `(batch, seq, v_heads, head_v)` and the advanced state.
pub fn gdn_recurrence(op: &GdnRecurrence<'_>) -> Result<Option<(Tensor, Tensor)>> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::gdn_recurrence(op);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = op;
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
    gated_rmsnorm_in(x, gate, weight, eps, group_size, GatedNorm::Mamba2)
}

/// A GLU's activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GluAct {
    Silu,
    Gelu,
    Relu,
    GeluErf,
    Sigmoid,
}

/// `act(a) * b` in one pass, `a` and `b` f32 of one shape; `None` where the device has no kernel for the layout.
pub fn glu(a: &Tensor, b: &Tensor, act: GluAct) -> Result<Option<Tensor>> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::glu(a, b, act);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (a, b, act);
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

/// Where a gated RMSNorm's gate goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedNorm {
    /// the norm of `x * silu(gate)`, then the weight (Mamba-2)
    Mamba2,
    /// the norm of `x`, the weight, then `silu(gate)` (Gated DeltaNet)
    NormThenSilu,
    /// as `NormThenSilu` with `sigmoid(gate)`
    NormThenSigmoid,
}

/// `x` (f32, `(.., k)`) times an f16 `(n, k)` weight transposed, on a Vulkan device's matrix cores: `(.., n)` f32, or
/// `None` where the device has none or the product is too small for them.
pub fn linear_half_weight(x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::linear_half_weight(x, weight);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (x, weight);
        Ok(None)
    }
}

/// `(x + residual, rms_norm(x + residual) * weight)` in one dispatch, for a decode's few f32 rows on Vulkan; `None`
/// where the kernel does not take the shapes.
pub fn add_rms_norm(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::add_rms_norm(x, residual, weight, eps);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (x, residual, weight, eps);
        Ok(None)
    }
}

/// [`gated_rmsnorm`] with the gate where `order` puts it.
pub fn gated_rmsnorm_in(
    x: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f32,
    group_size: usize,
    order: GatedNorm,
) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    return crate::vulkan_backend::gated_rmsnorm(x, gate, weight, eps, group_size, order);
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (x, gate, weight, eps, group_size, order);
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

/// Records this thread's work on `device` on `key`'s own stream (its own queue, beside other keys' work) until the
/// scope drops: `key` names one model, which shares nothing with the others but its loaded weights. Nothing on a device
/// with one stream, or another kind of device.
#[must_use]
pub struct StreamScope {
    #[cfg(feature = "vulkan")]
    _scope: Option<inference_vulkan::StreamScope>,
}

pub fn stream_scope(device: &crate::Device, key: usize) -> StreamScope {
    #[cfg(feature = "vulkan")]
    return StreamScope {
        _scope: device
            .as_vulkan_device()
            .ok()
            .map(|d| d.context().stream_scope(key)),
    };
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (device, key);
        StreamScope {}
    }
}

/// A Vulkan device the backend can open, as `vulkan:N` names it.
#[derive(Debug, Clone, PartialEq)]
pub struct VulkanDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    /// `discrete`, `integrated`, `virtual` or `cpu`.
    pub kind: &'static str,
    pub memory_bytes: u64,
    pub vendor_id: u32,
}

/// The Vulkan devices the backend can open, in `vulkan:N` order (discrete GPUs first), read without opening one.
pub fn devices() -> Result<Vec<VulkanDeviceInfo>> {
    #[cfg(feature = "vulkan")]
    return inference_vulkan::devices()
        .map(|devices| {
            devices
                .into_iter()
                .map(|d| VulkanDeviceInfo {
                    ordinal: d.ordinal,
                    name: d.name,
                    kind: d.kind,
                    memory_bytes: d.memory_bytes,
                    vendor_id: d.vendor_id,
                })
                .collect()
        })
        .map_err(|err| crate::Error::Msg(err.to_string()));
    #[cfg(not(feature = "vulkan"))]
    Err(crate::Error::NotCompiledWithVulkanSupport)
}

/// Frees `device`'s idle pooled buffers where they are host RAM (an iGPU); nothing elsewhere.
pub fn release_idle_memory(device: &crate::Device) {
    #[cfg(feature = "vulkan")]
    if let crate::Device::Vulkan(device) = device {
        device.context().release_idle();
    }
    #[cfg(not(feature = "vulkan"))]
    let _ = device;
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
