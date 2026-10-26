//! Qwen3.8-Flash-Next's QSA: a small indexer scores each complete block of `compress_ratio` keys by its pooled key,
//! and a query past the budget attends only to its top blocks plus the incomplete tail. The indexer's per-token keys
//! live in an auxiliary paged layer: raw keys in K, and in V at a block's last cell the block's pooled key.

use std::sync::Arc;

use inference_nn::cuda::qsa::{
    QSA_PADDED_ROW, qsa_cells_width, qsa_decode_plan, qsa_select_cells, qsa_select_mask,
};
use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::config::{QsaConfig, TextConfig};
use crate::{
    attention::SdpaParams,
    layers::GemmaRmsNorm,
    paged_attention::{
        GatheredAttention, PagedAttention, PagedAttentionInputMetadata, SelectedAttention,
    },
    utils::unvarbuilder::UnVarBuilder,
};

// query rows scored and attended at once: bounds the score and mask buffers of a long prefill chunk
const QSA_QUERY_SLICE: usize = 128;
// steps with at most this many query rows (decode, verification) gather each row's kept cells instead of masking
const QSA_GATHER_ROWS: usize = 16;

enum IndexerProjection {
    // the checkpoint's `index_qk_proj`: the queries' heads, then the key
    Fused(Arc<dyn QuantMethod>),
    Split {
        q: Arc<dyn QuantMethod>,
        k: Arc<dyn QuantMethod>,
    },
}

pub(super) struct QsaIndexer {
    projection: IndexerProjection,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    cfg: QsaConfig,
    cache_layer: usize,
}

/// What one full-attention layer's QSA needs from the step: the plan and its indexer cache.
#[derive(Clone, Copy)]
pub(super) struct QsaLayerInput<'a> {
    pub(super) step: &'a QsaStep,
    pub(super) cache: (&'a Tensor, &'a Tensor),
}

/// One forward's QSA plan, shared by every QSA layer: the blocks this step completes and, past the budget, the
/// pooled keys to score and each query row's position.
pub(super) struct QsaStep {
    pooling: Option<Pooling>,
    selection: Option<Selection>,
}

struct Pooling {
    // the members' cells in block order, then each block's last cell (where its pooled key goes, so the first cell's
    // V keeps the cos | sin the block rotates with when a rejected draft's pooling is redone)
    members: Tensor,
    lasts: Tensor,
}

struct Selection {
    // (batch, pools) cells of each sequence's block ends, where the pooled keys are
    pool_cells: Tensor,
    // (batch, query_len) positions, QSA_PADDED_ROW on padding
    positions: Tensor,
    seq_lens: Tensor,
    n_kv: usize,
    gather: Option<Gather>,
}

// Per query row (batch-major): its sequence's block table row, the cells it keeps, and the compact cache's table.
struct Gather {
    table: Tensor,
    kept: Tensor,
    compact_table: Tensor,
}

/// A batch row of this step: its first position and how many real tokens it carries.
#[derive(Clone, Copy, Debug)]
pub(super) struct QsaRow {
    pub(super) start: usize,
    pub(super) len: usize,
}

impl QsaIndexer {
    pub(super) fn load(
        vb: ShardedVarBuilder,
        vb_norms: ShardedVarBuilder,
        cfg: &TextConfig,
        qsa: QsaConfig,
        cache_layer: usize,
    ) -> Result<Self> {
        let linear = |name: &str, outputs: usize| {
            inference_quant::ReplicatedLayer::new(
                cfg.hidden_size,
                outputs,
                &cfg.quantization_config,
                false,
                vb.pp(name),
            )
        };
        let q_width = qsa.n_heads * qsa.head_dim;
        let projection = if vb.contains_tensor("index_qk_proj.weight") {
            IndexerProjection::Fused(linear(
                "index_qk_proj",
                q_width + qsa.kv_heads * qsa.head_dim,
            )?)
        } else {
            IndexerProjection::Split {
                q: linear("index_q_proj", q_width)?,
                k: linear("index_k_proj", qsa.kv_heads * qsa.head_dim)?,
            }
        };
        Ok(Self {
            projection,
            q_norm: GemmaRmsNorm::new(qsa.head_dim, cfg.rms_norm_eps, vb_norms.pp("q_layernorm"))?,
            k_norm: GemmaRmsNorm::new(qsa.head_dim, cfg.rms_norm_eps, vb_norms.pp("k_layernorm"))?,
            cfg: qsa,
            cache_layer,
        })
    }

    pub(super) fn cache_layer(&self) -> usize {
        self.cache_layer
    }

    // `(b, s, heads, d)` queries and `(b, s, 1, d)` raw keys
    fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let d = self.cfg.head_dim;
        let q_width = self.cfg.n_heads * d;
        let (q, k) = match &self.projection {
            IndexerProjection::Fused(qk) => {
                let qk = qk.forward(x)?;
                (
                    qk.narrow(D::Minus1, 0, q_width)?,
                    qk.narrow(D::Minus1, q_width, d)?,
                )
            }
            IndexerProjection::Split { q, k } => (q.forward(x)?, k.forward(x)?),
        };
        Ok((
            q.reshape((b, s, self.cfg.n_heads, d))?,
            k.reshape((b, s, 1, d))?,
        ))
    }

    /// Writes this step's raw keys, pools the blocks it completes, and past the budget returns the normed, rotated
    /// `(b, heads, s, d)` queries that score them.
    pub(super) fn prepare(
        &self,
        x: &Tensor,
        cos_sin: &(Tensor, Tensor),
        input: QsaLayerInput<'_>,
        paged_attn: &PagedAttention,
        input_metadata: &PagedAttentionInputMetadata,
    ) -> Result<Option<Tensor>> {
        let (q, raw) = self.project(x)?;
        let (k_cache, v_cache) = input.cache;
        let device = x.device();
        let slots = input_metadata
            .slot_mappings
            .get(&device.location())
            .ok_or_else(|| {
                inference_tensor::Error::msg("QSA has no slot mapping for its device")
            })?;
        // each V row holds its token's cos | sin until its block completes and takes the pooled key, so pooling
        // rotates to the block's first position as the step placed it, M-RoPE included
        let half = cos_sin.0.dim(D::Minus1)?;
        let (b, s, _) = x.dims3()?;
        let rotation = Tensor::cat(
            &[
                &cos_sin.0.to_device(device)?.to_dtype(raw.dtype())?,
                &cos_sin.1.to_device(device)?.to_dtype(raw.dtype())?,
                &Tensor::zeros((b, s, self.cfg.head_dim - 2 * half), raw.dtype(), device)?,
            ],
            D::Minus1,
        )?
        .unsqueeze(1)?;
        paged_attn.write_rows(&raw.transpose(1, 2)?, &rotation, k_cache, v_cache, slots)?;
        if let Some(pooling) = &input.step.pooling {
            self.pool(pooling, half, paged_attn, k_cache, v_cache)?;
        }
        if input.step.selection.is_none() {
            return Ok(None);
        }
        let q = crate::ops::rms_norm(&q, self.q_norm.weight(), self.q_norm.eps() as f32)?;
        let cos_sin = (cos_sin.0.to_device(device)?, cos_sin.1.to_device(device)?);
        Ok(Some(rope_prefix(&q.transpose(1, 2)?, &cos_sin)?))
    }

    fn pool(
        &self,
        pooling: &Pooling,
        half: usize,
        paged_attn: &PagedAttention,
        k_cache: &Tensor,
        v_cache: &Tensor,
    ) -> Result<()> {
        let d = self.cfg.head_dim;
        let cells = pooling.members.to_device(k_cache.device())?;
        let members = cache_rows(k_cache, d)?.index_select(&cells, 0)?;
        let blocks = members.dim(0)? / self.cfg.compress_ratio;
        let rotation = cache_rows(v_cache, d)?
            .index_select(&cells, 0)?
            .reshape((blocks, self.cfg.compress_ratio, d))?
            .narrow(1, 0, 1)?
            .reshape((1, blocks, d))?;
        let last = members
            .reshape((blocks, self.cfg.compress_ratio, d))?
            .narrow(1, self.cfg.compress_ratio - 1, 1)?
            .reshape((1, 1, blocks, d))?;
        // the reference means in f32, then norms and rotates to the block's first position
        let pooled = members
            .to_dtype(DType::F32)?
            .reshape((blocks, self.cfg.compress_ratio, d))?
            .mean(1)?
            .to_dtype(members.dtype())?
            .reshape((1, 1, blocks, d))?;
        let pooled = crate::ops::rms_norm(&pooled, self.k_norm.weight(), self.k_norm.eps() as f32)?;
        let pooled = rope_prefix(
            &pooled,
            &(
                rotation.narrow(D::Minus1, 0, half)?,
                rotation.narrow(D::Minus1, half, half)?,
            ),
        )?;
        // K keeps the last member's raw key; V takes the pooled key
        paged_attn.write_rows(
            &last,
            &pooled.contiguous()?,
            k_cache,
            v_cache,
            &pooling.lasts.to_device(k_cache.device())?,
        )
    }

    /// Attention over the cells QSA keeps for each query: scores, the mask, then fattn, a slice of rows at a time.
    pub(super) fn attend(&self, call: QsaAttention<'_>) -> Result<Tensor> {
        let QsaAttention {
            q_index,
            query,
            input,
            key_cache,
            value_cache,
            paged_attn,
            input_metadata,
            sdpa_params,
        } = call;
        let selection = input
            .step
            .selection
            .as_ref()
            .expect("prepare returned queries only with a selection");
        let device = query.device();
        let d = self.cfg.head_dim;
        let (b, heads, s, _) = q_index.dims4()?;
        let pools = selection.pool_cells.dim(1)?;
        let pooled = cache_rows(input.cache.1, d)?
            .index_select(&selection.pool_cells.to_device(device)?.flatten_all()?, 0)?
            .reshape((b, pools, d))?
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?;
        let positions = selection.positions.to_device(device)?;
        let scale = 1. / (d as f64).sqrt();
        let scores = |start: usize, rows: usize| -> Result<Tensor> {
            let q = q_index
                .narrow(2, start, rows)?
                .transpose(1, 2)?
                .reshape((b, rows * heads, d))?
                .to_dtype(DType::F32)?;
            (q.matmul(&pooled)?
                .relu()?
                .reshape((b, rows, heads, pools))?
                .sum(2)?
                * scale)?
                .reshape((b * rows, pools))
        };
        let k = self.cfg.budget / self.cfg.compress_ratio;
        if let Some(gather) = &selection.gather {
            let cells = qsa_select_cells(
                &scores(0, s)?,
                &positions.flatten_all()?,
                k,
                self.cfg.compress_ratio,
            )?;
            // each query row attends alone, as its own one-token sequence
            let (_, q_heads, _, q_dim) = query.dims4()?;
            let rows_query = query
                .transpose(1, 2)?
                .reshape((b * s, 1, q_heads, q_dim))?
                .transpose(1, 2)?;
            let out = paged_attn.attend_gathered(GatheredAttention {
                query: &rows_query,
                cells: &cells,
                kept: &gather.kept.to_device(device)?,
                block_table: &gather.table.to_device(device)?,
                compact_table: &gather.compact_table.to_device(device)?,
                key_cache,
                value_cache,
                sdpa_params,
            })?;
            return out
                .transpose(1, 2)?
                .reshape((b, s, q_heads, ()))?
                .transpose(1, 2);
        }
        let seq_lens = selection.seq_lens.to_device(device)?;
        let mut outputs = Vec::with_capacity(s.div_ceil(QSA_QUERY_SLICE));
        for start in (0..s).step_by(QSA_QUERY_SLICE) {
            let rows = QSA_QUERY_SLICE.min(s - start);
            let scores = scores(start, rows)?;
            let row_positions = positions.narrow(1, start, rows)?.flatten_all()?;
            let mask = qsa_select_mask(
                &scores,
                &row_positions,
                selection.n_kv,
                k,
                self.cfg.compress_ratio,
            )?;
            outputs.push(paged_attn.attend_selected(SelectedAttention {
                query: &query.narrow(2, start, rows)?,
                mask: &mask,
                key_cache,
                value_cache,
                input_metadata,
                seq_lens: &seq_lens,
                sdpa_params,
            })?);
        }
        Tensor::cat(&outputs, 2)
    }

    pub(super) fn residual_tensors(&self, uvb: &UnVarBuilder) {
        uvb.pp("q_layernorm")
            .add_tensor("weight", self.q_norm.original_weight().clone());
        uvb.pp("k_layernorm")
            .add_tensor("weight", self.k_norm.original_weight().clone());
    }
}

/// One full-attention layer's QSA attention: its indexer queries, its own queries `(b, heads, s, head_dim)` and caches.
pub(super) struct QsaAttention<'a> {
    pub(super) q_index: &'a Tensor,
    pub(super) query: &'a Tensor,
    pub(super) input: QsaLayerInput<'a>,
    pub(super) key_cache: &'a Tensor,
    pub(super) value_cache: &'a Tensor,
    pub(super) paged_attn: &'a PagedAttention,
    pub(super) input_metadata: &'a PagedAttentionInputMetadata,
    pub(super) sdpa_params: &'a SdpaParams,
}

// NeoX rotation of the leading `2 * cos.dim(-1)` dims of `(b, heads, s, d)`, with `(b, s, half)` cos/sin.
fn rope_prefix(x: &Tensor, (cos, sin): &(Tensor, Tensor)) -> Result<Tensor> {
    let half = cos.dim(D::Minus1)?;
    let d = x.dim(D::Minus1)?;
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?
        .unsqueeze(1)?
        .to_dtype(x.dtype())?;
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?
        .unsqueeze(1)?
        .to_dtype(x.dtype())?;
    let rotated = x.narrow(D::Minus1, 0, 2 * half)?;
    let turned = Tensor::cat(
        &[
            &rotated.narrow(D::Minus1, half, half)?.neg()?,
            &rotated.narrow(D::Minus1, 0, half)?,
        ],
        D::Minus1,
    )?;
    let rotated = (rotated.broadcast_mul(&cos)? + turned.broadcast_mul(&sin)?)?;
    Tensor::cat(
        &[&rotated, &x.narrow(D::Minus1, 2 * half, d - 2 * half)?],
        D::Minus1,
    )
}

// An indexer cache as one row per cell: QSA's single key head in the HND layout, `(blocks, 1, block_size, d)`.
fn cache_rows(cache: &Tensor, head_dim: usize) -> Result<Tensor> {
    let (blocks, heads, block_size, d) = cache.dims4()?;
    if heads != 1 || d != head_dim {
        inference_tensor::bail!(
            "QSA's indexer cache must be one {head_dim}-wide head in the HND layout, got {:?}",
            cache.shape()
        );
    }
    cache.reshape((blocks * block_size, d))
}

/// Each listed `(row, position)`'s paged cell, through the device block table.
fn cells(table: &Tensor, block_size: usize, at: &[(usize, usize)]) -> Result<Tensor> {
    let width = table.dim(1)?;
    let device = table.device();
    let (entries, offsets): (Vec<u32>, Vec<u32>) = at
        .iter()
        .map(|&(row, position)| {
            (
                (row * width + position / block_size) as u32,
                (position % block_size) as u32,
            )
        })
        .unzip();
    let blocks = table
        .flatten_all()?
        .index_select(&Tensor::new(entries.as_slice(), device)?, 0)?;
    blocks.affine(block_size as f64, 0.)? + Tensor::new(offsets.as_slice(), device)?
}

/// What [`QsaStep::new`] reads from the forward.
pub(super) struct QsaStepInput<'a> {
    pub(super) cfg: QsaConfig,
    pub(super) rows: &'a [QsaRow],
    pub(super) query_len: usize,
    pub(super) input_metadata: &'a PagedAttentionInputMetadata,
    pub(super) device: &'a Device,
}

impl QsaStep {
    pub(super) fn new(input: QsaStepInput<'_>) -> Result<Self> {
        let QsaStepInput {
            cfg,
            rows,
            query_len,
            input_metadata,
            device,
        } = input;
        let ratio = cfg.compress_ratio;
        let block_size = input_metadata
            .block_size
            .ok_or_else(|| inference_tensor::Error::msg("QSA needs the paged block size"))?;
        if block_size % ratio != 0 {
            inference_tensor::bail!(
                "QSA's {ratio}-cell blocks must tile the {block_size}-cell pages"
            );
        }
        let table = input_metadata
            .block_tables
            .as_ref()
            .and_then(|tables| {
                tables
                    .get(&device.location())
                    .or_else(|| tables.values().next())
            })
            .ok_or_else(|| inference_tensor::Error::msg("QSA needs the paged block tables"))?
            .to_device(device)?;
        let context_lens = input_metadata.context_lens.as_ref().and_then(|lens| {
            lens.get(&device.location())
                .or_else(|| lens.values().next())
        });
        // a prompt's context lens leave out its own tokens, so only a completion step's count this one
        let completion =
            !input_metadata.is_first_prompt_chunk && input_metadata.num_cached_tokens.is_none();
        // a verify step gives each query token its own table row, each covering up to its token; a sequence's last
        // row covers all of it
        let per_token = query_len > 1 && table.dim(0)? == rows.len() * query_len;
        if completion
            && (query_len == 1 || per_token)
            && rows.iter().all(|r| r.len == query_len)
            && let Some(context_lens) = context_lens
        {
            return Self::decode(
                &cfg,
                &table,
                &context_lens.to_device(device)?,
                block_size,
                query_len,
            );
        }
        let table_row = move |row: usize| {
            if per_token {
                row * query_len + query_len - 1
            } else {
                row
            }
        };

        // blocks whose last cell this step writes
        let completed = rows
            .iter()
            .enumerate()
            .flat_map(|(row, r)| {
                (r.start.saturating_sub(ratio - 1).div_ceil(ratio)..(r.start + r.len) / ratio)
                    .map(move |block| (table_row(row), block))
            })
            .collect::<Vec<_>>();
        let pooling = if completed.is_empty() {
            None
        } else {
            let members = completed
                .iter()
                .flat_map(|&(row, block)| (0..ratio).map(move |m| (row, block * ratio + m)))
                .collect::<Vec<_>>();
            let members = cells(&table, block_size, &members)?;
            let lasts = members
                .reshape((completed.len(), ratio))?
                .narrow(1, ratio - 1, 1)?
                .flatten_all()?
                .to_dtype(DType::I64)?;
            Some(Pooling { members, lasts })
        };

        let longest = rows.iter().map(|r| r.start + r.len).max().unwrap_or(0);
        let selection = if longest > cfg.dense_context_len() {
            let pools = longest / ratio;
            let pool_cells = (0..rows.len())
                .flat_map(|row| {
                    let complete = (rows[row].start + rows[row].len) / ratio;
                    // past a row's own blocks, any valid cell: the kernel never reads them
                    (0..pools).map(move |block| {
                        (
                            if block < complete { table_row(row) } else { 0 },
                            block * ratio + ratio - 1,
                        )
                    })
                })
                .collect::<Vec<_>>();
            let pool_cells =
                cells(&table, block_size, &pool_cells)?.reshape((rows.len(), pools))?;
            let positions = rows
                .iter()
                .flat_map(|r| {
                    (0..query_len).map(move |c| {
                        if c < r.len {
                            (r.start + c) as u32
                        } else {
                            QSA_PADDED_ROW
                        }
                    })
                })
                .collect::<Vec<_>>();
            let seq_lens = rows
                .iter()
                .map(|r| (r.start + r.len) as u32)
                .collect::<Vec<_>>();
            // the mask path reads one table row per sequence
            let gather = (per_token || rows.len() * query_len <= QSA_GATHER_ROWS)
                .then(|| gather_plan(&cfg, &table, block_size, &positions, query_len, per_token))
                .transpose()?;
            Some(Selection {
                pool_cells,
                positions: Tensor::new(positions.as_slice(), device)?
                    .reshape((rows.len(), query_len))?,
                seq_lens: Tensor::new(seq_lens.as_slice(), device)?,
                n_kv: PagedAttention::selected_kv_len(input_metadata, device)?,
                gather,
            })
        } else {
            None
        };
        Ok(Self { pooling, selection })
    }
}

impl QsaStep {
    // A decode or verify step's plan, all on the device: the same for every step of a table width, so a graph
    // replays it. A verify's query tokens are rows of their own; tokens of one block pool it alike, from the cells
    // the step has written.
    fn decode(
        cfg: &QsaConfig,
        table: &Tensor,
        context_lens: &Tensor,
        block_size: usize,
        query_len: usize,
    ) -> Result<Self> {
        let ratio = cfg.compress_ratio;
        let k = cfg.budget / ratio;
        let (rows, width) = table.dims2()?;
        let batch = rows / query_len;
        let capacity = width * block_size;
        let pools = capacity / ratio;
        let plan = qsa_decode_plan(table, context_lens, block_size, ratio, k, pools)?;
        let pooling = Pooling {
            members: plan.narrow(1, 0, ratio)?.flatten_all()?,
            lasts: plan
                .narrow(1, ratio - 1, 1)?
                .flatten_all()?
                .to_dtype(DType::I64)?,
        };
        // a table this short never reaches the budget, so every step of its width attends densely
        let selection = if capacity > cfg.dense_context_len() {
            let positions = plan
                .narrow(1, ratio, 1)?
                .contiguous()?
                .reshape((batch, query_len))?;
            let pages = qsa_cells_width(k, ratio).div_ceil(block_size);
            let compact = (0..(rows * pages) as u32).collect::<Vec<_>>();
            // a sequence's pooled keys and length are its last query token's
            let last = |t: &Tensor, cols: usize| -> Result<Tensor> {
                t.reshape((batch, query_len, cols))?
                    .narrow(1, query_len - 1, 1)?
                    .reshape((batch, cols))?
                    .contiguous()
            };
            Some(Selection {
                pool_cells: last(&plan.narrow(1, ratio + 2, pools)?.contiguous()?, pools)?,
                seq_lens: last(context_lens, 1)?.flatten_all()?,
                n_kv: capacity,
                gather: Some(Gather {
                    table: table.clone(),
                    kept: plan.narrow(1, ratio + 1, 1)?.flatten_all()?,
                    compact_table: Tensor::from_vec(compact, (rows, pages), table.device())?,
                }),
                positions,
            })
        } else {
            None
        };
        Ok(Self {
            pooling: Some(pooling),
            selection,
        })
    }
}

// The gather's per-row inputs for `positions` (batch-major, `query_len` per sequence), whose `table` rows are the
// sequences' or, `per_token`, the query tokens' own.
fn gather_plan(
    cfg: &QsaConfig,
    table: &Tensor,
    block_size: usize,
    positions: &[u32],
    query_len: usize,
    per_token: bool,
) -> Result<Gather> {
    let ratio = cfg.compress_ratio;
    let k = cfg.budget / ratio;
    let device = table.device();
    // past the budget a row keeps exactly k blocks and its tail; before it, everything up to its position
    let kept = positions
        .iter()
        .map(|&p| {
            if p == QSA_PADDED_ROW {
                return 1;
            }
            let seen = p as usize + 1;
            (if seen / ratio <= k {
                seen
            } else {
                k * ratio + seen % ratio
            }) as u32
        })
        .collect::<Vec<_>>();
    let sequence_of = (0..positions.len())
        .map(|row| (if per_token { row } else { row / query_len }) as u32)
        .collect::<Vec<_>>();
    let pages = qsa_cells_width(k, ratio).div_ceil(block_size);
    let compact = (0..(positions.len() * pages) as u32).collect::<Vec<_>>();
    Ok(Gather {
        table: table
            .index_select(&Tensor::new(sequence_of.as_slice(), device)?, 0)?
            .contiguous()?,
        kept: Tensor::new(kept.as_slice(), device)?,
        compact_table: Tensor::from_vec(compact, (positions.len(), pages), device)?,
    })
}
