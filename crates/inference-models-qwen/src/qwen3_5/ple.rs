//! Qwen3.8-Flash-Next's PLE: hashed bigram and trigram embeddings gated into every hyper-connection stream, plus a
//! dilated depthwise conv whose last positions are each sequence's recurrent state (`Qwen4ExpTextPLELayer`).

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::config::{PleConfig, PleHash, TextConfig};
use super::hyper::GroupedRmsNorm;
use crate::kv_cache::RecurrentStatePool;
use crate::model::TokenWindow;
use crate::utils::unvarbuilder::UnVarBuilder;

const MASK64: u128 = (1 << 64) - 1;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
// each PLE layer offsets its seed by this prime
const LAYER_SEED_PRIME: u64 = 10_007;
// the reference keeps the gate's square root away from zero
const GATE_FLOOR: f64 = 1e-6;
// decode rows one host gather serves: every graph batch bucket
#[cfg(feature = "cuda")]
const PLE_HOST_ROWS: usize = inference_nn::cuda::graph_capture::CUDA_GRAPH_MAX_BATCH_BUCKET;

fn splitmix64(value: u64) -> u64 {
    let value = value.wrapping_add(SPLITMIX_GAMMA);
    let value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    let value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

impl PleHash {
    /// The reference's constants for the PLE at `ple_index` (its position in `ple_layer_ids`).
    pub fn derive(cfg: &TextConfig, ple: &PleConfig, ple_index: u64) -> Self {
        let fields = &cfg.flash_next;
        let multiplier_max = (i64::MAX as u64) / (cfg.vocab_size as u64).max(1);
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = fields.seed.wrapping_add(LAYER_SEED_PRIME * ple_index);
        let layer_multipliers = (0..ple.ngram_size as u64)
            .map(|index| {
                let value = (u128::from(base_seed)
                    + u128::from(SPLITMIX_GAMMA) * u128::from(index + 1))
                    & MASK64;
                #[allow(clippy::cast_possible_truncation)]
                let value = value as u64;
                2 * (splitmix64(value) % half_bound) + 1
            })
            .collect();
        let heads = ple.ngram_heads() as u64;
        let (mut head_offsets, mut head_vocab_sizes) = (Vec::new(), Vec::new());
        let (mut total, mut prime) = (0, fields.ngram_vocab_size_base - 1);
        // head h's table size is the (global h + 1)-th prime from the base on; heads run in order, so step on
        for _ in 0..ple_index * heads {
            prime += 1;
            while !is_prime(prime) {
                prime += 1;
            }
        }
        for _ in 0..heads {
            prime += 1;
            while !is_prime(prime) {
                prime += 1;
            }
            head_offsets.push(total);
            head_vocab_sizes.push(prime);
            total += prime;
        }
        Self {
            layer_multipliers,
            head_offsets,
            head_vocab_sizes,
        }
    }

    /// The table rows `total` rounds up to.
    pub fn padded_rows(&self, divisor: usize) -> usize {
        let total = self.head_vocab_sizes.iter().sum::<u64>() as usize;
        total.div_ceil(divisor) * divisor
    }
}

/// Every chunk position's `ngram_heads` table rows, `[rows of the batch * chunk len, heads]`; a padded position takes
/// row 0, which the forward masks.
fn ngram_rows(
    window: &TokenWindow,
    hash: &PleHash,
    ple: &PleConfig,
    chunk_len: usize,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let heads = ple.ngram_heads();
    let context = ple.ngram_size - 1;
    let mut ids = Vec::with_capacity(window.rows.len() * chunk_len * heads);
    let mut real_lens = Vec::with_capacity(window.rows.len());
    for row in &window.rows {
        // the history the reference sees: EOS before the sequence's start
        let missing = context.saturating_sub(row.preceding);
        let skipped = row.preceding.saturating_sub(context);
        let mut history = vec![ple.eos_token_id; missing];
        history.extend_from_slice(&row.tokens[skipped..]);
        let real_len = row.tokens.len() - row.preceding;
        if real_len > chunk_len {
            inference_tensor::bail!(
                "a PLE row has {real_len} tokens for a {chunk_len}-token chunk"
            );
        }
        real_lens.push(real_len);
        for position in 0..chunk_len {
            if position >= real_len {
                ids.extend(std::iter::repeat_n(0, heads));
                continue;
            }
            let t = context + position;
            // shift k reads t - k unless an EOS sits strictly between, which resets the n-gram to EOS
            let shifted = |k: usize| {
                if (t - k + 1..t).any(|p| history[p] == ple.eos_token_id) {
                    ple.eos_token_id
                } else {
                    history[t - k]
                }
            };
            for ngram in 2..=ple.ngram_size {
                let mut mixed = u64::from(shifted(0)) * hash.layer_multipliers[0];
                for k in 1..ngram {
                    mixed ^= u64::from(shifted(k)) * hash.layer_multipliers[k];
                }
                let first_head = (ngram - 2) * ple.heads_per_ngram;
                for head in first_head..first_head + ple.heads_per_ngram {
                    let row = mixed % hash.head_vocab_sizes[head] + hash.head_offsets[head];
                    ids.push(u32::try_from(row).map_err(inference_tensor::Error::wrap)?);
                }
            }
        }
    }
    Ok((ids, real_lens))
}

/// What hashing a window into table rows and gathering them needs, shared with the host gather.
struct NgramLookup {
    cfg: PleConfig,
    hash: PleHash,
    table: NgramTable,
}

impl NgramLookup {
    // `[rows * chunk_len * heads, head_dim]` f32 on the host, with each row's real length
    fn rows(&self, window: &TokenWindow, chunk_len: usize) -> Result<(Tensor, Vec<usize>)> {
        let (ids, real_lens) = ngram_rows(window, &self.hash, &self.cfg, chunk_len)?;
        Ok((self.table.gather(&ids, DType::F32)?, real_lens))
    }
}

/// The n-gram table, one tensor (GGUF) or the checkpoint's row shards, kept on the host.
struct NgramTable {
    // (first row, rows, the shard)
    shards: Vec<(usize, usize, Arc<dyn QuantMethod>)>,
    // a GGUF's table read in place from its mapping: tens of GB of which each token reads a few rows
    mapped: Option<inference_quant::MappedBlocks>,
}

impl NgramTable {
    fn load(cfg: &TextConfig, rows: usize, head_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let table = |vb: ShardedVarBuilder, rows: usize| {
            inference_quant::ReplicatedLayer::new(
                head_dim,
                rows,
                &cfg.quantization_config,
                false,
                vb,
            )
        };
        if let Some(blocks) = vb
            .weight_source()
            .and_then(|source| source.mapped_blocks(&vb.prefix()))
            .filter(|blocks| blocks.dims == [rows, head_dim])
        {
            return Ok(Self {
                shards: Vec::new(),
                mapped: Some(blocks),
            });
        }
        if vb.contains_tensor("weight") {
            return Ok(Self {
                shards: vec![(0, rows, table(vb, rows)?)],
                mapped: None,
            });
        }
        let parts = (0..)
            .take_while(|part| vb.contains_tensor(&format!("shard_{part}.weight")))
            .count();
        if parts == 0 {
            inference_tensor::bail!("PLE has neither an n-gram table nor its shards");
        }
        // the reference splits the table into equal row chunks, the last one shorter
        let chunk = rows.div_ceil(parts);
        let mut shards = Vec::with_capacity(parts);
        for part in 0..parts {
            let vb = vb.pp(format!("shard_{part}"));
            let shard_rows = vb
                .tensor_shape("weight")
                .map_or(chunk.min(rows - part * chunk), |shape| shape[0]);
            let first = shards.last().map_or(0, |(first, rows, _)| first + rows);
            shards.push((first, shard_rows, table(vb, shard_rows)?));
        }
        Ok(Self {
            shards,
            mapped: None,
        })
    }

    /// The rows `ids`, `[ids, head_dim]` at `dtype`, gathered on the host shard by shard.
    fn gather(&self, ids: &[u32], dtype: inference_tensor::DType) -> Result<Tensor> {
        if let Some(blocks) = &self.mapped {
            let cols = blocks.dims[1];
            return Tensor::from_vec(blocks.gather_rows(ids)?, (ids.len(), cols), &Device::Cpu)?
                .to_dtype(dtype);
        }
        if let [(_, _, table)] = self.shards.as_slice() {
            let ids = Tensor::from_vec(ids.to_vec(), (ids.len(),), &Device::Cpu)?;
            return table.embedding_forward(&ids, dtype);
        }
        let mut parts = Vec::with_capacity(self.shards.len());
        let mut order = vec![0u32; ids.len()];
        let mut next = 0u32;
        for (first, rows, table) in &self.shards {
            let (positions, local): (Vec<usize>, Vec<u32>) = ids
                .iter()
                .enumerate()
                .filter(|(_, id)| (*first..first + rows).contains(&(**id as usize)))
                .map(|(position, id)| (position, *id - *first as u32))
                .unzip();
            if local.is_empty() {
                continue;
            }
            for position in positions {
                order[position] = next;
                next += 1;
            }
            let local = Tensor::from_vec(local.clone(), (local.len(),), &Device::Cpu)?;
            parts.push(table.embedding_forward(&local, dtype)?);
        }
        if next as usize != ids.len() {
            inference_tensor::bail!("an n-gram id falls outside every PLE table shard");
        }
        let order = Tensor::from_vec(order, (ids.len(),), &Device::Cpu)?;
        Tensor::cat(&parts, 0)?.index_select(&order, 0)
    }
}

/// Where a forward's token window is: the host's, or a decode step's in `TokenWindow::decode_rows`' layout on the
/// device (a captured step's, refreshed before each replay).
#[derive(Clone, Copy)]
pub(super) enum PleTokens<'a> {
    Host(&'a TokenWindow),
    Device(&'a Tensor),
}

pub(super) struct PleLayer {
    cfg: PleConfig,
    // the n-gram table stays on the host; each forward gathers its rows there
    lookup: Arc<NgramLookup>,
    #[cfg(feature = "cuda")]
    host_gather: std::sync::OnceLock<host_gather::HostGather>,
    key_proj: Arc<dyn QuantMethod>,
    value_proj: Arc<dyn QuantMethod>,
    norm_key: GroupedRmsNorm,
    norm_query: GroupedRmsNorm,
    norm_conv: GroupedRmsNorm,
    // `[streams * hidden, kernel]`
    conv: Tensor,
    streams: usize,
    hidden: usize,
}

impl PleLayer {
    /// `vb_quant` and `vb_plain` are the layer's `ple` prefix on its device; `vb_table` the table's, on the host.
    pub(super) fn load(
        cfg: &TextConfig,
        ple: PleConfig,
        hash: PleHash,
        streams: usize,
        vb_quant: ShardedVarBuilder,
        vb_plain: ShardedVarBuilder,
        vb_table: ShardedVarBuilder,
    ) -> Result<Self> {
        let width = streams * cfg.hidden_size;
        let rows = hash.padded_rows(cfg.flash_next.make_ngram_vocab_size_divisible_by);
        let linear = |vb: ShardedVarBuilder, inputs: usize, outputs: usize| {
            inference_quant::ReplicatedLayer::new(
                inputs,
                outputs,
                &cfg.quantization_config,
                false,
                vb,
            )
        };
        let conv = vb_plain
            .pp("conv1d")
            .get((width, 1, ple.conv_kernel), "weight")?;
        let table = NgramTable::load(
            cfg,
            rows,
            ple.head_dim(),
            vb_table.pp("ple_embedding").pp("ngram_embedding"),
        )?;
        Ok(Self {
            lookup: Arc::new(NgramLookup {
                cfg: ple,
                hash,
                table,
            }),
            #[cfg(feature = "cuda")]
            host_gather: std::sync::OnceLock::new(),
            key_proj: linear(vb_quant.pp("key_proj"), ple.embed_dim, width)?,
            value_proj: linear(vb_quant.pp("value_proj"), ple.embed_dim, cfg.hidden_size)?,
            norm_key: GroupedRmsNorm::new(
                width,
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_plain.pp("norm_key"),
            )?,
            norm_query: GroupedRmsNorm::new(
                width,
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_plain.pp("norm_query"),
            )?,
            norm_conv: GroupedRmsNorm::new(
                width,
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_plain.pp("norm_conv"),
            )?,
            conv: conv.squeeze(1)?,
            cfg: ple,
            streams,
            hidden: cfg.hidden_size,
        })
    }

    /// The tokens before each chunk the n-gram hash reads.
    pub(super) fn preceding_tokens(&self) -> usize {
        self.cfg.ngram_size - 1
    }

    /// The conv state each sequence keeps: the last `conv_state_len` normed inputs of every channel.
    pub(super) fn state_config(&self) -> crate::kv_cache::RecurrentLayerConfig {
        crate::kv_cache::RecurrentLayerConfig {
            conv_dim: self.streams * self.hidden,
            conv_width: self.cfg.conv_state_len(),
            state: crate::kv_cache::RecurrentStateSpec::Opaque { dims: vec![1] },
            recurrent_dtype: None,
        }
    }

    /// What PLE adds to the streams `xs` `[b, len, streams * hidden]`, updating each sequence's conv state; with
    /// `keep_history`, also the conv history (`[b, state + len, width]`) a speculative commit narrows to the kept rows.
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn forward(
        &self,
        xs: &Tensor,
        tokens: PleTokens<'_>,
        pool: &mut RecurrentStatePool,
        indices: &Tensor,
        keep_history: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (batch, len, width) = xs.dims3()?;
        let heads = self.cfg.ngram_heads();
        let (embeddings, real_lens) = self.embeddings(tokens, batch, len, xs.device())?;
        let embeddings = embeddings
            .reshape((batch, len, heads * self.cfg.head_dim()))?
            .to_dtype(xs.dtype())?;

        let split = (batch, len, self.streams, self.hidden);
        let key = self
            .norm_key
            .forward(&self.key_proj.forward(&embeddings)?)?
            .reshape(split)?;
        let value = self.value_proj.forward(&embeddings)?;
        let query = self.norm_query.forward(xs)?.reshape(split)?;
        let gate = ((key * query)?.sum_keepdim(D::Minus1)? / (self.hidden as f64).sqrt())?;
        let gate = (gate.abs()?.clamp(GATE_FLOOR, f64::INFINITY)?.sqrt()? * gate.sign()?)?;
        let gated = inference_tensor::nn::ops::sigmoid(&gate)?
            .broadcast_mul(&value.unsqueeze(2)?)?
            .reshape((batch, len, width))?;

        // padded positions carry nothing into the streams or the conv
        let mask = Tensor::from_vec(
            real_lens
                .iter()
                .flat_map(|real| {
                    (0..len).map(move |position| f32::from(u8::from(position < *real)))
                })
                .collect::<Vec<_>>(),
            (batch, len, 1),
            xs.device(),
        )?
        .to_dtype(xs.dtype())?;
        let gated = gated.broadcast_mul(&mask)?;
        let normed = self.norm_conv.forward(&gated)?.broadcast_mul(&mask)?;

        let state_len = self.cfg.conv_state_len();
        let state = pool
            .gather_conv_state(indices)?
            .transpose(1, 2)?
            .to_dtype(xs.dtype())?;
        let history = Tensor::cat(&[&state, &normed], 1)?;
        let dilation = self.cfg.ngram_size;
        let mut conv = None;
        for tap in 0..self.cfg.conv_kernel {
            let term = history.narrow(1, tap * dilation, len)?.broadcast_mul(
                &self
                    .conv
                    .narrow(1, tap, 1)?
                    .squeeze(1)?
                    .to_dtype(xs.dtype())?,
            )?;
            conv = Some(match conv {
                Some(sum) => (sum + term)?,
                None => term,
            });
        }
        let conv = inference_tensor::nn::ops::silu(&conv.expect("PLE conv has a kernel"))?;

        let next_state = real_lens
            .iter()
            .enumerate()
            .map(|(row, real)| history.narrow(0, row, 1)?.narrow(1, *real, state_len))
            .collect::<Result<Vec<_>>>()?;
        let next_state = Tensor::cat(&next_state, 0)?
            .transpose(1, 2)?
            .to_dtype(pool.conv_dtype())?
            .contiguous()?;
        pool.scatter_conv_state(indices, &next_state)?;
        let out = (gated + conv)?.broadcast_mul(&mask)?;
        Ok((out, keep_history.then_some(history)))
    }

    // `[batch * len * heads, head_dim]` f32 on `device`, with each row's real length
    fn embeddings(
        &self,
        tokens: PleTokens<'_>,
        batch: usize,
        len: usize,
        device: &Device,
    ) -> Result<(Tensor, Vec<usize>)> {
        // a CUDA decode step gathers in a host function, so it never waits on the host and a graph can replay it
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device
            && len == 1
            && batch <= PLE_HOST_ROWS
        {
            let preceding = self.preceding_tokens();
            let window = match tokens {
                PleTokens::Device(window) => Some(window.clone()),
                PleTokens::Host(window) => window
                    .decode_rows(preceding)
                    .map(|rows| {
                        Tensor::from_vec(
                            rows,
                            (batch, TokenWindow::decode_width(preceding)),
                            device,
                        )
                    })
                    .transpose()?,
            };
            if let Some(window) = window {
                return Ok((self.host_gather(cuda)?.gather(&window)?, vec![1; batch]));
            }
        }
        let window = match tokens {
            PleTokens::Host(window) => window,
            PleTokens::Device(window) => inference_tensor::bail!(
                "PLE reads a device token window ({:?}) only on a CUDA decode step",
                window.shape()
            ),
        };
        if window.rows.len() != batch {
            inference_tensor::bail!(
                "PLE has {} token rows for a {batch}-row batch",
                window.rows.len()
            );
        }
        let (rows, real_lens) = self.lookup.rows(window, len)?;
        Ok((rows.to_device(device)?, real_lens))
    }

    #[cfg(feature = "cuda")]
    fn host_gather(
        &self,
        device: &inference_tensor::CudaDevice,
    ) -> Result<&host_gather::HostGather> {
        if let Some(gather) = self.host_gather.get() {
            return Ok(gather);
        }
        let gather = host_gather::HostGather::new(self.lookup.clone(), device)?;
        Ok(self.host_gather.get_or_init(|| gather))
    }

    /// Sets each listed row's conv state to the window ending at its kept rows of a stashed `history`.
    pub(super) fn commit_kept_rows(
        &self,
        history: &Tensor,
        indices: &Tensor,
        rows: &[(usize, usize)],
        pool: &mut RecurrentStatePool,
    ) -> Result<()> {
        let state_len = self.cfg.conv_state_len();
        for &(row, keep) in rows {
            let state = history
                .narrow(0, row, 1)?
                .narrow(1, keep, state_len)?
                .transpose(1, 2)?
                .to_dtype(pool.conv_dtype())?
                .contiguous()?;
            pool.scatter_conv_state(&indices.narrow(0, row, 1)?, &state)?;
        }
        Ok(())
    }

    pub(super) fn residual_tensors(&self, uvb: &UnVarBuilder) {
        self.norm_key.residual_tensors(&uvb.pp("norm_key"));
        self.norm_query.residual_tensors(&uvb.pp("norm_query"));
        self.norm_conv.residual_tensors(&uvb.pp("norm_conv"));
        uvb.pp("conv1d").add_tensor(
            "weight",
            self.conv.unsqueeze(1).expect("conv weight is 2-D"),
        );
    }
}

#[cfg(feature = "cuda")]
mod host_gather {
    use std::sync::Arc;

    use inference_nn::cuda::mapped::{mapped_buffer, mapped_host, tensor_ptr};
    use inference_tensor::cuda::cudarc::driver::{CudaSlice, CudaStream, result};
    use inference_tensor::{CudaDevice, DType, Device, Result, Tensor};

    use super::{NgramLookup, PLE_HOST_ROWS, TokenWindow};

    /// What the host function reads and writes: the lookup and the mapped window and rows.
    struct Shared {
        lookup: Arc<NgramLookup>,
        window: CudaSlice<u32>,
        rows: CudaSlice<f32>,
    }

    impl Shared {
        fn width(&self) -> usize {
            TokenWindow::decode_width(self.lookup.cfg.ngram_size - 1)
        }

        fn row_len(&self) -> usize {
            self.lookup.cfg.ngram_heads() * self.lookup.cfg.head_dim()
        }

        fn gather(&self, batch: usize) -> Result<()> {
            let preceding = self.lookup.cfg.ngram_size - 1;
            let window = TokenWindow::from_decode_rows(
                mapped_host(&self.window, batch * self.width()),
                preceding,
            );
            let (rows, _) = self.lookup.rows(&window, 1)?;
            let rows = rows.flatten_all()?.to_vec1::<f32>()?;
            mapped_host(&self.rows, rows.len()).copy_from_slice(&rows);
            Ok(())
        }
    }

    /// One batch size's host function argument; the layer owns it, so a pointer to one stays valid.
    struct Call {
        shared: Arc<Shared>,
        batch: usize,
    }

    unsafe extern "C" fn gather_host(data: *mut std::ffi::c_void) {
        // SAFETY: `data` is a `Call` the layer owns for its whole life
        let call = unsafe { &*data.cast::<Call>() };
        if let Err(err) = call.shared.gather(call.batch) {
            // the step then adds nothing rather than stale rows
            tracing::error!("PLE failed to gather its n-gram rows on the host: {err}");
            mapped_host(&call.shared.rows, call.batch * call.shared.row_len()).fill(0.0);
        }
    }

    /// A decode step's rows gathered on the host without the stream waiting: the window goes out to mapped memory,
    /// a host function on a side stream hashes and gathers, and the rows come back, joined before the stream moves on.
    pub(super) struct HostGather {
        shared: Arc<Shared>,
        calls: Vec<Call>,
        side: Arc<CudaStream>,
        device: CudaDevice,
    }

    impl HostGather {
        pub(super) fn new(lookup: Arc<NgramLookup>, device: &CudaDevice) -> Result<Self> {
            let stream = device.cuda_stream();
            let width = TokenWindow::decode_width(lookup.cfg.ngram_size - 1);
            let row_len = lookup.cfg.ngram_heads() * lookup.cfg.head_dim();
            let shared = Arc::new(Shared {
                lookup,
                window: mapped_buffer(&stream, PLE_HOST_ROWS * width)?,
                rows: mapped_buffer(&stream, PLE_HOST_ROWS * row_len)?,
            });
            let calls = (1..=PLE_HOST_ROWS)
                .map(|batch| Call {
                    shared: shared.clone(),
                    batch,
                })
                .collect();
            Ok(Self {
                shared,
                calls,
                side: stream
                    .context()
                    .new_stream()
                    .map_err(inference_tensor::Error::wrap)?,
                device: device.clone(),
            })
        }

        /// The rows of each window row, `[batch * heads, head_dim]` f32.
        pub(super) fn gather(&self, window: &Tensor) -> Result<Tensor> {
            let (batch, width) = window.dims2()?;
            if batch == 0 || batch > PLE_HOST_ROWS || width != self.shared.width() {
                inference_tensor::bail!(
                    "a PLE host gather serves up to {PLE_HOST_ROWS} rows of {} tokens, got {:?}",
                    self.shared.width(),
                    window.shape()
                );
            }
            let cfg = &self.shared.lookup.cfg;
            let window = window.to_dtype(DType::U32)?.contiguous()?;
            let out = Tensor::zeros(
                (batch * cfg.ngram_heads(), cfg.head_dim()),
                DType::F32,
                &Device::Cuda(self.device.clone()),
            )?;
            let main = self.device.cuda_stream();
            let side = &self.side;
            let wrap = inference_tensor::Error::wrap;
            // SAFETY: the mapped buffers hold the largest batch; the window's copy out precedes the event the side
            // stream waits on, the host function reads it after that, and the stream waits for the side stream's copy
            // of the rows before it moves on, so the next step cannot overwrite a buffer still in use
            unsafe {
                result::memcpy_dtoh_async(
                    mapped_host(&self.shared.window, batch * width),
                    tensor_ptr::<u32>(&window, &main)?,
                    main.cu_stream(),
                )
                .map_err(wrap)?;
                side.wait(&main.record_event(None).map_err(wrap)?)
                    .map_err(wrap)?;
                let call = &self.calls[batch - 1] as *const Call as *mut std::ffi::c_void;
                result::stream::launch_host_function(side.cu_stream(), gather_host, call)
                    .map_err(wrap)?;
                result::memcpy_htod_async(
                    tensor_ptr::<f32>(&out, side)?,
                    mapped_host(&self.shared.rows, batch * self.shared.row_len()),
                    side.cu_stream(),
                )
                .map_err(wrap)?;
            }
            main.wait(&side.record_event(None).map_err(wrap)?)
                .map_err(wrap)?;
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PleHash, splitmix64};

    #[test]
    fn splitmix_matches_its_published_first_output() {
        // splitmix64 seeded with 0 yields 0xE220A8397B1DCDAF first (Vigna's reference)
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn head_tables_are_consecutive_primes_from_the_base() {
        let hash = PleHash {
            layer_multipliers: Vec::new(),
            head_offsets: vec![0, 20_000_003],
            head_vocab_sizes: vec![20_000_003, 20_000_023],
        };
        assert_eq!(
            hash.padded_rows(128),
            (20_000_003 + 20_000_023usize).div_ceil(128) * 128
        );
        assert!(super::is_prime(20_000_003) && super::is_prime(20_000_023));
        assert!((20_000_004..20_000_023).all(|v| !super::is_prime(v)));
    }
}
