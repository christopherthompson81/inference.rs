use std::{
    collections::HashMap,
    fs,
    io::Cursor,
    path::Path,
    sync::{Arc, RwLock},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use inference_tensor::{Context, D, DType, Device, Result, Tensor};
use serde::Serialize;

// routed rows buffered before their per-expert Grams run: one host sync and one gather per flush, not per forward
const ROUTED_FLUSH_ROWS: usize = 1 << 17;

#[derive(Debug)]
enum ImatrixLayerStats_ {
    Dense {
        row_counts: usize,
        ncalls: usize,
        row_accum: Tensor,
    },
    /// Routed (MoE): `accum [E, in]`, `counts [E]`; `rows` mirrors counts' total on the CPU so
    /// snapshots never sync the device.
    Routed {
        ncalls: usize,
        rows: usize,
        counts: Tensor,
        accum: Tensor,
    },
    /// The input's second moment `x^T x` `[in, in]` over every row, as EXL3's LDLQ weighs rounding errors.
    Hessian {
        ncalls: usize,
        rows: usize,
        h: Tensor,
    },
    /// Per expert of a routed (stacked) layer, the second moment over the rows routed to it, with their counts;
    /// an expert's is allocated when its first row arrives.
    RoutedHessian {
        ncalls: usize,
        rows: usize,
        in_dim: usize,
        counts: Vec<usize>,
        h: Vec<Option<Tensor>>,
        // forwards not yet added: each `(x, ids, a row per slot)`, and their routed rows
        pending: Vec<(Tensor, Tensor, bool)>,
        pending_rows: usize,
    },
    /// Every forward's input as it came (rows of `in`), for a caller pairing them with another model's.
    Capture {
        ncalls: usize,
        rows: usize,
        inputs: Vec<Tensor>,
    },
}

// `h` (zero when None) plus x^T x for `x` (rows, in), in f32
fn add_gram(h: Option<Tensor>, x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    let h = {
        let in_dim = x.dim(1)?;
        let h = match h {
            Some(h) => h,
            None => Tensor::zeros((in_dim, in_dim), DType::F32, x.device())?,
        };
        if add_gram_half(&h, x)? {
            return Ok(h);
        }
        Some(h)
    };
    let x = x.to_dtype(DType::F32)?;
    let add = x.t()?.matmul(&x)?;
    match h {
        Some(h) => h + add,
        None => Ok(add),
    }
}

// `h += x^T x` for `x` (rows, in) in bf16 or f16 on CUDA, one GEMM accumulating in f32 over the half inputs:
// their products are exact in f32, so this is the f32 product but for the order of its sums, at the tensor cores'
// rate and without a temporary. False where `x` is another dtype or device
#[cfg(feature = "cuda")]
fn add_gram_half(h: &Tensor, x: &Tensor) -> Result<bool> {
    use inference_tensor::Storage;
    use inference_tensor::cuda_backend::cudarc::cublas::{result, sys};

    let half = match x.dtype() {
        DType::BF16 => sys::cudaDataType_t::CUDA_R_16BF,
        DType::F16 => sys::cudaDataType_t::CUDA_R_16F,
        _ => return Ok(false),
    };
    let (Device::Cuda(dev), true) = (x.device(), h.dtype() == DType::F32 && h.is_contiguous())
    else {
        return Ok(false);
    };
    let x = x.contiguous()?;
    let (rows, k) = x.dims2()?;
    let (h_s, h_l) = h.storage_and_layout();
    let (x_s, x_l) = x.storage_and_layout();
    let (Storage::Cuda(h_c), Storage::Cuda(x_c)) = (&*h_s, &*x_s) else {
        return Ok(false);
    };
    let (h_ptr, _h) = crate::utils::slice_ptr(h_c.as_cuda_slice::<f32>()?, h_l.start_offset());
    let (x_ptr, _x) = match x.dtype() {
        DType::BF16 => {
            crate::utils::slice_ptr(x_c.as_cuda_slice::<half::bf16>()?, x_l.start_offset())
        }
        _ => crate::utils::slice_ptr(x_c.as_cuda_slice::<half::f16>()?, x_l.start_offset()),
    };
    let (alpha, beta) = (1.0f32, 1.0f32);
    let (k, rows) = (k as i32, rows as i32);
    // column-major, row-major x is x^T (k, rows): x^T (x^T)^T is x^T x, symmetric, so h's layout is either
    // SAFETY: both pointers span their tensors, `h` is f32 (k, k) and written in place on the blas handle's stream
    unsafe {
        result::gemm_ex(
            *dev.cublas_handle().handle(),
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_T,
            k,
            k,
            rows,
            (&raw const alpha).cast(),
            x_ptr as *const _,
            half,
            k,
            x_ptr as *const _,
            half,
            k,
            (&raw const beta).cast(),
            h_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            k,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .map_err(|e| inference_tensor::Error::Cuda(Box::new(e)))?;
    Ok(true)
}

#[derive(Debug, Clone)]
pub struct ImatrixLayerStats(Arc<RwLock<Option<ImatrixLayerStats_>>>);

impl ImatrixLayerStats {
    pub fn empty() -> Self {
        Self(Arc::new(RwLock::new(None)))
    }

    /// Start collecting; safe on shared layers since the state is interior-mutable.
    pub fn enable(&self, in_dim: usize, device: &Device) -> Result<()> {
        *self.0.write().unwrap() = Some(ImatrixLayerStats_::Dense {
            row_counts: 0,
            ncalls: 0,
            row_accum: Tensor::zeros((in_dim,), DType::F32, device)?,
        });
        Ok(())
    }

    /// Start collecting per-expert statistics for a routed (stacked `[E, out, in]`) layer.
    pub fn enable_routed(&self, num_experts: usize, in_dim: usize, device: &Device) -> Result<()> {
        *self.0.write().unwrap() = Some(ImatrixLayerStats_::Routed {
            ncalls: 0,
            rows: 0,
            counts: Tensor::zeros((num_experts,), DType::F32, device)?,
            accum: Tensor::zeros((num_experts, in_dim), DType::F32, device)?,
        });
        Ok(())
    }

    /// Start collecting the full input Hessian `[in, in]` (f32) of a dense layer.
    pub fn enable_hessian(&self, in_dim: usize, device: &Device) -> Result<()> {
        *self.0.write().unwrap() = Some(ImatrixLayerStats_::Hessian {
            ncalls: 0,
            rows: 0,
            h: Tensor::zeros((in_dim, in_dim), DType::F32, device)?,
        });
        Ok(())
    }

    /// Start collecting each expert's input Hessian `[in, in]` (f32) for a routed (stacked `[E, out, in]`) layer.
    pub fn enable_routed_hessian(&self, num_experts: usize, in_dim: usize) -> Result<()> {
        *self.0.write().unwrap() = Some(ImatrixLayerStats_::RoutedHessian {
            ncalls: 0,
            rows: 0,
            in_dim,
            counts: vec![0; num_experts],
            h: vec![None; num_experts],
            pending: Vec::new(),
            pending_rows: 0,
        });
        Ok(())
    }

    /// Start keeping every forward's input, for [`Self::take_captured`].
    pub fn enable_capture(&self) {
        *self.0.write().unwrap() = Some(ImatrixLayerStats_::Capture {
            ncalls: 0,
            rows: 0,
            inputs: Vec::new(),
        });
    }

    /// The inputs kept since [`Self::enable_capture`] or the last take, in order; capturing goes on.
    pub fn take_captured(&self) -> Result<Vec<Tensor>> {
        match self.0.write().unwrap().as_mut() {
            Some(ImatrixLayerStats_::Capture { inputs, .. }) => Ok(std::mem::take(inputs)),
            _ => inference_tensor::bail!("this layer is not capturing its inputs"),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.0.read().unwrap().is_some()
    }

    /// (forward calls, token rows) so far; routed rows are token x top-k slots.
    pub fn snapshot(&self) -> Option<(usize, usize)> {
        self.0.read().unwrap().as_ref().map(|s| match s {
            ImatrixLayerStats_::Dense {
                ncalls, row_counts, ..
            } => (*ncalls, *row_counts),
            ImatrixLayerStats_::Routed { ncalls, rows, .. }
            | ImatrixLayerStats_::Hessian { ncalls, rows, .. }
            | ImatrixLayerStats_::RoutedHessian { ncalls, rows, .. }
            | ImatrixLayerStats_::Capture { ncalls, rows, .. } => (*ncalls, *rows),
        })
    }

    pub fn process(&self, inp: &Tensor) -> Result<()> {
        if self.0.read().unwrap().is_none() {
            return Ok(());
        }
        let mut handle = self.0.write().unwrap();
        if let Some(ImatrixLayerStats_::Hessian { ncalls, rows, h }) = handle.as_mut() {
            let x = inp.reshape(((), inp.dim(D::Minus1)?))?;
            *ncalls += 1;
            *rows += x.dim(0)?;
            let _span = inference_tensor::profile::span("hessian.dense", Some(x.device()));
            *h = add_gram(Some(h.clone()), &x)?;
            return Ok(());
        }
        if let Some(ImatrixLayerStats_::Capture {
            ncalls,
            rows,
            inputs,
        }) = handle.as_mut()
        {
            let x = inp.reshape(((), inp.dim(D::Minus1)?))?;
            *ncalls += 1;
            *rows += x.dim(0)?;
            inputs.push(x);
            return Ok(());
        }
        // routed layers are observed by their router; a stray plain forward contributes nothing
        let Some(ImatrixLayerStats_::Dense {
            row_counts,
            ncalls,
            row_accum,
        }) = handle.as_mut()
        else {
            return Ok(());
        };

        let inp = inp.reshape(((), inp.dim(D::Minus1)?))?;
        *ncalls += 1;
        // Counts are token rows, yielding mean square per input column.
        *row_counts += inp.dim(0)?;
        *row_accum = (&*row_accum + inp.to_dtype(DType::F32)?.sqr()?.sum(0)?)?;
        Ok(())
    }

    /// Accumulate routed activations: `ids` is `(n, k)`; `x` is `(n, in)` (row scatters into all
    /// its experts) or `(n, k, in)` (slot `(t, s)` scatters into `ids[t, s]` only).
    pub fn process_routed(&self, x: &Tensor, ids: &Tensor) -> Result<()> {
        if self.0.read().unwrap().is_none() {
            return Ok(());
        }
        let mut handle = self.0.write().unwrap();
        if let Some(ImatrixLayerStats_::RoutedHessian {
            ncalls,
            rows,
            in_dim,
            counts,
            h,
            pending,
            pending_rows,
        }) = handle.as_mut()
        {
            let (n, k) = ids.dims2()?;
            pending.push((x.reshape(((), *in_dim))?, ids.clone(), x.rank() == 3));
            *pending_rows += n * k;
            *rows += n * k;
            *ncalls += 1;
            if *pending_rows >= ROUTED_FLUSH_ROWS {
                flush_routed(pending, counts, h)?;
                *pending_rows = 0;
            }
            return Ok(());
        }
        let Some(ImatrixLayerStats_::Routed {
            ncalls,
            rows,
            counts,
            accum,
        }) = handle.as_mut()
        else {
            return Ok(());
        };

        let (n, k) = ids.dims2()?;
        let in_dim = x.dim(D::Minus1)?;
        let x2 = match x.dims().len() {
            2 => x
                .to_dtype(DType::F32)?
                .sqr()?
                .unsqueeze(1)?
                .broadcast_as((n, k, in_dim))?
                .force_contiguous()?
                .reshape((n * k, in_dim))?,
            3 => x.to_dtype(DType::F32)?.sqr()?.reshape((n * k, in_dim))?,
            other => {
                inference_tensor::bail!("process_routed expects rank 2 or 3 input, got {other}")
            }
        };
        let ids_flat = ids.flatten_all()?.to_dtype(DType::U32)?;
        *accum = accum.index_add(&ids_flat, &x2, 0)?;
        let ones = Tensor::ones((n * k,), DType::F32, counts.device())?;
        *counts = counts.index_add(&ids_flat, &ones, 0)?;
        *rows += n * k;
        *ncalls += 1;
        Ok(())
    }

    /// Accumulate routed activations already grouped by expert: `counts[e]` consecutive rows of `x` (`(n, in)`) for
    /// expert `e`, as a grouped GEMM orders its rows.
    pub fn process_sorted(&self, x: &Tensor, counts: &[usize]) -> Result<()> {
        if self.0.read().unwrap().is_none() {
            return Ok(());
        }
        let mut handle = self.0.write().unwrap();
        let n = x.dim(0)?;
        if counts.iter().sum::<usize>() != n {
            inference_tensor::bail!(
                "{n} rows grouped as {} by expert",
                counts.iter().sum::<usize>()
            );
        }
        match handle.as_mut() {
            Some(ImatrixLayerStats_::RoutedHessian {
                ncalls,
                rows,
                counts: seen,
                h,
                pending,
                pending_rows,
                ..
            }) => {
                // queued as rows a slot each, with their experts, for the next flush's one gather
                let ids = counts
                    .iter()
                    .enumerate()
                    .flat_map(|(e, &c)| std::iter::repeat_n(e as u32, c))
                    .collect::<Vec<_>>();
                let ids = Tensor::from_vec(ids, (n, 1), x.device())?;
                pending.push((x.clone(), ids, true));
                *pending_rows += n;
                *rows += n;
                *ncalls += 1;
                if *pending_rows >= ROUTED_FLUSH_ROWS {
                    flush_routed(pending, seen, h)?;
                    *pending_rows = 0;
                }
            }
            Some(ImatrixLayerStats_::Routed {
                ncalls,
                rows,
                counts: seen,
                accum,
            }) => {
                let ids = counts
                    .iter()
                    .enumerate()
                    .flat_map(|(e, &c)| std::iter::repeat_n(e as u32, c))
                    .collect::<Vec<_>>();
                let ids = Tensor::from_vec(ids, n, x.device())?;
                *accum = accum.index_add(&ids, &x.to_dtype(DType::F32)?.sqr()?, 0)?;
                let ones = Tensor::ones(n, DType::F32, seen.device())?;
                *seen = seen.index_add(&ids, &ones, 0)?;
                *rows += n;
                *ncalls += 1;
            }
            _ => {}
        }
        Ok(())
    }

    /// Dense: `[in]`. Routed: `[E, in]` with all-zero rows for zero-traffic experts. Hessian: `x^T x` over the rows
    /// recorded, divided by their count, `[in, in]`; routed, one per expert `[E, in, in]` on the host (unit for an expert
    /// no row reached).
    pub fn compute_imatrix(&self) -> Result<Tensor> {
        let mut handle = self.0.write().unwrap();
        if let Some(ImatrixLayerStats_::RoutedHessian {
            counts,
            h,
            pending,
            pending_rows,
            ..
        }) = handle.as_mut()
        {
            flush_routed(pending, counts, h)?;
            *pending_rows = 0;
        }
        match handle.as_ref().context("Layer stats were deinitialized!")? {
            ImatrixLayerStats_::Dense {
                row_counts,
                ncalls,
                row_accum,
            } => {
                if *row_counts == 0 {
                    inference_tensor::bail!("No activations were recorded for this layer.");
                }
                (row_accum / *row_counts as f64)? * *ncalls as f64
            }
            ImatrixLayerStats_::Routed {
                ncalls,
                counts,
                accum,
                ..
            } => {
                let total = counts.sum_all()?.to_scalar::<f32>()?;
                if total == 0.0 {
                    inference_tensor::bail!("No activations were recorded for this layer.");
                }
                // Per-expert mean square; zero-traffic experts divide to zero, not NaN.
                let safe_counts = counts.maximum(1.0)?.unsqueeze(1)?;
                accum.broadcast_div(&safe_counts)? * *ncalls as f64
            }
            ImatrixLayerStats_::Hessian { rows, h, .. } => {
                if *rows == 0 {
                    inference_tensor::bail!("No activations were recorded for this layer.");
                }
                h / *rows as f64
            }
            ImatrixLayerStats_::RoutedHessian {
                rows,
                in_dim,
                counts,
                h,
                ..
            } => {
                if *rows == 0 {
                    inference_tensor::bail!("No activations were recorded for this layer.");
                }
                // stacked on the host: one device allocation of every expert's Hessian would be the layer's largest
                // an expert no row reached is encoded uncalibrated (exllamav3's fallback): a unit Hessian
                let per_expert = h
                    .iter()
                    .zip(counts)
                    .map(|(he, &count)| match he {
                        Some(he) => (he / count as f64)?.to_device(&Device::Cpu),
                        None => Tensor::eye(*in_dim, DType::F32, &Device::Cpu),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::stack(&per_expert, 0)
            }
            ImatrixLayerStats_::Capture { .. } => {
                inference_tensor::bail!("a capturing layer's inputs are taken, not summarized")
            }
        }
    }

    pub fn clear(&self) -> Result<()> {
        let mut handle = self.0.write().unwrap();
        *handle = None;
        Ok(())
    }
}

// the buffered forwards' rows added to their experts' Grams: the ids read back once, every routed row gathered in
// expert order in one select, each expert's Gram over its run of it
fn flush_routed(
    pending: &mut Vec<(Tensor, Tensor, bool)>,
    counts: &mut [usize],
    h: &mut [Option<Tensor>],
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let _span = inference_tensor::profile::span("hessian.routed", Some(pending[0].0.device()));
    let mut slots = vec![Vec::new(); counts.len()];
    let mut base = 0u32;
    let xs = pending.iter().map(|(x, ..)| x.clone()).collect::<Vec<_>>();
    let ids = Tensor::cat(
        &pending
            .iter()
            .map(|(_, ids, _)| ids.to_dtype(DType::U32)?.flatten_all())
            .collect::<Result<Vec<_>>>()?,
        0,
    )?
    .to_vec1::<u32>()?;
    let mut at = 0;
    for (x, forward_ids, per_slot) in pending.iter() {
        let (n, k) = forward_ids.dims2()?;
        // a row per token feeds all its experts; a row per slot only its own
        for (j, &e) in ids[at..at + n * k].iter().enumerate() {
            let row = if *per_slot { j } else { j / k };
            slots[e as usize].push(base + row as u32);
        }
        at += n * k;
        base += x.dim(0)? as u32;
    }
    let x = Tensor::cat(&xs, 0)?;
    let order = slots.iter().flatten().copied().collect::<Vec<_>>();
    let sorted = x.index_select(
        &Tensor::from_vec(order.clone(), order.len(), x.device())?,
        0,
    )?;
    let mut start = 0;
    for (e, rows_e) in slots.iter().enumerate().filter(|(_, r)| !r.is_empty()) {
        let xe = sorted.narrow(0, start, rows_e.len())?;
        h[e] = Some(add_gram(h[e].take(), &xe)?);
        counts[e] += rows_e.len();
        start += rows_e.len();
    }
    pending.clear();
    Ok(())
}

/// Collected imatrix data keyed by layer tracking key (`.cimatrix` format).
#[derive(Serialize, inference_json::FromJson)]
pub struct CollectedImatrixData(pub HashMap<String, Vec<f32>>);

impl CollectedImatrixData {
    pub fn save_imatrix<P: AsRef<Path>>(&self, fname: P) -> Result<()> {
        if let Some(ext) = fname.as_ref().extension()
            && ext != "cimatrix"
        {
            inference_tensor::bail!(
                "Expected a .cimatrix file to save collected imatrix data to, got {:?}",
                ext
            );
        }
        let mut buf: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&mut buf);

        cursor.write_u64::<LittleEndian>(self.0.len() as u64)?;
        for (key, data) in &self.0 {
            cursor.write_u64::<LittleEndian>(key.len() as u64)?;
            std::io::Write::write_all(&mut cursor, key.as_bytes())?;
            cursor.write_u64::<LittleEndian>(data.len() as u64)?;
            for x in data {
                cursor.write_f32::<LittleEndian>(*x)?;
            }
        }

        fs::write(fname, buf)?;
        Ok(())
    }

    pub fn load_imatrix<P: AsRef<Path>>(fname: P) -> Result<Self> {
        let buf = fs::read(fname)?;
        let mut cursor = Cursor::new(buf);

        let mut entries = HashMap::new();
        let num_entries = cursor.read_u64::<LittleEndian>()?;
        for _ in 0..num_entries {
            let key_len = cursor.read_u64::<LittleEndian>()? as usize;
            let mut key = vec![0u8; key_len];
            std::io::Read::read_exact(&mut cursor, &mut key)?;
            let key = String::from_utf8(key)
                .map_err(|_| inference_tensor::Error::Msg("Invalid cimatrix key".to_string()))?;
            let len_data = cursor.read_u64::<LittleEndian>()? as usize;
            let mut data = Vec::with_capacity(len_data);
            for _ in 0..len_data {
                data.push(cursor.read_f32::<LittleEndian>()?);
            }
            entries.insert(key, data);
        }

        Ok(Self(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantMethod, QuantMethodConfig};
    use inference_tensor::quantized::{GgmlDType, QTensor};

    fn manual_imatrix(rows: &[Vec<f32>], ncalls: usize) -> Vec<f32> {
        let in_dim = rows[0].len();
        let mut sums = vec![0f32; in_dim];
        for row in rows {
            for (s, x) in sums.iter_mut().zip(row) {
                *s += x * x;
            }
        }
        sums.iter()
            .map(|s| s / rows.len() as f32 * ncalls as f32)
            .collect()
    }

    #[test]
    fn gguf_layer_collects_imatrix() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &device)?;
        let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let layer = crate::GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: std::sync::Arc::new(q),
            b: None,
        })?;

        layer.begin_track_stats()?;
        let x1 = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
        let x2 = Tensor::randn(0f32, 1f32, (3, 32), &device)?;
        layer.forward_raw(&x1)?;
        layer.forward_raw(&x2)?;
        let imatrix = layer.end_track_stats()?.to_vec1::<f32>()?;

        let mut rows: Vec<Vec<f32>> = x1.to_vec2()?;
        rows.extend(x2.to_vec2::<f32>()?);
        let expected = manual_imatrix(&rows, 2);
        for (a, b) in imatrix.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-4, "got {a}, expected {b}");
        }
        Ok(())
    }

    // bf16 rows on CUDA accumulate by the half-precision GEMM to what their f32 values' products sum to
    #[test]
    fn bf16_rows_accumulate_as_their_f32_values_do() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (k, rows) = (384, 1000);
        let stats = ImatrixLayerStats::empty();
        stats.enable_hessian(k, &device)?;
        let mut reference = Tensor::zeros((k, k), DType::F32, &device)?;
        for _ in 0..3 {
            let x = Tensor::randn(0f32, 1f32, (rows, k), &device)?.to_dtype(DType::BF16)?;
            stats.process(&x)?;
            let x = x.to_dtype(DType::F32)?;
            reference = (reference + x.t()?.matmul(&x)?)?;
        }
        let h = stats.compute_imatrix()?;
        let reference = (reference / (3 * rows) as f64)?;
        let off = (h - &reference)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = reference.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(off < 1e-5 * scale, "{off} off at a scale of {scale}");
        Ok(())
    }

    // the Hessian is x^T x over every row across calls, over the row count; its diagonal the mean square
    #[test]
    fn hessian_is_the_mean_outer_product() -> Result<()> {
        let device = Device::Cpu;
        let stats = ImatrixLayerStats::empty();
        stats.enable_hessian(3, &device)?;
        let x1 = Tensor::new(&[[1f32, 2., 0.], [0., 1., 3.]], &device)?;
        let x2 = Tensor::new(&[[[2f32, 0., 1.]]], &device)?;
        stats.process(&x1)?;
        stats.process(&x2)?;
        assert_eq!(stats.snapshot(), Some((2, 3)));
        let h = stats.compute_imatrix()?.to_vec2::<f32>()?;
        let rows = [[1f32, 2., 0.], [0., 1., 3.], [2., 0., 1.]];
        for (i, row) in h.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let expected = rows.iter().map(|r| r[i] * r[j]).sum::<f32>() / 3.0;
                assert!(
                    (v - expected).abs() < 1e-6,
                    "H[{i}][{j}] {v} against {expected}"
                );
            }
        }
        Ok(())
    }

    // each expert's Hessian over the rows routed to it, over their count: a token row feeds every expert it is
    // routed to, a slot row only its own; an expert no row reached gets a unit Hessian
    #[test]
    fn routed_hessians_are_per_expert() -> Result<()> {
        let device = Device::Cpu;
        let stats = ImatrixLayerStats::empty();
        stats.enable_routed_hessian(3, 2)?;
        // token 0 -> experts {0, 1}, token 1 -> {1, 1}
        let x = Tensor::new(&[[1f32, 2.], [3., 0.]], &device)?;
        let ids = Tensor::new(&[[0u32, 1], [1, 1]], &device)?;
        stats.process_routed(&x, &ids)?;
        let h = stats.compute_imatrix()?.to_vec3::<f32>()?;
        assert_eq!(h[0], [[1., 2.], [2., 4.]]);
        // expert 1: token 0 once, token 1 twice, over 3 rows
        assert_eq!(h[1], [[(1. + 18.) / 3., 2. / 3.], [2. / 3., 4. / 3.]]);
        assert_eq!(h[2], [[1., 0.], [0., 1.]]);

        let stats = ImatrixLayerStats::empty();
        stats.enable_routed_hessian(2, 2)?;
        let slots = Tensor::new(&[[[1f32, 0.], [0., 2.]]], &device)?;
        stats.process_routed(&slots, &Tensor::new(&[[0u32, 1]], &device)?)?;
        let h = stats.compute_imatrix()?.to_vec3::<f32>()?;
        assert_eq!(h[0], [[1., 0.], [0., 0.]]);
        assert_eq!(h[1], [[0., 0.], [0., 4.]]);
        Ok(())
    }

    #[test]
    fn snapshot_visible_through_layer() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &device)?;
        let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let layer = crate::GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: std::sync::Arc::new(q),
            b: None,
        })?;
        assert!(layer.stats_snapshot().is_none());
        layer.begin_track_stats()?;
        assert_eq!(layer.stats_snapshot(), Some((0, 0)));
        let x = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
        layer.forward_raw(&x)?;
        assert_eq!(layer.stats_snapshot(), Some((1, 4)));
        Ok(())
    }

    #[test]
    fn routed_stats_scatter_per_expert() -> Result<()> {
        let device = Device::Cpu;
        let stats = ImatrixLayerStats::empty();
        stats.enable_routed(3, 2, &device)?;

        // two tokens, k=2: token0 -> experts {0, 2}, token1 -> experts {2, 1}
        let x = Tensor::new(&[[1f32, 2.], [3., 4.]], &device)?;
        let ids = Tensor::new(&[[0u32, 2], [2, 1]], &device)?;
        stats.process_routed(&x, &ids)?;

        assert_eq!(stats.snapshot(), Some((1, 4)));
        let m = stats.compute_imatrix()?.to_vec2::<f32>()?;
        // expert 0: token0 only; expert 1: token1 only; expert 2: mean of both
        assert_eq!(m[0], [1., 4.]);
        assert_eq!(m[1], [9., 16.]);
        assert_eq!(m[2], [(1. + 9.) / 2., (4. + 16.) / 2.]);
        Ok(())
    }

    #[test]
    fn routed_stats_rank3_per_slot() -> Result<()> {
        let device = Device::Cpu;
        let stats = ImatrixLayerStats::empty();
        stats.enable_routed(2, 2, &device)?;

        // per-slot inputs: slot (0,0)->e0 with [1,0]; slot (0,1)->e1 with [0,2]
        let x = Tensor::new(&[[[1f32, 0.], [0., 2.]]], &device)?;
        let ids = Tensor::new(&[[0u32, 1]], &device)?;
        stats.process_routed(&x, &ids)?;

        let m = stats.compute_imatrix()?.to_vec2::<f32>()?;
        assert_eq!(m[0], [1., 0.]);
        assert_eq!(m[1], [0., 4.]);
        Ok(())
    }

    #[test]
    fn dead_expert_rows_are_zero_not_nan() -> Result<()> {
        let device = Device::Cpu;
        let stats = ImatrixLayerStats::empty();
        stats.enable_routed(3, 2, &device)?;
        let x = Tensor::new(&[[1f32, 1.]], &device)?;
        let ids = Tensor::new(&[[0u32, 0]], &device)?;
        stats.process_routed(&x, &ids)?;
        let m = stats.compute_imatrix()?.to_vec2::<f32>()?;
        assert!(m[1].iter().all(|v| *v == 0.0));
        assert!(m.iter().flatten().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn expert_stack_per_slab_commutes() -> Result<()> {
        let device = Device::Cpu;
        // in-dim 256 so Q4K applies without fallback
        let stack = Tensor::randn(0f32, 1f32, (3, 8, 256), &device)?;
        let shared: Vec<f32> = (0..256).map(|i| 1.0 + (i % 7) as f32).collect();

        // shared vector through per-slab assembly is byte-identical to whole-stack quantize
        let per_slab = crate::GgufMatMul::quantize_expert_stack(
            &stack,
            crate::IsqType::Q4K,
            Some(&shared),
            &device,
            crate::QuantizeOntoGuard::new(),
        )?;
        let whole = QTensor::quantize_imatrix(&stack, &shared, GgmlDType::Q4K)?;
        assert_eq!(
            per_slab.data()?.to_vec(),
            whole.data()?.to_vec(),
            "per-slab assembly diverged from whole-stack quantization"
        );

        // per-expert vectors actually differentiate slabs
        let mut per_expert = shared.clone();
        per_expert.extend((0..256).map(|i| 100.0 + i as f32));
        per_expert.extend((0..256).map(|_| 1.0));
        let routed = crate::GgufMatMul::quantize_expert_stack(
            &stack,
            crate::IsqType::Q4K,
            Some(&per_expert),
            &device,
            crate::QuantizeOntoGuard::new(),
        )?;
        assert_ne!(
            routed.data()?.to_vec(),
            whole.data()?.to_vec(),
            "per-expert vectors had no effect"
        );
        // expert 0 shares its vector with the shared case and must be bit-identical
        let block_bytes = whole.data()?.len() / 3;
        assert_eq!(
            &routed.data()?[..block_bytes],
            &whole.data()?[..block_bytes],
            "expert 0 should match the shared quantization"
        );
        Ok(())
    }

    #[test]
    fn apply_isq_routes_rank3_imatrix_per_slab() -> Result<()> {
        use std::sync::{Arc, atomic::AtomicUsize};
        let device = Device::Cpu;
        let stack = Tensor::randn(0f32, 1f32, (2, 8, 256), &device)?;
        // expert 1 gets a wildly skewed importance vector so its blocks must quantize differently
        let mut routed: Vec<f32> = vec![1.0; 256];
        routed.extend((0..256).map(|i| if i < 128 { 1000.0 } else { 0.001 }));

        // unquantized resident (capture mode / from-source)
        let make = |im: Option<Vec<f32>>| -> Result<Tensor> {
            let unquant = Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
                inference_tensor::nn::Linear::new(stack.clone(), None),
            ))?) as Arc<dyn QuantMethod>;
            unquant
                .apply_isq(
                    Some(crate::IsqType::Q4K),
                    device.clone(),
                    &AtomicUsize::new(0),
                    im,
                    crate::QuantizeOntoGuard::new(),
                )?
                .dequantize_w()
        };
        let with_imatrix = make(Some(routed.clone()))?;
        assert_eq!(with_imatrix.dims(), [2, 8, 256]);
        let without = make(None)?;
        let diff = (&with_imatrix - &without)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff > 0.0, "routed imatrix had no effect on quantization");

        // quantized resident (apply without source weights)
        let q = Arc::new(crate::GgufMatMul::from_qtensor(
            QTensor::quantize(&stack, GgmlDType::Q8_0)?,
            None,
        )?) as Arc<dyn QuantMethod>;
        let q2 = q.apply_isq(
            Some(crate::IsqType::Q4K),
            device,
            &AtomicUsize::new(0),
            Some(routed),
            crate::QuantizeOntoGuard::new(),
        )?;
        assert_eq!(q2.dequantize_w()?.dims(), [2, 8, 256]);
        Ok(())
    }

    #[test]
    fn disabled_stats_are_free_of_state() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &device)?;
        let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let layer = crate::GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: std::sync::Arc::new(q),
            b: None,
        })?;
        let x = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
        layer.forward_raw(&x)?;
        assert!(layer.end_track_stats().is_err());
        Ok(())
    }
}
