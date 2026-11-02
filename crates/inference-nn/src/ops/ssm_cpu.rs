//! Mamba-2's prefill on the CPU in two passes over plain slices: the causal depthwise convolution, and the selective
//! scan with each head's state kept in cache across the sequence. The tensor-op loop it replaces ran a dozen ops per
//! timestep over the whole state, each one split across the thread pool.

use inference_tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;

/// The scan's inputs, as the CPU and CUDA scans take them (all f32, contiguous after the call's copies):
/// `x` `(batch, seq, heads, head_dim)`, `dt` `(batch, seq, heads)` before bias and softplus, `a` `(heads,)` (already
/// negated), `b` and `c` `(batch, seq, groups, state)` (heads split evenly over the groups), `d` and `dt_bias`
/// `(heads,)`.
#[derive(Clone, Copy)]
pub struct SelectiveScan<'a> {
    pub x: &'a Tensor,
    pub dt: &'a Tensor,
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    pub c: &'a Tensor,
    pub d: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub dt_min: f32,
    pub dt_max: f32,
}

// partial sums the dot with C keeps apart, as the AVX2 lanes do
const DOT_LANES: usize = 8;
// channels a task of the conv step covers
const CONV_CHANNELS_PER_TASK: usize = 512;

/// One state row's step: `row = row * decay + dtx * b`, then `row . c`.
#[inline(always)]
fn state_row_step(row: &mut [f32], b: &[f32], c: &[f32], decay: f32, dtx: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2_fma() && row.len().is_multiple_of(DOT_LANES) {
        // SAFETY: AVX2 and FMA were detected, and the row is whole lanes
        return unsafe { state_row_step_avx2(row, b, c, decay, dtx) };
    }
    let mut lanes = [0f32; DOT_LANES];
    for (i, ((sn, &bn), &cn)) in row.iter_mut().zip(b).zip(c).enumerate() {
        *sn = *sn * decay + dtx * bn;
        lanes[i % DOT_LANES] += *sn * cn;
    }
    lanes.iter().sum()
}

#[cfg(target_arch = "x86_64")]
fn avx2_fma() -> bool {
    static DETECTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DETECTED.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn state_row_step_avx2(row: &mut [f32], b: &[f32], c: &[f32], decay: f32, dtx: f32) -> f32 {
    use std::arch::x86_64::*;
    let (vdecay, vdtx) = (_mm256_set1_ps(decay), _mm256_set1_ps(dtx));
    let mut acc = _mm256_setzero_ps();
    for i in (0..row.len()).step_by(DOT_LANES) {
        // SAFETY: `row`, `b` and `c` hold whole lanes from `i`
        unsafe {
            let s = _mm256_loadu_ps(row.as_ptr().add(i));
            // the update as a multiply and an add, as the scalar and CUDA scans round it
            let s = _mm256_add_ps(
                _mm256_mul_ps(s, vdecay),
                _mm256_mul_ps(vdtx, _mm256_loadu_ps(b.as_ptr().add(i))),
            );
            _mm256_storeu_ps(row.as_mut_ptr().add(i), s);
            acc = _mm256_fmadd_ps(s, _mm256_loadu_ps(c.as_ptr().add(i)), acc);
        }
    }
    let mut lanes = [0f32; DOT_LANES];
    // SAFETY: `lanes` holds eight floats
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    lanes.iter().sum()
}

fn f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// `y` `(batch, seq, heads, head_dim)`, with `state` `(batch, heads, head_dim, state)` advanced over the sequence: per
/// step `dt = clamp(softplus(dt + dt_bias))`, `state = state * exp(dt * a) + dt * b * x`, `y = state . c + d * x`.
pub fn selective_scan_cpu(scan: SelectiveScan<'_>, state: &mut Tensor) -> Result<Tensor> {
    let inputs = ScanInputs::new(scan)?;
    let mut states = f32_vec(state)?;
    let y = inputs.scan(states.chunks_mut(inputs.head_state()).map(Some).collect());
    *state =
        Tensor::from_vec(states, state.shape().clone(), &Device::Cpu)?.to_dtype(state.dtype())?;
    inputs.output(y)
}

/// [`selective_scan_cpu`] on a recurrent pool: batch row `b` advances the pool's row `slots[b]` in place (`pool` f32
/// `(rows, heads, head_dim, state)`), a slot past `i32::MAX` marking a padding row that leaves the pool and scores 0.
pub fn selective_scan_cpu_slots(
    scan: SelectiveScan<'_>,
    pool: &Tensor,
    slots: &[u32],
) -> Result<Tensor> {
    let op = SlotScan {
        inputs: ScanInputs::new(scan)?,
        slots,
        y: std::cell::RefCell::new(Vec::new()),
    };
    pool.inplace_op1(&op)?;
    op.inputs.output(op.y.take())
}

struct ScanInputs {
    // batch, seq, heads, head_dim, groups, d_state
    shape: [usize; 6],
    x: Vec<f32>,
    dt: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
    dt_bias: Vec<f32>,
    dt_range: [f32; 2],
}

impl ScanInputs {
    fn new(scan: SelectiveScan<'_>) -> Result<Self> {
        let (batch, seq, heads, head_dim) = scan.x.dims4()?;
        Ok(Self {
            shape: [batch, seq, heads, head_dim, scan.b.dim(2)?, scan.b.dim(3)?],
            x: f32_vec(scan.x)?,
            dt: f32_vec(scan.dt)?,
            a: f32_vec(scan.a)?,
            b: f32_vec(scan.b)?,
            c: f32_vec(scan.c)?,
            d: f32_vec(scan.d)?,
            dt_bias: f32_vec(scan.dt_bias)?,
            dt_range: [scan.dt_min, scan.dt_max],
        })
    }

    fn head_state(&self) -> usize {
        self.shape[3] * self.shape[5]
    }

    fn output(&self, y: Vec<f32>) -> Result<Tensor> {
        let [batch, seq, heads, head_dim, ..] = self.shape;
        Tensor::from_vec(y, (batch, seq, heads, head_dim), &Device::Cpu)
    }

    // `states` holds each (batch, head) item's state block, `None` for a padding row's
    fn scan(&self, mut states: Vec<Option<&mut [f32]>>) -> Vec<f32> {
        let [batch, seq, heads, head_dim, groups, d_state] = self.shape;
        let heads_per_group = heads / groups;
        // one item per (batch, head): its state block and its outputs, strided through `y`
        let mut outputs: Vec<Vec<f32>> = vec![vec![0f32; seq * head_dim]; batch * heads];
        states
            .par_iter_mut()
            .zip(outputs.par_iter_mut())
            .enumerate()
            .for_each(|(item, (s, out))| {
                let Some(s) = s else { return };
                let (bi, h) = (item / heads, item % heads);
                for t in 0..seq {
                    let step = (bi * seq + t) * heads + h;
                    let raw = self.dt[step] + self.dt_bias[h];
                    let dt_t = (1.0 + raw.exp())
                        .ln()
                        .clamp(self.dt_range[0], self.dt_range[1]);
                    let decay = (dt_t * self.a[h]).exp();
                    let group_step = (bi * seq + t) * groups + h / heads_per_group;
                    let bt = &self.b[group_step * d_state..(group_step + 1) * d_state];
                    let ct = &self.c[group_step * d_state..(group_step + 1) * d_state];
                    let xt = &self.x[step * head_dim..(step + 1) * head_dim];
                    for (p, row) in s.chunks_exact_mut(d_state).enumerate() {
                        out[t * head_dim + p] =
                            state_row_step(row, bt, ct, decay, dt_t * xt[p]) + self.d[h] * xt[p];
                    }
                }
            });
        let mut y = vec![0f32; batch * seq * heads * head_dim];
        for (item, out) in outputs.iter().enumerate() {
            let (bi, h) = (item / heads, item % heads);
            for t in 0..seq {
                let at = ((bi * seq + t) * heads + h) * head_dim;
                y[at..at + head_dim].copy_from_slice(&out[t * head_dim..(t + 1) * head_dim]);
            }
        }
        y
    }
}

struct SlotScan<'a> {
    inputs: ScanInputs,
    slots: &'a [u32],
    y: std::cell::RefCell<Vec<f32>>,
}

impl inference_tensor::InplaceOp1 for SlotScan<'_> {
    fn name(&self) -> &'static str {
        "selective-scan-cpu-slots"
    }

    fn cpu_fwd(
        &self,
        storage: &mut inference_tensor::CpuStorage,
        layout: &inference_tensor::Layout,
    ) -> Result<()> {
        let inference_tensor::CpuStorage::F32(data) = storage else {
            inference_tensor::bail!("selective-scan-cpu-slots: the pool must be f32");
        };
        if !layout.is_contiguous() {
            inference_tensor::bail!("selective-scan-cpu-slots: the pool must be contiguous");
        }
        let head_state = self.inputs.head_state();
        let row_len = self.inputs.shape[2] * head_state;
        let pool = &mut data[layout.start_offset()..][..layout.shape().elem_count()];
        let mut rows: Vec<Option<&mut [f32]>> = pool.chunks_mut(row_len).map(Some).collect();
        let mut states = Vec::with_capacity(self.slots.len() * self.inputs.shape[2]);
        for &slot in self.slots {
            if slot > i32::MAX as u32 {
                states.extend((0..self.inputs.shape[2]).map(|_| None));
                continue;
            }
            let Some(row) = rows.get_mut(slot as usize).and_then(Option::take) else {
                inference_tensor::bail!(
                    "selective-scan-cpu-slots: slot {slot} is past the pool or repeated"
                );
            };
            states.extend(row.chunks_mut(head_state).map(Some));
        }
        *self.y.borrow_mut() = self.inputs.scan(states);
        Ok(())
    }
}

/// One decode step's causal conv on a conv-state pool, in place: batch row `b` shifts `x[b]` into the pool's row
/// `slots[b]` (`pool` f32 `(rows, channels, kernel)`), then returns `silu(window . weight + bias)` `(batch, channels)`;
/// a slot past `i32::MAX` marks a padding row that leaves the pool and scores 0.
pub fn causal_conv1d_step_cpu_slots(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    pool: &Tensor,
    slots: &[u32],
) -> Result<Tensor> {
    let (batch, channels) = x.dims2()?;
    let op = SlotConvStep {
        x: f32_vec(x)?,
        weight: f32_vec(weight)?,
        bias: bias.map(f32_vec).transpose()?,
        channels,
        slots,
        out: std::cell::RefCell::new(Vec::new()),
    };
    pool.inplace_op1(&op)?;
    Tensor::from_vec(op.out.take(), (batch, channels), &Device::Cpu)
}

struct SlotConvStep<'a> {
    x: Vec<f32>,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    channels: usize,
    slots: &'a [u32],
    out: std::cell::RefCell<Vec<f32>>,
}

impl inference_tensor::InplaceOp1 for SlotConvStep<'_> {
    fn name(&self) -> &'static str {
        "causal-conv1d-step-cpu-slots"
    }

    fn cpu_fwd(
        &self,
        storage: &mut inference_tensor::CpuStorage,
        layout: &inference_tensor::Layout,
    ) -> Result<()> {
        let inference_tensor::CpuStorage::F32(data) = storage else {
            inference_tensor::bail!("causal-conv1d-step-cpu-slots: the pool must be f32");
        };
        if !layout.is_contiguous() {
            inference_tensor::bail!("causal-conv1d-step-cpu-slots: the pool must be contiguous");
        }
        let (_, channels, kernel) = layout.shape().dims3()?;
        if channels != self.channels || self.weight.len() != channels * kernel {
            inference_tensor::bail!(
                "causal-conv1d-step-cpu-slots: {channels} channels of {kernel} taps"
            );
        }
        let pool = &mut data[layout.start_offset()..][..layout.shape().elem_count()];
        let mut out = vec![0f32; self.slots.len() * channels];
        let mut rows: Vec<Option<&mut [f32]>> =
            pool.chunks_mut(channels * kernel).map(Some).collect();
        for (b, (&slot, dst)) in self.slots.iter().zip(out.chunks_mut(channels)).enumerate() {
            if slot > i32::MAX as u32 {
                continue;
            }
            let Some(row) = rows.get_mut(slot as usize).and_then(Option::take) else {
                inference_tensor::bail!(
                    "causal-conv1d-step-cpu-slots: slot {slot} is past the pool or repeated"
                );
            };
            let (x, weight, bias) = (
                &self.x[b * channels..(b + 1) * channels],
                &self.weight,
                &self.bias,
            );
            // channels split across the pool's threads: a serial pass here stalls every worker
            row.par_chunks_mut(kernel * CONV_CHANNELS_PER_TASK)
                .zip(dst.par_chunks_mut(CONV_CHANNELS_PER_TASK))
                .enumerate()
                .for_each(|(task, (windows, dst))| {
                    let ch0 = task * CONV_CHANNELS_PER_TASK;
                    for (i, (window, o)) in windows.chunks_mut(kernel).zip(dst).enumerate() {
                        let ch = ch0 + i;
                        window.copy_within(1.., 0);
                        window[kernel - 1] = x[ch];
                        let w = &weight[ch * kernel..(ch + 1) * kernel];
                        let mut sum = 0f32;
                        for j in 0..kernel {
                            sum += window[j] * w[j];
                        }
                        if let Some(bias) = bias {
                            sum += bias[ch];
                        }
                        *o = sum / (1.0 + (-sum).exp());
                    }
                });
        }
        *self.out.borrow_mut() = out;
        Ok(())
    }
}

/// The causal depthwise convolution over `padded` `(batch, channels, kernel - 1 + seq)` with `weight`
/// `(channels, kernel)`, as `(batch, seq, channels)` in `padded`'s dtype: output step `t` sums taps `t..t + kernel`.
pub fn causal_conv1d_cpu(padded: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (batch, channels, width) = padded.dims3()?;
    let kernel = weight.dim(1)?;
    let seq = width + 1 - kernel;
    let (input, weight) = (f32_vec(padded)?, f32_vec(weight)?);
    let mut out = vec![0f32; batch * seq * channels];
    // each output row (one step's channels) reads every channel's window at that step
    out.par_chunks_mut(channels)
        .enumerate()
        .for_each(|(row, dst)| {
            let (bi, t) = (row / seq, row % seq);
            for (ch, o) in dst.iter_mut().enumerate() {
                let src = &input[(bi * channels + ch) * width + t..][..kernel];
                let w = &weight[ch * kernel..(ch + 1) * kernel];
                // an index loop: the zipped sum stayed an out-of-line call per channel in the dev profile
                let mut sum = 0f32;
                for j in 0..kernel {
                    sum += src[j] * w[j];
                }
                *o = sum;
            }
        });
    Tensor::from_vec(out, (batch, seq, channels), &Device::Cpu)?.to_dtype(padded.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADS: usize = 4;
    const GROUPS: usize = 2;
    const HEAD_DIM: usize = 8;
    const D_STATE: usize = 16;
    const CHANNELS: usize = 12;
    const KERNEL: usize = 4;
    const POOL_ROWS: usize = 3;
    // pool rows 2 and 0, with a padding row between
    const SLOTS: [u32; 3] = [2, u32::MAX, 0];
    const LIVE: [(usize, usize); 2] = [(0, 2), (2, 0)];

    fn rand(shape: &[usize]) -> Result<Tensor> {
        Tensor::randn(0f32, 0.5, shape, &Device::Cpu)
    }

    // each live batch row scores as the plain scan does on its pool row, the padding row scores 0 and no pool row
    // outside the slots changes
    #[test]
    fn a_slot_scan_advances_the_pool_rows_in_place() -> Result<()> {
        let batch = SLOTS.len();
        let inputs = [
            rand(&[batch, 1, HEADS, HEAD_DIM])?,
            rand(&[batch, 1, HEADS])?,
            (rand(&[HEADS])?.abs()? * -1.0)?,
            rand(&[batch, 1, GROUPS, D_STATE])?,
            rand(&[batch, 1, GROUPS, D_STATE])?,
            rand(&[HEADS])?,
            rand(&[HEADS])?,
        ];
        let scan = |t: &[Tensor], b: Option<usize>| -> Result<[Tensor; 7]> {
            Ok(std::array::from_fn(|i| match (b, i) {
                (Some(b), 0 | 1 | 3 | 4) => t[i].narrow(0, b, 1).unwrap(),
                _ => t[i].clone(),
            }))
        };
        fn view(t: &[Tensor; 7]) -> SelectiveScan<'_> {
            SelectiveScan {
                x: &t[0],
                dt: &t[1],
                a: &t[2],
                b: &t[3],
                c: &t[4],
                d: &t[5],
                dt_bias: &t[6],
                dt_min: 0.0,
                dt_max: f32::MAX,
            }
        }
        let pool = rand(&[POOL_ROWS, HEADS, HEAD_DIM, D_STATE])?;
        let before = pool.copy()?;
        let all = scan(&inputs, None)?;
        let y = selective_scan_cpu_slots(view(&all), &pool, &SLOTS)?;
        for (b, row) in LIVE {
            let mut state = before.narrow(0, row, 1)?.copy()?;
            let one = scan(&inputs, Some(b))?;
            let expected = selective_scan_cpu(view(&one), &mut state)?;
            assert_eq!(
                y.narrow(0, b, 1)?.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
            assert_eq!(
                pool.narrow(0, row, 1)?.flatten_all()?.to_vec1::<f32>()?,
                state.flatten_all()?.to_vec1::<f32>()?
            );
        }
        assert!(
            y.narrow(0, 1, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|&v| v == 0.0)
        );
        assert_eq!(
            pool.narrow(0, 1, 1)?.flatten_all()?.to_vec1::<f32>()?,
            before.narrow(0, 1, 1)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    // the window shifts the step in, the output is silu of the window's dot with the weight plus the bias
    #[test]
    fn a_slot_conv_step_shifts_the_pool_windows_in_place() -> Result<()> {
        let x = rand(&[SLOTS.len(), CHANNELS])?;
        let weight = rand(&[CHANNELS, KERNEL])?;
        let bias = rand(&[CHANNELS])?;
        let pool = rand(&[POOL_ROWS, CHANNELS, KERNEL])?;
        let before = pool.copy()?;
        let out = causal_conv1d_step_cpu_slots(&x, &weight, Some(&bias), &pool, &SLOTS)?;
        for (b, row) in LIVE {
            let window = Tensor::cat(
                &[
                    before.narrow(0, row, 1)?.narrow(2, 1, KERNEL - 1)?,
                    x.narrow(0, b, 1)?.unsqueeze(2)?,
                ],
                2,
            )?;
            assert_eq!(
                pool.narrow(0, row, 1)?.flatten_all()?.to_vec1::<f32>()?,
                window.flatten_all()?.to_vec1::<f32>()?
            );
            let expected = inference_tensor::nn::ops::silu(
                &(window.squeeze(0)? * &weight)?.sum(1)?.add(&bias)?,
            )?;
            let got = out.narrow(0, b, 1)?.squeeze(0)?.to_vec1::<f32>()?;
            for (g, e) in got.iter().zip(expected.to_vec1::<f32>()?) {
                assert!((g - e).abs() <= 1e-6 * e.abs().max(1.0), "{g} vs {e}");
            }
        }
        assert!(
            out.narrow(0, 1, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|&v| v == 0.0)
        );
        assert_eq!(
            pool.narrow(0, 1, 1)?.flatten_all()?.to_vec1::<f32>()?,
            before.narrow(0, 1, 1)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
