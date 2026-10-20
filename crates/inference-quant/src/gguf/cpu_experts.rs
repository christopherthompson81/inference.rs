//! A GLU expert FFN run on the CPU straight from GGUF blocks in host memory, for the routes a GPU expert cache misses.

use inference_tensor::quantized::GgmlType;
use inference_tensor::quantized::k_quants::{BlockQ8K, QK_K};
use inference_tensor::{Result, bail};
use rayon::prelude::*;

use super::cpu_dot;
use super::kernel::GgufType;
use crate::GluActivationType;
use crate::utils::apply_cpu_activation;

// Output rows a rayon task computes; one row of a 2048-wide IQ3_S expert is under 1 KB
const ROWS_PER_TASK: usize = 32;

/// A weight's GGUF blocks in host memory, with its logical dims (outermost first).
pub struct CpuGgufBlocks<'a> {
    pub ty: GgufType,
    pub dims: &'a [usize],
    pub bytes: std::borrow::Cow<'a, [u8]>,
}

impl CpuGgufBlocks<'_> {
    /// The blocks as every expert's matrix, when the dims are `[experts, rows, cols]`.
    pub fn expert_projection(&self) -> Option<CpuExpertProjection<'_>> {
        let [_, rows, cols] = *self.dims else {
            return None;
        };
        Some(CpuExpertProjection {
            ty: self.ty,
            bytes: &self.bytes,
            rows,
            cols,
        })
    }
}

/// Whether the CPU expert path runs this type at this width.
pub fn supports(ty: GgufType, cols: usize) -> bool {
    cpu_dot::supports(ty) && cols.is_multiple_of(QK_K)
}

/// One projection of every expert: `experts` matrices of `rows x cols`, expert-major, as GGUF stores `*_exps`.
#[derive(Clone, Copy)]
pub struct CpuExpertProjection<'a> {
    pub ty: GgufType,
    pub bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

impl CpuExpertProjection<'_> {
    fn row_bytes(&self) -> usize {
        self.cols / QK_K * self.ty.type_size()
    }

    fn row(&self, expert: usize, row: usize) -> &[u8] {
        let row_bytes = self.row_bytes();
        &self.bytes[(expert * self.rows + row) * row_bytes..][..row_bytes]
    }

    fn check(&self, name: &str) -> Result<()> {
        if !supports(self.ty, self.cols) {
            bail!(
                "{name}: {:?} rows of {} cannot run on the CPU expert path",
                self.ty,
                self.cols
            );
        }
        if !self
            .bytes
            .len()
            .is_multiple_of(self.rows * self.row_bytes())
        {
            bail!("{name}: {} bytes are not whole experts", self.bytes.len());
        }
        Ok(())
    }
}

/// gate, up and down of one MoE layer's experts.
#[derive(Clone, Copy)]
pub struct CpuExperts<'a> {
    pub gate: CpuExpertProjection<'a>,
    pub up: CpuExpertProjection<'a>,
    pub down: CpuExpertProjection<'a>,
    pub act: GluActivationType,
}

/// One routed pick: token row `token` sends to `expert` with routing weight `weight`.
#[derive(Clone, Copy, Debug)]
pub struct CpuExpertRoute {
    pub token: usize,
    pub expert: usize,
    pub weight: f32,
}

fn quantize_rows(xs: &[f32], cols: usize) -> Vec<BlockQ8K> {
    let mut ys = vec![BlockQ8K::zeros(); xs.len() / QK_K];
    ys.par_chunks_mut(cols / QK_K)
        .zip(xs.par_chunks(cols))
        .for_each(|(ys, xs)| BlockQ8K::from_float(xs, ys));
    ys
}

impl CpuExperts<'_> {
    /// Adds each route's weighted expert output into `out` (`tokens x hidden`, as `xs`).
    pub fn forward(&self, xs: &[f32], routes: &[CpuExpertRoute], out: &mut [f32]) -> Result<()> {
        self.gate.check("gate")?;
        self.up.check("up")?;
        self.down.check("down")?;
        let (hidden, inter) = (self.gate.cols, self.gate.rows);
        if self.up.cols != hidden
            || self.up.rows != inter
            || self.down.cols != inter
            || self.down.rows != hidden
            || !xs.len().is_multiple_of(hidden)
            || out.len() != xs.len()
        {
            bail!("CPU experts: projection or activation shapes disagree");
        }
        if routes.is_empty() {
            return Ok(());
        }
        let q8 = quantize_rows(xs, hidden);
        let blocks = hidden / QK_K;

        let mut gated = vec![0f32; routes.len() * inter];
        gated
            .par_chunks_mut(ROWS_PER_TASK)
            .enumerate()
            .for_each(|(task, values)| {
                for (i, value) in values.iter_mut().enumerate() {
                    let flat = task * ROWS_PER_TASK + i;
                    let (route, row) = (&routes[flat / inter], flat % inter);
                    let x = &q8[route.token * blocks..][..blocks];
                    let gate = cpu_dot::vec_dot(self.gate.ty, self.gate.row(route.expert, row), x);
                    let up = cpu_dot::vec_dot(self.up.ty, self.up.row(route.expert, row), x);
                    *value = apply_cpu_activation(gate, self.act) * up;
                }
            });

        let gated = quantize_rows(&gated, inter);
        let inter_blocks = inter / QK_K;
        let mut routed = vec![0f32; routes.len() * hidden];
        routed
            .par_chunks_mut(ROWS_PER_TASK)
            .enumerate()
            .for_each(|(task, values)| {
                for (i, value) in values.iter_mut().enumerate() {
                    let flat = task * ROWS_PER_TASK + i;
                    let (r, row) = (flat / hidden, flat % hidden);
                    let h = &gated[r * inter_blocks..][..inter_blocks];
                    *value =
                        cpu_dot::vec_dot(self.down.ty, self.down.row(routes[r].expert, row), h);
                }
            });

        for (route, y) in routes.iter().zip(routed.chunks_exact(hidden)) {
            for (o, v) in out[route.token * hidden..][..hidden].iter_mut().zip(y) {
                *o += route.weight * v;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPERTS: usize = 4;
    const HIDDEN: usize = 512;
    const INTER: usize = 256;

    fn random_bytes(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }

    // random blocks whose leading f16 scale is small and finite
    fn projection_bytes(ty: GgufType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        let size = ty.type_size();
        let mut bytes = random_bytes(EXPERTS * rows * cols / QK_K * size, seed);
        for (i, block) in bytes.chunks_exact_mut(size).enumerate() {
            let d = half::f16::from_f32(0.002 + 0.0001 * (i % 7) as f32).to_le_bytes();
            block[..2].copy_from_slice(&d);
        }
        bytes
    }

    fn dense(projection: &CpuExpertProjection, expert: usize) -> Vec<f32> {
        let expert_bytes = projection.rows * projection.row_bytes();
        super::super::raw::dequantize_rows(
            projection.ty,
            projection.cols,
            &projection.bytes[expert * expert_bytes..][..expert_bytes],
        )
        .unwrap()
    }

    fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
        w.chunks_exact(x.len())
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    #[test]
    fn routed_forward_matches_dense_experts() -> Result<()> {
        let gate = projection_bytes(GgufType::Iq3S, INTER, HIDDEN, 1);
        let up = projection_bytes(GgufType::Iq3S, INTER, HIDDEN, 2);
        let down = projection_bytes(GgufType::Iq4Xs, HIDDEN, INTER, 3);
        let experts = CpuExperts {
            gate: CpuExpertProjection {
                ty: GgufType::Iq3S,
                bytes: &gate,
                rows: INTER,
                cols: HIDDEN,
            },
            up: CpuExpertProjection {
                ty: GgufType::Iq3S,
                bytes: &up,
                rows: INTER,
                cols: HIDDEN,
            },
            down: CpuExpertProjection {
                ty: GgufType::Iq4Xs,
                bytes: &down,
                rows: HIDDEN,
                cols: INTER,
            },
            act: GluActivationType::Silu,
        };
        let xs = (0..2 * HIDDEN)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) / 24.0)
            .collect::<Vec<_>>();
        let routes = [
            CpuExpertRoute {
                token: 0,
                expert: 3,
                weight: 0.7,
            },
            CpuExpertRoute {
                token: 0,
                expert: 1,
                weight: 0.3,
            },
            CpuExpertRoute {
                token: 1,
                expert: 3,
                weight: 1.0,
            },
        ];
        let mut out = vec![0f32; xs.len()];
        experts.forward(&xs, &routes, &mut out)?;

        let mut expected = vec![0f32; xs.len()];
        for route in &routes {
            let x = &xs[route.token * HIDDEN..][..HIDDEN];
            let g = matvec(&dense(&experts.gate, route.expert), x);
            let u = matvec(&dense(&experts.up, route.expert), x);
            let h = g
                .iter()
                .zip(&u)
                .map(|(g, u)| apply_cpu_activation(*g, GluActivationType::Silu) * u)
                .collect::<Vec<_>>();
            let y = matvec(&dense(&experts.down, route.expert), &h);
            for (e, v) in expected[route.token * HIDDEN..][..HIDDEN].iter_mut().zip(y) {
                *e += route.weight * v;
            }
        }
        let norm = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
        let err = out
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        // both activations are quantized to Q8_K on this path
        assert!(err < 0.02 * norm, "relative error {}", err / norm);
        Ok(())
    }

    // the decode shape of a 35B-A3B layer (2048 x 512 experts, IQ3_S gate/up, IQ4_XS down), 8 picks a token over
    // 256 experts so the weights come from RAM rather than L3
    #[test]
    #[ignore = "benchmark"]
    fn decode_layer_bandwidth() -> Result<()> {
        const ALL: usize = 256;
        const PICKS: usize = 8;
        const ITERS: usize = 400;
        let (hidden, inter) = (2048, 512);
        let bytes = |ty: GgufType, rows: usize, cols: usize, seed| {
            random_bytes(ALL * rows * cols / QK_K * ty.type_size(), seed)
        };
        let (gate, up, down) = (
            bytes(GgufType::Iq3S, inter, hidden, 1),
            bytes(GgufType::Iq3S, inter, hidden, 2),
            bytes(GgufType::Iq4Xs, hidden, inter, 3),
        );
        let experts = CpuExperts {
            gate: CpuExpertProjection {
                ty: GgufType::Iq3S,
                bytes: &gate,
                rows: inter,
                cols: hidden,
            },
            up: CpuExpertProjection {
                ty: GgufType::Iq3S,
                bytes: &up,
                rows: inter,
                cols: hidden,
            },
            down: CpuExpertProjection {
                ty: GgufType::Iq4Xs,
                bytes: &down,
                rows: hidden,
                cols: inter,
            },
            act: GluActivationType::Silu,
        };
        let xs = vec![0.01f32; hidden];
        let mut out = vec![0f32; hidden];
        let per_expert = (gate.len() + up.len() + down.len()) / ALL;
        let start = std::time::Instant::now();
        for iter in 0..ITERS {
            let routes = (0..PICKS)
                .map(|k| CpuExpertRoute {
                    token: 0,
                    expert: (iter * 37 + k * 31) % ALL,
                    weight: 0.1,
                })
                .collect::<Vec<_>>();
            experts.forward(&xs, &routes, &mut out)?;
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{:.1} us a layer, {:.1} GB/s",
            secs / ITERS as f64 * 1e6,
            (ITERS * PICKS * per_expert) as f64 / secs / 1e9
        );
        Ok(())
    }
}
