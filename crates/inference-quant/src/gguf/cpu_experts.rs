//! A GLU expert FFN run on the CPU straight from GGUF blocks in host memory, for the routes a GPU expert cache misses.

use inference_tensor::quantized::GgmlType;
use inference_tensor::quantized::k_quants::{BlockQ8K, QK_K};
use inference_tensor::{Result, bail};

use super::cpu_dot;

// Tokens one expert row is dotted against per pass, decoding its blocks once for all of them
const DOT_TOKENS: usize = 4;
// Weights one work item decodes into its i16 tile; its row count is a power of two from 2 to QK_K, so it divides
// every projection's rows
const TILE_ELEMS: usize = 8 * 1024;
// An expert's routed tokens from which decoding its rows once into a tile beats the per-quad kernels
const TILE_MIN_TOKENS: usize = 8;
// Most routes one work item runs; past it an expert's routes split, decoding its tiles once per span
const SPAN_TOKENS: usize = 32;
// Grabs of the pool's chunked loop that each drain the item cursor; at least the pool's thread count
const DRAINERS: usize = 256;
// Consecutive items (an expert's neighbouring row tiles) one grab takes, keeping its output stores off other cores' lines
const GRAB_ITEMS: usize = 16;
// Q8_K blocks from which quantizing activations is split over the pool, and the blocks one item quantizes
const PARALLEL_QUANT_BLOCKS: usize = 256;
const QUANT_ITEM_BLOCKS: usize = 64;
// Tokens one item of the final weighted sum adds up
const SUM_ITEM_TOKENS: usize = 8;

use super::cpu_dot::TileRow;
use super::kernel::GgufType;
use crate::GluActivationType;
use crate::utils::apply_cpu_activation;

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
        self.ty.row_scale_bytes() + self.cols / QK_K * self.ty.type_size()
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

// Q8_K blocks of `xs`, and their values widened to i16 when `wide`, split over the pool once the rows are many
fn quantize_rows(xs: &[f32], wide: bool) -> (Vec<BlockQ8K>, Vec<i16>) {
    let n = xs.len() / QK_K;
    let mut ys = vec![BlockQ8K::zeros(); n];
    let mut qs = vec![0i16; if wide { xs.len() } else { 0 }];
    let run = |blocks: std::ops::Range<usize>, ys: &mut [BlockQ8K], qs: &mut [i16]| {
        BlockQ8K::from_float(&xs[blocks.start * QK_K..blocks.end * QK_K], ys);
        if wide {
            cpu_dot::widen_into(ys, qs);
        }
    };
    if n < PARALLEL_QUANT_BLOCKS {
        run(0..n, &mut ys, &mut qs);
        return (ys, qs);
    }
    let (ys_out, qs_out) = (Disjoint(ys.as_mut_ptr()), Disjoint(qs.as_mut_ptr()));
    let wide_len = if wide { QK_K } else { 0 };
    let items = n.div_ceil(QUANT_ITEM_BLOCKS);
    for_each_parallel(items, |item| {
        let blocks = item * QUANT_ITEM_BLOCKS..n.min((item + 1) * QUANT_ITEM_BLOCKS);
        // SAFETY: each item's blocks (and their values) belong to it alone
        let (ys, qs) = unsafe {
            (
                ys_out.slice(blocks.start, blocks.len()),
                qs_out.slice(blocks.start * wide_len, blocks.len() * wide_len),
            )
        };
        run(blocks, ys, qs);
    });
    (ys, qs)
}

// The quantized matmuls' barrier pool spins between calls, so a second pool here would fight it for the cores.
// Items cost what their span's length makes them, so each worker drains one shared cursor a few items at a time.
fn for_each_parallel(n_items: usize, f: impl Fn(usize) + Sync) {
    if n_items == 1 {
        return f(0);
    }
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    inference_tensor::utils::barrier_pool().execute_chunked(DRAINERS, |_| {
        loop {
            let start = cursor.fetch_add(GRAB_ITEMS, std::sync::atomic::Ordering::Relaxed);
            if start >= n_items {
                break;
            }
            (start..n_items.min(start + GRAB_ITEMS)).for_each(&f);
        }
    });
}

/// A shared output each parallel item writes its own disjoint elements of.
struct Disjoint<T>(*mut T);

// SAFETY: every writer stores to indices no other item touches, and the buffer outlives the parallel call
unsafe impl<T> Sync for Disjoint<T> {}

impl<T> Disjoint<T> {
    // SAFETY: `i` is in bounds and written by this item alone
    unsafe fn set(&self, i: usize, value: T) {
        unsafe { *self.0.add(i) = value };
    }

    // SAFETY: `start..start + len` is in bounds and touched by this item alone
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

/// A worker's decoded weights and the two projections' results of its current item.
#[derive(Default)]
struct Tile {
    w: Vec<i16>,
    d: Vec<f32>,
    out: [Vec<f32>; 2],
}

impl Tile {
    // rows `row0..row0 + rows_per_item(p.cols)` of `expert` against `tokens`, into `self.out[slot]` as `[row][token]`
    fn project(
        &mut self,
        slot: usize,
        p: &CpuExpertProjection,
        expert: usize,
        row0: usize,
        tokens: &[TileRow],
    ) {
        let row_bytes = p.row_bytes();
        let rows = rows_per_item(p.cols);
        self.w.resize(rows * p.cols, 0);
        self.d.resize(rows * p.cols / QK_K, 0.0);
        self.out[slot].resize(rows * tokens.len(), 0.0);
        let bytes = &p.bytes[(expert * p.rows + row0) * row_bytes..][..rows * row_bytes];
        cpu_dot::decode_tile(p.ty, bytes, p.cols, &mut self.w, &mut self.d);
        cpu_dot::tile_dot(&self.w, &self.d, tokens, &mut self.out[slot]);
    }
}

thread_local! {
    static TILE: std::cell::RefCell<Tile> = std::cell::RefCell::default();
    // the gated and routed rows, reused so a prompt's tens of MB are not faulted in again for every layer
    static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> = std::cell::RefCell::default();
}

// each tiled group's activation rows, in the group's route order
fn tile_rows<'a>(
    groups: &[&[usize]],
    tiled: bool,
    row: impl Fn(usize) -> TileRow<'a>,
) -> Vec<Option<Vec<TileRow<'a>>>> {
    groups
        .iter()
        .map(|g| (tiled && g.len() >= TILE_MIN_TOKENS).then(|| g.iter().map(|&r| row(r)).collect()))
        .collect()
}

// the rows one item of a projection with `cols` columns covers
fn rows_per_item(cols: usize) -> usize {
    1 << (TILE_ELEMS / cols).clamp(2, QK_K).ilog2()
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
        let blocks = hidden / QK_K;
        // routes grouped by expert, so each weight row is read once for every token sent to it
        let mut order = (0..routes.len()).collect::<Vec<_>>();
        order.sort_by_key(|&r| routes[r].expert);
        // and a busy expert's routes split into even spans, so no one work item dwarfs the rest
        let groups = order
            .chunk_by(|&a, &b| routes[a].expert == routes[b].expert)
            .flat_map(|g| g.chunks(g.len().div_ceil(g.len().div_ceil(SPAN_TOKENS))))
            .collect::<Vec<_>>();
        let tiled = |p: &CpuExpertProjection| {
            cpu_dot::tile_supports(p.ty) && groups.iter().any(|g| g.len() >= TILE_MIN_TOKENS)
        };
        let tiled_gate_up = tiled(&self.gate) && tiled(&self.up);
        let tiled_down = tiled(&self.down);

        let (q8, q16) = quantize_rows(xs, tiled_gate_up);
        let gate_up_rows = tile_rows(&groups, tiled_gate_up, |r| TileRow {
            qs: &q16[routes[r].token * hidden..][..hidden],
            blocks: &q8[routes[r].token * blocks..][..blocks],
        });
        // every element of both is written before it is read
        let (mut gated_rows, mut routed) = SCRATCH.take();
        gated_rows.resize(routes.len() * inter, 0.0);
        let gated_out = Disjoint(gated_rows.as_mut_ptr());
        let item_rows = rows_per_item(hidden);
        let chunks = inter / item_rows;
        for_each_parallel(groups.len() * chunks, |item| {
            let (g, row0) = (item / chunks, item % chunks * item_rows);
            let group = groups[g];
            let expert = routes[group[0]].expert;
            let store = |r: usize, row: usize, gate: f32, up: f32| {
                // SAFETY: route r's row `row` belongs to this item alone
                unsafe {
                    gated_out.set(r * inter + row, apply_cpu_activation(gate, self.act) * up)
                };
            };
            if let Some(tokens) = &gate_up_rows[g] {
                TILE.with_borrow_mut(|tile| {
                    let n = tokens.len();
                    tile.project(0, &self.gate, expert, row0, tokens);
                    tile.project(1, &self.up, expert, row0, tokens);
                    for k in 0..item_rows {
                        for (t, &r) in group.iter().enumerate() {
                            store(r, row0 + k, tile.out[0][k * n + t], tile.out[1][k * n + t]);
                        }
                    }
                });
                return;
            }
            let x = |r: usize| &q8[routes[r].token * blocks..][..blocks];
            for row in row0..row0 + item_rows {
                let (gate_row, up_row) = (self.gate.row(expert, row), self.up.row(expert, row));
                let (quads, rest) = group.as_chunks::<DOT_TOKENS>();
                for quad in quads {
                    let xs = quad.map(x);
                    let gate = cpu_dot::vec_dot_x4(self.gate.ty, gate_row, xs);
                    let up = cpu_dot::vec_dot_x4(self.up.ty, up_row, xs);
                    for (k, &r) in quad.iter().enumerate() {
                        store(r, row, gate[k], up[k]);
                    }
                }
                for &r in rest {
                    let (gate, up) = (
                        cpu_dot::vec_dot(self.gate.ty, gate_row, x(r)),
                        cpu_dot::vec_dot(self.up.ty, up_row, x(r)),
                    );
                    store(r, row, gate, up);
                }
            }
        });

        let (gated, gated16) = quantize_rows(&gated_rows, tiled_down);
        let inter_blocks = inter / QK_K;
        let down_rows = tile_rows(&groups, tiled_down, |r| TileRow {
            qs: &gated16[r * inter..][..inter],
            blocks: &gated[r * inter_blocks..][..inter_blocks],
        });
        routed.resize(routes.len() * hidden, 0.0);
        let routed_out = Disjoint(routed.as_mut_ptr());
        let item_rows = rows_per_item(inter);
        let chunks = hidden / item_rows;
        for_each_parallel(groups.len() * chunks, |item| {
            let (g, row0) = (item / chunks, item % chunks * item_rows);
            let group = groups[g];
            let expert = routes[group[0]].expert;
            // SAFETY: as above, for the down projection's rows
            let store = |r: usize, row: usize, value: f32| unsafe {
                routed_out.set(r * hidden + row, value)
            };
            if let Some(tokens) = &down_rows[g] {
                TILE.with_borrow_mut(|tile| {
                    let n = tokens.len();
                    tile.project(0, &self.down, expert, row0, tokens);
                    for k in 0..item_rows {
                        for (t, &r) in group.iter().enumerate() {
                            store(r, row0 + k, tile.out[0][k * n + t]);
                        }
                    }
                });
                return;
            }
            let h = |r: usize| &gated[r * inter_blocks..][..inter_blocks];
            for row in row0..row0 + item_rows {
                let down_row = self.down.row(expert, row);
                let (quads, rest) = group.as_chunks::<DOT_TOKENS>();
                for quad in quads {
                    let ys = cpu_dot::vec_dot_x4(self.down.ty, down_row, quad.map(h));
                    for (k, &r) in quad.iter().enumerate() {
                        store(r, row, ys[k]);
                    }
                }
                for &r in rest {
                    store(r, row, cpu_dot::vec_dot(self.down.ty, down_row, h(r)));
                }
            }
        });

        // each token's routes, so the weighted sum splits over tokens
        let tokens = xs.len() / hidden;
        let mut starts = vec![0usize; tokens + 1];
        for route in routes {
            starts[route.token + 1] += 1;
        }
        for t in 0..tokens {
            starts[t + 1] += starts[t];
        }
        let mut by_token = vec![0usize; routes.len()];
        let mut next = starts.clone();
        for (r, route) in routes.iter().enumerate() {
            by_token[next[route.token]] = r;
            next[route.token] += 1;
        }
        let sum_out = Disjoint(out.as_mut_ptr());
        for_each_parallel(tokens.div_ceil(SUM_ITEM_TOKENS), |item| {
            for t in item * SUM_ITEM_TOKENS..tokens.min((item + 1) * SUM_ITEM_TOKENS) {
                // SAFETY: token t's output row belongs to this item alone
                let out = unsafe { sum_out.slice(t * hidden, hidden) };
                for &r in &by_token[starts[t]..starts[t + 1]] {
                    for (o, v) in out.iter_mut().zip(&routed[r * hidden..][..hidden]) {
                        *o += routes[r].weight * v;
                    }
                }
            }
        });
        SCRATCH.set((gated_rows, routed));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPERTS: usize = 4;
    const HIDDEN: usize = 512;
    const INTER: usize = 256;
    const BENCH_EXPERTS: usize = 256;
    const PICKS: usize = 8;

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

    // random blocks whose leading f16 scale, or the row's f32 one, is small and finite
    fn projection_bytes(ty: GgufType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        let size = ty.type_size();
        let meta = ty.row_scale_bytes();
        let row_bytes = meta + cols / QK_K * size;
        let mut bytes = random_bytes(EXPERTS * rows * row_bytes, seed);
        for (r, row) in bytes.chunks_exact_mut(row_bytes).enumerate() {
            let d = 0.002 + 0.0001 * (r % 7) as f32;
            if meta > 0 {
                row[..meta].copy_from_slice(&(d / 64.0).to_le_bytes());
                continue;
            }
            for block in row.chunks_exact_mut(size) {
                block[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
            }
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

    // the expert types each test runs: Unsloth's 35B-A3B mix, and IQ4_KSS throughout (its rows carry a scale)
    const TYPE_SETS: [[GgufType; 2]; 2] = [
        [GgufType::Iq3S, GgufType::Iq4Xs],
        [GgufType::Iq4Kss, GgufType::Iq4Kss],
    ];

    fn check_against_dense(tokens: usize, routes: &[CpuExpertRoute]) -> Result<()> {
        for types in TYPE_SETS {
            check_types_against_dense(types, tokens, routes)?;
        }
        Ok(())
    }

    fn check_types_against_dense(
        [gate_up_ty, down_ty]: [GgufType; 2],
        tokens: usize,
        routes: &[CpuExpertRoute],
    ) -> Result<()> {
        let gate = projection_bytes(gate_up_ty, INTER, HIDDEN, 1);
        let up = projection_bytes(gate_up_ty, INTER, HIDDEN, 2);
        let down = projection_bytes(down_ty, HIDDEN, INTER, 3);
        let experts = CpuExperts {
            gate: CpuExpertProjection {
                ty: gate_up_ty,
                bytes: &gate,
                rows: INTER,
                cols: HIDDEN,
            },
            up: CpuExpertProjection {
                ty: gate_up_ty,
                bytes: &up,
                rows: INTER,
                cols: HIDDEN,
            },
            down: CpuExpertProjection {
                ty: down_ty,
                bytes: &down,
                rows: HIDDEN,
                cols: INTER,
            },
            act: GluActivationType::Silu,
        };
        let xs = (0..tokens * HIDDEN)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) / 24.0)
            .collect::<Vec<_>>();
        let mut out = vec![0f32; xs.len()];
        experts.forward(&xs, routes, &mut out)?;

        let mut expected = vec![0f32; xs.len()];
        for route in routes {
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
        for (t, (out, expected)) in out
            .as_chunks::<HIDDEN>()
            .0
            .iter()
            .zip(expected.as_chunks::<HIDDEN>().0)
            .enumerate()
        {
            let norm = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
            let err = out
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            // both activations are quantized to Q8_K on this path
            assert!(
                err < 0.02 * norm,
                "{gate_up_ty:?} / {down_ty:?} token {t}: relative error {}",
                err / norm
            );
        }
        Ok(())
    }

    #[test]
    fn routed_forward_matches_dense_experts() -> Result<()> {
        let route = |token, expert, weight| CpuExpertRoute {
            token,
            expert,
            weight,
        };
        check_against_dense(2, &[route(0, 3, 0.7), route(0, 1, 0.3), route(1, 3, 1.0)])
    }

    // expert 2 gets enough tokens to run from a decoded tile, with an odd count so the last quad is short
    #[test]
    fn tiled_forward_matches_dense_experts() -> Result<()> {
        const TOKENS: usize = 2 * TILE_MIN_TOKENS + 1;
        let mut routes = (0..TOKENS)
            .map(|token| CpuExpertRoute {
                token,
                expert: 2,
                weight: 0.6,
            })
            .collect::<Vec<_>>();
        routes.extend((0..3).map(|token| CpuExpertRoute {
            token: 2 * token,
            expert: 0,
            weight: 0.4,
        }));
        check_against_dense(TOKENS, &routes)
    }

    // a 35B-A3B layer's experts (2048 x 512, IQ3_S gate/up, IQ4_XS down), all 256 of them so the weights come from RAM
    fn bench_layer(run: impl Fn(&CpuExperts, usize) -> (usize, usize)) -> Result<()> {
        let (hidden, inter) = (2048, 512);
        let bytes = |ty: GgufType, rows: usize, cols: usize, seed| {
            random_bytes(BENCH_EXPERTS * rows * cols / QK_K * ty.type_size(), seed)
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
        let start = std::time::Instant::now();
        let (iters, routes) = run(&experts, hidden);
        let secs = start.elapsed().as_secs_f64();
        let per_expert = (gate.len() + up.len() + down.len()) / BENCH_EXPERTS;
        println!(
            "{:.1} us a layer, {:.1} GB/s of expert rows, {:.0} GOPS",
            secs / iters as f64 * 1e6,
            (iters * routes * per_expert) as f64 / secs / 1e9,
            (iters * routes * 6 * hidden * inter) as f64 / secs / 1e9
        );
        Ok(())
    }

    #[test]
    #[ignore = "benchmark"]
    fn decode_layer_bandwidth() -> Result<()> {
        const ITERS: usize = 400;
        bench_layer(|experts, hidden| {
            let xs = vec![0.01f32; hidden];
            let mut out = vec![0f32; hidden];
            for iter in 0..ITERS {
                let routes = (0..PICKS)
                    .map(|k| CpuExpertRoute {
                        token: 0,
                        expert: (iter * 37 + k * 31) % BENCH_EXPERTS,
                        weight: 0.1,
                    })
                    .collect::<Vec<_>>();
                experts.forward(&xs, &routes, &mut out).unwrap();
            }
            (ITERS, PICKS)
        })
    }

    fn bench_prefill(expert: impl Fn(usize) -> usize) -> Result<()> {
        const ITERS: usize = 5;
        const TOKENS: usize = 512;
        bench_layer(|experts, hidden| {
            let xs = (0..TOKENS * hidden)
                .map(|i| ((i * 29 % 97) as f32 - 48.0) / 240.0)
                .collect::<Vec<_>>();
            let mut out = vec![0f32; xs.len()];
            let routes = (0..TOKENS * PICKS)
                .map(|i| CpuExpertRoute {
                    token: i / PICKS,
                    expert: expert(i),
                    weight: 0.1,
                })
                .collect::<Vec<_>>();
            for _ in 0..ITERS {
                experts.forward(&xs, &routes, &mut out).unwrap();
            }
            (ITERS, routes.len())
        })
    }

    // a 512-token prompt, 8 picks a token spread evenly over the experts
    #[test]
    #[ignore = "benchmark"]
    fn prefill_layer_rate() -> Result<()> {
        bench_prefill(|i| (i * 37 + i / PICKS * 11) % BENCH_EXPERTS)
    }

    // the same with the picks piled onto a few experts (one gets about 16% of them)
    #[test]
    #[ignore = "benchmark"]
    fn prefill_skewed_layer_rate() -> Result<()> {
        bench_prefill(|i| {
            let u = ((i * 2_654_435_761) % 65_536) as f64 / 65_536.0;
            (BENCH_EXPERTS as f64 * u * u * u) as usize
        })
    }
}
