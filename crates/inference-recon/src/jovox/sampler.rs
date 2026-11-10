//! JoVox's discrete sampler over its two-level token pyramid: 64 root tokens (the 4^3 cells, each holding the 16^3
//! occupancy of its 8 children as 8 byte-valued slots) and a token per occupied 16^3 cell (the 64^3 occupancy below
//! it, the same way). Each step remasks the earliest-revealed slots, draws every slot from one forward and reveals a
//! share of the masked ones, as the schedule's plan says.

use rand::Rng;
use rand::rngs::StdRng;

/// A slot not yet drawn.
pub const MASK: u16 = 256;
/// Root tokens: the 4^3 cells.
pub const ROOT_TOKENS: usize = 64;
/// Second-level tokens: the 16^3 cells.
pub const CELL_TOKENS: usize = 4096;
pub const SLOTS: usize = 8;
// the root tokens sit at level 2 (4^3), the cell tokens at level 4 (16^3), the output at level 6 (64^3)
const ROOT_LEVEL: u32 = 2;
const CELL_LEVEL: u32 = 4;
pub const OUTPUT_SIDE: usize = 64;
// the remask tie-break: a random share of a step, so slots revealed in one step leave in random order
const TIE_BREAK: f32 = 0.5;

/// The training schedule: `kappa0 = min(t / e, 1)^p` for the root level, `kappa1 = t^p` for the cells.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub power: f64,
    pub cleanup_steps: usize,
    pub remask_steps: usize,
}

impl Schedule {
    /// Per step `[q0, q1, h0, h1]`: the share of known slots each level keeps, then the share of masked ones it
    /// reveals, over `steps` grid steps ending the root window at `root_end`, then `cleanup_steps` final steps.
    pub fn plan(&self, root_end: f64, steps: usize, phase: f64) -> Vec<[f64; 4]> {
        let n = steps as f64;
        let mut grid = vec![0.0];
        grid.extend((1..=steps).map(|i| ((i as f64 - 1.0 + phase) / n).clamp(0.0, 1.0)));
        grid.push(1.0);
        grid.sort_by(f64::total_cmp);
        grid.dedup();
        let p = self.power;
        let k0 = |x: f64| (x.max(0.0) / root_end).min(1.0).powf(p);
        let k1 = |x: f64| x.clamp(0.0, 1.0).powf(p);
        let last = grid.len() - 2;
        let mut plan = Vec::with_capacity(grid.len() - 1 + self.cleanup_steps);
        for i in 0..=last {
            let (t, b) = (grid[i], grid[i + 1]);
            let s = if i >= 1 && self.remask_steps > 0 {
                grid[i.saturating_sub(self.remask_steps)]
            } else {
                t
            };
            let (k0t, k0s) = (k0(t), k0(s));
            let q0 = if k0t > 0.0 && k0t < 1.0 && s < t {
                k0s / k0t
            } else {
                1.0
            };
            let h0 = if k0s >= 1.0 {
                1.0
            } else {
                (k0(b) - k0s) / (1.0 - k0s)
            };
            let (k1t, k1s) = (k1(t), k1(s));
            let q1 = if k1t > 0.0 && s < t { k1s / k1t } else { 1.0 };
            let h1 = if i == last {
                1.0 / (self.cleanup_steps + 1) as f64
            } else {
                (k1(b) - k1s) / (1.0 - k1s)
            };
            plan.push([q0, q1, h0, h1]);
        }
        plan.extend(
            (0..self.cleanup_steps).map(|j| [1.0, 1.0, 0.0, 1.0 / (self.cleanup_steps - j) as f64]),
        );
        plan
    }
}

/// `xyz` of `key` at `level` (`x << 2L | y << L | z`).
pub fn unkey(key: usize, level: u32) -> [usize; 3] {
    let m = (1 << level) - 1;
    [(key >> (2 * level)) & m, (key >> level) & m, key & m]
}

fn key([x, y, z]: [usize; 3], level: u32) -> usize {
    (x << (2 * level)) | (y << level) | z
}

fn octant([x, y, z]: [usize; 3]) -> usize {
    ((x & 1) << 2) | ((y & 1) << 1) | (z & 1)
}

fn octant_offset(o: usize) -> [usize; 3] {
    [(o >> 2) & 1, (o >> 1) & 1, o & 1]
}

/// The tokens one forward sees: each one's level (0 root, 1 cell), its cell `xyz` at that level and its slots.
pub struct Tokens {
    pub levels: Vec<u8>,
    pub coords: Vec<[usize; 3]>,
    pub slots: Vec<[u16; SLOTS]>,
    /// The known share of each level's slots, the progress the model is conditioned on
    pub ratios: [f32; 2],
}

/// One slot level's values, when each was revealed (-1 masked) and whether it may be remasked.
struct Level {
    val: Vec<u16>,
    born: Vec<i64>,
    lock: Vec<bool>,
}

impl Level {
    fn new(tokens: usize) -> Self {
        Self {
            val: vec![MASK; tokens * SLOTS],
            born: vec![-1; tokens * SLOTS],
            lock: vec![false; tokens * SLOTS],
        }
    }

    fn mask(&mut self, i: usize) {
        self.val[i] = MASK;
        self.born[i] = -1;
    }
}

struct State {
    root: Level,
    cells: Level,
    // per 16^3 cell: the root slot holding its bit and which bit
    parent: Vec<(usize, usize)>,
}

impl State {
    fn new() -> Self {
        let parent = (0..CELL_TOKENS)
            .map(|c| {
                let xyz = unkey(c, CELL_LEVEL);
                let root = key(xyz.map(|v| v >> 2), ROOT_LEVEL);
                (root * SLOTS + octant(xyz.map(|v| v >> 1)), octant(xyz))
            })
            .collect();
        Self {
            root: Level::new(ROOT_TOKENS),
            cells: Level::new(CELL_TOKENS),
            parent,
        }
    }

    // the 16^3 cells the known root slots say are occupied
    fn exist(&self) -> Vec<bool> {
        self.parent
            .iter()
            .map(|&(slot, bit)| {
                let v = self.root.val[slot];
                v != MASK && (v >> bit) & 1 == 1
            })
            .collect()
    }

    fn pending(&self, exist: &[bool]) -> bool {
        self.root.val.contains(&MASK)
            || exist
                .iter()
                .enumerate()
                .any(|(c, &e)| e && self.cells.val[c * SLOTS..][..SLOTS].contains(&MASK))
    }

    fn ratios(&self, exist: &[bool]) -> [f32; 2] {
        let root = self.root.val.iter().filter(|&&v| v != MASK).count() as f32
            / (ROOT_TOKENS * SLOTS) as f32;
        let existing = exist.iter().filter(|&&e| e).count();
        let known = exist
            .iter()
            .enumerate()
            .filter(|&(_, &e)| e)
            .map(|(c, _)| {
                self.cells.val[c * SLOTS..][..SLOTS]
                    .iter()
                    .filter(|&&v| v != MASK)
                    .count()
            })
            .sum::<usize>();
        [root, known as f32 / (existing * SLOTS).max(1) as f32]
    }

    fn tokens(&self, exist: &[bool]) -> Tokens {
        let mut tokens = Tokens {
            levels: Vec::new(),
            coords: Vec::new(),
            slots: Vec::new(),
            ratios: self.ratios(exist),
        };
        let slots = |level: &Level, t: usize| -> [u16; SLOTS] {
            level.val[t * SLOTS..][..SLOTS].try_into().expect("8 slots")
        };
        for t in 0..ROOT_TOKENS {
            tokens.levels.push(0);
            tokens.coords.push(unkey(t, ROOT_LEVEL));
            tokens.slots.push(slots(&self.root, t));
        }
        for (c, _) in exist.iter().enumerate().filter(|&(_, &e)| e) {
            tokens.levels.push(1);
            tokens.coords.push(unkey(c, CELL_LEVEL));
            tokens.slots.push(slots(&self.cells, c));
        }
        tokens
    }
}

/// Uniform draws in [0, 1), a whole level's worth at a time, as the reference draws `torch.rand(shape)`.
pub trait Uniform {
    fn fill(&mut self, n: usize) -> Vec<f32>;
}

impl Uniform for StdRng {
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.random::<f32>()).collect()
    }
}

// The `count` slots of `candidates` with the smallest keys
fn smallest(candidates: impl Iterator<Item = (f32, usize)>, count: usize) -> Vec<usize> {
    let mut keyed: Vec<(f32, usize)> = candidates.collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    keyed.into_iter().take(count).map(|(_, i)| i).collect()
}

// Remasks `round(known * (1 - keep))` of the eligible slots, earliest revealed first, ties in random order
fn erase(level: &mut Level, eligible: &[bool], known: usize, keep: f64, rng: &mut dyn Uniform) {
    let count = (known as f64 * (1.0 - keep)).round_ties_even() as usize;
    if count == 0 {
        return;
    }
    let noise = rng.fill(eligible.len());
    let born = &level.born;
    let picked = smallest(
        eligible
            .iter()
            .enumerate()
            .filter(|&(_, &e)| e)
            .map(|(i, _)| (born[i] as f32 + noise[i] * TIE_BREAK, i)),
        count,
    );
    for i in picked {
        level.mask(i);
    }
}

// Picks `ceil(masked * share)` (at least one) of the masked slots at random
fn reveal(masked: &[bool], share: f64, rng: &mut dyn Uniform) -> Vec<usize> {
    let noise = rng.fill(masked.len());
    let total = masked.iter().filter(|&&m| m).count();
    if total == 0 {
        return Vec::new();
    }
    let count = ((total as f64 * share).ceil() as usize).max(1);
    smallest(
        masked
            .iter()
            .enumerate()
            .filter(|&(_, &m)| m)
            .map(|(i, _)| (noise[i], i)),
        count,
    )
}

/// Runs `plan`, drawing every slot of the tokens `draw` is handed (one value per slot, token by token), and returns
/// the occupied `xyz` cells of the 64^3 grid, in key order.
pub fn run(
    plan: &[[f64; 4]],
    rng: &mut dyn Uniform,
    draw: &mut dyn FnMut(&Tokens) -> inference_tensor::Result<Vec<[u16; SLOTS]>>,
) -> inference_tensor::Result<Vec<[usize; 3]>> {
    let mut st = State::new();
    for (k, &row) in plan.iter().enumerate() {
        let k = k as i64;
        // the plan crosses as f32, as the reference's step table is
        let [q0, q1, h0, h1] = row.map(|v| v as f32 as f64);
        if !st.pending(&st.exist()) {
            break;
        }
        let root_known: Vec<bool> = st.root.val.iter().map(|&v| v != MASK).collect();
        let eligible: Vec<bool> = (0..root_known.len())
            .map(|i| root_known[i] && st.root.born[i] < k - 1 && !st.root.lock[i])
            .collect();
        let known = root_known.iter().filter(|&&v| v).count();
        erase(&mut st.root, &eligible, known, q0, rng);

        let exist = st.exist();
        for (c, &e) in exist.iter().enumerate() {
            if !e {
                for i in c * SLOTS..(c + 1) * SLOTS {
                    st.cells.mask(i);
                    st.cells.lock[i] = false;
                }
            }
        }
        let cell_known: Vec<bool> = st.cells.val.iter().map(|&v| v != MASK).collect();
        let eligible: Vec<bool> = (0..cell_known.len())
            .map(|i| cell_known[i] && st.cells.born[i] < k - 1 && !st.cells.lock[i])
            .collect();
        let known = cell_known.iter().filter(|&&v| v).count();
        erase(&mut st.cells, &eligible, known, q1, rng);

        let tokens = st.tokens(&exist);
        let drawn = draw(&tokens)?;
        let root_masked: Vec<bool> = st.root.val.iter().map(|&v| v == MASK).collect();
        let cell_masked: Vec<bool> = (0..st.cells.val.len())
            .map(|i| st.cells.val[i] == MASK && exist[i / SLOTS])
            .collect();
        let root_revealed = reveal(&root_masked, h0, rng);
        let cell_revealed = reveal(&cell_masked, h1, rng);
        // the cell tokens follow the roots in `tokens`, in cell order
        let mut row_of = vec![usize::MAX; CELL_TOKENS];
        for (row, (c, _)) in exist.iter().enumerate().filter(|&(_, &e)| e).enumerate() {
            row_of[c] = ROOT_TOKENS + row;
        }
        let single = root_revealed.len() + cell_revealed.len() == 1;
        for &i in &root_revealed {
            st.root.val[i] = drawn[i / SLOTS][i % SLOTS];
            st.root.born[i] = k;
            st.root.lock[i] |= single;
        }
        for &i in &cell_revealed {
            st.cells.val[i] = drawn[row_of[i / SLOTS]][i % SLOTS];
            st.cells.born[i] = k;
            st.cells.lock[i] |= single;
        }
    }
    let exist = st.exist();
    if st.pending(&exist) {
        inference_tensor::bail!("masked slots remain after the last step");
    }
    let mut out = Vec::new();
    for (c, _) in exist.iter().enumerate().filter(|&(_, &e)| e) {
        let cell = unkey(c, CELL_LEVEL);
        for slot in 0..SLOTS {
            let v = st.cells.val[c * SLOTS + slot] as usize;
            let mid = octant_offset(slot);
            for bit in (0..SLOTS).filter(|b| (v >> b) & 1 == 1) {
                let fine = octant_offset(bit);
                out.push(std::array::from_fn(|a| {
                    (cell[a] * 2 + mid[a]) * 2 + fine[a]
                }));
            }
        }
    }
    out.sort_by_key(|&xyz| key(xyz, 6));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    const STEPS: usize = 36;

    fn plan() -> Vec<[f64; 4]> {
        Schedule {
            power: 2.0,
            cleanup_steps: 8,
            remask_steps: 1,
        }
        .plan(0.8, STEPS, 1.0)
    }

    fn sample(value: u16) -> Vec<[usize; 3]> {
        let mut rng = StdRng::seed_from_u64(0);
        run(&plan(), &mut rng, &mut |tokens| {
            Ok(vec![[value; SLOTS]; tokens.slots.len()])
        })
        .expect("the plan reveals every slot")
    }

    #[test]
    fn full_slots_fill_the_grid_and_empty_ones_leave_it_empty() {
        assert_eq!(sample(255).len(), OUTPUT_SIDE.pow(3));
        assert!(sample(0).is_empty());
    }

    #[test]
    fn a_single_child_bit_lands_in_each_cells_first_corner() {
        // every slot holds only bit 0, the (0, 0, 0) child: one 16^3 cell per 8^3 cell, then one voxel per 32^3 cell
        let voxels = sample(1);
        assert_eq!(voxels.len(), (OUTPUT_SIDE / 4).pow(3));
        assert!(voxels.iter().all(|v| v.iter().all(|&c| c % 2 == 0)));
    }

    #[test]
    fn cell_tokens_appear_only_under_occupied_root_slots() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut most = 0;
        run(&plan(), &mut rng, &mut |tokens| {
            // the root's only occupied child is the first 16^3 cell of each 8^3 cell
            for (level, xyz) in tokens.levels.iter().zip(&tokens.coords) {
                if *level == 1 {
                    assert!(
                        xyz.iter().all(|&c| c % 2 == 0),
                        "cell {xyz:?} has no parent bit"
                    );
                }
            }
            most = most.max(tokens.slots.len() - ROOT_TOKENS);
            Ok(vec![[1; SLOTS]; tokens.slots.len()])
        })
        .expect("the plan reveals every slot");
        assert_eq!(most, CELL_TOKENS / 8);
    }
}
