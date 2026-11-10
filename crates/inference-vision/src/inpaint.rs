//! OpenCV's Telea inpainting (`cv::inpaint` with `INPAINT_TELEA`, OpenCV 5's `photo/src/inpaint.cpp`), ported to
//! match it: a fast-marching front moves into the unknown region from its one-pixel band, and each pixel it reaches
//! takes a weighted mean of the known pixels within `radius` (direction, distance and level-set weights), plus their
//! gradient term. OpenCV's quirks are kept (central image differences doubled, `+0.5` before rounding to even).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

const KNOWN: u8 = 0;
const BAND: u8 = 1;
const INSIDE: u8 = 2;
const CHANGE: u8 = 3;
const FAR: f32 = 1.0e6;
const MAX_RADIUS: i32 = 100;
const EPS: f32 = 1.0e-20;
const MIN_DIR: f64 = 0.01;
const TINY_DIR: f32 = 0.000001;

#[derive(PartialEq)]
struct Elem {
    t: f32,
    order: u32,
    i: i32,
    j: i32,
}

impl Eq for Elem {}

impl Ord for Elem {
    // reversed: BinaryHeap pops the largest, OpenCV the smallest T, ties first in
    fn cmp(&self, o: &Self) -> Ordering {
        o.t.total_cmp(&self.t).then(o.order.cmp(&self.order))
    }
}

impl PartialOrd for Elem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[derive(Default)]
struct Queue {
    heap: BinaryHeap<Elem>,
    next: u32,
}

impl Queue {
    fn push(&mut self, i: i32, j: i32, t: f32) {
        self.heap.push(Elem {
            t,
            order: self.next,
            i,
            j,
        });
        self.next += 1;
    }
    fn pop(&mut self) -> Option<(i32, i32)> {
        self.heap.pop().map(|e| (e.i, e.j))
    }
}

// the padded grids (one pixel of border), indexed (row, col)
struct Grid<T> {
    data: Vec<T>,
    cols: i32,
}

impl<T: Copy> Grid<T> {
    fn at(&self, i: i32, j: i32) -> T {
        self.data[(i * self.cols + j) as usize]
    }
    fn set(&mut self, i: i32, j: i32, v: T) {
        self.data[(i * self.cols + j) as usize] = v;
    }
}

fn solve(i1: i32, j1: i32, i2: i32, j2: i32, f: &Grid<u8>, t: &Grid<f32>) -> f32 {
    let (a11, a22) = (f64::from(t.at(i1, j1)), f64::from(t.at(i2, j2)));
    let m12 = a11.min(a22);
    let sol = match (f.at(i1, j1) != INSIDE, f.at(i2, j2) != INSIDE) {
        (true, true) if (a11 - a22).abs() >= 1.0 => 1.0 + m12,
        (true, true) => (a11 + a22 + (2.0 - (a11 - a22) * (a11 - a22)).sqrt()) * 0.5,
        (true, false) => 1.0 + a11,
        (false, true) => 1.0 + a22,
        (false, false) => 1.0 + m12,
    };
    sol as f32
}

fn arrival(i: i32, j: i32, f: &Grid<u8>, t: &Grid<f32>) -> f32 {
    let a = solve(i - 1, j, i, j - 1, f, t).min(solve(i + 1, j, i, j - 1, f, t));
    let b = solve(i - 1, j, i, j + 1, f, t).min(solve(i + 1, j, i, j + 1, f, t));
    a.min(b)
}

const NEIGHBORS: [(i32, i32); 4] = [(-1, 0), (0, -1), (1, 0), (0, 1)];

// the distances outward from the band into `f`'s INSIDE ring, negated
fn march_out(f: &mut Grid<u8>, t: &mut Grid<f32>, queue: &mut Queue, rows: i32, cols: i32) {
    while let Some((ii, jj)) = queue.pop() {
        f.set(ii, jj, CHANGE);
        for (di, dj) in NEIGHBORS {
            let (i, j) = (ii + di, jj + dj);
            if i <= 0 || j <= 0 || i > rows || j > cols {
                continue;
            }
            if f.at(i, j) == INSIDE {
                let dist = arrival(i, j, f, t);
                t.set(i, j, dist);
                f.set(i, j, BAND);
                queue.push(i, j, dist);
            }
        }
    }
    for (fv, tv) in f.data.iter_mut().zip(&mut t.data) {
        if *fv == CHANGE {
            *fv = KNOWN;
            *tv = -*tv;
        }
    }
}

// one-sided or central difference of `t` along an axis, zero when both sides are unknown
fn grad_t(f: &Grid<u8>, t: &Grid<f32>, i: i32, j: i32, di: i32, dj: i32) -> f32 {
    let (fp, fm) = (
        f.at(i + di, j + dj) != INSIDE,
        f.at(i - di, j - dj) != INSIDE,
    );
    match (fp, fm) {
        (true, true) => (t.at(i + di, j + dj) - t.at(i - di, j - dj)) * 0.5,
        (true, false) => t.at(i + di, j + dj) - t.at(i, j),
        (false, true) => t.at(i, j) - t.at(i - di, j - dj),
        (false, false) => 0.0,
    }
}

/// Fills the pixels of `image` (`height` x `width` x `channels`, row-major u8) where `unknown` is set, as
/// `cv2.inpaint(image, unknown, radius, cv2.INPAINT_TELEA)` does.
pub fn inpaint_telea(
    image: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    unknown: &[bool],
    radius: f64,
) {
    let [schedule] = telea_schedules(width, height, unknown, [radius]);
    schedule.inpaint(image, channels);
}

/// The fast march of [`inpaint_telea`] for one mask and radius, reusable across images: when each pixel enters the
/// band, its arrival time and level-set gradient there, and the pixels grouped into levels whose values depend only on
/// earlier levels. A pixel's value reads only pixels that entered before it, at their final values and times, so a
/// level is filled in parallel with the sequential port's arithmetic and result.
pub struct TeleaSchedule {
    front: Arc<Front>,
    range: i32,
    // the arrival times: the inward march's inside the region, the outward march's (negated) outside it
    t: Vec<f32>,
    // entry indices by level, and each level's end in it
    by_level: Vec<u32>,
    level_ends: Vec<usize>,
}

// the inward march, which does not depend on the radius
struct Front {
    width: usize,
    erows: i32,
    ecols: i32,
    // per padded pixel: 0 when it never was INSIDE, its entry index + 1 when it entered, NEVER when it never did
    entry: Vec<u32>,
    // per entry: the pixel (padded) and its gradient of T
    entered: Vec<(i32, i32, (f32, f32))>,
    t: Vec<f32>,
}

const NEVER: u32 = u32::MAX;
const MAX_CHANNELS: usize = 3;

// the known pixels within `reach` of an INSIDE one (4-neighbours only when `cross`), border excluded
fn dilated(mask: &Grid<u8>, erows: i32, reach: i32, cross: bool) -> Vec<bool> {
    use rayon::prelude::*;
    let ecols = mask.cols;
    let mut out = vec![false; (erows * ecols) as usize];
    out.par_chunks_mut(ecols as usize)
        .enumerate()
        .for_each(|(i, row)| {
            let i = i as i32;
            for (j, o) in row.iter_mut().enumerate() {
                let j = j as i32;
                if i == 0 || j == 0 || i == erows - 1 || j == ecols - 1 || mask.at(i, j) == INSIDE {
                    continue;
                }
                'near: for di in -reach..=reach {
                    for dj in -reach..=reach {
                        if cross && di != 0 && dj != 0 {
                            continue;
                        }
                        let (k, l) = (i + di, j + dj);
                        if k >= 0 && l >= 0 && k < erows && l < ecols && mask.at(k, l) == INSIDE {
                            *o = true;
                            break 'near;
                        }
                    }
                }
            }
        });
    out
}

/// [`TeleaSchedule`]s for one mask at each of `radii`, sharing the inward march; the outward marches (one per
/// radius) run beside it.
pub fn telea_schedules<const N: usize>(
    width: usize,
    height: usize,
    unknown: &[bool],
    radii: [f64; N],
) -> [TeleaSchedule; N] {
    use rayon::prelude::*;
    let (rows, cols) = (height as i32, width as i32);
    let (erows, ecols) = (rows + 2, cols + 2);
    let size = (erows * ecols) as usize;
    let mut mask = Grid {
        data: vec![KNOWN; size],
        cols: ecols,
    };
    for i in 0..rows {
        for j in 0..cols {
            if unknown[(i * cols + j) as usize] {
                mask.set(i + 1, j + 1, INSIDE);
            }
        }
    }
    let band = dilated(&mask, erows, 1, true);
    let ranges = radii.map(|r| (r.round_ties_even() as i32).clamp(1, MAX_RADIUS));
    let outward = |range: i32| -> Vec<f32> {
        let ring = dilated(&mask, erows, range, false);
        let mut t = Grid {
            data: vec![FAR; size],
            cols: ecols,
        };
        let mut queue = Queue::default();
        let mut out = Grid {
            data: vec![KNOWN; size],
            cols: ecols,
        };
        for (p, (&b, &r)) in band.iter().zip(&ring).enumerate() {
            if b {
                queue.push(p as i32 / ecols, p as i32 % ecols, 0.0);
                t.data[p] = 0.0;
            } else if r {
                out.data[p] = INSIDE;
            }
        }
        march_out(&mut out, &mut t, &mut queue, rows, cols);
        t.data
    };
    let inward = || -> Front {
        let mut t = Grid {
            data: vec![FAR; size],
            cols: ecols,
        };
        let mut heap = Queue::default();
        for (p, &b) in band.iter().enumerate() {
            if b {
                heap.push(p as i32 / ecols, p as i32 % ecols, 0.0);
                // the outward march leaves the band at -0
                t.data[p] = -0.0;
            }
        }
        let mut entry: Vec<u32> = mask
            .data
            .iter()
            .map(|&m| if m == INSIDE { NEVER } else { 0 })
            .collect();
        let mut f = Grid {
            data: mask.data.clone(),
            cols: ecols,
        };
        let mut entered = Vec::new();
        while let Some((ii, jj)) = heap.pop() {
            f.set(ii, jj, KNOWN);
            for (di, dj) in NEIGHBORS {
                let (i, j) = (ii + di, jj + dj);
                if i <= 0 || j <= 0 || i > erows - 1 || j > ecols - 1 || f.at(i, j) != INSIDE {
                    continue;
                }
                let dist = arrival(i, j, &f, &t);
                t.set(i, j, dist);
                let gt = (grad_t(&f, &t, i, j, 0, 1), grad_t(&f, &t, i, j, 1, 0));
                entered.push((i, j, gt));
                entry[(i * ecols + j) as usize] = entered.len() as u32;
                f.set(i, j, BAND);
                heap.push(i, j, dist);
            }
        }
        Front {
            width,
            erows,
            ecols,
            entry,
            entered,
            t: t.data,
        }
    };
    let (front, outs) = rayon::join(inward, || {
        ranges.par_iter().map(|&r| outward(r)).collect::<Vec<_>>()
    });
    let front = Arc::new(front);
    let schedules: Vec<TeleaSchedule> = ranges
        .par_iter()
        .zip(outs)
        .map(|(&range, mut t)| {
            for (p, tv) in t.iter_mut().enumerate() {
                if mask.data[p] == INSIDE {
                    *tv = front.t[p];
                }
            }
            TeleaSchedule::new(front.clone(), range, t)
        })
        .collect();
    match schedules.try_into() {
        Ok(schedules) => schedules,
        Err(_) => unreachable!("one schedule per radius"),
    }
}

impl TeleaSchedule {
    fn new(front: Arc<Front>, range: i32, t: Vec<f32>) -> Self {
        let (erows, ecols) = (front.erows, front.ecols);
        // a pixel's level: one past the deepest pixel it reads (its window and the window's 4-neighbours) that
        // entered before it; pixels that never were INSIDE are level 0
        let reach = range + 1;
        let mut level = vec![0u32; front.entered.len()];
        for (n, &(i, j, _)) in front.entered.iter().enumerate() {
            let e = n as u32 + 1;
            let mut deepest = 0;
            for k in (i - reach).max(0)..=(i + reach).min(erows - 1) {
                for l in (j - reach).max(0)..=(j + reach).min(ecols - 1) {
                    let q = front.entry[(k * ecols + l) as usize];
                    if q != 0 && q < e {
                        deepest = deepest.max(level[q as usize - 1]);
                    }
                }
            }
            level[n] = deepest + 1;
        }
        let levels = level.iter().copied().max().unwrap_or(0) as usize;
        let mut level_ends = vec![0usize; levels + 1];
        for &l in &level {
            level_ends[l as usize] += 1;
        }
        for l in 1..=levels {
            level_ends[l] += level_ends[l - 1];
        }
        let mut cursor = level_ends.clone();
        let mut by_level = vec![0u32; level.len()];
        for (n, &l) in level.iter().enumerate() {
            cursor[l as usize - 1] += 1;
            by_level[cursor[l as usize - 1] - 1] = n as u32;
        }
        Self {
            front,
            range,
            t,
            by_level,
            level_ends,
        }
    }

    /// Fills `image`'s unknown pixels (`channels` per pixel, at most 3), level by level.
    pub fn inpaint(&self, image: &mut [u8], channels: usize) {
        use rayon::prelude::*;
        assert!(channels <= MAX_CHANNELS, "at most {MAX_CHANNELS} channels");
        let width = self.front.width;
        let mut start = 0;
        let mut values = Vec::new();
        for &end in &self.level_ends[1..] {
            self.by_level[start..end]
                .par_iter()
                .map(|&n| self.value(image, channels, n as usize))
                .collect_into_vec(&mut values);
            for (&n, v) in self.by_level[start..end].iter().zip(&values) {
                let (i, j, _) = self.front.entered[n as usize];
                let p = ((i - 1) as usize * width + (j - 1) as usize) * channels;
                image[p..p + channels].copy_from_slice(&v[..channels]);
            }
            start = end;
        }
    }

    // entry `n`'s value, as the sequential port computes it when the pixel enters the band
    fn value(&self, image: &[u8], channels: usize, n: usize) -> [u8; MAX_CHANNELS] {
        let front = &*self.front;
        let (i, j, gt) = front.entered[n];
        let e = n as u32 + 1;
        let (erows, ecols, range) = (front.erows, front.ecols, self.range);
        let cols = ecols - 2;
        // whether a pixel was outside the unknown region when entry `n` entered (KNOWN or BAND)
        let known = |k: i32, l: i32| front.entry[(k * ecols + l) as usize] < e;
        let t = |k: i32, l: i32| self.t[(k * ecols + l) as usize];
        let px = |k: i32, l: i32| (k * cols + l) as usize * channels;
        let mut ia = [0f32; MAX_CHANNELS];
        let mut jx = [0f32; MAX_CHANNELS];
        let mut jy = [0f32; MAX_CHANNELS];
        let mut s = [EPS; MAX_CHANNELS];
        for k in i - range..=i + range {
            let km = k - 1 + i32::from(k == 1);
            let kp = k - 1 - i32::from(k == erows - 2);
            for l in j - range..=j + range {
                let lm = l - 1 + i32::from(l == 1);
                let lp = l - 1 - i32::from(l == ecols - 2);
                if !(k > 0 && l > 0 && k < erows - 1 && l < ecols - 1) {
                    continue;
                }
                if !known(k, l) || (l - j) * (l - j) + (k - i) * (k - i) > range * range {
                    continue;
                }
                let r = ((j - l) as f32, (i - k) as f32);
                let len = r.0 * r.0 + r.1 * r.1;
                // OpenCV's 3-channel path takes this root in double, its 1-channel path in float
                let dst = if channels == 3 {
                    (1.0 / (f64::from(len) * f64::from(len).sqrt())) as f32
                } else {
                    (1.0 / f64::from(len * len.sqrt())) as f32
                };
                // OpenCV's unqualified `fabs` takes the C library's double one, so 1 + |dt| is not rounded to float
                let lev = (1.0 / (1.0 + f64::from((t(k, l) - t(i, j)).abs()))) as f32;
                let mut dir = r.0 * gt.0 + r.1 * gt.1;
                if f64::from(dir.abs()) <= MIN_DIR {
                    dir = TINY_DIR;
                }
                let w = (dst * lev * dir).abs();
                let (fr, fl) = (known(k, l + 1), known(k, l - 1));
                let (fd, fu) = (known(k + 1, l), known(k - 1, l));
                for c in 0..channels {
                    let v = |a: i32, b: i32| f32::from(image[px(a, b) + c]);
                    let gx = match (fr, fl) {
                        (true, true) => (v(km, lp + 1) - v(km, lm - 1)) * 2.0,
                        (true, false) => v(km, lp + 1) - v(km, lm),
                        (false, true) => v(km, lp) - v(km, lm - 1),
                        (false, false) => 0.0,
                    };
                    let gy = match (fd, fu) {
                        (true, true) => (v(kp + 1, lm) - v(km - 1, lm)) * 2.0,
                        (true, false) => v(kp + 1, lm) - v(km, lm),
                        (false, true) => v(kp, lm) - v(km - 1, lm),
                        (false, false) => 0.0,
                    };
                    ia[c] += w * v(k - 1, l - 1);
                    jx[c] -= w * (gx * r.0);
                    jy[c] -= w * (gy * r.1);
                    s[c] += w;
                }
            }
        }
        std::array::from_fn(|c| {
            if c >= channels {
                return 0;
            }
            let sat =
                ia[c] / s[c] + (jx[c] + jy[c]) / ((jx[c] * jx[c] + jy[c] * jy[c]).sqrt() + EPS);
            (f64::from(sat) + 0.5).round_ties_even().clamp(0.0, 255.0) as u8
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hole_in_a_flat_image_fills_as_opencv_does() {
        let (w, h) = (16, 12);
        let mut image = vec![90u8; w * h * 3];
        let mut unknown = vec![false; w * h];
        for y in 3..8 {
            for x in 4..11 {
                unknown[y * w + x] = true;
                image[(y * w + x) * 3..][..3].fill(0);
            }
        }
        inpaint_telea(&mut image, w, h, 3, &unknown, 3.0);
        // cv2 5.0.0 on the same input: its normalized gradient term moves a flat fill by a level or two
        let expected = [
            [90, 90, 90, 90, 90, 90, 90],
            [91, 90, 91, 90, 92, 90, 90],
            [89, 91, 90, 90, 91, 90, 90],
            [92, 90, 90, 89, 89, 90, 90],
            [90, 89, 89, 91, 89, 92, 90],
        ];
        for (y, row) in expected.iter().enumerate() {
            for (x, &v) in row.iter().enumerate() {
                assert_eq!(
                    image[((y + 3) * w + x + 4) * 3..][..3],
                    [v; 3],
                    "at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn a_tie_at_the_border_rounds_as_opencv_does() {
        // cv2 5.0.0 on this case: the right-border pixel lands on 101.5 only with 1 + |dt| kept in double
        let mut image = vec![
            98, 99, 99, 98, 100, 99, 98, 98, 99, 98, 98, 98, 100, 100, 100, 98, 99, 100, 98, 99,
        ];
        let mask = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1];
        let unknown: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
        inpaint_telea(&mut image, 5, 4, 1, &unknown, 1.0);
        assert_eq!(
            image,
            [
                98, 99, 99, 100, 99, 99, 99, 98, 100, 98, 98, 99, 100, 100, 102, 98, 99, 100, 98,
                100
            ]
        );
    }
}
