//! OpenCV's Telea inpainting (`cv::inpaint` with `INPAINT_TELEA`, OpenCV 5's `photo/src/inpaint.cpp`), ported to
//! match it: a fast-marching front moves into the unknown region from its one-pixel band, and each pixel it reaches
//! takes a weighted mean of the known pixels within `radius` (direction, distance and level-set weights), plus their
//! gradient term. OpenCV's quirks are kept (central image differences doubled, `+0.5` before rounding to even).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
    let range = (radius.round_ties_even() as i32).clamp(1, MAX_RADIUS);
    let (rows, cols) = (height as i32, width as i32);
    let (erows, ecols) = (rows + 2, cols + 2);
    let size = (erows * ecols) as usize;
    let border = |i: i32, j: i32| i == 0 || j == 0 || i == erows - 1 || j == ecols - 1;
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
    let dilated = |reach: i32, cross: bool| -> Vec<bool> {
        let mut out = vec![false; size];
        for i in 0..erows {
            for j in 0..ecols {
                if border(i, j) || mask.at(i, j) == INSIDE {
                    continue;
                }
                'near: for di in -reach..=reach {
                    for dj in -reach..=reach {
                        if cross && di != 0 && dj != 0 {
                            continue;
                        }
                        let (k, l) = (i + di, j + dj);
                        if k >= 0 && l >= 0 && k < erows && l < ecols && mask.at(k, l) == INSIDE {
                            out[(i * ecols + j) as usize] = true;
                            break 'near;
                        }
                    }
                }
            }
        }
        out
    };
    let band = dilated(1, true);
    let ring = dilated(range, false);
    let mut t = Grid {
        data: vec![FAR; size],
        cols: ecols,
    };
    let mut heap = Queue::default();
    let mut out_queue = Queue::default();
    let mut out = Grid {
        data: vec![KNOWN; size],
        cols: ecols,
    };
    for (p, (&b, &r)) in band.iter().zip(&ring).enumerate() {
        let (i, j) = (p as i32 / ecols, p as i32 % ecols);
        if b {
            heap.push(i, j, 0.0);
            out_queue.push(i, j, 0.0);
            t.data[p] = 0.0;
        } else if r {
            out.data[p] = INSIDE;
        }
    }
    march_out(&mut out, &mut t, &mut out_queue, rows, cols);
    let f = &mut mask;
    let px = |k: i32, l: i32| (k * cols + l) as usize * channels;
    let mut ia = vec![0f32; channels];
    let mut jx = vec![0f32; channels];
    let mut jy = vec![0f32; channels];
    let mut s = vec![0f32; channels];
    while let Some((ii, jj)) = heap.pop() {
        f.set(ii, jj, KNOWN);
        for (di, dj) in NEIGHBORS {
            let (i, j) = (ii + di, jj + dj);
            if i <= 0 || j <= 0 || i > erows - 1 || j > ecols - 1 || f.at(i, j) != INSIDE {
                continue;
            }
            let dist = arrival(i, j, f, &t);
            t.set(i, j, dist);
            let gt = (grad_t(f, &t, i, j, 0, 1), grad_t(f, &t, i, j, 1, 0));
            ia.fill(0.0);
            jx.fill(0.0);
            jy.fill(0.0);
            s.fill(EPS);
            for k in i - range..=i + range {
                let km = k - 1 + i32::from(k == 1);
                let kp = k - 1 - i32::from(k == erows - 2);
                for l in j - range..=j + range {
                    let lm = l - 1 + i32::from(l == 1);
                    let lp = l - 1 - i32::from(l == ecols - 2);
                    if !(k > 0 && l > 0 && k < erows - 1 && l < ecols - 1) {
                        continue;
                    }
                    if f.at(k, l) == INSIDE || (l - j) * (l - j) + (k - i) * (k - i) > range * range
                    {
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
                    let lev = (1.0 / (1.0 + f64::from((t.at(k, l) - t.at(i, j)).abs()))) as f32;
                    let mut dir = r.0 * gt.0 + r.1 * gt.1;
                    if f64::from(dir.abs()) <= MIN_DIR {
                        dir = TINY_DIR;
                    }
                    let w = (dst * lev * dir).abs();
                    let (fr, fl) = (f.at(k, l + 1) != INSIDE, f.at(k, l - 1) != INSIDE);
                    let (fd, fu) = (f.at(k + 1, l) != INSIDE, f.at(k - 1, l) != INSIDE);
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
            for c in 0..channels {
                let sat =
                    ia[c] / s[c] + (jx[c] + jy[c]) / ((jx[c] * jx[c] + jy[c] * jy[c]).sqrt() + EPS);
                image[px(i - 1, j - 1) + c] =
                    (f64::from(sat) + 0.5).round_ties_even().clamp(0.0, 255.0) as u8;
            }
            f.set(i, j, BAND);
            heap.push(i, j, dist);
        }
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
