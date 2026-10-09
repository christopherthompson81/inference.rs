//! Region outlines from per-query masks, vertex for vertex as transformers' PPDocLayoutV3ImageProcessor draws them.

/// `approxPolyDP`'s epsilon as a share of the contour's perimeter.
const EPSILON_RATIO: f64 = 0.004;
/// A kept corner this close to 45 degrees is replaced by a point along its bisector.
const SHARP_ANGLE_DEG: f64 = 45.;
const SHARP_ANGLE_TOLERANCE_DEG: f64 = 1.;
/// Fewer vertices than this fall back to the box.
const MIN_POLYGON_VERTICES: usize = 4;
/// OpenCV refines the closed contour's two farthest points this many times.
const DP_INIT_ITERS: usize = 3;
/// The mask is predicted at a quarter of the model's input resolution.
const MASK_STRIDE: f32 = 4.;
/// Marks a pixel on a followed border in the padded contour grid (1 is unvisited foreground, 0 background).
const FOLLOWED: i8 = 2;

/// A query's mask logits, `mask_h x mask_w` row-major, at the model's input resolution over `MASK_STRIDE`.
pub struct MaskView<'a> {
    pub logits: &'a [f32],
    pub height: usize,
    pub width: usize,
}

/// A region's outline in image pixels, else its box's corners; `scale` maps image pixels to model input pixels.
pub fn region_outline(
    mask: &MaskView<'_>,
    bbox: [f32; 4],
    scale: (f32, f32),
    threshold: f32,
) -> Vec<[f32; 2]> {
    // numpy's astype(int32) truncates toward zero
    let [x_min, y_min, x_max, y_max] = bbox.map(|v| v as i32);
    let rect = vec![
        [x_min as f32, y_min as f32],
        [x_max as f32, y_min as f32],
        [x_max as f32, y_max as f32],
        [x_min as f32, y_max as f32],
    ];
    let (box_w, box_h) = (x_max - x_min, y_max - y_min);
    if box_w <= 0 || box_h <= 0 {
        return rect;
    }
    let (sx, sy) = (scale.0 / MASK_STRIDE, scale.1 / MASK_STRIDE);
    // Python's round() is half to even, as f32::round_ties_even
    let clip = |v: f32, hi: usize| (v.round_ties_even() as i64).clamp(0, hi as i64) as usize;
    let (x0, x1) = (
        clip(x_min as f32 * sx, mask.width),
        clip(x_max as f32 * sx, mask.width),
    );
    let (y0, y1) = (
        clip(y_min as f32 * sy, mask.height),
        clip(y_max as f32 * sy, mask.height),
    );
    if x1 <= x0 || y1 <= y0 {
        return rect;
    }
    // sigmoid(logit) > threshold
    let cut = (threshold / (1. - threshold)).ln();
    let (crop_w, crop_h) = (x1 - x0, y1 - y0);
    let crop: Vec<bool> = (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (y, x)))
        .map(|(y, x)| mask.logits[y * mask.width + x] > cut)
        .collect();
    if !crop.iter().any(|&on| on) {
        return rect;
    }
    // where downsampling drops every on-pixel transformers returns None; the box serves instead
    let grid = resize_nearest_padded(&crop, crop_w, crop_h, box_w as usize, box_h as usize);
    match mask_polygon(grid, box_w as usize, box_h as usize) {
        Some(polygon) if polygon.len() >= MIN_POLYGON_VERTICES => polygon
            .into_iter()
            .map(|[x, y]| [(x + f64::from(x_min)) as f32, (y + f64::from(y_min)) as f32])
            .collect(),
        _ => rect,
    }
}

// cv2.resize INTER_NEAREST into a zero-framed i8 grid; resizeNN's index floor(x * (1 / (dst / src))) rounds unlike src/dst
fn resize_nearest_padded(
    src: &[bool],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<i8> {
    let (fx, fy) = (
        1. / (dst_w as f64 / src_w as f64),
        1. / (dst_h as f64 / src_h as f64),
    );
    let col: Vec<usize> = (0..dst_w)
        .map(|x| ((x as f64 * fx).floor() as usize).min(src_w - 1))
        .collect();
    let stride = dst_w + 2;
    let mut grid = vec![0i8; stride * (dst_h + 2)];
    for y in 0..dst_h {
        let row = ((y as f64 * fy).floor() as usize).min(src_h - 1);
        let out = &mut grid[(y + 1) * stride + 1..(y + 1) * stride + 1 + dst_w];
        for (o, &c) in out.iter_mut().zip(&col) {
            *o = i8::from(src[row * src_w + c]);
        }
    }
    grid
}

// `_mask2polygon`: the largest outer contour, simplified, then the sharp-corner pass
fn mask_polygon(grid: Vec<i8>, width: usize, height: usize) -> Option<Vec<[f64; 2]>> {
    let contour = outer_contours(grid, width, height)
        .into_iter()
        .map(|c| (contour_area(&c), c))
        // cv2 lists contours in reverse raster order and max keeps the first of equal areas: the last one found here
        .reduce(|best, next| if next.0 >= best.0 { next } else { best })?
        .1;
    let epsilon = EPSILON_RATIO * arc_length(&contour);
    Some(sharp_corners(&approx_poly_dp(&contour, epsilon)))
}

type Point = (i64, i64);

// Moore neighbourhood in OpenCV's order (counter-clockwise from east, y down): E, NE, N, NW, W, SW, S, SE
const NEIGHBOURS: [Point; 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

// cv2.findContours(.., CHAIN_APPROX_SIMPLE) outer borders; a trace started right of a hole circles it, never the largest
fn outer_contours(mut img: Vec<i8>, width: usize, height: usize) -> Vec<Vec<Point>> {
    let (w, h) = (width as i64 + 2, height as i64 + 2);
    let mut contours = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = (y * w + x) as usize;
            if img[i] == 1 && img[i - 1] == 0 {
                let points = follow_border(&mut img, w, (x, y));
                contours.push(
                    points
                        .into_iter()
                        .map(|(px, py)| (px - 1, py - 1))
                        .collect(),
                );
            }
        }
    }
    contours
}

// OpenCV's icvFetchContour from `start` (background to its west), keeping a point where the chain turns
fn follow_border(img: &mut [i8], w: i64, start: Point) -> Vec<Point> {
    let at = |p: Point| (p.1 * w + p.0) as usize;
    let step = |c: Point, d: usize| (c.0 + NEIGHBOURS[d].0, c.1 + NEIGHBOURS[d].1);
    // clockwise from the west (OpenCV's s_end = 4, decrementing first) for the first foreground neighbour
    let mut s = 4usize;
    let mut found = None;
    for _ in 0..8 {
        s = (s + 7) % 8;
        if img[at(step(start, s))] != 0 {
            found = Some(s);
            break;
        }
    }
    let Some(s1) = found else {
        img[at(start)] = FOLLOWED;
        return vec![start];
    };
    let first = step(start, s1);
    let mut points = Vec::new();
    let mut prev_s = s1 ^ 4;
    let mut cur = start;
    let mut s = s1;
    loop {
        // counter-clockwise from the one after the way back
        let mut next = cur;
        for _ in 0..8 {
            s = (s + 1) % 8;
            next = step(cur, s);
            if img[at(next)] != 0 {
                break;
            }
        }
        img[at(cur)] = FOLLOWED;
        if s != prev_s {
            points.push(cur);
            prev_s = s;
        }
        if next == start && cur == first {
            break;
        }
        cur = next;
        s = (s + 4) % 8;
    }
    points
}

// cv2.contourArea: the shoelace area, unsigned
fn contour_area(points: &[Point]) -> f64 {
    let n = points.len();
    let twice: i64 = (0..n)
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % n]);
            a.0 * b.1 - b.0 * a.1
        })
        .sum();
    (twice as f64 / 2.).abs()
}

// cv2.arcLength(closed=True): each edge's length in f32, from the closing edge on, summed in f64
fn arc_length(points: &[Point]) -> f64 {
    let Some(&last) = points.last() else {
        return 0.;
    };
    let mut prev = (last.0 as f32, last.1 as f32);
    points.iter().fold(0., |perimeter, &(x, y)| {
        let (x, y) = (x as f32, y as f32);
        let (dx, dy) = (x - prev.0, y - prev.1);
        prev = (x, y);
        perimeter + f64::from((dx * dx + dy * dy).sqrt())
    })
}

// cv2.approxPolyDP(closed=True) after OpenCV's approxPolyDP_, with distance to the segment rather than its line
fn approx_poly_dp(src: &[Point], epsilon: f64) -> Vec<Point> {
    let count = src.len();
    if count == 0 {
        return Vec::new();
    }
    let eps = epsilon * epsilon;
    let mut dst: Vec<Point> = Vec::with_capacity(count);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0;
    let mut right_start = 0;
    let mut start_pt = src[0];
    let mut le_eps = false;
    for _ in 0..DP_INIT_ITERS {
        pos = (pos + right_start) % count;
        start_pt = src[pos];
        pos = (pos + 1) % count;
        let mut max_dist = 0.;
        for j in 1..count {
            let pt = src[pos];
            pos = (pos + 1) % count;
            let dist = ((pt.0 - start_pt.0).pow(2) + (pt.1 - start_pt.1).pow(2)) as f64;
            if dist > max_dist {
                max_dist = dist;
                right_start = j;
            }
        }
        le_eps = max_dist <= eps;
    }
    if le_eps {
        dst.push(start_pt);
    } else {
        let slice_start = pos % count;
        let right = ((right_start + slice_start) % count, slice_start);
        stack.push(right);
        stack.push((slice_start, right.0));
    }
    while let Some((start, end)) = stack.pop() {
        let end_pt = src[end];
        let mut p = start;
        let first = src[p];
        p = (p + 1) % count;
        let mut cut = start;
        let le = if p != end {
            let (dx, dy) = ((end_pt.0 - first.0) as f64, (end_pt.1 - first.1) as f64);
            let segment_len_2 = dx * dx + dy * dy;
            // squared distance to the segment (not its line), times its squared length
            let mut max_dist = 0.;
            while p != end {
                let pt = src[p];
                p = (p + 1) % count;
                let (px, py) = ((pt.0 - first.0) as f64, (pt.1 - first.1) as f64);
                let projection = px * dx + py * dy;
                let dist = if projection < 0. {
                    (px * px + py * py) * segment_len_2
                } else if projection > segment_len_2 {
                    let (ex, ey) = ((pt.0 - end_pt.0) as f64, (pt.1 - end_pt.1) as f64);
                    (ex * ex + ey * ey) * segment_len_2
                } else {
                    let cross = py * dx - px * dy;
                    cross * cross
                };
                if dist > max_dist {
                    max_dist = dist;
                    cut = (p + count - 1) % count;
                }
            }
            max_dist <= eps * segment_len_2
        } else {
            true
        };
        if le {
            dst.push(first);
        } else {
            stack.push((cut, end));
            stack.push((start, cut));
        }
    }
    drop_collinear(dst, eps)
}

// approxPolyDP_'s last stage on a closed polygon: points on an almost straight line between their neighbours go
fn drop_collinear(mut dst: Vec<Point>, eps: f64) -> Vec<Point> {
    let count = dst.len();
    let mut new_count = count;
    let mut pos = count - 1;
    let read = |dst: &[Point], pos: &mut usize| {
        let pt = dst[*pos];
        *pos = (*pos + 1) % count;
        pt
    };
    let mut start_pt = read(&dst, &mut pos);
    let mut wpos = pos;
    let mut pt = read(&dst, &mut pos);
    let mut i = 0;
    while i < count && new_count > 2 {
        let end_pt = read(&dst, &mut pos);
        let (dx, dy) = (
            (end_pt.0 - start_pt.0) as f64,
            (end_pt.1 - start_pt.1) as f64,
        );
        let dist = ((pt.0 - start_pt.0) as f64 * dy - (pt.1 - start_pt.1) as f64 * dx).abs();
        let successive =
            (pt.0 - start_pt.0) * (end_pt.0 - pt.0) + (pt.1 - start_pt.1) * (end_pt.1 - pt.1);
        if dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx != 0. && dy != 0. && successive >= 0
        {
            new_count -= 1;
            start_pt = end_pt;
            dst[wpos] = end_pt;
            wpos = (wpos + 1) % count;
            pt = read(&dst, &mut pos);
            i += 2;
            continue;
        }
        start_pt = pt;
        dst[wpos] = pt;
        wpos = (wpos + 1) % count;
        pt = end_pt;
        i += 1;
    }
    dst.truncate(new_count);
    dst
}

// `extract_custom_vertices`: corners turning one way are kept, one near 45 degrees moved into its wedge on the bisector
fn sharp_corners(polygon: &[Point]) -> Vec<[f64; 2]> {
    let n = polygon.len();
    let to_f = |p: Point| [p.0 as f64, p.1 as f64];
    (0..n)
        .filter_map(|i| {
            let (prev, cur, next) = (
                to_f(polygon[(i + n - 1) % n]),
                to_f(polygon[i]),
                to_f(polygon[(i + 1) % n]),
            );
            let v1 = [prev[0] - cur[0], prev[1] - cur[1]];
            let v2 = [next[0] - cur[0], next[1] - cur[1]];
            let cross = v1[1] * v2[0] - v1[0] * v2[1];
            if cross >= 0. {
                return None;
            }
            let (n1, n2) = (v1[0].hypot(v1[1]), v2[0].hypot(v2[1]));
            let cos = ((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)).clamp(-1., 1.);
            if (cos.acos().to_degrees() - SHARP_ANGLE_DEG).abs() < SHARP_ANGLE_TOLERANCE_DEG {
                let dir = [v1[0] / n1 + v2[0] / n2, v1[1] / n1 + v2[1] / n2];
                let norm = dir[0].hypot(dir[1]);
                let step = (n1 + n2) / 2.;
                Some([cur[0] + dir[0] / norm * step, cur[1] + dir[1] / norm * step])
            } else {
                Some(cur)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mask logits standing in for the model's: well past the 0.5 threshold either way
    const ON_LOGIT: f32 = 5.;
    const THRESHOLD: f32 = 0.5;
    // The goldens print float vertices; the 45-degree bisector points are irrational
    const VERTEX_TOLERANCE: f64 = 1e-4;

    #[derive(serde::Deserialize)]
    struct Goldens {
        model_size: (f32, f32),
        page_size: (f32, f32),
        cases: Vec<Case>,
    }

    #[derive(serde::Deserialize)]
    struct Case {
        kind: String,
        mask: Vec<Vec<u8>>,
        bbox: [f32; 4],
        polygon: Vec<[f64; 2]>,
    }

    // transformers' `_extract_polygon_points_by_masks` on synthetic masks (make_outline_goldens.py), vertex for vertex
    #[test]
    fn outlines_match_transformers() {
        let goldens: Goldens =
            serde_json::from_str(include_str!("fixtures/outline_goldens.json")).unwrap();
        let scale = (
            goldens.model_size.0 / goldens.page_size.0,
            goldens.model_size.1 / goldens.page_size.1,
        );
        for case in &goldens.cases {
            let (height, width) = (case.mask.len(), case.mask[0].len());
            let logits: Vec<f32> = case
                .mask
                .iter()
                .flatten()
                .map(|&on| if on == 1 { ON_LOGIT } else { -ON_LOGIT })
                .collect();
            let mask = MaskView {
                logits: &logits,
                height,
                width,
            };
            let outline = region_outline(&mask, case.bbox, scale, THRESHOLD);
            let same = outline.len() == case.polygon.len()
                && outline.iter().zip(&case.polygon).all(|(a, e)| {
                    (f64::from(a[0]) - e[0]).abs() < VERTEX_TOLERANCE
                        && (f64::from(a[1]) - e[1]).abs() < VERTEX_TOLERANCE
                });
            assert!(same, "{}: {outline:?} vs {:?}", case.kind, case.polygon);
        }
    }
}
