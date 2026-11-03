//! `torch.nn.functional.interpolate(mode="bicubic", antialias=True, align_corners=False)` on float images.

// the cubic kernel's coefficient on PyTorch's antialiased path (plain bicubic's is -0.75)
const AA_CUBIC_A: f64 = -0.5;

fn aa_cubic(x: f64) -> f64 {
    let (a, x) = (AA_CUBIC_A, x.abs());
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

/// Each output index's (input index, weight) taps for resampling `input` samples to `output` with antialiased bicubic.
pub fn aa_bicubic_weights(input: usize, output: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = input as f64 / output as f64;
    let (support, inv) = if scale >= 1.0 {
        (2.0 * scale, 1.0 / scale)
    } else {
        (2.0, 1.0)
    };
    (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let lo = ((center - support + 0.5) as isize).max(0) as usize;
            let hi = ((center + support + 0.5) as usize).min(input);
            let taps = (lo..hi)
                .map(|j| (j, aa_cubic((j as f64 - center + 0.5) * inv)))
                .collect::<Vec<_>>();
            let total: f64 = taps.iter().map(|t| t.1).sum();
            taps.into_iter().map(|(j, w)| (j, w / total)).collect()
        })
        .collect()
}

/// Planar `[channels][height][width]` floats resampled to `(out_height, out_width)`, width first as PyTorch does.
pub fn aa_bicubic_resize_planar(
    planes: &[f32],
    channels: usize,
    (height, width): (usize, usize),
    (out_height, out_width): (usize, usize),
) -> Vec<f32> {
    let cols = aa_bicubic_weights(width, out_width);
    let rows = aa_bicubic_weights(height, out_height);
    let mut wide = vec![0f32; channels * height * out_width];
    for (src, dst) in planes
        .chunks_exact(width)
        .zip(wide.chunks_exact_mut(out_width))
    {
        for (o, taps) in dst.iter_mut().zip(&cols) {
            *o = taps.iter().map(|&(j, w)| src[j] * w as f32).sum();
        }
    }
    let mut out = vec![0f32; channels * out_height * out_width];
    for (src, dst) in wide
        .chunks_exact(height * out_width)
        .zip(out.chunks_exact_mut(out_height * out_width))
    {
        for (row, taps) in dst.chunks_exact_mut(out_width).zip(&rows) {
            for &(j, w) in taps {
                for (o, v) in row.iter_mut().zip(&src[j * out_width..][..out_width]) {
                    *o += v * w as f32;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_sum_to_one_and_a_same_size_resize_is_the_identity() {
        for (input, output) in [(7, 3), (3, 7), (16, 16), (128, 37)] {
            for taps in aa_bicubic_weights(input, output) {
                assert!((taps.iter().map(|t| t.1).sum::<f64>() - 1.0).abs() < 1e-12);
            }
        }
        let planes = (0..2 * 5 * 6).map(|i| i as f32).collect::<Vec<_>>();
        let same = aa_bicubic_resize_planar(&planes, 2, (5, 6), (5, 6));
        for (a, b) in same.iter().zip(&planes) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }
}
