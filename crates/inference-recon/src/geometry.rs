//! Small geometry helpers the reconstruction ports share: 4 x 4 products and inverses, PyTorch3D's quaternion, torch's
//! interpolated quantile.

// PyTorch3D's matrix_to_quaternion (the best-conditioned of four candidates), scalar last, real part non-negative
pub(crate) fn rotation_to_quaternion(m: [[f32; 3]; 3]) -> [f32; 4] {
    let [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = m;
    let q_abs = [
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ]
    .map(|v| if v > 0.0 { v.sqrt() } else { 0.0 });
    // the quaternion (r, i, j, k) times each of r, i, j, k
    let by_rijk = [
        [q_abs[0] * q_abs[0], m21 - m12, m02 - m20, m10 - m01],
        [m21 - m12, q_abs[1] * q_abs[1], m10 + m01, m02 + m20],
        [m02 - m20, m10 + m01, q_abs[2] * q_abs[2], m12 + m21],
        [m10 - m01, m20 + m02, m21 + m12, q_abs[3] * q_abs[3]],
    ];
    let best = (0..4)
        .max_by(|&a, &b| q_abs[a].total_cmp(&q_abs[b]).then(b.cmp(&a)))
        .expect("four candidates");
    let denom = 2.0 * q_abs[best].max(0.1);
    let rijk = by_rijk[best].map(|v| v / denom);
    let q = [rijk[1], rijk[2], rijk[3], rijk[0]];
    if q[3] < 0.0 { q.map(|v| -v) } else { q }
}

pub(crate) fn mul4(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

// A general 4 x 4 inverse by Gauss-Jordan elimination with partial pivoting
pub(crate) fn invert4(m: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut a = *m;
    let mut inv = [[0.0; 4]; 4];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..4 {
        let pivot = (col..4)
            .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
            .expect("rows remain");
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let p = a[col][col];
        for j in 0..4 {
            a[col][j] /= p;
            inv[col][j] /= p;
        }
        for r in 0..4 {
            if r != col {
                let f = a[r][col];
                for j in 0..4 {
                    a[r][j] -= f * a[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
    }
    inv
}

// The `q` quantile with linear interpolation, as `torch.quantile`
pub(crate) fn quantile(values: &[f32], q: f32) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let pos = q * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f32)
}

/// The similarity `q ~ s R p + t` that best maps points `p` onto `q` in least squares (Horn's closed form: the
/// rotation from the top eigenvector of his 4 x 4 matrix, then the scale and translation).
pub(crate) fn similarity_fit(p: &[[f64; 3]], q: &[[f64; 3]]) -> (f64, [[f64; 3]; 3], [f64; 3]) {
    let n = p.len() as f64;
    let mean = |v: &[[f64; 3]]| [0, 1, 2].map(|d| v.iter().map(|x| x[d]).sum::<f64>() / n);
    let (mp, mq) = (mean(p), mean(q));
    let mut m = [[0.0; 3]; 3];
    let mut norm_p = 0.0;
    for (a, b) in p.iter().zip(q) {
        let (x, y) = (
            [0, 1, 2].map(|d| a[d] - mp[d]),
            [0, 1, 2].map(|d| b[d] - mq[d]),
        );
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += x[i] * y[j];
            }
        }
        norm_p += x.iter().map(|v| v * v).sum::<f64>();
    }
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = m;
    let mut k = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let mut vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    // cyclic Jacobi rotations until the off-diagonal vanishes
    for _ in 0..64 {
        let off = (0..4)
            .flat_map(|i| (0..4).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| k[i][j].abs())
            .sum::<f64>();
        if off < 1e-14 {
            break;
        }
        for a in 0..3 {
            for b in a + 1..4 {
                if k[a][b].abs() < 1e-300 {
                    continue;
                }
                let theta = (k[b][b] - k[a][a]) / (2.0 * k[a][b]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for row in k.iter_mut() {
                    let (ka, kb) = (row[a], row[b]);
                    row[a] = c * ka - s * kb;
                    row[b] = s * ka + c * kb;
                }
                let (lo, hi) = k.split_at_mut(b);
                for (ka, kb) in lo[a].iter_mut().zip(hi[0].iter_mut()) {
                    let (va, vb) = (*ka, *kb);
                    *ka = c * va - s * vb;
                    *kb = s * va + c * vb;
                }
                for v in &mut vecs {
                    let (va, vb) = (v[a], v[b]);
                    v[a] = c * va - s * vb;
                    v[b] = s * va + c * vb;
                }
            }
        }
    }
    let top = (0..4)
        .max_by(|&i, &j| k[i][i].total_cmp(&k[j][j]))
        .expect("four eigenvalues");
    let [w, x, y, z] = [0, 1, 2, 3].map(|r| vecs[r][top]);
    let r = [
        [
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ];
    let rot = |v: [f64; 3]| [0, 1, 2].map(|i| (0..3).map(|j| r[i][j] * v[j]).sum::<f64>());
    let mut dot = 0.0;
    for (a, b) in p.iter().zip(q) {
        let rx = rot([0, 1, 2].map(|d| a[d] - mp[d]));
        dot += (0..3).map(|d| rx[d] * (b[d] - mq[d])).sum::<f64>();
    }
    let s = if norm_p > 0.0 { dot / norm_p } else { 1.0 };
    let rmp = rot(mp);
    (s, r, [0, 1, 2].map(|d| mq[d] - s * rmp[d]))
}

#[cfg(test)]
mod tests {
    // a known similarity is recovered from noiseless points
    #[test]
    fn similarity_fit_recovers_a_known_transform() {
        let (a, b) = (0.7f64, -0.4f64);
        let r = [
            [a.cos(), -a.sin(), 0.0],
            [a.sin(), a.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ];
        let r2 = [
            [1.0, 0.0, 0.0],
            [0.0, b.cos(), -b.sin()],
            [0.0, b.sin(), b.cos()],
        ];
        let rr =
            [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| r2[i][k] * r[k][j]).sum::<f64>()));
        let (s, t) = (2.5, [0.3, -1.0, 4.0]);
        let p = (0..10)
            .map(|i| [i as f64 * 0.3, (i * i) as f64 * 0.1 - 1.0, (i % 3) as f64])
            .collect::<Vec<_>>();
        let q = p
            .iter()
            .map(|v| [0, 1, 2].map(|d| s * (0..3).map(|k| rr[d][k] * v[k]).sum::<f64>() + t[d]))
            .collect::<Vec<_>>();
        let (fs, fr, ft) = super::similarity_fit(&p, &q);
        assert!((fs - s).abs() < 1e-9, "{fs}");
        for i in 0..3 {
            assert!((ft[i] - t[i]).abs() < 1e-9);
            for j in 0..3 {
                assert!((fr[i][j] - rr[i][j]).abs() < 1e-9);
            }
        }
    }
}
