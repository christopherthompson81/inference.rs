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
