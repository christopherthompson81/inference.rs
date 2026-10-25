//! Gaussian splats as SPZ (Niantic's open format, version 3): positions in 24-bit fixed point and every other attribute
//! in a byte, gzipped, about a tenth of the PLY. Version 3 is the gzip form web viewers read; version 4 moved to zstd.

use std::io::Write;
use std::path::Path;

use inference_tensor::Result;

const MAGIC: u32 = 0x5053_474e;
const VERSION: u32 = 3;
const FRACTIONAL_BITS: u8 = 12;
// SPZ's colours are the SH DC term scaled by this, around 0.5
const COLOR_SCALE: f32 = 0.15;
// log scales stored as (s + 10) * 16
const SCALE_OFFSET: f32 = 10.0;
const SCALE_STEPS: f32 = 16.0;
// a smallest-three component's magnitude is at most 1/sqrt(2), kept in 9 bits
const COMPONENT_BITS: u32 = 9;

/// Splats as the trainer exports them: per splat 3 position, 3 SH DC, 1 opacity logit, 3 log scale, and a unit
/// quaternion in w x y z order.
pub struct SplatRows<'a> {
    pub means: &'a [[f32; 3]],
    pub sh0: &'a [[f32; 3]],
    pub logits: &'a [f32],
    pub log_scales: &'a [[f32; 3]],
    pub quats_wxyz: &'a [[f32; 4]],
}

fn byte(x: f32) -> u8 {
    x.round().clamp(0.0, 255.0) as u8
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// the largest component's index in the top two bits, then the other three as sign and 9-bit magnitude, in index order
fn smallest_three(xyzw: [f32; 4]) -> u32 {
    let norm = xyzw
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
        .max(f32::MIN_POSITIVE);
    let q = xyzw.map(|v| v / norm);
    let largest = (0..4)
        .max_by(|&a, &b| q[a].abs().total_cmp(&q[b].abs()))
        .expect("four components");
    let negate = q[largest] < 0.0;
    let max = ((1u32 << COMPONENT_BITS) - 1) as f32;
    let mut packed = largest as u32;
    for (i, &v) in q.iter().enumerate() {
        if i == largest {
            continue;
        }
        let negative = u32::from((v < 0.0) ^ negate);
        let magnitude = (max * (v.abs() / std::f32::consts::FRAC_1_SQRT_2) + 0.5) as u32;
        packed = (packed << (COMPONENT_BITS + 1))
            | (negative << COMPONENT_BITS)
            | magnitude.min(max as u32);
    }
    packed
}

/// Writes `rows` to `path` as SPZ with no higher-order SH; the count written.
pub fn write_spz(path: &Path, rows: &SplatRows) -> Result<usize> {
    let n = rows.means.len();
    let mut raw = Vec::with_capacity(16 + n * (9 + 1 + 3 + 3 + 4));
    raw.extend_from_slice(&MAGIC.to_le_bytes());
    raw.extend_from_slice(&VERSION.to_le_bytes());
    raw.extend_from_slice(&(n as u32).to_le_bytes());
    raw.extend_from_slice(&[0, FRACTIONAL_BITS, 0, 0]);
    let unit = (1u32 << FRACTIONAL_BITS) as f32;
    for p in rows.means {
        for v in p {
            let fixed = (v * unit).round() as i32;
            raw.extend_from_slice(&fixed.to_le_bytes()[..3]);
        }
    }
    raw.extend(rows.logits.iter().map(|&l| byte(sigmoid(l) * 255.0)));
    for c in rows.sh0 {
        raw.extend(c.map(|v| byte(v * COLOR_SCALE * 255.0 + 0.5 * 255.0)));
    }
    for s in rows.log_scales {
        raw.extend(s.map(|v| byte((v + SCALE_OFFSET) * SCALE_STEPS)));
    }
    for &[w, x, y, z] in rows.quats_wxyz {
        raw.extend_from_slice(&smallest_three([x, y, z, w]).to_le_bytes());
    }
    let file = std::fs::File::create(path)?;
    let mut gz = flate2::write::GzEncoder::new(
        std::io::BufWriter::new(file),
        flate2::Compression::default(),
    );
    gz.write_all(&raw)?;
    gz.finish()?.flush()?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    // the decode the viewers do (Spark, and PlayCanvas for the shared fields), applied to what was written
    fn read_back(path: &Path) -> (Vec<u8>, usize) {
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap())
            .read_to_end(&mut raw)
            .unwrap();
        let n = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        (raw, n)
    }

    fn unpack_quat(packed: u32) -> [f32; 4] {
        let largest = (packed >> 30) as usize;
        let mut q = [0f32; 4];
        let mut rest = packed;
        let mut sum = 0.0;
        for i in (0..4).rev() {
            if i == largest {
                continue;
            }
            let mut v = (rest & 0x1ff) as f32 * (std::f32::consts::FRAC_1_SQRT_2 / 511.0);
            if rest & 0x200 != 0 {
                v = -v;
            }
            q[i] = v;
            sum += v * v;
            rest >>= 10;
        }
        q[largest] = (1.0 - sum).max(0.0).sqrt();
        q
    }

    #[test]
    fn splats_read_back_as_written() {
        let means = [[1.25f32, -3.5, 0.000244], [-100.0, 7.0, 2047.0]];
        let sh0 = [[0.0f32, 1.0, -1.0], [3.0, -3.3, 0.5]];
        let logits = [0.0f32, 2.0];
        let log_scales = [[-4.0f32, -9.0, 0.5], [-2.0, -2.0, -2.0]];
        let quats_wxyz = [[0.9f32, 0.1, -0.3, 0.2], [-0.2, 0.7, 0.1, -0.6]];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("splats.spz");
        let written = write_spz(
            &path,
            &SplatRows {
                means: &means,
                sh0: &sh0,
                logits: &logits,
                log_scales: &log_scales,
                quats_wxyz: &quats_wxyz,
            },
        )
        .unwrap();
        let (raw, n) = read_back(&path);
        assert_eq!((written, n), (2, 2));
        assert_eq!(u32::from_le_bytes(raw[0..4].try_into().unwrap()), MAGIC);
        assert_eq!(u32::from_le_bytes(raw[4..8].try_into().unwrap()), VERSION);
        assert_eq!(raw[13], FRACTIONAL_BITS);
        let positions = &raw[16..16 + 9 * n];
        let alphas = &raw[16 + 9 * n..16 + 10 * n];
        let colors = &raw[16 + 10 * n..16 + 13 * n];
        let scales = &raw[16 + 13 * n..16 + 16 * n];
        let rotations = &raw[16 + 16 * n..16 + 20 * n];
        assert_eq!(raw.len(), 16 + 20 * n);
        for i in 0..n {
            for d in 0..3 {
                let b = &positions[9 * i + 3 * d..9 * i + 3 * d + 3];
                let v =
                    ((i32::from(b[0]) | i32::from(b[1]) << 8 | i32::from(b[2]) << 16) << 8) >> 8;
                assert!((v as f32 / 4096.0 - means[i][d]).abs() <= 0.5 / 4096.0 + 1e-6);
                let c = (f32::from(colors[3 * i + d]) / 255.0 - 0.5) / COLOR_SCALE;
                assert!((c - sh0[i][d].clamp(-0.5 / COLOR_SCALE, 0.5 / COLOR_SCALE)).abs() < 0.03);
                let s = f32::from(scales[3 * i + d]) / 16.0 - 10.0;
                assert!((s - log_scales[i][d]).abs() <= 1.0 / 32.0 + 1e-6);
            }
            assert!((f32::from(alphas[i]) / 255.0 - sigmoid(logits[i])).abs() < 0.003);
            let q = unpack_quat(u32::from_le_bytes(
                rotations[4 * i..4 * i + 4].try_into().unwrap(),
            ));
            let [w, x, y, z] = quats_wxyz[i];
            let norm = (w * w + x * x + y * y + z * z).sqrt();
            let expected = [x / norm, y / norm, z / norm, w / norm];
            // q and -q are the same rotation
            let dot = q
                .iter()
                .zip(expected)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                .abs();
            assert!(dot > 0.9999, "{q:?} against {expected:?}");
        }
    }
}
