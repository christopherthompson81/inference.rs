//! The generator's host-side signal path: the NSF harmonic source and the STFT around the network.

use std::f64::consts::PI;

use inference_tensor::Result;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_isaac::Isaac64Rng;

pub const SAMPLE_RATE: usize = 24_000;
// SineGen(harmonic_num = 8): the fundamental and eight overtones
pub const HARMONICS: usize = 9;
const SINE_AMP: f32 = 0.1;
const VOICED_NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.;

/// The source's Gaussian noise, one draw per sample and harmonic: seeded, or given (row-major, samples x harmonics).
pub enum SourceNoise {
    Seeded(Box<Isaac64Rng>),
    Given(Vec<f32>),
}

impl SourceNoise {
    pub fn seeded(seed: u64) -> Self {
        Self::Seeded(Box::new(Isaac64Rng::seed_from_u64(seed)))
    }

    fn draw(&mut self, len: usize) -> Result<Vec<f32>> {
        match self {
            Self::Seeded(rng) => Ok((0..len)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect()),
            Self::Given(noise) if noise.len() == len => Ok(std::mem::take(noise)),
            Self::Given(noise) => {
                inference_tensor::bail!("{} noise values given, {len} needed", noise.len())
            }
        }
    }
}

fn hann(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|n| (0.5 - 0.5 * (2. * PI * n as f64 / n_fft as f64).cos()) as f32)
        .collect()
}

/// `SourceModuleHnNSF`'s merged excitation, `upsample` samples per F0 frame; its random initial phase never reaches it.
pub fn harmonic_source(
    f0: &[f32],
    upsample: usize,
    weights: &[f32],
    bias: f32,
    noise: &mut SourceNoise,
) -> Result<Vec<f32>> {
    let frames = f0.len();
    let len = frames * upsample;
    // torch's CPU cumsum accumulates in double
    let mut acc = [0f64; HARMONICS];
    let mut phase = vec![[0f32; HARMONICS]; frames];
    for (d, &f) in f0.iter().enumerate() {
        for h in 0..HARMONICS {
            let rad = f * (h + 1) as f32 / SAMPLE_RATE as f32;
            acc[h] += f64::from(rad - rad.floor());
            phase[d][h] = (acc[h] as f32 * 2.) * PI as f32 * upsample as f32;
        }
    }
    let noise = noise.draw(len * HARMONICS)?;
    let scale = (1. / upsample as f64) as f32;
    let mut out = Vec::with_capacity(len);
    for n in 0..len {
        let f = f0[n / upsample];
        let voiced = f > VOICED_THRESHOLD;
        let src = (scale * (n as f32 + 0.5) - 0.5).max(0.);
        let i0 = src as usize;
        let i1 = (i0 + 1).min(frames - 1);
        let w1 = src - i0 as f32;
        let mut sum = bias;
        for h in 0..HARMONICS {
            let p = (1. - w1) * phase[i0][h] + w1 * phase[i1][h];
            let z = noise[n * HARMONICS + h];
            let wave = if voiced {
                p.sin() * SINE_AMP + VOICED_NOISE_STD * z
            } else {
                SINE_AMP / 3. * z
            };
            sum += weights[h] * wave;
        }
        out.push(sum.tanh());
    }
    Ok(out)
}

/// `torch.stft(center=True, reflect)` with a periodic Hann window: magnitudes then phases, (n_fft + 2) x frames flat.
pub fn stft(xs: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, usize) {
    let pad = n_fft / 2;
    let len = xs.len();
    let at = |i: isize| -> f32 {
        let i = if i < 0 {
            -i
        } else if i >= len as isize {
            2 * (len as isize - 1) - i
        } else {
            i
        };
        xs[i as usize]
    };
    let window = hann(n_fft);
    let bins = n_fft / 2 + 1;
    let frames = len / hop + 1;
    let mut out = vec![0f32; 2 * bins * frames];
    for t in 0..frames {
        let start = (t * hop) as isize - pad as isize;
        for k in 0..bins {
            let (mut re, mut im) = (0f64, 0f64);
            for (n, w) in window.iter().enumerate() {
                let v = f64::from(at(start + n as isize) * w);
                let angle = 2. * PI * (k * n) as f64 / n_fft as f64;
                re += v * angle.cos();
                im -= v * angle.sin();
            }
            // a real FFT's DC and Nyquist bins are exactly real, so their angle is 0 or +pi, never -pi
            if k == 0 || k == bins - 1 {
                im = 0.;
            }
            out[k * frames + t] = re.hypot(im) as f32;
            out[(bins + k) * frames + t] = im.atan2(re) as f32;
        }
    }
    (out, frames)
}

/// `torch.istft(center=True)` of `exp(rows[..bins]) * e^(i sin(rows[bins..]))`, the generator's spectral head.
pub fn istft(rows: &[Vec<f32>], n_fft: usize, hop: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let frames = rows[0].len();
    let window = hann(n_fft);
    let total = n_fft + hop * (frames - 1);
    let (mut ola, mut env) = (vec![0f64; total], vec![0f64; total]);
    for t in 0..frames {
        let spec = (0..bins)
            .map(|k| {
                let mag = f64::from(rows[k][t].exp());
                let phase = f64::from(rows[bins + k][t].sin());
                (mag * phase.cos(), mag * phase.sin())
            })
            .collect::<Vec<_>>();
        for (n, &w) in window.iter().enumerate() {
            // irfft drops the imaginary parts of the DC and Nyquist bins
            let mut v = spec[0].0 + spec[bins - 1].0 * if n % 2 == 0 { 1. } else { -1. };
            for (k, &(re, im)) in spec.iter().enumerate().take(bins - 1).skip(1) {
                let angle = 2. * PI * (k * n) as f64 / n_fft as f64;
                v += 2. * (re * angle.cos() - im * angle.sin());
            }
            let w = f64::from(w);
            ola[t * hop + n] += v / n_fft as f64 * w;
            env[t * hop + n] += w * w;
        }
    }
    let pad = n_fft / 2;
    (pad..pad + hop * (frames - 1))
        .map(|i| (ola[i] / env[i]) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A one-bin frame restarts its cosine, so frames agree only when k * hop % n_fft == 0; torch scales by sum w / sum w^2
    #[test]
    fn istft_of_one_bin_is_a_cosine() {
        let (n_fft, hop, frames, bin, amplitude) = (20, 5, 12, 4, 4f32);
        let bins = n_fft / 2 + 1;
        // the head is log magnitude then a pre-sin phase; a very negative log magnitude is an empty bin
        let rows = (0..2 * bins)
            .map(|r| {
                vec![
                    if r == bin {
                        amplitude.ln()
                    } else if r < bins {
                        -100.
                    } else {
                        0.
                    };
                    frames
                ]
            })
            .collect::<Vec<_>>();
        let out = istft(&rows, n_fft, hop);
        assert_eq!(out.len(), hop * (frames - 1));
        let window = hann(n_fft);
        for (i, v) in out.iter().enumerate() {
            let n = i + n_fft / 2;
            let (mut w, mut w2) = (0f32, 0f32);
            for t in 0..frames {
                if let Some(&x) = n.checked_sub(t * hop).and_then(|j| window.get(j)) {
                    w += x;
                    w2 += x * x;
                }
            }
            let tone = amplitude * 2. / n_fft as f32
                * (2. * PI as f32 * (bin * n) as f32 / n_fft as f32).cos();
            let want = tone * w / w2;
            assert!((v - want).abs() < 1e-5, "sample {i}: {v} vs {want}");
        }
    }

    #[test]
    fn stft_of_a_cosine_peaks_in_its_bin() {
        let (n_fft, hop, bin) = (20, 5, 2);
        let xs = (0..100)
            .map(|n| (2. * PI * (bin * n) as f64 / n_fft as f64).cos() as f32)
            .collect::<Vec<_>>();
        let (spec, frames) = stft(&xs, n_fft, hop);
        // an interior frame: the periodic Hann window turns the cosine into n_fft / 4 at its bin
        let t = frames / 2;
        for k in 0..n_fft / 2 + 1 {
            let mag = spec[k * frames + t];
            let want = if k == bin {
                n_fft as f32 / 4.
            } else if k.abs_diff(bin) == 1 {
                n_fft as f32 / 8.
            } else {
                0.
            };
            assert!((mag - want).abs() < 1e-4, "bin {k}: {mag} vs {want}");
        }
    }
}
