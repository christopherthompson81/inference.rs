//! LPIPS with VGG16 features (torchmetrics' `LearnedPerceptualImagePatchSimilarity(net_type="vgg")`), differentiable
//! through the tensor graph so the splatting trainer can descend it.

use std::path::Path;

use inference_tensor::{D, Device, Result, Tensor};

// torchvision vgg16 `features` conv indices, grouped by LPIPS slice (each slice after the first opens with a 2x2 pool)
const VGG_SLICES: [&[usize]; 5] = [
    &[0, 2],
    &[5, 7],
    &[10, 12, 14],
    &[17, 19, 21],
    &[24, 26, 28],
];
const SHIFT: [f32; 3] = [-0.030, -0.088, -0.188];
const SCALE: [f32; 3] = [0.458, 0.448, 0.450];
const NORM_EPS: f64 = 1e-8;

pub struct Lpips {
    convs: Vec<Vec<(Tensor, Tensor)>>,
    heads: Vec<Tensor>,
    shift: Tensor,
    scale: Tensor,
}

impl Lpips {
    /// `path` is a safetensors file with torchvision vgg16's `features.{i}.weight/bias` and the LPIPS v0.1 heads
    /// `lin{0..4}.model.1.weight`.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut tensors = inference_tensor::safetensors::load(path, device)?;
        let mut take = |name: &str| -> Result<Tensor> {
            tensors.remove(name).ok_or_else(|| {
                inference_tensor::Error::msg(format!("{}: missing {name}", path.display()))
            })
        };
        let convs = VGG_SLICES
            .iter()
            .map(|slice| {
                slice
                    .iter()
                    .map(|i| {
                        Ok((
                            take(&format!("features.{i}.weight"))?,
                            take(&format!("features.{i}.bias"))?.reshape((1, (), 1, 1))?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let heads = (0..VGG_SLICES.len())
            .map(|i| take(&format!("lin{i}.model.1.weight")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            convs,
            heads,
            shift: Tensor::new(&SHIFT, device)?.reshape((1, 3, 1, 1))?,
            scale: Tensor::new(&SCALE, device)?.reshape((1, 3, 1, 1))?,
        })
    }

    // the five unit-normalized feature taps of `x` ([b, 3, h, w], used as given: no [0, 1] to [-1, 1] remap)
    fn features(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut h = x.broadcast_sub(&self.shift)?.broadcast_div(&self.scale)?;
        let mut taps = Vec::with_capacity(self.convs.len());
        for (i, slice) in self.convs.iter().enumerate() {
            if i > 0 {
                h = h.max_pool2d_with_stride(2, 2)?;
            }
            for (w, b) in slice {
                h = h.conv2d(w, 1, 1, 1, 1)?.broadcast_add(b)?.relu()?;
            }
            let norm = (h.sqr()?.sum_keepdim(1)? + NORM_EPS)?.sqrt()?;
            taps.push(h.broadcast_div(&norm)?);
        }
        Ok(taps)
    }

    /// Mean LPIPS distance between `a` and `b` (`[b, 3, h, w]`); gradients flow to whichever is tracked.
    pub fn forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let (fa, fb) = (self.features(a)?, self.features(&b.detach())?);
        let mut total: Option<Tensor> = None;
        for ((x, y), lin) in fa.iter().zip(&fb).zip(&self.heads) {
            let d = (x - y)?
                .sqr()?
                .conv2d(lin, 0, 1, 1, 1)?
                .mean(D::Minus1)?
                .mean(D::Minus1)?;
            total = Some(match total {
                Some(t) => (t + d)?,
                None => d,
            });
        }
        total.expect("five taps").mean_all()
    }
}
