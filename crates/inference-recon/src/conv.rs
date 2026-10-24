//! 2D convolutions the reconstruction heads share: a conv that runs pointwise kernels as matmuls, and a transposed conv
//! whose stride equals its kernel.

use inference_tensor::nn::VarBuilder;
use inference_tensor::{Result, Tensor};

pub(crate) struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    stride: usize,
}

impl Conv {
    /// A pointwise or unpadded stride-1 conv from its weight `[cout, cin, k, k]` and bias.
    pub(crate) fn from_parts(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            weight,
            bias,
            padding: 0,
            stride: 1,
        })
    }

    pub(crate) fn load(
        cin: usize,
        cout: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            weight: vb.get((cout, cin, kernel, kernel), "weight")?,
            bias: if bias {
                Some(vb.get(cout, "bias")?)
            } else {
                None
            },
            padding,
            stride,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (cout, cin, kh, kw) = self.weight.dims4()?;
        // a pointwise conv is a matmul per frame, with no im2col copy
        let ys = if (kh, kw, self.stride, self.padding) == (1, 1, 1, 0) {
            let (frames, _, h, w) = xs.dims4()?;
            let weight = self.weight.reshape((cout, cin))?;
            let per_frame = (0..frames)
                .map(|f| weight.matmul(&xs.get(f)?.reshape((cin, h * w))?))
                .collect::<Result<Vec<_>>>()?;
            Tensor::stack(&per_frame, 0)?.reshape((frames, cout, h, w))?
        } else {
            xs.conv2d(&self.weight, self.padding, self.stride, 1, 1)?
        };
        match &self.bias {
            Some(b) => ys.broadcast_add(&b.reshape((1, (), 1, 1))?),
            None => Ok(ys),
        }
    }
}

// A transposed conv whose stride equals its kernel: each input pixel paints its own k x k output block, so it is one
// matmul per frame, `[cout * k * k, cin] x [cin, h * w]`, and a reshuffle
pub(crate) struct Upsample {
    // the kernel `[cin, cout, k, k]` as `[cout * k * k, cin]`
    weight: Tensor,
    bias: Tensor,
    channels: usize,
    kernel: usize,
}

impl Upsample {
    pub(crate) fn load(channels: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        Self::load_with(channels, channels, kernel, vb)
    }

    pub(crate) fn load_with(
        cin: usize,
        cout: usize,
        kernel: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb
            .get((cin, cout, kernel, kernel), "weight")?
            .reshape((cin, cout * kernel * kernel))?
            .t()?
            .contiguous()?;
        Ok(Self {
            weight,
            bias: vb.get(cout, "bias")?,
            channels: cout,
            kernel,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (frames, cin, h, w) = xs.dims4()?;
        let (c, k) = (self.channels, self.kernel);
        let per_frame = (0..frames)
            .map(|f| {
                self.weight
                    .matmul(&xs.get(f)?.reshape((cin, h * w))?)?
                    .reshape((c, k, k, h, w))?
                    .permute((0, 3, 1, 4, 2))?
                    .reshape((c, h * k, w * k))
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&per_frame, 0)?.broadcast_add(&self.bias.reshape((1, (), 1, 1))?)
    }
}
