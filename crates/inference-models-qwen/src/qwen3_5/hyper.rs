//! Qwen3.8-Flash-Next's hyper-connections: the residual is `streams` parallel copies of the hidden state, each block
//! reads a learned mix of them and writes its output back into every stream with its own weight.

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::{D, Result, Tensor};

use super::config::{HyperConnections, TextConfig};
use crate::utils::unvarbuilder::UnVarBuilder;

/// An RMSNorm over each `group`-wide slice of the last dimension, with one zero-centred weight per element.
pub(super) struct GroupedRmsNorm {
    // the checkpoint's weight, and the `1 + w` the forward multiplies by
    weight: Tensor,
    scale: Tensor,
    unit: Tensor,
    group: usize,
    eps: f64,
}

impl GroupedRmsNorm {
    pub(super) fn new(size: usize, group: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self {
            scale: (&weight + 1.0)?,
            unit: Tensor::ones(group, weight.dtype(), weight.device())?,
            weight,
            group,
            eps,
        })
    }

    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if inference_nn::cuda::hyper::supports(x) {
            #[allow(clippy::cast_possible_truncation)]
            return inference_nn::cuda::hyper::group_norm(
                x,
                &self.scale,
                self.group,
                self.eps as f32,
            );
        }
        let dims = x.dims().to_vec();
        let last = *dims.last().expect("a normed tensor has a last dimension");
        let mut grouped = dims.clone();
        *grouped.last_mut().expect("checked above") = last / self.group;
        grouped.push(self.group);
        #[allow(clippy::cast_possible_truncation)]
        let normed = crate::ops::rms_norm(&x.reshape(grouped)?, &self.unit, self.eps as f32)?;
        normed.reshape(dims)?.broadcast_mul(&self.scale)
    }

    pub(super) fn residual_tensors(&self, uvb: &UnVarBuilder) {
        uvb.add_tensor("weight", self.weight.clone());
    }
}

// The hyper-connection reads' projections: 8-bit activations here cost about 0.04 KLD against an F32 read, so a Q8_0
// weight multiplies the activations as they are (investigation Run 33)
fn hc_project(w: &Arc<dyn QuantMethod>, x: &Tensor) -> Result<Tensor> {
    match inference_quant::unquantized_activation_forward(w.as_ref(), x)? {
        Some(out) => Ok(out),
        None => w.forward(x),
    }
}

/// One block's read from and write into the hyper-connection streams (`Qwen4ExpTextGatedResidual`).
pub(super) struct GatedResidual {
    hc_norm: GroupedRmsNorm,
    mix_down: Arc<dyn QuantMethod>,
    mix_up: Arc<dyn QuantMethod>,
    // None on the final mixer, which only reads
    inject: Option<Arc<dyn QuantMethod>>,
    streams: usize,
    hidden: usize,
}

/// A block's mixed input and what it needs to write its output back.
pub(super) struct HyperRead {
    pub(super) input: Tensor,
    streams_in: Tensor,
    // `[.., streams]`, already `2 sigmoid(inject / streams)`
    inject: Tensor,
}

impl GatedResidual {
    pub(super) fn load(
        cfg: &TextConfig,
        hc: HyperConnections,
        vb_quant: ShardedVarBuilder,
        vb_plain: ShardedVarBuilder,
        combine: bool,
    ) -> Result<Self> {
        let width = hc.streams * cfg.hidden_size;
        let linear = |vb: ShardedVarBuilder, inputs: usize, outputs: usize| {
            inference_quant::ReplicatedLayer::new(
                inputs,
                outputs,
                &cfg.quantization_config,
                false,
                vb,
            )
        };
        Ok(Self {
            hc_norm: GroupedRmsNorm::new(
                width,
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_plain.pp("hc_norm"),
            )?,
            mix_down: linear(vb_quant.pp("input_mix_weight_down"), width, hc.lowrank)?,
            mix_up: linear(vb_quant.pp("input_mix_weight_up"), hc.lowrank, width)?,
            inject: combine
                .then(|| linear(vb_quant.pp("block_inject_weight"), width, hc.streams))
                .transpose()?,
            streams: hc.streams,
            hidden: cfg.hidden_size,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn mixed(&self, normed: &Tensor) -> Result<Tensor> {
        let streams = self.streams as f64;
        #[cfg(feature = "cuda")]
        if inference_nn::cuda::hyper::supports(normed) {
            use inference_nn::cuda::hyper;
            #[allow(clippy::cast_possible_truncation)]
            let down =
                hyper::scaled_silu(&hc_project(&self.mix_down, normed)?, (1.0 / streams) as f32)?;
            let up = hc_project(&self.mix_up, &down)?.to_dtype(normed.dtype())?;
            return hyper::mix(&up, normed, self.streams, self.hidden);
        }
        let mix = (hc_project(&self.mix_down, normed)? / streams)?;
        let mix = inference_tensor::nn::ops::sigmoid(&hc_project(
            &self.mix_up,
            &inference_tensor::nn::ops::silu(&mix)?,
        )?)?;
        let mut split = normed.dims().to_vec();
        split.pop();
        split.extend([self.streams, self.hidden]);
        (mix.reshape(split.clone())? * normed.reshape(split)?)?.mean(D::Minus2)
    }

    /// The block's input, the streams it read, and their write-back weights.
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn read(&self, streams_in: &Tensor) -> Result<HyperRead> {
        let normed = self.hc_norm.forward(streams_in)?;
        let inject = self
            .inject
            .as_ref()
            .map(|inject| hc_project(inject, &normed))
            .expect("a block's hyper-connection writes back")?;
        #[cfg(feature = "cuda")]
        let inject = if inference_nn::cuda::hyper::supports(&inject) {
            #[allow(clippy::cast_possible_truncation)]
            inference_nn::cuda::hyper::inject_weights(&inject, (1.0 / self.streams as f64) as f32)?
        } else {
            (inference_tensor::nn::ops::sigmoid(&(inject / self.streams as f64)?)? * 2.0)?
        };
        #[cfg(not(feature = "cuda"))]
        let inject = (inference_tensor::nn::ops::sigmoid(&(inject / self.streams as f64)?)? * 2.0)?;
        Ok(HyperRead {
            input: self.mixed(&normed)?,
            streams_in: streams_in.clone(),
            inject,
        })
    }

    /// The final mixer's read: the trunk's output from its streams.
    pub(super) fn collapse(&self, streams_in: &Tensor) -> Result<Tensor> {
        self.mixed(&self.hc_norm.forward(streams_in)?)
    }

    pub(super) fn residual_tensors(&self, uvb: &UnVarBuilder) {
        self.hc_norm.residual_tensors(&uvb.pp("hc_norm"));
    }
}

impl HyperRead {
    /// The branch the block writes into every stream: `output * inject_s`, `[.., streams * hidden]`.
    pub(super) fn injection(&self, output: &Tensor) -> Result<Tensor> {
        let dims = output.dims().to_vec();
        let mut split = dims.clone();
        split.pop();
        split.extend([
            1,
            *dims.last().expect("a block output has a last dimension"),
        ]);
        let mut weights = self.inject.dims().to_vec();
        weights.push(1);
        let mut flat = dims;
        *flat.last_mut().expect("checked above") *= self.inject.dim(D::Minus1)?;
        output
            .reshape(split)?
            .broadcast_mul(&self.inject.reshape(weights)?)?
            .reshape(flat)
    }

    /// `streams + output * inject_s` into every stream: the block's write-back.
    pub(super) fn write_back(&self, streams: &Tensor, output: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if inference_nn::cuda::hyper::supports(streams) {
            return inference_nn::cuda::hyper::inject_add(streams, output, &self.inject);
        }
        streams + self.injection(output)?
    }

    pub(super) fn streams_in(&self) -> &Tensor {
        &self.streams_in
    }
}
