//! The `img_projection` Phi-3-V and Phi-4-MM put after their vision towers, and the 6-D reshape their HD crops use.

use std::fmt::Debug;
use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::Module;
use inference_tensor::{DType, Device, Result, Shape, Tensor, shape::ShapeWithOneHole};

pub(crate) trait ModuleWithMetadata: Module + Debug + Send + Sync {
    fn device(&self) -> Device;
    fn dtype(&self) -> DType;
}

#[derive(Debug)]
struct QuantMethodWrapper(Arc<dyn QuantMethod>);

impl Module for QuantMethodWrapper {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

impl ModuleWithMetadata for QuantMethodWrapper {
    fn device(&self) -> Device {
        self.0.unquant_weight_bias().unwrap().0.device().clone()
    }
    fn dtype(&self) -> DType {
        self.0.unquant_weight_bias().unwrap().0.dtype()
    }
}

impl ModuleWithMetadata for inference_tensor::nn::Activation {
    fn device(&self) -> Device {
        unreachable!()
    }
    fn dtype(&self) -> DType {
        unreachable!()
    }
}

#[derive(Debug)]
pub(crate) struct BigShapeWithOneHole(pub (usize, usize, usize, usize, usize, ()));

fn hole_size(el_count: usize, prod_d: usize, s: &dyn std::fmt::Debug) -> Result<usize> {
    if prod_d == 0 {
        inference_tensor::bail!("cannot reshape tensor of {el_count} elements to {s:?}")
    }
    if !el_count.is_multiple_of(prod_d) {
        inference_tensor::bail!("cannot reshape tensor with {el_count} elements to {s:?}")
    }
    Ok(el_count / prod_d)
}

impl ShapeWithOneHole for BigShapeWithOneHole {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, d4, d5, ()) = self.0;
        let d = hole_size(el_count, d1 * d2 * d3 * d4 * d5, &self)?;
        Ok((d1, d2, d3, d4, d5, d).into())
    }
}

#[derive(Debug)]
pub(crate) struct EmbeddingLayers(pub Vec<Box<dyn ModuleWithMetadata>>);

impl Module for EmbeddingLayers {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.0 {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

/// `img_projection`: one linear, or a two-layer GELU MLP whose input is `hd_dim_in` wide under the HD transform.
/// Each layer's unquantized weights are pushed onto `tensors` under their checkpoint names.
pub(crate) fn image_projection(
    projection_cls: &str,
    use_hd_transform: bool,
    image_dim_out: usize,
    hd_dim_in: usize,
    hidden_size: usize,
    vb: &ShardedVarBuilder,
    tensors: &mut Vec<(String, Tensor)>,
) -> Result<EmbeddingLayers> {
    let mut linear =
        |in_dim: usize, out_dim: usize, name: &str| -> Result<Box<dyn ModuleWithMetadata>> {
            let layer = inference_quant::linear_b(in_dim, out_dim, true, &None, vb.pp(name))?;
            let (weight, bias) = layer.unquant_weight_bias().unwrap();
            tensors.push((format!("{name}.weight"), weight));
            if let Some(bias) = bias {
                tensors.push((format!("{name}.bias"), bias));
            }
            Ok(Box::new(QuantMethodWrapper(layer)))
        };
    let layers = match projection_cls {
        "linear" => vec![linear(image_dim_out, hidden_size, "img_projection")?],
        "mlp" => {
            let dim_in = if use_hd_transform {
                hd_dim_in
            } else {
                image_dim_out
            };
            vec![
                linear(dim_in, hidden_size, "img_projection.0")?,
                Box::new(inference_tensor::nn::Activation::Gelu),
                linear(hidden_size, hidden_size, "img_projection.2")?,
            ]
        }
        _ => inference_tensor::bail!("projection_cls=`{projection_cls}` not implemented."),
    };
    Ok(EmbeddingLayers(layers))
}
