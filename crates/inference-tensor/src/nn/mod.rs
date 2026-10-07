//! Neural-net building blocks (layers, activations, ops, weight loading), derived from candle-nn.

pub mod activation;
pub mod batch_norm;
pub mod conv;
pub mod embedding;
pub mod group_norm;
pub mod init;
pub mod layer_norm;
pub mod linear;
pub mod ops;
pub mod var_builder;
pub mod var_map;

pub use activation::{prelu, Activation, PReLU};
pub use batch_norm::{batch_norm, BatchNorm, BatchNormConfig};
pub use conv::{
    conv1d, conv1d_no_bias, conv2d, conv2d_no_bias, conv_transpose1d, conv_transpose1d_no_bias,
    conv_transpose2d, conv_transpose2d_no_bias, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig,
    ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig,
};
pub use embedding::{embedding, Embedding};
pub use group_norm::{group_norm, GroupNorm};
pub use init::Init;
pub use layer_norm::{
    layer_norm, layer_norm_no_bias, rms_norm, LayerNorm, LayerNormConfig, RmsNorm,
};
pub use linear::{linear, linear_b, linear_no_bias, Linear};
pub use ops::Dropout;
pub use var_builder::VarBuilder;
pub use var_map::VarMap;

pub use crate::{Module, ModuleT};
