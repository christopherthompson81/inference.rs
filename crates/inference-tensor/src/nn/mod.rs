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

pub use activation::Activation;
pub use batch_norm::{BatchNorm, BatchNormConfig};
pub use conv::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, ConvTranspose1d, ConvTranspose1dConfig, conv2d,
};
pub use embedding::{Embedding, embedding};
pub use group_norm::GroupNorm;
pub use init::Init;
pub use layer_norm::{LayerNorm, LayerNormConfig, layer_norm};
pub use linear::{Linear, linear};
pub use ops::Dropout;
pub use var_builder::VarBuilder;
pub use var_map::VarMap;

pub use crate::{Module, ModuleT};
