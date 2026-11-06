//! A Parakeet FastConformer audio tower in llama.cpp's mmproj layout (`clip.audio.projector_type parakeet`): one table
//! of llama.cpp names against transformers' `ParakeetEncoder` names drives both the writer and the binding.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use inference_quant::{GgufBindingMap, GgufTensorBinding};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

use super::mmproj_writer::{Checkpoint, MmprojSpec, MmprojTensor, Role, Source};
use super::multimodal_binding_utils::TensorInventory;

pub const PROJECTOR: &str = "parakeet";
pub const AUDIO_PROJECTOR_TYPE: &str = "clip.audio.projector_type";
const BLOCK_PREFIX: &str = "a.blk.";
const SUBSAMPLING_CONVS: [usize; 5] = [0, 2, 3, 5, 6];
// llama.cpp's Parakeet graph: 16 kHz, stride-2 subsampling convs, the encoder's LayerNorm epsilon
const SAMPLE_RATE: u32 = 16_000;
const SUBSAMPLING_STRIDE: usize = 2;
const LAYER_NORM_EPS: f32 = 1e-5;

/// How a tensor's checkpoint shape and its GGUF shape (both row-major) relate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Same,
    /// A conv bias `(c)` stored as `(1, c, 1, 1)`.
    ConvBias,
    /// A depthwise kernel `(c, 1, k)` stored as `(1, 1, c, k)`.
    Depthwise,
    /// A pointwise conv `(o, i, 1)` stored as the matrix `(o, i)`.
    Pointwise,
}

impl Layout {
    fn to_gguf(self, native: &[usize]) -> Vec<usize> {
        match (self, native) {
            (Self::ConvBias, [c]) => vec![1, *c, 1, 1],
            (Self::Depthwise, [c, 1, k]) => vec![1, 1, *c, *k],
            (Self::Pointwise, [o, i, 1]) => vec![*o, *i],
            _ => native.to_vec(),
        }
    }

    fn to_native(self, gguf: &[usize]) -> Vec<usize> {
        match (self, gguf) {
            (Self::ConvBias, [1, c, 1, 1]) => vec![*c],
            (Self::Depthwise, [1, 1, c, k]) => vec![*c, 1, *k],
            (Self::Pointwise, [o, i]) => vec![*o, *i, 1],
            _ => gguf.to_vec(),
        }
    }
}

// (llama.cpp name, checkpoint name, role, layout) of one encoder block; `{}` is the block's index
const BLOCK: &[(&str, &str, Role, Layout)] = &[
    (
        "attn_q.weight",
        "self_attn.q_proj.weight",
        Role::Matrix,
        Layout::Same,
    ),
    (
        "attn_q.bias",
        "self_attn.q_proj.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "attn_k.weight",
        "self_attn.k_proj.weight",
        Role::Matrix,
        Layout::Same,
    ),
    (
        "attn_k.bias",
        "self_attn.k_proj.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "attn_v.weight",
        "self_attn.v_proj.weight",
        Role::Matrix,
        Layout::Same,
    ),
    (
        "attn_v.bias",
        "self_attn.v_proj.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "attn_out.weight",
        "self_attn.o_proj.weight",
        Role::Matrix,
        Layout::Same,
    ),
    (
        "attn_out.bias",
        "self_attn.o_proj.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "linear_pos.weight",
        "self_attn.relative_k_proj.weight",
        Role::Matrix,
        Layout::Same,
    ),
    ("pos_bias_u", "self_attn.bias_u", Role::Float, Layout::Same),
    ("pos_bias_v", "self_attn.bias_v", Role::Float, Layout::Same),
    (
        "ffn_up.weight",
        "feed_forward1.linear1.weight",
        Role::Matrix,
        Layout::Same,
    ),
    (
        "ffn_up.bias",
        "feed_forward1.linear1.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "ffn_down.weight",
        "feed_forward1.linear2.weight",
        Role::Matrix,
        Layout::Same,
    ),
    (
        "ffn_down.bias",
        "feed_forward1.linear2.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "ffn_up_1.weight",
        "feed_forward2.linear1.weight",
        Role::Matrix,
        Layout::Same,
    ),
    (
        "ffn_up_1.bias",
        "feed_forward2.linear1.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "ffn_down_1.weight",
        "feed_forward2.linear2.weight",
        Role::Matrix,
        Layout::Same,
    ),
    (
        "ffn_down_1.bias",
        "feed_forward2.linear2.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "conv_pw1.weight",
        "conv.pointwise_conv1.weight",
        Role::Float,
        Layout::Pointwise,
    ),
    (
        "conv_pw1.bias",
        "conv.pointwise_conv1.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "conv_dw.weight",
        "conv.depthwise_conv.weight",
        Role::Float,
        Layout::Depthwise,
    ),
    (
        "conv_dw.bias",
        "conv.depthwise_conv.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "conv_pw2.weight",
        "conv.pointwise_conv2.weight",
        Role::Float,
        Layout::Pointwise,
    ),
    (
        "conv_pw2.bias",
        "conv.pointwise_conv2.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "conv_norm.weight",
        "conv.norm.weight",
        Role::Float,
        Layout::Same,
    ),
    (
        "conv_norm.bias",
        "conv.norm.bias",
        Role::Float,
        Layout::Same,
    ),
    (
        "conv_norm_mean",
        "conv.norm.running_mean",
        Role::Float,
        Layout::Same,
    ),
    (
        "conv_norm_var",
        "conv.norm.running_var",
        Role::Float,
        Layout::Same,
    ),
    (
        "ffn_norm.%",
        "norm_feed_forward1.%",
        Role::Float,
        Layout::Same,
    ),
    ("ln1.%", "norm_self_att.%", Role::Float, Layout::Same),
    ("norm_conv.%", "norm_conv.%", Role::Float, Layout::Same),
    (
        "ffn_norm_1.%",
        "norm_feed_forward2.%",
        Role::Float,
        Layout::Same,
    ),
    ("ln2.%", "norm_out.%", Role::Float, Layout::Same),
];
// a `%` in a `BLOCK` name is a LayerNorm's weight and bias
const NORM_PARTS: [&str; 2] = ["weight", "bias"];

/// Every (llama.cpp name, checkpoint name, role, layout) of the tower under `encoder` (transformers' `encoder.`) and
/// `projection` (the projector into the decoder), for `layers` blocks.
fn pairs(encoder: &str, projection: &str, layers: usize) -> Vec<(String, String, Role, Layout)> {
    let mut out = Vec::new();
    for conv in SUBSAMPLING_CONVS {
        let native = format!("{encoder}.subsampling.layers.{conv}");
        out.push((
            format!("a.conv1d.{conv}.weight"),
            format!("{native}.weight"),
            Role::Float,
            Layout::Same,
        ));
        out.push((
            format!("a.conv1d.{conv}.bias"),
            format!("{native}.bias"),
            Role::Float,
            Layout::ConvBias,
        ));
    }
    out.push((
        "a.pre_encode.out.weight".into(),
        format!("{encoder}.subsampling.linear.weight"),
        Role::Matrix,
        Layout::Same,
    ));
    out.push((
        "a.pre_encode.out.bias".into(),
        format!("{encoder}.subsampling.linear.bias"),
        Role::Float,
        Layout::Same,
    ));
    for (gguf, native) in [("a.mel_filters", "fb"), ("a.window", "window")] {
        out.push((
            gguf.into(),
            format!("{encoder}.feature_extractor.featurizer.{native}"),
            Role::Float,
            Layout::Same,
        ));
    }
    let norm = |gguf: &str, native: &str| {
        NORM_PARTS.map(|part| {
            (
                gguf.replace('%', part),
                native.replace('%', part),
                Role::Float,
                Layout::Same,
            )
        })
    };
    for layer in 0..layers {
        let theirs = format!("{BLOCK_PREFIX}{layer}");
        let ours = format!("{encoder}.layers.{layer}");
        let entries =
            BLOCK
                .iter()
                .flat_map(|&(gguf, native, role, layout)| match gguf.contains('%') {
                    true => norm(gguf, native).to_vec(),
                    false => vec![(gguf.to_string(), native.to_string(), role, layout)],
                });
        for (gguf, native, role, layout) in entries {
            out.push((
                format!("{theirs}.{gguf}"),
                format!("{ours}.{native}"),
                role,
                layout,
            ));
        }
    }
    out.push((
        "mm.a.norm_pre.weight".into(),
        format!("{projection}.norm.weight"),
        Role::Float,
        Layout::Same,
    ));
    for (gguf, native) in [("mm.a.mlp.1", "linear1"), ("mm.a.mlp.2", "linear2")] {
        out.push((
            format!("{gguf}.weight"),
            format!("{projection}.{native}.weight"),
            Role::Matrix,
            Layout::Same,
        ));
        out.push((
            format!("{gguf}.bias"),
            format!("{projection}.{native}.bias"),
            Role::Float,
            Layout::Same,
        ));
    }
    out
}

fn config_usize(config: &serde_json::Value, key: &str) -> Result<usize> {
    config[key]
        .as_u64()
        .map(|value| value as usize)
        .with_context(|| format!("`sound_config.{key}` is required"))
}

/// The tower's keys and tensors for the writer, from transformers' `ParakeetEncoderConfig` (`sound`) and the
/// checkpoint's tensors under `encoder` and `projection`; tensors the table names but the checkpoint lacks (biases a
/// config turns off) are left out.
pub fn audio_spec(
    checkpoint: &Checkpoint,
    sound: &serde_json::Value,
    encoder: &str,
    projection: &str,
    lm_hidden: usize,
) -> Result<MmprojSpec> {
    let factor = config_usize(sound, "subsampling_factor")?;
    if config_usize(sound, "subsampling_conv_stride")? != SUBSAMPLING_STRIDE
        || factor != SUBSAMPLING_STRIDE.pow(3)
    {
        bail!(
            "llama.cpp's Parakeet runs three stride-2 subsampling convs, not this `sound_config`'s"
        );
    }
    let layers = config_usize(sound, "num_hidden_layers")?;
    let u32_of = |key: &str| -> Result<Value> { Ok(Value::U32(config_usize(sound, key)? as u32)) };
    let metadata = vec![
        ("clip.has_audio_encoder".to_string(), Value::Bool(true)),
        (
            AUDIO_PROJECTOR_TYPE.to_string(),
            Value::String(PROJECTOR.to_string()),
        ),
        (
            "clip.audio.num_mel_bins".to_string(),
            u32_of("num_mel_bins")?,
        ),
        (
            "clip.audio.attention.layer_norm_epsilon".to_string(),
            Value::F32(LAYER_NORM_EPS),
        ),
        (
            "clip.audio.subsampling_factor".to_string(),
            Value::U32(factor as u32),
        ),
        (
            "clip.audio.conv_kernel_size".to_string(),
            u32_of("conv_kernel_size")?,
        ),
        (
            "clip.audio.projection_dim".to_string(),
            Value::U32(lm_hidden as u32),
        ),
        (
            "clip.audio.embedding_length".to_string(),
            u32_of("hidden_size")?,
        ),
        (
            "clip.audio.feed_forward_length".to_string(),
            u32_of("intermediate_size")?,
        ),
        (
            "clip.audio.block_count".to_string(),
            Value::U32(layers as u32),
        ),
        (
            "clip.audio.attention.head_count".to_string(),
            u32_of("num_attention_heads")?,
        ),
    ];
    let mut tensors = Vec::new();
    for (gguf, native, role, layout) in pairs(encoder, projection, layers) {
        if !checkpoint.contains(&native) {
            continue;
        }
        let shape = checkpoint.shape(&native)?;
        let source = match layout {
            Layout::Same => Source::tensor(native),
            _ => Source::tensor(native).reshape(layout.to_gguf(&shape)),
        };
        tensors.push(MmprojTensor::new(gguf, source, role));
    }
    if !tensors.iter().any(|t| t.name.starts_with(BLOCK_PREFIX)) {
        bail!("the checkpoint has no `{encoder}.layers.*` tensors");
    }
    Ok(MmprojSpec { metadata, tensors })
}

/// The tower's tensors under the checkpoint's names (`encoder`, `projection`), as `audio_spec` wrote them.
pub(super) fn bind_audio(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    encoder: &str,
    projection: &str,
) -> Result<()> {
    let layers = inventory.require_layers(BLOCK_PREFIX, PROJECTOR)?.len();
    for (gguf, native, _, layout) in pairs(encoder, projection, layers) {
        if !inventory.contains(&gguf) {
            continue;
        }
        let binding = match layout {
            Layout::Same => GgufTensorBinding::tensor(gguf),
            _ => {
                let shape = layout.to_native(inventory.shape(&gguf)?);
                GgufTensorBinding::tensor(gguf).reshape(shape)
            }
        };
        bindings.insert(native, binding);
    }
    Ok(())
}

fn metadata_usize(metadata: &HashMap<String, Value>, key: &str) -> Result<usize> {
    match metadata.get(key) {
        Some(Value::U32(value)) => Ok(*value as usize),
        Some(Value::U64(value)) => Ok(*value as usize),
        Some(_) => bail!("GGUF metadata `{key}` must be an integer"),
        None => bail!("GGUF metadata `{key}` is required"),
    }
}

/// transformers' `ParakeetEncoderConfig` (with the projector's width) that an mmproj's audio keys and shapes describe.
pub fn sound_config(
    metadata: &HashMap<String, Value>,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<serde_json::Value> {
    let shape = |name: &str| {
        shapes
            .get(name)
            .with_context(|| format!("the {PROJECTOR} projector has no `{name}`"))
    };
    // the first subsampling conv `(channels, 1, k, k)`, the projector's first matrix `(hidden, encoder width)`
    let first_conv = shape("a.conv1d.0.weight")?;
    let projector_in = shape("mm.a.mlp.1.weight")?;
    Ok(json!({
        "model_type": PROJECTOR,
        "hidden_size": metadata_usize(metadata, "clip.audio.embedding_length")?,
        "num_attention_heads": metadata_usize(metadata, "clip.audio.attention.head_count")?,
        "num_hidden_layers": metadata_usize(metadata, "clip.audio.block_count")?,
        "intermediate_size": metadata_usize(metadata, "clip.audio.feed_forward_length")?,
        "conv_kernel_size": metadata_usize(metadata, "clip.audio.conv_kernel_size")?,
        "convolution_bias": shapes.contains_key("a.blk.0.conv_dw.bias"),
        "attention_bias": shapes.contains_key("a.blk.0.attn_q.bias"),
        "subsampling_conv_channels": first_conv[0],
        "subsampling_conv_kernel_size": first_conv[first_conv.len() - 1],
        "subsampling_conv_stride": SUBSAMPLING_STRIDE,
        "subsampling_factor": metadata_usize(metadata, "clip.audio.subsampling_factor")?,
        "num_mel_bins": metadata_usize(metadata, "clip.audio.num_mel_bins")?,
        "projection_hidden_size": projector_in[0],
        "projection_bias": shapes.contains_key("mm.a.mlp.1.bias"),
        "sampling_rate": SAMPLE_RATE,
    }))
}

/// Whether the projector carries a `parakeet` audio tower.
pub fn has_audio(metadata: &HashMap<String, Value>) -> bool {
    matches!(metadata.get(AUDIO_PROJECTOR_TYPE), Some(Value::String(projector)) if projector == PROJECTOR)
}

#[cfg(test)]
mod tests {
    use super::Layout;

    // the writer's shape and the binding's are each other's inverse, for every reshaped kind
    #[test]
    fn each_layout_round_trips_its_checkpoint_shape() {
        for (layout, native, gguf) in [
            (Layout::ConvBias, vec![256], vec![1, 256, 1, 1]),
            (Layout::Depthwise, vec![1024, 1, 9], vec![1, 1, 1024, 9]),
            (Layout::Pointwise, vec![2048, 1024, 1], vec![2048, 1024]),
            (Layout::Same, vec![8, 128], vec![8, 128]),
        ] {
            assert_eq!(layout.to_gguf(&native), gguf, "{layout:?}");
            assert_eq!(layout.to_native(&gguf), native, "{layout:?}");
        }
    }
}
