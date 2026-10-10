//! NeMo's `.nemo` checkpoints: an uncompressed tar of `model_config.yaml` and a torch `model_weights.ckpt`, the
//! weights read in place from the archive.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inference_tensor::nn::VarBuilder;
use inference_tensor::pickle::PthTensors;
use inference_tensor::{DType, Device, Error, Result, Tensor};
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::parakeet::EncoderConfig;

pub const EXTENSION: &str = "nemo";
const CONFIG: &str = "model_config.yaml";
const WEIGHTS: &str = "model_weights.ckpt";
// older NeMo saved `.nemo` as a gzipped tar
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
// NeMo's dw_striding pre-encode hard-codes its convs rather than configuring them
const SUBSAMPLING_KERNEL: usize = 3;
const SUBSAMPLING_STRIDE: usize = 2;

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

/// An opened `.nemo`: its config text, and where its weights sit in the archive.
pub struct NemoArchive {
    path: PathBuf,
    config: String,
    weights: (u64, u64),
}

#[derive(Deserialize)]
struct Target {
    target: Option<String>,
}

impl NemoArchive {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut magic = [0u8; 2];
        std::io::Read::read_exact(&mut file, &mut magic)?;
        if magic == GZIP_MAGIC {
            return Err(msg(format!(
                "`{}` is a gzipped `.nemo`; unpack it to a plain tar (`gunzip -c`) to load it",
                path.display()
            )));
        }
        std::io::Seek::rewind(&mut file)?;
        let mut archive = tar::Archive::new(file);
        let (mut config, mut weights) = (None, None);
        for entry in archive.entries_with_seek()? {
            let mut entry = entry?;
            let name = entry
                .path()?
                .file_name()
                .map(|n| n.to_string_lossy().into_owned());
            match name.as_deref() {
                Some(CONFIG) => {
                    let mut text = String::new();
                    std::io::Read::read_to_string(&mut entry, &mut text)?;
                    config = Some(text);
                }
                Some(WEIGHTS) => weights = Some((entry.raw_file_position(), entry.size())),
                _ => {}
            }
        }
        let missing = |file: &str| msg(format!("`{}` holds no `{file}`", path.display()));
        Ok(Self {
            path: path.to_owned(),
            config: config.ok_or_else(|| missing(CONFIG))?,
            weights: weights.ok_or_else(|| missing(WEIGHTS))?,
        })
    }

    pub fn config<T: DeserializeOwned>(&self) -> Result<T> {
        serde_saphyr::from_str(&self.config).map_err(msg)
    }

    /// The NeMo class the checkpoint restores to, as `nemo.collections.asr.models.SortformerEncLabelModel`.
    pub fn target(&self) -> Result<Option<String>> {
        Ok(self.config::<Target>()?.target)
    }

    /// Every tensor, each name passed through `rename`, as a builder on `device` in `dtype`.
    pub fn var_builder(
        &self,
        rename: impl Fn(&str) -> String,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        let (offset, len) = self.weights;
        let tensors = PthTensors::in_range(&self.path, offset, len, None)?;
        let mut renamed: HashMap<String, Tensor> = HashMap::new();
        for name in tensors.tensor_infos().keys() {
            if let Some(t) = tensors.get(name)? {
                renamed.insert(rename(name), t);
            }
        }
        Ok(VarBuilder::from_tensors(renamed, dtype, device))
    }
}

/// Writes a `.nemo` of `config` and `tensors`, as tests build tiny checkpoints.
pub fn write_nemo(path: &Path, config: &str, tensors: &[(&str, &Tensor)]) -> Result<()> {
    let mut weights = std::io::Cursor::new(Vec::new());
    inference_tensor::pickle::write_pth(&mut weights, tensors)?;
    let mut builder = tar::Builder::new(std::fs::File::create(path)?);
    for (name, data) in [
        (CONFIG, config.as_bytes()),
        (WEIGHTS, weights.get_ref().as_slice()),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, name, data)?;
    }
    builder.finish()?;
    Ok(())
}

/// The `encoder:` section of a FastConformer checkpoint's config.
#[derive(Debug, Clone, Deserialize)]
pub struct NemoEncoderConfig {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub subsampling: String,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub self_attention_model: String,
    pub conv_kernel_size: usize,
    pub xscaling: bool,
    #[serde(default)]
    pub pos_emb_max_len: Option<usize>,
}

impl NemoEncoderConfig {
    /// The same encoder as transformers' Parakeet configures it; NeMo's linears and convs all carry biases.
    pub fn parakeet(&self) -> Result<EncoderConfig> {
        if self.subsampling != "dw_striding" || self.self_attention_model != "rel_pos" {
            return Err(msg(format!(
                "a `{}` pre-encode with `{}` attention is not a FastConformer this loads",
                self.subsampling, self.self_attention_model
            )));
        }
        Ok(EncoderConfig {
            hidden_size: self.d_model,
            intermediate_size: self.d_model * self.ff_expansion_factor,
            num_hidden_layers: self.n_layers,
            num_attention_heads: self.n_heads,
            num_mel_bins: self.feat_in,
            conv_kernel_size: self.conv_kernel_size,
            subsampling_conv_channels: self.subsampling_conv_channels,
            subsampling_conv_kernel_size: SUBSAMPLING_KERNEL,
            subsampling_conv_stride: SUBSAMPLING_STRIDE,
            subsampling_factor: self.subsampling_factor,
            attention_bias: true,
            convolution_bias: true,
            scale_input: self.xscaling,
        })
    }
}

// NeMo's FastConformer names to the transformers names the Parakeet encoder loads
const ENCODER_RENAMES: [(&str, &str); 10] = [
    (".pre_encode.conv.", ".subsampling.layers."),
    (".pre_encode.out.", ".subsampling.linear."),
    (".self_attn.linear_q.", ".self_attn.q_proj."),
    (".self_attn.linear_k.", ".self_attn.k_proj."),
    (".self_attn.linear_v.", ".self_attn.v_proj."),
    (".self_attn.linear_out.", ".self_attn.o_proj."),
    (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
    (".self_attn.pos_bias_u", ".self_attn.bias_u"),
    (".self_attn.pos_bias_v", ".self_attn.bias_v"),
    (".conv.batch_norm.", ".conv.norm."),
];

/// A NeMo checkpoint name under `encoder.` as the Parakeet encoder names it; others pass through.
pub fn parakeet_encoder_name(name: &str) -> String {
    if !name.starts_with("encoder.") {
        return name.to_string();
    }
    ENCODER_RENAMES
        .iter()
        .fold(name.to_string(), |n, (from, to)| n.replace(from, to))
}

/// The NeMo name of a Parakeet encoder name: the inverse of [`parakeet_encoder_name`].
pub fn nemo_encoder_name(name: &str) -> String {
    if !name.starts_with("encoder.") {
        return name.to_string();
    }
    ENCODER_RENAMES
        .iter()
        .fold(name.to_string(), |n, (nemo, ours)| n.replace(ours, nemo))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastconformer_names_become_the_parakeet_encoders() {
        for (nemo, ours) in [
            (
                "encoder.pre_encode.conv.5.weight",
                "encoder.subsampling.layers.5.weight",
            ),
            (
                "encoder.pre_encode.out.bias",
                "encoder.subsampling.linear.bias",
            ),
            (
                "encoder.layers.3.self_attn.linear_pos.weight",
                "encoder.layers.3.self_attn.relative_k_proj.weight",
            ),
            (
                "encoder.layers.0.self_attn.pos_bias_u",
                "encoder.layers.0.self_attn.bias_u",
            ),
            (
                "encoder.layers.16.conv.batch_norm.running_var",
                "encoder.layers.16.conv.norm.running_var",
            ),
            (
                "encoder.layers.2.conv.depthwise_conv.weight",
                "encoder.layers.2.conv.depthwise_conv.weight",
            ),
            (
                "transformer_encoder.layers.0.first_sub_layer.query_net.weight",
                "transformer_encoder.layers.0.first_sub_layer.query_net.weight",
            ),
        ] {
            assert_eq!(parakeet_encoder_name(nemo), ours);
            assert_eq!(nemo_encoder_name(ours), nemo);
        }
    }

    #[test]
    fn an_archive_yields_its_config_and_in_place_weights() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tiny.nemo");
        let bias = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        let yaml =
            "target: nemo.collections.asr.models.SortformerEncLabelModel\nsample_rate: 16000\n";
        write_nemo(&path, yaml, &[("encoder.pre_encode.out.bias", &bias)])?;

        let archive = NemoArchive::open(&path)?;
        assert_eq!(
            archive.target()?.as_deref(),
            Some("nemo.collections.asr.models.SortformerEncLabelModel")
        );
        let vb = archive.var_builder(parakeet_encoder_name, DType::F32, &Device::Cpu)?;
        let read = vb.get(3, "encoder.subsampling.linear.bias")?;
        assert_eq!(read.to_vec1::<f32>()?, [1., 2., 3.]);
        Ok(())
    }
}
