//! Builds a tiny random-weight SpeechBrain language identifier (VoxLingua107's layout) at test time: its
//! hyperparams, label encoder, `config.json` and the two torch checkpoints, under SpeechBrain's own names.
#![allow(dead_code)]

use inference_models_speech::voxlingua::{self, VoxLingua};
use inference_tensor::{Device, Tensor};

#[path = "recording.rs"]
mod recording;

const CONFIG: &str = r#"{"speechbrain_interface": "EncoderClassifier"}"#;
const HYPERPARAMS: &str = "\
n_mels: 20
compute_features: !new:speechbrain.lobes.features.Fbank
  n_mels: 20
embedding_model: !new:speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN
  input_size: 20
  channels: [16, 16, 16, 16, 48]
  kernel_sizes: [5, 3, 3, 3, 1]
  dilations: [1, 2, 3, 4, 1]
  attention_channels: 8
  se_channels: 8
  lin_neurons: 8
classifier: !new:speechbrain.lobes.models.Xvector.Classifier
  input_shape: [null, null, 8]
  lin_blocks: 1
  lin_neurons: 12
  out_neurons: !ref <n_mels>
";
pub const EMBEDDING_DIM: usize = 8;
const RUNNING_VAR: &str = "running_var";
const OUT_BIAS: &str = "classifier.out.w.bias";
// a bias this large on the first class makes it the answer for any audio
const FIRST_CLASS_BIAS: f32 = 50.0;

/// A random-weight language identifier over `labels` (`(code, name)`) in a new directory; the first one wins.
pub fn tiny_language_id(labels: &[(&str, &str)]) -> anyhow::Result<tempfile::TempDir> {
    let mut label_text: String = labels
        .iter()
        .enumerate()
        .map(|(i, (code, name))| format!("'{code}: {name}' => {i}\n"))
        .collect();
    label_text.push_str("================\n'starting_index' => 0\n");
    let dir = recording::record_checkpoint(&[], 1, &[], |vb, _| {
        VoxLingua::new(HYPERPARAMS, &label_text, &vb, &Device::Cpu).map(|_| ())
    })?;
    let weights = dir.path().join("model.safetensors");
    let tensors = inference_tensor::safetensors::load(&weights, &Device::Cpu)?;
    std::fs::remove_file(&weights)?;
    let mut parts: [Vec<(String, Tensor)>; 2] = [Vec::new(), Vec::new()];
    for (name, tensor) in tensors {
        // a BatchNorm's variance is positive
        let tensor = if name.ends_with(RUNNING_VAR) {
            (tensor.abs()? + 1.0)?
        } else if name == OUT_BIAS {
            let mut bias = vec![0f32; labels.len()];
            bias[0] = FIRST_CLASS_BIAS;
            Tensor::new(bias, &Device::Cpu)?
        } else {
            tensor
        };
        let (part, rest) = name.split_once('.').expect("a prefixed name");
        let index = usize::from(part == "classifier");
        parts[index].push((rest.to_string(), tensor));
    }
    for (file, part) in [voxlingua::EMBEDDING_MODEL, voxlingua::CLASSIFIER]
        .iter()
        .zip(&parts)
    {
        let named: Vec<(&str, &Tensor)> = part.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let out = std::fs::File::create(dir.path().join(file))?;
        inference_tensor::pickle::write_pth(std::io::BufWriter::new(out), &named)?;
    }
    std::fs::write(dir.path().join(voxlingua::HYPERPARAMS), HYPERPARAMS)?;
    std::fs::write(dir.path().join(voxlingua::LABELS), label_text)?;
    std::fs::write(dir.path().join("config.json"), CONFIG)?;
    Ok(dir)
}
