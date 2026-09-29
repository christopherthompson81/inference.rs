use std::fmt::{self, Display};
use std::sync::Arc;

use anyhow::Result;
use regex::Regex;

use crate::utils::varbuilder_utils::DeviceForLoadTensor;

pub const LAYER_INDEX_PATTERN: &str = r"\.layers\.(\d+)\.";

const NON_MAPPED_COMPONENTS: &[&str] = &[
    "audio",
    "audio_model",
    "audio_tower",
    "image",
    "mtp",
    "visual",
    "vision",
    "vision_encoder",
    "vision_model",
    "vision_tower",
];

pub fn standard_layer_index(tensor_name: &str) -> Option<usize> {
    let components = tensor_name.split('.').collect::<Vec<_>>();
    if components
        .iter()
        .any(|component| NON_MAPPED_COMPONENTS.contains(component))
    {
        return None;
    }
    components.windows(2).find_map(|window| {
        (window[0] == "layers")
            .then(|| window[1].parse::<usize>().ok())
            .flatten()
    })
}

#[derive(Clone, Debug)]
pub enum NonMappedSubModel {
    Vision,
    Audio,
}

impl Display for NonMappedSubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonMappedSubModel::Vision => write!(f, "vision"),
            NonMappedSubModel::Audio => write!(f, "audio"),
        }
    }
}

/// Places each tensor on the device of the layer its name indexes (capped at `num_layers`), else the base device.
pub fn layer_indexed_device(
    pattern: &str,
    num_layers: usize,
    loading_isq: bool,
) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
    if loading_isq {
        return Ok(Arc::new(|_| DeviceForLoadTensor::Base));
    }
    let re = Regex::new(pattern)?;
    Ok(Arc::new(move |name: String| {
        re.captures(&name)
            .and_then(|captures| captures.get(1))
            .and_then(|m| m.as_str().parse::<usize>().ok())
            .map(|l| DeviceForLoadTensor::Idx(l.min(num_layers)))
            .unwrap_or(DeviceForLoadTensor::Base)
    }))
}

#[cfg(test)]
mod tests {
    use super::{layer_indexed_device, DeviceForLoadTensor, LAYER_INDEX_PATTERN};

    #[test]
    fn layer_tensors_follow_their_layer_and_the_rest_stay_on_the_base_device() {
        let place = layer_indexed_device(LAYER_INDEX_PATTERN, 4, false).unwrap();
        let idx = |name: &str| match place(name.to_string()) {
            DeviceForLoadTensor::Idx(i) => Some(i),
            DeviceForLoadTensor::Base => None,
        };
        assert_eq!(idx("model.layers.2.mlp.up_proj.weight"), Some(2));
        assert_eq!(idx("model.layers.9.mlp.up_proj.weight"), Some(4));
        assert_eq!(idx("model.embed_tokens.weight"), None);

        let isq = layer_indexed_device(LAYER_INDEX_PATTERN, 4, true).unwrap();
        assert!(matches!(
            isq("model.layers.2.mlp.up_proj.weight".to_string()),
            DeviceForLoadTensor::Base
        ));
    }
}
