//! A `.cimatrix` (collected online, keyed by the layers' tracking keys) as an imatrix under a GGUF's tensor names.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

// llama.cpp's stand-in for an expert that saw no tokens
const UNSEEN_EXPERT_IMPORTANCE: f32 = 1.0;

/// What [`cimatrix_for_gguf`] matched.
#[derive(Debug, Default)]
pub struct ImatrixMapping {
    pub entries: HashMap<String, Vec<f32>>,
    /// Keys no GGUF tensor binds to (an audio tower beside a text GGUF, say).
    pub unmatched: Vec<String>,
    /// Expert slices that saw no tokens, filled as llama.cpp fills them.
    pub unseen_experts: usize,
}

/// `cimatrix`'s entries under the names of the tensors `gguf`'s layers load: a key matches a native name it equals or
/// ends with (after a dot), so a checkpoint's roots (`llm.`, `backbone.`) fall away.
pub fn cimatrix_for_gguf(cimatrix: &Path, gguf: &Path) -> Result<ImatrixMapping> {
    let collected = inference_quant::CollectedImatrixData::load_imatrix(cimatrix)?.0;
    let archive = inference_quant::GgufArchive::open([gguf])?;
    let names = crate::pipeline::gguf_tensor_names(&archive)?;
    let natives = names
        .iter()
        .filter_map(|(native, gguf)| Some((native.strip_suffix(".weight")?, gguf.as_str())))
        .collect::<Vec<_>>();
    let mut mapping = ImatrixMapping::default();
    let mut keys = collected.keys().collect::<Vec<_>>();
    keys.sort();
    for key in keys {
        let matched = natives.iter().find(|(native, _)| {
            key == native
                || key
                    .strip_suffix(native)
                    .is_some_and(|root| root.ends_with('.'))
        });
        let Some((_, gguf_name)) = matched else {
            mapping.unmatched.push(key.clone());
            continue;
        };
        let mut values = collected[key].clone();
        let shape = archive.tensors()[*gguf_name].shape();
        if shape.len() == 3 {
            let per_expert = values.len() / shape[0].max(1);
            for slice in values.chunks_mut(per_expert.max(1)) {
                if slice.iter().all(|v| *v == 0.0) {
                    slice.fill(UNSEEN_EXPERT_IMPORTANCE);
                    mapping.unseen_experts += 1;
                }
            }
        }
        mapping.entries.insert(gguf_name.to_string(), values);
    }
    Ok(mapping)
}
