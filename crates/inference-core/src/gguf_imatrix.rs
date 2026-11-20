//! A `.cimatrix` (collected online, keyed by the layers' tracking keys) as an imatrix under a GGUF's tensor names.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Result;

// llama.cpp's stand-in for an expert that saw no tokens
const UNSEEN_EXPERT_IMPORTANCE: f32 = 1.0;
// an engine Hessian dump's file extension, the tensor each holds, and the head's key
const HESSIAN_EXTENSION: &str = "safetensors";
pub(crate) const HESSIAN_TENSOR: &str = "hin";
// a symmetric Hessian stored as its packed upper triangle, row-major
pub(crate) const HESSIAN_UPPER_TENSOR: &str = "hin_upper";
const MIRROR_TILE: usize = 64;
const HEAD_KEY: &str = "lm_head";
const OUTPUT_TENSOR: &str = "output.weight";
const TOKEN_EMBD_TENSOR: &str = "token_embd.weight";
// the GGUF keys of a gated delta net's key heads (its groups) and value heads, and its output projection
const SSM_GROUP_COUNT: &str = "ssm.group_count";
const SSM_TIME_STEP_RANK: &str = "ssm.time_step_rank";
const SSM_OUT_SUFFIX: &str = ".ssm_out.weight";
const UP_PROJ_SUFFIX: &str = ".up_proj.weight";
const GATE_PROJ_SUFFIX: &str = ".gate_proj.weight";

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

/// An engine Hessian dump (`inference quantize hessians`) under a GGUF's tensor names.
pub struct HessianDir {
    dense: HashMap<String, PathBuf>,
    experts: HashMap<String, BTreeMap<usize, PathBuf>>,
    // the GDN output projections' column order: the converter tiles V heads by key head, the checkpoint groups them
    ssm_out_order: Option<Vec<usize>>,
    /// Files no GGUF tensor binds to.
    pub unmatched: Vec<String>,
}

impl HessianDir {
    /// The GGUF tensors given a Hessian.
    pub fn len(&self) -> usize {
        self.dense.len() + self.experts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// the full matrix from its packed upper triangle
fn unpack_upper(upper: &[f32]) -> (usize, Vec<f32>) {
    let n = ((8 * upper.len() + 1).isqrt() - 1) / 2;
    let mut full = vec![0f32; n * n];
    let mut k = 0;
    for i in 0..n {
        full[i * n + i..(i + 1) * n].copy_from_slice(&upper[k..k + n - i]);
        k += n - i;
    }
    // the lower triangle mirrored a tile at a time, so the strided side stays in cache
    for bi in (0..n).step_by(MIRROR_TILE) {
        for bj in (0..=bi).step_by(MIRROR_TILE) {
            for i in bi..(bi + MIRROR_TILE).min(n) {
                for j in bj..(bj + MIRROR_TILE).min(i) {
                    full[i * n + j] = full[j * n + i];
                }
            }
        }
    }
    (n, full)
}

fn load_hessian(path: &Path, order: Option<&[usize]>) -> inference_tensor::Result<Vec<f32>> {
    let mut tensors = inference_tensor::safetensors::load(path, &inference_tensor::Device::Cpu)?;
    let (n, values) = if let Some(upper) = tensors.remove(HESSIAN_UPPER_TENSOR) {
        unpack_upper(
            &upper
                .to_dtype(inference_tensor::DType::F32)?
                .to_vec1::<f32>()?,
        )
    } else {
        let h = tensors
            .remove(HESSIAN_TENSOR)
            .ok_or_else(|| {
                inference_tensor::Error::Msg(format!(
                    "{} holds no `{HESSIAN_TENSOR}`",
                    path.display()
                ))
            })?
            .to_dtype(inference_tensor::DType::F32)?;
        (h.dim(0)?, h.flatten_all()?.to_vec1::<f32>()?)
    };
    let Some(order) = order.filter(|o| o.len() == n) else {
        return Ok(values);
    };
    Ok((0..n * n)
        .map(|i| values[order[i / n] * n + order[i % n]])
        .collect())
}

impl inference_quant::requantize::HessianSource for HessianDir {
    fn hessian(
        &self,
        name: &str,
        expert: Option<usize>,
    ) -> inference_tensor::Result<Option<Vec<f32>>> {
        let order = name
            .ends_with(SSM_OUT_SUFFIX)
            .then_some(())
            .and(self.ssm_out_order.as_deref());
        let path = match expert {
            None => self.dense.get(name),
            Some(e) => self.experts.get(name).and_then(|m| m.get(&e)),
        };
        path.map(|p| load_hessian(p, order)).transpose()
    }
}

// the converter's `_reorder_v_heads` along one dimension: grouped by key head to tiled
fn v_head_order(num_k_heads: usize, num_v_heads: usize, n: usize) -> Vec<usize> {
    let per_k = num_v_heads / num_k_heads;
    let head_dim = n / num_v_heads;
    (0..n)
        .map(|j| {
            let (v, d) = (j / head_dim, j % head_dim);
            let (r, k) = (v / num_k_heads, v % num_k_heads);
            (k * per_k + r) * head_dim + d
        })
        .collect()
}

/// The engine Hessians in `dir` under the names of the tensors `gguf`'s layers load: a file `<linear>` binds to the
/// tensor its weight does, `<stack>.<e>.<proj>` to expert `e` of the stack, `lm_head` to the output (or, tied, the
/// token embedding). A GDN output projection's Hessian follows the converter's V-head order.
pub fn hessians_for_gguf(dir: &Path, gguf: &Path) -> Result<HessianDir> {
    let archive = inference_quant::GgufArchive::open([gguf])?;
    let names = crate::pipeline::gguf_tensor_names(&archive)?;
    let tensors = archive.tensors();
    let mut out = HessianDir {
        dense: HashMap::new(),
        experts: HashMap::new(),
        ssm_out_order: None,
        unmatched: Vec::new(),
    };
    let arch = archive
        .metadata()
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .unwrap_or_default();
    let meta = |key: &str| {
        archive
            .metadata()
            .get(&format!("{arch}.{key}"))
            .and_then(|v| v.to_u64().ok())
            .and_then(|v| usize::try_from(v).ok())
    };
    if let (Some(k), Some(v)) = (meta(SSM_GROUP_COUNT), meta(SSM_TIME_STEP_RANK))
        && k != v
        && k > 0
        && let Some(info) = tensors
            .iter()
            .find(|(n, _)| n.ends_with(SSM_OUT_SUFFIX))
            .map(|(_, i)| i)
    {
        let cols = *info.shape().last().unwrap_or(&0);
        out.ssm_out_order = Some(v_head_order(k, v, cols));
    }
    let mut entries = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != HESSIAN_EXTENSION) {
            continue;
        }
        let Some(key) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        if key == HEAD_KEY {
            let head = [OUTPUT_TENSOR, TOKEN_EMBD_TENSOR]
                .into_iter()
                .find(|n| tensors.contains_key(*n));
            match head {
                Some(n) => {
                    out.dense.insert(n.to_string(), path);
                }
                None => out.unmatched.push(key),
            }
            continue;
        }
        if let Some(gguf_name) = names.get(&format!("{key}.weight")) {
            out.dense.insert(gguf_name.clone(), path);
            continue;
        }
        // `<stack>.<e>.<proj>`: expert e of the stacked `<stack>.<proj>`
        let expert = key.rsplit_once('.').and_then(|(rest, proj)| {
            let (stack, e) = rest.rsplit_once('.')?;
            Some((format!("{stack}.{proj}.weight"), e.parse::<usize>().ok()?))
        });
        match expert.and_then(|(native, e)| Some((names.get(&native)?.clone(), e))) {
            Some((gguf_name, e)) => {
                out.experts.entry(gguf_name).or_default().insert(e, path);
            }
            None => out.unmatched.push(key),
        }
    }
    // a stacked up projection reads its gate's input: an expert Hessian written once, under the gate
    for (native, gguf_name) in &names {
        let Some(gate) = native
            .strip_suffix(UP_PROJ_SUFFIX)
            .map(|stack| format!("{stack}{GATE_PROJ_SUFFIX}"))
        else {
            continue;
        };
        if out.experts.contains_key(gguf_name) {
            continue;
        }
        if let Some(entries) = names.get(&gate).and_then(|g| out.experts.get(g)).cloned() {
            out.experts.insert(gguf_name.clone(), entries);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn packed_upper_triangle_unpacks_symmetric() {
        // [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
        let (n, full) = super::unpack_upper(&[1., 2., 3., 4., 5., 6.]);
        assert_eq!(n, 3);
        assert_eq!(full, [1., 2., 3., 2., 4., 5., 3., 5., 6.]);
        // past one tile: symmetric, the upper triangle in order
        let n = 2 * super::MIRROR_TILE + 3;
        let upper = (0u16..)
            .take(n * (n + 1) / 2)
            .map(f32::from)
            .collect::<Vec<_>>();
        let (m, full) = super::unpack_upper(&upper);
        assert_eq!(m, n);
        let mut k = 0;
        for i in 0..n {
            for j in i..n {
                assert_eq!(
                    (full[i * n + j], full[j * n + i]),
                    (upper[k], upper[k]),
                    "{i},{j}"
                );
                k += 1;
            }
        }
    }
}
