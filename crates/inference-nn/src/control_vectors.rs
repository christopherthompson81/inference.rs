//! Control vectors: a direction per layer added to the residual stream after the layer, scaled, as llama.cpp's
//! `--control-vector`. A file holds `direction.<layer>` (`[hidden]` F32, layer 1 up; llama.cpp never steers layer 0),
//! in a GGUF (llama.cpp's `cvector-generator`, repeng) or a safetensors file.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use inference_tensor::{DType, Device, Result, Tensor};

const DIRECTION_PREFIX: &str = "direction.";
// a direction's layer from 1: llama.cpp keeps no data for layer 0, which no file names
const FIRST_LAYER: usize = 1;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One file of directions and the scale they are added at.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlVectorFile {
    pub path: PathBuf,
    pub scale: f32,
}

/// The vectors a model is steered by: files summed at their scales, applied to the layers of `layers` (inclusive,
/// 0-based; unset, every layer a file names).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ControlVectorSet {
    pub files: Vec<ControlVectorFile>,
    pub layers: Option<(usize, usize)>,
}

impl ControlVectorSet {
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The summed directions, a row of `hidden` per layer of `layers` (row 0 and rows out of range zero).
    pub fn combined(&self, layers: usize, hidden: usize) -> Result<Vec<f32>> {
        let mut rows = vec![0f32; layers * hidden];
        for file in &self.files {
            for (layer, direction) in read_directions(&file.path)? {
                if direction.len() != hidden {
                    inference_tensor::bail!(
                        "{}: `direction.{layer}` holds {} values, the model's hidden size is {hidden}",
                        file.path.display(),
                        direction.len()
                    );
                }
                if layer >= layers {
                    inference_tensor::bail!(
                        "{}: `direction.{layer}` names a layer past the model's {layers}",
                        file.path.display()
                    );
                }
                if self
                    .layers
                    .is_some_and(|(first, last)| layer < first || layer > last)
                {
                    continue;
                }
                let row = &mut rows[layer * hidden..(layer + 1) * hidden];
                for (out, v) in row.iter_mut().zip(direction) {
                    *out += v * file.scale;
                }
            }
        }
        Ok(rows)
    }

    /// [`Self::key`] hashed (64-bit FNV-1a, stable across builds): `None` for no vectors, which leaves keys as they were.
    pub fn hash(&self) -> Option<u64> {
        (!self.is_empty()).then(|| {
            self.key().bytes().fold(FNV_OFFSET, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
            })
        })
    }

    /// A key that changes with the files, their scales and the range: what cached prompt states depend on.
    pub fn key(&self) -> String {
        let files = self
            .files
            .iter()
            .map(|f| format!("{}@{}", f.path.display(), f.scale))
            .collect::<Vec<_>>()
            .join(";");
        match self.layers {
            Some((first, last)) => format!("{files}|{first}-{last}"),
            None => files,
        }
    }
}

fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix(DIRECTION_PREFIX)?.parse().ok()
}

/// A file's directions by layer.
pub fn read_directions(path: &Path) -> Result<Vec<(usize, Vec<f32>)>> {
    let directions = if path.extension().is_some_and(|ext| ext == "safetensors") {
        // SAFETY: the file is only read
        let st = unsafe { inference_tensor::safetensors::MmapedSafetensors::new(path)? };
        st.tensors()
            .into_iter()
            .filter_map(|(name, _)| layer_of(&name).map(|layer| (layer, name)))
            .map(|(layer, name)| {
                let tensor = st.load(&name, &Device::Cpu)?.to_dtype(DType::F32)?;
                Ok((layer, tensor.flatten_all()?.to_vec1::<f32>()?))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let content = inference_tensor::quantized::gguf_file::Content::read(&mut reader)?;
        content
            .tensor_infos
            .keys()
            .filter_map(|name| layer_of(name).map(|layer| (layer, name.clone())))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(layer, name)| {
                let tensor = content
                    .tensor(&mut reader, &name, &Device::Cpu)?
                    .dequantize(&Device::Cpu)?;
                Ok((layer, tensor.flatten_all()?.to_vec1::<f32>()?))
            })
            .collect::<Result<Vec<_>>>()?
    };
    if directions.is_empty() {
        inference_tensor::bail!("{}: no `direction.<layer>` tensors", path.display());
    }
    if let Some((layer, _)) = directions.iter().find(|(layer, _)| *layer < FIRST_LAYER) {
        inference_tensor::bail!(
            "{}: `direction.{layer}`: directions start at layer 1",
            path.display()
        );
    }
    Ok(directions)
}

/// The rows a model adds, each on its layer's device at the activations' dtype; a set's values are written into them
/// in place, so CUDA graphs that read them keep their nodes.
pub struct ControlVectors {
    rows: Vec<Option<Tensor>>,
    set: ControlVectorSet,
}

impl ControlVectors {
    /// `set` laid out for `devices` (a layer's device each), `hidden` wide at `dtype`.
    pub fn new(
        set: ControlVectorSet,
        devices: &[Device],
        hidden: usize,
        dtype: DType,
    ) -> Result<Self> {
        let combined = set.combined(devices.len(), hidden)?;
        let rows = devices
            .iter()
            .enumerate()
            .map(|(layer, device)| {
                let row = &combined[layer * hidden..(layer + 1) * hidden];
                // a layer no file steers adds nothing, and gets no node to add it with
                if layer < FIRST_LAYER {
                    return Ok(None);
                }
                Ok(Some(
                    Tensor::from_slice(row, hidden, device)?.to_dtype(dtype)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { rows, set })
    }

    /// `set`'s values written into the rows in place.
    pub fn rewrite(&mut self, set: ControlVectorSet) -> Result<()> {
        let layers = self.rows.len();
        let hidden = self
            .rows
            .iter()
            .flatten()
            .next()
            .map_or(Ok(0), |row| row.dim(0))?;
        let combined = set.combined(layers, hidden)?;
        for (layer, row) in self.rows.iter().enumerate() {
            if let Some(row) = row {
                let values = Tensor::from_slice(
                    &combined[layer * hidden..(layer + 1) * hidden],
                    hidden,
                    row.device(),
                )?
                .to_dtype(row.dtype())?;
                row.slice_set(&values, 0, 0)?;
            }
        }
        self.set = set;
        Ok(())
    }

    pub fn set(&self) -> &ControlVectorSet {
        &self.set
    }

    /// `xs` with layer `layer`'s row added (each position the same).
    pub fn apply(&self, layer: usize, xs: Tensor) -> Result<Tensor> {
        match self.rows.get(layer).and_then(Option::as_ref) {
            Some(row) => xs.broadcast_add(row),
            None => Ok(xs),
        }
    }
}

/// Where a model holds its control vectors: empty until a set is attached, rewritten in place after.
#[derive(Clone, Default)]
pub struct ControlVectorSlot(Arc<RwLock<Option<ControlVectors>>>);

impl ControlVectorSlot {
    /// `xs` after layer `layer`, with its row added when a set is attached.
    pub fn apply(&self, layer: usize, xs: Tensor) -> Result<Tensor> {
        match &*self.0.read().expect("control vectors poisoned") {
            Some(vectors) => vectors.apply(layer, xs),
            None => Ok(xs),
        }
    }

    /// Attaches `set` (rewriting the rows in place when one is attached already), or detaches with an empty set.
    /// Returns whether the model's graphs need capturing again: a set attached where none was, or removed.
    pub fn replace(
        &self,
        set: ControlVectorSet,
        devices: &[Device],
        hidden: usize,
        dtype: DType,
    ) -> Result<bool> {
        let mut slot = self.0.write().expect("control vectors poisoned");
        match (slot.as_mut(), set.is_empty()) {
            (Some(_), true) => {
                *slot = None;
                Ok(true)
            }
            (None, true) => Ok(false),
            (Some(vectors), false) => {
                vectors.rewrite(set)?;
                Ok(false)
            }
            (None, false) => {
                *slot = Some(ControlVectors::new(set, devices, hidden, dtype)?);
                Ok(true)
            }
        }
    }

    /// The attached set, empty when none is.
    pub fn current(&self) -> ControlVectorSet {
        self.0
            .read()
            .expect("control vectors poisoned")
            .as_ref()
            .map(|v| v.set().clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    const HIDDEN: usize = 4;
    const LAYERS: usize = 3;

    fn file(dir: &Path, name: &str, directions: &[(usize, [f32; HIDDEN])]) -> PathBuf {
        let tensors = directions
            .iter()
            .map(|(layer, d)| {
                (
                    format!("{DIRECTION_PREFIX}{layer}"),
                    Tensor::new(d, &Device::Cpu).unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        let path = dir.join(name);
        inference_tensor::safetensors::save(&tensors, &path).unwrap();
        path
    }

    // each layer's row is the files' directions for it times their scales, layer 0 and layers out of range left out
    #[test]
    fn files_sum_their_scaled_directions_per_layer() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let a = file(
            dir.path(),
            "a.safetensors",
            &[(1, [1.0, 2.0, 3.0, 4.0]), (2, [1.0; HIDDEN])],
        );
        let b = file(dir.path(), "b.safetensors", &[(2, [0.5, 0.0, -0.5, 1.0])]);
        let set = ControlVectorSet {
            files: vec![
                ControlVectorFile {
                    path: a.clone(),
                    scale: 2.0,
                },
                ControlVectorFile {
                    path: b,
                    scale: -1.0,
                },
            ],
            layers: None,
        };
        let rows = set.combined(LAYERS, HIDDEN)?;
        assert_eq!(&rows[..HIDDEN], &[0.0; HIDDEN]);
        assert_eq!(&rows[HIDDEN..2 * HIDDEN], &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(&rows[2 * HIDDEN..], &[1.5, 2.0, 2.5, 1.0]);
        let ranged = ControlVectorSet {
            layers: Some((2, 2)),
            ..set.clone()
        };
        assert_eq!(
            &ranged.combined(LAYERS, HIDDEN)?[HIDDEN..2 * HIDDEN],
            &[0.0; HIDDEN]
        );
        assert_ne!(set.hash(), ranged.hash());
        assert_eq!(ControlVectorSet::default().hash(), None);
        // a direction wider than the model, or past its layers, is refused
        let wide = ControlVectorSet {
            files: vec![ControlVectorFile {
                path: a,
                scale: 1.0,
            }],
            layers: None,
        };
        assert!(wide.combined(LAYERS, HIDDEN + 1).is_err());
        assert!(wide.combined(2, HIDDEN).is_err());
        Ok(())
    }

    // a layer's row adds to every position, and rewriting the set changes the rows a captured graph reads in place
    #[test]
    fn a_slot_adds_its_row_and_rewrites_it_in_place() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let a = file(dir.path(), "a.safetensors", &[(1, [1.0, -1.0, 0.5, 0.0])]);
        let devices = vec![Device::Cpu; LAYERS];
        let slot = ControlVectorSlot::default();
        let set = |scale| ControlVectorSet {
            files: vec![ControlVectorFile {
                path: a.clone(),
                scale,
            }],
            layers: None,
        };
        assert!(slot.replace(set(1.0), &devices, HIDDEN, DType::F32)?);
        let xs = Tensor::ones((2, HIDDEN), DType::F32, &Device::Cpu)?;
        assert_eq!(
            slot.apply(0, xs.clone())?.to_vec2::<f32>()?,
            xs.to_vec2::<f32>()?
        );
        assert_eq!(
            slot.apply(1, xs.clone())?.to_vec2::<f32>()?,
            vec![vec![2.0, 0.0, 1.5, 1.0]; 2]
        );
        // a new scale keeps the rows: no recapture
        assert!(!slot.replace(set(2.0), &devices, HIDDEN, DType::F32)?);
        assert_eq!(
            slot.apply(1, xs.clone())?.to_vec2::<f32>()?,
            vec![vec![3.0, -1.0, 2.0, 1.0]; 2]
        );
        assert_eq!(slot.current(), set(2.0));
        assert!(slot.replace(ControlVectorSet::default(), &devices, HIDDEN, DType::F32)?);
        assert_eq!(
            slot.apply(1, xs.clone())?.to_vec2::<f32>()?,
            xs.to_vec2::<f32>()?
        );
        Ok(())
    }

    // llama.cpp numbers directions from layer 1: a file with a layer 0 direction is not one of its control vectors
    #[test]
    fn a_direction_for_layer_zero_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = file(dir.path(), "zero.safetensors", &[(0, [1.0; HIDDEN])]);
        assert!(read_directions(&path).is_err());
    }
}
