//! Control vectors: a direction per layer added to the residual stream after the layer, scaled, as llama.cpp's
//! `--control-vector`. A file holds `direction.<layer>` (`[hidden]` F32, layer 1 up; llama.cpp never steers layer 0),
//! in a GGUF (llama.cpp's `cvector-generator`, repeng) or a safetensors file. A looped model also takes loop guidance
//! (LoopCD-Hidden, arXiv:2610.02185): the coda's input pushed away from the first pass's output, `h + w (h - h1)`.

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
/// 0-based; unset, every layer a file names), and a looped model's loop guidance strength.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ControlVectorSet {
    pub files: Vec<ControlVectorFile>,
    pub layers: Option<(usize, usize)>,
    pub loop_guidance: Option<f32>,
}

impl ControlVectorSet {
    pub fn is_empty(&self) -> bool {
        self.files.is_empty() && self.loop_guidance.is_none()
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
        let key = match self.layers {
            Some((first, last)) => format!("{files}|{first}-{last}"),
            None => files,
        };
        match self.loop_guidance {
            Some(w) => format!("{key}|loop {w}"),
            None => key,
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
    // each layer's row; none at all without files
    rows: Option<Vec<Option<Tensor>>>,
    // the loop guidance strength, a scalar on the coda's device
    guidance: Option<Tensor>,
    set: ControlVectorSet,
}

impl ControlVectors {
    /// `set` laid out for `devices` (a layer's device each), `hidden` wide at `dtype`; its loop guidance on
    /// `guidance_device`.
    pub fn new(
        set: ControlVectorSet,
        devices: &[Device],
        hidden: usize,
        dtype: DType,
        guidance_device: Option<&Device>,
    ) -> Result<Self> {
        let rows = if set.files.is_empty() {
            None
        } else {
            let combined = set.combined(devices.len(), hidden)?;
            Some(
                devices
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
                    .collect::<Result<Vec<_>>>()?,
            )
        };
        let guidance = match (set.loop_guidance, guidance_device) {
            (Some(w), Some(device)) => Some(Tensor::from_slice(&[w], 1, device)?.to_dtype(dtype)?),
            _ => None,
        };
        Ok(Self {
            rows,
            guidance,
            set,
        })
    }

    // whether it has rows and guidance: what its graphs' nodes depend on
    fn shape(set: &ControlVectorSet) -> (bool, bool) {
        (!set.files.is_empty(), set.loop_guidance.is_some())
    }

    /// `set`'s values written into the rows and the guidance in place; `set` has the same shape as the current one.
    pub fn rewrite(&mut self, set: ControlVectorSet) -> Result<()> {
        if let Some(rows) = &self.rows {
            let layers = rows.len();
            let hidden = rows
                .iter()
                .flatten()
                .next()
                .map_or(Ok(0), |row| row.dim(0))?;
            let combined = set.combined(layers, hidden)?;
            for (layer, row) in rows.iter().enumerate() {
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
        }
        if let (Some(g), Some(w)) = (&self.guidance, set.loop_guidance) {
            g.slice_set(
                &Tensor::from_slice(&[w], 1, g.device())?.to_dtype(g.dtype())?,
                0,
                0,
            )?;
        }
        self.set = set;
        Ok(())
    }

    pub fn set(&self) -> &ControlVectorSet {
        &self.set
    }

    /// `xs` with layer `layer`'s row added (each position the same).
    pub fn apply(&self, layer: usize, xs: Tensor) -> Result<Tensor> {
        match self
            .rows
            .as_ref()
            .and_then(|rows| rows.get(layer))
            .and_then(Option::as_ref)
        {
            Some(row) => xs.broadcast_add(row),
            None => Ok(xs),
        }
    }
}

/// Where a model holds its control vectors: empty until a set is attached, rewritten in place after. It also records
/// each layer's output while a capture is armed, for deriving vectors from the model.
#[derive(Clone, Default)]
pub struct ControlVectorSlot {
    vectors: Arc<RwLock<Option<ControlVectors>>>,
    // each layer's outputs (one entry a visit) while armed
    capture: Arc<std::sync::Mutex<Option<Vec<Vec<Tensor>>>>>,
}

impl ControlVectorSlot {
    /// `xs` after layer `layer`, with its row added when a set is attached. While a capture is armed it records `xs`
    /// and adds nothing, so a derivation reads the unsteered model.
    pub fn apply(&self, layer: usize, xs: Tensor) -> Result<Tensor> {
        if let Some(capture) = self
            .capture
            .lock()
            .expect("control vector capture poisoned")
            .as_mut()
        {
            let hidden = xs.dim(inference_tensor::D::Minus1)?;
            if let Some(visits) = capture.get_mut(layer) {
                visits.push(xs.reshape(((), hidden))?);
            }
            return Ok(xs);
        }
        match &*self.vectors.read().expect("control vectors poisoned") {
            Some(vectors) => vectors.apply(layer, xs),
            None => Ok(xs),
        }
    }

    /// The loop guidance strength (a scalar tensor on the coda's device) when set and no capture is armed.
    pub fn loop_guidance(&self) -> Option<Tensor> {
        if self
            .capture
            .lock()
            .expect("control vector capture poisoned")
            .is_some()
        {
            return None;
        }
        self.vectors
            .read()
            .expect("control vectors poisoned")
            .as_ref()
            .and_then(|v| v.guidance.clone())
    }

    /// Records the next forwards' layer outputs, for `layers` layers.
    pub fn begin_capture(&self, layers: usize) {
        *self
            .capture
            .lock()
            .expect("control vector capture poisoned") = Some(vec![Vec::new(); layers]);
    }

    /// The recorded outputs, a layer's visits pooled as rows (`[rows, hidden]` F32 on the host), and the capture
    /// disarmed; `None` when none was armed.
    pub fn take_capture(&self) -> Result<Option<Vec<Tensor>>> {
        let Some(layers) = self
            .capture
            .lock()
            .expect("control vector capture poisoned")
            .take()
        else {
            return Ok(None);
        };
        layers
            .into_iter()
            .map(|visits| {
                let rows = visits
                    .iter()
                    .map(|v| v.to_dtype(DType::F32)?.to_device(&Device::Cpu))
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&rows, 0)
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// Attaches `set` (rewriting it in place when one of the same shape is attached already), or detaches with an
    /// empty set; `guidance_device` is a looped model's coda device, none for a model that takes no loop guidance.
    /// Returns whether the model's graphs need capturing again: a set attached where none was, removed, or reshaped
    /// (files or guidance added or dropped).
    pub fn replace(
        &self,
        set: ControlVectorSet,
        devices: &[Device],
        hidden: usize,
        dtype: DType,
        guidance_device: Option<&Device>,
    ) -> Result<bool> {
        if let Some(w) = set.loop_guidance {
            if guidance_device.is_none() {
                inference_tensor::bail!(
                    "loop guidance takes a looped model (Inkling 2); this one runs each layer once"
                );
            }
            if !w.is_finite() || w < 0.0 {
                inference_tensor::bail!("loop guidance must be 0 or more, got {w}");
            }
        }
        let mut slot = self.vectors.write().expect("control vectors poisoned");
        if set.is_empty() {
            return Ok(slot.take().is_some());
        }
        if let Some(vectors) = slot.as_mut()
            && ControlVectors::shape(vectors.set()) == ControlVectors::shape(&set)
        {
            vectors.rewrite(set)?;
            return Ok(false);
        }
        *slot = Some(ControlVectors::new(
            set,
            devices,
            hidden,
            dtype,
            guidance_device,
        )?);
        Ok(true)
    }

    /// The attached set, empty when none is.
    pub fn current(&self) -> ControlVectorSet {
        self.vectors
            .read()
            .expect("control vectors poisoned")
            .as_ref()
            .map(|v| v.set().clone())
            .unwrap_or_default()
    }
}

/// One prompt's recorded layer outputs (`[rows, hidden]` each, the same rows for a pair's two prompts).
pub type LayerOutputs = Vec<Tensor>;

/// llama.cpp's `mean` method: for each layer but the last, every pair's per-row difference (positive minus negative)
/// with the all-zero rows dropped (a prefix the pair shares), averaged over the pairs and normalized; the direction
/// from layer L's output is `direction.<L + 1>`, as llama.cpp names it. Returns (layer, direction) pairs.
pub fn mean_directions(pairs: &[(LayerOutputs, LayerOutputs)]) -> Result<Vec<(usize, Vec<f32>)>> {
    let Some((first, _)) = pairs.first() else {
        inference_tensor::bail!(
            "deriving a control vector needs at least one positive and negative pair"
        );
    };
    let layers = first.len();
    let mut directions = Vec::with_capacity(layers.saturating_sub(1));
    for layer in 0..layers.saturating_sub(1) {
        let mut sum: Vec<f64> = Vec::new();
        let mut rows = 0usize;
        for (positive, negative) in pairs {
            let (p, n) = (&positive[layer], &negative[layer]);
            if p.dims() != n.dims() {
                inference_tensor::bail!(
                    "layer {layer}: a pair's outputs are {:?} and {:?}; pad its prompts to one length",
                    p.dims(),
                    n.dims()
                );
            }
            let diff = (p - n)?.to_vec2::<f32>()?;
            for row in diff.iter().filter(|row| row.iter().any(|v| *v != 0.0)) {
                if sum.is_empty() {
                    sum = vec![0.0; row.len()];
                }
                for (s, v) in sum.iter_mut().zip(row) {
                    *s += f64::from(*v);
                }
                rows += 1;
            }
        }
        if rows == 0 {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = sum.iter().map(|s| s / rows as f64).collect::<Vec<_>>();
        let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        #[allow(clippy::cast_possible_truncation)]
        directions.push((
            layer + FIRST_LAYER,
            mean.iter().map(|v| (v / norm) as f32).collect(),
        ));
    }
    Ok(directions)
}

const GGUF_F32: u32 = 0;
const CONTROL_VECTOR_ARCHITECTURE: &str = "controlvector";

/// `directions` written as llama.cpp's control vector GGUF (`direction.<layer>` F32, `controlvector.*` metadata).
pub fn write_control_vector(
    path: &Path,
    directions: &[(usize, Vec<f32>)],
    model_hint: &str,
) -> Result<()> {
    use inference_tensor::quantized::gguf_file::{
        TensorEntry, Value, metadata_bytes, write_header, write_tensor_padding,
    };
    use std::io::Write;
    let names = directions
        .iter()
        .map(|(layer, _)| format!("{DIRECTION_PREFIX}{layer}"))
        .collect::<Vec<_>>();
    let dims = directions
        .iter()
        .map(|(_, d)| vec![d.len()])
        .collect::<Vec<_>>();
    let entries = names
        .iter()
        .zip(&dims)
        .map(|(name, dims)| TensorEntry {
            name,
            dims,
            ggml_dtype: GGUF_F32,
            size_in_bytes: dims[0] * std::mem::size_of::<f32>(),
        })
        .collect::<Vec<_>>();
    let architecture = Value::String(CONTROL_VECTOR_ARCHITECTURE.to_string());
    let hint = Value::String(model_hint.to_string());
    #[allow(clippy::cast_possible_truncation)]
    let layer_count = Value::U32(directions.len() as u32);
    let metadata = [
        ("general.architecture", &architecture),
        ("controlvector.model_hint", &hint),
        ("controlvector.layer_count", &layer_count),
    ];
    let partial = path.with_extension("gguf.partial");
    let mut writer = std::io::BufWriter::new(std::fs::File::create(&partial)?);
    write_header(
        &mut writer,
        metadata.len(),
        &metadata_bytes(&metadata)?,
        &entries,
    )?;
    for (_, direction) in directions {
        let bytes = direction
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        writer.write_all(&bytes)?;
        write_tensor_padding(&mut writer, bytes.len())?;
    }
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    std::fs::rename(&partial, path)?;
    Ok(())
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
            loop_guidance: None,
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
            loop_guidance: None,
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
            loop_guidance: None,
        };
        assert!(slot.replace(set(1.0), &devices, HIDDEN, DType::F32, None)?);
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
        assert!(!slot.replace(set(2.0), &devices, HIDDEN, DType::F32, None)?);
        assert_eq!(
            slot.apply(1, xs.clone())?.to_vec2::<f32>()?,
            vec![vec![3.0, -1.0, 2.0, 1.0]; 2]
        );
        assert_eq!(slot.current(), set(2.0));
        assert!(slot.replace(
            ControlVectorSet::default(),
            &devices,
            HIDDEN,
            DType::F32,
            None
        )?);
        assert_eq!(
            slot.apply(1, xs.clone())?.to_vec2::<f32>()?,
            xs.to_vec2::<f32>()?
        );
        Ok(())
    }

    // loop guidance alone is a set: its strength keys the cache, rewrites in place, and only a looped model takes it
    #[test]
    fn loop_guidance_attaches_rewrites_in_place_and_needs_a_looped_model() -> Result<()> {
        let devices = vec![Device::Cpu; LAYERS];
        let slot = ControlVectorSlot::default();
        let guided = |w| ControlVectorSet {
            loop_guidance: Some(w),
            ..Default::default()
        };
        assert!(!guided(0.2).is_empty());
        assert_ne!(guided(0.2).hash(), guided(0.3).hash());
        assert!(
            slot.replace(guided(0.2), &devices, HIDDEN, DType::F32, None)
                .is_err()
        );
        assert!(
            slot.replace(
                guided(-1.0),
                &devices,
                HIDDEN,
                DType::F32,
                Some(&Device::Cpu)
            )
            .is_err()
        );
        assert!(slot.replace(
            guided(0.2),
            &devices,
            HIDDEN,
            DType::F32,
            Some(&Device::Cpu)
        )?);
        let w = slot.loop_guidance().expect("guidance attached");
        // a new strength writes the same scalar a captured graph reads
        assert!(!slot.replace(
            guided(0.5),
            &devices,
            HIDDEN,
            DType::F32,
            Some(&Device::Cpu)
        )?);
        assert_eq!(w.to_vec1::<f32>()?, vec![0.5]);
        // no rows without files: layers pass through
        let xs = Tensor::ones((2, HIDDEN), DType::F32, &Device::Cpu)?;
        assert_eq!(
            slot.apply(1, xs.clone())?.to_vec2::<f32>()?,
            xs.to_vec2::<f32>()?
        );
        // an armed capture reads the unguided model
        slot.begin_capture(LAYERS);
        assert!(slot.loop_guidance().is_none());
        for layer in 0..LAYERS {
            slot.apply(layer, xs.clone())?;
        }
        slot.take_capture()?;
        assert!(slot.replace(
            ControlVectorSet::default(),
            &devices,
            HIDDEN,
            DType::F32,
            None
        )?);
        assert!(slot.loop_guidance().is_none());
        Ok(())
    }

    // the mean of each pair's row differences with shared (all-zero) rows dropped, normalized; layer L's output names
    // direction L + 1, and the last layer gives none
    #[test]
    fn mean_directions_follow_llama_cpps_mean_method() -> Result<()> {
        let t = |rows: [[f32; 2]; 2]| Tensor::new(&rows, &Device::Cpu).unwrap();
        // three layers; row 0 is a shared prefix, identical in both prompts
        let positive = vec![
            t([[1.0, 1.0], [3.0, 0.0]]),
            t([[0.0, 0.0], [0.0, 2.0]]),
            t([[9.0, 9.0], [9.0, 9.0]]),
        ];
        let negative = vec![
            t([[1.0, 1.0], [0.0, 0.0]]),
            t([[0.0, 0.0], [0.0, 0.0]]),
            t([[0.0, 0.0], [0.0, 0.0]]),
        ];
        let directions = mean_directions(&[(positive, negative)])?;
        assert_eq!(directions.len(), 2);
        assert_eq!(directions[0], (1, vec![1.0, 0.0]));
        assert_eq!(directions[1], (2, vec![0.0, 1.0]));
        Ok(())
    }

    #[test]
    fn a_written_control_vector_reads_back_as_its_directions() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("written.gguf");
        let directions = vec![(1, vec![0.6, 0.8, 0.0, 0.0]), (3, vec![0.0, 0.0, 1.0, 0.0])];
        write_control_vector(&path, &directions, "a tiny model")?;
        let mut read = read_directions(&path)?;
        read.sort_by_key(|(layer, _)| *layer);
        assert_eq!(read, directions);
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
