use std::{
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::Arc,
};

use inference_tensor::{Device, Result};

use super::{UqffReader, UqffTensor, uqff_version_tensors};
use crate::{IsqType, QuantMethod, Shard};

// the full key, kept in the file so a hash collision reads as a miss
const KEY_TENSOR: &str = "isq_cache.key";
const CACHE_EXTENSION: &str = "uqff";
const PARTIAL_EXTENSION: &str = "uqff.partial";

/// Layers quantized at load, kept in one UQFF file per `key` (the source weights and how they were quantized), so a
/// later load reads them instead of quantizing again. A cache is all or nothing: it is written only once complete,
/// and then replaces the other files of its `group` (the same model, quantized from older weights or another way).
pub struct IsqCache {
    path: PathBuf,
    group_prefix: String,
    reader: Option<UqffReader>,
    pending: Vec<UqffTensor>,
}

impl IsqCache {
    pub fn open(dir: &Path, group: &str, key: &str) -> Self {
        let hash = |text: &str| {
            let mut hasher = siphasher::sip::SipHasher13::new();
            text.hash(&mut hasher);
            hasher.finish()
        };
        let group_prefix = format!("{:016x}-", hash(group));
        let path = dir.join(format!(
            "{group_prefix}{:016x}.{CACHE_EXTENSION}",
            hash(key)
        ));
        let reader = path
            .is_file()
            .then(|| UqffReader::open(std::slice::from_ref(&path)).ok())
            .flatten()
            .filter(|reader| {
                reader
                    .load_tensor(KEY_TENSOR, &Device::Cpu)
                    .and_then(|stored| stored.to_vec1::<u8>())
                    .is_ok_and(|stored| stored == key.as_bytes())
            });
        let pending = if reader.is_some() {
            Vec::new()
        } else {
            let mut pending = uqff_version_tensors();
            pending.push(UqffTensor::from_raw_u8(
                KEY_TENSOR,
                key.as_bytes().to_vec(),
                vec![key.len()],
            ));
            pending
        };
        Self {
            path,
            group_prefix,
            reader,
            pending,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether an earlier load wrote this cache, so every layer reads from it.
    pub fn is_complete(&self) -> bool {
        self.reader.is_some()
    }

    /// Layer `name` from the cache on `device`, or built by `quantize` and kept for [`IsqCache::finish`].
    pub fn linear(
        &mut self,
        name: &str,
        ty: IsqType,
        device: &Device,
        quantize: impl FnOnce() -> Result<Arc<dyn QuantMethod>>,
    ) -> Result<Arc<dyn QuantMethod>> {
        if let Some(reader) = &self.reader {
            return reader
                .load_linear(name, device, Shard::default())?
                .ok_or_else(|| {
                    inference_tensor::Error::msg(format!(
                        "{} has no layer `{name}`; delete it to quantize again",
                        self.path.display()
                    ))
                });
        }
        let layer = quantize()?;
        self.pending.extend(layer.serialize_uqff(name, ty)?);
        Ok(layer)
    }

    /// Writes the layers this load quantized; a cache read from disk has nothing to write.
    pub fn finish(self) -> Result<()> {
        if self.reader.is_some() {
            return Ok(());
        }
        let io = |error: std::io::Error| inference_tensor::Error::msg(error.to_string());
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        let partial = self.path.with_extension(PARTIAL_EXTENSION);
        safetensors::serialize_to_file(
            self.pending.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &partial,
        )
        .map_err(|error| inference_tensor::Error::msg(error.to_string()))?;
        std::fs::rename(&partial, &self.path).map_err(io)?;
        let stale = self
            .path
            .parent()
            .and_then(|dir| std::fs::read_dir(dir).ok())
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                *path != self.path
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            name.starts_with(&self.group_prefix) && name.ends_with(CACHE_EXTENSION)
                        })
            });
        for path in stale {
            std::fs::remove_file(&path).map_err(io)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use inference_tensor::{
        Tensor,
        quantized::{GgmlDType, QTensor},
    };

    use super::*;
    use crate::GgufMatMul;

    const ROWS: usize = 4;
    const COLS: usize = 64;
    const GROUP: &str = "/models/drafter";

    fn layer() -> Result<Arc<dyn QuantMethod>> {
        let values = (0..ROWS * COLS)
            .map(|i| (i % 11) as f32 - 5.0)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(values, (ROWS, COLS), &Device::Cpu)?;
        Ok(Arc::new(GgufMatMul::from_qtensor(
            QTensor::quantize(&weight, GgmlDType::Q8_0)?,
            None,
        )?))
    }

    fn files(dir: &Path) -> usize {
        std::fs::read_dir(dir).map_or(0, |entries| entries.count())
    }

    #[test]
    fn a_written_cache_reads_back_and_replaces_its_group() -> Result<()> {
        let dir = tempfile::tempdir().map_err(inference_tensor::Error::wrap)?;
        let mut first = IsqCache::open(dir.path(), GROUP, "weights v1");
        assert!(!first.is_complete());
        let quantized = first.linear("proj", IsqType::Q8_0, &Device::Cpu, layer)?;
        first.finish()?;

        let mut again = IsqCache::open(dir.path(), GROUP, "weights v1");
        assert!(again.is_complete());
        let read = again.linear("proj", IsqType::Q8_0, &Device::Cpu, || {
            panic!("a complete cache quantizes nothing")
        })?;
        let input = Tensor::ones((1, COLS), inference_tensor::DType::F32, &Device::Cpu)?;
        let diff = (quantized.forward(&input)? - read.forward(&input)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(diff, 0.0);

        let mut updated = IsqCache::open(dir.path(), GROUP, "weights v2");
        assert!(!updated.is_complete(), "another key misses");
        updated.linear("proj", IsqType::Q8_0, &Device::Cpu, layer)?;
        let other_group = IsqCache::open(dir.path(), "/models/other", "weights v1");
        assert!(!other_group.is_complete());
        updated.finish()?;
        assert_eq!(files(dir.path()), 1, "the group's older file is removed");
        assert!(!IsqCache::open(dir.path(), GROUP, "weights v1").is_complete());
        Ok(())
    }
}
