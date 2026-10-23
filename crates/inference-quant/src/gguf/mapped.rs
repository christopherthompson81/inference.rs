//! A GGUF tensor's blocks read in place from the file mapping, so a load need not copy what it only reads.

use std::ops::Range;
use std::sync::Arc;

use inference_tensor::Result;

use super::archive::GgufArchive;
use super::kernel::GgufType;

/// A weight's GGUF blocks in its archive's mapping, which this keeps alive.
#[derive(Clone)]
pub struct MappedBlocks {
    pub ty: GgufType,
    /// Logical dims, outermost first.
    pub dims: Vec<usize>,
    archive: Arc<GgufArchive>,
    shard: usize,
    range: Range<usize>,
}

impl MappedBlocks {
    pub(crate) fn new(
        ty: GgufType,
        dims: Vec<usize>,
        archive: Arc<GgufArchive>,
        shard: usize,
        range: Range<usize>,
    ) -> Self {
        Self {
            ty,
            dims,
            archive,
            shard,
            range,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.archive.shard_bytes(self.shard)[self.range.clone()]
    }

    /// Mapped bytes after the blocks, which a kernel reading whole tiles past the end may touch.
    pub fn trailing(&self) -> usize {
        self.archive.shard_bytes(self.shard).len() - self.range.end
    }

    /// The whole shard mapping these blocks sit in, and their offset in it.
    pub fn shard_mapping(&self) -> (&[u8], usize) {
        (self.archive.shard_bytes(self.shard), self.range.start)
    }

    /// Identifies the mapping, so registrations of it can be shared.
    pub fn shard_key(&self) -> (usize, usize) {
        (Arc::as_ptr(&self.archive) as usize, self.shard)
    }

    /// Rows `ids` of a `[rows, cols]` weight, dequantized to f32 and laid out one after another.
    pub fn gather_rows(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let [rows, cols] = self.dims[..] else {
            inference_tensor::bail!("gathering rows needs a 2-D weight, got {:?}", self.dims);
        };
        let row_bytes = self.ty.row_bytes(cols).ok_or_else(|| {
            inference_tensor::Error::msg(format!("{:?} rows cannot hold {cols} values", self.ty))
        })?;
        let bytes = self.bytes();
        let mut out = Vec::with_capacity(ids.len() * cols);
        for &id in ids {
            let id = id as usize;
            if id >= rows {
                inference_tensor::bail!("row {id} is past the weight's {rows} rows");
            }
            out.extend(super::raw::dequantize_rows(
                self.ty,
                cols,
                &bytes[id * row_bytes..][..row_bytes],
            )?);
        }
        Ok(out)
    }
}
