use std::collections::HashMap;

use super::GGUFArchitecture;
use inference_tensor::{
    Device, Result,
    quantized::{
        QTensor,
        gguf_file::{self, TensorInfo, Value},
    },
};

// Internal invariant: contents and readers must be paired.
/// This abstracts the files for a GGUF model and enables multiple files to be used.
pub struct Content<'a, R: std::io::Seek + std::io::Read> {
    contents: Vec<gguf_file::Content>,
    readers: &'a mut [&'a mut R],
    arch: GGUFArchitecture,
    all_metadata: HashMap<String, Value>,
}

impl<'a, R: std::io::Seek + std::io::Read> Content<'a, R> {
    pub fn arch(&self) -> GGUFArchitecture {
        self.arch
    }

    /// Retrieve a tensor info, searching through each content.
    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        for ct in &self.contents {
            if let Some(tensor_info) = ct.tensor_infos.get(name) {
                return Ok(tensor_info);
            }
        }
        inference_tensor::bail!("Cannot find tensor info for {name}")
    }

    /// Retrieve a tensor, searching through each content.
    pub fn tensor(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        for (ct, reader) in self.contents.iter().zip(self.readers.iter_mut()) {
            if let Some(tensor_info) = ct.tensor_infos.get(name) {
                return tensor_info.read(reader, ct.tensor_data_offset, device);
            }
        }
        inference_tensor::bail!("Cannot find tensor info for {name}")
    }

    /// Check for a tensor, searching through each content.
    pub fn has_tensor(&self, name: &str) -> bool {
        for ct in self.contents.iter() {
            if ct.tensor_infos.contains_key(name) {
                return true;
            }
        }
        false
    }

    /// Get all metadatas
    pub fn get_metadata(&self) -> &HashMap<String, Value> {
        &self.all_metadata
    }
}
