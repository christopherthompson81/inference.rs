//! What a GGUF writer reports: the file's size and how many tensors took each type.

use inference_tensor::quantized::GgmlDType;

#[derive(Debug, Clone)]
pub struct GgufWriteSummary {
    pub bytes: u64,
    pub types: Vec<(String, usize)>,
}

impl GgufWriteSummary {
    /// The summary of the file at `out`, whose tensors took `dtypes`.
    pub fn of(
        out: &std::path::Path,
        dtypes: impl IntoIterator<Item = GgmlDType>,
    ) -> anyhow::Result<Self> {
        let mut types: Vec<(String, usize)> = Vec::new();
        for dtype in dtypes {
            let name = dtype
                .name()
                .map_or_else(|| format!("{dtype:?}"), str::to_string);
            match types.iter_mut().find(|(n, _)| *n == name) {
                Some((_, count)) => *count += 1,
                None => types.push((name, 1)),
            }
        }
        Ok(Self {
            bytes: std::fs::metadata(out)?.len(),
            types,
        })
    }
}
