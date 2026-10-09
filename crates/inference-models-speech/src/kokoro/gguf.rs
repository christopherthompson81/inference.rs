//! audio.cpp's Kokoro GGUF (`general.architecture = kokoro_tts`): numbered tensors and an embedded file table.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek};
use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::quantized::gguf_file::{Content, Value, peek_architecture};
use inference_tensor::{DType, Device, Result, bail};

use super::{KokoroConfig, VoicePack};

pub const ARCHITECTURE: &str = "kokoro_tts";
pub const GGUF_EXTENSION: &str = "gguf";
const ARCHITECTURE_KEY: &str = "general.architecture";
const TENSOR_NAMES_KEY: &str = "kokoro.tensor_names";
const TENSOR_SHAPE_PREFIX: &str = "kokoro.tensor_shape.";
// tensor i of `kokoro.tensor_names` is stored as `kokoro.<i>`
const TENSOR_PREFIX: &str = "kokoro.";
const FILE_NAMES_KEY: &str = "audiocpp.embedded_files.names";
const FILE_OFFSETS_KEY: &str = "audiocpp.embedded_files.offsets";
const FILE_DATA_KEY: &str = "audiocpp.embedded_files.data";
const CONFIG_FILE: &str = "config.json";
const VOICE_DIR: &str = "voices/";
const VOICE_EXT: &str = ".bin";

/// Everything a Kokoro GGUF carries: its config, F32 weights by PyTorch name, and its voice packs.
pub struct KokoroGguf {
    pub config: KokoroConfig,
    pub weights: VarBuilder<'static>,
    pub voices: BTreeMap<String, VoicePack>,
}

/// Whether the file at `path` is a Kokoro GGUF.
pub fn is_kokoro_gguf(path: &Path) -> bool {
    let read = || -> Result<bool> {
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
        Ok(peek_architecture(&mut file)?.is_some_and(|a| a == ARCHITECTURE))
    };
    path.extension().is_some_and(|e| e == GGUF_EXTENSION)
        && path.is_file()
        && read().unwrap_or(false)
}

fn strings(content: &Content, key: &str) -> Result<Vec<String>> {
    match content.metadata.get(key) {
        Some(v) => v.to_vec()?.iter().map(|s| s.to_string().cloned()).collect(),
        None => bail!("Kokoro GGUF has no {key}"),
    }
}

/// Reads a Kokoro GGUF, dequantizing every weight to F32 on `device`.
pub fn read_kokoro_gguf(path: &Path, device: &Device) -> Result<KokoroGguf> {
    let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
    let content = Content::read(&mut file)?;
    match content.metadata.get(ARCHITECTURE_KEY) {
        Some(Value::String(a)) if a == ARCHITECTURE => {}
        other => bail!(
            "{} is not a Kokoro GGUF (architecture {other:?})",
            path.display()
        ),
    }
    let files = embedded_files(&content, &mut file)?;
    let Some(config) = files.get(CONFIG_FILE) else {
        bail!("Kokoro GGUF embeds no {CONFIG_FILE}")
    };
    let config: KokoroConfig =
        serde_json::from_slice(config).map_err(inference_tensor::Error::wrap)?;
    let voices = files
        .iter()
        .filter_map(|(name, bytes)| {
            let voice = name.strip_prefix(VOICE_DIR)?.strip_suffix(VOICE_EXT)?;
            Some(VoicePack::from_f32_le(bytes).map(|pack| (voice.to_string(), pack)))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;

    let mut tensors = HashMap::new();
    for (i, name) in strings(&content, TENSOR_NAMES_KEY)?.into_iter().enumerate() {
        let stored = format!("{TENSOR_PREFIX}{i}");
        let t = content
            .tensor(&mut file, &stored, &Device::Cpu)
            .map_err(|e| inference_tensor::Error::Msg(format!("{name} ({stored}): {e}")))?
            .dequantize(device)?;
        let shape = match content
            .metadata
            .get(&format!("{TENSOR_SHAPE_PREFIX}{name}"))
        {
            Some(v) => v
                .to_vec()?
                .iter()
                .map(|d| d.to_i64().map(|d| d as usize))
                .collect::<Result<Vec<_>>>()?,
            None => t.dims().to_vec(),
        };
        tensors.insert(name, t.to_dtype(DType::F32)?.reshape(shape)?);
    }
    Ok(KokoroGguf {
        config,
        weights: VarBuilder::from_tensors(tensors, DType::F32, device),
        voices,
    })
}

/// The embedded file table: names, n + 1 offsets into one byte blob.
fn embedded_files<R: Read + Seek>(
    content: &Content,
    reader: &mut R,
) -> Result<HashMap<String, Vec<u8>>> {
    let names = strings(content, FILE_NAMES_KEY)?;
    let offsets = match content.metadata.get(FILE_OFFSETS_KEY) {
        Some(v) => v
            .to_vec()?
            .iter()
            .map(|o| o.to_u64())
            .collect::<Result<Vec<_>>>()?,
        None => bail!("Kokoro GGUF has no {FILE_OFFSETS_KEY}"),
    };
    if offsets.len() != names.len() + 1 {
        bail!(
            "{} embedded files need {} offsets, got {}",
            names.len(),
            names.len() + 1,
            offsets.len()
        )
    }
    // only the files the model reads, so the multi-hundred-MB G2P resources stay on disk
    names
        .into_iter()
        .zip(offsets.windows(2))
        .filter(|(name, _)| {
            name == CONFIG_FILE || (name.starts_with(VOICE_DIR) && name.ends_with(VOICE_EXT))
        })
        .map(|(name, range)| {
            let bytes = match content.metadata.get(FILE_DATA_KEY) {
                Some(Value::Array(small)) => small
                    .get(range[0] as usize..range[1] as usize)
                    .ok_or_else(|| {
                        inference_tensor::Error::Msg(format!("{name} is past the embedded data"))
                    })?
                    .iter()
                    .map(|b| b.to_u8())
                    .collect::<Result<Vec<_>>>()?,
                _ => content.read_large_array(reader, FILE_DATA_KEY, range[0]..range[1])?,
            };
            Ok((name, bytes))
        })
        .collect()
}
