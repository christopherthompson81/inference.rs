use std::{path::Path, str::FromStr};

use anyhow::Result;
use tokenizers::Tokenizer;

// tokenizers' from_file/from_bytes are generic and compile its deserializer in each caller; FromStr compiles once.
pub fn tokenizer_from_bytes(raw: &[u8]) -> Result<Tokenizer> {
    Tokenizer::from_str(std::str::from_utf8(raw)?).map_err(anyhow::Error::msg)
}

pub fn tokenizer_from_file(path: &Path) -> Result<Tokenizer> {
    tokenizer_from_bytes(&std::fs::read(path)?)
}
