//! Qwen2's byte-level BPE from `vocab.json` and `merges.txt`, as Qwen3-TTS ships it (no `tokenizer.json`).

use std::path::Path;

use tokenizers::models::bpe::{BPE, Merges, Vocab};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer};

const VOCAB: &str = "vocab.json";
const MERGES: &str = "merges.txt";
const TOKENIZER_CONFIG: &str = "tokenizer_config.json";
// Qwen2's pre-tokenizer split, from its tokenizer.json
const QWEN2_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(inference_json::FromJson)]
struct AddedTokenEntry {
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(inference_json::FromJson)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: std::collections::BTreeMap<String, AddedTokenEntry>,
}

/// The tokenizer in `dir`: its `tokenizer.json` when present, else built from the BPE files and the added tokens.
pub fn load(dir: &Path) -> anyhow::Result<Tokenizer> {
    let json = dir.join("tokenizer.json");
    if json.exists() {
        return Tokenizer::from_file(&json).map_err(anyhow::Error::msg);
    }
    let vocab: Vocab = serde_json::from_str(&std::fs::read_to_string(dir.join(VOCAB))?)?;
    let merges = std::fs::read_to_string(dir.join(MERGES))?
        .lines()
        .filter(|line| !line.starts_with("#version") && !line.trim().is_empty())
        .map(|line| {
            line.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .ok_or_else(|| anyhow::anyhow!("bad merge `{line}`"))
        })
        .collect::<anyhow::Result<Merges>>()?;
    let config: TokenizerConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join(TOKENIZER_CONFIG))?)?;
    let added = config
        .added_tokens_decoder
        .into_iter()
        .map(|(id, entry)| Ok((id.parse::<u32>()?, entry.content, entry.special)))
        .collect::<anyhow::Result<_>>()?;
    bpe(vocab, merges, added)
}

/// Qwen2's BPE over `vocab` and `merges`, with `added` tokens (id, text, special) at their ids.
pub fn bpe(
    vocab: Vocab,
    merges: Merges,
    mut added: Vec<(u32, String, bool)>,
) -> anyhow::Result<Tokenizer> {
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(anyhow::Error::msg)?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer
        .with_normalizer(Some(NFC))
        .map_err(anyhow::Error::msg)?;
    let split = Split::new(
        SplitPattern::Regex(QWEN2_SPLIT.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(anyhow::Error::msg)?;
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ])));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    added.sort_by_key(|(id, _, _)| *id);
    for (id, content, special) in added {
        let token = AddedToken::from(content.clone(), special);
        if special {
            tokenizer.add_special_tokens([token])
        } else {
            tokenizer.add_tokens([token])
        }
        .map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            tokenizer.token_to_id(&content) == Some(id),
            "added token `{content}` did not land at id {id}"
        );
    }
    Ok(tokenizer)
}
