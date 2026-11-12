use std::{collections::HashMap, path::Path};

use ahash::AHashMap;
use anyhow::{Result, anyhow};
use base64::Engine;
use inference_nn::utils::tokenizer::tokenizer_from_bytes;
use serde_json::Value;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BpeBuilder,
    pre_tokenizers::{
        PreTokenizerWrapper,
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
    tokenizer::{self, Tokenizer, normalizer::SplitDelimiterBehavior},
};

use super::tiktoken::token_bytes_to_string;

#[derive(inference_json::FromJson)]
struct AddedToken {
    id: usize,
    content: String,
}

// transformers adds the tokens `tokenizer_config.json` declares beside `tokenizer.json`; older repos list them only there
fn merge_config_added_tokens(raw: Vec<u8>, config: &Path) -> Result<Vec<u8>> {
    let Ok(config) = std::fs::read(config) else {
        return Ok(raw);
    };
    let config: Value = serde_json::from_slice(&config)?;
    let Some(declared) = config[ADDED_TOKENS_DECODER].as_object() else {
        return Ok(raw);
    };
    let mut tokenizer: Value = serde_json::from_slice(&raw)?;
    let present = tokenizer["added_tokens"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|token| token["content"].as_str().map(str::to_string))
        .collect::<std::collections::HashSet<_>>();
    let vocab = tokenizer["model"]["vocab"].as_object();
    let mut missing = declared
        .iter()
        .filter_map(|(id, token)| Some((id.parse::<u64>().ok()?, token)))
        .filter(|(id, token)| {
            token["content"].as_str().is_some_and(|content| {
                !present.contains(content)
                    && vocab
                        .and_then(|vocab| vocab.get(content))
                        .is_none_or(|known| known.as_u64() == Some(*id))
            })
        })
        .map(|(id, token)| {
            let mut token = token.clone();
            token["id"] = id.into();
            (id, token)
        })
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(raw);
    }
    missing.sort_by_key(|(id, _)| *id);
    if !tokenizer["added_tokens"].is_array() {
        tokenizer["added_tokens"] = Value::Array(Vec::new());
    }
    let added = tokenizer["added_tokens"]
        .as_array_mut()
        .expect("added_tokens was made an array above");
    added.extend(missing.into_iter().map(|(_, token)| token));
    Ok(serde_json::to_vec(&tokenizer)?)
}

fn repair_tokenizer_json(raw: Vec<u8>) -> Result<Vec<u8>> {
    let mut tokenizer: Value = serde_json::from_slice(&raw)?;
    let Some(vocab) = tokenizer["model"]["vocab"].as_object() else {
        return Ok(raw);
    };
    let added_tokens: Vec<AddedToken> = serde_json::from_value(tokenizer["added_tokens"].clone())?;
    let missing_tokens = added_tokens
        .into_iter()
        .filter(|token| !vocab.contains_key(&token.content))
        .collect::<Vec<_>>();

    if missing_tokens.is_empty() {
        return Ok(raw);
    }

    let vocab = tokenizer["model"]["vocab"]
        .as_object_mut()
        .expect("tokenizer model vocab was validated above");
    for token in missing_tokens {
        vocab.insert(token.content, token.id.into());
    }
    Ok(serde_json::to_vec(&tokenizer)?)
}

#[derive(inference_json::FromJson)]
struct TekkenVocabEntry {
    rank: usize,
    token_bytes: String,
}

#[derive(inference_json::FromJson)]
struct TekkenSpecialToken {
    rank: usize,
    token_str: String,
}

#[derive(inference_json::FromJson)]
struct TekkenConfig {
    pattern: String,
    default_vocab_size: usize,
    default_num_special_tokens: usize,
}

#[derive(inference_json::FromJson)]
struct TekkenTokenizer {
    config: TekkenConfig,
    vocab: Vec<TekkenVocabEntry>,
    #[serde(default)]
    special_tokens: Vec<TekkenSpecialToken>,
}

/// The tokenizer files a checkpoint may ship in place of `tokenizer.json`: Mistral's, Qwen 1's and Kimi's tiktoken
/// ranks, and RWKV's world vocabulary.
pub(crate) const TEKKEN_TOKENIZER: &str = "tekken.json";
pub(crate) const QWEN_TIKTOKEN_TOKENIZER: &str = "qwen.tiktoken";
pub(crate) const KIMI_TIKTOKEN_TOKENIZER: &str = "tiktoken.model";
// a slow tokenizer's files: the vocabulary, with its merges and config beside it
pub(crate) const BPE_VOCAB: &str = "vocab.json";
pub(crate) const BPE_MERGES: &str = "merges.txt";
pub(crate) const NATIVE_TOKENIZERS: [&str; 5] = [
    TEKKEN_TOKENIZER,
    QWEN_TIKTOKEN_TOKENIZER,
    KIMI_TIKTOKEN_TOKENIZER,
    inference_gguf::rwkv_tokenizer::RWKV_VOCAB_FILE,
    BPE_VOCAB,
];
// the slow tokenizer classes whose files build the fast one transformers converts them to
const QWEN2_TOKENIZER_CLASSES: [&str; 2] = ["Qwen2Tokenizer", "Qwen2TokenizerFast"];
const MERGES_HEADER: &str = "#version";
// tokenization_kimi.py: the specials follow the ranks, two past its reserved count
const KIMI_SPECIALS: usize = 256 + 2;
// the embedding's rows when a vocabulary file's config does not say
const RWKV_DEFAULT_VOCAB: usize = 65_536;
const RWKV_ADDED_TOKENS: &str = "added_tokens.json";

// tokenization_qwen.py: the specials follow the 151,643 ranks
const TOKENIZER_CONFIG: &str = "tokenizer_config.json";
const ADDED_TOKENS_DECODER: &str = "added_tokens_decoder";
const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN_SPECIALS: [&str; 3] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];
const QWEN_EXTRA_SPECIALS: usize = 205;

fn load_tekken_tokenizer<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
    let raw = std::fs::read(path)?;
    let tekken: TekkenTokenizer = serde_json::from_slice(&raw)?;

    let num_special = tekken.config.default_num_special_tokens;
    let inner_vocab_size = tekken.config.default_vocab_size - num_special;

    // Build bpe_ranks: bytes -> rank (only ranks < inner_vocab_size)
    let mut bpe_ranks: HashMap<Vec<u8>, u32> = HashMap::new();
    for entry in &tekken.vocab {
        if entry.rank >= inner_vocab_size {
            continue;
        }
        let token_bytes = base64::engine::general_purpose::STANDARD.decode(&entry.token_bytes)?;
        #[allow(clippy::cast_possible_truncation)]
        bpe_ranks.insert(token_bytes, entry.rank as u32);
    }

    // Build special token name lookup from the parsed special_tokens field
    let mut special_token_map: HashMap<usize, String> = HashMap::new();
    for st in &tekken.special_tokens {
        special_token_map.insert(st.rank, st.token_str.clone());
    }
    // special tokens occupy 0..num_special, so the ranks' ids are offset past them
    #[allow(clippy::cast_possible_truncation)]
    let specials = (0..num_special)
        .map(|id| {
            let name = special_token_map
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("<SPECIAL_{id}>"));
            (name, id as u32)
        })
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    bpe_from_ranks(
        &bpe_ranks,
        num_special as u32,
        specials,
        tekken.config.pattern,
    )
}

// Lines of `<base64 token> <rank>`
fn tiktoken_ranks(path: &Path) -> Result<HashMap<Vec<u8>, u32>> {
    let raw = std::fs::read_to_string(path)?;
    let mut bpe_ranks: HashMap<Vec<u8>, u32> = HashMap::new();
    for line in raw.lines().filter(|line| !line.is_empty()) {
        let (token, rank) = line
            .split_once(' ')
            .ok_or_else(|| anyhow!("tiktoken line without a rank: {line:?}"))?;
        let token = base64::engine::general_purpose::STANDARD.decode(token)?;
        bpe_ranks.insert(token, rank.parse()?);
    }
    Ok(bpe_ranks)
}

// Kimi's specials take the tokenizer config's names, `<|reserved_token_N|>` where it gives none
fn load_kimi_tiktoken(path: &Path) -> Result<Tokenizer> {
    let bpe_ranks = tiktoken_ranks(path)?;
    let names = std::fs::read_to_string(path.with_file_name(TOKENIZER_CONFIG))
        .ok()
        .and_then(|config| serde_json::from_str::<serde_json::Value>(&config).ok())
        .and_then(|config| config.get(ADDED_TOKENS_DECODER).cloned())
        .and_then(|decoder| decoder.as_object().cloned())
        .unwrap_or_default();
    #[allow(clippy::cast_possible_truncation)]
    let first_special = bpe_ranks.len() as u32;
    #[allow(clippy::cast_possible_truncation)]
    let specials = (first_special..first_special + KIMI_SPECIALS as u32)
        .map(|id| {
            let name = names
                .get(&id.to_string())
                .and_then(|token| token.get("content")?.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("<|reserved_token_{id}|>"));
            (name, id)
        })
        .collect();
    bpe_from_ranks(
        &bpe_ranks,
        0,
        specials,
        inference_gguf::KIMI_K2_REGEX.to_string(),
    )
}

fn load_qwen_tiktoken<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
    let bpe_ranks = tiktoken_ranks(path.as_ref())?;
    #[allow(clippy::cast_possible_truncation)]
    let first_special = bpe_ranks.len() as u32;
    let names = QWEN_SPECIALS
        .iter()
        .map(|name| name.to_string())
        .chain((0..QWEN_EXTRA_SPECIALS).map(|i| format!("<|extra_{i}|>")));
    let specials = names.zip(first_special..).collect();
    bpe_from_ranks(&bpe_ranks, 0, specials, QWEN_PATTERN.to_string())
}

// A byte-level BPE over tiktoken's ranks: ids are the ranks plus `offset`, merges come from each multi-byte token's
// splits into two ranked pieces (as tiktoken.rs builds them), and `specials` are added under their own ids
fn bpe_from_ranks(
    bpe_ranks: &HashMap<Vec<u8>, u32>,
    offset: u32,
    specials: Vec<(String, u32)>,
    pattern: String,
) -> Result<Tokenizer> {
    let mut vocab = AHashMap::new();
    for (token_bytes, rank) in bpe_ranks {
        vocab.insert(token_bytes_to_string(token_bytes), *rank + offset);
    }

    let mut merges = Vec::new();
    for (token, rank) in bpe_ranks {
        if token.len() == 1 {
            continue;
        }
        let mut local = Vec::new();
        for index in 1..token.len() {
            let piece_l = &token[..index];
            let piece_r = &token[index..];
            if let (Some(&rank_l), Some(&rank_r)) = (bpe_ranks.get(piece_l), bpe_ranks.get(piece_r))
            {
                local.push((piece_l.to_vec(), piece_r.to_vec(), *rank, rank_l, rank_r));
            }
        }
        local.sort_by_key(|(_, _, _, rank_l, rank_r)| (*rank_l, *rank_r));
        for (piece_l, piece_r, rank, _, _) in local {
            merges.push((piece_l, piece_r, rank));
        }
    }
    merges.sort_by_key(|(_, _, rank)| *rank);
    let merges: Vec<(String, String)> = merges
        .into_iter()
        .map(|(l, r, _)| (token_bytes_to_string(&l), token_bytes_to_string(&r)))
        .collect();

    for (name, id) in &specials {
        vocab.insert(name.clone(), *id);
    }
    let bpe_model = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!("Failed to build BPE model: {}", e))?;

    let mut tokenizer = Tokenizer::new(bpe_model);
    let split = Split::new(
        SplitPattern::Regex(pattern),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("Failed to create split pre-tokenizer: {}", e))?;
    // the split above is the whole pre-tokenization; ByteLevel only maps bytes, its GPT-2 regex would split again
    let byte_level = ByteLevel::new(false, true, false);
    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ]);
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(true, false, false)));
    tokenizer
        .add_special_tokens(
            specials
                .into_iter()
                .map(|(name, _)| tokenizer::AddedToken::from(name, true)),
        )
        .map_err(anyhow::Error::msg)?;

    Ok(tokenizer)
}

// transformers' conversion of `Qwen2Tokenizer`: NFC, Qwen's split, byte-level BPE over `vocab.json` and `merges.txt`,
// and the config's added tokens at their ids
fn load_qwen2_vocab(path: &Path) -> Result<Tokenizer> {
    let config: Value =
        serde_json::from_slice(&std::fs::read(path.with_file_name(TOKENIZER_CONFIG))?)?;
    let class = config["tokenizer_class"].as_str().unwrap_or_default();
    if !QWEN2_TOKENIZER_CLASSES.contains(&class) {
        anyhow::bail!(
            "`{}` holds a `{class}` vocabulary; only Qwen2's builds without a `tokenizer.json`",
            path.display()
        )
    }
    let vocab: AHashMap<String, u32> = serde_json::from_slice(&std::fs::read(path)?)?;
    let merges = std::fs::read_to_string(path.with_file_name(BPE_MERGES))?
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with(MERGES_HEADER))
        .map(|line| {
            line.split_once(' ')
                .map(|(l, r)| (l.to_string(), r.to_string()))
                .ok_or_else(|| anyhow!("`{line}` is not a merge"))
        })
        .collect::<Result<Vec<_>>>()?;
    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!("Failed to build BPE model: {e}"))?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer
        .with_normalizer(Some(tokenizers::normalizers::unicode::NFC))
        .map_err(anyhow::Error::msg)?;
    let split = Split::new(
        SplitPattern::Regex(QWEN_PATTERN.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("Failed to create split pre-tokenizer: {e}"))?;
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
    ])));
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(true, true, true)));
    let mut added = config[ADDED_TOKENS_DECODER]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(id, token)| {
            Some((
                id.parse::<u32>().ok()?,
                token["content"].as_str()?,
                token["special"].as_bool()?,
            ))
        })
        .collect::<Vec<_>>();
    added.sort_by_key(|(id, _, _)| *id);
    for (id, content, special) in added {
        let token = tokenizer::AddedToken::from(content, special);
        if special {
            tokenizer
                .add_special_tokens([token])
                .map_err(anyhow::Error::msg)?;
        } else {
            tokenizer.add_tokens([token]).map_err(anyhow::Error::msg)?;
        }
        if tokenizer.token_to_id(content) != Some(id) {
            anyhow::bail!("added token `{content}` does not follow the vocabulary at id {id}")
        }
    }
    Ok(tokenizer)
}

// RWKV's world vocabulary, padded to the config's vocabulary beside it as llama.cpp's converter pads it
fn load_rwkv_vocab(path: &Path) -> Result<Tokenizer> {
    let vocab_size = std::fs::read_to_string(path.with_file_name("config.json"))
        .ok()
        .and_then(|config| serde_json::from_str::<serde_json::Value>(&config).ok())
        .and_then(|config| usize::try_from(config.get("vocab_size")?.as_u64()?).ok())
        .unwrap_or(RWKV_DEFAULT_VOCAB);
    // the HF tokenizer's name for token 0 (its end of text), llama.cpp's `<s>` when none is given
    let first = std::fs::read_to_string(path.with_file_name(RWKV_ADDED_TOKENS))
        .ok()
        .and_then(|added| serde_json::from_str::<HashMap<String, u32>>(&added).ok())
        .and_then(|added| {
            added
                .into_iter()
                .find(|(_, id)| *id == 0)
                .map(|(name, _)| name)
        })
        .unwrap_or_else(|| inference_gguf::rwkv_tokenizer::RWKV_BOS.to_string());
    let tokens = inference_gguf::rwkv_tokenizer::parse_rwkv_vocab(
        &std::fs::read_to_string(path)?,
        &first,
        vocab_size,
    )?;
    inference_gguf::rwkv_tokenizer::rwkv_world_tokenizer(&tokens)
}

/// May fix the tokenizer according to: https://gist.github.com/jneuff/682d47b786329f19291d166957b3274a
pub(crate) fn get_tokenizer<P: AsRef<Path> + Clone>(
    p: P,
    processor_added_tokens: Option<&[&str]>,
) -> Result<Tokenizer> {
    let native = match p.as_ref().file_name().and_then(|name| name.to_str()) {
        Some(TEKKEN_TOKENIZER) => Some(load_tekken_tokenizer(p.clone())?),
        Some(QWEN_TIKTOKEN_TOKENIZER) => Some(load_qwen_tiktoken(p.clone())?),
        Some(KIMI_TIKTOKEN_TOKENIZER) => Some(load_kimi_tiktoken(p.as_ref())?),
        Some(inference_gguf::rwkv_tokenizer::RWKV_VOCAB_FILE) => Some(load_rwkv_vocab(p.as_ref())?),
        Some(BPE_VOCAB) => Some(load_qwen2_vocab(p.as_ref())?),
        _ => None,
    };
    if let Some(mut tokenizer) = native {
        if let Some(added_tokens) = processor_added_tokens {
            tokenizer
                .add_special_tokens(
                    added_tokens
                        .iter()
                        .map(|x| tokenizer::AddedToken::from(x.to_string(), true)),
                )
                .map_err(anyhow::Error::msg)?;
        }
        return Ok(tokenizer);
    }

    let mut tokenizer = {
        let raw = std::fs::read(p.clone()).map_err(anyhow::Error::msg)?;
        let raw = merge_config_added_tokens(raw, &p.as_ref().with_file_name(TOKENIZER_CONFIG))?;
        let raw_fixed = repair_tokenizer_json(raw)?;
        tokenizer_from_bytes(&raw_fixed)?
    };
    // the engine fits prompts itself; a saved fixed padding (as sentence-transformers exports) would pad every prompt
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(anyhow::Error::msg)?;
    if let Some(added_tokens) = processor_added_tokens {
        tokenizer
            .add_special_tokens(
                added_tokens
                    .iter()
                    .map(|x| tokenizer::AddedToken::from(x.to_string(), true)),
            )
            .map_err(anyhow::Error::msg)?;
    }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::{
        load_kimi_tiktoken, load_qwen_tiktoken, load_qwen2_vocab, merge_config_added_tokens,
        repair_tokenizer_json, tokenizer_from_bytes,
    };
    use base64::Engine;
    use serde_json::{Value, json};

    // every byte, then `<a`: the pattern keeps `<a` one piece, which GPT-2's regex would split into `<` and `a`
    const MERGED: &[u8] = b"<a";
    const MERGED_RANK: u32 = 256;

    // as transformers does, tokens only `tokenizer_config.json` declares join the tokenizer at their declared ids
    #[test]
    fn tokens_only_the_config_declares_join_at_their_ids() -> anyhow::Result<()> {
        let tokenizer = json!({
            "version": "1.0",
            "added_tokens": [],
            "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1, "[UNK]": 2}, "unk_token": "[UNK]"},
            "pre_tokenizer": {"type": "Whitespace"},
        });
        let flags = |content: &str| {
            json!({"content": content, "lstrip": false, "normalized": false, "rstrip": false,
                   "single_word": false, "special": true})
        };
        let dir = tempfile::tempdir()?;
        let config = dir.path().join(super::TOKENIZER_CONFIG);
        std::fs::write(
            &config,
            json!({"added_tokens_decoder": {"3": flags("<|x|>"), "4": flags("<|y|>")}}).to_string(),
        )?;
        let raw = merge_config_added_tokens(serde_json::to_vec(&tokenizer)?, &config)?;
        let tokenizer = tokenizer_from_bytes(&repair_tokenizer_json(raw)?)?;
        let encoding = tokenizer
            .encode("a <|x|> b<|y|>", false)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(encoding.get_ids(), [0, 3, 1, 4]);
        Ok(())
    }

    #[test]
    fn a_qwen_tiktoken_file_merges_by_rank_and_ids_its_specials_after_the_ranks()
    -> anyhow::Result<()> {
        let encode = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
        let mut ranks = (0..=u8::MAX)
            .map(|byte| format!("{} {byte}", encode(&[byte])))
            .collect::<Vec<_>>();
        ranks.push(format!("{} {MERGED_RANK}", encode(MERGED)));
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(super::QWEN_TIKTOKEN_TOKENIZER);
        std::fs::write(&path, ranks.join("\n"))?;
        let tokenizer = load_qwen_tiktoken(&path)?;
        let im_start = MERGED_RANK + 2;
        let encoding = tokenizer
            .encode("<|im_start|><a b", false)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(
            encoding.get_ids(),
            [im_start, MERGED_RANK, u32::from(b' '), u32::from(b'b')]
        );
        assert_eq!(
            tokenizer.token_to_id("<|extra_204|>"),
            Some(MERGED_RANK + 1 + 3 + 204)
        );
        Ok(())
    }

    // a slow Qwen2 tokenizer's files: byte-level BPE over the merges, the config's added tokens at their ids
    #[test]
    fn a_qwen2_vocabulary_builds_with_its_merges_and_added_tokens() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let vocab = dir.path().join(super::BPE_VOCAB);
        std::fs::write(
            &vocab,
            r#"{"a": 0, "b": 1, "\u0120": 2, "ab": 3, "\u0120b": 4}"#,
        )?;
        std::fs::write(
            dir.path().join(super::BPE_MERGES),
            "#version: 0.2\na b\n\u{120} b\n",
        )?;
        let config = |class: &str| {
            format!(
                r#"{{"tokenizer_class": "{class}", "added_tokens_decoder": {{
                    "5": {{"content": "<|im_start|>", "special": true}},
                    "6": {{"content": "<text>", "special": false}}}}}}"#
            )
        };
        let config_path = dir.path().join(super::TOKENIZER_CONFIG);
        std::fs::write(&config_path, config("Qwen2Tokenizer"))?;
        let tokenizer = load_qwen2_vocab(&vocab)?;
        let encoding = tokenizer
            .encode("<|im_start|>ab b<text>", false)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(encoding.get_ids(), [5, 3, 4, 6]);
        assert_eq!(
            tokenizer
                .decode(&[3, 4, 6], true)
                .map_err(anyhow::Error::msg)?,
            "ab b<text>"
        );
        // another class's vocabulary splits its own way
        std::fs::write(&config_path, config("GPT2Tokenizer"))?;
        assert!(load_qwen2_vocab(&vocab).is_err());
        Ok(())
    }

    // Kimi's specials follow the ranks under the config's names, unnamed ones reserved; a Han run splits from the
    // letters after it, which its split keeps out of the word runs
    #[test]
    fn a_kimi_tiktoken_file_names_its_specials_from_the_config_and_splits_han_apart()
    -> anyhow::Result<()> {
        let encode = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
        let han = "\u{4e2d}".as_bytes();
        let han_a = [han, b"a"].concat();
        // the byte ranks, then the Han character's prefix, the character, and the character with `a`
        let mut ranks = (0..=u8::MAX)
            .map(|byte| format!("{} {byte}", encode(&[byte])))
            .collect::<Vec<_>>();
        let pieces = [&han[..2], han, &han_a[..]];
        let base = 256u32;
        for (rank, piece) in (base..).zip(pieces) {
            ranks.push(format!("{} {rank}", encode(piece)));
        }
        let (han_id, first_special) = (base + 1, base + 3);
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(super::KIMI_TIKTOKEN_TOKENIZER);
        std::fs::write(&path, ranks.join("\n"))?;
        std::fs::write(
            dir.path().join(super::TOKENIZER_CONFIG),
            json!({"added_tokens_decoder": {first_special.to_string(): {"content": "[BOS]"}}})
                .to_string(),
        )?;
        let tokenizer = load_kimi_tiktoken(&path)?;
        assert_eq!(tokenizer.token_to_id("[BOS]"), Some(first_special));
        assert_eq!(
            tokenizer.token_to_id(&format!("<|reserved_token_{}|>", first_special + 1)),
            Some(first_special + 1)
        );
        let encoding = tokenizer
            .encode(std::str::from_utf8(&han_a)?, false)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(encoding.get_ids(), [han_id, u32::from(b'a')]);
        Ok(())
    }

    #[test]
    fn tokenizer_repair_preserves_bytes_when_vocab_is_complete() {
        let raw = serde_json::to_vec_pretty(&json!({
            "added_tokens": [{"id": 1, "content": "<special>"}],
            "model": {"vocab": {"token": 0, "<special>": 1}}
        }))
        .unwrap();

        assert_eq!(repair_tokenizer_json(raw.clone()).unwrap(), raw);
    }

    #[test]
    fn tokenizer_repair_adds_only_missing_tokens() {
        let raw = serde_json::to_vec(&json!({
            "added_tokens": [
                {"id": 1, "content": "<present>"},
                {"id": 2, "content": "<missing>"}
            ],
            "model": {"vocab": {"token": 0, "<present>": 1}}
        }))
        .unwrap();

        let repaired: Value = serde_json::from_slice(&repair_tokenizer_json(raw).unwrap()).unwrap();
        let vocab = repaired["model"]["vocab"].as_object().unwrap();
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab["<present>"], 1);
        assert_eq!(vocab["<missing>"], 2);
    }

    #[test]
    fn tokenizer_repair_preserves_unigram_array_vocab() {
        let raw = serde_json::to_vec_pretty(&json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{
                "id": 1,
                "content": "<special>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [["<unk>", 0.0], ["<special>", -1.0]],
                "byte_fallback": false
            }
        }))
        .unwrap();

        let repaired = repair_tokenizer_json(raw.clone()).unwrap();
        assert_eq!(repaired, raw);
        assert!(tokenizer_from_bytes(&repaired).is_ok());
    }
}
