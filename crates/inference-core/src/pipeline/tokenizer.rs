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

/// The tokenizer files a checkpoint may ship in place of `tokenizer.json`: Mistral's and Qwen 1's tiktoken ranks.
pub(crate) const TEKKEN_TOKENIZER: &str = "tekken.json";
pub(crate) const QWEN_TIKTOKEN_TOKENIZER: &str = "qwen.tiktoken";
pub(crate) const NATIVE_TOKENIZERS: [&str; 2] = [TEKKEN_TOKENIZER, QWEN_TIKTOKEN_TOKENIZER];

// tokenization_qwen.py: the specials follow the 151,643 ranks
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
fn load_qwen_tiktoken<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
    let raw = std::fs::read_to_string(path)?;
    let mut bpe_ranks: HashMap<Vec<u8>, u32> = HashMap::new();
    for line in raw.lines().filter(|line| !line.is_empty()) {
        let (token, rank) = line
            .split_once(' ')
            .ok_or_else(|| anyhow!("qwen.tiktoken line without a rank: {line:?}"))?;
        let token = base64::engine::general_purpose::STANDARD.decode(token)?;
        bpe_ranks.insert(token, rank.parse()?);
    }
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

/// May fix the tokenizer according to: https://gist.github.com/jneuff/682d47b786329f19291d166957b3274a
pub(crate) fn get_tokenizer<P: AsRef<Path> + Clone>(
    p: P,
    processor_added_tokens: Option<&[&str]>,
) -> Result<Tokenizer> {
    let native = match p.as_ref().file_name().and_then(|name| name.to_str()) {
        Some(TEKKEN_TOKENIZER) => Some(load_tekken_tokenizer(p.clone())?),
        Some(QWEN_TIKTOKEN_TOKENIZER) => Some(load_qwen_tiktoken(p.clone())?),
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
    use super::{load_qwen_tiktoken, repair_tokenizer_json, tokenizer_from_bytes};
    use base64::Engine;
    use serde_json::{Value, json};

    // every byte, then `<a`: the pattern keeps `<a` one piece, which GPT-2's regex would split into `<` and `a`
    const MERGED: &[u8] = b"<a";
    const MERGED_RANK: u32 = 256;

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
