// Adapted from toktrie_hf_tokenizers 1.4.0 (guidance-ai/llguidance, MIT, see crates/inference-core/third_party).

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{Result, anyhow, bail};
use tokenizers::{NormalizerWrapper, Tokenizer, normalizers::Sequence};
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use tracing::warn;

/// A Hugging Face tokenizer's vocabulary as the raw bytes of each token, the form toktrie and llguidance work in.
pub(crate) struct ByteTokenizer {
    hf_tokenizer: Tokenizer,
    info: TokRxInfo,
    token_bytes: Vec<Vec<u8>>,
}

// GPT-2 byte-level BPE maps bytes outside these ranges to chars from U+0100 up
fn is_self_mapped(c: char) -> bool {
    matches!(c, '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}')
}

fn build_char_map() -> HashMap<char, u8> {
    let mut res = HashMap::default();
    let mut k = 0x100u32;
    for byte in 0..=255u8 {
        let c = byte as char;
        if is_self_mapped(c) {
            res.insert(c, byte);
        } else {
            res.insert(char::from_u32(k).unwrap(), byte);
            k += 1;
        }
    }
    res
}

impl ByteTokenizer {
    pub(crate) fn from_tokenizer(mut hft: Tokenizer) -> Result<ByteTokenizer> {
        let mut is_byte_level = false;
        let mut is_byte_fallback = false;
        let mut space_ch = ' ';

        // the "prepend space" normalizer would add a space to every encoded fragment
        if let Some(n) = hft.get_normalizer() {
            let n = match n {
                NormalizerWrapper::Sequence(x) => NormalizerWrapper::Sequence(Sequence::new(
                    x.as_ref()
                        .iter()
                        .filter_map(|n| match n {
                            NormalizerWrapper::Prepend(_) => None,
                            _ => Some(n.clone()),
                        })
                        .collect(),
                )),
                _ => n.clone(),
            };
            hft.with_normalizer(Some(n)).map_err(anyhow::Error::msg)?;
        }

        if let Some(d) = hft.get_decoder() {
            // DecoderWrapper::Sequence does not expose its decoders, so read them from the serialized form
            let v = serde_json::to_value(d)?;
            if v["type"].as_str() == Some("ByteLevel") {
                is_byte_level = true;
            } else if v["type"].as_str() == Some("Sequence")
                && let Some(decoders) = v["decoders"].as_array()
            {
                for decoder in decoders {
                    if decoder["type"].as_str() == Some("ByteFallback") {
                        is_byte_fallback = true;
                    } else if decoder["type"].as_str() == Some("Replace")
                        && decoder["content"].as_str() == Some(" ")
                        && let Some(s) = decoder["pattern"]["String"].as_str()
                    {
                        let s: Vec<char> = s.chars().collect();
                        if s.len() == 1 {
                            space_ch = s[0];
                        }
                    }
                }
            }
        }

        if !is_byte_fallback && !is_byte_level {
            bail!("can't determine decoder type: {:?}", hft.get_decoder());
        }

        let vocab_size = u32::try_from(hft.get_vocab_size(true))?;
        let mut added = hft
            .get_added_tokens_decoder()
            .into_iter()
            .collect::<Vec<_>>();
        added.sort_by_key(|(id, _)| *id);

        let mut res = ByteTokenizer {
            info: TokRxInfo::new(vocab_size, 0),
            token_bytes: (0..vocab_size).map(|_| Vec::new()).collect(),
            hf_tokenizer: hft,
        };

        let mut specials = HashSet::new();

        for (id, info) in added.iter() {
            // every added token of the form <...> counts as special
            if info.special || (info.content.starts_with('<') && info.content.ends_with('>')) {
                match info.content.as_str() {
                    "</s>"
                    | "<|endoftext|>"
                    | "<|end_of_text|>"
                    | "<｜end▁of▁sentence｜>" // DeepSeek's fullwidth bars
                    | "<eos>" => res.info.tok_eos = *id,

                    "<|end|>" | "<|eot_id|>" | "<|im_end|>" => res.info.tok_end_of_turn = Some(*id),
                    "<unk>" | "<|unk|>" => res.info.tok_unk = Some(*id),
                    "<pad>" | "<|pad|>" => res.info.tok_pad = Some(*id),
                    _ => {}
                }
                specials.insert(*id);
            } else {
                res.token_bytes[*id as usize] = info.content.clone().into_bytes();
            }
        }

        let char_map = build_char_map();

        for tok_id in 0..vocab_size {
            let Some(tok_name) = res.hf_tokenizer.id_to_token(tok_id) else {
                warn!("missing token: {tok_id}");
                continue;
            };
            let bytes = if specials.contains(&tok_id) {
                let mut bytes = tok_name.as_bytes().to_vec();
                bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
                bytes
            } else if is_byte_fallback {
                if tok_name.len() == 6 && tok_name.starts_with("<0x") && tok_name.ends_with('>') {
                    vec![u8::from_str_radix(&tok_name[3..5], 16)?]
                } else {
                    assert!(!tok_name.starts_with("<0x"));
                    tok_name.replace(space_ch, " ").into_bytes()
                }
            } else {
                let bytes: Result<Vec<u8>> = tok_name
                    .chars()
                    .map(|c| {
                        char_map
                            .get(&c)
                            .copied()
                            .ok_or_else(|| anyhow!("missing char: {c}"))
                    })
                    .collect();
                match bytes {
                    Ok(b) => b,
                    Err(e) => {
                        warn!("error: {e} for {tok_name:?}");
                        continue;
                    }
                }
            };
            res.token_bytes[tok_id as usize] = bytes;
        }

        Ok(res)
    }

    pub(crate) fn tokrx_info(&self) -> TokRxInfo {
        self.info
    }

    pub(crate) fn token_bytes(&self) -> Vec<Vec<u8>> {
        self.token_bytes.clone()
    }
}

pub(crate) struct ByteTokenizerEnv {
    pub tokenizer: ByteTokenizer,
    pub tok_trie: TokTrie,
}

impl ByteTokenizerEnv {
    pub(crate) fn into_env(self) -> TokEnv {
        Arc::new(self)
    }
}

impl TokenizerEnv for ByteTokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tokenizer
                .hf_tokenizer
                .encode(s, false)
                .expect("tokenizer error")
                .get_ids()
                .to_vec()
        })
    }

    fn tokenize_bytes_special(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tok_trie.tokenize_with_special(s, |s| {
                self.tokenizer
                    .hf_tokenizer
                    .encode(s, false)
                    .expect("tokenizer error")
                    .get_ids()
                    .to_vec()
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    const MINIMAL_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "vocab": {
                "a": 0
            },
            "merges": []
        }
    }"#;

    #[test]
    fn tokenize_special_respects_toktrie_specials() {
        let hf_tokenizer = Tokenizer::from_str(MINIMAL_TOKENIZER_JSON).unwrap();
        let info = TokRxInfo::new(2, 0);
        let mut special_bytes = vec![TokTrie::SPECIAL_TOKEN_MARKER];
        special_bytes.extend_from_slice(b"<|end|>");
        let token_bytes = vec![b"a".to_vec(), special_bytes];
        let tok_trie = TokTrie::from(&info, &token_bytes);
        let env = ByteTokenizerEnv {
            tokenizer: ByteTokenizer {
                hf_tokenizer,
                info,
                token_bytes,
            },
            tok_trie,
        };
        let special_id = env.tok_trie().get_special_token("<|end|>").unwrap();
        assert_eq!(env.tokenize("<|end|>"), Vec::<TokenId>::new());
        assert_eq!(env.tokenize_special("<|end|>"), vec![special_id]);
    }

    #[test]
    fn byte_level_vocab_decodes_to_raw_bytes() {
        let json = MINIMAL_TOKENIZER_JSON.replace(r#""a": 0"#, r#""a": 0, "Ġb": 1"#);
        let bt = ByteTokenizer::from_tokenizer(Tokenizer::from_str(&json).unwrap()).unwrap();
        assert_eq!(bt.token_bytes(), vec![b"a".to_vec(), b" b".to_vec()]);
    }
}
