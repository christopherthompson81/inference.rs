//! RWKV's world tokenizer: a greedy longest match over byte tokens (`rwkv_vocab_v20230424.txt`, or a GGUF's `rwkv`
//! token list). It runs as a Hugging Face tokenizer: the bytes mapped to GPT-2's byte-level characters, then a split by
//! one regex alternating every token longest first (leftmost-first, so each match is the longest token at its
//! position, as the greedy match takes it), then each piece looked up whole.

use anyhow::{Context, Result};
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::tokenizer::normalizer::SplitDelimiterBehavior;
use tokenizers::{AddedToken, DecoderWrapper, ModelWrapper, PreTokenizerWrapper, Tokenizer};

/// The token llama.cpp's converter puts first, before the vocabulary's own (which start at id 1); the HF tokenizer
/// names it in `added_tokens.json`.
pub const RWKV_BOS: &str = "<s>";
/// The vocabulary file an RWKV checkpoint ships in place of a tokenizer.json.
pub const RWKV_VOCAB_FILE: &str = "rwkv_vocab_v20230424.txt";
// llama.cpp's padding past the vocabulary to the embedding's rows
const PAD_PREFIX: &str = "[PAD";

/// The vocabulary file's tokens by id, `first` at 0 (llama.cpp's `<s>`, the HF tokenizer's end of text) and padded
/// to `vocab_size` as llama.cpp's converter writes them.
pub fn parse_rwkv_vocab(text: &str, first: &str, vocab_size: usize) -> Result<Vec<Vec<u8>>> {
    let mut tokens = vec![first.as_bytes().to_vec()];
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        // `<id> <python literal> <byte length>`, the literal possibly holding spaces
        let (id, rest) = line
            .split_once(' ')
            .with_context(|| format!("RWKV vocab line {}: no id", n + 1))?;
        let (literal, len) = rest
            .rsplit_once(' ')
            .with_context(|| format!("RWKV vocab line {}: no length", n + 1))?;
        let token = python_literal(literal)
            .with_context(|| format!("RWKV vocab line {}: `{literal}`", n + 1))?;
        let (id, len) = (id.parse::<usize>()?, len.parse::<usize>()?);
        anyhow::ensure!(
            id == tokens.len() && len == token.len(),
            "RWKV vocab line {}: id {id} of {len} bytes, expected id {} of {} bytes",
            n + 1,
            tokens.len(),
            token.len()
        );
        tokens.push(token);
    }
    while tokens.len() < vocab_size {
        tokens.push(format!("{PAD_PREFIX}{}]", tokens.len()).into_bytes());
    }
    Ok(tokens)
}

// a Python `'...'` or `b'...'` literal: in a str `\xHH` is a code point, in bytes a byte
fn python_literal(literal: &str) -> Result<Vec<u8>> {
    let (bytes, rest) = match literal.strip_prefix('b') {
        Some(rest) => (true, rest),
        None => (false, literal),
    };
    let quote = rest.chars().next().context("empty literal")?;
    anyhow::ensure!(matches!(quote, '\'' | '"'), "not a quoted literal");
    let inner = rest
        .strip_prefix(quote)
        .and_then(|r| r.strip_suffix(quote))
        .context("unterminated literal")?;
    let mut out = Vec::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        let escaped = chars.next().context("a trailing backslash")?;
        let mut hex = |digits: usize| -> Result<u32> {
            let s = chars.by_ref().take(digits).collect::<String>();
            Ok(u32::from_str_radix(&s, 16)?)
        };
        let code = match escaped {
            'n' => u32::from('\n'),
            't' => u32::from('\t'),
            'r' => u32::from('\r'),
            '0' => 0,
            'x' => hex(2)?,
            'u' if !bytes => hex(4)?,
            'U' if !bytes => hex(8)?,
            other => u32::from(other),
        };
        if bytes && escaped == 'x' {
            out.push(u8::try_from(code)?);
        } else {
            let c = char::from_u32(code).context("not a code point")?;
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    }
    Ok(out)
}

/// A GGUF `rwkv` token as llama.cpp unescapes it: `\t`, `\n`, `\r`, `\xHH` and `\<c>` for any other `c`.
pub fn unescape_gguf_rwkv_token(escaped: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(escaped.len());
    let bytes = escaped.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'\\' || i + 1 >= bytes.len() {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        match bytes[i + 1] {
            b't' => out.push(b'\t'),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b'x' if i + 4 <= bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 2..i + 4]).unwrap_or("00");
                out.push(u8::from_str_radix(hex, 16).unwrap_or(0));
                i += 4;
                continue;
            }
            other => out.push(other),
        }
        i += 2;
    }
    out
}

// GPT-2's byte-to-character map, as the byte-level pre-tokenizer applies it
fn byte_chars() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..=255u8 {
        let printable =
            (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        map[b as usize] = if printable {
            char::from(b)
        } else {
            extra += 1;
            char::from_u32(255 + extra).expect("a code point")
        };
    }
    map
}

/// The tokenizer over `tokens` (by id): greedy longest match, token 0 the only special token.
pub fn rwkv_world_tokenizer(tokens: &[Vec<u8>]) -> Result<Tokenizer> {
    let first = String::from_utf8(tokens.first().context("an empty vocabulary")?.clone())?;
    let chars = byte_chars();
    let mapped = tokens
        .iter()
        .map(|t| t.iter().map(|&b| chars[b as usize]).collect::<String>())
        .collect::<Vec<_>>();
    let mut vocab = ahash::AHashMap::with_capacity(mapped.len());
    for (id, token) in mapped.iter().enumerate() {
        // a later duplicate (the padding never collides with a real token) keeps the first id, as the trie does
        vocab.entry(token.clone()).or_insert(u32::try_from(id)?);
    }
    // the vocabulary's own tokens, longest first; `<s>` and the padding are never matched in text
    let mut alternatives = mapped
        .iter()
        .enumerate()
        .filter(|(id, t)| *id != 0 && !t.starts_with(PAD_PREFIX))
        .map(|(_, t)| t.as_str())
        .collect::<Vec<_>>();
    alternatives.sort_by_key(|t| std::cmp::Reverse(t.chars().count()));
    let pattern = alternatives
        .iter()
        .map(|t| fancy_regex_escape(t))
        .collect::<Vec<_>>()
        .join("|");
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token(first.clone())
        .build()
        .map_err(anyhow::Error::msg)?;
    let mut tokenizer = Tokenizer::new(ModelWrapper::WordLevel(model));
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        PreTokenizerWrapper::Split(
            Split::new(
                SplitPattern::Regex(pattern),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(anyhow::Error::msg)?,
        ),
    ]))));
    tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevelDecoder::new(
        false, false, false,
    ))));
    tokenizer
        .add_special_tokens([AddedToken::from(first, true)])
        .map_err(anyhow::Error::msg)?;
    Ok(tokenizer)
}

// regex syntax characters escaped; every other character (the byte-level ones included) stands for itself
fn fancy_regex_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if "\\.+*?()|[]{}^$#&-~".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_literals_read_as_their_bytes() -> Result<()> {
        assert_eq!(python_literal(r"'\x00'")?, [0]);
        assert_eq!(python_literal(r"b'\xe2\x80'")?, [0xe2, 0x80]);
        assert_eq!(python_literal("'\\xe9'")?, "\u{e9}".as_bytes());
        assert_eq!(python_literal(r#""it's""#)?, b"it's");
        assert_eq!(python_literal(r"'a b\n'")?, b"a b\n");
        assert_eq!(
            unescape_gguf_rwkv_token(r"\xff\n\\x"),
            [0xff, b'\n', b'\\', b'x']
        );
        Ok(())
    }

    // the longest token at each position, as a greedy trie match takes it, not the fewest tokens overall
    #[test]
    fn the_tokenizer_takes_the_longest_token_at_each_position() -> Result<()> {
        let tokens =
            ["<s>", "a", "b", "c", "ab", "bc", "abc", " ", " a"].map(|t| t.as_bytes().to_vec());
        let tokenizer = rwkv_world_tokenizer(&tokens)?;
        let ids = |text: &str| -> Result<Vec<u32>> {
            Ok(tokenizer
                .encode_fast(text, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec())
        };
        assert_eq!(ids("abcab")?, [6, 4]);
        assert_eq!(ids("abc a")?, [6, 8]);
        assert_eq!(ids("cbc")?, [3, 5]);
        let decoded = tokenizer
            .decode(&[6, 8], false)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(decoded, "abc a");
        Ok(())
    }

    // the greedy match written out, as llama.cpp's trie runs it
    fn greedy(tokens: &[Vec<u8>], text: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < text.len() {
            let (id, len) = (1..tokens.len())
                .filter(|&id| text[at..].starts_with(&tokens[id]))
                .map(|id| (id, tokens[id].len()))
                .max_by_key(|&(id, len)| (len, std::cmp::Reverse(id)))
                .expect("every byte is a token");
            out.push(u32::try_from(id).unwrap());
            at += len;
        }
        out
    }

    // random tokens over a few bytes (spaces, newlines and a non-UTF-8 byte among them), on random text
    #[test]
    fn the_tokenizer_matches_the_greedy_match_on_random_text() -> Result<()> {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        const ALPHABET: &[u8] = b"ab \n\xff";
        const TOKENS: usize = 200;
        const MAX_LEN: usize = 7;
        const TEXT: usize = 3000;
        let rng = &mut StdRng::seed_from_u64(5);
        let mut tokens = vec![RWKV_BOS.as_bytes().to_vec()];
        tokens.extend(ALPHABET.iter().map(|&b| vec![b]));
        while tokens.len() < TOKENS {
            let len = rng.random_range(2..=MAX_LEN);
            let token = (0..len)
                .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())])
                .collect::<Vec<_>>();
            if !tokens.contains(&token) {
                tokens.push(token);
            }
        }
        // text of ASCII alone, so it is a string; the tokens still hold the 0xff byte
        let text = (0..TEXT)
            .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len() - 1)])
            .collect::<Vec<_>>();
        let tokenizer = rwkv_world_tokenizer(&tokens)?;
        let ids = tokenizer
            .encode_fast(std::str::from_utf8(&text)?, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        assert_eq!(ids, greedy(&tokens, &text));
        Ok(())
    }
}
