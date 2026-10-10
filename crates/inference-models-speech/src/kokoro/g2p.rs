//! Text to Kokoro phonemes: vernacula-phonemizer's canonical IPA rendered into the alphabet Kokoro was trained on.
//! A port of vernacula's `KokoroFormat` and `KokoroPhonemizer`, whose rules and measurements live in its
//! docs/investigations/kokoro_*_investigation.md; keep the two in step.

use inference_tensor::{Error, Result, bail};
use unicode_normalization::UnicodeNormalization;
pub use vernacula_phonemizer::core::data_source::{
    DataError, DataSource, resolve_data_root, set_data_source,
};
use vernacula_phonemizer::core::trace::Trace;

const DATA_ENV: &str = "VERNACULA_DATA_DIR";
/// The Hugging Face repo holding the phonemizer's `data/` tree, laid out so a data key is a repo path.
pub const DATA_REPO: &str = "christopherthompson81/vernacula-phonemizer-data";
/// The data's tag, which is the phonemizer commit this crate is pinned to; the two must move together.
pub const DATA_REVISION: &str = "03fa865d";
// Kokoro's voice prefixes (its lang_code) and the phonemizer's codes for the same languages
const VOICE_LANGUAGES: [(char, &str); 9] = [
    ('a', "en"),
    ('b', "en-GB"),
    ('e', "es"),
    ('f', "fr"),
    ('h', "hi"),
    ('i', "it"),
    ('p', "pt-BR"),
    ('j', "ja"),
    ('z', "cmn"),
];
const ENGLISH: [&str; 3] = ["en", "en-GB", "en-US"];

// English: misaki's espeak post-processing, keyed on what the phonemizer emits; longer keys before their prefixes
const COMMON: &[(&str, &str)] = &[
    ("\u{361}", ""),
    ("ʰ", ""),
    ("ʲ", ""),
    ("t\u{32c}", "T"),
    ("d\u{32c}", "d"),
    ("ɫ", "l"),
    ("oᶷ", "O"),
    ("eᶦ", "A"),
    ("aᶦ", "I"),
    ("aᶷ", "W"),
    ("ɔᶦ", "Y"),
    ("ᶦ", "ɪ"),
    ("ᶷ", "ʊ"),
    ("dʒ", "ʤ"),
    ("tʃ", "ʧ"),
    ("ɝ", "ɜɹ"),
    ("ɚ", "əɹ"),
    ("ɐ", "ə"),
    ("r", "ɹ"),
    ("x", "k"),
    ("ç", "k"),
    ("ɬ", "l"),
    ("\u{303}", ""),
    ("ʔ", "t"),
    ("ɾ", "T"),
];
// the alphabet's own conventions, which hold in every language; no allophone collapses
const ALPHABET: &[(&str, &str)] = &[
    ("\u{361}", ""),
    ("oᶷ", "O"),
    ("eᶦ", "A"),
    ("aᶦ", "I"),
    ("aᶷ", "W"),
    ("ɔᶦ", "Y"),
    ("ᶦ", "ɪ"),
    ("ᶷ", "ʊ"),
    ("dʒ", "ʤ"),
    ("tʃ", "ʧ"),
    ("t\u{32c}", "d"),
    ("d\u{32c}", "d"),
    ("ɫ", "l"),
    ("ɝ", "ɜɹ"),
];
const MANDARIN: &[(&str, &str)] = &[
    ("ʈʂ", "ꭧ"),
    ("ts", "ʦ"),
    ("tɕ", "ʨ"),
    ("ʐ", "ɻ"),
    ("ɹ\u{329}", "ɨ"),
    ("ɹ", "ɨ"),
    ("\u{329}", ""),
    ("ᵘ", "u"),
    ("ⁱ", "i"),
    ("ɑ", "a"),
    ("æ", "ɛ"),
];
const JAPANESE: &[(&str, &str)] = &[
    ("ts", "ʦ"),
    ("tɕ", "ʨ"),
    ("dʑ", "ʥ"),
    ("ʑ", "ʥ"),
    ("\u{e4}", "a"),
    ("\u{31e}", ""),
    ("ɴ", "n"),
    ("ꜜ", ""),
];
const HINDI: &[(&str, &str)] = &[("ɦ", "h"), ("ʱ", "ʰ"), ("\u{32a}", "")];
const SYLLABIC: char = '\u{329}';
const EXTRA_SHORT_SCHWA: &str = "ə\u{306}";
// a flap before a word-final unstressed vowel is the tap, whose duration Kokoro predicts in proportion
const FLAP_VOWELS: &str = "əɐaeiouɑɔɛɪʊʌæɜAIOWYᵻ";
const WORD_END: &str = " ,.;:!?…—";
// clause marks Kokoro reads as pauses; the phonemizer spaces them out, Kokoro's data attaches them
const DETACHED_PUNCTUATION: &str = ",.;:!?…—";
const MANDARIN_NUCLEI: &str = "aeiouyɛɤəɨʊɔɚ";
// languages without spaces between words, whose trace segments them
const TRACED_WORD_LANGUAGES: [&str; 2] = ["ja", "cmn"];
// CJK unified ideographs and extension A, as UTF-16 units
const HAN: [std::ops::RangeInclusive<u16>; 2] = [0x4e00..=0x9fff, 0x3400..=0x4dbf];
const TONE_LETTERS: std::ops::RangeInclusive<char> = '\u{2e5}'..='\u{2e9}';
const KOKORO_VOWELS: &str = "əɐaeiouɑɔɛɪʊʌæɜAIOWYᵻᵊ";
const WORD_TRIM: &[char] = &['.', ',', ';', ':', '!', '?', '"', '\'', '(', ')', '—', '-'];

/// The phonemizer's language for a Kokoro voice, from its first letter as Kokoro's own pipeline chooses.
pub fn voice_language(voice: &str) -> Result<&'static str> {
    let prefix = voice.trim().chars().next();
    match VOICE_LANGUAGES.iter().find(|(c, _)| Some(*c) == prefix) {
        Some((_, code)) => Ok(code),
        None => bail!("no language for voice `{voice}`; pass `phonemes`"),
    }
}

/// Whether the phonemizer reads `lang`, without loading anything.
pub fn supported(lang: &str) -> Result<()> {
    if !vernacula_phonemizer::LANGUAGES.contains(&lang) {
        bail!(
            "the phonemizer reads {} so far, not `{lang}`; pass `phonemes`",
            vernacula_phonemizer::LANGUAGES.join(", ")
        )
    }
    Ok(())
}

/// Whether the phonemizer reads `lang` and can load its data.
pub fn readable(lang: &str) -> Result<()> {
    supported(lang)?;
    // the engine is built once per process, so this pays the data load on the first request only
    vernacula_phonemizer::phonemize("", lang)
        .map(|_| ())
        .map_err(phonemizer_error)
}

fn phonemizer_error(e: vernacula_phonemizer::PhonemizeError) -> Error {
    use vernacula_phonemizer::PhonemizeError::*;
    Error::Msg(match e {
        UnknownLanguage(code) => format!("the phonemizer has no `{code}`; pass `phonemes`"),
        Data(why) => format!(
            "Kokoro's text input needs vernacula-phonemizer's data ({DATA_REPO}, or {DATA_ENV}): {why}"
        ),
        Neural(why) => format!("the phonemizer's neural reader failed: {why}"),
        Input(why) => format!("the phonemizer cannot read this input: {why}"),
    })
}

fn replace_all(mut s: String, pairs: &[(&str, &str)]) -> String {
    for (from, to) in pairs {
        s = s.replace(from, to);
    }
    s
}

/// Canonical IPA in Kokoro's alphabet for `lang`; `in_vocab` decides what non-English text decomposes into.
pub fn render(ipa: &str, lang: &str, in_vocab: impl Fn(char) -> bool) -> String {
    if ENGLISH.contains(&lang) {
        render_english(ipa, lang == "en-GB")
    } else {
        render_other(ipa, lang, in_vocab)
    }
}

fn render_english(ipa: &str, british: bool) -> String {
    let mut ps = ipa.trim().to_string();
    if british {
        ps = ps.replace("əᶷ", "Q");
    }
    ps = replace_all(ps, COMMON);
    ps = if british {
        ps.replace("ɛə", "ɛː")
    } else {
        ps.replace('ː', "")
    };
    ps = ps.replace('o', "ɔ").replace(EXTRA_SHORT_SCHWA, "ᵊ");
    ps = syllabic_to_schwa(&ps).replace(SYLLABIC, "");
    ps = attach_punctuation(&ps);
    if !british {
        ps = word_final_flap(&ps);
    }
    ps
}

fn render_other(ipa: &str, lang: &str, in_vocab: impl Fn(char) -> bool) -> String {
    let mut ps = replace_all(ipa.trim().to_string(), ALPHABET);
    ps = match lang {
        "cmn" => place_mandarin_tone(&normalize_mandarin(&replace_all(ps, MANDARIN))),
        "ja" => replace_all(ps, JAPANESE),
        "hi" => replace_all(ps, HINDI),
        _ => ps,
    };
    attach_punctuation(&decompose_unknown(&ps, in_vocab))
}

// a syllabic consonant (U+0329 after it) takes misaki's small schwa (U+1D4A) before it instead
fn syllabic_to_schwa(s: &str) -> String {
    let chars = s.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_whitespace() && chars.get(i + 1) == Some(&SYLLABIC) {
            out.push('ᵊ');
            out.push(chars[i]);
            i += 2;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

// spaces before a word-final run of detached punctuation drop, so it attaches to the word before
fn attach_punctuation(s: &str) -> String {
    let chars = s.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ' ' {
            let spaces_end = (i..chars.len())
                .find(|&j| chars[j] != ' ')
                .unwrap_or(chars.len());
            let punct_end = (spaces_end..chars.len())
                .find(|&j| !DETACHED_PUNCTUATION.contains(chars[j]))
                .unwrap_or(chars.len());
            if punct_end > spaces_end && chars.get(punct_end).is_none_or(|&c| c == ' ') {
                out.extend(&chars[spaces_end..punct_end]);
                i = punct_end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

// T before a word-final vowel is a flap
fn word_final_flap(s: &str) -> String {
    let chars = s.chars().collect::<Vec<_>>();
    (0..chars.len())
        .map(|i| {
            let final_vowel = chars.get(i + 1).is_some_and(|&v| FLAP_VOWELS.contains(v))
                && chars.get(i + 2).is_none_or(|&e| WORD_END.contains(e));
            if chars[i] == 'T' && final_vowel {
                'ɾ'
            } else {
                chars[i]
            }
        })
        .collect()
}

fn is_nucleus(c: Option<char>) -> bool {
    c.is_some_and(|c| MANDARIN_NUCLEI.contains(c))
}

// A Mandarin syllable in the shape the engine's own pinyin table writes, before its tone is placed.
fn normalize_mandarin(ps: &str) -> String {
    ps.split(' ')
        .map(|syllable| {
            let mut syl = syllable.chars().collect::<Vec<_>>();
            // a zero-initial glide that only duplicates its vowel is dropped: yi is i, wu is u, yu is y
            if syl.len() > 1
                && ((syl[0] == 'j' && matches!(syl[1], 'i' | 'y'))
                    || (syl[0] == 'w' && syl[1] == 'u'))
            {
                syl.remove(0);
            }
            let syl = syl
                .iter()
                .collect::<String>()
                .replace("ər", "ɚ")
                .chars()
                .collect::<Vec<_>>();
            let mut out: Vec<char> = Vec::with_capacity(syl.len() + 2);
            for (k, &c) in syl.iter().enumerate() {
                let next = syl.get(k + 1).copied();
                let prev = out.last().copied();
                // prenuclear high vowels are glides: ʨia -> ʨja, tuan -> twan, ɕye -> ɕɥe
                if is_nucleus(next) {
                    match c {
                        'i' => {
                            out.push('j');
                            continue;
                        }
                        'u' => {
                            out.push('w');
                            continue;
                        }
                        'y' => {
                            out.push('ɥ');
                            continue;
                        }
                        _ => {}
                    }
                }
                // the apical vowel after a retroflex or sibilant, looking past chi/ci's aspiration
                let onset = if prev == Some('ʰ') && out.len() > 1 {
                    out.get(out.len() - 2).copied()
                } else {
                    prev
                };
                if c == 'ɻ' && matches!(onset, Some('ʂ' | 'ꭧ' | 'ɻ' | 's')) {
                    out.push('ɨ');
                    continue;
                }
                // after a palatal glide the mid vowel's height follows the coda
                if c == 'ɛ' && matches!(prev, Some('ɥ' | 'j')) {
                    out.push(if matches!(next, Some('n' | 'ŋ')) {
                        'ɛ'
                    } else {
                        'e'
                    });
                    continue;
                }
                out.push(c);
            }
            out.into_iter().collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// Tone letters at a syllable's end become Kokoro's arrow, moved to just after the nucleus (zhong1 = ꭧʊ→ŋ).
fn place_mandarin_tone(ps: &str) -> String {
    ps.split(' ')
        .map(|syllable| {
            let chars = syllable.chars().collect::<Vec<_>>();
            let Some(start) = chars.iter().position(|c| TONE_LETTERS.contains(c)) else {
                return syllable.to_string();
            };
            let end = (start..chars.len())
                .find(|&i| !TONE_LETTERS.contains(&chars[i]))
                .unwrap_or(chars.len());
            let arrow = tone_arrow(&chars[start..end]);
            let bare = chars
                .iter()
                .filter(|&&c| !TONE_LETTERS.contains(&c))
                .copied()
                .collect::<Vec<_>>();
            if arrow.is_empty() {
                return bare.into_iter().collect();
            }
            match bare.iter().rposition(|&c| MANDARIN_NUCLEI.contains(c)) {
                Some(n) => bare[..=n]
                    .iter()
                    .chain(arrow.chars().collect::<Vec<_>>().iter())
                    .chain(&bare[n + 1..])
                    .collect(),
                None => bare.into_iter().collect::<String>() + arrow,
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn tone_arrow(letters: &[char]) -> &'static str {
    match letters {
        ['\u{2e5}', '\u{2e5}'] => "→",
        ['\u{2e7}', '\u{2e5}'] => "↗",
        ['\u{2e8}', '\u{2e9}', '\u{2e6}'] => "↓",
        ['\u{2e5}', '\u{2e9}'] => "↘",
        // a contour variant still lands on a real tone, judged by its shape
        _ if letters.len() < 2 => "",
        [first, .., last] if last > first => "↗",
        [first, .., last] if last < first => "↘",
        _ => "→",
    }
}

// A codepoint the vocabulary lacks becomes its canonical decomposition when every piece is in it (Portuguese õ).
fn decompose_unknown(ps: &str, in_vocab: impl Fn(char) -> bool) -> String {
    if ps.chars().all(&in_vocab) {
        return ps.to_string();
    }
    let mut out = String::with_capacity(ps.len() + 8);
    for c in ps.chars() {
        if in_vocab(c) {
            out.push(c);
            continue;
        }
        let decomposed = c.to_string().nfd().collect::<String>();
        if decomposed.chars().count() > 1 && decomposed.chars().all(&in_vocab) {
            out.push_str(&decomposed);
        } else {
            out.push(c);
        }
    }
    out
}

/// Space-delimited groups that contain a letter, i.e. spoken words rather than stand-alone punctuation.
fn count_word_groups(ipa: &str) -> usize {
    ipa.split_whitespace()
        .filter(|g| g.chars().any(char::is_alphabetic))
        .count()
}

/// A word of the input as UTF-16 offsets, the units the phonemizer's trace spans count in.
#[derive(Debug, Clone, Copy)]
struct WordSpan {
    start: usize,
    end: usize,
}

// languages that don't space take their words from the trace; whitespace stays the fallback when it can't segment
fn source_words(text: &[u16], lang: &str, trace: &Trace, ipa: &[u16]) -> Vec<WordSpan> {
    if !TRACED_WORD_LANGUAGES.contains(&lang) {
        return whitespace_words(text);
    }
    let traced = traced_words(text, trace, ipa);
    if traced.is_empty() {
        whitespace_words(text)
    } else {
        traced
    }
}

// ja: a token per phrase; cmn: a Han run's syllables map onto its hanzi only when counts agree (digits break it)
fn traced_words(text: &[u16], trace: &Trace, ipa: &[u16]) -> Vec<WordSpan> {
    if !trace.traced {
        return Vec::new();
    }
    let mut words = Vec::new();
    for tok in &trace.tokens {
        let Some((from, to)) = tok.input_span else {
            continue;
        };
        let groups = tok.ipa_span.as_ref().map_or(0, |span| {
            count_word_groups(&String::from_utf16_lossy(&ipa[span.0..span.1]))
        });
        // a token that says nothing is punctuation (Japanese trailing full stop), not a word
        if to <= from || groups == 0 {
            continue;
        }
        let body = &text[from..to];
        let han = body.iter().filter(|&&u| is_han(u)).count();
        if groups > 1 && groups == han && body.iter().all(|&u| is_han(u) || is_space(u)) {
            for k in (from..to).filter(|&k| is_han(text[k])) {
                append_word(
                    &mut words,
                    WordSpan {
                        start: k,
                        end: k + 1,
                    },
                );
            }
        } else {
            append_word(
                &mut words,
                WordSpan {
                    start: from,
                    end: to,
                },
            );
        }
    }
    words
}

// tokens claiming the same characters (a rewrite stamps its whole match on each token it makes) are one word
fn append_word(words: &mut Vec<WordSpan>, span: WordSpan) {
    match words.last_mut() {
        Some(last) if span.start < last.end => {
            last.start = last.start.min(span.start);
            last.end = last.end.max(span.end);
        }
        _ => words.push(span),
    }
}

fn is_han(u: u16) -> bool {
    HAN.iter().any(|r| r.contains(&u))
}

fn is_space(u: u16) -> bool {
    char::from_u32(u32::from(u)).is_some_and(char::is_whitespace)
}

fn whitespace_words(text: &[u16]) -> Vec<WordSpan> {
    let mut words = Vec::new();
    let mut start = None;
    for (i, &u) in text.iter().enumerate() {
        match (is_space(u), start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                words.push(WordSpan { start: s, end: i });
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        words.push(WordSpan {
            start: s,
            end: text.len(),
        });
    }
    words
}

/// One source-word index per spoken IPA group, from the trace's spans; None when a token has no input span.
fn group_source_words(
    trace: &Trace,
    ipa: &[u16],
    text: &[u16],
    words: &[WordSpan],
) -> Option<Vec<usize>> {
    if !trace.traced || words.is_empty() {
        return None;
    }
    // a character between two words takes the index of the word that follows it
    let mut word_at = vec![0usize; text.len() + 1];
    let mut next = 0;
    for (i, slot) in word_at.iter_mut().enumerate().take(text.len()) {
        if next < words.len() && i >= words[next].end {
            next += 1;
        }
        *slot = if next < words.len() && i >= words[next].start {
            next
        } else {
            next.min(words.len() - 1)
        };
    }
    word_at[text.len()] = words.len() - 1;

    let mut map = Vec::new();
    let mut last_span: Option<(usize, usize)> = None;
    let mut last_word = 0usize;
    for tok in &trace.tokens {
        let input = tok.input_span?;
        let groups = if let Some(span) = tok.ipa_span.as_ref() {
            count_word_groups(&String::from_utf16_lossy(&ipa[span.0..span.1]))
        } else if !tok.emitted.is_empty() {
            tok.emitted.len()
        } else {
            // a token that says nothing (Japanese 。) contributes nothing
            continue;
        };
        if groups == 0 {
            continue;
        }
        let span = input;
        let last_in_span = word_at[input.1.saturating_sub(1).min(text.len())];
        let first = word_at[input.0];
        // one token covering several word units carries one group per unit (a Mandarin sentence)
        if last_span != Some(span) && groups > 1 && last_in_span + 1 == first + groups {
            map.extend(first..first + groups);
            last_span = Some(span);
            last_word = last_in_span;
            continue;
        }
        // tokens from one normalizer rewrite take successive words of a span that covers several
        let word = if last_span == Some(span) {
            (last_word + 1).min(last_in_span)
        } else {
            first
        };
        map.extend(std::iter::repeat_n(word, groups));
        last_span = Some(span);
        last_word = word;
    }
    Some(map)
}

// misaki writes a de-/re- prefix's reduced vowel as schwa (1,057 times, against 9 for barred i); needs the source word
fn reduce_english_prefix_vowel(
    rendered: &str,
    text: &[u16],
    map: &[usize],
    words: &[WordSpan],
) -> String {
    if map.is_empty() || !rendered.contains('ᵻ') || count_word_groups(rendered) != map.len() {
        return rendered.to_string();
    }
    let mut out = String::with_capacity(rendered.len());
    let mut group = 0;
    let mut token = String::new();
    let flush = |token: &mut String, out: &mut String, group: &mut usize| {
        if token.is_empty() {
            return;
        }
        if !token.chars().any(char::is_alphabetic) {
            out.push_str(token);
        } else {
            let word = map.get(*group).and_then(|&w| words.get(w));
            *group += 1;
            match word {
                Some(w) => out.push_str(&with_prefix_schwa(
                    token,
                    &String::from_utf16_lossy(&text[w.start..w.end]),
                )),
                None => out.push_str(token),
            }
        }
        token.clear();
    };
    for c in rendered.chars() {
        if c.is_whitespace() {
            flush(&mut token, &mut out, &mut group);
            out.push(c);
        } else {
            token.push(c);
        }
    }
    flush(&mut token, &mut out, &mut group);
    out
}

fn with_prefix_schwa(token: &str, word: &str) -> String {
    let bare = word.trim_matches(WORD_TRIM).to_lowercase();
    if bare.starts_with("ded") || !(bare.starts_with("de") || bare.starts_with("re")) {
        return token.to_string();
    }
    let chars = token.chars().collect::<Vec<_>>();
    match chars.iter().position(|&c| KOKORO_VOWELS.contains(c)) {
        Some(v) if chars[v] == 'ᵻ' => chars[..v]
            .iter()
            .chain(['ə'].iter())
            .chain(&chars[v + 1..])
            .collect(),
        _ => token.to_string(),
    }
}

/// `text` read as `lang` and rendered into Kokoro's alphabet.
pub fn phonemes(text: &str, lang: &str, in_vocab: impl Fn(char) -> bool) -> Result<String> {
    Ok(phonemize(text, lang, in_vocab)?.0)
}

/// The rendered phonemes and, when the trace accounts for every group, each group's source-word index.
fn phonemize(
    text: &str,
    lang: &str,
    in_vocab: impl Fn(char) -> bool,
) -> Result<(String, Option<Vec<usize>>)> {
    let traced = vernacula_phonemizer::phonemize_trace(text, lang).map_err(phonemizer_error)?;
    let text16 = text.encode_utf16().collect::<Vec<_>>();
    let traced16 = traced.ipa.encode_utf16().collect::<Vec<_>>();
    let words = source_words(&text16, lang, &traced.trace, &traced16);
    let map = group_source_words(&traced.trace, &traced16, &text16, &words);
    // the best reading routes unknown English words through the BiLSTM; the trace's reading is the fallback
    let mut ipa =
        vernacula_phonemizer::phonemize_best(text, lang).unwrap_or_else(|_| traced.ipa.clone());
    // the map was built from the traced reading, so use it only for a reading of the same shape
    if map
        .as_ref()
        .is_some_and(|m| count_word_groups(&ipa) != m.len())
    {
        ipa = traced.ipa.clone();
    }
    let rendered = render(&ipa, lang, in_vocab);
    let rendered = match (&map, ENGLISH.contains(&lang)) {
        (Some(map), true) => reduce_english_prefix_vowel(&rendered, &text16, map, &words),
        _ => rendered,
    };
    Ok((rendered, map))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Kokoro v1.0's vocabulary, as the release's config.json lists it
    const VOCAB: &str = ";:,.!?—…\"()“” \u{303}ʣʥʦʨᵝꭧAIOQSTWYᵊabcdefhijklmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ↓→↗↘ᵻ";

    fn en(ipa: &str) -> String {
        render(ipa, "en", |c| VOCAB.contains(c))
    }

    fn lang(ipa: &str, lang: &str) -> String {
        render(ipa, lang, |c| VOCAB.contains(c))
    }

    fn all_in_vocab(s: &str) -> bool {
        s.chars().all(|c| VOCAB.contains(c))
    }

    // the vectors of vernacula's KokoroFormatTests, which the renderer must keep byte for byte
    #[test]
    fn renders_american() {
        let cases = [
            (
                "həlˈoᶷ wˈɝɫd . ðɪs ɪz ə tʰˈɛst .",
                "həlˈO wˈɜɹld. ðɪs ɪz ə tˈɛst.",
            ),
            ("t͡ʃˈɝt͡ʃ d͡ʒˈʌd͡ʒ bˈʌt̬ən hˈɪd̬ən", "ʧˈɜɹʧ ʤˈʌʤ bˈʌTən hˈɪdən"),
            ("lˈeᶦzi bɹˈaᶷn aᶦ pʰˈɔᶦnt oᶷvɚ", "lˈAzi bɹˈWn I pˈYnt Ovəɹ"),
            ("fˈɑːks θˈɔːt jɚˈeᶦniʲəm", "fˈɑks θˈɔt jəɹˈAniəm"),
            ("ɹᵻmˈɛmbɚ jˈɛstɚd̬ˌeᶦz", "ɹᵻmˈɛmbəɹ jˈɛstəɹdˌAz"),
            (
                "wˈeᶦt , hiː sˈɛd . ˈɪzənt ɪt ?",
                "wˈAt, hi sˈɛd. ˈɪzənt ɪt?",
            ),
        ];
        for (ipa, want) in cases {
            assert_eq!(en(ipa), want, "{ipa}");
        }
    }

    #[test]
    fn word_final_flap_becomes_a_tap() {
        let cases = [
            ("ðə dˈeᶦt̬ə ɹᵻkwˈaᶦɚd", "ðə dˈAɾə ɹᵻkwˈIəɹd"),
            ("ðə bˈeᶦt̬ə , ðə d̬ˈeᶦt̬ə .", "ðə bˈAɾə, ðə dˈAɾə."),
            ("ðə sˈɪt̬i hæz kwˈɑːlᵻt̬i", "ðə sˈɪɾi hæz kwˈɑlᵻɾi"),
            ("ɹˈaᶦt̬ɚ ɹˈaᶦd̬ɚ lˈæt̬ɚ lˈæd̬ɚ", "ɹˈITəɹ ɹˈIdəɹ lˈæTəɹ lˈædəɹ"),
            (
                "mˈiːt̬ɪŋ bˈɛt̬ɚ ɹᵻlˈeᶦt̬ᵻd lˈɪmᵻt̬ᵻd",
                "mˈiTɪŋ bˈɛTəɹ ɹᵻlˈATᵻd lˈɪmᵻTᵻd",
            ),
        ];
        for (ipa, want) in cases {
            assert_eq!(en(ipa), want, "{ipa}");
        }
        assert!(en("ðə dˈeᶦt̬ə").contains('ɾ') && all_in_vocab(&en("ðə dˈeᶦt̬ə")));
    }

    #[test]
    fn renders_british() {
        let cases = [
            ("həlˈəᶷ wˈɜːɫd . ðˈɛə hˈɪə", "həlˈQ wˈɜːld. ðˈɛː hˈɪə"),
            ("ɡˈəᶷ hˈəᶷm nˈaᶷ !", "ɡˈQ hˈQm nˈW!"),
            ("fˈaᶦə , ʃˈɛə , kjˈʊə .", "fˈIə, ʃˈɛː, kjˈʊə."),
        ];
        for (ipa, want) in cases {
            assert_eq!(lang(ipa, "en-GB"), want, "{ipa}");
        }
    }

    #[test]
    fn every_output_codepoint_is_in_the_vocab() {
        let cases = [
            (
                "mˈɪstɚ smˈɪθ ɚˈaᶦvd æt tʰˈɛn θˈɝd̬iː ˈeᶦ ˈɛm ˈɑːn tʰˈuːzdi , mˈɑːɹt͡ʃ θˈɝd , twˈɛnti twˈɛnti fˈɔːɹ .",
                "en",
            ),
            (
                "jˈɛstədˌeᶦz wˈɛðə wˈɒz bˈɛtə ðæn tədˈeᶦz , wˈɒzənt ɪt ?",
                "en-GB",
            ),
            ("sˈɪŋɪŋ , θˈɪŋkɪŋ , lˈɛŋkθ , ðə kʰˈɪŋz ɹˈɪŋ .", "en"),
        ];
        for (ipa, l) in cases {
            let ps = lang(ipa, l);
            assert!(all_in_vocab(&ps), "{ps}");
        }
        assert_eq!(en(""), "");
    }

    #[test]
    fn a_non_english_contrast_survives() {
        let cases = [
            ("es", "pˈeɾo", "pˈeɾo"),
            ("es", "pˈero", "pˈero"),
            ("es", "xamˈon", "xamˈon"),
            ("it", "kˈorre", "kˈorre"),
            ("fr", "bɔ̃", "bɔ̃"),
            ("pt-BR", "avˈo", "avˈo"),
            ("pt-BR", "avˈɔ", "avˈɔ"),
            ("hi", "kʰaː", "kʰaː"),
            ("hi", "ɦɛ", "hɛ"),
            ("hi", "ɡʱoʃ", "ɡʰoʃ"),
            ("hi", "sˈəkt̪a", "sˈəkta"),
        ];
        for (l, ipa, want) in cases {
            assert_eq!(lang(ipa, l), want, "{l} {ipa}");
        }
        for nasal in ["õ", "ĩ", "ũ", "ẽ"] {
            let ps = lang(nasal, "pt-BR");
            assert!(all_in_vocab(&ps) && ps.contains('\u{303}'), "{nasal}: {ps}");
        }
    }

    #[test]
    fn japanese_lands_on_kokoros_kana_inventory() {
        let cases = [
            ("t\u{361}sɯᵝ", "ʦɯᵝ"),
            ("t\u{361}ɕi", "ʨi"),
            ("d\u{361}ʑi", "ʥi"),
            ("ʑi", "ʥi"),
            ("k\u{e4}", "ka"),
            ("e\u{31e}", "e"),
            ("s\u{e4}ɴ", "san"),
        ];
        for (ipa, want) in cases {
            assert_eq!(lang(ipa, "ja"), want, "{ipa}");
        }
        let pitch = lang("o\u{31e}ꜜː", "ja");
        assert!(!pitch.contains('ꜜ') && all_in_vocab(&pitch), "{pitch}");
    }

    #[test]
    fn mandarin_tone_moves_after_the_nucleus() {
        let cases = [
            ("a\u{2e5}\u{2e5}", "a→"),
            ("a\u{2e7}\u{2e5}", "a↗"),
            ("ai\u{2e8}\u{2e9}\u{2e6}", "ai↓"),
            ("ai\u{2e5}\u{2e9}", "ai↘"),
            ("a", "a"),
            ("tɑ\u{2e8}\u{2e9}\u{2e6}n", "ta↓n"),
            ("sɹ\u{329}\u{2e5}\u{2e9}", "sɨ↘"),
            ("ʂʐ\u{329}\u{2e5}\u{2e9}", "ʂɨ↘"),
            ("ʐʐ\u{329}\u{2e5}\u{2e9}", "ɻɨ↘"),
            ("t\u{361}ɕiɑ\u{2e5}\u{2e5}", "ʨja→"),
            ("tuɑn\u{2e5}\u{2e9}", "twa↘n"),
            ("ji\u{2e5}\u{2e5}", "i→"),
            ("wɑŋ\u{2e8}\u{2e9}\u{2e6}", "wa↓ŋ"),
            ("ʈʂʊŋ\u{2e5}\u{2e5}", "ꭧʊ→ŋ"),
            ("ər\u{2e5}\u{2e9}", "ɚ↘"),
        ];
        for (ipa, want) in cases {
            assert_eq!(lang(ipa, "cmn"), want, "{ipa}");
        }
    }

    #[test]
    fn a_reduced_slot_becomes_the_small_schwa() {
        let cases = [
            ("ˈeᶦbɫ̩", "ˈAbᵊl"),
            ("θˈaᶷzn̩d", "θˈWzᵊnd"),
            ("əkʰˈʌmpə\u{306}ni", "əkˈʌmpᵊni"),
            ("ˈænə\u{306}lˌaᶦz", "ˈænᵊlˌIz"),
            ("əbˈɑːmə\u{306}nəbɫ̩", "əbˈɑmᵊnəbᵊl"),
        ];
        for (ipa, want) in cases {
            assert_eq!(en(ipa), want, "{ipa}");
            assert!(all_in_vocab(&en(ipa)));
        }
        assert!(!en("\u{329} ˈeᶦbɫ̩").contains(SYLLABIC));
    }

    #[test]
    fn voices_pick_their_language() -> Result<()> {
        assert_eq!(voice_language("af_heart")?, "en");
        assert_eq!(voice_language("bm_george")?, "en-GB");
        assert_eq!(voice_language("zf_xiaobei")?, "cmn");
        assert!(voice_language("xx_none").is_err());
        Ok(())
    }

    // vernacula's KokoroPhonemizerTests, through the real phonemizer; they skip without its data
    fn read(text: &str, l: &str) -> Option<(String, Option<Vec<usize>>)> {
        if let Err(e) = readable(l) {
            eprintln!("skipped: {e}");
            return None;
        }
        Some(phonemize(text, l, |c| VOCAB.contains(c)).unwrap())
    }

    #[test]
    fn one_group_per_spoken_word_with_punctuation_attached() {
        let Some((ps, map)) = read("Hello world. This is a test, isn't it?", "en") else {
            return;
        };
        let groups = ps.split(' ').filter(|g| !g.is_empty()).collect::<Vec<_>>();
        assert_eq!(groups.len(), 8, "{ps}");
        assert!(
            groups[1].ends_with('.') && groups[5].ends_with(',') && groups[7].ends_with('?'),
            "{ps}"
        );
        assert_eq!(map, Some((0..8).collect()));
        assert!(ps.chars().all(|c| c == ' ' || VOCAB.contains(c)), "{ps}");
    }

    #[test]
    fn expanded_numbers_collapse_onto_their_written_word() {
        let Some((ps, map)) = read("I paid about it $3.14 yesterday.", "en") else {
            return;
        };
        let map = map.expect("a map");
        assert_eq!(ps.split_whitespace().count(), map.len());
        assert!(map.iter().filter(|&&w| w == 4).count() >= 2, "{map:?}");
        assert_eq!(map.last(), Some(&5));
        assert!(map.windows(2).all(|w| w[0] <= w[1]), "{map:?}");
    }

    #[test]
    fn british_uses_the_gb_reading() {
        let (Some((us, _)), Some((gb, _))) = (read("go home", "en"), read("go home", "en-GB"))
        else {
            return;
        };
        assert_eq!(us, "ɡˈO hˈOm");
        assert_eq!(gb, "ɡˈQ hˈQm");
    }

    #[test]
    fn the_prefix_vowel_is_a_schwa_except_where_it_must_not_be() {
        let cases = [
            ("determine", "dətˈɜɹmən"),
            ("describe", "dəskɹˈIb"),
            ("reduce", "ɹədˈus"),
            ("remember", "ɹəmˈɛmbəɹ"),
            ("deduce", "dᵻdˈus"),
            ("deduct", "dᵻdˈʌkt"),
            ("dejection", "dəʤˈɛkʃən"),
            ("degeneracy", "dəʤˈɛnəɹəsi"),
            ("before", "bᵻfˈɔɹ"),
            ("become", "bᵻkˈʌm"),
            ("dejected", "dəʤˈɛktᵻd"),
        ];
        for (word, want) in cases {
            let Some((ps, _)) = read(word, "en") else {
                return;
            };
            assert_eq!(ps, want, "{word}");
            assert!(ps.chars().all(|c| VOCAB.contains(c)), "{ps}");
        }
        // onsets without evidence render exactly as without the rule
        for word in [
            "precede", "precise", "predict", "preclude", "prefer", "before", "become", "begin",
        ] {
            let Some((ps, _)) = read(word, "en") else {
                return;
            };
            let ipa = vernacula_phonemizer::phonemize_best(word, "en").unwrap();
            assert_eq!(ps, render(&ipa, "en", |c| VOCAB.contains(c)), "{word}");
        }
        let Some((represent, _)) = read("represent", "en") else {
            return;
        };
        assert!(
            represent.starts_with('ɹ') && !represent.contains("ɹəp"),
            "{represent}"
        );
        let Some((dedans, _)) = read("dedans", "en") else {
            return;
        };
        assert!(dedans.starts_with("dᵻd"), "{dedans}");
    }

    #[test]
    fn a_de_or_re_prefix_reads_with_a_schwa() {
        assert_eq!(with_prefix_schwa("dᵻtˈɜɹmᵻn", "determine"), "dətˈɜɹmᵻn");
        assert_eq!(with_prefix_schwa("ɹᵻdˈus", "Reduce,"), "ɹədˈus");
        // ded- is the deduce family, and a later ᵻ is no prefix
        assert_eq!(with_prefix_schwa("dᵻdˈus", "deduce"), "dᵻdˈus");
        assert_eq!(with_prefix_schwa("ɹˌɛpɹᵻzˈɛnt", "represent"), "ɹˌɛpɹᵻzˈɛnt");
        assert_eq!(with_prefix_schwa("pɹᵻfˈɜɹ", "prefer"), "pɹᵻfˈɜɹ");
    }

    // vernacula's WordSegmentationTests: the words a language without spaces takes from its trace
    fn words(text: &str, l: &str) -> Option<Vec<String>> {
        if let Err(e) = readable(l) {
            eprintln!("skipped: {e}");
            return None;
        }
        let traced = vernacula_phonemizer::phonemize_trace(text, l).unwrap();
        let text16 = text.encode_utf16().collect::<Vec<_>>();
        let ipa16 = traced.ipa.encode_utf16().collect::<Vec<_>>();
        let spans = source_words(&text16, l, &traced.trace, &ipa16);
        assert!(spans.windows(2).all(|w| w[1].start >= w[0].end), "{text}");
        assert!(
            spans
                .iter()
                .all(|w| w.start < w.end && w.end <= text16.len()),
            "{text}"
        );
        Some(
            spans
                .iter()
                .map(|w| String::from_utf16_lossy(&text16[w.start..w.end]))
                .collect(),
        )
    }

    #[test]
    fn japanese_splits_into_phrases_where_whitespace_sees_one_word() {
        let text = "科学者たちが発表しました。";
        let Some(found) = words(text, "ja") else {
            return;
        };
        assert!(found.len() >= 2, "{found:?}");
        assert_eq!(
            whitespace_words(&text.encode_utf16().collect::<Vec<_>>()).len(),
            1
        );
        // the trailing full stop says nothing, so it is no word
        assert!(found.iter().all(|w| w != "。"), "{found:?}");
        let other = words("彼女は新しい本を読んでいます。", "ja").unwrap();
        assert!(other.len() >= 2, "{other:?}");
    }

    // PDF reads as a katakana expansion whose tokens all claim the whole sentence; they merge, never repeat
    #[test]
    fn mixed_script_japanese_offers_no_two_words_over_the_same_characters() {
        let Some(found) = words("PDFファイルを開いてください。", "ja") else {
            return;
        };
        assert!(!found.is_empty());
    }

    #[test]
    fn mandarin_walks_syllables_onto_hanzi() {
        let Some(found) = words("今天天气很好。", "cmn") else {
            return;
        };
        assert_eq!(found, ["今", "天", "天", "气", "很", "好"]);
    }

    // 11 speaks two syllables, so a walk would shift every hanzi after it; the token stays whole instead
    #[test]
    fn mandarin_with_digits_keeps_the_token_whole() {
        let text = "11点20分警察要求。";
        let Some(found) = words(text, "cmn") else {
            return;
        };
        let han = text.encode_utf16().filter(|&u| is_han(u)).count();
        assert!(!found.is_empty() && found.len() < han, "{found:?}");
    }

    #[test]
    fn japanese_maps_each_group_to_a_phrase() {
        let text = "科学者たちが発表しました。";
        let Some((ps, map)) = read(text, "ja") else {
            return;
        };
        assert!(ps.chars().all(|c| c == ' ' || VOCAB.contains(c)), "{ps}");
        let map = map.expect("the trace accounts for every group");
        assert_eq!(map.len(), count_word_groups(&ps), "{ps}");
        assert!(
            map.windows(2).all(|w| w[0] <= w[1]) && map.last() > Some(&0),
            "{map:?}"
        );
    }

    #[test]
    fn data_revision_is_the_pinned_phonemizer_commit() {
        let manifest = include_str!("../../Cargo.toml");
        assert!(
            manifest.contains(&format!("rev = \"{DATA_REVISION}\"")),
            "Cargo.toml's vernacula-phonemizer rev and DATA_REVISION differ"
        );
    }
}
