//! Text cut into pieces that each fit Kokoro's context, measured in phonemes rather than characters. A port of
//! vernacula's `KokoroChunker` and `ParagraphChunker`: paragraphs, then sentences, clauses, words, characters.

use inference_tensor::Result;

// Kokoro's BERT holds 512 tokens with 2 pads; pack below that, then verify against the hard limit
const PACK_BUDGET: usize = 460;
const HARD_LIMIT: usize = 508;
// paragraphs past this are split on sentences before any phoneme counting
const MAX_CHARS_PER_CHUNK: usize = 600;
const MIN_CHARS_FOR_CHUNKING: usize = 200;
const SENTENCE_ENDS: &str = ".!?";
// CJK terminators end a sentence with no following space
const CJK_SENTENCE_ENDS: &str = "\u{3002}\u{ff01}\u{ff1f}";
const WORD_SEPARATOR: &str = " ";
const CLAUSE_ENDS: &str = ",;:\u{2014}";
const CJK_CLAUSE_ENDS: &str = "\u{3001}\u{ff0c}\u{ff1b}";

#[derive(Clone, Copy)]
enum Finer {
    Clauses,
    Words,
    Nothing,
}

/// Synthesis pieces of `text` whose phoneme counts (`count`) stay inside Kokoro's context; every split is at
/// whitespace except within a script that has none, so the words are unchanged.
pub fn chunk_for_synthesis(
    text: &str,
    count: &mut dyn FnMut(&str) -> Result<usize>,
) -> Result<Vec<String>> {
    let mut chunker = Chunker {
        count,
        out: Vec::new(),
    };
    for chunk in paragraph_chunks(text) {
        chunker.split_to_budget(&chunk)?;
    }
    Ok(chunker.out)
}

struct Chunker<'a> {
    count: &'a mut dyn FnMut(&str) -> Result<usize>,
    out: Vec<String>,
}

impl Chunker<'_> {
    fn split_to_budget(&mut self, chunk: &str) -> Result<()> {
        if (self.count)(chunk)? <= HARD_LIMIT {
            self.out.push(chunk.to_string());
            return Ok(());
        }
        self.pack(
            split_after(chunk, SENTENCE_ENDS, CJK_SENTENCE_ENDS),
            Finer::Clauses,
            WORD_SEPARATOR,
        )
    }

    fn split_finer(&mut self, segment: &str, finer: Finer) -> Result<()> {
        match finer {
            Finer::Clauses => self.pack(
                split_after(segment, CLAUSE_ENDS, CJK_CLAUSE_ENDS),
                Finer::Words,
                WORD_SEPARATOR,
            ),
            Finer::Words => {
                let words = segment
                    .split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                // a script without spaces is one word; its characters are the only cut left, rejoined as they were
                if words.len() <= 1 && segment.chars().count() > 1 {
                    self.pack(
                        segment.chars().map(String::from).collect(),
                        Finer::Nothing,
                        "",
                    )
                } else {
                    self.pack(words, Finer::Nothing, WORD_SEPARATOR)
                }
            }
            Finer::Nothing => unreachable!("only packing at a finer level splits further"),
        }
    }

    // greedy packing to PACK_BUDGET, a separator costing one; a segment over budget goes to the next finer split
    fn pack(&mut self, segments: Vec<String>, finer: Finer, sep: &str) -> Result<()> {
        let (mut buf, mut buf_tokens) = (String::new(), 0);
        for raw in segments {
            let s = raw.trim();
            if s.is_empty() {
                continue;
            }
            let tokens = (self.count)(s)?;
            if tokens > PACK_BUDGET && !matches!(finer, Finer::Nothing) {
                self.flush(&mut buf, &mut buf_tokens)?;
                self.split_finer(s, finer)?;
                continue;
            }
            let mut cost = tokens + usize::from(!buf.is_empty() && !sep.is_empty());
            if buf_tokens > 0 && buf_tokens + cost > PACK_BUDGET {
                self.flush(&mut buf, &mut buf_tokens)?;
                cost = tokens;
            }
            if !buf.is_empty() {
                buf.push_str(sep);
            }
            buf.push_str(s);
            buf_tokens += cost;
        }
        self.flush(&mut buf, &mut buf_tokens)
    }

    fn flush(&mut self, buf: &mut String, buf_tokens: &mut usize) -> Result<()> {
        if !buf.is_empty() {
            self.emit_verified(&std::mem::take(buf))?;
            *buf_tokens = 0;
        }
        Ok(())
    }

    // the packing estimate can undercount; a piece really over the limit is halved on words, else characters
    fn emit_verified(&mut self, piece: &str) -> Result<()> {
        if (self.count)(piece)? <= HARD_LIMIT {
            self.out.push(piece.to_string());
            return Ok(());
        }
        let words = piece.split_whitespace().collect::<Vec<_>>();
        if words.len() <= 1 {
            let chars = piece.chars().collect::<Vec<_>>();
            if chars.len() <= 1 {
                self.out.push(piece.to_string());
                return Ok(());
            }
            let half = chars.len() / 2;
            self.emit_verified(&chars[..half].iter().collect::<String>())?;
            return self.emit_verified(&chars[half..].iter().collect::<String>());
        }
        let mid = words.len() / 2;
        self.emit_verified(&words[..mid].join(" "))?;
        self.emit_verified(&words[mid..].join(" "))
    }
}

/// Splits after a mark followed by whitespace (consumed), or right after a CJK mark.
fn split_after(text: &str, marks: &str, cjk_marks: &str) -> Vec<String> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if cjk_marks.contains(chars[i]) {
            pieces.push(chars[start..=i].iter().collect());
            start = i + 1;
        } else if marks.contains(chars[i]) && chars.get(i + 1).is_some_and(|c| c.is_whitespace()) {
            pieces.push(chars[start..=i].iter().collect());
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            start = j;
            i = j;
            continue;
        }
        i += 1;
    }
    if start < chars.len() || pieces.is_empty() {
        pieces.push(chars[start..].iter().collect());
    }
    pieces
}

// Paragraph breaks are a newline, optional spaces, and one or more newlines.
fn paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut blank_run = false;
    for line in text.split('\n') {
        let line = line.trim_end_matches('\r');
        if line.trim_matches([' ', '\t', '\r']).is_empty() {
            blank_run = true;
            continue;
        }
        if blank_run && !current.is_empty() {
            out.push(std::mem::take(&mut current).join("\n"));
        }
        blank_run = false;
        current.push(line);
    }
    if !current.is_empty() {
        out.push(current.join("\n"));
    }
    out
}

fn paragraph_chunks(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.chars().count() < MIN_CHARS_FOR_CHUNKING {
        return vec![trimmed.to_string()];
    }
    let mut chunks = Vec::new();
    for p in paragraphs(trimmed)
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
    {
        if p.chars().count() <= MAX_CHARS_PER_CHUNK {
            chunks.push(p.to_string());
            continue;
        }
        let mut buf = String::new();
        for sentence in split_after(p, SENTENCE_ENDS, "") {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let projected =
                buf.chars().count() + usize::from(!buf.is_empty()) + sentence.chars().count();
            if projected > MAX_CHARS_PER_CHUNK && !buf.is_empty() {
                chunks.push(std::mem::take(&mut buf));
            }
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(sentence);
        }
        if !buf.is_empty() {
            chunks.push(buf);
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    // a stand-in phoneme count: one token per character, as a phonemized string roughly is
    fn chars(s: &str) -> Result<usize> {
        Ok(s.chars().count())
    }

    #[test]
    fn short_text_is_one_piece() -> Result<()> {
        assert_eq!(
            chunk_for_synthesis("Hello there.", &mut chars)?,
            ["Hello there."]
        );
        assert!(chunk_for_synthesis("   ", &mut chars)?.is_empty());
        Ok(())
    }

    #[test]
    fn long_text_splits_on_sentences_then_clauses_and_stays_in_budget() -> Result<()> {
        let sentence =
            "This sentence is a fairly ordinary one, with a clause or two, and it ends here. ";
        let text = sentence.repeat(40);
        let pieces = chunk_for_synthesis(&text, &mut chars)?;
        assert!(pieces.len() > 1);
        assert!(pieces.iter().all(|p| p.chars().count() <= HARD_LIMIT));
        // whitespace-only cuts keep every word in order
        assert_eq!(
            pieces.join(" ").split_whitespace().collect::<Vec<_>>(),
            text.split_whitespace().collect::<Vec<_>>()
        );
        assert!(pieces.iter().all(|p| p.ends_with('.')));
        Ok(())
    }

    #[test]
    fn an_over_long_clause_falls_to_words() -> Result<()> {
        let text = "word ".repeat(300);
        let pieces = chunk_for_synthesis(&text, &mut chars)?;
        assert!(pieces.iter().all(|p| p.chars().count() <= HARD_LIMIT));
        assert_eq!(pieces.join(" ").split_whitespace().count(), 300);
        Ok(())
    }

    #[test]
    fn an_over_long_word_is_cut_without_spaces() -> Result<()> {
        let text = "7".repeat(1200);
        let pieces = chunk_for_synthesis(&text, &mut chars)?;
        assert!(pieces.len() > 1);
        assert!(pieces.iter().all(|p| p.chars().count() <= HARD_LIMIT));
        assert_eq!(pieces.concat(), text);
        Ok(())
    }

    #[test]
    fn splits_follow_marks_and_cjk_terminators() {
        assert_eq!(
            split_after("A b. C d! E", SENTENCE_ENDS, CJK_SENTENCE_ENDS),
            ["A b.", "C d!", "E"]
        );
        assert_eq!(split_after("v1.2 stays", SENTENCE_ENDS, ""), ["v1.2 stays"]);
        assert_eq!(
            split_after("一。二！三", SENTENCE_ENDS, CJK_SENTENCE_ENDS),
            ["一。", "二！", "三"]
        );
        assert_eq!(
            paragraphs("one\n\n  \ntwo\r\n\r\nthree"),
            ["one", "two", "three"]
        );
    }
}
