import { marked, type Token, type Tokens } from "marked";

export type BlockKind = "heading" | "paragraph" | "item" | "quote" | "table";

/** One readable block of a markdown document: what it shows and what is spoken for it. */
export interface TextBlock {
  kind: BlockKind;
  /** A heading's depth; 0 otherwise. */
  level: number;
  /** The block's text with its markup removed. */
  text: string;
  /** What the voice reads: the text, with a heading or table row closed by a full stop so it ends as a sentence. */
  speech: string;
}

// a heading or table row read without a pause runs into what follows
const SENTENCE_END = /[.!?:;]$/;

// the text of inline tokens, markup dropped; images say nothing
function inlineText(tokens: Token[] | undefined, fallback = ""): string {
  if (!tokens) return fallback;
  return tokens
    .map((token) => {
      switch (token.type) {
        case "image":
        case "html":
          return "";
        case "br":
          return " ";
        case "escape":
        case "codespan":
          return (token as Tokens.Codespan).text;
        default:
          return "tokens" in token && token.tokens ? inlineText(token.tokens) : "text" in token ? String(token.text) : "";
      }
    })
    .join("");
}

const tidy = (text: string) => text.replace(/\s+/g, " ").trim();
const closed = (text: string) => (SENTENCE_END.test(text) ? text : `${text}.`);

/** The readable blocks of `markdown` in order; code, rules and raw HTML aren't read. */
export function markdownBlocks(markdown: string): TextBlock[] {
  const blocks: TextBlock[] = [];
  const push = (kind: BlockKind, level: number, text: string, speech = text) => {
    const shown = tidy(text);
    if (shown) blocks.push({ kind, level, text: shown, speech: tidy(speech) });
  };
  const walk = (tokens: Token[], within?: BlockKind) => {
    for (const token of tokens) {
      switch (token.type) {
        case "heading": {
          const text = tidy(inlineText((token as Tokens.Heading).tokens, token.text));
          push("heading", (token as Tokens.Heading).depth, text, closed(text));
          break;
        }
        case "paragraph":
        case "text":
          push(within ?? "paragraph", 0, inlineText((token as Tokens.Paragraph).tokens, token.text));
          break;
        case "list":
          for (const item of (token as Tokens.List).items) walk(item.tokens, "item");
          break;
        case "blockquote":
          walk((token as Tokens.Blockquote).tokens, "quote");
          break;
        case "table": {
          const table = token as Tokens.Table;
          const rows = [table.header, ...table.rows].map((row) =>
            closed(tidy(row.map((cell) => inlineText(cell.tokens, cell.text)).join(", "))),
          );
          push("table", 0, rows.join(" "));
          break;
        }
        default:
          break;
      }
    }
  };
  walk(marked.lexer(markdown));
  return blocks;
}
