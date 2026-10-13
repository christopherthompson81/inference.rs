/** A region of a page as read: PP-DocLayoutV3's label for it, its box in page pixels and the OCR model's text. */
export interface ReadRegion {
  label: string;
  score: number;
  bbox: [number, number, number, number];
  text: string;
}

export interface ReadPage {
  name: string;
  width: number;
  height: number;
  regions: ReadRegion[];
  /** The whole page's text, for a page read without a layout. */
  text: string;
}

export type DocumentFormat = "md" | "txt" | "html" | "json";

export const DOCUMENT_FORMATS: { format: DocumentFormat; label: string; mime: string }[] = [
  { format: "md", label: "Markdown", mime: "text/markdown" },
  { format: "html", label: "HTML", mime: "text/html" },
  { format: "txt", label: "Plain text", mime: "text/plain" },
  { format: "json", label: "JSON (regions and boxes)", mime: "application/json" },
];

// page furniture the reading leaves out of the document's text; JSON keeps everything
const FURNITURE = new Set(["header", "footer", "number", "formula_number", "header_image", "footer_image"]);
const PICTURES = new Set(["image", "seal"]);
const FORMULAS = new Set(["display_formula", "inline_formula"]);
// PaddleOCR-VL writes tables in OTSL: a cell per tag, rows ended by <nl>
const OTSL_CELL = /<(fcel|ecel|lcel|ucel|xcel)>([^<]*)/g;
const OTSL_ROW = "<nl>";

/** An OTSL table's rows of cell texts; merged and empty cells are empty. Null for text that isn't OTSL. */
export function otslRows(text: string): string[][] | null {
  if (!text.includes("<fcel>") && !text.includes("<ecel>")) return null;
  const rows = text
    .split(OTSL_ROW)
    .map((row) => [...row.matchAll(OTSL_CELL)].map((cell) => (cell[1] === "fcel" ? cell[2].trim() : "")))
    .filter((row) => row.length > 0);
  return rows.length > 0 ? rows : null;
}

const escapeHtml = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const markdownCell = (text: string) => text.replace(/\|/g, "\\|").replace(/\n/g, " ");

function markdownTable(rows: string[][]): string {
  const width = Math.max(...rows.map((row) => row.length));
  const pad = (row: string[]) => [...row, ...Array(width - row.length).fill("")];
  const line = (row: string[]) => `| ${pad(row).map(markdownCell).join(" | ")} |`;
  return [line(rows[0]), `|${" --- |".repeat(width)}`, ...rows.slice(1).map(line)].join("\n");
}

function htmlTable(rows: string[][]): string {
  const cells = (row: string[], tag: string) => row.map((cell) => `<${tag}>${escapeHtml(cell)}</${tag}>`).join("");
  return `<table>\n<thead><tr>${cells(rows[0], "th")}</tr></thead>\n<tbody>\n${rows
    .slice(1)
    .map((row) => `<tr>${cells(row, "td")}</tr>`)
    .join("\n")}\n</tbody>\n</table>`;
}

// the regions a reading keeps, in the order the detector gave them
const readable = (page: ReadPage) => page.regions.filter((r) => !FURNITURE.has(r.label) && (r.text || PICTURES.has(r.label)));

function pageMarkdown(page: ReadPage): string {
  if (page.regions.length === 0) return page.text.trim();
  return readable(page)
    .map((region) => {
      const text = region.text.trim();
      if (PICTURES.has(region.label)) return `*[${region.label.replace(/_/g, " ")}]*`;
      if (region.label === "doc_title") return `# ${text}`;
      if (region.label === "paragraph_title") return `## ${text}`;
      if (FORMULAS.has(region.label)) return `$$\n${text}\n$$`;
      const rows = region.label === "table" ? otslRows(text) : null;
      return rows ? markdownTable(rows) : text;
    })
    .join("\n\n");
}

function pageText(page: ReadPage): string {
  if (page.regions.length === 0) return page.text.trim();
  return readable(page)
    .filter((region) => !PICTURES.has(region.label))
    .map((region) => {
      const rows = region.label === "table" ? otslRows(region.text) : null;
      return rows ? rows.map((row) => row.join("\t")).join("\n") : region.text.trim();
    })
    .join("\n\n");
}

function pageHtml(page: ReadPage): string {
  if (page.regions.length === 0) return `<p>${escapeHtml(page.text.trim())}</p>`;
  return readable(page)
    .map((region) => {
      const text = region.text.trim();
      if (PICTURES.has(region.label)) return `<p><em>[${escapeHtml(region.label.replace(/_/g, " "))}]</em></p>`;
      if (region.label === "doc_title") return `<h1>${escapeHtml(text)}</h1>`;
      if (region.label === "paragraph_title") return `<h2>${escapeHtml(text)}</h2>`;
      if (FORMULAS.has(region.label)) return `<pre class="formula">${escapeHtml(text)}</pre>`;
      const rows = region.label === "table" ? otslRows(text) : null;
      return rows ? htmlTable(rows) : `<p>${escapeHtml(text)}</p>`;
    })
    .join("\n");
}

/** The read document in `format`, its pages in order. */
export function renderDocument(title: string, pages: ReadPage[], format: DocumentFormat): string {
  switch (format) {
    case "md":
      return pages.map(pageMarkdown).join("\n\n---\n\n") + "\n";
    case "txt":
      return pages.map(pageText).join("\n\n\f\n\n") + "\n";
    case "html":
      return `<!doctype html>\n<html><head><meta charset="utf-8"><title>${escapeHtml(title)}</title>
<style>body{font-family:system-ui,sans-serif;max-width:48rem;margin:2rem auto;line-height:1.5;padding:0 1rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.25rem .5rem}.page{border-top:1px solid #ddd;margin-top:2rem;padding-top:1rem}</style>
</head><body>\n${pages.map((page, i) => `<section class="page" id="page-${i + 1}">\n${pageHtml(page)}\n</section>`).join("\n")}\n</body></html>\n`;
    case "json":
      return JSON.stringify({ title, pages: pages.map((page, i) => ({ page: i + 1, ...page })) }, null, 2);
  }
}
