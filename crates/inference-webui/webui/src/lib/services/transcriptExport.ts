/** A timed word, in seconds from the start of the audio. */
export interface TimedWord {
  word: string;
  start: number;
  end: number;
}

/** One speaker's turn: its time, its text and the words in it. */
export interface TranscriptSegment {
  start: number;
  end: number;
  /** The diarizer's label (`Speaker 0`), which `names` maps to a display name; empty without one. */
  speaker: string;
  text: string;
  words?: TimedWord[];
}

export type ExportFormat = "srt" | "vtt" | "csv" | "json" | "txt";

export const EXPORT_FORMATS: { format: ExportFormat; label: string; mime: string }[] = [
  { format: "srt", label: "Subtitles (SRT)", mime: "application/x-subrip" },
  { format: "vtt", label: "Subtitles (WebVTT)", mime: "text/vtt" },
  { format: "csv", label: "Spreadsheet (CSV)", mime: "text/csv" },
  { format: "json", label: "JSON", mime: "application/json" },
  { format: "txt", label: "Plain text", mime: "text/plain" },
];

// `HH:MM:SS,mmm` for SubRip, `HH:MM:SS.mmm` for WebVTT
function stamp(seconds: number, separator: string): string {
  const millis = Math.round(Math.max(0, seconds) * 1000);
  const pad = (value: number, width = 2) => String(value).padStart(width, "0");
  return `${pad(Math.floor(millis / 3_600_000))}:${pad(Math.floor(millis / 60_000) % 60)}:${pad(Math.floor(millis / 1000) % 60)}${separator}${pad(millis % 1000, 3)}`;
}

const csvField = (value: string) => (/[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value);

/** The transcript in `format`, each segment under its speaker's display name. */
export function renderTranscript(
  segments: TranscriptSegment[],
  names: Record<string, string>,
  format: ExportFormat,
): string {
  const who = (segment: TranscriptSegment) => names[segment.speaker]?.trim() || segment.speaker;
  const line = (segment: TranscriptSegment) => (who(segment) ? `${who(segment)}: ${segment.text}` : segment.text);
  switch (format) {
    case "srt":
      return segments
        .map((s, i) => `${i + 1}\n${stamp(s.start, ",")} --> ${stamp(s.end, ",")}\n${line(s)}\n`)
        .join("\n");
    case "vtt":
      return `WEBVTT\n\n${segments
        .map((s) => `${stamp(s.start, ".")} --> ${stamp(s.end, ".")}\n${who(s) ? `<v ${who(s)}>` : ""}${s.text}\n`)
        .join("\n")}`;
    case "csv":
      return [
        "speaker,start,end,text",
        ...segments.map((s) => [who(s), s.start.toFixed(3), s.end.toFixed(3), s.text].map(csvField).join(",")),
      ].join("\n");
    case "json":
      return JSON.stringify(
        segments.map((s) => ({ speaker: who(s), start: s.start, end: s.end, text: s.text, words: s.words ?? [] })),
        null,
        2,
      );
    case "txt":
      return segments.map(line).join("\n\n");
  }
}

/** Saves `text` as `filename`. */
export function download(text: string, filename: string, mime: string) {
  const url = URL.createObjectURL(new Blob([text], { type: `${mime};charset=utf-8` }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
