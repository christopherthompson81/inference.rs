const RIFF_HEADER_BYTES = 12;
const CHUNK_HEADER_BYTES = 8;

interface Wav {
  format: Uint8Array<ArrayBuffer>;
  data: Uint8Array<ArrayBuffer>;
}

function parse(bytes: ArrayBuffer): Wav {
  const view = new DataView(bytes);
  let format: Uint8Array<ArrayBuffer> | null = null;
  let data: Uint8Array<ArrayBuffer> | null = null;
  for (let at = RIFF_HEADER_BYTES; at + CHUNK_HEADER_BYTES <= bytes.byteLength; ) {
    const id = String.fromCharCode(...new Uint8Array(bytes, at, 4));
    const size = view.getUint32(at + 4, true);
    const body = new Uint8Array(bytes, at + CHUNK_HEADER_BYTES, Math.min(size, bytes.byteLength - at - CHUNK_HEADER_BYTES));
    if (id === "fmt ") format = body;
    else if (id === "data") data = body;
    // chunks are padded to an even size
    at += CHUNK_HEADER_BYTES + size + (size % 2);
  }
  if (!format || !data) throw new Error("not a WAV file");
  return { format, data };
}

/** The WAVs played one after another as one file; they must share a format, as one model's audio does. */
export async function joinWavs(parts: Blob[]): Promise<Blob> {
  const wavs = await Promise.all(parts.map(async (part) => parse(await part.arrayBuffer())));
  if (wavs.length === 0) throw new Error("nothing to join");
  const format = wavs[0].format;
  const same = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((byte, i) => byte === b[i]);
  if (!wavs.every((wav) => same(wav.format, format))) throw new Error("the parts have different audio formats");
  const dataBytes = wavs.reduce((sum, wav) => sum + wav.data.length, 0);
  const header = new ArrayBuffer(RIFF_HEADER_BYTES + CHUNK_HEADER_BYTES * 2 + format.length);
  const view = new DataView(header);
  const text = (at: number, value: string) => [...value].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, header.byteLength - CHUNK_HEADER_BYTES + dataBytes, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, format.length, true);
  new Uint8Array(header, 20, format.length).set(format);
  const dataAt = 20 + format.length;
  text(dataAt, "data");
  view.setUint32(dataAt + 4, dataBytes, true);
  return new Blob([header, ...wavs.map((wav) => wav.data)], { type: "audio/wav" });
}
