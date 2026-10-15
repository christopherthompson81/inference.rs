// the rate the speech models run at, which the server would resample to anyway
const MODEL_RATE = 16_000;
const WAV_HEADER_BYTES = 44;
const PCM16_MAX = 0x7fff;
// a RIFF header's fields: channels at 22, sample rate at 24, bits per sample at 34
const FORMAT_PROBE_BYTES = 36;
// the EBML magic that opens a Matroska or WebM file
const EBML_MAGIC = [0x1a, 0x45, 0xdf, 0xa3];

// Matroska often carries Opus or AC-3, which the server has no decoder for, so the browser decodes it
async function matroska(file: Blob): Promise<boolean> {
  const head = new Uint8Array(await file.slice(0, EBML_MAGIC.length).arrayBuffer());
  return EBML_MAGIC.every((byte, i) => head[i] === byte);
}

// a WAV wider than the models need: more channels, a higher rate or deeper samples than 16 kHz mono 16-bit
async function oversizedWav(file: Blob): Promise<boolean> {
  const head = new DataView(await file.slice(0, FORMAT_PROBE_BYTES).arrayBuffer());
  if (head.byteLength < FORMAT_PROBE_BYTES) return false;
  const tag = (offset: number) => String.fromCharCode(...new Uint8Array(head.buffer, offset, 4));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") return false;
  return head.getUint16(22, true) > 1 || head.getUint32(24, true) > MODEL_RATE || head.getUint16(34, true) > 16;
}

// `file` decoded, mixed to mono and resampled to the models' rate in one offline render
async function monoAtModelRate(file: Blob): Promise<Float32Array> {
  const probe = new OfflineAudioContext(1, 1, MODEL_RATE);
  const decoded = await probe.decodeAudioData(await file.arrayBuffer());
  const frames = Math.max(1, Math.ceil(decoded.duration * MODEL_RATE));
  const context = new OfflineAudioContext(1, frames, MODEL_RATE);
  const source = context.createBufferSource();
  source.buffer = decoded;
  source.connect(context.destination);
  source.start();
  return (await context.startRendering()).getChannelData(0);
}

function wav16(pcm: Float32Array): Blob {
  const buffer = new ArrayBuffer(WAV_HEADER_BYTES + pcm.length * 2);
  const view = new DataView(buffer);
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + pcm.length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, MODEL_RATE, true);
  view.setUint32(28, MODEL_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) {
    view.setInt16(WAV_HEADER_BYTES + i * 2, Math.round(Math.max(-1, Math.min(1, pcm[i])) * PCM16_MAX), true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/** What to upload for `file`: a WAV wider than the models need goes as 16 kHz mono when that is smaller, and a
 * Matroska or WebM file always does; any other file, or one the browser can't decode, goes as it is. */
export async function compactAudio(file: File): Promise<File> {
  const container = await matroska(file);
  if (!container && !(await oversizedWav(file))) return file;
  try {
    const compact = wav16(await monoAtModelRate(file));
    const name = container ? file.name.replace(/\.[^.]*$/, "") + ".wav" : file.name;
    return container || compact.size < file.size ? new File([compact], name, { type: "audio/wav" }) : file;
  } catch {
    return file;
  }
}
