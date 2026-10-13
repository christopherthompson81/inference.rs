// the rate the speech models run at; clips are cut and sent at it, so nothing is resampled twice
const CLIP_RATE = 16_000;
const WAV_HEADER_BYTES = 44;
const PCM16_MAX = 0x7fff;

/** `file` decoded to mono at 16 kHz, for cutting clips from. */
export async function decodeMono(file: Blob): Promise<Float32Array> {
  const bytes = await file.arrayBuffer();
  // decoding through an offline context at the clip rate resamples and mixes down in one pass
  const probe = new OfflineAudioContext(1, 1, CLIP_RATE);
  const decoded = await probe.decodeAudioData(bytes);
  const frames = Math.ceil(decoded.duration * CLIP_RATE);
  const context = new OfflineAudioContext(1, Math.max(frames, 1), CLIP_RATE);
  const source = context.createBufferSource();
  source.buffer = decoded;
  source.connect(context.destination);
  source.start();
  return (await context.startRendering()).getChannelData(0);
}

/** Seconds of audio `pcm` holds. */
export const durationOf = (pcm: Float32Array) => pcm.length / CLIP_RATE;

/** The 16-bit WAV of `pcm` from `start` to `end` seconds. */
export function wavClip(pcm: Float32Array, start: number, end: number): Blob {
  const from = Math.max(0, Math.floor(start * CLIP_RATE));
  const to = Math.min(pcm.length, Math.ceil(end * CLIP_RATE));
  const samples = Math.max(0, to - from);
  const buffer = new ArrayBuffer(WAV_HEADER_BYTES + samples * 2);
  const view = new DataView(buffer);
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + samples * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, CLIP_RATE, true);
  view.setUint32(28, CLIP_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples * 2, true);
  for (let i = 0; i < samples; i++) {
    const sample = Math.max(-1, Math.min(1, pcm[from + i]));
    view.setInt16(WAV_HEADER_BYTES + i * 2, Math.round(sample * PCM16_MAX), true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
