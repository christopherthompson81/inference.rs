import { authedFetch } from "./auth";

export type JobKind = "transcription" | "speech" | "images" | "documents";

/** One finished job a view can be restored from: its fields in `data`, its media in `files`. */
export interface Job<Data = Record<string, unknown>> {
  id: string;
  kind: JobKind;
  created_at: string;
  title: string;
  model?: string;
  data: Data;
  files: string[];
}

export interface JobFile {
  name: string;
  blob: Blob;
}

function apiUrl(path: string): string {
  const base = document.querySelector("base")?.getAttribute("href") ?? "/ui/";
  return `${base}api/${path}`;
}

async function check(response: Response): Promise<Response> {
  if (!response.ok) throw new Error((await response.text()) || response.statusText);
  return response;
}

export function jobFileUrl(job: Job<unknown>, name: string): string {
  return apiUrl(`job_file/${job.id}/${encodeURIComponent(name)}`);
}

/** The job's file, as a `File` a request can send again. */
export async function jobFile(job: Job<unknown>, name: string): Promise<File> {
  const response = await check(await authedFetch(jobFileUrl(job, name)));
  return new File([await response.blob()], name, { type: response.headers.get("content-type") ?? "" });
}

export async function listJobs<Data>(kind: JobKind): Promise<Job<Data>[]> {
  const response = await check(await authedFetch(apiUrl(`list_jobs?kind=${kind}`)));
  return (await response.json()).jobs;
}

/** Keeps a job, or with `id` replaces its fields and adds `files` to it. */
export async function saveJob<Data>(
  job: { id?: string; kind: JobKind; title: string; model?: string; data: Data },
  files: JobFile[] = [],
): Promise<Job<Data>> {
  const form = new FormData();
  form.append("job", JSON.stringify(job));
  for (const file of files) form.append("file", file.blob, file.name);
  const response = await check(await authedFetch(apiUrl("save_job"), { method: "POST", body: form }));
  return response.json();
}

export async function deleteJob(id: string): Promise<void> {
  await check(
    await authedFetch(apiUrl("delete_job"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    }),
  );
}

// the extensions the server keeps (`JOB_FILE_TYPES` in jobs.rs), and the other spellings of them a file may carry
const KEPT_EXTENSIONS: Record<"image" | "audio", Record<string, string>> = {
  image: { png: "png", jpg: "jpg", jpeg: "jpg", webp: "webp", gif: "gif", bmp: "bmp" },
  audio: {
    wav: "wav", "x-wav": "wav", wave: "wav", mp3: "mp3", mpeg: "mp3", ogg: "ogg", oga: "ogg", opus: "opus",
    flac: "flac", "x-flac": "flac", m4a: "m4a", "x-m4a": "m4a", mp4: "m4a", aac: "aac", webm: "webm",
    aiff: "aiff", "x-aiff": "aiff", aif: "aiff",
  },
};

/** The extension the server keeps `file` under, from its name or else its type; null for a type it doesn't keep. */
export function keptExtension(file: Blob & { name?: string }, kind: "image" | "audio"): string | null {
  const known = KEPT_EXTENSIONS[kind];
  const named = file.name?.includes(".") ? file.name.split(".").pop()!.toLowerCase() : "";
  return known[named] ?? known[file.type.split("/")[1] ?? ""] ?? null;
}

async function fetched(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return response;
}

/** A blob from a data URL, as the image routes return them. */
export async function blobOf(url: string): Promise<Blob> {
  return (await fetched(url)).blob();
}

/** The job's file as a data URL, which the media routes take where they don't take a URL. */
export async function jobDataUrl(job: Job<unknown>, name: string): Promise<string> {
  const blob = await (await check(await authedFetch(jobFileUrl(job, name)))).blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/** `image` redrawn as a PNG, for a type the server doesn't keep. */
export async function pngOf(image: Blob): Promise<Blob> {
  const bitmap = await createImageBitmap(image);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0);
  bitmap.close();
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("the image could not be redrawn"))), "image/png"),
  );
}
