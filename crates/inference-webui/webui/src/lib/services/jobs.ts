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

/** The extension a kept copy of `file` takes, from its name or else its type. */
export function extensionOf(file: Blob & { name?: string }, fallback: string): string {
  const named = file.name?.split(".").pop()?.toLowerCase();
  if (file.name?.includes(".") && named && /^[a-z0-9]+$/.test(named)) return named;
  return file.type.split("/")[1]?.replace("mpeg", "mp3").replace("x-wav", "wav") || fallback;
}

/** A blob from a data URL, as the image routes return them. */
export async function blobOf(url: string): Promise<Blob> {
  return (await fetch(url)).blob();
}
