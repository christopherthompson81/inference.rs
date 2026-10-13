import { apiRoot, authedFetch } from "./auth";

/** How a model serves requests, as the server's model manager names it. */
export type ModelKind =
  | "text"
  | "multimodal"
  | "embedding"
  | "image_generation"
  | "speech"
  | "transcription"
  | "voice_activity"
  | "diarization"
  | "layout"
  | "projector";

/** One loadable model the server's folders or Hugging Face cache hold. */
export interface InventoryModel {
  path: string;
  source: "local" | "hf_cache";
  repo_id: string | null;
  format: "gguf" | "safetensors" | "nemo" | "other";
  kind: ModelKind | null;
  loader: string | null;
  architecture: string | null;
  name: string | null;
  size_label: string | null;
  quantization: string | null;
  context_length: number | null;
  base_model: string | null;
  size_bytes: number;
  modified: number | null;
  files: string[];
  projectors: string[];
  error: string | null;
}

export interface Inventory {
  models: InventoryModel[];
  hf_cache: { path: string; repos: { repo_id: string; size_bytes: number }[] } | null;
  errors: { path: string; error: string }[];
}

/** A served model and where it came from. */
export interface ManagedModel {
  model_id: string;
  kind: Exclude<ModelKind, "projector">;
  source: string | null;
  projector: string | null;
  status: "loaded" | "unloaded" | "reloading";
}

export interface LoadRequest {
  source: string;
  model_id: string;
  paged_context_len?: number;
  max_seqs?: number;
  mtp?: { n_predict?: number };
  mmproj?: string;
  device?: string;
}

export interface LoadProgress {
  stage: string;
  done: number;
  total: number;
  unit: "items" | "bytes";
}

export interface DownloadRequest {
  repo_id: string;
  revision?: string;
  files?: string[];
  quant?: string;
}

export interface DownloadProgress {
  file: string;
  done: number;
  total: number;
}

export interface LayoutRegion {
  label: string;
  score: number;
  bbox: [number, number, number, number];
  polygon: [number, number][];
}

export interface LayoutResult {
  width: number;
  height: number;
  regions: LayoutRegion[];
}

/** The OpenAI-style error a route answered with, or the status line when it gave none. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string | null = null,
  ) {
    super(message);
  }
}

function url(path: string): string {
  return `${apiRoot()}${path}`;
}

async function failure(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    const error = body?.error ?? body;
    return new ApiError(error?.message ?? response.statusText, response.status, error?.code ?? null);
  } catch {
    return new ApiError(response.statusText || `HTTP ${response.status}`, response.status);
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await authedFetch(url(path));
  if (!response.ok) throw await failure(response);
  return response.json();
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await authedFetch(url(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await failure(response);
  const text = await response.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

async function postForm<T>(path: string, form: FormData): Promise<T> {
  const response = await authedFetch(url(path), { method: "POST", body: form });
  if (!response.ok) throw await failure(response);
  return response.json();
}

/** Reads a POST's event stream, calling `onEvent` per event, until it ends or `signal` aborts it. */
async function postEvents(
  path: string,
  body: unknown,
  onEvent: (event: string, data: unknown) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await authedFetch(url(path), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) throw await failure(response);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    // a stream may end without the blank line after its last event
    if (done) buffer += "\n\n";
    // a proxy may rewrite line endings to CRLF
    else buffer += value.replace(/\r\n?/g, "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      // keep-alive comments carry no data
      if (data.length === 0) continue;
      const text = data.join("\n");
      let parsed: unknown = text;
      try {
        parsed = JSON.parse(text);
      } catch {
        // a plain-text event
      }
      onEvent(event, parsed);
    }
    if (done) break;
  }
}

export const inventory = () => getJson<Inventory>("v1/manager/inventory");

export const servedModels = () => getJson<{ data: ManagedModel[] }>("v1/manager/models").then((r) => r.data);

/** Loads a model, reporting its progress; resolves with the loaded model, or rejects with the load's error. */
export async function loadModel(
  request: LoadRequest,
  onProgress: (progress: LoadProgress) => void,
): Promise<ManagedModel> {
  let loaded: ManagedModel | null = null;
  let error: ApiError | null = null;
  await postEvents("v1/manager/load", request, (event, data) => {
    if (event === "progress") onProgress(data as LoadProgress);
    else if (event === "loaded") loaded = data as ManagedModel;
    else if (event === "error") {
      const body = (data as { error?: { message?: string; code?: string } })?.error;
      error = new ApiError(body?.message ?? "the load failed", 400, body?.code ?? null);
    }
  });
  if (error) throw error;
  if (!loaded) throw new ApiError("the load ended without a result", 500);
  return loaded;
}

export const unloadModel = (model_id: string) => postJson<void>("v1/manager/unload", { model_id });

/** Downloads into the server's cache until done, canceled (by `signal`) or failed. */
export async function downloadModel(
  request: DownloadRequest,
  onProgress: (progress: DownloadProgress) => void,
  signal?: AbortSignal,
): Promise<"done" | "canceled"> {
  let outcome: "done" | "canceled" = "canceled";
  let error: ApiError | null = null;
  try {
    await postEvents(
      "v1/manager/download",
      request,
      (event, data) => {
        if (event === "progress") onProgress(data as DownloadProgress);
        else if (event === "done") outcome = "done";
        else if (event === "error") {
          const body = (data as { error?: { message?: string } })?.error;
          error = new ApiError(body?.message ?? "the download failed", 400);
        }
      },
      signal,
    );
  } catch (e) {
    if (signal?.aborted) return "canceled";
    throw e;
  }
  if (error) throw error;
  return outcome;
}

export const deleteModel = (target: { repo_id?: string; path?: string }) =>
  postJson<{ deleted: string; freed_bytes: number }>("v1/manager/delete", target);

export const detectLayout = (model: string, image: string, threshold?: number) =>
  postJson<LayoutResult>("v1/layout/detect", { model, image, threshold });

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

export function transcribe(model: string, audio: File, language?: string) {
  const form = new FormData();
  form.append("file", audio);
  form.append("model", model);
  form.append("response_format", "verbose_json");
  if (language) form.append("language", language);
  return postForm<{ text: string; language?: string; duration?: number; segments?: TranscriptSegment[] }>(
    "v1/audio/transcriptions",
    form,
  );
}

export function diarize(model: string, audio: File) {
  const form = new FormData();
  form.append("file", audio);
  form.append("model", model);
  return postForm<{ num_speakers: number; segments: { speaker: number; start: number; end: number }[] }>(
    "v1/audio/diarization",
    form,
  );
}

/** Speech for `input`, as a WAV blob. */
export async function speak(request: { model: string; input: string; voice?: string; speed?: number }): Promise<Blob> {
  const response = await authedFetch(url("v1/audio/speech"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...request, response_format: "wav" }),
  });
  if (!response.ok) throw await failure(response);
  return response.blob();
}

export interface ImageOptions {
  width?: number;
  height?: number;
  steps?: number;
  guidance_scale?: number;
  seed?: number;
}

type ImageReply = { data: { b64_json?: string; url?: string }[] };

function imageSources(reply: ImageReply): string[] {
  return reply.data.map((image) => (image.b64_json ? `data:image/png;base64,${image.b64_json}` : (image.url ?? "")));
}

export async function generateImage(model: string, prompt: string, options: ImageOptions): Promise<string[]> {
  const reply = await postJson<ImageReply>("v1/images/generations", {
    model,
    prompt,
    response_format: "b64_json",
    ...options,
  });
  return imageSources(reply);
}

/** An edit of one reference image, or a new scene with several as subjects. */
export async function editImage(model: string, prompt: string, images: File[], options: ImageOptions): Promise<string[]> {
  const form = new FormData();
  form.append("model", model);
  form.append("prompt", prompt);
  form.append("response_format", "b64_json");
  for (const [key, value] of Object.entries(options)) {
    if (value != null) form.append(key, String(value));
  }
  for (const image of images) form.append("image", image);
  return imageSources(await postForm<ImageReply>("v1/images/edits", form));
}

/** One chat turn with an image, for reading a page region; the reply's text. */
export async function readImage(model: string, image: string, prompt: string): Promise<string> {
  const reply = await postJson<{ choices: { message: { content: string | null } }[] }>("v1/chat/completions", {
    model,
    messages: [
      {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: image } },
          { type: "text", text: prompt },
        ],
      },
    ],
    max_tokens: 2048,
    temperature: 0,
  });
  return reply.choices[0]?.message.content?.trim() ?? "";
}
