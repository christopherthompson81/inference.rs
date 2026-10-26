import { apiRoot, authedFetch } from "./auth";

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

/** How much a job may unload to run; null leaves it to the server. */
export type Placement = "fastest" | "when_quiet" | "no_eviction";

export interface JobStage {
  name: string;
  /** Steps in the stage; 0 when it has no count. */
  total: number;
  done: number;
  started_at: number;
  finished_at?: number | null;
}

export interface JobPreview {
  stage: string;
  caption: string;
  /** Relative to the job's folder. */
  file: string;
}

export interface JobOutput {
  name: string;
  file: string;
}

export interface JobRecord {
  id: string;
  kind: string;
  status: JobStatus;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  request: Record<string, unknown>;
  inputs: string[];
  stages: JobStage[];
  previews: JobPreview[];
  outputs: JobOutput[];
  error?: string | null;
  /** Why a queued job has not started. */
  waiting?: string | null;
  /** The CUDA device it ran on. */
  device?: number | null;
  report: unknown;
}

export type JobEvent =
  | { type: "status"; status: JobStatus }
  | { type: "waiting"; reason: string }
  | { type: "stage"; name: string; total: number }
  | { type: "progress"; done: number; total: number }
  | ({ type: "preview" } & JobPreview)
  | { type: "finished"; status: JobStatus; error?: string | null };

export interface WorldOptions {
  prompt: string;
  scene: "indoor" | "outdoor" | null;
  seed: number | null;
  splats: boolean;
  steps: number | null;
  placement: Placement | null;
}

export interface ReconstructOptions {
  target_size: number | null;
  fps: number | null;
  max_frames: number | null;
  placement: Placement | null;
}

export function isFinished(status: JobStatus): boolean {
  return status === "succeeded" || status === "failed" || status === "cancelled";
}

function url(path: string): string {
  return `${apiRoot()}${path}`;
}

async function failure(response: Response): Promise<Error> {
  try {
    const body = await response.json();
    return new Error(body?.error?.message ?? response.statusText);
  } catch {
    return new Error(response.statusText);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authedFetch(url(path), init);
  if (!response.ok) throw await failure(response);
  return response.status === 204 ? (undefined as T) : response.json();
}

// drops the options left to the server's defaults
function given(options: object): Record<string, unknown> {
  return Object.fromEntries(Object.entries(options).filter(([, value]) => value != null && value !== ""));
}

async function submit(kind: string, options: object, files: File[]): Promise<JobRecord> {
  const form = new FormData();
  form.append("request", JSON.stringify({ kind, ...given(options) }));
  for (const file of files) form.append("file", file);
  return request<JobRecord>("v1/jobs", { method: "POST", body: form });
}

/** A world from an equirectangular panorama: keyframes along camera paths, a reconstruction, then splats. */
export function submitWorld(panorama: File, options: WorldOptions): Promise<JobRecord> {
  return submit("world", options, [panorama]);
}

/** Cameras, depth, points and splats from photos of one scene, or a video. */
export function submitReconstruct(files: File[], options: ReconstructOptions): Promise<JobRecord> {
  return submit("reconstruct", options, files);
}

export async function listJobs(): Promise<JobRecord[]> {
  return (await request<{ data: JobRecord[] }>("v1/jobs")).data;
}

export function getJob(id: string): Promise<JobRecord> {
  return request<JobRecord>(`v1/jobs/${encodeURIComponent(id)}`);
}

export function cancelJob(id: string): Promise<JobRecord> {
  return request<JobRecord>(`v1/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
}

export function deleteJob(id: string): Promise<void> {
  return request<void>(`v1/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
}

/** Where a preview or output of the job is served. */
export function jobFileUrl(id: string, file: string): string {
  const path = file.split("/").map(encodeURIComponent).join("/");
  return url(`v1/jobs/${encodeURIComponent(id)}/files/${path}`);
}

/** Follows a job: its record as it stands, then each change until it ends; the returned function stops following. */
export function followJob(
  id: string,
  onRecord: (record: JobRecord) => void,
  onEvent: (event: JobEvent) => void,
  onDropped: () => void,
): () => void {
  const source = new EventSource(url(`v1/jobs/${encodeURIComponent(id)}/events`));
  source.addEventListener("record", (message) => onRecord(JSON.parse((message as MessageEvent).data)));
  // missed changes: the record as it stands now catches up
  source.addEventListener("lagged", () => void getJob(id).then(onRecord, () => {}));
  source.onmessage = (message) => {
    const event = JSON.parse(message.data) as JobEvent;
    onEvent(event);
    if (event.type === "finished") source.close();
  };
  // the stream ends with the job, and reconnecting would replay its record; a drop mid-job falls back to polling
  source.onerror = () => {
    source.close();
    onDropped();
  };
  return () => source.close();
}
