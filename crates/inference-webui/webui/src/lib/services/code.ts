import { authedFetch } from "./auth";

/** One OpenAI chat message as a Code session stores it. */
/** A content part of a message that shows the model images. */
/** A content part of a message that shows the model images; `tool_call_id` marks an image a tool call returned. */
export type CodePart =
  | { type: "text"; text: string; tool_call_id?: string }
  | { type: "image_url"; image_url: { url: string }; tool_call_id?: string };

export interface CodeMessage {
  role: "user" | "assistant" | "tool";
  content: string | CodePart[] | null;
  reasoning_content?: string;
  tool_calls?: { id: string; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
}

export interface CodeSessionInfo {
  id: string;
  title: string;
  modified: number;
}

/** Typed as a prompt, folds the session's earlier history into a summary instead of asking the model. */
export const COMPACT_COMMAND = "/compact";

export type PermissionMode = "ask" | "auto" | "read-only";
export type Decision = "once" | "always" | "deny";

/** What the agent streams while it works on a prompt. */
export type CodeEvent =
  /** The prompt the run is for, first of its events. */
  | { type: "prompt"; text: string; images: string[] }
  /** The model's context in tokens, where the server can tell. */
  | { type: "context"; tokens: number | null }
  /** A page attaching late has seen every event so far; the run started `elapsed_ms` ago. */
  | { type: "caught_up"; elapsed_ms: number }
  /** A request of about `prompt_tokens` (the harness's estimate) went to the model. */
  | { type: "round_start"; prompt_tokens: number }
  | { type: "text"; delta: string }
  | { type: "reasoning"; delta: string }
  | {
      type: "tool_start";
      id: string;
      name: string;
      args: Record<string, unknown>;
    }
  /** `images`: data URLs of the images the call returned, which the model sees after the round's results. */
  | { type: "tool_end"; id: string; text: string; is_error: boolean; images?: string[] }
  | {
      type: "approval";
      approval_id: string;
      id: string;
      name: string;
      args: Record<string, unknown>;
      kind: string;
    }
  | { type: "usage"; usage: Record<string, unknown> }
  /** What a run of thinking and tool calls was for, by the id of its first call; may follow `done`. */
  | { type: "label"; id: string; label: string }
  /** The history before message `end` is being folded into a summary, to fit the model's context. */
  | { type: "compacting"; end: number }
  /** It was: the model now sees the summary and the messages from `start` on (system prompt counted). */
  | { type: "compacted"; start: number; at: number | null; context_estimate: number }
  | {
      type: "done";
      end: "done" | "interrupted" | "round_limit" | "reply_limit" | "compacted" | "nothing_to_compact";
    }
  | { type: "error"; message: string };

function url(path: string): string {
  const base = document.querySelector("base")?.getAttribute("href") ?? "/ui/";
  return `${base}api/code/${path}`;
}

async function failure(res: Response): Promise<Error> {
  const text = await res.text().catch(() => res.statusText);
  return new Error(text || `HTTP ${res.status}`);
}

async function post(path: string, body: unknown, signal?: AbortSignal): Promise<Response> {
  const res = await authedFetch(url(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await failure(res);
  return res;
}

export async function listSessions(
  project: string,
): Promise<{ project: string; sessions: CodeSessionInfo[] }> {
  const res = await authedFetch(
    url(`sessions?project=${encodeURIComponent(project)}`),
  );
  if (!res.ok) throw await failure(res);
  return res.json();
}

export async function openSession(
  project: string,
  model: string | null,
  sessionId?: string,
): Promise<{
  session_id: string;
  project: string;
  messages: CodeMessage[];
  /** What each run of tool calls was for, by the id of its first call. */
  labels: Record<string, string>;
  /** Where the model's view begins past a summary of the earlier messages, as an index into `messages`. */
  compacted_before: number | null;
  /** When that summary was made, in seconds since the epoch. */
  compacted_at: number | null;
  context_tokens: number | null;
  /** About how many tokens the model's view of the session holds, by the harness's count. */
  context_estimate: number;
  /** The server's report on the last request of the last prompt run since the session loaded. */
  usage: Record<string, any> | null;
  /** A prompt is running: `messages` are the session as it found it, and `attach` streams its events. */
  running: boolean;
}> {
  return (await post("open", { project, session_id: sessionId, model })).json();
}

/** Makes a demo project (a small package whose tests fail) and opens a session there; send `prompt` to watch the agent fix it. */
export async function startDemo(
  model: string | null,
): Promise<{ session_id: string; project: string; prompt: string }> {
  return (await post("demo", { model })).json();
}

export async function approve(
  sessionId: string,
  approvalId: string,
  decision: Decision,
) {
  await post("approve", {
    session_id: sessionId,
    approval_id: approvalId,
    decision,
  });
}

export async function interrupt(sessionId: string) {
  await post("interrupt", { session_id: sessionId });
}

export interface SendOptions {
  model: string | null;
  mode: PermissionMode;
  enableThinking: boolean | null;
  describeActivity: boolean;
  /** Data URLs of the images the prompt shows the model. */
  images: string[];
}

/** Sends a prompt and calls `onEvent` for each event until the agent finishes; the prompt goes on if `signal` aborts. */
export async function send(
  sessionId: string,
  prompt: string,
  options: SendOptions,
  onEvent: (event: CodeEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const body: Record<string, unknown> = {
    session_id: sessionId,
    prompt,
    mode: options.mode,
    model: options.model,
    describe_activity: options.describeActivity,
    images: options.images,
  };
  if (options.enableThinking != null) body.enable_thinking = options.enableThinking;
  await readEvents(await post("send", body, signal), onEvent);
}

/** The running prompt's events from its first, for a session `openSession` found running. */
export async function attach(
  sessionId: string,
  onEvent: (event: CodeEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  await readEvents(await post("attach", { session_id: sessionId }, signal), onEvent);
}

async function readEvents(res: Response, onEvent: (event: CodeEvent) => void): Promise<void> {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("no response body");
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const event = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = event
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data) as CodeEvent);
    }
  }
}

export type ChangeStatus = "added" | "modified" | "deleted" | "no_original";

/** A file the session changed: relative to the project when inside it, else absolute. */
export interface FileChange {
  path: string;
  status: ChangeStatus;
  added: number;
  removed: number;
  binary: boolean;
}

export interface DiffLine {
  kind: "context" | "added" | "removed";
  old: number | null;
  new: number | null;
  text: string;
}

/** One file's diff; an unchanged file comes back whole as context, with no status. */
export interface FileDiff extends Omit<FileChange, "status"> {
  status: ChangeStatus | null;
  too_large: boolean;
  hunks: DiffLine[][];
}

export interface TreeEntry {
  name: string;
  dir: boolean;
}

async function get<T>(
  path: string,
  params: Record<string, string>,
): Promise<T> {
  const res = await authedFetch(url(`${path}?${new URLSearchParams(params)}`));
  if (!res.ok) throw await failure(res);
  return res.json();
}

export async function listChanges(sessionId: string): Promise<FileChange[]> {
  return (
    await get<{ files: FileChange[] }>("changes", { session_id: sessionId })
  ).files;
}

export async function fileDiff(
  sessionId: string,
  path: string,
): Promise<FileDiff> {
  return get("diff", { session_id: sessionId, path });
}

/** One directory of the project (`""` is its root), directories first. */
export async function listDir(
  sessionId: string,
  dir: string,
): Promise<TreeEntry[]> {
  return (
    await get<{ entries: TreeEntry[] }>("tree", {
      session_id: sessionId,
      path: dir,
    })
  ).entries;
}
