import { authedFetch } from "./auth";

/** One OpenAI chat message as a Code session stores it. */
export interface CodeMessage {
  role: "user" | "assistant" | "tool";
  content: string | null;
  reasoning_content?: string;
  tool_calls?: { id: string; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
}

export interface CodeSessionInfo {
  id: string;
  title: string;
  modified: number;
}

export type PermissionMode = "ask" | "auto" | "read-only";
export type Decision = "once" | "always" | "deny";

/** What the agent streams while it works on a prompt. */
export type CodeEvent =
  | { type: "text"; delta: string }
  | { type: "reasoning"; delta: string }
  | { type: "tool_start"; id: string; name: string; args: Record<string, unknown> }
  | { type: "tool_end"; id: string; text: string; is_error: boolean }
  | {
      type: "approval";
      approval_id: string;
      id: string;
      name: string;
      args: Record<string, unknown>;
      kind: string;
    }
  | { type: "usage"; usage: Record<string, unknown> }
  | { type: "done"; end: "done" | "interrupted" | "round_limit" | "reply_limit" }
  | { type: "error"; message: string };

function url(path: string): string {
  const base = document.querySelector("base")?.getAttribute("href") ?? "/ui/";
  return `${base}api/code/${path}`;
}

async function failure(res: Response): Promise<Error> {
  const text = await res.text().catch(() => res.statusText);
  return new Error(text || `HTTP ${res.status}`);
}

async function post(path: string, body: unknown): Promise<Response> {
  const res = await authedFetch(url(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await failure(res);
  return res;
}

export async function listSessions(
  project: string,
): Promise<{ project: string; sessions: CodeSessionInfo[] }> {
  const res = await authedFetch(url(`sessions?project=${encodeURIComponent(project)}`));
  if (!res.ok) throw await failure(res);
  return res.json();
}

export async function openSession(
  project: string,
  model: string | null,
  sessionId?: string,
): Promise<{ session_id: string; project: string; messages: CodeMessage[] }> {
  return (await post("open", { project, session_id: sessionId, model })).json();
}

/** Makes a demo project (a small package whose tests fail) and opens a session there; send `prompt` to watch the agent fix it. */
export async function startDemo(
  model: string | null,
): Promise<{ session_id: string; project: string; prompt: string }> {
  return (await post("demo", { model })).json();
}

export async function approve(sessionId: string, approvalId: string, decision: Decision) {
  await post("approve", { session_id: sessionId, approval_id: approvalId, decision });
}

export async function interrupt(sessionId: string) {
  await post("interrupt", { session_id: sessionId });
}

/** Sends a prompt and calls `onEvent` for each event until the agent finishes. */
export async function send(
  sessionId: string,
  model: string | null,
  prompt: string,
  mode: PermissionMode,
  enableThinking: boolean | null,
  onEvent: (event: CodeEvent) => void,
): Promise<void> {
  const body: Record<string, unknown> = { session_id: sessionId, prompt, mode, model };
  if (enableThinking != null) body.enable_thinking = enableThinking;
  const res = await post("send", body);
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
