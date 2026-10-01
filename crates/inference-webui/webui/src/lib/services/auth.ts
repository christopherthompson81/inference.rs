import { authStore } from "../stores/auth.svelte";

/** The server root the UI is mounted under: /ui/ -> /. */
export function apiRoot(): string {
  const base = document.querySelector("base")?.getAttribute("href") ?? "/ui/";
  try {
    return new URL("../", new URL(base, window.location.origin)).pathname;
  } catch {
    return "/";
  }
}

/** `fetch`, flagging a 401 so the sign-in prompt shows; a keyed server signs the browser in with a cookie. */
export async function authedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const response = await fetch(input, init);
  if (response.status === 401) authStore.required = true;
  return response;
}

/** Exchanges `key` for a session cookie and reloads, so every view starts again signed in. */
export async function signIn(key: string): Promise<void> {
  const response = await fetch(`${apiRoot()}auth/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  });
  if (!response.ok) throw new Error("That key isn't one this server knows.");
  window.location.reload();
}

export async function signOut(): Promise<void> {
  await fetch(`${apiRoot()}auth/session`, { method: "DELETE" });
  window.location.reload();
}
