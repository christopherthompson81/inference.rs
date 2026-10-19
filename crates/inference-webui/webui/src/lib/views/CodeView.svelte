<script lang="ts">
  import { onMount, tick } from "svelte";
  import ThinkingBlock from "../components/ThinkingBlock.svelte";
  import { modelStore } from "../stores/models.svelte";
  import { renderMarkdown } from "../utils/markdown";
  import * as code from "../services/code";
  import type { CodeEvent, CodeMessage, CodeSessionInfo, Decision, PermissionMode } from "../services/code";

  const PROJECT_KEY = "inference-rs-code-project";
  const MODE_KEY = "inference-rs-code-mode";
  const SUMMARY_CHARS = 100;
  // the agent's own words for a call the user turned down
  const DECLINED = "the user declined this call";
  const MODES: { id: PermissionMode; label: string }[] = [
    { id: "ask", label: "Ask before edits and commands" },
    { id: "auto", label: "Run everything" },
    { id: "read-only", label: "Read only" },
  ];

  interface ToolItem {
    kind: "tool";
    id: string;
    name: string;
    args: Record<string, unknown>;
    result: string | null;
    isError: boolean;
    running: boolean;
    declined: boolean;
    approval: { id: string; kind: string } | null;
  }
  type Item = { kind: "user" | "reasoning" | "text" | "notice"; text: string } | ToolItem;

  function stored<T extends string>(key: string, fallback: T): T {
    try {
      return (localStorage.getItem(key) as T) ?? fallback;
    } catch {
      return fallback;
    }
  }

  function store(key: string, value: string) {
    try {
      localStorage.setItem(key, value);
    } catch {
      // private windows may refuse storage
    }
  }

  let project = $state(stored(PROJECT_KEY, ""));
  let mode = $state<PermissionMode>(stored(MODE_KEY, "ask"));
  let thinking = $state<"default" | "on" | "off">("default");
  let sessions = $state<CodeSessionInfo[]>([]);
  let sessionId = $state<string | null>(null);
  let projectPath = $state("");
  let items = $state<Item[]>([]);
  let prompt = $state("");
  let running = $state(false);
  let error = $state("");
  let usage = $state<Record<string, any> | null>(null);
  let container = $state<HTMLDivElement | null>(null);

  let hasRoot = $derived(!!modelStore.capabilities.code_root);
  let available = $derived(!!modelStore.capabilities.admin);

  function parseArgs(text: string): Record<string, unknown> {
    try {
      const args = JSON.parse(text || "{}");
      return args && typeof args === "object" ? args : {};
    } catch {
      return { arguments: text };
    }
  }

  function cut(text: string, max: number): string {
    return text.length > max ? `${text.slice(0, max)}...` : text;
  }

  function summary(name: string, args: Record<string, unknown>): string {
    const field = (key: string) => (typeof args[key] === "string" ? (args[key] as string) : null);
    let detail: string | null;
    if (name === "bash") detail = field("command")?.split("\n")[0] ?? null;
    else if (name === "grep" || name === "glob")
      detail = field("pattern") && (field("path") ? `${field("pattern")} in ${field("path")}` : field("pattern"));
    else detail = field("path");
    return `${name}(${cut(detail ?? JSON.stringify(args), SUMMARY_CHARS)})`;
  }

  function fromMessages(messages: CodeMessage[]): Item[] {
    const out: Item[] = [];
    for (const message of messages) {
      if (message.role === "user") {
        out.push({ kind: "user", text: message.content ?? "" });
      } else if (message.role === "assistant") {
        if (message.reasoning_content?.trim()) out.push({ kind: "reasoning", text: message.reasoning_content.trim() });
        if (message.content?.trim()) out.push({ kind: "text", text: message.content.trim() });
        for (const call of message.tool_calls ?? []) {
          out.push({
            kind: "tool",
            id: call.id,
            name: call.function.name,
            args: parseArgs(call.function.arguments),
            result: null,
            isError: false,
            running: false,
            declined: false,
            approval: null,
          });
        }
      } else if (message.role === "tool") {
        const tool = findTool(out, message.tool_call_id ?? "");
        if (tool) {
          tool.result = message.content ?? "";
          tool.declined = tool.result === DECLINED;
        }
      }
    }
    return out;
  }

  function findTool(list: Item[], id: string): ToolItem | undefined {
    return list.findLast((item): item is ToolItem => item.kind === "tool" && item.id === id);
  }

  function appendText(kind: "text" | "reasoning", delta: string) {
    const last = items[items.length - 1];
    if (last?.kind === kind) {
      last.text += delta;
    } else {
      // a new block starts at its first visible character
      const trimmed = delta.replace(/^\s+/, "");
      if (trimmed) items.push({ kind, text: trimmed });
    }
  }

  function onEvent(event: CodeEvent) {
    switch (event.type) {
      case "text":
      case "reasoning":
        appendText(event.type, event.delta);
        break;
      case "approval":
        items.push({
          kind: "tool",
          id: event.id,
          name: event.name,
          args: event.args,
          result: null,
          isError: false,
          running: false,
          declined: false,
          approval: { id: event.approval_id, kind: event.kind },
        });
        break;
      case "tool_start": {
        const tool = findTool(items, event.id);
        if (tool) {
          tool.running = true;
          tool.approval = null;
        } else {
          items.push({
            kind: "tool",
            id: event.id,
            name: event.name,
            args: event.args,
            result: null,
            isError: false,
            running: true,
            declined: false,
            approval: null,
          });
        }
        break;
      }
      case "tool_end": {
        const tool = findTool(items, event.id);
        if (tool) {
          tool.result = event.text;
          tool.isError = event.is_error;
          tool.running = false;
        }
        break;
      }
      case "usage":
        usage = event.usage;
        break;
      case "done":
        if (event.end === "interrupted") items.push({ kind: "notice", text: "Interrupted" });
        if (event.end === "round_limit") items.push({ kind: "notice", text: "Stopped at the round limit" });
        if (event.end === "reply_limit")
          items.push({ kind: "notice", text: "A reply hit the token limit (the model was likely repeating itself)" });
        break;
      case "error":
        error = event.message;
        break;
    }
    void scrollDown();
  }

  async function scrollDown() {
    await tick();
    if (container) container.scrollTop = container.scrollHeight;
  }

  async function refreshSessions() {
    if (!hasRoot && !project) return;
    try {
      const listing = await code.listSessions(project);
      sessions = listing.sessions;
      projectPath = listing.project;
    } catch (e) {
      sessions = [];
      error = `${e}`;
    }
  }

  async function open(id?: string) {
    try {
      const session = await code.openSession(project, modelStore.selectedModel, id);
      sessionId = session.session_id;
      projectPath = session.project;
      items = fromMessages(session.messages);
      usage = null;
      error = "";
      store(PROJECT_KEY, project);
      await refreshSessions();
      void scrollDown();
    } catch (e) {
      error = `${e}`;
    }
  }

  async function runDemo() {
    if (running) return;
    try {
      const demo = await code.startDemo(modelStore.selectedModel);
      project = demo.project;
      sessionId = demo.session_id;
      projectPath = demo.project;
      items = [];
      usage = null;
      error = "";
      await refreshSessions();
      prompt = demo.prompt;
      await submit();
    } catch (e) {
      error = `${e}`;
    }
  }

  async function submit() {
    const text = prompt.trim();
    if (!text || running) return;
    if (!sessionId) await open();
    if (!sessionId) return;
    prompt = "";
    error = "";
    items.push({ kind: "user", text });
    running = true;
    void scrollDown();
    try {
      await code.send(sessionId, modelStore.selectedModel, text, mode, thinking === "default" ? null : thinking === "on", onEvent);
    } catch (e) {
      error = `${e}`;
    } finally {
      running = false;
      for (const item of items) if (item.kind === "tool") item.running = false;
      void refreshSessions();
    }
  }

  async function decide(tool: ToolItem, decision: Decision) {
    if (!sessionId || !tool.approval) return;
    const approvalId = tool.approval.id;
    tool.approval = null;
    if (decision === "deny") tool.declined = true;
    try {
      await code.approve(sessionId, approvalId, decision);
    } catch (e) {
      error = `${e}`;
    }
  }

  async function stop() {
    if (sessionId) await code.interrupt(sessionId).catch(() => {});
  }

  function onKeydown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      void submit();
    }
  }

  function setMode(next: PermissionMode) {
    mode = next;
    store(MODE_KEY, next);
  }

  function relativeTime(seconds: number): string {
    const minutes = Math.round((Date.now() / 1000 - seconds) / 60);
    if (minutes < 1) return "just now";
    if (minutes < 60) return `${minutes} min ago`;
    if (minutes < 60 * 24) return `${Math.round(minutes / 60)} h ago`;
    return new Date(seconds * 1000).toLocaleDateString();
  }

  function lineCount(text: string): string {
    const lines = text.replace(/\n$/, "").split("\n").length;
    return lines === 1 ? "1 line" : `${lines} lines`;
  }

  function str(value: unknown): string {
    return typeof value === "string" ? value : "";
  }

  onMount(() => {
    if (available) void refreshSessions();
  });

  $effect(() => {
    if (available && hasRoot && !projectPath) void refreshSessions();
  });
</script>

{#if !modelStore.capabilities.admin}
  <div class="m-auto max-w-lg p-6 text-sm text-gray-600 dark:text-gray-400">
    <h2 class="mb-2 text-base font-semibold text-gray-800 dark:text-gray-200">Code needs an admin</h2>
    <p>
      The coding agent runs commands on the server's machine, so only admins may use it: a key the server lists with
      <code class="rounded bg-gray-100 px-1 dark:bg-gray-800">--admin</code>, or on a server without keys a browser on the
      server's own machine.
    </p>
  </div>
{:else}
  <div class="flex min-h-0 min-w-0 flex-1 flex-col md:flex-row">
    <aside class="flex shrink-0 flex-col gap-3 border-b border-gray-200 p-3 md:w-64 md:border-r md:border-b-0 dark:border-gray-800">
      {#if hasRoot}
        <label class="text-xs font-medium text-gray-500 dark:text-gray-400" for="code-project">Project</label>
        <input
          id="code-project"
          class="min-w-0 rounded-md border border-gray-300 bg-white px-2 py-1 text-sm dark:border-gray-700 dark:bg-gray-900"
          placeholder="relative to the code root"
          bind:value={project}
          onchange={() => refreshSessions()}
        />
      {:else}
        <p class="text-xs text-gray-500 dark:text-gray-400">
          Start the server with <code>--code-root DIR</code> to open your own projects here.
        </p>
      {/if}
      <p class="truncate text-xs text-gray-400" title={projectPath}>{projectPath || modelStore.capabilities.code_root || ""}</p>
      {#if hasRoot}
        <button
          class="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          onclick={() => open()}
          disabled={running}
        >
          New session
        </button>
      {/if}
      <button
        class="rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-100 disabled:opacity-50 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
        onclick={runDemo}
        disabled={running}
        title="Make a temp project with two planted bugs and ask the agent to fix them"
      >
        Run demo
      </button>
      <div class="min-h-0 flex-1 overflow-y-auto">
        <!-- a session nothing was sent in is only worth listing while it's open -->
        {#each sessions.filter((s) => s.title || s.id === sessionId) as session (session.id)}
          <button
            class="mb-1 block w-full rounded-md px-2 py-1.5 text-left text-sm {session.id === sessionId
              ? 'bg-gray-200 dark:bg-gray-800'
              : 'hover:bg-gray-100 dark:hover:bg-gray-900'}"
            onclick={() => open(session.id)}
            disabled={running}
          >
            <span class="block truncate text-gray-800 dark:text-gray-200">{session.title || "(empty session)"}</span>
            <span class="block text-xs text-gray-400">{relativeTime(session.modified)}</span>
          </button>
        {:else}
          <p class="text-xs text-gray-400">No sessions in this project yet.</p>
        {/each}
      </div>
    </aside>

    <section class="flex min-h-0 min-w-0 flex-1 flex-col">
      <div class="flex flex-wrap items-center gap-2 border-b border-gray-200 px-4 py-2 text-xs text-gray-500 dark:border-gray-800">
        <span class="truncate">{sessionId ? `session ${sessionId}` : "no session open"}</span>
        <span class="ml-auto"></span>
        <select
          class="rounded border border-gray-300 bg-white px-1 py-0.5 dark:border-gray-700 dark:bg-gray-900"
          value={thinking}
          onchange={(e) => (thinking = e.currentTarget.value as typeof thinking)}
          aria-label="Thinking"
        >
          <option value="default">Thinking: model default</option>
          <option value="on">Thinking: on</option>
          <option value="off">Thinking: off</option>
        </select>
        <select
          class="rounded border border-gray-300 bg-white px-1 py-0.5 dark:border-gray-700 dark:bg-gray-900"
          value={mode}
          onchange={(e) => setMode(e.currentTarget.value as PermissionMode)}
          aria-label="Permission mode"
        >
          {#each MODES as option (option.id)}
            <option value={option.id}>{option.label}</option>
          {/each}
        </select>
      </div>

      <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4" bind:this={container}>
        <div class="mx-auto flex max-w-3xl flex-col gap-3">
          {#if items.length === 0}
            <p class="mt-8 text-center text-sm text-gray-400">
              Ask for a change in {projectPath || "the project"}: the agent reads files, edits them and runs commands
              there.
            </p>
          {/if}
          {#each items as item, index (index)}
            {#if item.kind === "user"}
              <div class="self-end rounded-2xl bg-blue-600 px-4 py-2 text-sm whitespace-pre-wrap text-white">{item.text}</div>
            {:else if item.kind === "reasoning"}
              <ThinkingBlock reasoning={item.text} streaming={running && index === items.length - 1} />
            {:else if item.kind === "text"}
              <div class="markdown-content text-sm text-gray-800 dark:text-gray-200">{@html renderMarkdown(item.text)}</div>
            {:else if item.kind === "notice"}
              <p class="text-xs text-gray-400">{item.text}</p>
            {:else}
              <div
                class="rounded-lg border text-sm {item.approval
                  ? 'border-amber-400 bg-amber-50 dark:border-amber-600 dark:bg-amber-950/30'
                  : item.isError || item.declined
                    ? 'border-red-200 dark:border-red-900'
                    : 'border-gray-200 dark:border-gray-800'}"
                data-tool={item.name}
              >
                <details>
                  <summary class="flex cursor-pointer items-center gap-2 px-3 py-1.5 font-mono text-xs">
                    <span
                      class="h-2 w-2 shrink-0 rounded-full {item.running
                        ? 'animate-pulse bg-amber-400'
                        : item.approval
                          ? 'bg-amber-500'
                          : item.isError || item.declined
                            ? 'bg-red-500'
                            : item.result != null
                              ? 'bg-green-500'
                              : 'bg-gray-300'}"
                    ></span>
                    <span class="truncate text-gray-800 dark:text-gray-200">{summary(item.name, item.args)}</span>
                    {#if item.declined}
                      <span class="ml-auto shrink-0 text-red-500">declined</span>
                    {:else if item.result != null}
                      <span class="ml-auto shrink-0 text-gray-400">{lineCount(item.result)}</span>
                    {/if}
                  </summary>
                  {#if item.result != null}
                    <pre class="max-h-80 overflow-auto border-t border-gray-200 px-3 py-2 text-xs dark:border-gray-800">{item.result}</pre>
                  {/if}
                </details>
                {#if item.approval}
                  <div class="border-t border-amber-200 px-3 py-2 dark:border-amber-800">
                    {#if item.name === "edit"}
                      <pre class="max-h-60 overflow-auto rounded bg-red-50 px-2 py-1 text-xs text-red-800 dark:bg-red-950/40 dark:text-red-300">{str(item.args.old_string)}</pre>
                      <pre class="mt-1 max-h-60 overflow-auto rounded bg-green-50 px-2 py-1 text-xs text-green-800 dark:bg-green-950/40 dark:text-green-300">{str(item.args.new_string)}</pre>
                    {:else if item.name === "write"}
                      <pre class="max-h-60 overflow-auto rounded bg-green-50 px-2 py-1 text-xs text-green-800 dark:bg-green-950/40 dark:text-green-300">{str(item.args.content)}</pre>
                    {:else if item.name === "bash"}
                      <pre class="max-h-60 overflow-auto rounded bg-gray-100 px-2 py-1 text-xs dark:bg-gray-900">$ {str(item.args.command)}</pre>
                    {/if}
                    <div class="mt-2 flex flex-wrap gap-2">
                      <button class="rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700" onclick={() => decide(item, "once")}>Allow</button>
                      <button class="rounded border border-gray-300 px-3 py-1 text-xs hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800" onclick={() => decide(item, "always")}>Always allow {item.name}</button>
                      <button class="rounded border border-gray-300 px-3 py-1 text-xs text-red-600 hover:bg-red-50 dark:border-gray-700 dark:hover:bg-red-950/30" onclick={() => decide(item, "deny")}>Deny</button>
                    </div>
                  </div>
                {/if}
              </div>
            {/if}
          {/each}
          {#if error}
            <p class="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950/40 dark:text-red-300">{error}</p>
          {/if}
        </div>
      </div>

      <div class="shrink-0 border-t border-gray-200 px-4 py-3 dark:border-gray-800">
        <div class="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            class="min-w-0 flex-1 resize-none rounded-xl border border-gray-300 bg-gray-50 px-4 py-2.5 text-sm outline-none focus:border-blue-400 focus:bg-white focus:ring-1 focus:ring-blue-400 dark:border-gray-600 dark:bg-gray-800 dark:focus:bg-gray-900"
            rows="2"
            placeholder="Describe a change (Enter sends, Shift+Enter for a new line)"
            bind:value={prompt}
            onkeydown={onKeydown}
          ></textarea>
          {#if running}
            <button class="rounded-xl bg-red-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-700" onclick={stop}>Stop</button>
          {:else}
            <button
              class="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
              onclick={submit}
              disabled={!prompt.trim()}
            >
              Send
            </button>
          {/if}
        </div>
        {#if usage}
          <p class="mx-auto mt-1 max-w-3xl text-xs text-gray-400">
            context {usage.total_tokens} tokens, {usage.prompt_tokens_details?.cached_tokens ?? 0} cached, prefill
            {Number(usage.total_prompt_time_sec ?? 0).toFixed(1)} s, {Number(usage.avg_compl_tok_per_sec ?? 0).toFixed(1)} tok/s
          </p>
        {/if}
      </div>
    </section>
  </div>
{/if}
