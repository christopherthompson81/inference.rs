<script lang="ts">
  import { onMount, tick } from "svelte";
  import CodeDiff from "../components/CodeDiff.svelte";
  import CodeFiles from "../components/CodeFiles.svelte";
  import ThinkingBlock from "../components/ThinkingBlock.svelte";
  import { modelStore } from "../stores/models.svelte";
  import { renderMarkdown } from "../utils/markdown";
  import * as code from "../services/code";
  import type {
    CodeEvent,
    CodeMessage,
    CodeSessionInfo,
    Decision,
    FileChange,
    FileDiff,
    PermissionMode,
  } from "../services/code";

  const PROJECT_KEY = "inference-rs-code-project";
  const MODE_KEY = "inference-rs-code-mode";
  const DIFF_WIDTH_KEY = "inference-rs-code-diff-width";
  const DESCRIBE_KEY = "inference-rs-code-describe-activity";
  // the diff pane's share of the width, kept between these so neither side vanishes
  const MIN_DIFF_SHARE = 0.2;
  const MAX_DIFF_SHARE = 0.8;
  const DEFAULT_DIFF_SHARE = 0.5;
  // calls after which the project may have changed
  const CHANGING_TOOLS = new Set(["edit", "write", "bash"]);
  const SUMMARY_CHARS = 100;
  // how often the agent clock redraws while a prompt runs
  const CLOCK_TICK_MS = 1000;
  // an activity line names this many files before it counts them instead
  const LABEL_FILES = 2;
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

  function storedShare(): number {
    const share = Number(stored(DIFF_WIDTH_KEY, ""));
    return share >= MIN_DIFF_SHARE && share <= MAX_DIFF_SHARE ? share : DEFAULT_DIFF_SHARE;
  }

  function store(key: string, value: string) {
    try {
      localStorage.setItem(key, value);
    } catch {
      // private windows may refuse storage
    }
  }

  let project = $state<string>(stored(PROJECT_KEY, ""));
  let mode = $state<PermissionMode>(stored(MODE_KEY, "ask"));
  let thinking = $state<"default" | "on" | "off">("default");
  // the model's words for what each activity line was for, by its first call; the calls' own summary until then
  let describe = $state(stored<"on" | "off">(DESCRIBE_KEY, "on") !== "off");
  let labels = $state<Record<string, string>>({});
  let sessions = $state<CodeSessionInfo[]>([]);
  let sessionId = $state<string | null>(null);
  let projectPath = $state("");
  let items = $state<Item[]>([]);
  let prompt = $state("");
  let running = $state(false);
  let error = $state("");
  let usage = $state<Record<string, any> | null>(null);
  // time the agent worked on the last prompt, less the time it waited for an approval
  let agentMs = $state(0);
  let activeSince = $state<number | null>(null);
  let now = $state(Date.now());
  let agentSeconds = $derived((agentMs + (activeSince != null ? now - activeSince : 0)) / 1000);
  let blocks = $derived(groupActivity(items));
  let container = $state<HTMLDivElement | null>(null);
  let sideTab = $state<"sessions" | "files">("sessions");
  let changes = $state<FileChange[]>([]);
  let selectedFile = $state<string | null>(null);
  let fileDiff = $state<FileDiff | null>(null);
  let diffError = $state("");
  let diffShare = $state(storedShare());
  let split = $state<HTMLDivElement | null>(null);

  let roots = $derived(modelStore.capabilities.code_roots ?? []);
  let hasRoot = $derived(roots.length > 0);
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

  type Entry = { item: Item; index: number };
  type Block = { kind: "one"; entry: Entry } | { kind: "activity"; entries: Entry[]; key: number };

  // Thinking and the tool calls after it fold into one line until the next text, as Claude Code shows them
  function groupActivity(list: Item[]): Block[] {
    const out: Block[] = [];
    let run: Entry[] = [];
    const flush = () => {
      if (run.some((entry) => entry.item.kind === "tool")) out.push({ kind: "activity", entries: run, key: run[0].index });
      else for (const entry of run) out.push({ kind: "one", entry });
      run = [];
    };
    list.forEach((item, index) => {
      if (item.kind === "reasoning" || item.kind === "tool") {
        run.push({ item, index });
      } else {
        flush();
        out.push({ kind: "one", entry: { item, index } });
      }
    });
    flush();
    return out;
  }

  function toolsOf(entries: Entry[]): ToolItem[] {
    return entries.map((entry) => entry.item).filter((item): item is ToolItem => item.kind === "tool");
  }

  function activityLabel(entries: Entry[]): string {
    const tools = toolsOf(entries);
    const described = describe && tools.length > 0 ? labels[tools[0].id] : undefined;
    if (described) return `${described} (${tools.length})`;
    const of = (names: string[]) => tools.filter((tool) => names.includes(tool.name));
    const files = (list: ToolItem[]) => [
      ...new Set(list.map((tool) => str(tool.args.path).split("/").pop() ?? "").filter(Boolean)),
    ];
    const named = (verb: string, list: ToolItem[]) => {
      const names = files(list);
      if (names.length > 0 && names.length <= LABEL_FILES) return `${verb} ${names.join(", ")}`;
      const count = Math.max(names.length, list.length);
      return `${verb} ${count} file${count === 1 ? "" : "s"}`;
    };
    const counted = (verb: string, noun: string, n: number) => `${verb} ${n} ${noun}${n === 1 ? "" : "s"}`;
    const parts: string[] = [];
    const reads = of(["read"]);
    const edits = of(["edit", "write"]);
    const searches = of(["grep", "glob"]);
    const commands = of(["bash"]);
    if (reads.length) parts.push(named("read", reads));
    if (edits.length) parts.push(named("edited", edits));
    if (searches.length) parts.push(counted("searched", "pattern", searches.length));
    if (commands.length) parts.push(counted("ran", "command", commands.length));
    const known = new Set(["read", "edit", "write", "grep", "glob", "bash"]);
    for (const name of new Set(tools.filter((tool) => !known.has(tool.name)).map((tool) => tool.name))) parts.push(`used ${name}`);
    const text = parts.join(", ");
    return `${text.charAt(0).toUpperCase()}${text.slice(1)} (${tools.length})`;
  }

  function activityState(entries: Entry[]): "running" | "waiting" | "error" | "done" {
    const tools = toolsOf(entries);
    if (tools.some((tool) => tool.approval)) return "waiting";
    if (tools.some((tool) => tool.running) || (running && entries[entries.length - 1].index === items.length - 1))
      return "running";
    if (tools.some((tool) => tool.isError || tool.declined)) return "error";
    return "done";
  }

  // runs while a prompt does, paused while an approval waits on the user
  function syncClock() {
    const t = Date.now();
    const waiting = items.some((item) => item.kind === "tool" && item.approval);
    if (running && !waiting) {
      if (activeSince == null) activeSince = t;
    } else if (activeSince != null) {
      agentMs += t - activeSince;
      activeSince = null;
    }
    now = t;
  }

  $effect(() => {
    if (!running) return;
    const timer = setInterval(() => (now = Date.now()), CLOCK_TICK_MS);
    return () => clearInterval(timer);
  });

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
          if (CHANGING_TOOLS.has(tool.name)) void refreshChanges();
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
    syncClock();
    void scrollDown();
  }

  async function scrollDown() {
    await tick();
    if (container) container.scrollTop = container.scrollHeight;
  }

  async function refreshSessions() {
    if (!project.trim()) return;
    try {
      const listing = await code.listSessions(project);
      sessions = listing.sessions;
      projectPath = listing.project;
    } catch (e) {
      sessions = [];
      error = `${e}`;
    }
  }

  async function refreshChanges() {
    if (!sessionId) return;
    const id = sessionId;
    try {
      const listed = await code.listChanges(id);
      if (id === sessionId) changes = listed;
    } catch (e) {
      error = `${e}`;
    }
    if (selectedFile) await loadDiff(selectedFile);
  }

  async function loadDiff(path: string) {
    if (!sessionId) return;
    try {
      const diff = await code.fileDiff(sessionId, path);
      if (path === selectedFile) {
        fileDiff = diff;
        diffError = "";
      }
    } catch (e) {
      if (path === selectedFile) diffError = `${e}`;
    }
  }

  function selectFile(path: string) {
    if (path === selectedFile) return;
    selectedFile = path;
    fileDiff = null;
    diffError = "";
    void loadDiff(path);
  }

  // a tool's path argument as the change list keys it: relative inside the project, absolute outside
  function changeKey(path: string): string {
    if (projectPath && path.startsWith(`${projectPath}/`)) return path.slice(projectPath.length + 1);
    return path.replace(/^(\.\/)+/, "");
  }

  function resetFiles() {
    changes = [];
    selectedFile = null;
    fileDiff = null;
  }

  function startDrag(event: PointerEvent) {
    const handle = event.currentTarget as HTMLElement;
    handle.setPointerCapture(event.pointerId);
    const move = (e: PointerEvent) => {
      if (!split) return;
      const rect = split.getBoundingClientRect();
      diffShare = Math.min(MAX_DIFF_SHARE, Math.max(MIN_DIFF_SHARE, (rect.right - e.clientX) / rect.width));
    };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      store(DIFF_WIDTH_KEY, String(diffShare));
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  }

  async function open(id?: string) {
    try {
      const session = await code.openSession(project, modelStore.selectedModel, id);
      if (session.session_id !== sessionId) resetFiles();
      sessionId = session.session_id;
      projectPath = session.project;
      items = fromMessages(session.messages);
      labels = session.labels ?? {};
      usage = null;
      agentMs = 0;
      error = "";
      store(PROJECT_KEY, project);
      await refreshSessions();
      void refreshChanges();
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
      resetFiles();
      sessionId = demo.session_id;
      projectPath = demo.project;
      items = [];
      labels = {};
      usage = null;
      agentMs = 0;
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
    agentMs = 0;
    activeSince = null;
    syncClock();
    void scrollDown();
    try {
      await code.send(sessionId, modelStore.selectedModel, text, mode, thinking === "default" ? null : thinking === "on", onEvent);
    } catch (e) {
      error = `${e}`;
    } finally {
      running = false;
      for (const item of items) if (item.kind === "tool") item.running = false;
      syncClock();
      void refreshSessions();
      void refreshChanges();
      void describeActivity();
    }
  }

  // once a prompt is answered, so the labels never share the GPU with the agent's own decoding
  async function describeActivity() {
    if (!describe || !sessionId || running) return;
    const groups = blocks
      .filter((block) => block.kind === "activity")
      .map((block) => toolsOf(block.entries).map((tool) => tool.id))
      .filter((ids) => ids.length > 0 && !labels[ids[0]]);
    if (groups.length === 0) return;
    const id = sessionId;
    try {
      const described = await code.labelActivity(id, groups);
      if (id === sessionId) labels = { ...labels, ...described };
    } catch {
      // the calls' own summary stays
    }
  }

  function setDescribe(on: boolean) {
    describe = on;
    store(DESCRIBE_KEY, on ? "on" : "off");
    void describeActivity();
  }

  async function decide(tool: ToolItem, decision: Decision) {
    if (!sessionId || !tool.approval) return;
    const approvalId = tool.approval.id;
    tool.approval = null;
    if (decision === "deny") tool.declined = true;
    syncClock();
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
    if (available && project && !projectPath) void refreshSessions();
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
    <aside class="flex max-h-[45vh] shrink-0 flex-col gap-3 border-b border-gray-200 p-3 md:max-h-none md:w-72 md:border-r md:border-b-0 dark:border-gray-800">
      {#if hasRoot}
        <label class="text-xs font-medium text-gray-500 dark:text-gray-400" for="code-project">Project</label>
        <input
          id="code-project"
          class="min-w-0 rounded-md border border-gray-300 bg-white px-2 py-1 text-sm dark:border-gray-700 dark:bg-gray-900"
          placeholder="absolute path, e.g. {roots[0]}/project"
          bind:value={project}
          onchange={() => refreshSessions()}
        />
      {:else}
        <p class="text-xs text-gray-500 dark:text-gray-400">
          Start the server with <code>--code-root DIR</code> (repeat for more) to open your own projects here.
        </p>
      {/if}
      {#if projectPath}
        <p class="truncate text-xs text-gray-400" title={projectPath}>{projectPath}</p>
      {:else if hasRoot}
        <p class="text-xs text-gray-400">Projects under {roots.join(", ")}</p>
      {/if}
      {#if hasRoot}
        <button
          class="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          onclick={() => open()}
          disabled={running || !project.trim()}
        >
          New session
        </button>
      {/if}
      <button
        class="rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-100 disabled:opacity-50 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
        onclick={runDemo}
        disabled={running}
        title="Make a temp project, a small package whose tests fail, and ask the agent to fix it"
      >
        Run demo
      </button>
      <div class="flex rounded-md border border-gray-300 text-xs dark:border-gray-700" role="tablist">
        <button
          class="flex-1 rounded-l-md px-2 py-1 {sideTab === 'sessions' ? 'bg-gray-200 dark:bg-gray-800' : ''}"
          role="tab"
          aria-selected={sideTab === "sessions"}
          onclick={() => (sideTab = "sessions")}>Sessions</button
        >
        <button
          class="flex-1 rounded-r-md px-2 py-1 {sideTab === 'files' ? 'bg-gray-200 dark:bg-gray-800' : ''}"
          role="tab"
          aria-selected={sideTab === "files"}
          onclick={() => (sideTab = "files")}>Files{changes.length ? ` (${changes.length})` : ""}</button
        >
      </div>
      {#if sideTab === "files"}
        <CodeFiles {sessionId} {changes} selected={selectedFile} onSelect={selectFile} />
      {:else}
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
      {/if}
    </aside>

    <div class="flex min-h-0 min-w-0 flex-1 flex-col md:flex-row" bind:this={split}>
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
          <label class="flex items-center gap-1" title="The model says in a few words what each run of tool calls was for">
            <input type="checkbox" checked={describe} onchange={(e) => setDescribe(e.currentTarget.checked)} />
            Describe activity
          </label>
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
            {#snippet entryView(item: Item, index: number)}
              {#if item.kind === "user"}
                <div class="self-end rounded-2xl bg-blue-600 px-4 py-2 text-sm whitespace-pre-wrap text-white">{item.text}</div>
              {:else if item.kind === "reasoning"}
                <ThinkingBlock reasoning={item.text} streaming={running && index === items.length - 1} />
              {:else if item.kind === "text"}
                <div class="markdown-content text-sm text-gray-800 dark:text-gray-200">{@html renderMarkdown(item.text)}</div>
              {:else if item.kind === "notice"}
                <p class="text-xs text-gray-400">{item.text}</p>
              {:else if item.kind === "tool"}
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
                        {#if (item.name === "edit" || item.name === "write") && !item.isError && str(item.args.path)}
                          <button
                            class="shrink-0 rounded px-1.5 text-blue-600 hover:bg-blue-50 dark:text-blue-400 dark:hover:bg-blue-950/40"
                            onclick={(e) => {
                              e.preventDefault();
                              selectFile(changeKey(str(item.args.path)));
                            }}>diff</button
                          >
                        {/if}
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
            {/snippet}
            {#each blocks as block (block.kind === "one" ? `o${block.entry.index}` : `a${block.key}`)}
              {#if block.kind === "one"}
                {@render entryView(block.entry.item, block.entry.index)}
              {:else}
                {@const phase = activityState(block.entries)}
                <details
                  class="group/act rounded-lg border border-gray-200 dark:border-gray-800"
                  open={phase === "waiting"}
                  data-activity
                >
                  <summary class="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm text-gray-600 select-none hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200">
                    <svg class="h-3.5 w-3.5 shrink-0 transition-transform group-open/act:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                    </svg>
                    <span
                      class="h-2 w-2 shrink-0 rounded-full {phase === 'running'
                        ? 'animate-pulse bg-amber-400'
                        : phase === 'waiting'
                          ? 'bg-amber-500'
                          : phase === 'error'
                            ? 'bg-red-500'
                            : 'bg-green-500'}"
                    ></span>
                    <span class="truncate">{activityLabel(block.entries)}</span>
                  </summary>
                  <div class="flex flex-col gap-2 border-t border-gray-200 px-2 py-2 dark:border-gray-800">
                    {#each block.entries as entry (entry.index)}
                      {@render entryView(entry.item, entry.index)}
                    {/each}
                  </div>
                </details>
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
          {#if usage || agentSeconds > 0}
            <p class="mx-auto mt-1 max-w-3xl text-xs text-gray-400" data-agent-seconds={agentSeconds.toFixed(1)}>
              agent {agentSeconds.toFixed(1)} s{#if usage}, context {usage.total_tokens} tokens, {usage.prompt_tokens_details
                  ?.cached_tokens ?? 0} cached, prefill {Number(usage.total_prompt_time_sec ?? 0).toFixed(1)} s, {Number(
                  usage.avg_compl_tok_per_sec ?? 0,
                ).toFixed(1)} tok/s{/if}
            </p>
          {/if}
        </div>
      </section>
      {#if selectedFile}
        <div
          class="hidden w-1 shrink-0 cursor-col-resize bg-gray-200 hover:bg-blue-400 md:block dark:bg-gray-800"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the diff"
          onpointerdown={startDrag}
        ></div>
        <div
          class="flex h-1/2 min-h-0 min-w-0 shrink-0 flex-col border-t border-gray-200 md:h-auto md:basis-(--diff-share) md:border-t-0 dark:border-gray-800"
          style="--diff-share: {diffShare * 100}%"
        >
          <CodeDiff path={selectedFile} diff={fileDiff} error={diffError} onClose={() => (selectedFile = null)} />
        </div>
      {/if}
    </div>
  </div>
{/if}
