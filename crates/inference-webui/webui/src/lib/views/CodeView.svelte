<script lang="ts">
  import { onDestroy, tick } from "svelte";
  import CodeDiff from "../components/CodeDiff.svelte";
  import CodeFiles from "../components/CodeFiles.svelte";
  import ThinkingBlock from "../components/ThinkingBlock.svelte";
  import { modelStore } from "../stores/models.svelte";
  import { renderMarkdown } from "../utils/markdown";
  import { when } from "../utils/when";
  import * as code from "../services/code";
  import type {
    CodeEvent,
    CodeMessage,
    CodePart,
    CodeFolder,
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
  // the session last open, reopened when the tab comes back (or the page reloads) and attached to if still running
  const SESSION_KEY = "inference-rs-code-session";
  // the diff pane's share of the width, kept between these so neither side vanishes
  const MIN_DIFF_SHARE = 0.2;
  const MAX_DIFF_SHARE = 0.8;
  const DEFAULT_DIFF_SHARE = 0.5;
  // calls after which the project may have changed
  const CHANGING_TOOLS = new Set(["edit", "write", "bash"]);
  const SUMMARY_CHARS = 100;
  // how often the agent clock and the live stats redraw while a prompt runs
  const CLOCK_TICK_MS = 100;
  // streamed characters per token, for the live estimate until the server's count arrives
  const CHARS_PER_TOKEN = 4;
  // the share of the context past which the agent summarizes the history (the harness's COMPACT_AT)
  const COMPACT_SHARE = 0.8;
  // within this many pixels of the bottom the log keeps following new output
  const FOLLOW_PX = 48;
  const RING_RADIUS = 7;
  const RING_LENGTH = 2 * Math.PI * RING_RADIUS;
  // an activity line names this many files before it counts them instead
  const LABEL_FILES = 2;
  const COMPACTED_NOTICE = "The earlier history was summarized: the model now sees the summary and the messages after this point";
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
    images: string[];
  }
  type Item =
    | { kind: "reasoning" | "text" | "notice"; text: string }
    | { kind: "user"; text: string; images: string[] }
    | ({ kind: "compaction"; state: "running" | "done" | "stopped"; at: number | null } & CompactionRun)
    | ToolItem;
  // a compaction seen live: when it began and ended, its summary's first text, how much streamed, and each request's usage
  interface CompactionRun {
    startedAt: number | null;
    finishedAt: number | null;
    firstAt: number | null;
    chars: number;
    requests: Record<string, any>[];
  }
  const NO_RUN: CompactionRun = { startedAt: null, finishedAt: null, firstAt: null, chars: 0, requests: [] };
  // the model request under way: when it went, its estimated prompt, and what has streamed back since
  interface Round {
    start: number;
    promptTokens: number;
    firstAt: number | null;
    chars: number;
    // false once a late page joined past the prefill, which it didn't see
    prefillSeen: boolean;
  }

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
  // the model's words for what each activity line was for, by its first call, sent as each run closes
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
  let contextTokens = $state<number | null>(null);
  let contextEstimate = $state<number | null>(null);
  let round = $state<Round | null>(null);
  let live = $derived.by(() => {
    if (!round) return null;
    const tokens = Math.round(round.chars / CHARS_PER_TOKEN);
    const decoding = round.firstAt != null ? (now - round.firstAt) / 1000 : 0;
    return {
      context: round.promptTokens + tokens,
      prefill: round.prefillSeen ? ((round.firstAt ?? now) - round.start) / 1000 : null,
      tokPerSec: decoding > 0 ? tokens / decoding : 0,
    };
  });
  // what the context holds: live while a reply streams, the server's count after it, the harness's estimate before
  let contextUsed = $derived(live?.context ?? usage?.total_tokens ?? contextEstimate);
  let contextShare = $derived(contextTokens && contextUsed != null ? Math.min(1, contextUsed / contextTokens) : null);
  let follow = true;
  // a late page replays the run's events before it shows them live
  let replaying = false;
  let stream: AbortController | null = null;
  let pendingImages = $state<string[]>([]);
  let imageInput = $state<HTMLInputElement | null>(null);
  let canSeeImages = $derived(modelStore.selectedInputModalities.includes("vision"));
  // the session menu, from its corner button or a right-click
  let menu = $state<{ session: CodeSessionInfo; x: number; y: number } | null>(null);
  // the folder picker: the folder shown (null for the code roots) and its folders
  let browsing = $state<{ path: string | null; folders: CodeFolder[] } | null>(null);
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

  function textOf(content: CodeMessage["content"]): string {
    if (!Array.isArray(content)) return content ?? "";
    return content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("\n");
  }

  function fromMessages(
    messages: CodeMessage[],
    compactedBefore: number | null = null,
    compactedAt: number | null = null,
  ): Item[] {
    const out: Item[] = [];
    for (const [index, message] of messages.entries()) {
      if (index === compactedBefore) out.push({ kind: "compaction", state: "done", at: compactedAt, ...NO_RUN });
      if (message.role === "user") {
        const parts: CodePart[] = Array.isArray(message.content) ? message.content : [];
        // the turn that showed the model the images tool calls returned: they go under those calls
        const returned = parts.filter((part) => part.tool_call_id);
        if (returned.length) {
          for (const part of returned) {
            const tool = findTool(out, part.tool_call_id ?? "");
            if (!tool) continue;
            // an image whose file is gone comes back as a note saying so
            if (part.type === "image_url") tool.images.push(part.image_url.url);
            else tool.result = `${tool.result ?? ""}\n${part.text}`;
          }
          continue;
        }
        const images = parts.flatMap((part) => (part.type === "image_url" ? [part.image_url.url] : []));
        out.push({ kind: "user", text: textOf(message.content), images });
      } else if (message.role === "assistant") {
        if (message.reasoning_content?.trim()) out.push({ kind: "reasoning", text: message.reasoning_content.trim() });
        const text = textOf(message.content).trim();
        if (text) out.push({ kind: "text", text });
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
            images: [],
          });
        }
      } else if (message.role === "tool") {
        const tool = findTool(out, message.tool_call_id ?? "");
        if (tool) {
          tool.result = textOf(message.content);
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
      case "prompt":
        items.push({ kind: "user", text: event.text, images: event.images ?? [] });
        follow = true;
        break;
      case "context":
        contextTokens = event.tokens;
        break;
      case "caught_up":
        replaying = false;
        agentMs = 0;
        activeSince = Date.now() - event.elapsed_ms;
        // the replayed output arrived at once: rate what streams from here, with what came before in the prompt
        if (round) {
          round.promptTokens += Math.round(round.chars / CHARS_PER_TOKEN);
          round.chars = 0;
          round.start = Date.now();
          if (round.firstAt != null) {
            round.firstAt = Date.now();
            round.prefillSeen = false;
          }
        }
        break;
      case "round_start":
        round = { start: Date.now(), promptTokens: event.prompt_tokens, firstAt: null, chars: 0, prefillSeen: true };
        break;
      case "text":
      case "reasoning":
        if (round) {
          round.firstAt ??= Date.now();
          round.chars += event.delta.length;
        }
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
          images: [],
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
            images: [],
          });
        }
        break;
      }
      case "tool_end": {
        const tool = findTool(items, event.id);
        if (tool) {
          tool.result = event.text;
          tool.isError = event.is_error;
          tool.images = event.images ?? [];
          tool.running = false;
          if (CHANGING_TOOLS.has(tool.name)) void refreshChanges();
        }
        break;
      }
      case "usage":
        usage = event.usage;
        round = null;
        break;
      case "label":
        labels[event.id] = event.label;
        break;
      case "compacting":
        items.push({ kind: "compaction", state: "running", at: null, ...NO_RUN, requests: [], startedAt: Date.now() });
        break;
      case "summary_text": {
        const item = runningCompaction();
        if (item) {
          item.firstAt ??= Date.now();
          item.chars += event.chars;
        }
        break;
      }
      case "summary_usage":
        runningCompaction()?.requests.push(event.usage);
        break;
      case "compacted": {
        const item = runningCompaction();
        if (item) {
          item.state = "done";
          item.at = event.at;
          item.finishedAt = Date.now();
        }
        usage = null;
        contextEstimate = event.context_estimate;
        break;
      }
      case "done":
        if (event.end === "nothing_to_compact")
          items.push({ kind: "notice", text: "Nothing before the latest prompt to summarize" });
        if (event.end === "interrupted") items.push({ kind: "notice", text: "Interrupted" });
        if (event.end === "round_limit") items.push({ kind: "notice", text: "Stopped at the round limit" });
        if (event.end === "reply_limit")
          items.push({ kind: "notice", text: "A reply hit the token limit (the model was likely repeating itself)" });
        break;
      case "error":
        error = event.message;
        break;
    }
    if (replaying) return;
    syncClock();
    void scrollDown();
  }

  function runningCompaction() {
    const item = items.findLast((item) => item.kind === "compaction" && item.state === "running");
    return item?.kind === "compaction" ? item : undefined;
  }

  // a compaction's figures: time so far (or taken), and the summary requests' reading and writing rates
  function compactionStats(item: CompactionRun): string {
    const parts: string[] = [];
    if (item.startedAt != null) parts.push(clock(((item.finishedAt ?? now) - item.startedAt) / 1000));
    const done = item.requests;
    if (done.length) {
      const sum = (key: string) => done.reduce((total, usage) => total + Number(usage[key] ?? 0), 0);
      const cached = done.reduce((total, usage) => total + Number(usage.prompt_tokens_details?.cached_tokens ?? 0), 0);
      const read = sum("prompt_tokens") - cached;
      const written = sum("completion_tokens");
      const readTime = sum("total_prompt_time_sec");
      const writeTime = sum("total_completion_time_sec");
      parts.push(`${done.length} request${done.length === 1 ? "" : "s"}`);
      if (readTime > 0) parts.push(`read ${count(read)} tokens at ${count(Math.round(read / readTime))} tok/s`);
      if (writeTime > 0) parts.push(`wrote ${count(written)} at ${(written / writeTime).toFixed(1)} tok/s`);
    }
    if (item.finishedAt == null && item.firstAt != null && now > item.firstAt) {
      parts.push(`~${(item.chars / CHARS_PER_TOKEN / ((now - item.firstAt) / 1000)).toFixed(1)} tok/s`);
    }
    return parts.join(" · ");
  }

  // prompt tokens the server read (not served from its cache) per second of prefill
  function prefillRate(usage: Record<string, any>): number | null {
    const time = Number(usage.total_prompt_time_sec ?? 0);
    const read = Number(usage.prompt_tokens ?? 0) - Number(usage.prompt_tokens_details?.cached_tokens ?? 0);
    return time > 0 && read > 0 ? read / time : null;
  }

  // draft tokens accepted per speculative step, and the share of proposed drafts that were
  function drafts(usage: Record<string, any>): string | null {
    const spec = usage.speculative;
    if (!spec?.steps) return null;
    const share = spec.proposed_tokens ? Math.round((100 * spec.accepted_tokens) / spec.proposed_tokens) : 0;
    return `drafts ${(spec.accepted_tokens / spec.steps).toFixed(2)}/step (${share}%)`;
  }

  async function scrollDown() {
    if (!follow) return;
    await tick();
    if (container && follow) container.scrollTop = container.scrollHeight;
  }

  // scrolling up stops the log following new output; back near the bottom it follows again
  function onScroll() {
    if (container) follow = container.scrollHeight - container.scrollTop - container.clientHeight < FOLLOW_PX;
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

  // `remembered`: the session the tab had open last, which may since have gone
  async function open(id?: string, remembered = false) {
    try {
      const session = await code.openSession(project, modelStore.selectedModel, id);
      if (session.session_id !== sessionId) resetFiles();
      sessionId = session.session_id;
      projectPath = session.project;
      items = fromMessages(session.messages, session.compacted_before, session.compacted_at);
      labels = session.labels ?? {};
      usage = session.usage;
      round = null;
      contextTokens = session.context_tokens;
      contextEstimate = session.context_estimate;
      agentMs = 0;
      error = "";
      store(PROJECT_KEY, project);
      store(SESSION_KEY, session.session_id);
      follow = true;
      await refreshSessions();
      void refreshChanges();
      void scrollDown();
      if (session.running) void run((signal) => code.attach(session.session_id, onEvent, signal), true);
    } catch (e) {
      if (remembered) store(SESSION_KEY, "");
      else error = `${e}`;
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
      store(SESSION_KEY, demo.session_id);
      items = [];
      labels = {};
      usage = null;
      contextEstimate = null;
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
    const images = canSeeImages ? pendingImages : [];
    if ((!text && !images.length) || running) return;
    if (!sessionId) await open();
    if (!sessionId) return;
    const id = sessionId;
    prompt = "";
    pendingImages = [];
    error = "";
    const options = {
      model: modelStore.selectedModel,
      mode,
      enableThinking: thinking === "default" ? null : thinking === "on",
      describeActivity: describe,
      images,
    };
    await run((signal) => code.send(id, text, options, onEvent, signal), false);
  }

  // follows a prompt's events to its end; leaving the tab aborts only the stream, never the prompt
  async function run(listen: (signal: AbortSignal) => Promise<void>, late: boolean) {
    stream?.abort();
    const controller = new AbortController();
    stream = controller;
    running = true;
    replaying = late;
    agentMs = 0;
    activeSince = null;
    if (!late) syncClock();
    try {
      await listen(controller.signal);
    } catch (e) {
      if (!controller.signal.aborted) error = `${e}`;
    } finally {
      if (stream === controller) {
        stream = null;
        running = false;
        replaying = false;
        round = null;
        for (const item of items) {
          if (item.kind === "tool") item.running = false;
          if (item.kind === "compaction" && item.state === "running") item.state = "stopped";
        }
        syncClock();
        void refreshSessions();
        void refreshChanges();
        void scrollDown();
      }
    }
  }

  function readImage(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }

  async function addImages(files: Iterable<File>) {
    for (const file of files) {
      if (!file.type.startsWith("image/")) continue;
      try {
        pendingImages.push(await readImage(file));
      } catch (e) {
        error = `${e}`;
      }
    }
  }

  function onPaste(event: ClipboardEvent) {
    if (!canSeeImages) return;
    const files = [...(event.clipboardData?.files ?? [])].filter((file) => file.type.startsWith("image/"));
    if (!files.length) return;
    event.preventDefault();
    void addImages(files);
  }

  function onDrop(event: DragEvent) {
    if (!canSeeImages || !event.dataTransfer?.files.length) return;
    event.preventDefault();
    void addImages(event.dataTransfer.files);
  }

  // HH:MM:SS.S
  function clock(seconds: number): string {
    const tenths = Math.floor(seconds * 10);
    const pad = (n: number) => String(n).padStart(2, "0");
    const hours = Math.floor(tenths / 36000);
    const minutes = Math.floor(tenths / 600) % 60;
    return `${pad(hours)}:${pad(minutes)}:${((tenths % 600) / 10).toFixed(1).padStart(4, "0")}`;
  }

  function count(tokens: number): string {
    return tokens.toLocaleString();
  }

  function showMenu(event: MouseEvent, session: CodeSessionInfo) {
    event.preventDefault();
    event.stopPropagation();
    menu = { session, x: event.clientX, y: event.clientY };
  }

  async function deleteSession(session: CodeSessionInfo) {
    menu = null;
    const title = session.title || "(empty session)";
    if (!confirm(`Delete the session "${title}"? Its history and the copies it kept for diffs go too; the project's files stay as they are.`))
      return;
    try {
      await code.deleteSession(session.id);
      if (session.id === sessionId) {
        sessionId = null;
        items = [];
        usage = null;
        contextEstimate = null;
        resetFiles();
        store(SESSION_KEY, "");
      }
    } catch (e) {
      error = `${e}`;
    }
    await refreshSessions();
  }

  async function browse(path: string | null) {
    try {
      const listing = await code.listFolders(path ?? "");
      browsing = { path: listing.path, folders: listing.folders };
    } catch (e) {
      error = `${e}`;
    }
  }

  // a code root's parent is the roots list
  function parentOf(path: string): string | null {
    if (roots.includes(path)) return null;
    return path.slice(0, path.lastIndexOf("/")) || null;
  }

  function useFolder(path: string) {
    project = path;
    projectPath = "";
    browsing = null;
    store(PROJECT_KEY, path);
    void refreshSessions();
  }

  function setDescribe(on: boolean) {
    describe = on;
    store(DESCRIBE_KEY, on ? "on" : "off");
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

  let restored = false;
  // once the page knows it may use the tab (capabilities arrive after mount on a fresh load)
  $effect(() => {
    if (!available || restored) return;
    restored = true;
    void refreshSessions();
    const last = stored(SESSION_KEY, "");
    if (last && project.trim()) void open(last, true);
  });

  onDestroy(() => stream?.abort());

  $effect(() => {
    if (!menu) return;
    const close = () => (menu = null);
    window.addEventListener("click", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", close);
    };
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
        <button
          class="-mt-1 self-start text-xs text-blue-600 hover:underline dark:text-blue-400"
          onclick={() => (browsing ? (browsing = null) : browse(projectPath || null))}
          >{browsing ? "Close the folder list" : "Browse folders..."}</button
        >
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
      {#if browsing}
        <div class="flex min-h-0 flex-1 flex-col gap-1" data-browser>
          <div class="flex items-center gap-1 text-xs text-gray-500">
            {#if browsing.path}
              <button
                class="rounded px-1 hover:bg-gray-100 dark:hover:bg-gray-800"
                onclick={() => browsing && browse(parentOf(browsing.path ?? ""))}
                aria-label="Up a folder"
                title="Up a folder">&uarr;</button
              >
              <span class="min-w-0 flex-1 truncate" title={browsing.path}>{browsing.path}</span>
              <button
                class="rounded bg-blue-600 px-2 py-0.5 text-white hover:bg-blue-700"
                onclick={() => browsing?.path && useFolder(browsing.path)}>Use this folder</button
              >
            {:else}
              <span>Code roots</span>
            {/if}
          </div>
          <div class="min-h-0 flex-1 overflow-y-auto">
            {#each browsing.folders as folder (folder.path)}
              <div class="flex items-center gap-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-900">
                <button class="min-w-0 flex-1 truncate px-2 py-1 text-left text-sm" onclick={() => browse(folder.path)} title={folder.path}>
                  {folder.name}
                </button>
                {#if folder.sessions}
                  <button
                    class="shrink-0 rounded px-1.5 text-xs text-blue-600 hover:bg-blue-50 dark:text-blue-400 dark:hover:bg-blue-950/40"
                    onclick={() => useFolder(folder.path)}
                    title="Open this project's sessions"
                    >{folder.sessions} session{folder.sessions === 1 ? "" : "s"}</button
                  >
                {/if}
                {#if folder.sessions_below}
                  <span class="shrink-0 pr-2 text-xs text-gray-400" title="Sessions in folders below this one">+{folder.sessions_below} below</span>
                {/if}
              </div>
            {:else}
              <p class="px-2 text-xs text-gray-400">No folders here.</p>
            {/each}
          </div>
        </div>
      {:else if sideTab === "files"}
        <CodeFiles {sessionId} {changes} selected={selectedFile} onSelect={selectFile} />
      {:else}
        <div class="min-h-0 flex-1 overflow-y-auto">
          <!-- a session nothing was sent in is only worth listing while it's open -->
          {#each sessions.filter((s) => s.title || s.id === sessionId) as session (session.id)}
            <div class="group/session relative mb-1" data-session={session.id}>
              <button
                class="block w-full rounded-md py-1.5 pr-7 pl-2 text-left text-sm {session.id === sessionId
                  ? 'bg-gray-200 dark:bg-gray-800'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-900'}"
                onclick={() => open(session.id)}
                oncontextmenu={(e) => showMenu(e, session)}
                disabled={running}
              >
                <span class="block truncate text-gray-800 dark:text-gray-200">{session.title || "(empty session)"}</span>
                <span class="block text-xs text-gray-400">{relativeTime(session.modified)}</span>
              </button>
              <button
                class="absolute top-1.5 right-1 rounded px-1 text-gray-400 opacity-0 group-hover/session:opacity-100 hover:bg-gray-300 hover:text-gray-700 focus:opacity-100 dark:hover:bg-gray-700 dark:hover:text-gray-200"
                onclick={(e) => showMenu(e, session)}
                aria-label="Session actions"
                title="Session actions">&#8943;</button
              >
            </div>
          {:else}
            <p class="text-xs text-gray-400">No sessions in this project yet.</p>
          {/each}
        </div>
      {/if}
    </aside>
    {#if menu}
      <div
        class="fixed z-50 min-w-36 rounded-md border border-gray-200 bg-white py-1 text-sm shadow-lg dark:border-gray-700 dark:bg-gray-900"
        style="left: {menu.x}px; top: {menu.y}px"
        role="menu"
      >
        <button
          class="block w-full px-3 py-1.5 text-left text-red-600 hover:bg-red-50 disabled:opacity-50 dark:hover:bg-red-950/40"
          role="menuitem"
          disabled={running && menu.session.id === sessionId}
          onclick={() => menu && deleteSession(menu.session)}>Delete session</button
        >
      </div>
    {/if}

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

        <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4" bind:this={container} onscroll={onScroll}>
          <div class="mx-auto flex max-w-3xl flex-col gap-3">
            {#if items.length === 0}
              <p class="mt-8 text-center text-sm text-gray-400">
                Ask for a change in {projectPath || "the project"}: the agent reads files, edits them and runs commands
                there.
              </p>
            {/if}
            {#snippet entryView(item: Item, index: number)}
              {#if item.kind === "user"}
                {#if item.images.length}
                  <div class="flex flex-wrap justify-end gap-2 self-end">
                    {#each item.images as image, i (i)}
                      <img src={image} alt="attached" class="max-h-40 max-w-60 rounded-lg border border-gray-200 object-contain dark:border-gray-800" />
                    {/each}
                  </div>
                {/if}
                {#if item.text}
                  <div class="self-end rounded-2xl bg-blue-600 px-4 py-2 text-sm whitespace-pre-wrap text-white">{item.text}</div>
                {/if}
              {:else if item.kind === "compaction"}
                <div class="flex items-center gap-2 text-xs text-gray-400" data-compaction={item.state} title={COMPACTED_NOTICE}>
                  <span class="h-px flex-1 bg-gray-200 dark:bg-gray-800"></span>
                  {#if item.state === "running"}
                    <svg class="h-3.5 w-3.5 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                    </svg>
                    <span class="animate-pulse">Summarizing the earlier history to fit the context...</span>
                    <span class="tabular-nums">{compactionStats(item)}</span>
                  {:else if item.state === "stopped"}
                    <span>Summarizing stopped; the history is as it was</span>
                  {:else}
                    <span>History summarized{item.at ? ` ${when(new Date(item.at * 1000).toISOString())}` : ""}</span>
                    {#if item.startedAt != null}<span class="tabular-nums">· {compactionStats(item)}</span>{/if}
                  {/if}
                  <span class="h-px flex-1 bg-gray-200 dark:bg-gray-800"></span>
                </div>
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
                      {:else if item.images.length}
                        <span class="ml-auto shrink-0 text-gray-400">{item.images.length === 1 ? "image" : `${item.images.length} images`}</span>
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
                    {#if item.images.length}
                      <div class="flex flex-wrap gap-2 border-t border-gray-200 px-3 py-2 dark:border-gray-800">
                        {#each item.images as image, i (i)}
                          <img src={image} alt="returned by {item.name}" class="max-h-80 max-w-full rounded border border-gray-200 object-contain dark:border-gray-800" />
                        {/each}
                      </div>
                    {/if}
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

        <div
          class="shrink-0 border-t border-gray-200 px-4 py-3 dark:border-gray-800"
          role="group"
          ondragover={(e) => canSeeImages && e.preventDefault()}
          ondrop={onDrop}
        >
          {#if pendingImages.length}
            <div class="mx-auto mb-2 flex max-w-3xl flex-wrap gap-2">
              {#each pendingImages as image, i (i)}
                <div class="relative">
                  <img src={image} alt="to send" class="h-16 w-16 rounded-md border border-gray-200 object-cover dark:border-gray-700" />
                  <button
                    class="absolute -top-1.5 -right-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-gray-700 text-xs text-white hover:bg-red-600"
                    onclick={() => pendingImages.splice(i, 1)}
                    aria-label="Remove image">&times;</button
                  >
                </div>
              {/each}
            </div>
          {/if}
          <div class="mx-auto flex max-w-3xl items-end gap-2">
            {#if canSeeImages}
              <input
                type="file"
                accept="image/*"
                multiple
                class="hidden"
                bind:this={imageInput}
                onchange={(e) => {
                  void addImages(e.currentTarget.files ?? []);
                  e.currentTarget.value = "";
                }}
              />
              <button
                class="rounded-xl border border-gray-300 px-3 py-2.5 text-gray-500 hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-800"
                onclick={() => imageInput?.click()}
                title="Attach images (or paste or drop them here)"
                aria-label="Attach images"
              >
                <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                </svg>
              </button>
            {/if}
            <textarea
              class="min-w-0 flex-1 resize-none rounded-xl border border-gray-300 bg-gray-50 px-4 py-2.5 text-sm outline-none focus:border-blue-400 focus:bg-white focus:ring-1 focus:ring-blue-400 dark:border-gray-600 dark:bg-gray-800 dark:focus:bg-gray-900"
              rows="2"
              placeholder={`Describe a change (Enter sends, Shift+Enter for a new line; ${code.COMPACT_COMMAND} summarizes the history)`}
              bind:value={prompt}
              onkeydown={onKeydown}
              onpaste={onPaste}
            ></textarea>
            {#if running}
              <button class="rounded-xl bg-red-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-700" onclick={stop}>Stop</button>
            {:else}
              <button
                class="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                onclick={submit}
                disabled={!prompt.trim() && !(canSeeImages && pendingImages.length)}
              >
                Send
              </button>
            {/if}
          </div>
          {#if sessionId}
            <div class="mx-auto mt-1 flex max-w-3xl flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-400 tabular-nums">
              {#if contextTokens && contextShare != null && contextUsed != null}
                <span
                  class="flex items-center gap-1.5"
                  data-context-used={contextUsed}
                  title="The model's context: what this session holds of it (summarized at {COMPACT_SHARE * 100}%)"
                >
                  <svg class="h-4 w-4 -rotate-90" viewBox="0 0 18 18" aria-hidden="true">
                    <circle cx="9" cy="9" r={RING_RADIUS} fill="none" stroke-width="2.5" class="stroke-gray-200 dark:stroke-gray-700" />
                    <circle
                      cx="9"
                      cy="9"
                      r={RING_RADIUS}
                      fill="none"
                      stroke-width="2.5"
                      stroke-linecap="round"
                      stroke-dasharray={RING_LENGTH}
                      stroke-dashoffset={RING_LENGTH * (1 - contextShare)}
                      class={contextShare >= COMPACT_SHARE ? "stroke-amber-500" : "stroke-blue-500"}
                    />
                  </svg>
                  {live || !usage ? "~" : ""}{count(contextUsed)} / {count(contextTokens)} tokens ({Math.round(contextShare * 100)}%)
                </span>
              {/if}
              {#if agentSeconds > 0}
                <span data-agent-seconds={agentSeconds.toFixed(1)}>agent {clock(agentSeconds)}</span>
              {/if}
              {#if live}
                {#if live.prefill != null}<span>prefill {live.prefill.toFixed(1)} s</span>{/if}
                {#if live.tokPerSec > 0}<span>~{live.tokPerSec.toFixed(1)} tok/s</span>{/if}
                {#if usage}<span>{count(usage.prompt_tokens_details?.cached_tokens ?? 0)} cached (last request)</span>{/if}
              {:else if usage}
                {@const rate = prefillRate(usage)}
                {@const drafted = drafts(usage)}
                <span>{count(usage.prompt_tokens_details?.cached_tokens ?? 0)} cached</span>
                <span>prefill {Number(usage.total_prompt_time_sec ?? 0).toFixed(1)} s{rate ? ` (${count(Math.round(rate))} tok/s)` : ""}</span>
                <span>{Number(usage.avg_compl_tok_per_sec ?? 0).toFixed(1)} tok/s</span>
                {#if drafted}<span title="Speculative decoding: draft tokens the model accepted per step, and the share of drafts accepted">{drafted}</span>{/if}
              {/if}
            </div>
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
