<script lang="ts">
  import { untrack } from "svelte";
  import { SvelteSet } from "svelte/reactivity";
  import * as code from "../services/code";
  import type { FileChange, TreeEntry } from "../services/code";

  interface Props {
    sessionId: string | null;
    changes: FileChange[];
    selected: string | null;
    onSelect: (path: string) => void;
  }
  let { sessionId, changes, selected, onSelect }: Props = $props();

  const STATUS_MARK: Record<string, string> = { added: "A", modified: "M", deleted: "D", no_original: "?" };
  const STATUS_TITLE: Record<string, string> = {
    added: "added",
    modified: "modified",
    deleted: "deleted",
    no_original: "changed by a command, with no earlier copy",
  };

  let view = $state<"changed" | "all">("changed");
  let listings = $state<Record<string, TreeEntry[]>>({});
  let error = $state("");
  // changed view: everything starts open; full tree: everything starts closed
  const collapsed = new SvelteSet<string>();
  const expanded = new SvelteSet<string>();

  let inside = $derived(changes.filter((c) => !c.path.startsWith("/")));
  let outside = $derived(changes.filter((c) => c.path.startsWith("/")));
  let byPath = $derived(new Map(changes.map((c) => [c.path, c])));
  let totals = $derived(
    changes.reduce((sum, c) => ({ added: sum.added + c.added, removed: sum.removed + c.removed }), { added: 0, removed: 0 }),
  );

  $effect(() => {
    // a different session is a different project
    void sessionId;
    listings = {};
    expanded.clear();
    collapsed.clear();
  });

  $effect(() => {
    if (view === "all" && sessionId && !listings[""]) void load("");
  });

  $effect(() => {
    // new changes may be new files, so the open directories are listed again
    void changes;
    untrack(() => {
      if (view !== "all") return;
      for (const dir of Object.keys(listings)) void load(dir);
    });
  });

  function join(dir: string, name: string): string {
    return dir ? `${dir}/${name}` : name;
  }

  async function load(dir: string) {
    if (!sessionId) return;
    try {
      listings[dir] = await code.listDir(sessionId, dir);
      error = "";
    } catch (e) {
      error = `${e}`;
    }
  }

  function changedEntries(dir: string): TreeEntry[] {
    const prefix = dir ? `${dir}/` : "";
    const seen = new Map<string, boolean>();
    for (const change of inside) {
      if (!change.path.startsWith(prefix)) continue;
      const rest = change.path.slice(prefix.length);
      const slash = rest.indexOf("/");
      const name = slash < 0 ? rest : rest.slice(0, slash);
      seen.set(name, (seen.get(name) ?? false) || slash >= 0);
    }
    return [...seen]
      .map(([name, dir]) => ({ name, dir }))
      .sort((a, b) => Number(b.dir) - Number(a.dir) || a.name.localeCompare(b.name));
  }

  function dirTotals(dir: string): { added: number; removed: number; files: number } {
    const prefix = `${dir}/`;
    let added = 0;
    let removed = 0;
    let files = 0;
    for (const change of inside) {
      if (!change.path.startsWith(prefix)) continue;
      added += change.added;
      removed += change.removed;
      files += 1;
    }
    return { added, removed, files };
  }

  function isOpen(path: string): boolean {
    return view === "changed" ? !collapsed.has(path) : expanded.has(path);
  }

  function toggle(path: string) {
    if (view === "changed") {
      if (collapsed.has(path)) collapsed.delete(path);
      else collapsed.add(path);
      return;
    }
    if (expanded.has(path)) {
      expanded.delete(path);
    } else {
      expanded.add(path);
      if (!listings[path]) void load(path);
    }
  }

  function entries(dir: string): TreeEntry[] | undefined {
    return view === "changed" ? changedEntries(dir) : listings[dir];
  }
</script>

{#snippet counts(added: number, removed: number, binary = false)}
  {#if binary}
    <span class="shrink-0 text-gray-400">bin</span>
  {:else}
    {#if added}<span class="shrink-0 text-green-600 dark:text-green-400">+{added}</span>{/if}
    {#if removed}<span class="shrink-0 text-red-600 dark:text-red-400">-{removed}</span>{/if}
  {/if}
{/snippet}

{#snippet node(dir: string, depth: number)}
  {#each entries(dir) ?? [] as entry (entry.name)}
    {@const path = join(dir, entry.name)}
    {#if entry.dir}
      {@const sum = dirTotals(path)}
      <button
        class="flex w-full items-center gap-1 rounded px-1 py-0.5 text-left hover:bg-gray-100 dark:hover:bg-gray-900"
        style="padding-left: {depth * 12 + 4}px"
        onclick={() => toggle(path)}
      >
        <span class="w-3 shrink-0 text-gray-400">{isOpen(path) ? "v" : ">"}</span>
        <span class="truncate text-gray-700 dark:text-gray-300">{entry.name}</span>
        <span class="ml-auto flex gap-1">{@render counts(sum.added, sum.removed)}</span>
      </button>
      {#if isOpen(path)}
        {#if view === "all" && !listings[path]}
          <p class="text-gray-400" style="padding-left: {depth * 12 + 20}px">...</p>
        {:else}
          {@render node(path, depth + 1)}
        {/if}
      {/if}
    {:else}
      {@const change = byPath.get(path)}
      <button
        class="flex w-full items-center gap-1 rounded px-1 py-0.5 text-left {selected === path
          ? 'bg-blue-100 dark:bg-blue-950/50'
          : 'hover:bg-gray-100 dark:hover:bg-gray-900'}"
        style="padding-left: {depth * 12 + 16}px"
        onclick={() => onSelect(path)}
        title={change ? `${path} (${STATUS_TITLE[change.status]})` : path}
      >
        <span
          class="truncate {change?.status === 'deleted'
            ? 'text-red-600 line-through dark:text-red-400'
            : change
              ? 'text-gray-900 dark:text-gray-100'
              : 'text-gray-600 dark:text-gray-400'}">{entry.name}</span
        >
        {#if change}
          <span class="ml-auto flex gap-1">
            {@render counts(change.added, change.removed, change.binary)}
            <span class="w-3 shrink-0 text-center text-amber-600 dark:text-amber-400">{STATUS_MARK[change.status]}</span>
          </span>
        {/if}
      </button>
    {/if}
  {/each}
{/snippet}

<div class="flex min-h-0 flex-1 flex-col gap-2 text-xs">
  <div class="flex items-center gap-1">
    <div class="flex rounded-md border border-gray-300 dark:border-gray-700" role="group" aria-label="Files shown">
      <button
        class="rounded-l-md px-2 py-0.5 {view === 'changed' ? 'bg-gray-200 dark:bg-gray-800' : ''}"
        onclick={() => (view = "changed")}>Changed</button
      >
      <button
        class="rounded-r-md px-2 py-0.5 {view === 'all' ? 'bg-gray-200 dark:bg-gray-800' : ''}"
        onclick={() => (view = "all")}>All files</button
      >
    </div>
    <span class="ml-auto flex gap-1 font-mono">
      {#if changes.length}
        <span class="text-gray-500">{changes.length} {changes.length === 1 ? "file" : "files"}</span>
        {@render counts(totals.added, totals.removed)}
      {/if}
    </span>
  </div>
  <div class="min-h-0 flex-1 overflow-y-auto font-mono" data-testid="code-files">
    {#if !sessionId}
      <p class="font-sans text-gray-400">Open a session to see its project's files.</p>
    {:else if view === "changed" && changes.length === 0}
      <p class="font-sans text-gray-400">No changes in this session yet.</p>
    {:else}
      {@render node("", 0)}
      {#if view === "changed" && outside.length}
        <p class="mt-2 mb-1 font-sans text-gray-500">Outside the project</p>
        {#each outside as change (change.path)}
          <button
            class="flex w-full items-center gap-1 rounded px-1 py-0.5 text-left {selected === change.path
              ? 'bg-blue-100 dark:bg-blue-950/50'
              : 'hover:bg-gray-100 dark:hover:bg-gray-900'}"
            onclick={() => onSelect(change.path)}
            title={change.path}
          >
            <span class="truncate" style="direction: rtl">{change.path}</span>
            <span class="ml-auto flex gap-1">{@render counts(change.added, change.removed, change.binary)}</span>
          </button>
        {/each}
      {/if}
    {/if}
    {#if error}
      <p class="mt-2 font-sans text-red-600">{error}</p>
    {/if}
  </div>
</div>
