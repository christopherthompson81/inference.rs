<script lang="ts">
  export interface HistoryEntry {
    id: string;
    title: string;
    detail: string;
  }

  // `busy` holds the history still while the view runs a job; `onRename` adds a rename button to each entry
  let {
    entries,
    active,
    newLabel,
    onNew,
    onOpen,
    onDelete,
    onRename,
    busy = false,
    error = "",
    empty,
  }: {
    entries: HistoryEntry[];
    active: string | null;
    newLabel: string;
    onNew: () => void;
    onOpen: (id: string) => void;
    onDelete: (id: string) => void;
    onRename?: (id: string, title: string) => void;
    busy?: boolean;
    error?: string;
    empty: string;
  } = $props();

  let editing = $state<string | null>(null);
  let draft = $state("");

  function rename(id: string) {
    if (editing !== id) return;
    editing = null;
    if (draft.trim()) onRename?.(id, draft.trim());
  }

  const icon = "rounded p-0.5 text-gray-400 dark:text-gray-500";
</script>

<aside
  class="flex max-h-48 shrink-0 flex-col border-b border-gray-200 bg-gray-50 md:max-h-none md:w-64 md:border-r md:border-b-0 dark:border-gray-800 dark:bg-gray-900"
  aria-label="History"
>
  <div class="flex items-center justify-between gap-2 px-3 pt-3 pb-1">
    <span class="text-xs font-medium text-gray-500">History</span>
    <button
      class="rounded-md border border-gray-300 px-2 py-0.5 text-xs text-gray-700 hover:bg-gray-100 disabled:opacity-50 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
      disabled={busy}
      onclick={onNew}
    >
      {newLabel}
    </button>
  </div>
  {#if error}
    <div class="px-3 text-xs text-red-600">{error}</div>
  {/if}
  <div class="min-h-0 flex-1 overflow-y-auto p-2">
    {#each entries as entry (entry.id)}
      {@const current = active === entry.id}
      <div
        class="group flex items-center rounded-lg px-2 py-1.5 text-sm {current
          ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200'
          : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'}"
      >
        {#if editing === entry.id}
          <!-- svelte-ignore a11y_autofocus -->
          <input
            class="min-w-0 flex-1 rounded border border-blue-300 bg-white px-1.5 py-0.5 text-sm dark:border-blue-600 dark:bg-gray-800"
            aria-label="Title"
            autofocus
            bind:value={draft}
            onkeydown={(e) => {
              if (e.key === "Enter") rename(entry.id);
              if (e.key === "Escape") editing = null;
            }}
            onblur={() => rename(entry.id)}
          />
        {:else}
          <button class="flex min-w-0 flex-1 flex-col items-start text-left disabled:opacity-60" disabled={busy} onclick={() => onOpen(entry.id)}>
            <span class="w-full truncate">{entry.title || "Untitled"}</span>
            <span class="w-full truncate font-mono text-[10.5px] {current ? 'text-blue-600 dark:text-blue-300' : 'text-gray-400 dark:text-gray-500'}">
              {entry.detail}
            </span>
          </button>
          <div class="flex shrink-0 opacity-0 group-hover:opacity-100 focus-within:opacity-100">
            {#if onRename}
              <button
                class="{icon} hover:text-gray-600 dark:hover:text-gray-300"
                onclick={() => ((editing = entry.id), (draft = entry.title))}
                title="Rename"
                aria-label="Rename"
              >
                <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </button>
            {/if}
            <button
              class="{icon} hover:text-red-500 dark:hover:text-red-400"
              disabled={busy && current}
              onclick={() => onDelete(entry.id)}
              title="Delete"
              aria-label="Delete"
            >
              <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        {/if}
      </div>
    {:else}
      <div class="px-2 text-xs text-gray-400">{empty}</div>
    {/each}
  </div>
</aside>
