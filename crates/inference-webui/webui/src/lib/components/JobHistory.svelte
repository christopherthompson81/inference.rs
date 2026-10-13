<script lang="ts">
  import { onMount } from "svelte";
  import { jobsStore } from "../stores/jobs.svelte";
  import type { Job, JobKind } from "../services/jobs";

  // `busy` holds the history still while the view runs a job
  let { kind, onOpen, busy = false }: { kind: JobKind; onOpen: (job: Job) => void; busy?: boolean } = $props();
  let error = $state("");

  // the entry the tab showed last, reopened once the list is in; one deleted, or another owner's, is skipped
  onMount(async () => {
    const remembered = jobsStore.current[kind];
    await jobsStore.refresh(kind);
    const job = remembered ? jobsStore.lists[kind].find((entry) => entry.id === remembered) : undefined;
    if (job && jobsStore.current[kind] === remembered) onOpen(job);
    else if (remembered && !job) jobsStore.current[kind] = null;
  });

  $effect(() => {
    void jobsStore.current[kind];
    jobsStore.persistCurrent();
  });

  async function remove(job: Job) {
    try {
      error = "";
      await jobsStore.remove(kind, job.id);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  const when = (date: string) =>
    new Date(date).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
</script>

<aside
  class="flex max-h-48 shrink-0 flex-col border-b border-gray-200 bg-gray-50 md:max-h-none md:w-64 md:border-r md:border-b-0 dark:border-gray-800 dark:bg-gray-900"
  aria-label="History"
>
  <div class="px-3 pt-3 pb-1 text-xs font-medium text-gray-500">History</div>
  {#if error}
    <div class="px-3 text-xs text-red-600">{error}</div>
  {/if}
  <div class="min-h-0 flex-1 overflow-y-auto p-2">
    {#each jobsStore.lists[kind] as job (job.id)}
      {@const active = jobsStore.current[kind] === job.id}
      <div
        class="group flex items-center rounded-lg px-2 py-1.5 text-sm {active
          ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200'
          : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'}"
      >
        <button class="flex min-w-0 flex-1 flex-col items-start text-left disabled:opacity-60" disabled={busy} onclick={() => onOpen(job)}>
          <span class="w-full truncate">{job.title || "Untitled"}</span>
          <span class="w-full truncate font-mono text-[10.5px] {active ? 'text-blue-600 dark:text-blue-300' : 'text-gray-400 dark:text-gray-500'}">
            {when(job.created_at)}{job.model ? `, ${job.model}` : ""}
          </span>
        </button>
        <button
          class="shrink-0 rounded p-0.5 text-gray-400 opacity-0 group-hover:opacity-100 hover:text-red-500 focus:opacity-100 dark:text-gray-500 dark:hover:text-red-400"
          disabled={busy && active}
          onclick={() => remove(job)}
          title="Delete"
          aria-label="Delete"
        >
          <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>
    {:else}
      <div class="px-2 text-xs text-gray-400">Finished jobs show here.</div>
    {/each}
  </div>
</aside>
