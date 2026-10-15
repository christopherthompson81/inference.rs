<script lang="ts">
  import { onMount } from "svelte";
  import HistoryPanel from "./HistoryPanel.svelte";
  import { jobsStore } from "../stores/jobs.svelte";
  import { when } from "../utils/when";
  import type { Job, JobKind } from "../services/jobs";

  // `busy` holds the history still while the view runs a job
  let {
    kind,
    onOpen,
    onNew,
    newLabel,
    busy = false,
  }: { kind: JobKind; onOpen: (job: Job) => void; onNew: () => void; newLabel: string; busy?: boolean } = $props();
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

  async function remove(id: string) {
    try {
      error = "";
      await jobsStore.remove(kind, id);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  function open(id: string) {
    const job = jobsStore.lists[kind].find((entry) => entry.id === id);
    if (job) onOpen(job);
  }

  let entries = $derived(
    jobsStore.lists[kind].map((job) => ({
      id: job.id,
      title: job.title,
      detail: `${when(job.created_at)}${job.model ? `, ${job.model}` : ""}`,
    })),
  );
</script>

<HistoryPanel
  {entries}
  active={jobsStore.current[kind]}
  {newLabel}
  {onNew}
  onOpen={open}
  onDelete={remove}
  {busy}
  {error}
  empty="Finished jobs show here."
/>
