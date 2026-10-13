<script lang="ts">
  import { managerStore, SLOTS, displayName, baseName, formatBytes, type Slot, type SlotId } from "../stores/manager.svelte";

  let { slot }: { slot: SlotId } = $props();

  // the select's value for a source the inventory does not list: a Hugging Face repository
  const REPO = "__repo__";

  let spec: Slot = $derived(SLOTS[slot]);
  let form = $derived(managerStore.form(slot));
  let served = $derived(managerStore.servedIn(slot));
  let activity = $derived(managerStore.activityOf(slot));
  let candidates = $derived(managerStore.candidates(slot));
  let listed = $derived(candidates.some((model) => model.path === form.source));
  let choice = $state("");
  let repo = $state("");
  // until the user picks, the form follows what the slot serves
  let picked = $state(false);
  let entry = $derived(listed ? managerStore.entry(form.source) : undefined);
  let percent = $derived(
    activity.progress && activity.progress.total > 0
      ? Math.round((100 * activity.progress.done) / activity.progress.total)
      : null,
  );

  let servedName = $derived.by(() => {
    if (!served?.source) return served?.model_id ?? "";
    const known = managerStore.entry(served.source);
    if (known) return displayName(known);
    return served.source.startsWith("/") ? baseName(served.source) : served.source;
  });

  $effect(() => {
    const source = served?.source;
    if (source && !picked && form.source !== source) {
      form.source = source;
      form.mmproj = served?.projector ?? "";
    }
  });

  // the select follows the form: a listed source picks its entry, anything else is a repository
  $effect(() => {
    if (listed) choice = form.source;
    else if (form.source) {
      choice = REPO;
      repo = form.source;
    } else if (candidates[0]) form.source = candidates[0].path;
    else choice = REPO;
  });

  function pick(value: string) {
    picked = true;
    choice = value;
    form.source = value === REPO ? repo.trim() : value;
    form.mmproj = "";
    managerStore.persistForms();
  }

  function details(path: string): string {
    const model = managerStore.entry(path);
    if (!model) return "";
    const parts = [model.quantization, model.size_label, formatBytes(model.size_bytes)];
    if (model.source === "hf_cache") parts.push("HF cache");
    return parts.filter(Boolean).join(", ");
  }

  async function load() {
    if (choice === REPO) form.source = repo.trim();
    if (!form.source) return;
    await managerStore.load(slot);
  }
</script>

<section class="rounded-lg border border-gray-200 bg-gray-50 p-3 text-sm dark:border-gray-800 dark:bg-gray-900/60">
  <div class="flex flex-wrap items-center gap-2">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">{spec.title}</h3>
    {#if served}
      <span class="rounded bg-green-100 px-1.5 py-0.5 font-mono text-[10px] uppercase text-green-700 dark:bg-green-900/40 dark:text-green-300">
        {served.status}
      </span>
      <span class="truncate font-mono text-xs text-gray-700 dark:text-gray-200" title={served.source ?? served.model_id}>
        {servedName}
      </span>
      {#if served.projector}
        <span class="truncate font-mono text-[11px] text-gray-400" title={served.projector}>+ {baseName(served.projector)}</span>
      {/if}
    {:else}
      <span class="text-xs text-gray-400">{managerStore.admin ? "Nothing loaded" : "Nothing loaded; an admin loads models"}</span>
    {/if}
  </div>

  {#if managerStore.admin}
    <div class="mt-2 flex flex-wrap items-end gap-2">
      <label class="flex min-w-64 flex-1 flex-col gap-1">
        <span class="text-[11px] text-gray-500 dark:text-gray-400">{spec.hint}</span>
        <select
          class="rounded-md border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-700 dark:bg-gray-950"
          value={choice}
          onchange={(e) => pick(e.currentTarget.value)}
          disabled={activity.busy}
        >
          {#each candidates as model (model.path)}
            <option value={model.path}>{displayName(model)}{details(model.path) ? `  (${details(model.path)})` : ""}</option>
          {/each}
          <option value={REPO}>Hugging Face repository...</option>
        </select>
      </label>
      {#if choice === REPO}
        <input
          class="min-w-56 rounded-md border border-gray-300 bg-white px-2 py-1 font-mono text-xs dark:border-gray-700 dark:bg-gray-950"
          placeholder="org/model"
          bind:value={repo}
          onchange={() => pick(REPO)}
          disabled={activity.busy}
        />
      {/if}
      {#if entry && entry.projectors.length > 0}
        <label class="flex flex-col gap-1">
          <span class="text-[11px] text-gray-500 dark:text-gray-400">Projector (images, audio)</span>
          <select
            class="max-w-56 rounded-md border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-700 dark:bg-gray-950"
            bind:value={form.mmproj}
            onchange={() => managerStore.persistForms()}
            disabled={activity.busy}
          >
            <option value="">{entry.projectors.length === 1 ? "The one beside it" : "None (text only)"}</option>
            {#each entry.projectors as projector (projector)}
              <option value={projector}>{baseName(projector)}</option>
            {/each}
          </select>
        </label>
      {/if}
      {#if spec.chat}
        <label class="flex flex-col gap-1">
          <span class="text-[11px] text-gray-500 dark:text-gray-400">Context (tokens)</span>
          <input
            type="number"
            min="0"
            step="1024"
            class="w-24 rounded-md border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-700 dark:bg-gray-950"
            bind:value={form.contextLength}
            onchange={() => managerStore.persistForms()}
            disabled={activity.busy}
          />
        </label>
        <label class="flex items-center gap-1.5 pb-1 text-xs text-gray-600 dark:text-gray-300" title="Speculative decoding with the checkpoint's own MTP head">
          <input type="checkbox" class="accent-blue-600" bind:checked={form.mtp} onchange={() => managerStore.persistForms()} disabled={activity.busy} />
          MTP
        </label>
        {#if form.mtp}
          <label class="flex flex-col gap-1">
            <span class="text-[11px] text-gray-500 dark:text-gray-400">Drafts (0: default)</span>
            <input
              type="number"
              min="0"
              max="8"
              class="w-20 rounded-md border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-700 dark:bg-gray-950"
              bind:value={form.mtpDrafts}
              onchange={() => managerStore.persistForms()}
              disabled={activity.busy}
            />
          </label>
        {/if}
      {/if}
      {#if spec.device}
        <label class="flex flex-col gap-1">
          <span class="text-[11px] text-gray-500 dark:text-gray-400">Device</span>
          <select
            class="rounded-md border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-700 dark:bg-gray-950"
            bind:value={form.device}
            onchange={() => managerStore.persistForms()}
            disabled={activity.busy}
          >
            <option value="auto">The engine's</option>
            <option value="cpu">CPU</option>
            <option value="cuda:0">cuda:0</option>
          </select>
        </label>
      {/if}
      <div class="flex gap-1.5 pb-px">
        <button
          class="rounded-md bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          onclick={load}
          disabled={activity.busy || (choice === REPO ? !repo.trim() : !form.source)}
        >
          {served ? "Replace" : "Load"}
        </button>
        <button
          class="rounded-md border border-gray-300 px-3 py-1 text-xs text-gray-700 hover:bg-gray-100 disabled:opacity-50 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-800"
          onclick={() => managerStore.unload(slot)}
          disabled={activity.busy || !served}
        >
          Unload
        </button>
      </div>
    </div>
    {#if candidates.length === 0 && managerStore.inventory}
      <p class="mt-1 text-[11px] text-gray-400">
        No model of this kind in the server's folders or cache; name a Hugging Face repository, or download one in Models.
      </p>
    {/if}
  {/if}

  {#if activity.busy}
    <div class="mt-2 flex items-center gap-2">
      <div class="h-1.5 flex-1 overflow-hidden rounded bg-gray-200 dark:bg-gray-800">
        <div
          class="h-full bg-blue-500 transition-all {percent == null ? 'w-1/3 animate-pulse' : ''}"
          style={percent == null ? "" : `width: ${percent}%`}
        ></div>
      </div>
      <span class="font-mono text-[11px] text-gray-500">
        {activity.progress ? `${activity.progress.stage} ${percent ?? ""}${percent == null ? "" : "%"}` : "Working"}
      </span>
    </div>
  {/if}
  {#if activity.error}
    <p class="mt-2 whitespace-pre-wrap text-xs text-red-600 dark:text-red-400">{activity.error}</p>
  {/if}
</section>
