<script lang="ts">
  import { onMount, tick } from "svelte";
  import { managerStore, SLOTS, displayName, formatBytes, baseName, fits, type SlotId } from "../stores/manager.svelte";
  import { deleteModel, downloadModel, type InventoryModel } from "../services/manager";
  import SlotBar from "../components/SlotBar.svelte";

  let { onUse }: { onUse: (slot: SlotId) => void } = $props();

  const KIND_LABELS: Record<string, string> = {
    image_generation: "image generation",
    voice_activity: "voice activity",
  };

  const SETTINGS_HASH = "#settings-";
  // the inventory, or every slot's settings (a slot bar's Settings link opens them at its slot)
  let view = $state<"inventory" | "settings">(location.hash.startsWith(SETTINGS_HASH) ? "settings" : "inventory");

  async function followHash() {
    if (!location.hash.startsWith(SETTINGS_HASH)) return;
    view = "settings";
    await tick();
    document.getElementById(location.hash.slice(1))?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  onMount(() => {
    void followHash();
    window.addEventListener("hashchange", followHash);
    return () => window.removeEventListener("hashchange", followHash);
  });

  let filter = $state("");
  let showProjectors = $state(false);
  let showUnsupported = $state(false);
  let selected = $state<InventoryModel | null>(null);
  let confirming = $state(false);
  let notice = $state("");

  let downloadRepo = $state("");
  let downloadQuant = $state("");
  let downloadFiles = $state("");
  let downloading = $state<AbortController | null>(null);
  let downloadStatus = $state("");
  let downloadPercent = $state<number | null>(null);

  let entries = $derived(
    (managerStore.inventory?.models ?? [])
      .filter((model) => showProjectors || model.kind !== "projector")
      .filter((model) => showUnsupported || model.kind != null)
      .filter((model) => {
        const needle = filter.trim().toLowerCase();
        if (!needle) return true;
        return [displayName(model), model.architecture, model.quantization, model.path, model.kind]
          .some((field) => field?.toLowerCase().includes(needle));
      })
      .sort((a, b) => displayName(a).localeCompare(displayName(b))),
  );

  function sourceName(source: string | null): string {
    if (!source) return "";
    const known = managerStore.entry(source);
    if (known) return displayName(known);
    return source.startsWith("/") ? baseName(source) : source;
  }

  function slotsFor(model: InventoryModel): SlotId[] {
    return (Object.keys(SLOTS) as SlotId[]).filter((slot) => fits(slot, model));
  }

  function kindLabel(model: InventoryModel): string {
    if (model.kind == null) return "unsupported";
    return KIND_LABELS[model.kind] ?? model.kind;
  }

  // what a delete removes: a cached repository whole, or a model folder's GGUF and its shards
  function deletion(model: InventoryModel): { target: { repo_id?: string; path?: string }; what: string } | null {
    if (model.source === "hf_cache" && model.repo_id) {
      return { target: { repo_id: model.repo_id }, what: `the cached repository ${model.repo_id}, every revision` };
    }
    if (model.source === "local" && model.format === "gguf") {
      return { target: { path: model.path }, what: `${model.files.length} GGUF file(s): ${model.files.map(baseName).join(", ")}` };
    }
    return null;
  }

  async function use(model: InventoryModel, slot: SlotId) {
    managerStore.assign(slot, model.path);
    onUse(slot);
    await managerStore.load(slot);
  }

  async function remove(model: InventoryModel) {
    const plan = deletion(model);
    if (!plan) return;
    confirming = false;
    try {
      const deleted = await deleteModel(plan.target);
      notice = `Deleted ${deleted.deleted}, freeing ${formatBytes(deleted.freed_bytes)}`;
      selected = null;
      await managerStore.scan();
    } catch (e) {
      notice = e instanceof Error ? e.message : String(e);
    }
  }

  async function download() {
    const controller = new AbortController();
    downloading = controller;
    downloadPercent = null;
    downloadStatus = "Starting";
    const files = downloadFiles.split(",").map((file) => file.trim()).filter(Boolean);
    try {
      const outcome = await downloadModel(
        { repo_id: downloadRepo.trim(), quant: downloadQuant.trim() || undefined, files },
        (progress) => {
          downloadPercent = progress.total > 0 ? Math.round((100 * progress.done) / progress.total) : null;
          downloadStatus = `${baseName(progress.file)}: ${formatBytes(progress.done)} of ${formatBytes(progress.total)}`;
        },
        controller.signal,
      );
      downloadStatus = outcome === "done" ? `Downloaded ${downloadRepo.trim()}` : "Canceled";
      if (outcome === "done") await managerStore.scan();
    } catch (e) {
      downloadStatus = e instanceof Error ? e.message : String(e);
    } finally {
      downloading = null;
      downloadPercent = null;
    }
  }

  const field = "rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-950";
</script>

{#if !managerStore.admin}
  <div class="m-auto max-w-md p-8 text-center text-sm text-gray-500">
    {#if managerStore.keyed}
      Loading, downloading and deleting models is for admins: the keys the server names with <code>--admin</code>.
    {:else}
      Loading, downloading and deleting models is for a browser on the server's own machine, at
      <code>localhost</code>, its address or its name. From another device, start the server with keys and
      <code>--admin</code>.
    {/if}
  </div>
{:else}
  <div class="flex min-h-0 w-full flex-1 flex-col gap-3 overflow-hidden p-4">
    <div class="flex gap-1" role="tablist">
      {#each [{ id: "inventory", label: "Inventory" }, { id: "settings", label: "Slot settings" }] as item (item.id)}
        <button
          role="tab"
          aria-selected={view === item.id}
          class="rounded-md px-3 py-1 text-sm {view === item.id
            ? 'bg-gray-200 font-medium text-gray-900 dark:bg-gray-800 dark:text-gray-100'
            : 'text-gray-500 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200'}"
          onclick={() => (view = item.id as typeof view)}
        >
          {item.label}
        </button>
      {/each}
    </div>
  {#if view === "settings"}
    <div class="flex min-h-0 flex-col gap-3 overflow-y-auto">
      <p class="text-xs text-gray-500">
        Each slot's model and how it runs. A load keeps its settings on the server, so every browser sees them, and a
        slot marked "Load at startup" loads again when the server starts. Save keeps changes without loading.
        {#if managerStore.savedError}<span class="text-red-600">{managerStore.savedError}</span>{/if}
      </p>
      <div class="grid gap-3 xl:grid-cols-2">
        {#each Object.keys(SLOTS) as slot (slot)}
          <SlotBar slot={slot as SlotId} full />
        {/each}
      </div>
    </div>
  {:else}
    <div class="flex flex-wrap items-center gap-2">
      <button class="rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 disabled:opacity-50 dark:border-gray-700 dark:hover:bg-gray-800" onclick={() => managerStore.scan()} disabled={managerStore.scanning}>
        {managerStore.scanning ? "Scanning..." : "Refresh"}
      </button>
      <input class="min-w-64 flex-1 {field}" placeholder="Filter by name, architecture, quantization or path" bind:value={filter} />
      <label class="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-300"><input type="checkbox" class="accent-blue-600" bind:checked={showProjectors} />Projectors</label>
      <label class="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-300"><input type="checkbox" class="accent-blue-600" bind:checked={showUnsupported} />Unsupported</label>
    </div>
    <p class="text-xs text-gray-500">
      {entries.length} of {managerStore.inventory?.models.length ?? 0} models{managerStore.inventory?.hf_cache ? `, cache at ${managerStore.inventory.hf_cache.path}` : ""}.
      {notice}
      {#if managerStore.inventoryError}<span class="text-red-600">{managerStore.inventoryError}</span>{/if}
    </p>

    <div class="grid min-h-0 flex-1 gap-3 lg:grid-cols-[1fr_24rem]">
      <div class="min-h-0 overflow-y-auto rounded-lg border border-gray-200 dark:border-gray-800">
        <table class="w-full text-left text-sm">
          <thead class="sticky top-0 bg-gray-50 text-xs text-gray-500 dark:bg-gray-900">
            <tr><th class="px-2 py-1.5 font-medium">Model</th><th class="font-medium">Kind</th><th class="font-medium">Quant</th><th class="pr-2 text-right font-medium">Size</th></tr>
          </thead>
          <tbody>
            {#each entries as model (model.path)}
              <tr
                class="cursor-pointer border-t border-gray-100 hover:bg-gray-50 dark:border-gray-800 dark:hover:bg-gray-900 {selected?.path === model.path ? 'bg-blue-50 dark:bg-blue-900/20' : ''}"
                onclick={() => {
                  selected = model;
                  confirming = false;
                }}
              >
                <td class="px-2 py-1">
                  <div class="font-medium">{displayName(model)}</div>
                  <div class="truncate text-[11px] text-gray-400">{model.source === "hf_cache" ? "HF cache" : model.path}</div>
                </td>
                <td class="text-xs">{kindLabel(model)}</td>
                <td class="font-mono text-xs">{model.quantization ?? ""}</td>
                <td class="pr-2 text-right font-mono text-xs">{formatBytes(model.size_bytes)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>

      <div class="flex min-h-0 flex-col gap-3 overflow-y-auto">
        {#if selected}
          {@const plan = deletion(selected)}
          <div class="rounded-lg border border-gray-200 p-3 text-sm dark:border-gray-800">
            <div class="font-semibold">{displayName(selected)}</div>
            <div class="text-xs text-gray-500">{kindLabel(selected)}{selected.architecture ? `, ${selected.architecture}` : ""}{selected.size_label ? `, ${selected.size_label}` : ""}</div>
            <div class="mt-1 font-mono text-[11px] break-all text-gray-400">{selected.path}</div>
            {#if selected.base_model}<div class="mt-1 text-xs">Base model: {selected.base_model}</div>{/if}
            {#if selected.context_length}<div class="text-xs">Context: {selected.context_length.toLocaleString()} tokens</div>{/if}
            {#if selected.projectors.length > 0}
              <div class="mt-1 text-xs">Projectors: {selected.projectors.map(baseName).join(", ")}</div>
            {/if}
            {#if selected.error}<div class="mt-1 text-xs text-red-600">{selected.error}</div>{/if}
            <div class="mt-2 flex flex-wrap items-center gap-1.5">
              {#each slotsFor(selected) as slot (slot)}
                <button class="rounded-md bg-blue-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50" onclick={() => use(selected!, slot)} disabled={managerStore.activityOf(slot).busy}>
                  Load as {SLOTS[slot].title.toLowerCase().replace(" (optional)", "")}
                </button>
              {/each}
              {#if plan}
                <button class="ml-auto rounded-md border border-red-300 px-2.5 py-1 text-xs text-red-700 hover:bg-red-50 dark:border-red-800 dark:text-red-300 dark:hover:bg-red-950" onclick={() => (confirming = true)}>
                  Delete...
                </button>
              {/if}
            </div>
            {#if confirming && plan}
              <div class="mt-2 rounded border border-red-200 bg-red-50 p-2 text-xs dark:border-red-900 dark:bg-red-950/40">
                This deletes {plan.what}.
                <div class="mt-1.5 flex gap-1.5">
                  <button class="rounded bg-red-600 px-2 py-0.5 text-white hover:bg-red-700" onclick={() => remove(selected!)}>Delete</button>
                  <button class="rounded border border-gray-300 px-2 py-0.5 dark:border-gray-700" onclick={() => (confirming = false)}>Keep</button>
                </div>
              </div>
            {/if}
          </div>
        {:else}
          <p class="text-xs text-gray-400">Select a model to see its details and load it.</p>
        {/if}

        <div class="rounded-lg border border-gray-200 p-3 dark:border-gray-800">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-500">Served</h3>
          {#each managerStore.served as model (model.model_id)}
            <div class="mt-1 flex items-center gap-2 text-xs">
              <span class="font-mono">{model.model_id}</span>
              <span class="text-gray-400">{model.kind}</span>
              <span class="truncate text-gray-400" title={model.source ?? ""}>{sourceName(model.source)}</span>
            </div>
          {:else}
            <p class="mt-1 text-xs text-gray-400">Nothing yet.</p>
          {/each}
        </div>

        <div class="flex flex-col gap-2 rounded-lg border border-gray-200 p-3 dark:border-gray-800">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-500">Download from Hugging Face</h3>
          <input class={field} placeholder="Repository, e.g. unsloth/Qwen3-8B-GGUF" bind:value={downloadRepo} disabled={downloading != null} />
          <div class="flex gap-2">
            <input class="w-32 {field}" placeholder="Quant, e.g. Q4_K_M" bind:value={downloadQuant} disabled={downloading != null} />
            <input class="min-w-0 flex-1 {field}" placeholder="Files, comma separated" bind:value={downloadFiles} disabled={downloading != null} />
          </div>
          <p class="text-[11px] text-gray-400">A quant downloads its GGUF and the projector beside it; with neither a quant nor files, the whole repository.</p>
          <div class="flex gap-2">
            <button class="rounded-md bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50" onclick={download} disabled={downloading != null || !downloadRepo.trim()}>Download</button>
            <button class="rounded-md border border-gray-300 px-3 py-1 text-xs disabled:opacity-50 dark:border-gray-700" onclick={() => downloading?.abort()} disabled={downloading == null}>Cancel</button>
          </div>
          {#if downloading}
            <div class="h-1.5 overflow-hidden rounded bg-gray-200 dark:bg-gray-800">
              <div class="h-full bg-blue-500 {downloadPercent == null ? 'w-1/3 animate-pulse' : ''}" style={downloadPercent == null ? "" : `width: ${downloadPercent}%`}></div>
            </div>
          {/if}
          <p class="text-xs break-all text-gray-500">{downloadStatus}</p>
        </div>
      </div>
    </div>
  {/if}
  </div>
{/if}
