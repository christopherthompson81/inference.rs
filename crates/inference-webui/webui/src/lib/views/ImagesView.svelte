<script lang="ts">
  import SlotBar from "../components/SlotBar.svelte";
  import { managerStore, SLOTS } from "../stores/manager.svelte";
  import { editImage, generateImage, type ImageOptions } from "../services/manager";

  let prompt = $state("A lighthouse on a rocky coast at dusk, oil painting");
  let width = $state(1024);
  let height = $state(1024);
  // 0 and -1 leave the model's own choice
  let steps = $state(0);
  let guidance = $state(0);
  let seed = $state(-1);
  let references = $state<File[]>([]);
  let previews = $state<string[]>([]);
  let images = $state<string[]>([]);
  let busy = $state(false);
  let status = $state("");

  let ready = $derived(managerStore.servedIn("images") != null);

  function chooseReferences(files: FileList | null) {
    for (const url of previews) URL.revokeObjectURL(url);
    references = files ? [...files] : [];
    previews = references.map((file) => URL.createObjectURL(file));
  }

  async function run() {
    busy = true;
    status = references.length > 0 ? "Editing" : "Generating";
    const started = performance.now();
    const options: ImageOptions = {};
    // 0 leaves the size to the model, which for an edit is the reference's
    if (width > 0 && height > 0) Object.assign(options, { width, height });
    if (steps > 0) options.steps = steps;
    if (guidance > 0) options.guidance_scale = guidance;
    if (seed >= 0) options.seed = seed;
    try {
      images =
        references.length > 0
          ? await editImage(SLOTS.images.id, prompt.trim(), references, options)
          : await generateImage(SLOTS.images.id, prompt.trim(), options);
      status = `Done in ${((performance.now() - started) / 1000).toFixed(1)} s`;
    } catch (e) {
      status = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  const field = "rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-950";
</script>

<div class="mx-auto flex w-full max-w-5xl flex-col gap-3 overflow-y-auto p-4">
  <SlotBar slot="images" />

  <textarea class="min-h-20 {field}" bind:value={prompt}></textarea>
  <div class="flex flex-wrap items-end gap-2">
    <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Width (0: model's)</span><input type="number" step="64" min="0" class="w-24 {field}" bind:value={width} /></label>
    <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Height</span><input type="number" step="64" min="0" class="w-24 {field}" bind:value={height} /></label>
    <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Steps (0: model's)</span><input type="number" min="0" class="w-24 {field}" bind:value={steps} /></label>
    <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Guidance (0: model's)</span><input type="number" min="0" step="0.5" class="w-24 {field}" bind:value={guidance} /></label>
    <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Seed (-1: random)</span><input type="number" min="-1" class="w-28 {field}" bind:value={seed} /></label>
    <label class="cursor-pointer rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800" title="One image to edit, or several as subjects for a new scene (HiDream-O1)">
      Reference images...
      <input type="file" accept="image/*" multiple class="hidden" onchange={(e) => chooseReferences(e.currentTarget.files)} />
    </label>
    {#if references.length > 0}
      <button class="text-xs text-gray-500 underline" onclick={() => chooseReferences(null)}>clear {references.length}</button>
    {/if}
    <button
      class="rounded-md bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      onclick={run}
      disabled={!ready || !prompt.trim() || busy}
    >
      {references.length > 0 ? "Edit" : "Generate"}
    </button>
    <span class="text-xs text-gray-500">{ready ? status : "Load an image model first"}</span>
  </div>
  {#if previews.length > 0}
    <div class="flex flex-wrap gap-2">
      {#each previews as preview (preview)}
        <img src={preview} alt="Reference" class="h-24 rounded border border-gray-200 object-cover dark:border-gray-800" />
      {/each}
    </div>
  {/if}
  <div class="grid grid-cols-1 gap-3 md:grid-cols-2">
    {#each images as image, index (index)}
      <a href={image} download="image-{index + 1}.png">
        <img src={image} alt="Generated" class="w-full rounded-lg border border-gray-200 dark:border-gray-800" />
      </a>
    {/each}
  </div>
</div>
