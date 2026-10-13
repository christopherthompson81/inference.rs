<script lang="ts">
  import { onDestroy } from "svelte";
  import SlotBar from "../components/SlotBar.svelte";
  import { managerStore, SLOTS } from "../stores/manager.svelte";
  import { speak } from "../services/manager";

  let text = $state("Hello! This sentence was spoken by inference.rs, from the web UI.");
  let voice = $state("af_heart");
  let speed = $state(1.0);
  let busy = $state(false);
  let status = $state("");
  let audioUrl = $state<string | null>(null);

  let ready = $derived(managerStore.servedIn("speech") != null);
  onDestroy(() => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  });

  async function run() {
    busy = true;
    status = "Generating";
    const started = performance.now();
    try {
      const audio = await speak({
        model: SLOTS.speech.id,
        input: text.trim(),
        voice: voice.trim() || undefined,
        speed,
      });
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      audioUrl = URL.createObjectURL(audio);
      status = `${(audio.size / 1024).toFixed(0)} KiB of audio in ${((performance.now() - started) / 1000).toFixed(1)} s`;
    } catch (e) {
      status = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
</script>

<div class="mx-auto flex w-full max-w-4xl flex-col gap-3 overflow-y-auto p-4">
  <SlotBar slot="speech" />

  <textarea
    class="min-h-32 rounded-md border border-gray-300 bg-white p-2 text-sm dark:border-gray-700 dark:bg-gray-950"
    bind:value={text}
  ></textarea>
  <div class="flex flex-wrap items-end gap-2">
    <label class="flex flex-col gap-1">
      <span class="text-[11px] text-gray-500" title="Kokoro: af_heart, bm_george, ...; comma-separated voices blend">Voice</span>
      <input class="w-40 rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-950" bind:value={voice} />
    </label>
    <label class="flex flex-col gap-1">
      <span class="text-[11px] text-gray-500">Speed</span>
      <input
        type="number"
        min="0.5"
        max="2"
        step="0.1"
        class="w-20 rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-950"
        bind:value={speed}
      />
    </label>
    <button
      class="rounded-md bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      onclick={run}
      disabled={!ready || !text.trim() || busy}
    >
      Speak
    </button>
    {#if audioUrl}
      <a class="rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800" href={audioUrl} download="speech.wav">
        Save
      </a>
    {/if}
    <span class="text-xs text-gray-500">{ready ? status : "Load a speech synthesis model first"}</span>
  </div>
  {#if audioUrl}
    <audio controls autoplay src={audioUrl} class="w-full"></audio>
  {/if}
</div>
