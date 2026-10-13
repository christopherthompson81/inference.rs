<script lang="ts">
  import { onDestroy } from "svelte";
  import JobHistory from "../components/JobHistory.svelte";
  import SlotBar from "../components/SlotBar.svelte";
  import { jobsStore } from "../stores/jobs.svelte";
  import { managerStore, SLOTS } from "../stores/manager.svelte";
  import { speak } from "../services/manager";
  import { jobFileUrl, type Job } from "../services/jobs";

  const AUDIO_FILE = "speech.wav";

  interface SpeechJob {
    text: string;
    voice: string;
    speed: number;
  }

  let text = $state("Hello! This sentence was spoken by inference.rs, from the web UI.");
  let voice = $state("af_heart");
  let speed = $state(1.0);
  let busy = $state(false);
  let status = $state("");
  let audioUrl = $state<string | null>(null);
  // only new speech plays by itself, not a job opened from the history
  let fresh = $state(false);

  let ready = $derived(managerStore.servedIn("speech") != null);
  onDestroy(() => show(null));

  // a kept job's audio is served, not a blob of this page's
  function show(url: string | null) {
    if (audioUrl?.startsWith("blob:")) URL.revokeObjectURL(audioUrl);
    audioUrl = url;
  }

  function open(job: Job) {
    const data = job.data as unknown as SpeechJob;
    text = data.text;
    voice = data.voice;
    speed = data.speed;
    show(job.files.includes(AUDIO_FILE) ? jobFileUrl(job, AUDIO_FILE) : null);
    status = "";
    fresh = false;
    jobsStore.current.speech = job.id;
  }

  async function run() {
    busy = true;
    status = "Generating";
    const started = performance.now();
    const job: SpeechJob = { text: text.trim(), voice: voice.trim(), speed };
    try {
      const audio = await speak({
        model: SLOTS.speech.id,
        input: job.text,
        voice: job.voice || undefined,
        speed,
      });
      fresh = true;
      show(URL.createObjectURL(audio));
      status = `${(audio.size / 1024).toFixed(0)} KiB of audio in ${((performance.now() - started) / 1000).toFixed(1)} s`;
      await jobsStore
        .save({ kind: "speech", title: job.text, slot: "speech", data: job }, [{ name: AUDIO_FILE, blob: audio }])
        .catch((e) => (status += `; not kept: ${e instanceof Error ? e.message : String(e)}`));
    } catch (e) {
      status = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
</script>

<div class="flex min-h-0 w-full flex-1 flex-col md:flex-row">
  <JobHistory kind="speech" onOpen={open} {busy} />
  <div class="min-h-0 flex-1 overflow-y-auto">
    <div class="mx-auto flex w-full max-w-4xl flex-col gap-3 p-4">
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
        <audio controls autoplay={fresh} src={audioUrl} class="w-full"></audio>
      {/if}
    </div>
  </div>
</div>
