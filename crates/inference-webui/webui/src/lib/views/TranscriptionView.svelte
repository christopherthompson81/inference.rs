<script lang="ts">
  import { onDestroy } from "svelte";
  import JobHistory from "../components/JobHistory.svelte";
  import SlotBar from "../components/SlotBar.svelte";
  import { jobsStore } from "../stores/jobs.svelte";
  import { managerStore, SLOTS } from "../stores/manager.svelte";
  import { diarize, transcribe } from "../services/manager";
  import { jobFile, jobFileUrl, keptExtension, type Job } from "../services/jobs";

  const AUDIO_STEM = "audio";

  interface Segment {
    start: number;
    end: number;
    text: string;
    speaker: string;
  }

  interface TranscriptionJob {
    audioName: string;
    language: string;
    transcript: string;
    segments: Segment[];
  }

  let audio = $state<File | null>(null);
  // the history entry whose audio is shown, read back from the server when transcribed again
  let restored = $state<Job | null>(null);
  let audioUrl = $state<string | null>(null);
  let language = $state("");
  let busy = $state(false);
  let status = $state("");
  let transcript = $state("");
  let segments = $state<Segment[]>([]);

  let ready = $derived(managerStore.servedIn("transcription") != null);
  onDestroy(() => show(null));
  let diarizer = $derived(managerStore.servedIn("diarization") != null);

  function show(url: string | null) {
    if (audioUrl?.startsWith("blob:")) URL.revokeObjectURL(audioUrl);
    audioUrl = url;
  }

  function choose(file: File | undefined) {
    if (!file) return;
    audio = file;
    restored = null;
    jobsStore.current.transcription = null;
    show(URL.createObjectURL(file));
    transcript = "";
    segments = [];
    status = file.name;
  }

  // an entry deleted while it is shown can no longer be read back
  $effect(() => {
    if (restored && !jobsStore.lists.transcription.some((job) => job.id === restored!.id)) {
      restored = null;
      if (!audio) show(null);
    }
  });

  function open(job: Job) {
    const data = $state.snapshot(job.data) as unknown as TranscriptionJob;
    const name = job.files.find((file) => file.startsWith(`${AUDIO_STEM}.`));
    audio = null;
    restored = name ? job : null;
    show(name ? jobFileUrl(job, name) : null);
    language = data.language;
    transcript = data.transcript;
    segments = data.segments;
    status = data.audioName;
    jobsStore.current.transcription = job.id;
  }

  // the audio to send: the one chosen, or the history entry's
  async function source(): Promise<{ file: File; name: string } | null> {
    if (audio) return { file: audio, name: audio.name };
    const name = restored?.files.find((file) => file.startsWith(`${AUDIO_STEM}.`));
    if (!restored || !name) return null;
    return { file: await jobFile(restored, name), name: (restored.data as unknown as TranscriptionJob).audioName };
  }

  // each transcript segment takes the speaker whose turns overlap it most
  function label(
    parts: { start: number; end: number; text: string }[],
    turns: { speaker: number; start: number; end: number }[],
  ): Segment[] {
    return parts.map((part) => {
      const overlap = new Map<number, number>();
      for (const turn of turns) {
        const shared = Math.min(part.end, turn.end) - Math.max(part.start, turn.start);
        if (shared > 0) overlap.set(turn.speaker, (overlap.get(turn.speaker) ?? 0) + shared);
      }
      const best = [...overlap.entries()].sort((a, b) => b[1] - a[1])[0];
      return { ...part, text: part.text.trim(), speaker: best ? `Speaker ${best[0]}` : "" };
    });
  }

  async function run() {
    busy = true;
    status = "Transcribing";
    const started = performance.now();
    const again = restored;
    try {
      const input = await source();
      if (!input) return;
      // the speakers are optional: a diarizer that fails leaves the transcript unlabeled
      let speakersFailed = "";
      const [reply, turns] = await Promise.all([
        transcribe(SLOTS.transcription.id, input.file, language.trim() || undefined),
        diarizer
          ? diarize(SLOTS.diarization.id, input.file).then(
              (r) => r.segments,
              (e) => {
                speakersFailed = e instanceof Error ? e.message : String(e);
                return [];
              },
            )
          : Promise.resolve([]),
      ]);
      transcript = reply.text;
      segments = label(reply.segments ?? [], turns);
      status = `Done in ${((performance.now() - started) / 1000).toFixed(1)} s`;
      if (speakersFailed) status += `; no speakers: ${speakersFailed}`;
      const data: TranscriptionJob = { audioName: input.name, language: language.trim(), transcript, segments };
      // transcribing a history entry again updates it rather than keeping its audio twice
      const extension = keptExtension(input.file, "audio");
      const files = again || !extension ? [] : [{ name: `${AUDIO_STEM}.${extension}`, blob: input.file }];
      if (!again && !extension) status += "; this audio type is not kept, only the transcript";
      try {
        const kept = await jobsStore.save(
          { id: again?.id, kind: "transcription", title: input.name, slot: "transcription", data },
          files,
        );
        // with its audio kept, the entry is what runs again; without, the chosen file still is
        if (again || files.length > 0) {
          restored = kept;
          audio = null;
        }
      } catch (e) {
        status += `; not kept: ${e instanceof Error ? e.message : String(e)}`;
      }
    } catch (e) {
      status = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  const time = (seconds: number) => seconds.toFixed(2).padStart(7, " ");
</script>

<div class="flex min-h-0 w-full flex-1 flex-col md:flex-row">
  <JobHistory kind="transcription" onOpen={open} {busy} />
  <div class="min-h-0 flex-1 overflow-y-auto">
    <div class="mx-auto flex w-full max-w-5xl flex-col gap-3 p-4">
      <SlotBar slot="transcription" />
      <SlotBar slot="diarization" />

      <div class="flex flex-wrap items-center gap-2">
        <label class="cursor-pointer rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800">
          Choose audio...
          <input type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a" class="hidden" disabled={busy} onchange={(e) => choose(e.currentTarget.files?.[0])} />
        </label>
        <input
          class="w-36 rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-950"
          placeholder="Language (auto)"
          aria-label="Language"
          bind:value={language}
        />
        <button
          class="rounded-md bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          onclick={run}
          disabled={!ready || !(audio || restored) || busy}
        >
          Transcribe
        </button>
        <span class="text-xs text-gray-500">{ready ? status : "Load a speech recognition model first"}</span>
      </div>
      {#if audioUrl}
        <audio controls src={audioUrl} class="w-full"></audio>
      {/if}

      {#if transcript}
        <div class="rounded-lg border border-gray-200 p-3 text-sm leading-relaxed whitespace-pre-wrap dark:border-gray-800">{transcript}</div>
      {/if}
      {#if segments.length > 0}
        <table class="w-full text-left text-sm">
          <thead class="text-xs text-gray-500">
            <tr><th class="w-40 py-1 font-medium">Time</th><th class="w-28 font-medium">Speaker</th><th class="font-medium">Text</th></tr>
          </thead>
          <tbody>
            {#each segments as segment, index (index)}
              <tr class="border-t border-gray-100 align-top dark:border-gray-800">
                <td class="py-1 font-mono text-xs whitespace-pre text-gray-500">{time(segment.start)} - {time(segment.end)}</td>
                <td class="py-1 text-xs text-gray-600 dark:text-gray-300">{segment.speaker}</td>
                <td class="py-1">{segment.text}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </div>
  </div>
</div>
