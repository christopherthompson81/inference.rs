<script lang="ts">
  import { onDestroy, tick } from "svelte";
  import AudioTransport from "../components/AudioTransport.svelte";
  import JobHistory from "../components/JobHistory.svelte";
  import SlotBar from "../components/SlotBar.svelte";
  import { jobsStore } from "../stores/jobs.svelte";
  import { managerStore, SLOTS } from "../stores/manager.svelte";
  import { transcribe } from "../services/manager";
  import { compactAudio } from "../services/audioUpload";
  import { jobFile, jobFileUrl, keptExtension, type Job } from "../services/jobs";
  import {
    EXPORT_FORMATS,
    download,
    renderTranscript,
    type ExportFormat,
    type TimedWord,
    type TranscriptSegment,
  } from "../services/transcriptExport";

  const AUDIO_STEM = "audio";
  // a word belongs to a sentence it starts within, give or take rounding in the reply's times
  const WORD_EDGE_SECONDS = 0.01;
  // a renamed speaker is saved once typing pauses this long
  const NAMES_SAVE_DELAY_MS = 800;
  const SPEAKER_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#ea580c", "#7c3aed", "#0d9488", "#db2777", "#4b5563"];

  interface TranscriptionJob {
    audioName: string;
    language: string;
    segments: TranscriptSegment[];
    names?: Record<string, string>;
  }

  interface Progress {
    stage: string;
    done: number;
    /** 0 while the stage can't count its work. */
    total: number;
  }

  let audio = $state<File | null>(null);
  // the history entry whose audio is shown, read back from the server when transcribed again
  let restored = $state<Job | null>(null);
  let audioUrl = $state<string | null>(null);
  let audioName = $state("");
  let language = $state("");
  let busy = $state(false);
  let status = $state("");
  let progress = $state<Progress | null>(null);
  let controller: AbortController | null = null;
  let segments = $state<TranscriptSegment[]>([]);
  let names = $state<Record<string, string>>({});

  let player = $state<HTMLAudioElement | null>(null);
  let activeSegment = $state(-1);
  let activeWord = $state(-1);
  let paused = $state(true);
  let position = $state(0);
  let length = $state(0);
  let frame = 0;
  let namesTimer: ReturnType<typeof setTimeout> | null = null;

  let ready = $derived(managerStore.servedIn("transcription") != null);
  let diarizer = $derived(managerStore.servedIn("diarization") != null);
  let speakers = $derived([...new Set(segments.map((segment) => segment.speaker).filter(Boolean))]);

  onDestroy(() => {
    show(null);
    cancelAnimationFrame(frame);
    controller?.abort();
  });

  function show(url: string | null) {
    if (audioUrl?.startsWith("blob:")) URL.revokeObjectURL(audioUrl);
    audioUrl = url;
  }

  function reset() {
    segments = [];
    names = {};
    activeSegment = -1;
    activeWord = -1;
  }

  function startNew() {
    player?.pause();
    audio = null;
    restored = null;
    audioName = "";
    show(null);
    reset();
    position = 0;
    length = 0;
    status = "";
    jobsStore.current.transcription = null;
  }

  function choose(input: HTMLInputElement) {
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;
    audio = file;
    audioName = file.name;
    restored = null;
    jobsStore.current.transcription = null;
    show(URL.createObjectURL(file));
    reset();
    status = "";
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
    reset();
    audioName = data.audioName;
    language = data.language ?? "";
    segments = data.segments ?? [];
    names = data.names ?? {};
    status = "";
    jobsStore.current.transcription = job.id;
  }

  // the audio to send: the one chosen, or the history entry's
  async function source(): Promise<File | null> {
    if (audio) return audio;
    const name = restored?.files.find((file) => file.startsWith(`${AUDIO_STEM}.`));
    return restored && name ? jobFile(restored, name) : null;
  }

  // a reply's sentences, each with the words that start inside it and the diarizer's speaker when it ran
  function segmentsOf(reply: {
    text: string;
    segments?: { start: number; end: number; text: string; speaker?: number }[];
    words?: TimedWord[];
  }): TranscriptSegment[] {
    const words = (reply.words ?? []).map((w) => ({ word: w.word, start: w.start, end: w.end }));
    const parts = reply.segments?.length
      ? reply.segments
      : [{ start: words[0]?.start ?? 0, end: words.at(-1)?.end ?? 0, text: reply.text }];
    return parts
      .map((part) => ({
        start: part.start,
        end: part.end,
        speaker: "speaker" in part && part.speaker !== undefined ? `Speaker ${part.speaker}` : "",
        text: part.text.trim(),
        words: words.filter((w) => w.start >= part.start - WORD_EDGE_SECONDS && w.start < part.end + WORD_EDGE_SECONDS),
      }))
      .filter((segment) => segment.text);
  }

  async function run() {
    busy = true;
    status = "";
    controller = new AbortController();
    const signal = controller.signal;
    const started = performance.now();
    const again = restored;
    // a speaker's name outlives transcribing again; the segments don't
    segments = [];
    activeSegment = -1;
    activeWord = -1;
    try {
      progress = { stage: "Reading the audio", done: 0, total: 0 };
      const file = await source();
      if (!file) return;
      const options = { language: language.trim() || undefined, signal };

      // one upload: the server diarizes the same audio while it transcribes, and labels the sentences
      progress = { stage: diarizer ? "Transcribing and finding the speakers" : "Transcribing", done: 0, total: 0 };
      let speakersFailed = "";
      // what the model hears, at the size it needs, which the history keeps too
      const upload = await compactAudio(file);
      let reply;
      try {
        reply = await transcribe(SLOTS.transcription.id, upload, {
          ...options,
          diarizationModel: diarizer ? SLOTS.diarization.id : undefined,
        });
      } catch (e) {
        if (signal.aborted || !diarizer) throw e;
        speakersFailed = e instanceof Error ? e.message : String(e);
        reply = await transcribe(SLOTS.transcription.id, upload, options);
      }
      segments = segmentsOf(reply);
      progress = null;
      status = `Done in ${((performance.now() - started) / 1000).toFixed(1)} s`;
      if (speakersFailed) status += `; no speakers: ${speakersFailed}`;
      await keep(upload, again);
    } catch (e) {
      status = signal.aborted ? "Canceled" : e instanceof Error ? e.message : String(e);
    } finally {
      progress = null;
      busy = false;
      controller = null;
    }
  }

  function jobData(): TranscriptionJob {
    return { audioName, language: language.trim(), segments, names };
  }

  async function keep(file: File, again: Job | null) {
    // transcribing a history entry again updates it rather than keeping its audio twice
    const extension = keptExtension(file, "audio");
    const files = again || !extension ? [] : [{ name: `${AUDIO_STEM}.${extension}`, blob: file }];
    if (!again && !extension) status += "; this audio type is not kept, only the transcript";
    try {
      const kept = await jobsStore.save(
        { id: again?.id, kind: "transcription", title: audioName, slot: "transcription", data: jobData() },
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
  }

  function rename(speaker: string, name: string) {
    names = { ...names, [speaker]: name };
    const id = jobsStore.current.transcription;
    if (!id || busy) return;
    if (namesTimer) clearTimeout(namesTimer);
    namesTimer = setTimeout(() => {
      jobsStore
        .save({ id, kind: "transcription", title: audioName, slot: "transcription", data: jobData() })
        .catch((e) => (status = `Names not kept: ${e instanceof Error ? e.message : String(e)}`));
    }, NAMES_SAVE_DELAY_MS);
  }

  const who = (speaker: string) => names[speaker]?.trim() || speaker;
  const color = (speaker: string) => SPEAKER_COLORS[Math.max(0, speakers.indexOf(speaker)) % SPEAKER_COLORS.length];

  function exportAs(format: ExportFormat) {
    const spec = EXPORT_FORMATS.find((entry) => entry.format === format)!;
    const base = audioName.replace(/\.[^.]+$/, "") || "transcript";
    download(renderTranscript(segments, names, format), `${base}.${format}`, spec.mime);
  }

  // keeps the highlighted segment and word under the playhead while the audio plays
  function follow() {
    cancelAnimationFrame(frame);
    const step = () => {
      if (!player) return;
      locate(player.currentTime);
      if (!player.paused) frame = requestAnimationFrame(step);
    };
    step();
  }

  function locate(time: number) {
    position = time;
    const current = segments[activeSegment];
    const segment =
      current && time >= current.start && time < current.end
        ? activeSegment
        : segments.findIndex((s) => time >= s.start && time < s.end);
    const word = segment >= 0 ? (segments[segment].words ?? []).findLastIndex((w) => w.start <= time) : -1;
    if (segment !== activeSegment) {
      activeSegment = segment;
      if (segment >= 0) {
        void tick().then(() => document.getElementById(`segment-${segment}`)?.scrollIntoView({ block: "nearest" }));
      }
    }
    if (word !== activeWord) activeWord = word;
  }

  function seek(time: number) {
    if (!player) return;
    player.currentTime = time;
    void player.play();
  }

  function toggle() {
    if (!player) return;
    if (player.paused) void player.play();
    else player.pause();
  }

  // the sentence before or after the one playing, or after the playhead between sentences
  function step(direction: 1 | -1) {
    const time = player?.currentTime ?? 0;
    const from = activeSegment >= 0 ? activeSegment : segments.findLastIndex((s) => s.start <= time) + (direction < 0 ? 1 : 0);
    const target = segments[from + direction];
    if (target) seek(target.start);
  }

  const clock = (seconds: number) => {
    const total = Math.max(0, Math.floor(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = String(Math.floor(total / 60) % 60).padStart(2, "0");
    const secs = String(total % 60).padStart(2, "0");
    return hours > 0 ? `${hours}:${minutes}:${secs}` : `${minutes}:${secs}`;
  };

  const field = "rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-950";
  const plain = "rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800";
</script>

<div class="flex min-h-0 w-full flex-1 flex-col md:flex-row">
  <JobHistory kind="transcription" newLabel="New transcription" onOpen={open} onNew={startNew} {busy} />
  <div class="min-h-0 flex-1 overflow-y-auto">
    <div class="mx-auto flex w-full max-w-5xl flex-col gap-3 p-4">
      <SlotBar slot="transcription" />
      <SlotBar slot="diarization" />

      <div class="flex flex-wrap items-center gap-2">
        <label class="cursor-pointer {plain}">
          Choose audio...
          <input type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.mka,.mkv,.webm,video/x-matroska,video/webm,video/mp4" class="hidden" disabled={busy} onchange={(e) => choose(e.currentTarget)} />
        </label>
        <input class="w-36 {field}" placeholder="Language (auto)" aria-label="Language" bind:value={language} disabled={busy} />
        {#if busy}
          <button class={plain} onclick={() => controller?.abort()}>Cancel</button>
        {:else}
          <button
            class="rounded-md bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            onclick={run}
            disabled={!ready || !(audio || restored)}
          >
            Transcribe
          </button>
        {/if}
        <span class="min-w-0 truncate text-xs text-gray-500">
          {ready ? [audioName, status].filter(Boolean).join(": ") : "Load a speech recognition model first"}
        </span>
      </div>

      {#if progress}
        <div
          class="flex flex-col gap-1"
          role="progressbar"
          aria-label={progress.stage}
          aria-valuemin={0}
          aria-valuemax={progress.total || undefined}
          aria-valuenow={progress.total ? progress.done : undefined}
        >
          <div class="text-xs text-gray-600 dark:text-gray-300">
            {progress.stage}{progress.total ? ` ${progress.done} of ${progress.total}` : ""}
          </div>
          <div class="h-2 overflow-hidden rounded bg-gray-200 dark:bg-gray-800">
            {#if progress.total}
              <div class="h-full rounded bg-blue-600 transition-[width] duration-300" style="width: {(100 * progress.done) / progress.total}%"></div>
            {:else}
              <div class="indeterminate h-full w-2/5 rounded bg-blue-600"></div>
            {/if}
          </div>
        </div>
      {/if}

      {#if audioUrl}
        <audio
          bind:this={player}
          src={audioUrl}
          class="hidden"
          onplay={() => ((paused = false), follow())}
          onpause={() => (paused = true)}
          onloadedmetadata={() => player && ((length = player.duration), (position = player.currentTime))}
          onseeked={() => player && locate(player.currentTime)}
        ></audio>
        <AudioTransport {paused} {position} {length} unit="sentence" ontoggle={toggle} onseek={seek} onstep={step} />
      {/if}

      {#if segments.length > 0}
        <div class="grid gap-4 lg:grid-cols-[1fr_16rem]">
          <div class="flex min-w-0 flex-col gap-2">
            {#each segments as segment, index (index)}
              <div
                id="segment-{index}"
                class="rounded-lg border p-3 text-sm leading-relaxed {index === activeSegment
                  ? 'border-blue-400 bg-blue-50/60 dark:border-blue-700 dark:bg-blue-950/30'
                  : 'border-gray-200 dark:border-gray-800'}"
              >
                <div class="mb-1 flex items-baseline gap-2">
                  {#if segment.speaker}
                    <span class="font-semibold" style="color: {color(segment.speaker)}">{who(segment.speaker)}</span>
                  {/if}
                  <button class="font-mono text-xs text-gray-500 hover:text-blue-600" title="Play from here" onclick={() => seek(segment.start)}>
                    {clock(segment.start)} - {clock(segment.end)}
                  </button>
                </div>
                {#if segment.words?.length}
                  <p>
                    {#each segment.words as word, w (w)}
                      <button
                        class="rounded px-px text-left {index === activeSegment && w === activeWord
                          ? 'bg-yellow-200 text-gray-900 dark:bg-yellow-500/60 dark:text-white'
                          : index === activeSegment && w < activeWord
                            ? 'text-gray-900 dark:text-gray-100'
                            : 'text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800'}"
                        onclick={() => seek(word.start)}>{word.word}</button
                      >{" "}
                    {/each}
                  </p>
                {:else}
                  <p class="text-gray-700 dark:text-gray-300">{segment.text}</p>
                {/if}
              </div>
            {/each}
          </div>

          <aside class="flex flex-col gap-4 self-start lg:sticky lg:top-16">
            {#if speakers.length > 0}
              <section class="flex flex-col gap-2">
                <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-500">Speakers</h3>
                {#each speakers as speaker (speaker)}
                  <label class="flex items-center gap-2">
                    <span class="h-3 w-3 shrink-0 rounded-full" style="background: {color(speaker)}"></span>
                    <input
                      class="min-w-0 flex-1 {field}"
                      placeholder={speaker}
                      aria-label="Name for {speaker}"
                      value={names[speaker] ?? ""}
                      oninput={(e) => rename(speaker, e.currentTarget.value)}
                    />
                  </label>
                {/each}
              </section>
            {/if}
            <section class="flex flex-col gap-1.5">
              <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-500">Export</h3>
              {#each EXPORT_FORMATS as { format, label } (format)}
                <button class="text-left disabled:opacity-50 {plain}" disabled={busy} onclick={() => exportAs(format)}>{label}</button>
              {/each}
            </section>
          </aside>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .indeterminate {
    animation: sweep 1.4s ease-in-out infinite;
  }
  @keyframes sweep {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(250%);
    }
  }
</style>
