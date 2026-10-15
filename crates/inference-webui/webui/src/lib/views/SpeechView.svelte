<script lang="ts">
  import { onDestroy, tick } from "svelte";
  import AudioTransport from "../components/AudioTransport.svelte";
  import JobHistory from "../components/JobHistory.svelte";
  import SlotBar from "../components/SlotBar.svelte";
  import { jobsStore } from "../stores/jobs.svelte";
  import { managerStore, SLOTS } from "../stores/manager.svelte";
  import { speakTimed, type SpokenWord } from "../services/manager";
  import { jobFile, jobFileUrl, type Job } from "../services/jobs";
  import { markdownBlocks, type TextBlock } from "../services/markdownBlocks";
  import { joinWavs } from "../services/wav";
  import { download } from "../services/transcriptExport";

  const SAMPLE =
    "# A short reading\n\nThis paragraph was spoken by inference.rs. Each paragraph is its own clip, generated in order.\n\nPlayback highlights the word being read; click any word to play from it.";
  // the reader waits this long after typing stops before splitting the text again
  const REPARSE_DELAY_MS = 400;
  const IPA_KEY = "inference-rs-tts-ipa";
  const SKIPPED_TITLE = "Nothing here to read aloud";
  const HASH_OFFSET = 0x811c9dc5;
  const HASH_PRIME = 0x01000193;

  // a skipped block has nothing to read aloud (a rule, a lone symbol), so playback passes over it
  type Status = "pending" | "changed" | "generating" | "ready" | "skipped" | "failed";

  interface Block extends TextBlock {
    /** What the audio depends on: the spoken text, the voice and the speed. */
    key: string;
    status: Status;
    audio: Blob | null;
    audioUrl: string | null;
    /** The kept job's file holding this block's audio. */
    file: string | null;
    duration: number;
    words: SpokenWord[];
    error: string;
  }

  interface SpeechJob {
    markdown: string;
    voice: string;
    speed: number;
    /** The audio kept for each block key, by job file. */
    clips?: { key: string; file: string; duration: number; words: SpokenWord[] }[];
    /** The keys of blocks with nothing to read aloud. */
    skipped?: string[];
  }

  let markdown = $state(SAMPLE);
  let voice = $state("af_heart");
  let speed = $state(1.0);
  let blocks = $state<Block[]>([]);
  let editing = $state(true);
  let ipa = $state(readIpa());
  let busy = $state(false);
  let status = $state("");
  let controller: AbortController | null = null;
  let reparseTimer: ReturnType<typeof setTimeout> | null = null;

  let player = $state<HTMLAudioElement | null>(null);
  // the block playing, and whether playback carries on into the blocks after it
  let playing = $state(-1);
  let continuous = $state(false);
  let activeWord = $state(-1);
  let frame = 0;
  let paused = $state(true);
  // seconds into the whole reading: the ready paragraphs played one after another
  let position = $state(0);
  // a seek asked for before a paragraph's audio has loaded
  let pendingSeek: number | null = null;

  let ready = $derived(managerStore.servedIn("speech") != null);
  let generated = $derived(blocks.filter((block) => block.status === "ready").length);
  // paragraphs needing no more work: those with audio and those with nothing to read aloud
  let finished = $derived(generated + blocks.filter((block) => block.status === "skipped").length);
  let total = $derived(blocks.length);
  // where each paragraph starts in the whole reading; a paragraph not yet generated takes no time
  let offsets = $derived(
    blocks.reduce<number[]>((at, block, i) => [...at, (at[i] ?? 0) + (block.status === "ready" ? block.duration : 0)], [0]),
  );
  let length = $derived(offsets[offsets.length - 1] ?? 0);

  function readIpa(): boolean {
    try {
      return localStorage.getItem(IPA_KEY) === "1";
    } catch {
      return false;
    }
  }

  function setIpa(on: boolean) {
    ipa = on;
    try {
      localStorage.setItem(IPA_KEY, on ? "1" : "0");
    } catch {
      // private windows may refuse storage
    }
  }

  function hash(text: string): string {
    let h = HASH_OFFSET;
    for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), HASH_PRIME) >>> 0;
    return h.toString(16).padStart(8, "0");
  }

  const keyOf = (block: TextBlock) => JSON.stringify([block.speech, voice.trim(), speed]);

  function release(block: Block) {
    if (block.audioUrl?.startsWith("blob:")) URL.revokeObjectURL(block.audioUrl);
  }

  // the blocks of the text now, keeping the audio of every block whose speech, voice and speed are unchanged
  function reparse(kept: Map<string, Partial<Block>> = new Map()) {
    const previous = new Map(
      blocks.filter((b) => b.status === "ready" || b.status === "skipped").map((b) => [b.key, b]),
    );
    const used = new Set<string>();
    const next = markdownBlocks(markdown).map((text, index): Block => {
      const key = keyOf(text);
      const reuse = previous.get(key) ?? kept.get(key);
      const done = reuse?.status === "ready" || reuse?.status === "skipped";
      if (done) used.add(key);
      const wasThere = blocks[index] && blocks[index].status !== "pending";
      return {
        ...text,
        key,
        status: done ? reuse.status! : wasThere ? "changed" : "pending",
        audio: reuse?.audio ?? null,
        audioUrl: reuse?.audioUrl ?? null,
        file: reuse?.file ?? null,
        duration: reuse?.duration ?? 0,
        words: reuse?.words ?? [],
        error: "",
      };
    });
    for (const block of previous.values()) if (!used.has(block.key)) release(block);
    if (playing >= 0) stop();
    blocks = next;
  }

  function edited() {
    if (reparseTimer) clearTimeout(reparseTimer);
    reparseTimer = setTimeout(() => reparse(), REPARSE_DELAY_MS);
  }

  reparse();

  onDestroy(() => {
    controller?.abort();
    cancelAnimationFrame(frame);
    for (const block of blocks) release(block);
  });

  async function chooseFile(file: File | undefined) {
    if (!file) return;
    markdown = await file.text();
    jobsStore.current.speech = null;
    status = file.name;
    reparse();
    editing = false;
  }

  async function generate() {
    busy = true;
    status = "";
    controller = new AbortController();
    const signal = controller.signal;
    const started = performance.now();
    let made = 0;
    let skipped = 0;
    try {
      for (const [index, block] of blocks.entries()) {
        if (block.status === "ready" || block.status === "skipped") continue;
        blocks[index].status = "generating";
        try {
          const spoken = await speakTimed(
            { model: SLOTS.speech.id, input: block.speech, voice: voice.trim() || undefined, speed },
            signal,
          );
          // the text may have been split again while this block was speaking
          const current = blocks[index];
          if (!current || current.key !== block.key) continue;
          if (spoken.duration === 0) {
            current.status = "skipped";
            skipped++;
            continue;
          }
          current.audio = spoken.audio;
          current.audioUrl = URL.createObjectURL(spoken.audio);
          current.duration = spoken.duration;
          current.words = spoken.words;
          current.file = null;
          current.status = "ready";
          made++;
          // playback waiting at the edge of what was generated carries on
          const waiting =
            playing >= 0 && playing < index && blocks.slice(playing + 1, index).every((b) => b.status === "skipped");
          if (continuous && waiting && player?.ended) play(index);
        } catch (e) {
          if (signal.aborted) throw e;
          blocks[index].status = "failed";
          blocks[index].error = e instanceof Error ? e.message : String(e);
        }
      }
      status = made > 0 ? `${made} generated in ${((performance.now() - started) / 1000).toFixed(1)} s` : "";
      if (made > 0 || skipped > 0) await keep();
    } catch (e) {
      status = signal.aborted ? "Canceled" : e instanceof Error ? e.message : String(e);
      for (const block of blocks) if (block.status === "generating") block.status = "pending";
    } finally {
      busy = false;
      controller = null;
    }
  }

  async function keep() {
    const fresh = blocks.filter((block) => block.status === "ready" && !block.file && block.audio);
    const files = fresh.map((block) => ({ name: `clip-${hash(block.key)}.wav`, blob: block.audio! }));
    for (const block of fresh) block.file = `clip-${hash(block.key)}.wav`;
    const data: SpeechJob = {
      markdown,
      voice: voice.trim(),
      speed,
      clips: blocks
        .filter((block) => block.status === "ready" && block.file)
        .map((block) => ({ key: block.key, file: block.file!, duration: block.duration, words: block.words })),
      skipped: blocks.filter((block) => block.status === "skipped").map((block) => block.key),
    };
    const title = blocks[0]?.text ?? "Speech";
    try {
      await jobsStore.save(
        { id: jobsStore.current.speech ?? undefined, kind: "speech", title, slot: "speech", data },
        files,
      );
    } catch (e) {
      status += `; not kept: ${e instanceof Error ? e.message : String(e)}`;
      for (const block of fresh) block.file = null;
    }
  }

  async function open(job: Job) {
    const data = $state.snapshot(job.data) as unknown as SpeechJob & { text?: string };
    stop();
    markdown = data.markdown ?? data.text ?? "";
    voice = data.voice ?? voice;
    speed = data.speed ?? speed;
    jobsStore.current.speech = job.id;
    status = "";
    for (const block of blocks) release(block);
    blocks = [];
    const kept = new Map<string, Partial<Block>>(
      (data.clips ?? [])
        .filter((clip) => job.files.includes(clip.file))
        .map((clip) => [
          clip.key,
          { status: "ready", file: clip.file, audioUrl: jobFileUrl(job, clip.file), duration: clip.duration, words: clip.words },
        ]),
    );
    for (const key of data.skipped ?? []) kept.set(key, { status: "skipped" });
    reparse(kept);
    editing = false;
  }

  // a kept clip is fetched only when something needs its bytes
  async function audioOf(block: Block): Promise<Blob | null> {
    if (block.audio) return block.audio;
    const id = jobsStore.current.speech;
    const job = jobsStore.lists.speech.find((entry) => entry.id === id);
    return job && block.file ? jobFile(job, block.file) : null;
  }

  function play(index: number, from = 0) {
    const block = blocks[index];
    if (!player || !block || block.status !== "ready" || !block.audioUrl) return;
    playing = index;
    activeWord = -1;
    if (player.src !== new URL(block.audioUrl, location.href).href) {
      // a seek before the new audio's metadata is in would be lost; it is applied once it loads
      pendingSeek = from;
      player.src = block.audioUrl;
    } else {
      player.currentTime = from;
    }
    void player.play();
    void tick().then(() => document.getElementById(`block-${index}`)?.scrollIntoView({ block: "nearest" }));
  }

  function loaded() {
    if (player && pendingSeek != null) player.currentTime = pendingSeek;
    pendingSeek = null;
  }

  // play, or carry on from where playback paused
  function toggle() {
    if (!player) return;
    if (!paused) return player.pause();
    continuous = true;
    if (playing >= 0) return void player.play();
    seekTo(position < length ? position : 0);
  }

  // the paragraph holding `time` in the whole reading, played from there
  function seekTo(time: number) {
    const index = blocks.findLastIndex((block, i) => block.status === "ready" && offsets[i] <= time);
    const first = blocks.findIndex((block) => block.status === "ready");
    const target = index >= 0 ? index : first;
    if (target < 0) return;
    continuous = true;
    play(target, Math.max(0, time - offsets[target]));
  }

  function step(direction: 1 | -1) {
    const from = playing >= 0 ? playing : 0;
    for (let i = from + direction; i >= 0 && i < blocks.length; i += direction) {
      if (blocks[i].status === "ready") return void ((continuous = true), play(i));
    }
  }

  function stop() {
    player?.pause();
    playing = -1;
    activeWord = -1;
    cancelAnimationFrame(frame);
  }

  function ended() {
    cancelAnimationFrame(frame);
    if (!continuous) return void (playing = -1);
    let next = playing + 1;
    while (blocks[next]?.status === "skipped") next++;
    // an unready block is either still to come (generation resumes playback) or the end of the reading
    if (blocks[next]?.status === "ready") play(next);
    else if (!(busy && blocks[next])) {
      playing = -1;
      position = 0;
    }
  }

  function follow() {
    cancelAnimationFrame(frame);
    const step = () => {
      if (!player || playing < 0) return;
      const words = blocks[playing]?.words ?? [];
      const word = words.findLastIndex((w) => w.start <= player!.currentTime);
      if (word !== activeWord) activeWord = word;
      position = (offsets[playing] ?? 0) + player.currentTime;
      if (!player.paused) frame = requestAnimationFrame(step);
    };
    step();
  }

  async function exportAs(format: "wav" | "json" | "md") {
    const base = (blocks[0]?.text ?? "speech").replace(/[^\w-]+/g, "-").slice(0, 40) || "speech";
    try {
      if (format === "md") return download(markdown, `${base}.md`, "text/markdown");
      const done = blocks.filter((block) => block.status === "ready");
      if (format === "json") {
        let offset = 0;
        const timed = done.map((block) => {
          const entry = {
            kind: block.kind,
            text: block.text,
            start: offset,
            end: offset + block.duration,
            words: block.words.map((w) => ({ ...w, start: w.start + offset, end: w.end + offset })),
          };
          offset += block.duration;
          return entry;
        });
        return download(JSON.stringify(timed, null, 2), `${base}.json`, "application/json");
      }
      const parts = await Promise.all(done.map(audioOf));
      const joined = await joinWavs(parts.filter((part): part is Blob => part != null));
      const url = URL.createObjectURL(joined);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${base}.wav`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (e) {
      status = `Export failed: ${e instanceof Error ? e.message : String(e)}`;
    }
  }

  // the words as shown: a heading's spoken full stop is the voice's, not the text's
  function shownWords(block: Block): SpokenWord[] {
    if (block.kind !== "heading" || block.speech === block.text || block.words.length === 0) return block.words;
    const words = [...block.words];
    const last = words.length - 1;
    words[last] = { ...words[last], word: words[last].word.replace(/\.$/, "") };
    return words;
  }

  const BADGES: Record<Status, string> = {
    pending: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300",
    changed: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200",
    generating: "bg-blue-100 text-blue-700 animate-pulse dark:bg-blue-900/40 dark:text-blue-200",
    ready: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-200",
    skipped: "bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-400",
    failed: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200",
  };
  const badge = (block: Block) =>
    block.status === "ready" ? `${block.duration.toFixed(1)} s` : block.status === "changed" ? "changed" : block.status;
  const HEADING_SIZES = ["text-2xl", "text-xl", "text-lg", "text-base", "text-base", "text-base"];
  const textClass = (block: Block) =>
    block.kind === "heading"
      ? `font-semibold ${HEADING_SIZES[Math.min(block.level, 6) - 1]}`
      : block.kind === "quote"
        ? "italic text-gray-600 dark:text-gray-400"
        : "";

  const field = "rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-950";
  const plain = "rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 disabled:opacity-50 dark:border-gray-700 dark:hover:bg-gray-800";
</script>

<div class="flex min-h-0 w-full flex-1 flex-col md:flex-row">
  <JobHistory kind="speech" onOpen={open} {busy} />
  <div class="min-h-0 flex-1 overflow-y-auto">
    <div class="mx-auto flex w-full max-w-4xl flex-col gap-3 p-4">
      <SlotBar slot="speech" />

      <div class="flex flex-wrap items-end gap-2">
        <label class="flex flex-col gap-1">
          <span class="text-[11px] text-gray-500" title="Kokoro: af_heart, bm_george, ...; comma-separated voices blend">Voice</span>
          <input class="w-36 {field}" bind:value={voice} onchange={() => reparse()} disabled={busy} />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-[11px] text-gray-500">Speed</span>
          <input type="number" min="0.5" max="2" step="0.1" class="w-20 {field}" bind:value={speed} onchange={() => reparse()} disabled={busy} />
        </label>
        <label class="cursor-pointer {plain}">
          Open text...
          <input type="file" accept=".md,.markdown,.txt,text/markdown,text/plain" class="hidden" disabled={busy} onchange={(e) => chooseFile(e.currentTarget.files?.[0])} />
        </label>
        <button class={plain} onclick={() => (editing = !editing)} aria-pressed={editing}>{editing ? "Done editing" : "Edit text"}</button>
        {#if busy}
          <button class={plain} onclick={() => controller?.abort()}>Cancel</button>
        {:else}
          <button
            class="rounded-md bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            onclick={generate}
            disabled={!ready || finished === total}
          >
            Generate
          </button>
        {/if}
        <label class="flex items-center gap-1.5 pb-1.5 text-sm" title="The IPA of each word above it, once its paragraph is generated">
          <input type="checkbox" class="accent-blue-600" checked={ipa} onchange={(e) => setIpa(e.currentTarget.checked)} />
          IPA
        </label>
      </div>

      <div class="flex flex-wrap items-center gap-2 text-xs text-gray-500">
        <span>{ready ? `${finished} of ${total} paragraphs generated` : "Load a speech synthesis model first"}</span>
        {#if status}<span>{status}</span>{/if}
        <span class="ml-auto flex gap-1.5">
          <button class="{plain} px-2 py-1 text-xs" onclick={() => exportAs("wav")} disabled={generated === 0 || busy}>WAV</button>
          <button class="{plain} px-2 py-1 text-xs" onclick={() => exportAs("json")} disabled={generated === 0}>Timings (JSON)</button>
          <button class="{plain} px-2 py-1 text-xs" onclick={() => exportAs("md")}>Markdown</button>
        </span>
      </div>
      {#if total > 0}
        <div class="h-1.5 overflow-hidden rounded bg-gray-200 dark:bg-gray-800" role="progressbar" aria-label="Paragraphs generated" aria-valuemin={0} aria-valuemax={total} aria-valuenow={finished}>
          <div class="h-full bg-green-600 transition-[width] duration-300" style="width: {(100 * finished) / total}%"></div>
        </div>
      {/if}

      {#if editing}
        <textarea
          class="min-h-48 font-mono {field}"
          aria-label="Text to read, in markdown"
          bind:value={markdown}
          oninput={edited}
          disabled={busy}
        ></textarea>
      {/if}

      <audio
        bind:this={player}
        class="hidden"
        onplay={() => ((paused = false), follow())}
        onpause={() => (paused = true)}
        onseeked={follow}
        onended={ended}
        onloadedmetadata={loaded}
      ></audio>

      {#if generated > 0}
        <AudioTransport {paused} {position} {length} unit="paragraph" ontoggle={toggle} onseek={seekTo} onstep={step} />
      {/if}

      <div class="flex flex-col gap-2">
        {#each blocks as block, index (index)}
          <div
            id="block-{index}"
            class="group rounded-lg border p-3 {index === playing
              ? 'border-blue-400 bg-blue-50/60 dark:border-blue-700 dark:bg-blue-950/30'
              : 'border-gray-200 dark:border-gray-800'}"
          >
            <div class="mb-1 flex items-center gap-2 text-[11px]">
              <span class="text-gray-400">{block.kind === "item" ? "list item" : block.kind} {index + 1}</span>
              <span class="rounded px-1.5 py-0.5 font-medium {BADGES[block.status]}" title={block.error || (block.status === "skipped" ? SKIPPED_TITLE : undefined)}>{badge(block)}</span>
              {#if block.status === "ready"}
                <button class="text-gray-500 hover:text-blue-600" onclick={() => ((continuous = false), play(index))} title="Play this paragraph">Play</button>
                <button class="text-gray-500 hover:text-blue-600" onclick={() => ((continuous = true), play(index))} title="Play from this paragraph on">Play from here</button>
              {/if}
            </div>
            <p class="leading-loose {textClass(block)}" class:pl-4={block.kind === "item"}>
              {#if block.kind === "item"}<span class="-ml-3 mr-1 text-gray-400">&bull;</span>{/if}
              {#if block.status === "ready" && block.words.length}
                {#each shownWords(block) as word, w (w)}
                  <button
                    class="rounded px-px text-left {index === playing && w === activeWord
                      ? 'bg-yellow-200 text-gray-900 dark:bg-yellow-500/60 dark:text-white'
                      : 'hover:bg-gray-100 dark:hover:bg-gray-800'}"
                    onclick={() => ((continuous = true), play(index, word.start))}
                  >
                    {#if ipa && word.ipa}
                      <ruby>{word.word}<rt class="ipa text-gray-500 dark:text-gray-400">{word.ipa}</rt></ruby>
                    {:else}
                      {word.word}
                    {/if}
                  </button>{" "}
                {/each}
              {:else}
                {block.text}
              {/if}
            </p>
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>

<style>
  .ipa {
    font-family: "Charis SIL", "Doulos SIL", "Gentium Plus", "DejaVu Sans", "Noto Sans", sans-serif;
    font-size: 0.6em;
    font-weight: normal;
    font-style: normal;
  }
  ruby {
    ruby-position: over;
  }
</style>
