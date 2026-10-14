<script lang="ts">
  import { onDestroy, tick } from "svelte";
  import ImageGallery, { type GalleryItem } from "../components/ImageGallery.svelte";
  import JobHistory from "../components/JobHistory.svelte";
  import SlotBar from "../components/SlotBar.svelte";
  import { jobsStore } from "../stores/jobs.svelte";
  import { managerStore, SLOTS } from "../stores/manager.svelte";
  import { editImage, generateImage, type ImageOptions } from "../services/manager";
  import { blobOf, jobFile, jobFileUrl, keptExtension, pngOf, type Job } from "../services/jobs";

  const IMAGE_STEM = "image-";
  const REFERENCE_STEM = "reference-";
  const TURN_STEM = "turn-";
  // a refinement starts from this file of its job: the image that was anchored
  const ANCHOR_FILE = `${REFERENCE_STEM}1.png`;
  const SOURCE_SUFFIX = "-source.png";
  const REFINE_TITLE = "Refine: ";

  interface ThreadTurn {
    prompt: string;
    /** The job file holding the turn's image. */
    image: string;
    /** The job file it edited; absent, the turn before's image, or the first turn's the anchored file. */
    source?: string;
  }

  interface ImagesJob {
    prompt: string;
    width: number;
    height: number;
    steps: number;
    guidance: number;
    seed: number;
    /** The refinements, each an edit of an image in the job: one it made, or the anchored file. */
    thread?: ThreadTurn[];
  }

  interface Turn {
    prompt: string;
    url: string | null;
    file: string | null;
    source: string | null;
    status: "working" | "done" | "failed";
    error: string;
  }

  let prompt = $state("A lighthouse on a rocky coast at dusk, oil painting");
  // 0 leaves the size to the server: the model's default, or for an edit the reference's
  let width = $state(0);
  let height = $state(0);
  // 0 and -1 leave the model's own choice
  let steps = $state(0);
  let guidance = $state(0);
  let seed = $state(-1);
  let references = $state<File[]>([]);
  let previews = $state<string[]>([]);
  let images = $state<string[]>([]);
  // the job's file for each of `images`, once kept
  let imageFiles = $state<string[]>([]);
  let busy = $state(false);
  let status = $state("");

  let gallery = $state<{ items: GalleryItem[]; start: number } | null>(null);
  // the job shown: what it generated, from which prompt and references, then the refinements of its images
  let jobId = $state<string | null>(null);
  let made = $state<ImagesJob | null>(null);
  let madeReferences = $state<string[]>([]);
  let thread = $state<Turn[]>([]);
  // the image the refinement edits next; `file` is the job's file holding it, null for one from elsewhere
  let anchor = $state<{ url: string; caption: string; file: string | null } | null>(null);
  let refinePrompt = $state("");

  let ready = $derived(managerStore.servedIn("images") != null);
  onDestroy(() => release(previews));

  function release(urls: string[]) {
    for (const url of urls) if (url.startsWith("blob:")) URL.revokeObjectURL(url);
  }

  function chooseReferences(files: File[] | FileList | null) {
    release(previews);
    references = files ? [...files] : [];
    previews = references.map((file) => URL.createObjectURL(file));
  }

  // every image the history holds, newest first, each under the prompt that made it
  function historyItems(): GalleryItem[] {
    return jobsStore.lists.images.flatMap((job) => {
      const data = job.data as unknown as ImagesJob;
      const turns = (data.thread ?? []).filter((turn) => job.files.includes(turn.image));
      const generated = job.files.filter((name) => name.startsWith(IMAGE_STEM));
      return [
        ...[...turns].reverse().map((turn) => ({ url: jobFileUrl(job, turn.image), caption: turn.prompt, filename: turn.image })),
        ...generated.map((name) => ({ url: jobFileUrl(job, name), caption: data.prompt, filename: name })),
      ];
    });
  }

  function view(url: string) {
    const items = historyItems();
    const at = items.findIndex((item) => item.url === url);
    if (at >= 0) return void (gallery = { items, start: at });
    // an image not kept in the history is shown with the ones beside it on the page
    const shown = [
      ...images.map((image, i) => ({ url: image, caption: prompt, filename: `image-${i + 1}.png` })),
      ...thread.filter((turn) => turn.url).map((turn, i) => ({ url: turn.url!, caption: turn.prompt, filename: `turn-${i + 1}.png` })),
    ];
    gallery = { items: shown, start: Math.max(0, shown.findIndex((item) => item.url === url)) };
  }

  // the job's file holding the image at `url`, if it is one of the job shown
  function fileOf(url: string): string | null {
    const at = images.indexOf(url);
    if (at >= 0) return imageFiles[at] ?? null;
    return thread.find((turn) => turn.url === url)?.file ?? null;
  }

  // an image of the job shown is refined within it; one from elsewhere starts a refinement of its own
  async function anchorOn(item: GalleryItem) {
    gallery = null;
    const shown = images.includes(item.url) || thread.some((turn) => turn.url === item.url);
    if (!shown) clear();
    const file = fileOf(item.url);
    anchor = { url: item.url, caption: item.caption, file };
    refinePrompt = "";
    await tick();
    document.getElementById("refine")?.scrollIntoView({ behavior: "smooth", block: "start" });
    document.getElementById("refine-prompt")?.focus();
  }

  async function refineFile(file: File | undefined) {
    if (!file) return;
    // the thread keeps its starting image as a PNG, so another type is redrawn as one
    const image = file.type === "image/png" ? file : await pngOf(file);
    const url = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(image);
    });
    await anchorOn({ url, caption: file.name, filename: file.name });
  }

  function clear() {
    jobId = null;
    made = null;
    madeReferences = [];
    images = [];
    imageFiles = [];
    thread = [];
    anchor = null;
  }

  // the job as its conversation: what it generated, then each refinement, ready to go on from the last image
  async function open(job: Job) {
    const data = job.data as unknown as ImagesJob;
    ({ prompt, width, height, steps, guidance, seed } = data);
    status = "";
    chooseReferences(null);
    jobId = job.id;
    imageFiles = job.files.filter((name) => name.startsWith(IMAGE_STEM));
    images = imageFiles.map((name) => jobFileUrl(job, name));
    made = images.length > 0 ? data : null;
    const names = job.files.filter((name) => name.startsWith(REFERENCE_STEM));
    madeReferences = made ? names.map((name) => jobFileUrl(job, name)) : [];
    let previous = !made && job.files.includes(ANCHOR_FILE) ? ANCHOR_FILE : null;
    thread = (data.thread ?? []).map((turn) => {
      const source = turn.source ?? previous;
      previous = turn.image;
      return {
        prompt: turn.prompt,
        url: job.files.includes(turn.image) ? jobFileUrl(job, turn.image) : null,
        file: turn.image,
        source,
        status: "done",
        error: "",
      };
    });
    const last = thread.findLast((turn) => turn.url);
    anchor = last
      ? { url: last.url!, caption: last.prompt, file: last.file }
      : !made && job.files.includes(ANCHOR_FILE)
        ? { url: jobFileUrl(job, ANCHOR_FILE), caption: data.prompt, file: ANCHOR_FILE }
        : null;
    jobsStore.current.images = job.id;
    if (!made) return;
    // the references come back as files, so the job can be run again as it was
    busy = true;
    try {
      chooseReferences(await Promise.all(names.map((name) => jobFile(job, name))));
    } catch (e) {
      status = `References not read: ${e instanceof Error ? e.message : String(e)}`;
    } finally {
      busy = false;
    }
  }

  function formOptions(): ImageOptions | null {
    const options: ImageOptions = {};
    if (width > 0 && height > 0) Object.assign(options, { width, height });
    else if (width > 0 || height > 0) return null;
    if (steps > 0) options.steps = steps;
    if (guidance > 0) options.guidance_scale = guidance;
    if (seed >= 0) options.seed = seed;
    return options;
  }

  async function run() {
    const options = formOptions();
    if (!options) return void (status = "Give both width and height, or neither");
    busy = true;
    status = references.length > 0 ? "Editing" : "Generating";
    const started = performance.now();
    const job: ImagesJob = { prompt: prompt.trim(), width, height, steps, guidance, seed };
    const sent = [...references];
    clear();
    made = job;
    madeReferences = [...previews];
    try {
      images =
        sent.length > 0
          ? await editImage(SLOTS.images.id, job.prompt, sent, options)
          : await generateImage(SLOTS.images.id, job.prompt, options);
      status = `Done in ${((performance.now() - started) / 1000).toFixed(1)} s`;
      try {
        const outputs = await Promise.all(
          images.map(async (image, index) => ({ name: `${IMAGE_STEM}${index + 1}.png`, blob: await blobOf(image) })),
        );
        const given = await Promise.all(
          sent.map(async (file, index) => {
            const extension = keptExtension(file, "image");
            const blob = extension ? file : await pngOf(file);
            return { name: `${REFERENCE_STEM}${index + 1}.${extension ?? "png"}`, blob };
          }),
        );
        const kept = await jobsStore.save({ kind: "images", title: job.prompt, slot: "images", data: job }, [...outputs, ...given]);
        // shown from the history, so the gallery finds them among the rest
        jobId = kept.id;
        imageFiles = outputs.map((file) => file.name);
        images = imageFiles.map((name) => jobFileUrl(kept, name));
        madeReferences = given.map((file) => jobFileUrl(kept, file.name));
      } catch (e) {
        status += `; not kept: ${e instanceof Error ? e.message : String(e)}`;
      }
    } catch (e) {
      made = null;
      status = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  // one refinement: the anchored image edited as asked, which becomes the next anchor
  async function refine() {
    const ask = refinePrompt.trim();
    if (!anchor || !ask) return;
    const options = formOptions() ?? {};
    // the edit keeps the anchor's size unless the form names one
    delete options.width;
    delete options.height;
    busy = true;
    refinePrompt = "";
    const turn: Turn = { prompt: ask, url: null, file: null, source: anchor.file, status: "working", error: "" };
    thread = [...thread, turn];
    const at = thread.length - 1;
    try {
      const source = await blobOf(anchor.url);
      const start = new File([source], "anchor.png", { type: source.type || "image/png" });
      const [result] = await editImage(SLOTS.images.id, ask, [start], options);
      if (!result) throw new Error("the model returned no image");
      thread[at].url = result;
      thread[at].status = "done";
      anchor = { url: result, caption: ask, file: null };
      await keepTurn(at, start, result);
    } catch (e) {
      thread[at].status = "failed";
      thread[at].error = e instanceof Error ? e.message : String(e);
      refinePrompt = ask;
    } finally {
      busy = false;
      await tick();
      document.getElementById(`turn-${at}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  // each turn kept in the job it refines: a generation's, or one of its own holding the anchored image
  async function keepTurn(at: number, start: File, result: string) {
    const name = `${TURN_STEM}${at + 1}.png`;
    const files = [{ name, blob: await blobOf(result) }];
    if (!thread[at].source) {
      // an image from elsewhere: a new refinement's anchored file, or kept beside the turn that edited it
      const source = jobId ? `${TURN_STEM}${at + 1}${SOURCE_SUFFIX}` : ANCHOR_FILE;
      files.unshift({ name: source, blob: start });
      thread[at].source = source;
    }
    thread[at].file = name;
    const base = made ?? { prompt: thread[0].prompt, width: 0, height: 0, steps, guidance, seed };
    const data: ImagesJob = {
      ...base,
      thread: thread
        .filter((turn) => turn.file)
        .map((turn) => ({ prompt: turn.prompt, image: turn.file!, source: turn.source ?? undefined })),
    };
    const title = made ? made.prompt : `${REFINE_TITLE}${thread[0].prompt}`;
    try {
      const kept = await jobsStore.save({ id: jobId ?? undefined, kind: "images", title, slot: "images", data }, files);
      jobId = kept.id;
      const url = jobFileUrl(kept, name);
      thread[at].url = url;
      if (anchor?.url === result) anchor = { url, caption: anchor.caption, file: name };
    } catch (e) {
      thread[at].file = null;
      thread[at].error = `not kept: ${e instanceof Error ? e.message : String(e)}`;
    }
  }

  // the image a turn edited, shown beside its prompt unless it is the result just above
  function sourceOf(index: number): { url: string; label: string } | null {
    const source = thread[index].source;
    const above = index > 0 ? thread[index - 1].file : imageFiles.length === 1 ? imageFiles[0] : null;
    if (!source || source === above) return null;
    const at = imageFiles.indexOf(source);
    if (at >= 0) return { url: images[at], label: `image ${at + 1}` };
    const turn = thread.findIndex((other) => other.file === source);
    if (turn >= 0 && thread[turn].url) return { url: thread[turn].url!, label: `the result of "${thread[turn].prompt}"` };
    return jobId ? { url: jobFileUrl({ id: jobId } as Job, source), label: "the image it started from" } : null;
  }

  function refineKey(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void refine();
    }
  }

  const field = "rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-950";
  const plain = "rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 disabled:opacity-50 dark:border-gray-700 dark:hover:bg-gray-800";
</script>

<div class="flex min-h-0 w-full flex-1 flex-col md:flex-row">
  <JobHistory kind="images" onOpen={open} {busy} />
  <div class="min-h-0 flex-1 overflow-y-auto">
    <div class="mx-auto flex w-full max-w-5xl flex-col gap-3 p-4">
      <SlotBar slot="images" />

      <textarea class="min-h-20 {field}" aria-label="Prompt" bind:value={prompt}></textarea>
      <div class="flex flex-wrap items-end gap-2">
        <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Width (0: default)</span><input type="number" step="64" min="0" class="w-24 {field}" bind:value={width} /></label>
        <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Height (0: default)</span><input type="number" step="64" min="0" class="w-24 {field}" bind:value={height} /></label>
        <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Steps (0: model's)</span><input type="number" min="0" class="w-24 {field}" bind:value={steps} /></label>
        <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Guidance (0: model's)</span><input type="number" min="0" step="0.5" class="w-24 {field}" bind:value={guidance} /></label>
        <label class="flex flex-col gap-1"><span class="text-[11px] text-gray-500">Seed (-1: random)</span><input type="number" min="-1" class="w-28 {field}" bind:value={seed} /></label>
        <label class="cursor-pointer {plain}" title="One image to edit, or several as subjects for a new scene (HiDream-O1)">
          Reference images...
          <input type="file" accept="image/*" multiple class="hidden" disabled={busy} onchange={(e) => chooseReferences(e.currentTarget.files)} />
        </label>
        {#if references.length > 0}
          <button class="text-xs text-gray-500 underline" onclick={() => chooseReferences(null)}>clear {references.length}</button>
        {/if}
        <label class="cursor-pointer {plain}" title="Start a refinement from an image of your own: describe each change and the model edits it">
          Refine an image...
          <input type="file" accept="image/*" class="hidden" disabled={busy} onchange={(e) => refineFile(e.currentTarget.files?.[0])} />
        </label>
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
      {#if made || thread.length > 0 || anchor}
        <section class="flex flex-col gap-3" aria-label="Conversation">
          {#if made}
            <div class="flex flex-col items-end gap-1.5 self-end">
              <div class="rounded-2xl rounded-br-sm bg-blue-600 px-3 py-2 text-sm whitespace-pre-wrap text-white">{made.prompt}</div>
              {#if madeReferences.length > 0}
                <div class="flex flex-wrap justify-end gap-1.5">
                  {#each madeReferences as reference (reference)}
                    <img src={reference} alt="Reference" class="h-12 rounded border border-gray-200 object-cover dark:border-gray-800" />
                  {/each}
                </div>
              {/if}
            </div>
            {#if images.length === 0}
              <div class="flex h-40 w-40 animate-pulse items-center justify-center rounded-lg bg-gray-100 text-xs text-gray-500 dark:bg-gray-800">Generating...</div>
            {/if}
            <div class="grid grid-cols-1 gap-3 md:grid-cols-2">
              {#each images as image, index (index)}
                <figure class="flex flex-col gap-1.5">
                  <button
                    class="overflow-hidden rounded-lg border {anchor?.url === image ? 'border-blue-500 ring-2 ring-blue-500' : 'border-gray-200 dark:border-gray-800'}"
                    onclick={() => view(image)}
                    aria-label="View image {index + 1}"
                  >
                    <img src={image} alt="Generated" class="w-full" />
                  </button>
                  <figcaption class="flex gap-1.5">
                    {#if anchor?.url !== image}
                      <button
                        class="rounded-md bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                        onclick={() => anchorOn({ url: image, caption: made!.prompt, filename: `image-${index + 1}.png` })}
                        disabled={busy}
                      >
                        Refine
                      </button>
                    {/if}
                    <button class="{plain} px-2 py-1 text-xs" onclick={() => view(image)}>View</button>
                    <a class="{plain} px-2 py-1 text-xs" href={image} download="image-{index + 1}.png">Download</a>
                  </figcaption>
                </figure>
              {/each}
            </div>
            {#if !anchor && thread.length === 0}
              <p class="text-xs text-gray-500">Refine an image to edit it step by step with the model, or click it to browse the gallery.</p>
            {/if}
          {/if}

          {#each thread as turn, index (index)}
            {@const from = sourceOf(index)}
            <div id="turn-{index}" class="flex flex-col gap-2">
              <div class="flex items-end gap-2 self-end">
                {#if from}
                  <button onclick={() => view(from.url)} aria-label="View the image this edits" title="Edits {from.label}">
                    <img src={from.url} alt="Edited" class="h-10 w-10 rounded object-cover opacity-80" />
                  </button>
                {/if}
                <div class="rounded-2xl rounded-br-sm bg-blue-600 px-3 py-2 text-sm whitespace-pre-wrap text-white">{turn.prompt}</div>
              </div>
              <div class="flex items-end gap-2 self-start">
                {#if turn.status === "working"}
                  <div class="flex h-40 w-40 items-center justify-center rounded-lg bg-gray-100 text-xs text-gray-500 animate-pulse dark:bg-gray-800">Editing...</div>
                {:else if turn.status === "failed"}
                  <div class="rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/40 dark:text-red-300">{turn.error}</div>
                {:else if turn.url}
                  <button onclick={() => view(turn.url!)} aria-label="View the result of: {turn.prompt}">
                    <img
                      src={turn.url}
                      alt={turn.prompt}
                      class="max-h-72 rounded-lg border {anchor?.url === turn.url ? 'border-blue-500 ring-2 ring-blue-500' : 'border-gray-200 dark:border-gray-800'}"
                    />
                  </button>
                  {#if anchor?.url !== turn.url}
                    <button class="{plain} px-2 py-1 text-xs" onclick={() => anchorOn({ url: turn.url!, caption: turn.prompt, filename: turn.file ?? "" })} disabled={busy}>
                      Anchor here
                    </button>
                  {/if}
                {/if}
                {#if turn.status === "done" && turn.error}<span class="text-xs text-amber-600">{turn.error}</span>{/if}
              </div>
            </div>
          {/each}

          {#if anchor}
            <div id="refine" class="flex items-end gap-2 rounded-lg border border-gray-200 p-2 dark:border-gray-800">
              <button onclick={() => view(anchor!.url)} aria-label="View the anchored image" title="The next edit starts from this image">
                <img src={anchor.url} alt="Anchored" class="h-14 w-14 rounded object-cover ring-2 ring-blue-500" />
              </button>
              <textarea
                id="refine-prompt"
                class="min-h-10 flex-1 resize-y {field}"
                rows="2"
                placeholder="Describe a change, e.g. make it night, add a boat"
                aria-label="Refinement"
                bind:value={refinePrompt}
                onkeydown={refineKey}
                disabled={busy}
              ></textarea>
              <button
                class="rounded-md bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                onclick={refine}
                disabled={!ready || !refinePrompt.trim() || busy}
              >
                Send
              </button>
              <button class={plain} onclick={() => (anchor = null)} disabled={busy} title="Stop refining; the conversation stays">Close</button>
            </div>
          {/if}
        </section>
      {/if}
    </div>
  </div>
</div>

{#if gallery}
  <ImageGallery items={gallery.items} start={gallery.start} onClose={() => (gallery = null)} onAnchor={anchorOn} />
{/if}
