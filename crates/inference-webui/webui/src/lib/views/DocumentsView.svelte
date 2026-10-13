<script lang="ts">
  import { onDestroy } from "svelte";
  import type { PDFDocumentProxy } from "pdfjs-dist";
  import JobHistory from "../components/JobHistory.svelte";
  import SlotBar from "../components/SlotBar.svelte";
  import { jobsStore } from "../stores/jobs.svelte";
  import { managerStore, SLOTS, type SlotId } from "../stores/manager.svelte";
  import { detectLayout, readImage } from "../services/manager";
  import { blobOf, jobFileUrl, type Job } from "../services/jobs";
  import { openPdf, renderPage } from "../services/pdfPages";
  import {
    DOCUMENT_FORMATS,
    otslRows,
    renderDocument,
    type DocumentFormat,
    type ReadPage,
    type ReadRegion,
  } from "../services/documentExport";
  import { download } from "../services/transcriptExport";

  // PaddleOCR-VL's task prompts by PP-DocLayoutV3 label; every other text kind takes the text prompt
  const TASK_PROMPTS: Record<string, string> = {
    table: "Table Recognition:",
    display_formula: "Formula Recognition:",
    inline_formula: "Formula Recognition:",
    chart: "Chart Recognition:",
  };
  // the PP-DocLayoutV3 labels of regions with no text to read
  const PICTURES = new Set(["image", "header_image", "footer_image", "seal"]);
  // region badges scale with the page, so they read the same on a phone photo and a 300 dpi scan
  const BADGE_PAGE_FRACTION = 1 / 40;
  // a kept single-page job from before documents had pages
  const LEGACY_PAGE_FILE = "page.png";
  const pageFile = (index: number) => `page-${index + 1}.png`;
  // regions read at once: the engine batches them (3x at its default 4 sequences); past the OCR model's sequence
  // limit they wait in its queue, so a model loaded with more sequences reads faster still
  const READ_CONCURRENCY = 16;

  type Status = "new" | "working" | "laid out" | "read" | "failed";

  interface Page extends ReadPage {
    /** What shows the page: a blob URL of its render, or its kept job file. */
    url: string | null;
    blob: Blob | null;
    /** The job file holding the page, once kept. */
    file: string | null;
    status: Status;
    error: string;
  }

  interface DocumentsJob {
    name: string;
    textPrompt: string;
    pages: (ReadPage & { file: string | null })[];
  }

  interface LegacyJob {
    pageName: string;
    textPrompt: string;
    regions: ReadRegion[];
    text: string;
  }

  let name = $state("");
  let pages = $state<Page[]>([]);
  let current = $state(0);
  let pdf: PDFDocumentProxy | null = null;
  let textPrompt = $state("OCR:");
  let status = $state("");
  let busy = $state(false);
  let progress = $state<{ label: string; done: number; total: number } | null>(null);
  let controller: AbortController | null = null;

  let layoutReady = $derived(managerStore.servedIn("layout") != null);
  let ocrReady = $derived(managerStore.servedIn("ocr") != null);
  let page = $derived(pages[current]);
  let badge = $derived(Math.max(8, (page?.width ?? 0) * BADGE_PAGE_FRACTION));
  let readCount = $derived(pages.filter((p) => p.status === "read").length);

  onDestroy(() => {
    controller?.abort();
    release();
    void pdf?.destroy();
  });

  function release() {
    for (const p of pages) if (p.url?.startsWith("blob:")) URL.revokeObjectURL(p.url);
  }

  function blank(pageName: string): Page {
    return { name: pageName, width: 0, height: 0, regions: [], text: "", url: null, blob: null, file: null, status: "new", error: "" };
  }

  function reset(next: Page[], docName: string) {
    release();
    void pdf?.destroy();
    pdf = null;
    pages = next;
    current = 0;
    name = docName;
    jobsStore.current.documents = null;
  }

  // drawn upright into a PNG first: the browser applies a photo's EXIF rotation and the server's decoder does not,
  // so the boxes it returns and the crops cut here must come from the same pixels
  async function uprightPng(file: File): Promise<{ blob: Blob; width: number; height: number }> {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    canvas.getContext("2d")!.drawImage(bitmap, 0, 0);
    bitmap.close();
    const blob = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((png) => (png ? resolve(png) : reject(new Error("the image could not be drawn"))), "image/png"),
    );
    return { blob, width: canvas.width, height: canvas.height };
  }

  async function choose(file: File | undefined) {
    if (!file) return;
    status = "";
    try {
      if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
        status = "Opening the PDF";
        const opened = await openPdf(file);
        reset(
          Array.from({ length: opened.numPages }, (_, i) => blank(`${file.name}, page ${i + 1}`)),
          file.name,
        );
        pdf = opened;
        await show(0);
        status = `${opened.numPages} page${opened.numPages === 1 ? "" : "s"}`;
      } else {
        const drawn = await uprightPng(file);
        reset([{ ...blank(file.name), ...drawn, url: URL.createObjectURL(drawn.blob) }], file.name);
        status = `${drawn.width}x${drawn.height}`;
      }
    } catch (e) {
      status = `${file.name}: ${e instanceof Error ? e.message : String(e)}`;
    }
  }

  // the page's image, rendered from the PDF the first time it's needed, or read back from the job
  async function ensureImage(index: number): Promise<Blob> {
    const p = pages[index];
    if (p.blob) return p.blob;
    if (p.url) {
      p.blob = await blobOf(p.url);
      return p.blob;
    }
    if (!pdf) throw new Error("the page's image is gone");
    const rendered = await renderPage(pdf, index + 1);
    Object.assign(pages[index], { ...rendered, url: URL.createObjectURL(rendered.blob) });
    return rendered.blob;
  }

  // the page's image rendered and decoded, so swapping it in doesn't flash an empty frame
  async function ready(index: number) {
    await ensureImage(index);
    const url = pages[index].url;
    if (!url) return;
    const image = new Image();
    image.src = url;
    await image.decode().catch(() => {});
  }

  // the shown page stays until the next is ready; its neighbour renders meanwhile, for the page after
  async function show(index: number) {
    const target = Math.min(Math.max(index, 0), pages.length - 1);
    try {
      await ready(target);
      current = target;
      if (target + 1 < pages.length && pdf) void ready(target + 1).catch(() => {});
    } catch (e) {
      status = e instanceof Error ? e.message : String(e);
    }
  }

  const dataUrl = (blob: Blob) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });

  const loadImage = (blob: Blob) =>
    new Promise<HTMLImageElement>((resolve, reject) => {
      const image = new Image();
      const url = URL.createObjectURL(blob);
      image.onload = () => (URL.revokeObjectURL(url), resolve(image));
      image.onerror = () => (URL.revokeObjectURL(url), reject(new Error("the page image could not be read")));
      image.src = url;
    });

  // the region's box of `image`, clamped to it, as a PNG data URL
  function crop(image: HTMLImageElement, box: [number, number, number, number]): string {
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    const left = Math.min(Math.max(Math.floor(box[0]), 0), width - 1);
    const top = Math.min(Math.max(Math.floor(box[1]), 0), height - 1);
    const right = Math.min(Math.max(Math.ceil(box[2]), left + 1), width);
    const bottom = Math.min(Math.max(Math.ceil(box[3]), top + 1), height);
    const canvas = document.createElement("canvas");
    canvas.width = right - left;
    canvas.height = bottom - top;
    const context = canvas.getContext("2d")!;
    // a transparent page reads as paper, not black
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, left, top, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/png");
  }

  // one page through what's loaded: the layout when asked and not done yet, then the reading of each region
  async function processPage(index: number, steps: { layout: boolean; read: boolean }, signal: AbortSignal, total: number) {
    const p = pages[index];
    const label = (stage: string) => `${total > 1 ? `Page ${index + 1} of ${pages.length}: ` : ""}${stage}`;
    p.status = "working";
    p.error = "";
    try {
      const blob = await ensureImage(index);
      if (steps.layout) {
        progress = { label: label("detecting the layout"), done: 0, total: 0 };
        const result = await detectLayout(SLOTS.layout.id, await dataUrl(blob));
        p.regions = result.regions.map((region) => ({ ...region, text: "" }));
        p.text = "";
        p.status = "laid out";
      }
      if (steps.read) {
        const image = await loadImage(blob);
        const jobs =
          p.regions.length === 0
            ? [{ region: null as ReadRegion | null, box: [0, 0, image.naturalWidth, image.naturalHeight] as [number, number, number, number], prompt: textPrompt }]
            : p.regions
                .filter((region) => !PICTURES.has(region.label))
                .map((region) => ({ region, box: region.bbox, prompt: TASK_PROMPTS[region.label] ?? textPrompt }));
        p.text = "";
        let next = 0;
        let done = 0;
        progress = { label: label(`reading ${jobs.length} regions`), done: 0, total: jobs.length };
        // a few workers take the regions in turn; each result lands on its own region, so the order holds
        let failed = false;
        const worker = async () => {
          while (!failed && next < jobs.length) {
            if (signal.aborted) throw new DOMException("canceled", "AbortError");
            const job = jobs[next++];
            try {
              const read = await readImage(SLOTS.ocr.id, crop(image, job.box), job.prompt);
              if (job.region) job.region.text = read;
              else p.text = read;
            } catch (e) {
              // the other workers stop taking regions once one has failed
              failed = true;
              throw e;
            }
            done++;
            progress = { label: label(`read ${done} of ${jobs.length} regions`), done, total: jobs.length };
          }
        };
        await Promise.all(Array.from({ length: Math.min(READ_CONCURRENCY, jobs.length) }, worker));
        p.status = "read";
      }
      if (!steps.layout && !steps.read) p.status = "new";
    } catch (e) {
      p.status = signal.aborted ? (p.regions.length ? "laid out" : "new") : "failed";
      p.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      await keep(steps.read ? "ocr" : "layout");
    }
  }

  async function run(which: "layout" | "read" | "all") {
    busy = true;
    status = "";
    controller = new AbortController();
    const signal = controller.signal;
    const started = performance.now();
    const targets = which === "all" ? pages.map((_, i) => i).filter((i) => pages[i].status !== "read") : [current];
    try {
      for (const [n, index] of targets.entries()) {
        if (signal.aborted) break;
        const p = pages[index];
        const steps = {
          layout: which === "layout" || (which === "all" && layoutReady && p.status === "new"),
          read: which === "read" || (which === "all" && ocrReady),
        };
        if (which === "all") await show(index);
        await processPage(index, steps, signal, which === "all" ? targets.length : 1);
        if (which === "all") progress = { label: `Page ${n + 1} of ${targets.length} done`, done: n + 1, total: targets.length };
      }
      status = `Done in ${((performance.now() - started) / 1000).toFixed(1)} s`;
    } catch (e) {
      status = signal.aborted ? "Canceled" : e instanceof Error ? e.message : String(e);
    } finally {
      progress = null;
      busy = false;
      controller = null;
    }
  }

  // the document kept as one job: each worked page's image once, and every page's reading
  async function keep(slot: SlotId) {
    const fresh = pages
      .map((p, index) => ({ p, index }))
      .filter(({ p }) => !p.file && p.blob && p.status !== "new" && p.status !== "working");
    const files = fresh.map(({ p, index }) => ({ name: pageFile(index), blob: p.blob! }));
    const data: DocumentsJob = {
      name,
      textPrompt,
      pages: pages.map((p, index) => ({
        name: p.name,
        width: p.width,
        height: p.height,
        regions: p.regions,
        text: p.text,
        file: p.file ?? (fresh.some((f) => f.index === index) ? pageFile(index) : null),
      })),
    };
    try {
      await jobsStore.save({ id: jobsStore.current.documents ?? undefined, kind: "documents", title: name, slot, data }, files);
      for (const { p, index } of fresh) p.file = pageFile(index);
    } catch (e) {
      status = `Not kept: ${e instanceof Error ? e.message : String(e)}`;
    }
  }

  function open(job: Job) {
    const data = $state.snapshot(job.data) as unknown as DocumentsJob & Partial<LegacyJob>;
    const kept: DocumentsJob["pages"] = data.pages ?? [
      {
        name: data.pageName ?? "",
        width: 0,
        height: 0,
        regions: data.regions ?? [],
        text: data.text ?? "",
        file: job.files.includes(LEGACY_PAGE_FILE) ? LEGACY_PAGE_FILE : null,
      },
    ];
    reset(
      kept.map((p) => ({
        ...p,
        url: p.file && job.files.includes(p.file) ? jobFileUrl(job, p.file) : null,
        blob: null,
        status: p.text || p.regions.some((r) => r.text) ? "read" : p.regions.length ? "laid out" : "new",
        error: "",
      })),
      data.name ?? data.pageName ?? job.title,
    );
    textPrompt = data.textPrompt ?? textPrompt;
    jobsStore.current.documents = job.id;
    status = "";
  }

  // a page kept before its size was recorded is measured once its image shows
  function measured(event: Event) {
    const image = event.currentTarget as HTMLImageElement;
    if (page && !page.width) Object.assign(page, { width: image.naturalWidth, height: image.naturalHeight });
  }

  function exportAs(format: DocumentFormat) {
    const spec = DOCUMENT_FORMATS.find((entry) => entry.format === format)!;
    const base = name.replace(/\.[^.]+$/, "") || "document";
    const read = pages.map(({ name, width, height, regions, text }) => ({ name, width, height, regions, text }));
    download(renderDocument(base, read, format), `${base}.${format}`, spec.mime);
  }

  const STATUS_COLORS: Record<Status, string> = {
    new: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300",
    working: "bg-blue-100 text-blue-700 animate-pulse dark:bg-blue-900/40 dark:text-blue-200",
    "laid out": "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200",
    read: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-200",
    failed: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200",
  };
  const button = "rounded-md px-4 py-1.5 text-sm font-medium disabled:opacity-50";
  const plain = "rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 disabled:opacity-50 dark:border-gray-700 dark:hover:bg-gray-800";
</script>

<div class="flex min-h-0 w-full flex-1 flex-col md:flex-row">
  <JobHistory kind="documents" onOpen={open} {busy} />
  <div class="min-h-0 flex-1 overflow-y-auto">
    <div class="flex w-full flex-col gap-3 p-4">
      <div class="grid gap-3 lg:grid-cols-2">
        <SlotBar slot="layout" />
        <SlotBar slot="ocr" />
      </div>

      <div class="flex flex-wrap items-center gap-2">
        <label class="cursor-pointer {plain}">
          Open page or PDF...
          <input type="file" accept="image/*,application/pdf,.pdf" class="hidden" disabled={busy} onchange={(e) => choose(e.currentTarget.files?.[0])} />
        </label>
        <button class="{button} bg-blue-600 text-white hover:bg-blue-700" onclick={() => run("layout")} disabled={!layoutReady || !page || busy}>
          Detect layout
        </button>
        <button class="{button} bg-blue-600 text-white hover:bg-blue-700" onclick={() => run("read")} disabled={!ocrReady || !page || busy}>
          Read
        </button>
        {#if pages.length > 1}
          <button
            class="{button} bg-blue-600 text-white hover:bg-blue-700"
            onclick={() => run("all")}
            disabled={!(layoutReady || ocrReady) || busy || readCount === pages.length}
            title="The layout of each page not yet laid out, then its reading, for every page not yet read"
          >
            Process all pages
          </button>
        {/if}
        {#if busy}
          <button class={plain} onclick={() => controller?.abort()}>Cancel</button>
        {/if}
        <label class="flex items-center gap-1 text-xs text-gray-500" title="The prompt for text regions, and for the whole page when nothing was detected">
          Text prompt
          <input class="w-28 rounded-md border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-700 dark:bg-gray-950" bind:value={textPrompt} />
        </label>
        <span class="text-xs text-gray-500">{[name, status].filter(Boolean).join(": ")}</span>
      </div>

      {#if progress}
        <div class="flex flex-col gap-1" role="progressbar" aria-label={progress.label} aria-valuemin={0} aria-valuemax={progress.total || undefined} aria-valuenow={progress.total ? progress.done : undefined}>
          <div class="text-xs text-gray-600 dark:text-gray-300">{progress.label}</div>
          <div class="h-2 overflow-hidden rounded bg-gray-200 dark:bg-gray-800">
            {#if progress.total}
              <div class="h-full rounded bg-blue-600 transition-[width] duration-300" style="width: {(100 * progress.done) / progress.total}%"></div>
            {:else}
              <div class="indeterminate h-full w-2/5 rounded bg-blue-600"></div>
            {/if}
          </div>
        </div>
      {/if}

      {#if pages.length > 0}
        <div class="flex flex-wrap items-center gap-2">
          {#if pages.length > 1}
            <button class="{plain} px-2 py-1" onclick={() => show(current - 1)} disabled={current === 0} aria-label="Previous page">&lsaquo;</button>
            <span class="text-sm">Page {current + 1} of {pages.length}</span>
            <button class="{plain} px-2 py-1" onclick={() => show(current + 1)} disabled={current === pages.length - 1} aria-label="Next page">&rsaquo;</button>
            <span class="text-xs text-gray-500">{readCount} of {pages.length} read</span>
            <div class="flex flex-wrap gap-1">
              {#each pages as p, index (index)}
                <button
                  class="h-6 min-w-6 rounded px-1 text-[11px] font-medium {STATUS_COLORS[p.status]} {index === current ? 'ring-2 ring-blue-500' : ''}"
                  onclick={() => show(index)}
                  title="Page {index + 1}: {p.status}{p.error ? ` (${p.error})` : ''}"
                  aria-label="Page {index + 1}, {p.status}"
                  aria-current={index === current ? "page" : undefined}
                >
                  {index + 1}
                </button>
              {/each}
            </div>
          {/if}
          <span class="ml-auto flex flex-wrap items-center gap-1.5">
            <span class="text-xs text-gray-500">Export</span>
            {#each DOCUMENT_FORMATS as { format, label } (format)}
              <button class="{plain} px-2 py-1 text-xs" onclick={() => exportAs(format)} disabled={readCount === 0 && !pages.some((p) => p.regions.length)}>
                {label}
              </button>
            {/each}
          </span>
        </div>
      {/if}

      {#if page}
        {#if page.error}
          <div class="rounded bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/40 dark:text-red-300">{page.error}</div>
        {/if}
        <div class="grid min-h-0 gap-3 lg:grid-cols-2">
          <div class="self-start">
            {#if page.url}
              <!-- measures a page kept before its size was recorded; the page shows in the SVG with its boxes -->
              <img src={page.url} alt="" class="hidden" onload={measured} />
            {/if}
            {#if page.url && page.width > 0}
              <svg class="w-full rounded border border-gray-200 dark:border-gray-800" viewBox="{-badge * 2} 0 {page.width + badge * 2} {page.height}">
                <image href={page.url} x="0" y="0" width={page.width} height={page.height} />
                {#each page.regions as region, index (index)}
                  <rect
                    x={region.bbox[0]}
                    y={region.bbox[1]}
                    width={region.bbox[2] - region.bbox[0]}
                    height={region.bbox[3] - region.bbox[1]}
                    fill="rgba(59,130,246,0.08)"
                    stroke="rgb(59,130,246)"
                    stroke-width={Math.max(1, page.width / 600)}
                  />
                  <text x={region.bbox[0] - badge * 1.8} y={region.bbox[1] + badge} font-size={badge} fill="rgb(37,99,235)" font-weight="600">
                    {index + 1}
                  </text>
                {/each}
              </svg>
            {:else}
              <div class="flex h-64 items-center justify-center rounded border border-dashed border-gray-300 text-xs text-gray-500 dark:border-gray-700">
                Rendering the page...
              </div>
            {/if}
          </div>
          <div class="flex flex-col gap-2">
            {#each page.regions as region, index (index)}
              <div class="rounded border border-gray-200 p-2 text-sm dark:border-gray-800">
                <div class="text-xs text-gray-500">{index + 1}. {region.label} ({region.score.toFixed(2)})</div>
                {#if region.text}
                  {@const rows = region.label === "table" ? otslRows(region.text) : null}
                  {#if rows}
                    <div class="mt-1 overflow-x-auto">
                      <table class="text-xs">
                        {#each rows as row, r (r)}
                          <tr>
                            {#each row as cell, c (c)}
                              <td class="border border-gray-300 px-1.5 py-0.5 dark:border-gray-700 {r === 0 ? 'bg-gray-50 font-medium dark:bg-gray-900' : ''}">{cell}</td>
                            {/each}
                          </tr>
                        {/each}
                      </table>
                    </div>
                  {:else}
                    <div class="mt-1 font-mono text-xs whitespace-pre-wrap">{region.text}</div>
                  {/if}
                {/if}
              </div>
            {/each}
            {#if page.text && page.regions.length === 0}
              <div class="rounded border border-gray-200 p-2 font-mono text-xs whitespace-pre-wrap dark:border-gray-800">{page.text}</div>
            {/if}
          </div>
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
