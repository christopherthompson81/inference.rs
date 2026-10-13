<script lang="ts">
  import JobHistory from "../components/JobHistory.svelte";
  import SlotBar from "../components/SlotBar.svelte";
  import { jobsStore } from "../stores/jobs.svelte";
  import { managerStore, SLOTS, type SlotId } from "../stores/manager.svelte";
  import { detectLayout, readImage, type LayoutRegion } from "../services/manager";
  import { blobOf, jobFileUrl, type Job } from "../services/jobs";

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

  const PAGE_FILE = "page.png";

  interface Region extends LayoutRegion {
    text: string;
  }

  interface DocumentsJob {
    pageName: string;
    textPrompt: string;
    regions: Region[];
    text: string;
  }

  let page = $state<string | null>(null);
  let pageName = $state("");
  let pageImage = $state<HTMLImageElement | null>(null);
  let size = $state({ width: 0, height: 0 });
  let regions = $state<Region[]>([]);
  let textPrompt = $state("OCR:");
  let text = $state("");
  let status = $state("");
  let busy = $state(false);

  let layoutReady = $derived(managerStore.servedIn("layout") != null);
  let ocrReady = $derived(managerStore.servedIn("ocr") != null);
  let badge = $derived(Math.max(8, size.width * BADGE_PAGE_FRACTION));

  // drawn upright into a PNG first: the browser applies a photo's EXIF rotation and the server's decoder does not,
  // so the boxes it returns and the crops cut here must come from the same pixels
  async function choose(file: File | undefined) {
    if (!file) return;
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext("2d")!.drawImage(bitmap, 0, 0);
      bitmap.close();
      page = canvas.toDataURL("image/png");
      pageName = file.name;
      jobsStore.current.documents = null;
      regions = [];
      text = "";
      status = file.name;
    } catch (e) {
      status = `${file.name}: ${e instanceof Error ? e.message : String(e)}`;
    }
  }

  function open(job: Job) {
    const data = job.data as unknown as DocumentsJob;
    page = job.files.includes(PAGE_FILE) ? jobFileUrl(job, PAGE_FILE) : null;
    pageImage = null;
    size = { width: 0, height: 0 };
    ({ pageName, textPrompt, regions, text } = data);
    status = pageName;
    jobsStore.current.documents = job.id;
  }

  // a page's detection and reading are one job: the first keeps the page, the next update it
  async function keep(slot: SlotId) {
    if (!page) return;
    const id = jobsStore.current.documents ?? undefined;
    const data: DocumentsJob = { pageName, textPrompt, regions, text };
    try {
      const files = id ? [] : [{ name: PAGE_FILE, blob: await blobOf(page) }];
      await jobsStore.save({ id, kind: "documents", title: pageName, slot, data }, files);
    } catch (e) {
      status += `; not kept: ${e instanceof Error ? e.message : String(e)}`;
    }
  }

  function loaded(event: Event) {
    const image = event.currentTarget as HTMLImageElement;
    pageImage = image;
    size = { width: image.naturalWidth, height: image.naturalHeight };
    status = `${pageName}: ${size.width}x${size.height}`;
  }

  async function detect() {
    if (!page) return;
    busy = true;
    status = "Detecting layout";
    const started = performance.now();
    try {
      const result = await detectLayout(SLOTS.layout.id, page);
      regions = result.regions.map((region) => ({ ...region, text: "" }));
      status = `${regions.length} regions in ${((performance.now() - started) / 1000).toFixed(1)} s`;
      await keep("layout");
    } catch (e) {
      status = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  // the region's box of `image`, clamped to it, as a PNG data URL
  function crop(image: HTMLImageElement, size: { width: number; height: number }, box: [number, number, number, number]): string {
    const left = Math.min(Math.max(Math.floor(box[0]), 0), size.width - 1);
    const top = Math.min(Math.max(Math.floor(box[1]), 0), size.height - 1);
    const right = Math.min(Math.max(Math.ceil(box[2]), left + 1), size.width);
    const bottom = Math.min(Math.max(Math.ceil(box[3]), top + 1), size.height);
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

  async function read() {
    if (!page || !pageImage) return;
    const jobs: { region: Region | null; box: [number, number, number, number]; prompt: string }[] =
      regions.length === 0
        ? [{ region: null, box: [0, 0, size.width, size.height], prompt: textPrompt }]
        : regions
            .filter((region) => !PICTURES.has(region.label))
            .map((region) => ({ region, box: region.bbox, prompt: TASK_PROMPTS[region.label] ?? textPrompt }));
    busy = true;
    text = "";
    const started = performance.now();
    // the page as it was when reading began
    const image = pageImage;
    const pageSize = { ...size };
    try {
      for (const [index, job] of jobs.entries()) {
        status = `Reading ${index + 1} of ${jobs.length}`;
        const read = await readImage(SLOTS.ocr.id, crop(image, pageSize, job.box), job.prompt);
        if (job.region) job.region.text = read;
        text += `${read}\n\n`;
      }
      status = `Read ${jobs.length} regions in ${((performance.now() - started) / 1000).toFixed(1)} s`;
    } catch (e) {
      status = e instanceof Error ? e.message : String(e);
    } finally {
      // what was read before a failure is kept too
      if (text) await keep("ocr");
      busy = false;
    }
  }

  const button = "rounded-md px-4 py-1.5 text-sm font-medium disabled:opacity-50";
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
        <label class="cursor-pointer rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800">
          Choose page...
          <input type="file" accept="image/*" class="hidden" disabled={busy} onchange={(e) => choose(e.currentTarget.files?.[0])} />
        </label>
        <button class="{button} bg-blue-600 text-white hover:bg-blue-700" onclick={detect} disabled={!layoutReady || !page || busy}>
          Detect layout
        </button>
        <button class="{button} bg-blue-600 text-white hover:bg-blue-700" onclick={read} disabled={!ocrReady || !page || busy}>
          Read
        </button>
        <label class="flex items-center gap-1 text-xs text-gray-500" title="The prompt for text regions, and for the whole page when nothing was detected">
          Text prompt
          <input class="w-28 rounded-md border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-700 dark:bg-gray-950" bind:value={textPrompt} />
        </label>
        <span class="text-xs text-gray-500">{status}</span>
      </div>

      {#if page}
        <div class="grid min-h-0 gap-3 lg:grid-cols-2">
          <div class="self-start">
            <!-- measured and cropped from; the page shows in the SVG, in page pixels with the boxes -->
            <img src={page} alt="" class="hidden" onload={loaded} />
            {#if size.width > 0}
              <svg class="w-full rounded border border-gray-200 dark:border-gray-800" viewBox="{-badge * 2} 0 {size.width + badge * 2} {size.height}">
                <image href={page} x="0" y="0" width={size.width} height={size.height} />
                {#each regions as region, index (index)}
                  <rect
                    x={region.bbox[0]}
                    y={region.bbox[1]}
                    width={region.bbox[2] - region.bbox[0]}
                    height={region.bbox[3] - region.bbox[1]}
                    fill="rgba(59,130,246,0.08)"
                    stroke="rgb(59,130,246)"
                    stroke-width={Math.max(1, size.width / 600)}
                  />
                  <text x={region.bbox[0] - badge * 1.8} y={region.bbox[1] + badge} font-size={badge} fill="rgb(37,99,235)" font-weight="600">
                    {index + 1}
                  </text>
                {/each}
              </svg>
            {/if}
          </div>
          <div class="flex flex-col gap-2">
            {#each regions as region, index (index)}
              <div class="rounded border border-gray-200 p-2 text-sm dark:border-gray-800">
                <div class="text-xs text-gray-500">{index + 1}. {region.label} ({region.score.toFixed(2)})</div>
                {#if region.text}
                  <div class="mt-1 font-mono text-xs whitespace-pre-wrap">{region.text}</div>
                {/if}
              </div>
            {/each}
            {#if text && regions.length === 0}
              <div class="rounded border border-gray-200 p-2 font-mono text-xs whitespace-pre-wrap dark:border-gray-800">{text}</div>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>
