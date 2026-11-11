<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import ImageGallery, { type GalleryItem } from "../components/ImageGallery.svelte";
  import MeshViewer, { type MeshFormat } from "../components/MeshViewer.svelte";
  import SplatViewer, { type SplatFrame } from "../components/SplatViewer.svelte";
  import {
    cancelJob,
    deleteJob,
    followJob,
    getJob,
    isFinished,
    jobFileUrl,
    listJobs,
    submitImageTo3d,
    submitReconstruct,
    submitVoxels,
    submitWorld,
    type JobEvent,
    type JobRecord,
    type JobStatus,
    type Placement,
    type Trellis2Sampler,
    type Trellis2Variant,
  } from "../services/queuedJobs";

  // what the job list and a dropped event stream poll at
  const POLL_MS = 5000;
  const TICK_MS = 1000;
  const DEFAULT_STEPS = 8000;
  // the file each kind's viewer opens, and the splat report a world job's transform comes from
  const WORLD_SPLATS = "splats/splats.spz";
  const WORLD_SPLAT_REPORT = "splats/splats.json";
  const RECONSTRUCTION_SPLATS = "reconstruction/gaussians.ply";
  // the image jobs' meshes
  const MODEL_GLB = "model/model.glb";
  const VOXELS_PLY = "voxels/voxels.ply";
  const VOXELS_MESH = "voxels/mesh.ply";
  const DEFAULT_VOXEL_STEPS = 72;
  const TEXTURE_SIZES = [1024, 2048, 4096];
  const BADGES: Record<JobStatus, string> = {
    queued: "bg-gray-200 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
    running: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
    succeeded: "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300",
    failed: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
    cancelled: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  };
  const KINDS: Record<string, string> = { world: "World", reconstruct: "Reconstruction", image_to_3d: "Image to 3D", voxels: "Voxels" };
  type Mode = "world" | "reconstruct" | "image_to_3d" | "voxels";
  const MODES: { id: Mode; label: string }[] = [
    { id: "image_to_3d", label: "Object from an image" },
    { id: "voxels", label: "Voxels from an image" },
    { id: "world", label: "World from a panorama" },
    { id: "reconstruct", label: "Scene from photos or a video" },
  ];

  let jobs = $state<JobRecord[]>([]);
  let selected = $state<JobRecord | null>(null);
  let mode = $state<Mode>("image_to_3d");
  let panorama = $state<File | null>(null);
  let panoramaUrl = $state<string | null>(null);
  let photos = $state<File[]>([]);
  let image = $state<File | null>(null);
  let imageUrl = $state<string | null>(null);
  let variant = $state<Trellis2Variant>("1024_cascade");
  let textureSize = $state<number | null>(null);
  let vertexColors = $state(false);
  let remesh = $state(false);
  let faces = $state<number | null>(null);
  // pipeline.json's samplers, shown as placeholders
  const SAMPLER_STAGES = [
    { key: "structure", label: "Structure", defaults: [12, 7.5, 0.7, 5] },
    { key: "shape", label: "Shape", defaults: [12, 7.5, 0.5, 3] },
    { key: "texture", label: "Texture", defaults: [12, 1, 0, 3] },
  ] as const;
  const SAMPLER_FIELDS = [
    { key: "steps", label: "Steps", step: "1" },
    { key: "guidance_strength", label: "Guidance", step: "0.1" },
    { key: "guidance_rescale", label: "Rescale", step: "0.05" },
    { key: "rescale_t", label: "Time shift", step: "0.5" },
  ] as const;
  const blankSampler = (): Trellis2Sampler => ({ steps: null, guidance_strength: null, guidance_rescale: null, rescale_t: null });
  let samplers = $state<Record<(typeof SAMPLER_STAGES)[number]["key"], Trellis2Sampler>>({
    structure: blankSampler(),
    shape: blankSampler(),
    texture: blankSampler(),
  });
  // a sampler with nothing set is left out, and an emptied number input reads as null
  function changed(sampler: Trellis2Sampler): Trellis2Sampler | null {
    const clean = Object.fromEntries(
      Object.entries(sampler).map(([k, v]) => [k, typeof v === "number" && Number.isFinite(v) ? v : null]),
    ) as unknown as Trellis2Sampler;
    return Object.values(clean).some((v) => v !== null) ? clean : null;
  }
  let removeBackground = $state(false);
  let voxelSteps = $state<number | null>(null);
  let voxelMesh = $state(true);
  let prompt = $state("");
  let scene = $state<"" | "indoor" | "outdoor">("");
  let seed = $state<number | null>(null);
  let splats = $state(true);
  let steps = $state<number | null>(null);
  let targetSize = $state<number | null>(null);
  let placement = $state<"" | Placement>("");
  let busy = $state(false);
  let status = $state("");
  let now = $state(Date.now() / 1000);
  let gallery = $state<{ items: GalleryItem[]; start: number } | null>(null);
  let viewer = $state<{ url: string; frame: SplatFrame; sceneToSplats: number[][] | null; title: string } | null>(null);
  let meshViewer = $state<{ url: string; format: MeshFormat; flat: boolean; title: string } | null>(null);

  let unfollow: (() => void) | null = null;
  let poll: ReturnType<typeof setInterval> | null = null;
  let tick: ReturnType<typeof setInterval> | null = null;

  onMount(() => {
    void refresh();
    poll = setInterval(() => void refresh(), POLL_MS);
    tick = setInterval(() => (now = Date.now() / 1000), TICK_MS);
  });
  onDestroy(() => {
    unfollow?.();
    if (poll) clearInterval(poll);
    if (tick) clearInterval(tick);
    if (panoramaUrl) URL.revokeObjectURL(panoramaUrl);
    if (imageUrl) URL.revokeObjectURL(imageUrl);
  });

  async function refresh() {
    try {
      jobs = await listJobs();
      // a job whose stream dropped catches up from the list
      if (selected && !isFinished(selected.status) && !unfollow) {
        const fresh = jobs.find((job) => job.id === selected?.id);
        if (fresh) selected = fresh;
      }
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    }
  }

  function open(job: JobRecord) {
    unfollow?.();
    unfollow = null;
    selected = job;
    if (isFinished(job.status)) {
      void getJob(job.id).then((record) => (selected = record), () => {});
      return;
    }
    unfollow = followJob(
      job.id,
      (record) => (selected = record),
      (event) => apply(job.id, event),
      () => (unfollow = null),
    );
  }

  function apply(id: string, event: JobEvent) {
    // the list shows the followed job's status as it changes, not at the next poll
    const listed = jobs.find((entry) => entry.id === id);
    if (listed && (event.type === "status" || event.type === "finished")) listed.status = event.status;
    const job = selected;
    if (!job || job.id !== id) return;
    const at = Date.now() / 1000;
    if (event.type === "status") {
      job.status = event.status;
      if (event.status === "running") job.waiting = null;
    } else if (event.type === "waiting") job.waiting = event.reason;
    else if (event.type === "stage") {
      const last = job.stages.at(-1);
      if (last && last.finished_at == null) {
        last.finished_at = at;
        last.done = last.total;
      }
      job.stages.push({ name: event.name, total: event.total, done: 0, started_at: at });
    } else if (event.type === "progress") {
      const last = job.stages.at(-1);
      if (last) last.done = event.done;
    } else if (event.type === "preview") {
      job.previews.push({ stage: event.stage, caption: event.caption, file: event.file });
    } else {
      unfollow = null;
      // the record holds the outputs and report the events leave out
      void getJob(id).then((record) => (selected?.id === id ? (selected = record) : null), () => {});
      void refresh();
    }
  }

  function startNew() {
    unfollow?.();
    unfollow = null;
    selected = null;
    status = "";
  }

  function choosePanorama(files: FileList | null) {
    if (panoramaUrl) URL.revokeObjectURL(panoramaUrl);
    panorama = files?.[0] ?? null;
    panoramaUrl = panorama ? URL.createObjectURL(panorama) : null;
  }

  function chooseImage(files: FileList | null) {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    image = files?.[0] ?? null;
    imageUrl = image ? URL.createObjectURL(image) : null;
  }

  function send(): Promise<JobRecord> {
    const room = placement || null;
    switch (mode) {
      case "world":
        return submitWorld(panorama as File, { prompt, scene: scene || null, seed, splats, steps: splats ? steps : null, placement: room });
      case "reconstruct":
        return submitReconstruct(photos, { target_size: targetSize, fps: null, max_frames: null, placement: room });
      case "image_to_3d":
        return submitImageTo3d(image as File, {
          seed,
          variant,
          remove_background: removeBackground || null,
          vertex_colors: vertexColors || null,
          texture_size: vertexColors ? null : textureSize,
          remesh: remesh || null,
          faces: typeof faces === "number" && Number.isFinite(faces) ? faces : null,
          structure_sampler: changed(samplers.structure),
          shape_sampler: changed(samplers.shape),
          texture_sampler: changed(samplers.texture),
          placement: room,
        });
      case "voxels":
        return submitVoxels(image as File, { seed, steps: voxelSteps, remove_background: removeBackground || null, mesh: voxelMesh, placement: room });
    }
  }

  async function submit() {
    busy = true;
    status = "";
    try {
      const job = await send();
      await refresh();
      open(job);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function cancel(job: JobRecord) {
    try {
      await cancelJob(job.id);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    }
  }

  async function remove(job: JobRecord) {
    try {
      await deleteJob(job.id);
      startNew();
      await refresh();
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    }
  }

  function previewItems(job: JobRecord): GalleryItem[] {
    return job.previews.map((preview) => ({
      url: jobFileUrl(job.id, preview.file),
      caption: `${preview.stage}: ${preview.caption}`,
      filename: preview.file.split("/").pop() ?? preview.file,
    }));
  }

  async function view(job: JobRecord) {
    status = "";
    try {
      if (job.outputs.some((output) => output.file === WORLD_SPLATS)) {
        const report = await fetch(jobFileUrl(job.id, WORLD_SPLAT_REPORT)).then((r) => r.json());
        viewer = { url: jobFileUrl(job.id, WORLD_SPLATS), frame: "z_up", sceneToSplats: report.transform ?? null, title: title(job) };
      } else {
        viewer = { url: jobFileUrl(job.id, RECONSTRUCTION_SPLATS), frame: "opencv", sceneToSplats: null, title: title(job) };
      }
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    }
  }

  function viewable(job: JobRecord): boolean {
    return job.outputs.some((output) => output.file === WORLD_SPLATS || output.file === RECONSTRUCTION_SPLATS);
  }

  // the meshes a finished image job can be turned around in
  function meshes(job: JobRecord): { label: string; file: string; format: MeshFormat; flat: boolean }[] {
    const made = (file: string) => job.outputs.some((output) => output.file === file);
    const out: { label: string; file: string; format: MeshFormat; flat: boolean }[] = [];
    if (made(MODEL_GLB)) out.push({ label: "View model", file: MODEL_GLB, format: "glb", flat: false });
    if (made(VOXELS_PLY)) out.push({ label: "View voxels", file: VOXELS_PLY, format: "ply", flat: true });
    if (made(VOXELS_MESH)) out.push({ label: "View mesh", file: VOXELS_MESH, format: "ply", flat: false });
    return out;
  }

  function title(job: JobRecord): string {
    const prompt = typeof job.request.prompt === "string" ? job.request.prompt : "";
    if (prompt) return prompt;
    if (job.kind === "reconstruct" && job.inputs.length > 1) return `${job.inputs.length} photos`;
    return job.inputs[0] || KINDS[job.kind] || job.kind;
  }

  function duration(seconds: number): string {
    const s = Math.max(0, Math.round(seconds));
    return s < 60 ? `${s} s` : s < 3600 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
  }

  function when(seconds: number): string {
    return new Date(seconds * 1000).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" });
  }

  let ready = $derived(mode === "world" ? panorama != null : mode === "reconstruct" ? photos.length > 0 : image != null);
  const field = "rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-950";
  const plain = "rounded-md border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100 disabled:opacity-50 dark:border-gray-700 dark:hover:bg-gray-800";
  const primary = "rounded-md bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50";
</script>

<div class="flex min-h-0 w-full flex-1 flex-col md:flex-row">
  <aside class="flex max-h-48 shrink-0 flex-col border-b border-gray-200 md:max-h-none md:w-64 md:border-r md:border-b-0 dark:border-gray-800">
    <div class="p-3">
      <button class="w-full {plain}" onclick={startNew}>New</button>
    </div>
    <div class="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
      {#each jobs as job (job.id)}
        <button
          class="mb-1 flex w-full flex-col gap-0.5 rounded-md px-2 py-1.5 text-left text-sm {selected?.id === job.id
            ? 'bg-gray-100 dark:bg-gray-800'
            : 'hover:bg-gray-50 dark:hover:bg-gray-900'}"
          onclick={() => open(job)}
        >
          <span class="truncate">{title(job)}</span>
          <span class="flex items-center gap-1.5 text-[11px] text-gray-500">
            <span class="rounded px-1.5 py-0.5 font-medium {BADGES[job.status]}">{job.status}</span>
            {KINDS[job.kind] ?? job.kind}, {when(job.created_at)}
          </span>
        </button>
      {:else}
        <p class="px-2 text-xs text-gray-500">No jobs yet.</p>
      {/each}
    </div>
  </aside>

  <div class="min-h-0 flex-1 overflow-y-auto">
    <div class="mx-auto flex w-full max-w-5xl flex-col gap-4 p-4">
      {#if status}
        <p class="rounded-md bg-red-50 px-3 py-2 text-sm text-red-800 dark:bg-red-950/40 dark:text-red-300" role="alert">{status}</p>
      {/if}

      {#if !selected}
        <div class="flex flex-wrap gap-1 self-start rounded-lg bg-gray-100 p-1 text-sm dark:bg-gray-900" role="tablist">
          {#each MODES as entry (entry.id)}
            <button role="tab" aria-selected={mode === entry.id} class="rounded-md px-3 py-1 {mode === entry.id ? 'bg-white shadow-sm dark:bg-gray-800' : ''}" onclick={() => (mode = entry.id)}>{entry.label}</button>
          {/each}
        </div>

        {#if mode === "image_to_3d" || mode === "voxels"}
          {#if mode === "image_to_3d"}
            <p class="text-sm text-gray-600 dark:text-gray-400">
              TRELLIS.2 builds a textured mesh of the object in one photo: its background removed with BiRefNet, then a voxel structure, the shape and its PBR material, baked into a GLB. About 40 s at 1024 on a 24 GB card, with a 6 GB GPU peak.
            </p>
          {:else}
            <p class="text-sm text-gray-600 dark:text-gray-400">
              JoVox generates the 64^3 voxel structure of the object in one photo (its background removed with BiRefNet); TRELLIS.2's shape stage can then mesh it. Seconds for the voxels alone.
            </p>
          {/if}
          <label class="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-gray-300 p-4 text-sm text-gray-500 hover:border-blue-400 dark:border-gray-700">
            {#if imageUrl}
              <img src={imageUrl} alt="The chosen photo" class="max-h-64 w-full rounded object-contain" />
              <span>{image?.name}</span>
            {:else}
              <span>Choose a photo of one object (a cut-out with transparency keeps its own outline)</span>
            {/if}
            <input type="file" accept="image/*" class="hidden" onchange={(e) => chooseImage(e.currentTarget.files)} disabled={busy} />
          </label>
          <div class="flex flex-wrap items-end gap-3">
            {#if mode === "image_to_3d"}
              <label class="flex flex-col gap-1">
                <span class="text-[11px] text-gray-500">Resolution</span>
                <select class={field} bind:value={variant} disabled={busy}>
                  <option value="1024_cascade">1024 (512, then 1024)</option>
                  <option value="1536_cascade">1536 (512, then 1536; slower, finer)</option>
                  <option value="1024">1024 (direct)</option>
                  <option value="512">512 (faster, coarser)</option>
                </select>
              </label>
              <label class="flex flex-col gap-1">
                <span class="text-[11px] text-gray-500">Texture size</span>
                <select class={field} bind:value={textureSize} disabled={busy || vertexColors}>
                  <option value={null}>2048 (default)</option>
                  {#each TEXTURE_SIZES as size (size)}
                    <option value={size}>{size}</option>
                  {/each}
                </select>
              </label>
              <label class="flex items-center gap-1.5 pb-1.5 text-sm" title="A closed, single-sided shell around the surface before baking">
                <input type="checkbox" class="accent-blue-600" bind:checked={remesh} disabled={busy || vertexColors} />
                Remesh
              </label>
              <label class="flex items-center gap-1.5 pb-1.5 text-sm" title="Color and material per vertex instead of baked textures: no unwrapping or baking">
                <input type="checkbox" class="accent-blue-600" bind:checked={vertexColors} disabled={busy} />
                Vertex colors
              </label>
            {:else}
              <label class="flex flex-col gap-1">
                <span class="text-[11px] text-gray-500">Sampling steps</span>
                <input type="number" min="36" max="108" class="w-28 {field}" bind:value={voxelSteps} placeholder={String(DEFAULT_VOXEL_STEPS)} disabled={busy} />
              </label>
              <label class="flex items-center gap-1.5 pb-1.5 text-sm">
                <input type="checkbox" class="accent-blue-600" bind:checked={voxelMesh} disabled={busy} />
                Mesh it with TRELLIS.2
              </label>
            {/if}
            <label class="flex flex-col gap-1">
              <span class="text-[11px] text-gray-500">Seed</span>
              <input type="number" class="w-28 {field}" bind:value={seed} placeholder="0" disabled={busy} />
            </label>
            <label class="flex items-center gap-1.5 pb-1.5 text-sm" title="An opaque photo always has its background removed">
              <input type="checkbox" class="accent-blue-600" bind:checked={removeBackground} disabled={busy} />
              Remove the background of a cut-out too
            </label>
          </div>
          {#if mode === "image_to_3d"}
            <details class="text-sm">
              <summary class="cursor-pointer text-gray-600 dark:text-gray-400">Advanced: face count and samplers</summary>
              <div class="mt-2 flex flex-col gap-3">
                <label class="flex flex-col gap-1">
                  <span class="text-[11px] text-gray-500">Faces in the exported mesh</span>
                  <input type="number" min="1000" max="10000000" step="1000" class="w-36 {field}" bind:value={faces} placeholder="1000000" disabled={busy} />
                </label>
                <div class="overflow-x-auto">
                  <table class="text-left">
                    <thead>
                      <tr class="text-[11px] text-gray-500">
                        <th class="pr-2 font-normal"></th>
                        {#each SAMPLER_FIELDS as f (f.key)}
                          <th class="pr-2 font-normal">{f.label}</th>
                        {/each}
                      </tr>
                    </thead>
                    <tbody>
                      {#each SAMPLER_STAGES as stage (stage.key)}
                        <tr>
                          <td class="pr-2 text-gray-600 dark:text-gray-400">{stage.label}</td>
                          {#each SAMPLER_FIELDS as f, i (f.key)}
                            <td class="pr-2 pb-1">
                              <input
                                type="number"
                                step={f.step}
                                class="w-20 {field}"
                                bind:value={samplers[stage.key][f.key]}
                                placeholder={String(stage.defaults[i])}
                                disabled={busy}
                              />
                            </td>
                          {/each}
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
                <p class="text-[11px] text-gray-500">Blank fields keep the model's defaults. Guidance 1 turns guidance off; steps run 1 to 100.</p>
              </div>
            </details>
          {/if}
        {:else if mode === "world"}
          <p class="text-sm text-gray-600 dark:text-gray-400">
            WorldStereo-2 generates views along camera paths around an equirectangular panorama, WorldMirror 2.0 reconstructs them, and Gaussian splats are trained on the result. It takes the GPU to itself, about half an hour on a 24 GB card.
          </p>
          <label class="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-gray-300 p-4 text-sm text-gray-500 hover:border-blue-400 dark:border-gray-700">
            {#if panoramaUrl}
              <img src={panoramaUrl} alt="The chosen panorama" class="max-h-64 w-full rounded object-contain" />
              <span>{panorama?.name}</span>
            {:else}
              <span>Choose a panorama (2:1 equirectangular image)</span>
            {/if}
            <input type="file" accept="image/*" class="hidden" onchange={(e) => choosePanorama(e.currentTarget.files)} disabled={busy} />
          </label>
          <label class="flex flex-col gap-1">
            <span class="text-[11px] text-gray-500">Caption every view is generated with</span>
            <textarea class="min-h-16 {field}" bind:value={prompt} placeholder="A sunlit courtyard with stone paths, potted trees and wooden benches." disabled={busy}></textarea>
          </label>
          <div class="flex flex-wrap items-end gap-3">
            <label class="flex flex-col gap-1">
              <span class="text-[11px] text-gray-500">Scene</span>
              <select class={field} bind:value={scene} disabled={busy}>
                <option value="">Decide from the sky</option>
                <option value="indoor">Indoor</option>
                <option value="outdoor">Outdoor</option>
              </select>
            </label>
            <label class="flex flex-col gap-1">
              <span class="text-[11px] text-gray-500">Seed</span>
              <input type="number" class="w-28 {field}" bind:value={seed} placeholder="1024" disabled={busy} />
            </label>
            <label class="flex items-center gap-1.5 pb-1.5 text-sm">
              <input type="checkbox" class="accent-blue-600" bind:checked={splats} disabled={busy} />
              Train splats
            </label>
            {#if splats}
              <label class="flex flex-col gap-1">
                <span class="text-[11px] text-gray-500">Training steps</span>
                <input type="number" min="100" step="100" class="w-28 {field}" bind:value={steps} placeholder={String(DEFAULT_STEPS)} disabled={busy} />
              </label>
            {/if}
          </div>
        {:else}
          <p class="text-sm text-gray-600 dark:text-gray-400">
            WorldMirror 2.0 recovers cameras, depth, a point cloud and Gaussian splats from photos of one scene, or from a video sampled at one frame a second. It takes seconds.
          </p>
          <label class="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-gray-300 p-6 text-sm text-gray-500 hover:border-blue-400 dark:border-gray-700">
            <span>{photos.length ? `${photos.length} file${photos.length === 1 ? "" : "s"} chosen` : "Choose photos (up to 32) or one video"}</span>
            <input type="file" accept="image/*,video/*" multiple class="hidden" onchange={(e) => (photos = Array.from(e.currentTarget.files ?? []))} disabled={busy} />
          </label>
          <label class="flex flex-col gap-1 self-start">
            <span class="text-[11px] text-gray-500">Longest edge to infer at</span>
            <input type="number" min="140" step="14" class="w-28 {field}" bind:value={targetSize} placeholder="518" disabled={busy} />
          </label>
        {/if}

        <div class="flex flex-wrap items-end gap-3">
          <label class="flex flex-col gap-1">
            <span class="text-[11px] text-gray-500" title="What the job may unload from its GPU to make room">Make room</span>
            <select class={field} bind:value={placement} disabled={busy}>
              <option value="">The server's setting</option>
              <option value="fastest">Unload what it needs</option>
              <option value="when_quiet">Unload only idle models</option>
              <option value="no_eviction">Unload nothing; wait for room</option>
            </select>
          </label>
          <button class={primary} onclick={submit} disabled={!ready || busy}>{busy ? "Starting..." : "Start"}</button>
        </div>
      {:else}
        {@const job = selected}
        <div class="flex flex-wrap items-center gap-2">
          <h2 class="min-w-0 flex-1 truncate text-lg font-medium">{title(job)}</h2>
          <span class="rounded px-2 py-0.5 text-xs font-medium {BADGES[job.status]}">{job.status}</span>
          {#if job.device != null}
            <span class="text-xs text-gray-500">GPU {job.device}</span>
          {/if}
          {#if job.started_at}
            <span class="text-xs text-gray-500">{duration((job.finished_at ?? now) - job.started_at)}</span>
          {/if}
          {#if viewable(job)}
            <button class={primary} onclick={() => view(job)}>View splats</button>
          {/if}
          {#each meshes(job) as mesh, index (mesh.file)}
            <button class={index === 0 ? primary : plain} onclick={() => (meshViewer = { url: jobFileUrl(job.id, mesh.file), format: mesh.format, flat: mesh.flat, title: title(job) })}>{mesh.label}</button>
          {/each}
          {#if !isFinished(job.status)}
            <button class={plain} onclick={() => cancel(job)}>Cancel</button>
          {:else}
            <button class={plain} onclick={() => remove(job)}>Delete</button>
          {/if}
        </div>
        {#if job.error}
          <p class="rounded-md bg-red-50 px-3 py-2 text-sm text-red-800 dark:bg-red-950/40 dark:text-red-300">{job.error}</p>
        {/if}
        {#if job.status === "queued"}
          <p class="text-sm text-gray-500">{job.waiting ? `Waiting: ${job.waiting}` : "Waiting for the jobs ahead of it, then for room on a GPU."}</p>
        {/if}

        {#if job.stages.length}
          <ol class="flex flex-col gap-2">
            {#each job.stages as stage, index (index)}
              {@const running = stage.finished_at == null && job.status === "running"}
              <li class="flex flex-col gap-1">
                <div class="flex items-center gap-2 text-sm">
                  <span class="flex-1 capitalize {running ? 'font-medium' : 'text-gray-600 dark:text-gray-400'}">{stage.name}</span>
                  {#if stage.total > 0}<span class="text-xs text-gray-500">{stage.done} of {stage.total}</span>{/if}
                  <span class="w-24 text-right text-xs text-gray-500">{duration((stage.finished_at ?? (running ? now : stage.started_at)) - stage.started_at)}</span>
                </div>
                <div class="h-1.5 overflow-hidden rounded bg-gray-200 dark:bg-gray-800" role="progressbar" aria-label={stage.name} aria-valuemin={0} aria-valuemax={stage.total || 1} aria-valuenow={stage.finished_at != null ? stage.total || 1 : stage.done}>
                  {#if stage.finished_at != null}
                    <div class="h-full w-full bg-green-600"></div>
                  {:else if stage.total > 0}
                    <div class="h-full bg-blue-600 transition-[width] duration-300" style="width: {(100 * stage.done) / stage.total}%"></div>
                  {:else if running}
                    <div class="h-full w-1/3 animate-pulse bg-blue-600"></div>
                  {/if}
                </div>
              </li>
            {/each}
          </ol>
        {/if}

        {#if job.previews.length}
          <div class="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
            {#each previewItems(job) as item, index (index)}
              <button class="group flex flex-col gap-1 text-left" onclick={() => (gallery = { items: previewItems(job), start: index })}>
                <img src={item.url} alt={item.caption} loading="lazy" class="aspect-video w-full rounded bg-gray-100 object-contain group-hover:opacity-90 dark:bg-gray-900" />
                <span class="truncate text-[11px] text-gray-500">{item.caption}</span>
              </button>
            {/each}
          </div>
        {/if}

        {#if job.outputs.length}
          <div class="flex flex-wrap gap-2 text-sm">
            {#each job.outputs as output (output.file)}
              <a class={plain} href={jobFileUrl(job.id, output.file)} download={output.file.split("/").pop()}>{output.name}</a>
            {/each}
          </div>
        {/if}
      {/if}
    </div>
  </div>
</div>

{#if gallery}
  <ImageGallery items={gallery.items} start={gallery.start} onClose={() => (gallery = null)} />
{/if}
{#if meshViewer}
  <MeshViewer url={meshViewer.url} format={meshViewer.format} flat={meshViewer.flat} title={meshViewer.title} onClose={() => (meshViewer = null)} />
{/if}
{#if viewer}
  <SplatViewer url={viewer.url} frame={viewer.frame} sceneToSplats={viewer.sceneToSplats} title={viewer.title} onClose={() => (viewer = null)} />
{/if}
