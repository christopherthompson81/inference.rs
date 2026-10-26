<script lang="ts" module>
  /** How a splat file's axes sit: a world job's scene frame is +z up, a reconstruction's the first camera's (OpenCV). */
  export type SplatFrame = "z_up" | "opencv";
</script>

<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  // walking speed a second as a share of the splats' median distance from the start: scenes come in arbitrary units,
  // and a far sky dome would make their extent useless
  const MOVE_FRACTION = 0.25;
  // splats sampled for that median
  const SPEED_SAMPLES = 20000;

  let {
    url,
    frame,
    sceneToSplats = null,
    title,
    onClose,
  }: {
    url: string;
    frame: SplatFrame;
    /** Row-major, from the scene frame to the frame the splats were trained in; undone so the scene stands upright. */
    sceneToSplats?: number[][] | null;
    title: string;
    onClose: () => void;
  } = $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let dialog = $state<HTMLDivElement | null>(null);
  let status = $state("Loading splats...");
  let stop: (() => void) | null = null;

  onMount(() => {
    dialog?.focus();
    void start().catch((error) => (status = `Could not show the splats: ${error instanceof Error ? error.message : error}`));
  });
  onDestroy(() => stop?.());

  async function start() {
    // loaded on first use: the renderer is a few megabytes the rest of the UI never needs
    const THREE = await import("three");
    const { SparkControls, SparkRenderer, SplatMesh } = await import("@sparkjsdev/spark");
    if (!canvas) return;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101014);
    const camera = new THREE.PerspectiveCamera(70, 1, 0.01, 1000);
    scene.add(new SparkRenderer({ renderer }));

    const splats = new SplatMesh({ url, fileName: url.split("/").pop() });
    const placement = new THREE.Matrix4();
    if (sceneToSplats) placement.set(...(sceneToSplats.flat() as Parameters<typeof placement.set>)).invert();
    // three.js is +y up and looks down -z: a +z-up scene turns a quarter about x, an OpenCV frame half a turn
    const axes = new THREE.Matrix4().makeRotationX(frame === "z_up" ? -Math.PI / 2 : Math.PI);
    splats.matrixAutoUpdate = false;
    splats.matrix.multiplyMatrices(axes, placement);
    scene.add(splats);
    const controls = new SparkControls({ canvas });
    splats.initialized.then(
      () => {
        const distance = medianDistance(splats, THREE);
        if (distance > 0) controls.fpsMovement.moveSpeed = distance * MOVE_FRACTION;
        status = "";
      },
      (error: unknown) => (status = `Could not load the splats: ${error instanceof Error ? error.message : error}`),
    );
    const resize = () => {
      if (!canvas) return;
      const { clientWidth: width, clientHeight: height } = canvas;
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();
    renderer.setAnimationLoop(() => {
      controls.update(camera);
      renderer.render(scene, camera);
    });
    stop = () => {
      renderer.setAnimationLoop(null);
      observer.disconnect();
      splats.dispose();
      renderer.dispose();
    };
  }

  // the median distance of a sample of the splats from the camera's start, the origin, once placed
  function medianDistance(
    splats: import("@sparkjsdev/spark").SplatMesh,
    THREE: typeof import("three"),
  ): number {
    const count = splats.packedSplats?.numSplats ?? splats.extSplats?.numSplats ?? 0;
    const stride = Math.max(1, Math.floor(count / SPEED_SAMPLES));
    const distances: number[] = [];
    const placed = new THREE.Vector3();
    splats.forEachSplat((index, center) => {
      if (index % stride === 0) distances.push(placed.copy(center).applyMatrix4(splats.matrix).length());
    });
    distances.sort((a, b) => a - b);
    return distances[Math.floor(distances.length / 2)] ?? 0;
  }

  function key(event: KeyboardEvent) {
    if (event.key === "Escape") onClose();
  }
</script>

<div
  bind:this={dialog}
  class="fixed inset-0 z-50 flex flex-col bg-black outline-none"
  role="dialog"
  aria-modal="true"
  aria-label={title}
  tabindex="-1"
  onkeydown={key}
>
  <div class="flex shrink-0 items-center gap-3 px-4 py-3 text-sm text-white">
    <span class="min-w-0 flex-1 truncate">{title}</span>
    <span class="hidden text-xs text-white/60 sm:inline">Drag to look, WASD to move, E and Q for up and down, Shift to go faster</span>
    <button class="rounded-md border border-white/30 px-3 py-1.5 hover:bg-white/10" onclick={onClose} aria-label="Close viewer">Close</button>
  </div>
  <div class="relative min-h-0 flex-1">
    <canvas bind:this={canvas} class="h-full w-full touch-none"></canvas>
    {#if status}
      <div class="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-white/80">{status}</div>
    {/if}
  </div>
</div>
