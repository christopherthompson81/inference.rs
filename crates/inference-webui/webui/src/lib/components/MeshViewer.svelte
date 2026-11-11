<script lang="ts" module>
  /** A GLB as exported (+y up), or a PLY in TRELLIS's +z-up frame (an image job's mesh, or its voxels). */
  export type MeshFormat = "glb" | "ply";
</script>

<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  // the camera's distance as a multiple of what just fits the model's bounding sphere
  const FIT_MARGIN = 1.3;
  const FOV = 40;
  // a PLY carries no material: a neutral clay
  const CLAY = 0xb9bcc2;
  const CLAY_ROUGHNESS = 0.85;
  const BACKGROUND = 0x18181c;
  const ENVIRONMENT_BLUR = 0.04;
  // the models' albedo sits well below their photos' brightness, so the default exposure lifts it
  const DEFAULT_BRIGHTNESS = 1.5;
  const MIN_BRIGHTNESS = 0.5;
  const MAX_BRIGHTNESS = 3;
  const BRIGHTNESS_KEY = "meshViewer.brightness";

  function storedBrightness(): number {
    try {
      const v = Number(localStorage.getItem(BRIGHTNESS_KEY));
      return v >= MIN_BRIGHTNESS && v <= MAX_BRIGHTNESS ? v : DEFAULT_BRIGHTNESS;
    } catch {
      return DEFAULT_BRIGHTNESS;
    }
  }

  let {
    url,
    format,
    flat = false,
    title,
    onClose,
  }: {
    url: string;
    format: MeshFormat;
    /** Facet the shading (voxels), rather than smoothing it across shared vertices */
    flat?: boolean;
    title: string;
    onClose: () => void;
  } = $props();

  let canvas = $state<HTMLCanvasElement | null>(null);
  let dialog = $state<HTMLDivElement | null>(null);
  let status = $state("Loading the model...");
  let brightness = $state(storedBrightness());
  let renderer: import("three").WebGLRenderer | null = null;
  let stop: (() => void) | null = null;

  $effect(() => {
    if (renderer) renderer.toneMappingExposure = brightness;
    try {
      localStorage.setItem(BRIGHTNESS_KEY, String(brightness));
    } catch {
      // storage unavailable: the setting lasts this view
    }
  });

  onMount(() => {
    dialog?.focus();
    void start().catch((error) => (status = `Could not show the model: ${error instanceof Error ? error.message : error}`));
  });
  onDestroy(() => stop?.());

  async function start() {
    // loaded on first use, as the splat viewer's renderer is
    const THREE = await import("three");
    const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
    const { RoomEnvironment } = await import("three/examples/jsm/environments/RoomEnvironment.js");
    if (!canvas) return;
    const view = new THREE.WebGLRenderer({ canvas, antialias: true });
    view.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    view.toneMapping = THREE.NeutralToneMapping;
    view.toneMappingExposure = brightness;
    renderer = view;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(BACKGROUND);
    const pmrem = new THREE.PMREMGenerator(view);
    const environment = pmrem.fromScene(new RoomEnvironment(), ENVIRONMENT_BLUR).texture;
    scene.environment = environment;
    const camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 100);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;

    let model: import("three").Object3D;
    if (format === "glb") {
      const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
      model = (await new GLTFLoader().loadAsync(url)).scene;
    } else {
      const { PLYLoader } = await import("three/examples/jsm/loaders/PLYLoader.js");
      const geometry = await new PLYLoader().loadAsync(url);
      geometry.computeVertexNormals();
      const material = new THREE.MeshStandardMaterial({ color: CLAY, roughness: CLAY_ROUGHNESS, flatShading: flat, side: THREE.DoubleSide });
      model = new THREE.Mesh(geometry, material);
      // +z up to three.js's +y up
      model.rotation.x = -Math.PI / 2;
    }
    scene.add(model);
    frame(model, camera, controls, THREE);
    status = "";

    const resize = () => {
      if (!canvas) return;
      const { clientWidth: width, clientHeight: height } = canvas;
      view.setSize(width, height, false);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();
    view.setAnimationLoop(() => {
      controls.update();
      view.render(scene, camera);
    });
    stop = () => {
      view.setAnimationLoop(null);
      observer.disconnect();
      controls.dispose();
      model.traverse((node) => {
        const mesh = node as import("three").Mesh;
        mesh.geometry?.dispose();
        const materials = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
        for (const material of materials) {
          for (const value of Object.values(material)) if (value instanceof THREE.Texture) value.dispose();
          material.dispose();
        }
      });
      environment.dispose();
      pmrem.dispose();
      view.dispose();
      renderer = null;
    };
  }

  // centers the model at the origin and backs the camera off until its bounding sphere fits, a little from above
  function frame(
    model: import("three").Object3D,
    camera: import("three").PerspectiveCamera,
    controls: import("three/examples/jsm/controls/OrbitControls.js").OrbitControls,
    THREE: typeof import("three"),
  ) {
    model.updateMatrixWorld(true);
    const sphere = new THREE.Box3().setFromObject(model).getBoundingSphere(new THREE.Sphere());
    model.position.sub(sphere.center);
    const radius = Math.max(sphere.radius, 1e-3);
    const distance = (FIT_MARGIN * radius) / Math.sin(THREE.MathUtils.degToRad(FOV / 2));
    camera.position.set(0, 0.35, 1).normalize().multiplyScalar(distance);
    camera.near = distance / 100;
    camera.far = distance * 10;
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    controls.minDistance = radius / 4;
    controls.maxDistance = distance * 4;
    controls.update();
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
    <label class="flex items-center gap-2 text-xs text-white/80">
      Brightness
      <input type="range" min={MIN_BRIGHTNESS} max={MAX_BRIGHTNESS} step="0.1" bind:value={brightness} class="w-24 accent-blue-500 sm:w-32" aria-label="Brightness" />
      <span class="w-6 tabular-nums">{brightness.toFixed(1)}</span>
    </label>
    <span class="hidden text-xs text-white/60 lg:inline">Drag to turn, scroll to zoom, right-drag to move</span>
    <button class="rounded-md border border-white/30 px-3 py-1.5 hover:bg-white/10" onclick={onClose} aria-label="Close viewer">Close</button>
  </div>
  <div class="relative min-h-0 flex-1">
    <canvas bind:this={canvas} class="h-full w-full touch-none"></canvas>
    {#if status}
      <div class="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-white/80">{status}</div>
    {/if}
  </div>
</div>
