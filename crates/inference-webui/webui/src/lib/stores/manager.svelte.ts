import * as manager from "../services/manager";
import type { Inventory, InventoryModel, LoadProgress, ManagedModel, ModelKind } from "../services/manager";

/** A job's model: the id it is served under, the kinds that fit it, and the load options it offers. */
export interface Slot {
  id: string;
  title: string;
  kinds: ModelKind[];
  /** Offers the KV context length and MTP speculative decoding. */
  chat?: boolean;
  /** Offers steering by a control vector (a text model's: the chat and code tabs share it). */
  steer?: boolean;
  /** Offers a device, for the layout detector that runs beside the engine. */
  device?: boolean;
  /** Also takes a text GGUF with a projector beside it, which the inventory lists as text. */
  visionGguf?: boolean;
  /** The KV context a load offers first, in place of the chat default. */
  defaultContext?: number;
  /** Offers a voice activity model to cut long recordings at their silences. */
  vad?: boolean;
  hint: string;
}

export const SLOTS = {
  chat: {
    id: "chat",
    title: "Chat model",
    kinds: ["text", "multimodal"],
    chat: true,
    steer: true,
    hint: "A GGUF (with its projector for images), a model directory, or a Hugging Face repo",
  },
  transcription: {
    id: "transcription",
    title: "Speech recognition model",
    kinds: ["transcription"],
    vad: true,
    hint: "Whisper, Parakeet, Nemotron ASR or Qwen3-ASR: a .nemo, directory, GGUF or repo",
  },
  diarization: {
    id: "diarization",
    title: "Speaker diarization model (optional)",
    kinds: ["diarization"],
    hint: "Nemotron-3 Diarization or Streaming Sortformer",
  },
  speech: {
    id: "speech",
    title: "Speech synthesis model",
    kinds: ["speech"],
    hint: "Kokoro or Dia",
  },
  images: {
    id: "images",
    title: "Image model",
    kinds: ["image_generation"],
    hint: "HiDream-O1 or FLUX",
  },
  layout: {
    id: "layout",
    title: "Layout model",
    kinds: ["layout"],
    device: true,
    hint: "PP-DocLayoutV3, as its directory or GGUF",
  },
  ocr: {
    id: "ocr",
    title: "OCR model",
    kinds: ["multimodal"],
    chat: true,
    visionGguf: true,
    // the cache all regions in flight share: a region's prompt is often 750-1,500 image tokens, and at 8,192 only a
    // handful fit, the rest waiting or preempted (re-encoding their images)
    defaultContext: 32768,
    hint: "PaddleOCR-VL, or another vision-language model",
  },
} satisfies Record<string, Slot>;

export type SlotId = keyof typeof SLOTS;

/** What a slot's form holds between loads, kept per browser. */
export interface SlotForm {
  source: string;
  contextLength: number;
  /** Sequences the model serves at once; 0 leaves the server's `--max-seqs`. */
  maxSeqs: number;
  mtp: boolean;
  mtpDrafts: number;
  /** A drafter checkpoint (DFlash) to draft with; "" uses the model's own MTP head. */
  mtpDrafter: string;
  /** The drafter's weight type (`q4k`, `q8_0`); "" takes the server's (q8_0 beside a GGUF, else the model's). */
  mtpDraftIsq: string;
  /** Whether each step drafts or decodes plainly, whichever the drafter's recent acceptance says is faster. */
  mtpAutoDepth?: boolean;
  mmproj: string;
  /** Load a recipe-loaded checkpoint's vision tower too */
  vision?: boolean;
  /** The voice activity model to load beside it: unset takes the first the inventory lists, "" none. */
  vad?: string;
  device: string;
  /** The KV cache's K and V storage (`auto`, `f8e4m3`, `q2`..`q8`); "" keeps the server's. */
  cacheK: string;
  cacheV: string;
  /** Per-layer cache storage (`0=q8,20-27=q6` or `uniform`); "" takes the model's profile or the general rule. */
  cacheLayers: string;
  /** Whether the server loads this slot again when it starts. */
  loadAtStart: boolean;
  /** Steering by a control vector derived from the texts below. */
  steer?: boolean;
  /** One text a line; line i pairs with line i of the other. */
  steerPositive?: string;
  steerNegative?: string;
  /** The vector's scale: its directions are unit length, so about 4 nudges and 8 is strong; negative turns it. */
  steerScale?: number;
  /** `first-last` (0-based) to steer only those layers; "" steers every layer. */
  steerLayers?: string;
  /** A looped model's loop guidance strength (about 0.2 to 0.3); null or 0 for none. */
  loopGuidance?: number | null;
}

/** The steering a slot's form sets up, or null when it sets none. */
export function controlVectors(form: SlotForm): manager.ControlVectors | null {
  const lines = (text: string | undefined) =>
    (text ?? "")
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line !== "");
  const positive = lines(form.steerPositive);
  const negative = lines(form.steerNegative);
  const vectors: manager.ControlVectors = {};
  if (form.steer && positive.length > 0 && negative.length > 0) {
    vectors.derived = [{ positive, negative, scale: form.steerScale ?? DEFAULT_STEER_SCALE }];
    const range = /^\s*(\d+)\s*-\s*(\d+)\s*$/.exec(form.steerLayers ?? "");
    if (range) vectors.layer_range = [Number(range[1]), Number(range[2])];
  }
  const guidance = form.loopGuidance;
  if (typeof guidance === "number" && Number.isFinite(guidance) && guidance > 0) vectors.loop_guidance = guidance;
  return Object.keys(vectors).length > 0 ? vectors : null;
}

/** Whether a model loops part of its stack (Inkling 2), so it takes loop guidance. */
export function loops(model: InventoryModel | undefined): boolean {
  return /inkling2/i.test(model?.architecture ?? "");
}

/** The scale a new slot's steering starts at. */
export const DEFAULT_STEER_SCALE = 4;

/** The KV cache storages a slot's form offers; quantized widths may differ for K and V. */
export const CACHE_TYPES = ["auto", "f8e4m3", "q8", "q7", "q6", "q5", "q4", "q3", "q2"] as const;

const isQuantized = (cache: string) => /^q[2-8]$/.test(cache);

/** The load's `paged_cache_type` for a form's K and V choices; undefined keeps the server's. */
export function cacheType(form: SlotForm): string | undefined {
  if (!form.cacheK) return undefined;
  if (!isQuantized(form.cacheK) || !isQuantized(form.cacheV) || form.cacheV === form.cacheK) return form.cacheK;
  return `${form.cacheK}/${form.cacheV}`;
}

// A `paged_cache_type` as the form's K and V choices.
function cacheChoices(cache: string | undefined): [string, string] {
  if (!cache) return ["", ""];
  const [k, v] = cache.split("/");
  return [k, v ?? k];
}

/** A load or unload under way. */
export interface SlotActivity {
  busy: boolean;
  progress: LoadProgress | null;
  error: string | null;
}

const FORMS_KEY = "inference-rs-slot-forms";
const REFRESH_MS = 10_000;
const DEFAULT_CONTEXT = 8192;

function defaultForm(slot: SlotId): SlotForm {
  const contextLength = (SLOTS[slot] as Slot).defaultContext ?? DEFAULT_CONTEXT;
  return {
    source: "",
    contextLength,
    maxSeqs: 0,
    mtp: false,
    mtpDrafts: 0,
    mtpDrafter: "",
    mtpDraftIsq: "",
    mmproj: "",
    device: "auto",
    cacheK: "",
    cacheV: "",
    cacheLayers: "",
    loadAtStart: false,
  };
}

// every slot's form, from what this browser kept, so reading one never creates state mid-render
function storedForms(): Record<SlotId, SlotForm> {
  let stored: Partial<Record<SlotId, Partial<SlotForm>>> = {};
  try {
    stored = JSON.parse(localStorage.getItem(FORMS_KEY) ?? "{}");
  } catch {
    // unreadable storage starts fresh
  }
  const forms = {} as Record<SlotId, SlotForm>;
  for (const slot of Object.keys(SLOTS) as SlotId[]) {
    const form = { ...defaultForm(slot), ...stored[slot] };
    // a form kept before the slot had its own default holds the old shared one, which nobody chose for it
    if (stored[slot]?.contextLength === DEFAULT_CONTEXT) form.contextLength = defaultForm(slot).contextLength;
    forms[slot] = form;
  }
  return forms;
}

class ManagerStore {
  /** Whether this client may load, unload, download and delete models. */
  admin = $state(false);
  keyed = $state(false);
  served = $state<ManagedModel[]>([]);
  inventory = $state<Inventory | null>(null);
  inventoryError = $state<string | null>(null);
  scanning = $state(false);
  forms = $state<Record<SlotId, SlotForm>>(storedForms());
  /** What the server keeps per model id; a slot's form starts from its entry. */
  saved = $state<Record<string, manager.SavedModel>>({});
  savedError = $state<string | null>(null);
  activity = $state<Record<string, SlotActivity>>({});
  /** Bumped after each load or unload, so views that list models refresh. */
  changes = $state(0);
  private timer: ReturnType<typeof setInterval> | null = null;
  // a poll sent before a load finished must not overwrite the list read after it
  private refreshes = 0;

  start(admin: boolean, keyed: boolean) {
    this.admin = admin;
    this.keyed = keyed;
    void this.refresh();
    if (admin) {
      void this.scan();
      void this.readSaved(true);
    }
    this.timer ??= setInterval(() => void this.refresh(), REFRESH_MS);
  }

  async refresh() {
    const sent = ++this.refreshes;
    try {
      const served = await manager.servedModels();
      if (sent === this.refreshes) this.served = served;
    } catch {
      // a server without the manager serves no slots
    }
  }

  async scan() {
    this.scanning = true;
    try {
      this.inventory = await manager.inventory();
      this.inventoryError = null;
    } catch (e) {
      this.inventoryError = e instanceof Error ? e.message : String(e);
    } finally {
      this.scanning = false;
    }
  }

  /** Reads the server's kept loads; `apply` puts each slot's entry into its form. */
  async readSaved(apply = false) {
    try {
      this.saved = await manager.savedModels();
      this.savedError = null;
    } catch (e) {
      // a server without kept loads (an older one, or none configured) leaves the forms as this browser kept them
      this.savedError = e instanceof Error ? e.message : String(e);
      return;
    }
    if (!apply) return;
    for (const slot of Object.keys(SLOTS) as SlotId[]) {
      const kept = this.saved[SLOTS[slot].id];
      if (!kept) continue;
      const request = kept.request;
      const form = this.forms[slot];
      const [cacheK, cacheV] = cacheChoices(request.paged_cache_type);
      Object.assign(form, {
        source: request.source,
        contextLength: request.paged_context_len ?? form.contextLength,
        maxSeqs: request.max_seqs ?? 0,
        mtp: request.mtp != null,
        mtpDrafts: request.mtp?.n_predict ?? 0,
        mtpDrafter: request.mtp?.model ?? "",
        mtpDraftIsq: request.mtp?.draft_isq ?? "",
        mtpAutoDepth: request.mtp?.auto_depth ?? false,
        mmproj: request.mmproj ?? "",
        vision: request.vision ?? false,
        vad: (SLOTS[slot] as Slot).vad ? (request.vad ?? "") : form.vad,
        device: request.device ?? "auto",
        cacheK,
        cacheV,
        cacheLayers: request.paged_cache_layers ?? "",
        loadAtStart: kept.load_at_start,
        ...steerFields(request.control_vectors),
      });
    }
    this.persistForms();
  }

  /** The load a slot's form describes. */
  requestFor(slot: SlotId): manager.LoadRequest {
    const spec = SLOTS[slot] as Slot;
    const form = this.form(slot);
    const request: manager.LoadRequest = { source: form.source.trim(), model_id: spec.id };
    if (spec.chat) {
      if (form.contextLength > 0) request.paged_context_len = form.contextLength;
      if (form.maxSeqs > 0) request.max_seqs = form.maxSeqs;
      if (form.mtp) {
        request.mtp = {};
        if (form.mtpDrafts > 0) request.mtp.n_predict = form.mtpDrafts;
        if (form.mtpDrafter) request.mtp.model = form.mtpDrafter;
        if (form.mtpDrafter && form.mtpDraftIsq) request.mtp.draft_isq = form.mtpDraftIsq;
        if (form.mtpAutoDepth) request.mtp.auto_depth = true;
      }
      if (form.mmproj) request.mmproj = form.mmproj;
      if (form.vision && this.entry(request.source)?.recipe_vision) request.vision = true;
      const cache = cacheType(form);
      if (cache) request.paged_cache_type = cache;
      if (form.cacheLayers.trim()) request.paged_cache_layers = form.cacheLayers.trim();
      const vectors = spec.steer ? controlVectors(form) : null;
      if (vectors) request.control_vectors = vectors;
    }
    if (spec.vad) {
      const vad = form.vad ?? this.vads[0]?.path ?? "";
      if (vad) request.vad = vad;
    }
    if (spec.device && form.device !== "auto") request.device = form.device;
    return request;
  }

  /** Keeps a slot's settings on the server without loading it. */
  async saveSettings(slot: SlotId) {
    const form = this.form(slot);
    this.persistForms();
    const saved = { request: this.requestFor(slot), load_at_start: form.loadAtStart };
    try {
      await manager.saveModel(SLOTS[slot].id, saved);
      this.activity[slot] = { ...this.activityOf(slot), error: null };
    } catch (e) {
      this.activity[slot] = { ...this.activityOf(slot), error: e instanceof Error ? e.message : String(e) };
    }
    await this.readSaved();
  }

  servedIn(slot: SlotId): ManagedModel | undefined {
    const spec = SLOTS[slot] as Slot;
    const own = this.served.find((model) => model.model_id === spec.id);
    if (own) return own;
    // a model the server started with (`serve -m`) is served under its own id, not a slot's
    const slotIds = new Set(Object.values(SLOTS).map((s) => s.id as string));
    return this.served.find((model) => !slotIds.has(model.model_id) && spec.kinds.includes(model.kind));
  }

  form(slot: SlotId): SlotForm {
    return this.forms[slot];
  }

  persistForms() {
    try {
      localStorage.setItem(FORMS_KEY, JSON.stringify(this.forms));
    } catch {
      // private windows may refuse storage
    }
  }

  /** The inventory's models a slot can load, by name. */
  /** The drafters (DFlash, or a checkpoint carrying one) a chat load can draft with in place of its own MTP head. */
  get drafters(): InventoryModel[] {
    return (this.inventory?.models ?? []).filter((model) => model.kind === "drafter" || model.carries_drafter);
  }

  /** The voice activity models a speech recognition load can cut its long recordings with. */
  get vads(): InventoryModel[] {
    return (this.inventory?.models ?? []).filter((model) => model.kind === "voice_activity");
  }

  candidates(slot: SlotId): InventoryModel[] {
    return (this.inventory?.models ?? [])
      .filter((model) => fits(slot, model))
      .sort((a, b) => displayName(a).localeCompare(displayName(b)));
  }

  entry(path: string): InventoryModel | undefined {
    return this.inventory?.models.find((model) => model.path === path);
  }

  activityOf(slot: SlotId): SlotActivity {
    return this.activity[slot] ?? { busy: false, progress: null, error: null };
  }

  /** Puts `source` in the slot's form, as the models view does before loading it. */
  assign(slot: SlotId, source: string, mmproj = "") {
    const form = this.form(slot);
    form.source = source;
    form.mmproj = mmproj;
    this.persistForms();
  }

  async load(slot: SlotId) {
    this.persistForms();
    const request = this.requestFor(slot);
    const startup = this.servedIn(slot);
    this.activity[slot] = { busy: true, progress: null, error: null };
    try {
      // a replaced startup model is unloaded first, so the two never hold memory at once
      if (startup && startup.model_id !== request.model_id) await manager.unloadModel(startup.model_id);
      await manager.loadModel(request, (progress) => {
        this.activity[slot] = { busy: true, progress, error: null };
      });
      this.activity[slot] = { busy: false, progress: null, error: null };
    } catch (e) {
      this.activity[slot] = { busy: false, progress: null, error: e instanceof Error ? e.message : String(e) };
    }
    await this.refresh();
    // the server kept this load; the slot's startup choice rides along with it
    if (this.form(slot).loadAtStart && !this.activityOf(slot).error) await this.saveSettings(slot);
    else await this.readSaved();
    this.changes++;
  }

  /** Steers the slot's loaded model by its form's vectors from its next request, or stops when steering is off. */
  async steer(slot: SlotId) {
    this.persistForms();
    const served = this.servedIn(slot);
    if (!served) return;
    this.activity[slot] = { busy: true, progress: null, error: null };
    try {
      await manager.setControlVectors(served.model_id, controlVectors(this.form(slot)) ?? {});
      this.activity[slot] = { busy: false, progress: null, error: null };
    } catch (e) {
      this.activity[slot] = { busy: false, progress: null, error: e instanceof Error ? e.message : String(e) };
      return;
    }
    // a slot the server keeps reloads steered as it is now
    if (this.saved[SLOTS[slot].id]) await this.saveSettings(slot);
  }

  async unload(slot: SlotId) {
    this.activity[slot] = { busy: true, progress: null, error: null };
    try {
      await manager.unloadModel(this.servedIn(slot)?.model_id ?? SLOTS[slot].id);
      this.activity[slot] = { busy: false, progress: null, error: null };
    } catch (e) {
      this.activity[slot] = { busy: false, progress: null, error: e instanceof Error ? e.message : String(e) };
    }
    await this.refresh();
    this.changes++;
  }
}

// a kept load's steering as the form's fields: its first derived vector (the form makes one)
function steerFields(vectors: manager.ControlVectors | undefined): Partial<SlotForm> {
  const derived = vectors?.derived?.[0];
  const loopGuidance = vectors?.loop_guidance ?? null;
  if (!derived) return { steer: false, loopGuidance };
  const range = vectors?.layer_range;
  return {
    loopGuidance,
    steer: true,
    steerPositive: derived.positive.join("\n"),
    steerNegative: derived.negative.join("\n"),
    steerScale: derived.scale ?? DEFAULT_STEER_SCALE,
    steerLayers: range ? `${range[0]}-${range[1]}` : "",
  };
}

/** Whether `slot` can load `model`. */
export function fits(slot: SlotId, model: InventoryModel): boolean {
  const spec: Slot = SLOTS[slot];
  if (model.kind == null) return false;
  if (spec.kinds.includes(model.kind)) return true;
  return spec.visionGguf === true && model.kind === "text" && model.projectors.length > 0;
}

/** How a model is listed: a GGUF or `.nemo` by its file, anything else by its repository or folder name. */
export function displayName(model: InventoryModel): string {
  const file = model.path.replace(/\/+$/, "").split("/").pop() ?? model.path;
  if (model.format === "gguf" || model.format === "nemo") return file;
  return model.repo_id ?? model.name ?? file;
}

/** The last part of a path, for showing a source or projector compactly. */
export function baseName(path: string): string {
  return path.replace(/\/+$/, "").split("/").pop() ?? path;
}

export function formatBytes(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

export const managerStore = new ManagerStore();
