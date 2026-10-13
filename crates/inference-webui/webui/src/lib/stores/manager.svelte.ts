import * as manager from "../services/manager";
import type { Inventory, InventoryModel, LoadProgress, ManagedModel, ModelKind } from "../services/manager";

/** A job's model: the id it is served under, the kinds that fit it, and the load options it offers. */
export interface Slot {
  id: string;
  title: string;
  kinds: ModelKind[];
  /** Offers the KV context length and MTP speculative decoding. */
  chat?: boolean;
  /** Offers a device, for the layout detector that runs beside the engine. */
  device?: boolean;
  /** Also takes a text GGUF with a projector beside it, which the inventory lists as text. */
  visionGguf?: boolean;
  hint: string;
}

export const SLOTS = {
  chat: {
    id: "chat",
    title: "Chat model",
    kinds: ["text", "multimodal"],
    chat: true,
    hint: "A GGUF (with its projector for images), a model directory, or a Hugging Face repo",
  },
  transcription: {
    id: "transcription",
    title: "Speech recognition model",
    kinds: ["transcription"],
    hint: "Parakeet, Nemotron ASR or Qwen3-ASR: a .nemo, directory, GGUF or repo",
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
    hint: "PaddleOCR-VL, or another vision-language model",
  },
} satisfies Record<string, Slot>;

export type SlotId = keyof typeof SLOTS;

/** What a slot's form holds between loads, kept per browser. */
export interface SlotForm {
  source: string;
  contextLength: number;
  mtp: boolean;
  mtpDrafts: number;
  mmproj: string;
  device: string;
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

function defaultForm(): SlotForm {
  return { source: "", contextLength: DEFAULT_CONTEXT, mtp: false, mtpDrafts: 0, mmproj: "", device: "auto" };
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
  for (const slot of Object.keys(SLOTS) as SlotId[]) forms[slot] = { ...defaultForm(), ...stored[slot] };
  return forms;
}

class ManagerStore {
  /** Whether this client may load, unload, download and delete models. */
  admin = $state(false);
  served = $state<ManagedModel[]>([]);
  inventory = $state<Inventory | null>(null);
  inventoryError = $state<string | null>(null);
  scanning = $state(false);
  forms = $state<Record<SlotId, SlotForm>>(storedForms());
  activity = $state<Record<string, SlotActivity>>({});
  /** Bumped after each load or unload, so views that list models refresh. */
  changes = $state(0);
  private timer: ReturnType<typeof setInterval> | null = null;
  // a poll sent before a load finished must not overwrite the list read after it
  private refreshes = 0;

  start(admin: boolean) {
    this.admin = admin;
    void this.refresh();
    if (admin) void this.scan();
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

  servedIn(slot: SlotId): ManagedModel | undefined {
    return this.served.find((model) => model.model_id === SLOTS[slot].id);
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
    const spec = SLOTS[slot] as Slot;
    const form = this.form(slot);
    this.persistForms();
    const request: manager.LoadRequest = { source: form.source.trim(), model_id: spec.id };
    if (spec.chat) {
      if (form.contextLength > 0) request.paged_context_len = form.contextLength;
      if (form.mtp) request.mtp = form.mtpDrafts > 0 ? { n_predict: form.mtpDrafts } : {};
      if (form.mmproj) request.mmproj = form.mmproj;
    }
    if (spec.device && form.device !== "auto") request.device = form.device;
    this.activity[slot] = { busy: true, progress: null, error: null };
    try {
      await manager.loadModel(request, (progress) => {
        this.activity[slot] = { busy: true, progress, error: null };
      });
      this.activity[slot] = { busy: false, progress: null, error: null };
    } catch (e) {
      this.activity[slot] = { busy: false, progress: null, error: e instanceof Error ? e.message : String(e) };
    }
    await this.refresh();
    this.changes++;
  }

  async unload(slot: SlotId) {
    this.activity[slot] = { busy: true, progress: null, error: null };
    try {
      await manager.unloadModel(SLOTS[slot].id);
      this.activity[slot] = { busy: false, progress: null, error: null };
    } catch (e) {
      this.activity[slot] = { busy: false, progress: null, error: e instanceof Error ? e.message : String(e) };
    }
    await this.refresh();
    this.changes++;
  }
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
