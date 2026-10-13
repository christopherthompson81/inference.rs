import * as jobs from "../services/jobs";
import type { Job, JobFile, JobKind } from "../services/jobs";
import { managerStore, type SlotId } from "./manager.svelte";

const TITLE_LENGTH = 60;
// which entry each tab last showed, so a reload or a return to the tab opens it again
const CURRENT_KEY = "inference-rs-current-jobs";
const NONE: Record<JobKind, string | null> = { transcription: null, speech: null, images: null, documents: null };

function storedCurrent(): Record<JobKind, string | null> {
  try {
    return { ...NONE, ...JSON.parse(localStorage.getItem(CURRENT_KEY) ?? "{}") };
  } catch {
    return { ...NONE };
  }
}
// a Hugging Face cache path: `.../models--<org>--<name>/snapshots/<revision>[/<file>]`
const HF_CACHE_REPO = /models--([^/]+?)--([^/]+)\/snapshots\/[^/]+(?:\/(.+))?$/;

class JobsStore {
  lists = $state<Record<JobKind, Job[]>>({ transcription: [], speech: [], images: [], documents: [] });
  /** The job each view shows, so its history marks it. */
  current = $state<Record<JobKind, string | null>>(storedCurrent());

  /** Keeps which entry each tab shows, for the next visit. */
  persistCurrent() {
    try {
      localStorage.setItem(CURRENT_KEY, JSON.stringify(this.current));
    } catch {
      // private windows may refuse storage
    }
  }

  async refresh(kind: JobKind) {
    try {
      this.lists[kind] = await jobs.listJobs(kind);
    } catch (e) {
      console.error(`Failed to list ${kind} jobs:`, e);
    }
  }

  /** Keeps a job and marks it current; `id` adds to one already kept. */
  async save<Data>(
    job: { id?: string; kind: JobKind; title: string; slot: SlotId; data: Data },
    files: JobFile[] = [],
  ): Promise<Job<Data>> {
    const { slot, ...rest } = job;
    const kept = await jobs.saveJob({ ...rest, title: title(job.title), model: modelName(slot) }, files);
    this.current[job.kind] = kept.id;
    await this.refresh(job.kind);
    return kept;
  }

  async remove(kind: JobKind, id: string) {
    await jobs.deleteJob(id);
    if (this.current[kind] === id) this.current[kind] = null;
    await this.refresh(kind);
  }
}

function title(text: string): string {
  const line = text.trim().replace(/\s+/g, " ");
  return line.length > TITLE_LENGTH ? `${line.slice(0, TITLE_LENGTH)}...` : line;
}

// the served model's repo or file name, which says more than the slot's id
function modelName(slot: SlotId): string | undefined {
  const served = managerStore.servedIn(slot);
  const source = served?.source?.replace(/\/+$/, "");
  if (!source) return served?.model_id;
  const cached = HF_CACHE_REPO.exec(source);
  if (cached) return cached[3]?.split("/").pop() ?? `${cached[1]}/${cached[2]}`;
  return source.split("/").pop();
}

export const jobsStore = new JobsStore();
