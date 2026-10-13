<script lang="ts" module>
  export interface GalleryItem {
    url: string;
    caption: string;
    /** The name a download saves it under. */
    filename: string;
  }
</script>

<script lang="ts">
  import { onMount, tick } from "svelte";

  let {
    items,
    start = 0,
    onClose,
    onAnchor,
  }: { items: GalleryItem[]; start?: number; onClose: () => void; onAnchor: (item: GalleryItem) => void } = $props();

  let index = $state(0);
  let dialog = $state<HTMLDivElement | null>(null);
  let strip = $state<HTMLDivElement | null>(null);
  let item = $derived(items[index]);

  onMount(() => {
    index = Math.min(Math.max(start, 0), items.length - 1);
    dialog?.focus();
  });

  function go(step: number) {
    index = (index + step + items.length) % items.length;
    void tick().then(() => strip?.querySelector(`[data-index="${index}"]`)?.scrollIntoView({ block: "nearest", inline: "center" }));
  }

  function key(event: KeyboardEvent) {
    if (event.key === "Escape") onClose();
    else if (event.key === "ArrowRight") go(1);
    else if (event.key === "ArrowLeft") go(-1);
    else return;
    event.preventDefault();
  }

  const button =
    "rounded-md bg-white/10 px-3 py-1.5 text-sm text-white hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-white";
</script>

{#if item}
  <div
    bind:this={dialog}
    class="fixed inset-0 z-50 flex flex-col bg-black/95 outline-none"
    role="dialog"
    aria-modal="true"
    aria-label="Image gallery"
    tabindex="-1"
    onkeydown={key}
  >
    <div class="flex shrink-0 items-center gap-2 px-4 py-3 text-sm text-white">
      <span class="font-mono text-xs text-white/70">{index + 1} / {items.length}</span>
      <span class="min-w-0 flex-1 truncate" title={item.caption}>{item.caption}</span>
      <a class={button} href={item.url} download={item.filename}>Download</a>
      <button class={button} onclick={() => onAnchor(item)} title="Use this image as the subject of the next edit">Anchor for editing</button>
      <button class={button} onclick={onClose} aria-label="Close gallery">Close</button>
    </div>

    <div class="relative flex min-h-0 flex-1 items-center justify-center px-14">
      <!-- a click on the dark space around the image closes the gallery -->
      <button class="absolute inset-0 cursor-default" onclick={onClose} aria-label="Close gallery" tabindex="-1"></button>
      <img src={item.url} alt={item.caption} class="relative max-h-full max-w-full object-contain shadow-2xl" />
      {#if items.length > 1}
        <button class="absolute top-1/2 left-2 -translate-y-1/2 rounded-full {button} px-3 py-3" onclick={() => go(-1)} aria-label="Previous image">
          <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 18l-6-6 6-6" /></svg>
        </button>
        <button class="absolute top-1/2 right-2 -translate-y-1/2 rounded-full {button} px-3 py-3" onclick={() => go(1)} aria-label="Next image">
          <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6" /></svg>
        </button>
      {/if}
    </div>

    {#if items.length > 1}
      <div bind:this={strip} class="flex shrink-0 gap-2 overflow-x-auto px-4 py-3">
        {#each items as entry, i (i)}
          <button
            data-index={i}
            class="h-16 w-16 shrink-0 overflow-hidden rounded border-2 {i === index ? 'border-white' : 'border-transparent opacity-60 hover:opacity-100'}"
            onclick={() => (index = i)}
            aria-label="Image {i + 1}: {entry.caption}"
            aria-current={i === index ? "true" : undefined}
          >
            <img src={entry.url} alt="" class="h-full w-full object-cover" loading="lazy" />
          </button>
        {/each}
      </div>
    {/if}
  </div>
{/if}
