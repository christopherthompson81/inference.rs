<script lang="ts">
  interface Props {
    paused: boolean;
    position: number;
    length: number;
    /** What the previous and next buttons step between, as in "paragraph". */
    unit: string;
    ontoggle: () => void;
    onseek: (time: number) => void;
    onstep: (direction: 1 | -1) => void;
  }

  let { paused, position, length, unit, ontoggle, onseek, onstep }: Props = $props();

  const clock = (seconds: number) => {
    const total = Math.max(0, Math.floor(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor(total / 60) % 60;
    const secs = String(total % 60).padStart(2, "0");
    return hours > 0 ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
  };

  const side = "rounded p-1 text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800";
</script>

<!-- stays in view above the text it plays -->
<div class="sticky top-0 z-10 flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 shadow-sm dark:border-gray-800 dark:bg-gray-950">
  <button class={side} onclick={() => onstep(-1)} aria-label="Previous {unit}" title="Previous {unit}">
    <svg class="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h2v14H6zM20 5v14L9 12z" /></svg>
  </button>
  <button class="rounded-full bg-blue-600 p-2 text-white hover:bg-blue-700" onclick={ontoggle} aria-label={paused ? "Play" : "Pause"} title={paused ? "Play" : "Pause"}>
    {#if paused}
      <svg class="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4v16l13-8z" /></svg>
    {:else}
      <svg class="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zM14 4h4v16h-4z" /></svg>
    {/if}
  </button>
  <button class={side} onclick={() => onstep(1)} aria-label="Next {unit}" title="Next {unit}">
    <svg class="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M16 5h2v14h-2zM4 5v14l11-7z" /></svg>
  </button>
  <input
    type="range"
    class="min-w-0 flex-1 accent-blue-600"
    min="0"
    max={length}
    step="0.1"
    value={position}
    aria-label="Position in the audio"
    oninput={(e) => onseek(Number(e.currentTarget.value))}
  />
  <span class="shrink-0 font-mono text-xs text-gray-500">{clock(position)} / {clock(length)}</span>
</div>
