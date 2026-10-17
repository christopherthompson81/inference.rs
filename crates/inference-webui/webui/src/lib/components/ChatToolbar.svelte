<script lang="ts">
  import { modelStore } from "../stores/models.svelte";
  import { settingsStore } from "../stores/settings.svelte";
  import ParamPopover from "./ParamPopover.svelte";

  type ParamKey = "temp" | "top_p" | "top_k" | "rep" | "max";

  let openParam = $state<ParamKey | null>(null);

  let modelDefaults = $derived(
    modelStore.models.find((m) => m.name === modelStore.selectedModel)?.generation_defaults,
  );

  let connection = $derived(`${window.location.hostname}:${window.location.port || (window.location.protocol === "https:" ? "443" : "80")}`);
</script>

<header class="flex h-12 shrink-0 items-center gap-3 border-b border-gray-200 bg-white px-4 dark:border-gray-800 dark:bg-gray-950">
  <!-- Inline sampling params (click to edit) -->
  <div class="hidden items-center gap-1 font-mono text-[11px] text-gray-500 md:flex dark:text-gray-400">
    {#snippet param(key: ParamKey, label: string, value: number | null | undefined)}
      <div class="relative">
        <button
          class="rounded px-1.5 py-0.5 hover:bg-gray-100 dark:hover:bg-gray-800"
          onclick={() => (openParam = openParam === key ? null : key)}
        >
          <span class="text-gray-400 dark:text-gray-500">{label}</span>
          <!-- an f32 default (1.1) comes back as 1.100000023841858 -->
          <span class="text-gray-700 dark:text-gray-200">{value == null ? "-" : +value.toFixed(4)}</span>
        </button>
        {#if openParam === key}
          {#if key === "temp"}
            <ParamPopover
              label="temperature"
              value={settingsStore.temperature}
              min={0} max={2} step={0.05}
              defaultValue={modelDefaults?.temperature ?? 0.7}
              onCommit={(v) => { if (v != null) { settingsStore.temperature = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {:else if key === "top_p"}
            <ParamPopover
              label="top_p"
              value={settingsStore.topP}
              min={0} max={1} step={0.01}
              defaultValue={modelDefaults?.top_p ?? 0.9}
              onCommit={(v) => { if (v != null) { settingsStore.topP = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {:else if key === "top_k"}
            <ParamPopover
              label="top_k"
              value={settingsStore.topK}
              min={0} step={1}
              defaultValue={modelDefaults?.top_k ?? 40}
              onCommit={(v) => { if (v != null) { settingsStore.topK = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {:else if key === "rep"}
            <ParamPopover
              label="repetition penalty"
              value={settingsStore.repetitionPenalty}
              min={0.5} max={2} step={0.01}
              defaultValue={modelDefaults?.repetition_penalty ?? 1.1}
              onCommit={(v) => { if (v != null) { settingsStore.repetitionPenalty = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {:else if key === "max"}
            <ParamPopover
              label="max tokens"
              value={settingsStore.maxTokens}
              min={1} step={1}
              defaultValue={modelDefaults?.max_tokens ?? 2048}
              onCommit={(v) => { if (v != null) { settingsStore.maxTokens = v; settingsStore.persist(); } }}
              onClose={() => (openParam = null)}
            />
          {/if}
        {/if}
      </div>
    {/snippet}

    <label
      class="flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-gray-100 dark:hover:bg-gray-800"
      title="Always take the likeliest token, with no repetition penalty, over the sampling values"
    >
      <input
        type="checkbox"
        class="accent-blue-600"
        bind:checked={settingsStore.greedy}
        onchange={() => settingsStore.persist()}
      />
      greedy
    </label>
    <div class="flex items-center gap-1 {settingsStore.greedy ? 'pointer-events-none line-through opacity-40' : ''}">
      {@render param("temp", "temp", settingsStore.temperature)}
      {@render param("top_p", "top_p", settingsStore.topP)}
      {@render param("top_k", "top_k", settingsStore.topK)}
      {@render param("rep", "rep", settingsStore.repetitionPenalty)}
    </div>
    {@render param("max", "max", settingsStore.maxTokens)}

    <button
      class="rounded px-1.5 py-0.5 hover:bg-gray-100 dark:hover:bg-gray-800 {settingsStore.reasoningEffort === 'off' ? 'text-gray-400 dark:text-gray-600' : 'text-amber-600 dark:text-amber-400'}"
      onclick={() => (settingsStore.settingsOpen = true)}
      title="Set reasoning effort"
    >
      reasoning:{settingsStore.reasoningEffort || "default"}
    </button>
  </div>

  <!-- Active tool indicators -->
  <div class="ml-auto flex items-center gap-1.5">
    {#if settingsStore.enableSearch && modelStore.capabilities.search_enabled}
      <span class="flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 dark:bg-green-900/40 dark:text-green-300" title="Web search active">
        <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        Search
      </span>
    {/if}
    {#if settingsStore.enableCodeExecution && modelStore.capabilities.code_execution_enabled}
      <span class="flex items-center gap-1 rounded-full bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-700 dark:bg-purple-900/40 dark:text-purple-300" title="Code execution active">
        <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
        Code
      </span>
    {/if}
    {#if settingsStore.enableShell && modelStore.capabilities.shell_enabled}
      <span class="flex items-center gap-1 rounded-full bg-teal-100 px-2 py-0.5 text-xs font-medium text-teal-700 dark:bg-teal-900/40 dark:text-teal-300" title="Shell active">
        <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 17l6-6-6-6M12 19h8" />
        </svg>
        Shell
      </span>
    {/if}
    <span class="hidden items-center gap-1.5 rounded-full border border-gray-200 px-2 py-0.5 font-mono text-[11px] text-gray-500 md:inline-flex dark:border-gray-700 dark:text-gray-400" title="Server endpoint">
      <span class="h-1.5 w-1.5 rounded-full bg-green-500"></span>
      {connection}
    </span>
  </div>

  <button
    class="rounded-md p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-200"
    onclick={() => (settingsStore.toolsOpen = !settingsStore.toolsOpen)}
    aria-label="Tools panel"
    title="Tools"
  >
    <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z" />
    </svg>
  </button>

  <button
    class="rounded-md p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-200"
    onclick={() => (settingsStore.settingsOpen = !settingsStore.settingsOpen)}
    aria-label="Settings"
  >
    <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  </button>
</header>
