<script lang="ts">
  import type { FileDiff } from "../services/code";

  interface Props {
    path: string;
    diff: FileDiff | null;
    error: string;
    onClose: () => void;
  }
  let { path, diff, error, onClose }: Props = $props();

  const STATUS_LABEL: Record<string, string> = {
    added: "added",
    modified: "modified",
    deleted: "deleted",
    no_original: "changed by a command",
  };
</script>

<div class="flex min-h-0 min-w-0 flex-1 flex-col" data-testid="code-diff">
  <div class="flex items-center gap-2 border-b border-gray-200 px-3 py-2 text-xs dark:border-gray-800">
    <span class="truncate font-mono text-gray-800 dark:text-gray-200" title={path}>{path}</span>
    {#if diff?.status}
      <span class="shrink-0 rounded bg-gray-100 px-1.5 py-0.5 text-gray-600 dark:bg-gray-800 dark:text-gray-400">
        {STATUS_LABEL[diff.status]}
      </span>
      {#if !diff.binary && !diff.too_large}
        <span class="shrink-0 font-mono text-green-600 dark:text-green-400">+{diff.added}</span>
        <span class="shrink-0 font-mono text-red-600 dark:text-red-400">-{diff.removed}</span>
      {/if}
    {:else if diff}
      <span class="shrink-0 text-gray-400">unchanged</span>
    {/if}
    <button
      class="ml-auto shrink-0 rounded px-2 py-0.5 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
      onclick={onClose}
      aria-label="Close diff">Close</button
    >
  </div>
  <div class="min-h-0 flex-1 overflow-auto">
    {#if error}
      <p class="p-3 text-sm text-red-600">{error}</p>
    {:else if !diff}
      <p class="p-3 text-sm text-gray-400">Loading...</p>
    {:else if diff.too_large}
      <p class="p-3 text-sm text-gray-400">Too large to show.</p>
    {:else if diff.binary}
      <p class="p-3 text-sm text-gray-400">A binary file.</p>
    {:else}
      {#if diff.status === "no_original"}
        <p class="border-b border-gray-200 px-3 py-2 text-xs text-gray-500 dark:border-gray-800">
          A command changed this file and there is no earlier copy to compare with, so this is the file as it is now.
        </p>
      {/if}
      <table class="w-full border-collapse font-mono text-xs leading-5">
        <tbody>
          {#each diff.hunks as hunk, index (index)}
            {#if index > 0}
              <tr><td colspan="4" class="bg-gray-50 px-2 text-gray-400 dark:bg-gray-900">...</td></tr>
            {/if}
            {#each hunk as line, n (n)}
              <tr
                class={line.kind === "added"
                  ? "bg-green-50 dark:bg-green-950/40"
                  : line.kind === "removed"
                    ? "bg-red-50 dark:bg-red-950/40"
                    : ""}
              >
                <td class="w-px px-2 text-right text-gray-400 select-none">{line.old ?? ""}</td>
                <td class="w-px px-2 text-right text-gray-400 select-none">{line.new ?? ""}</td>
                <td
                  class="w-px pl-1 select-none {line.kind === 'added'
                    ? 'text-green-700 dark:text-green-400'
                    : 'text-red-700 dark:text-red-400'}"
                >
                  {line.kind === "added" ? "+" : line.kind === "removed" ? "-" : ""}
                </td>
                <td class="pr-3 pl-1 whitespace-pre text-gray-800 dark:text-gray-200">{line.text}</td>
              </tr>
            {/each}
          {/each}
        </tbody>
      </table>
    {/if}
  </div>
</div>
