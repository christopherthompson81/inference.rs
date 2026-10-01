<script lang="ts">
  import { signIn } from "../services/auth";

  let key = $state("");
  let error = $state("");
  let busy = $state(false);

  async function submit(event: SubmitEvent) {
    event.preventDefault();
    if (!key.trim()) return;
    busy = true;
    error = "";
    try {
      await signIn(key.trim());
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      busy = false;
    }
  }
</script>

<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
  <form
    class="w-80 rounded-lg border border-gray-200 bg-white p-5 shadow-lg dark:border-gray-800 dark:bg-gray-950"
    onsubmit={submit}
  >
    <h2 class="mb-1 text-base font-semibold">Sign in</h2>
    <p class="mb-3 text-sm text-gray-500 dark:text-gray-400">This server needs an API key.</p>
    <input
      type="password"
      autocomplete="current-password"
      class="mb-2 w-full rounded border border-gray-300 bg-transparent px-2 py-1.5 text-sm dark:border-gray-700"
      placeholder="API key"
      bind:value={key}
    />
    {#if error}
      <p class="mb-2 text-sm text-red-600 dark:text-red-400">{error}</p>
    {/if}
    <button
      type="submit"
      class="w-full rounded bg-gray-900 px-3 py-1.5 text-sm text-white disabled:opacity-50 dark:bg-gray-100 dark:text-gray-900"
      disabled={busy}
    >
      Sign in
    </button>
  </form>
</div>
