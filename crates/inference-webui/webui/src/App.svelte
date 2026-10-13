<script lang="ts">
  import ModelBar from "./lib/components/ModelBar.svelte";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import MessageList from "./lib/components/MessageList.svelte";
  import ChatInput from "./lib/components/ChatInput.svelte";
  import SettingsPanel from "./lib/components/SettingsPanel.svelte";
  import ToolsPanel from "./lib/components/ToolsPanel.svelte";
  import StatusBar from "./lib/components/StatusBar.svelte";
  import SignIn from "./lib/components/SignIn.svelte";
  import SlotBar from "./lib/components/SlotBar.svelte";
  import TranscriptionView from "./lib/views/TranscriptionView.svelte";
  import SpeechView from "./lib/views/SpeechView.svelte";
  import ImagesView from "./lib/views/ImagesView.svelte";
  import DocumentsView from "./lib/views/DocumentsView.svelte";
  import ModelsView from "./lib/views/ModelsView.svelte";
  import { authStore } from "./lib/stores/auth.svelte";
  import { managerStore, SLOTS, type SlotId } from "./lib/stores/manager.svelte";
  import { modelStore } from "./lib/stores/models.svelte";
  import { settingsStore } from "./lib/stores/settings.svelte";
  import * as api from "./lib/services/api";
  import { onMount } from "svelte";

  const TABS = [
    { id: "chat", label: "Chat" },
    { id: "transcription", label: "Speech to text" },
    { id: "speech", label: "Text to speech" },
    { id: "images", label: "Images" },
    { id: "documents", label: "Documents" },
    { id: "models", label: "Models" },
  ] as const;
  type Tab = (typeof TABS)[number]["id"];
  const TAB_KEY = "inference-rs-tab";
  // the tab each slot is loaded from
  const SLOT_TABS: Record<SlotId, Tab> = {
    chat: "chat",
    transcription: "transcription",
    diarization: "transcription",
    speech: "speech",
    images: "images",
    layout: "documents",
    ocr: "documents",
  };

  function storedTab(): Tab {
    try {
      const tab = localStorage.getItem(TAB_KEY);
      return TABS.some((t) => t.id === tab) ? (tab as Tab) : "chat";
    } catch {
      return "chat";
    }
  }

  let tab = $state<Tab>(storedTab());
  // set once onMount has applied the server's saved selection, which the effect below must not race
  let started = $state(false);

  function show(next: Tab) {
    tab = next;
    try {
      localStorage.setItem(TAB_KEY, next);
    } catch {
      // private windows may refuse storage
    }
  }

  // the chat's model list follows what is loaded, and moves to the chat slot when it is the only choice left
  async function refreshChatModels() {
    try {
      const { models } = await api.listModels();
      modelStore.models = models;
      const current = models.some((model) => model.name === modelStore.selectedModel);
      // the selection is the server's: change it only when the selected model went away, or none was selected
      if (!current && models.length > 0) {
        const preferred = models.find((model) => model.name === SLOTS.chat.id) ?? models[0];
        modelStore.selectModel(preferred.name);
        await api.selectModel(preferred.name).catch(() => {});
      } else if (models.length === 0) {
        modelStore.selectedModel = null;
      }
    } catch (e) {
      console.error("Failed to list models:", e);
    }
  }

  $effect(() => {
    void managerStore.changes;
    if (started) void refreshChatModels();
  });

  onMount(async () => {
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      document.documentElement.classList.add("dark");
    }

    try {
      const { models } = await api.listModels();
      modelStore.models = models;

      const settings = await api.getSettings();
      if (settings.model) {
        modelStore.selectModel(settings.model);
      } else if (models.length > 0) {
        modelStore.selectModel(models[0].name);
      }

      if (settings.defaults) {
        settingsStore.applyDefaults(settings.defaults);
      }

      try {
        const caps = await api.getCapabilities();
        modelStore.capabilities = caps;
        if (caps.search_enabled) settingsStore.enableSearch = true;
        if (caps.code_execution_enabled) settingsStore.enableCodeExecution = true;
        if (caps.shell_enabled) settingsStore.enableShell = true;
        managerStore.start(caps.admin ?? false);
      } catch {
        // Capabilities endpoint might not exist on older servers
      }
    } catch (e) {
      console.error("Failed to initialize:", e);
    } finally {
      started = true;
    }
  });
</script>

<div class="flex h-screen flex-col overflow-hidden">
  <!-- the tabs scroll on a phone rather than widen the page -->
  <nav class="flex h-10 shrink-0 items-center gap-1 overflow-x-auto border-b border-gray-200 bg-gray-50 px-3 dark:border-gray-800 dark:bg-gray-900">
    <img src="favicon.png" alt="" class="mr-1 h-5 w-5 shrink-0" />
    <span class="mr-3 shrink-0 text-sm font-semibold text-gray-800 dark:text-gray-100">inference.rs</span>
    {#each TABS as item (item.id)}
      <button
        class="shrink-0 rounded-md px-3 py-1 text-sm whitespace-nowrap {tab === item.id
          ? 'bg-white font-medium text-gray-900 shadow-sm dark:bg-gray-800 dark:text-gray-100'
          : 'text-gray-500 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200'}"
        onclick={() => show(item.id)}
        aria-current={tab === item.id ? "page" : undefined}
      >
        {item.label}
      </button>
    {/each}
  </nav>

  {#if tab === "chat"}
    <ModelBar />

    <div class="flex min-h-0 flex-1">
      {#if settingsStore.sidebarOpen}
        <button
          class="fixed inset-0 z-30 bg-black/30 md:hidden"
          onclick={() => (settingsStore.sidebarOpen = false)}
          aria-label="Close sidebar"
        ></button>
      {/if}

      {#if settingsStore.sidebarOpen}
        <div
          class="fixed top-22 bottom-7 z-40 flex w-72 flex-col border-r border-gray-200 bg-gray-50 md:hidden dark:border-gray-800 dark:bg-gray-900"
        >
          <Sidebar />
        </div>
        <div
          class="hidden w-72 shrink-0 flex-col border-r border-gray-200 bg-gray-50 md:flex dark:border-gray-800 dark:bg-gray-900"
        >
          <Sidebar />
        </div>
      {/if}

      <div class="flex min-w-0 flex-1 flex-col">
        <div class="px-4 pt-3">
          <SlotBar slot="chat" />
        </div>
        <MessageList />
        <ChatInput />
      </div>

      {#if settingsStore.settingsOpen}
        <button
          class="fixed inset-0 z-30 bg-black/30"
          onclick={() => (settingsStore.settingsOpen = false)}
          aria-label="Close settings"
        ></button>
        <div class="fixed right-0 top-22 bottom-7 z-40 w-80 overflow-y-auto border-l border-gray-200 bg-white shadow-lg dark:border-gray-800 dark:bg-gray-950">
          <SettingsPanel />
        </div>
      {/if}

      {#if settingsStore.toolsOpen}
        <button
          class="fixed inset-0 z-30 bg-black/30"
          onclick={() => (settingsStore.toolsOpen = false)}
          aria-label="Close tools"
        ></button>
        <div class="fixed right-0 top-22 bottom-7 z-40 w-96 overflow-hidden border-l border-gray-200 bg-white shadow-lg dark:border-gray-800 dark:bg-gray-950">
          <ToolsPanel />
        </div>
      {/if}
    </div>
  {:else}
    <main class="flex min-h-0 flex-1 overflow-hidden">
      {#if tab === "transcription"}
        <TranscriptionView />
      {:else if tab === "speech"}
        <SpeechView />
      {:else if tab === "images"}
        <ImagesView />
      {:else if tab === "documents"}
        <DocumentsView />
      {:else}
        <ModelsView onUse={(slot) => show(SLOT_TABS[slot])} />
      {/if}
    </main>
  {/if}

  <StatusBar />
</div>

{#if authStore.required}
  <SignIn />
{/if}
