<script lang="ts">
  import ChatToolbar from "./lib/components/ChatToolbar.svelte";
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
  import CodeView from "./lib/views/CodeView.svelte";
  import { authStore } from "./lib/stores/auth.svelte";
  import { managerStore, SLOTS, type SlotId } from "./lib/stores/manager.svelte";
  import { modelStore } from "./lib/stores/models.svelte";
  import { settingsStore } from "./lib/stores/settings.svelte";
  import * as api from "./lib/services/api";
  import { onMount } from "svelte";
  import type { UiModelInfo } from "./lib/types";

  const TABS = [
    { id: "chat", label: "Chat" },
    { id: "code", label: "Code" },
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

  // the chat talks to the chat slot's model, or with none loaded there the first text model the server holds
  async function pickChatModel(models: UiModelInfo[]) {
    modelStore.models = models;
    const chosen = models.find((model) => model.name === SLOTS.chat.id) ?? models[0];
    if (!chosen) return void (modelStore.selectedModel = null);
    if (chosen.name === modelStore.selectedModel) return;
    modelStore.selectModel(chosen.name);
    await api.selectModel(chosen.name).catch(() => {});
  }

  async function refreshChatModels() {
    try {
      await pickChatModel((await api.listModels()).models);
    } catch (e) {
      console.error("Failed to list models:", e);
    }
  }

  $effect(() => {
    void managerStore.changes;
    if (started) void refreshChatModels();
  });

  // a slot bar's Settings link names its slot's settings, which live on the Models tab
  function followSettingsLink() {
    if (location.hash.startsWith("#settings-")) show("models");
  }

  onMount(() => {
    followSettingsLink();
    window.addEventListener("hashchange", followSettingsLink);
    return () => window.removeEventListener("hashchange", followSettingsLink);
  });

  onMount(async () => {
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      document.documentElement.classList.add("dark");
    }

    try {
      await pickChatModel((await api.listModels()).models);
      const settings = await api.getSettings();

      if (settings.defaults) {
        settingsStore.applyDefaults(settings.defaults);
      }

      try {
        const caps = await api.getCapabilities();
        modelStore.capabilities = caps;
        if (caps.search_enabled) settingsStore.enableSearch = true;
        if (caps.code_execution_enabled) settingsStore.enableCodeExecution = true;
        if (caps.shell_enabled) settingsStore.enableShell = true;
        managerStore.start(caps.admin ?? false, caps.keyed ?? false);
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
    <div class="flex min-h-0 flex-1 flex-col md:flex-row">
      <Sidebar />

      <div class="flex min-h-0 min-w-0 flex-1 flex-col">
        <ChatToolbar />
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
      {:else if tab === "code"}
        <CodeView />
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
