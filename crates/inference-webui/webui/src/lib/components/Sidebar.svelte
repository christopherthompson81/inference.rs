<script lang="ts">
  import HistoryPanel from "./HistoryPanel.svelte";
  import { chatStore } from "../stores/chat.svelte";
  import { SLOTS } from "../stores/manager.svelte";
  import { when } from "../utils/when";
  import * as api from "../services/api";
  import type { ChatFile } from "../types";

  const LABEL_LENGTH = 40;

  let chats = $state<ChatFile[]>([]);
  let error = $state("");

  async function refreshChats() {
    try {
      const result = await api.listChats();
      chats = result.chats.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    } catch {
      // the list stays as it was
    }
  }

  // a chat is created by its first message, so the list follows the chat shown
  $effect(() => {
    void chatStore.currentChatId;
    void refreshChats();
  });

  async function act(action: () => Promise<unknown>) {
    try {
      error = "";
      await action();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
    await refreshChats();
  }

  function label(chat: ChatFile): string {
    if (chat.title) return chat.title;
    const first = chat.messages.find((m) => m.role === "user");
    if (!first) return `Chat ${chat.id}`;
    return first.content.slice(0, LABEL_LENGTH) + (first.content.length > LABEL_LENGTH ? "..." : "");
  }

  // the slot's alias says nothing the tab doesn't, so only another model's name is shown
  function detail(chat: ChatFile): string {
    const turns = chat.messages.filter((m) => m.role === "user").length;
    const parts = [when(chat.created_at)];
    if (chat.model && chat.model !== SLOTS.chat.id) parts.push(chat.model);
    if (turns > 0) parts.push(`${turns} turn${turns === 1 ? "" : "s"}`);
    return parts.join(", ");
  }

  let entries = $derived(chats.map((chat) => ({ id: chat.id!, title: label(chat), detail: detail(chat) })));
</script>

<HistoryPanel
  {entries}
  active={chatStore.currentChatId}
  newLabel="New chat"
  onNew={() => act(() => chatStore.newChat())}
  onOpen={(id) => act(() => chatStore.loadChat(id))}
  onDelete={(id) => act(() => chatStore.deleteChat(id))}
  onRename={(id, title) => act(() => api.renameChat(id, title))}
  busy={chatStore.isStreaming}
  {error}
  empty="Chats show here."
/>
