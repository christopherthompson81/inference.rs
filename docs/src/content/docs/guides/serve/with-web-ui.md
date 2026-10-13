---
title: Use the built-in web UI
description: What you get from the built-in web UI mounted at /ui, and how to customize it.
---

```bash
inference serve --model-folder ~/models
```

Open `http://localhost:1234/ui`. The built-in web UI is mounted at `/ui` by default whenever you run `inference serve`. It is a single-page app bundled into the binary; nothing is fetched from the network at runtime.

Started without `-m` or `-f`, the server loads nothing until you choose models in the UI. It has a tab per job, each with its own model:

- **Chat**: the chat panel below, over a text or multimodal model.
- **Speech to text**: an audio file's transcript, sentence by sentence. With a diarization model loaded too, it finds the speakers first and transcribes each one's turns, with a progress bar counting them; each speaker can be given a name. Playback highlights each word as it is spoken, and clicking a word or a segment's time plays from there. The transcript exports as SRT or WebVTT subtitles, CSV, JSON (with word times) or plain text, under the speakers' names.
- **Text to speech**: a reader for long text. Paste or open a markdown file and it splits into paragraphs (headings, list items, quotes and table rows; code isn't read), each generated as its own clip with a badge showing whether it is waiting, generating, ready (with its length) or changed since. Editing the text re-generates only the paragraphs that changed, and changing the voice or speed re-generates them all. Playback highlights the word being read and runs from one paragraph into the next; clicking a word plays from it. The **IPA** option shows each word's IPA above it once its paragraph is generated. Exports the whole reading as one WAV, its word timings as JSON, or the markdown.
- **Images**: text to image with HiDream-O1 or FLUX, and with reference images, HiDream-O1's edits.
- **Documents**: PP-DocLayoutV3 layout regions drawn over a page image, each then read by an OCR model such as PaddleOCR-VL with the prompt for its kind (text, table, formula, chart).
- **Models**: the models in the `--model-folder` folders and the Hugging Face cache, loaded into a tab with a click, downloaded from the Hub with progress, or deleted.

Each tab's model bar picks a model from those folders or the cache, or names a Hugging Face repository, and loads it with its options: the KV context length, the concurrent sequences (each holds memory while it runs; the server's `--max-seqs`, 4, when left at 0) and MTP speculative decoding for chat models, the projector beside a text GGUF, the device for the layout model. Loading shows its progress and replaces the tab's previous model; **Unload** frees it. A model started with `-m` still loads at startup and can be chatted with as before.

Each job tab keeps a history beside it, as the chat keeps its chats: every finished transcription, speech, image and document is saved on the server with its inputs, its results and the model that made them. Clicking an entry shows it again, ready to run again; transcribing a history entry again, or reading a page after detecting its layout, updates that entry rather than adding another. Jobs are kept under `~/.cache/inference-rs/jobs` (or `$XDG_CACHE_HOME`).

Loading, unloading, downloading and deleting are an admin's. On a server without keys that is a browser on the server's own machine, whether it opens the UI at `localhost`, the machine's address or its hostname (not a dotted name such as `.local`, which the network may answer); everyone else, such as another device on the network, sees which models are loaded and uses them. With keys, the owners named with `--admin` are admins. See [the model manager](/reference/http-api/#model-manager) for the routes behind the tabs.

The chat panel provides:

- Markdown rendering, code block syntax highlighting, and streamed output.
- A settings drawer with sampling controls (temperature, top-p, top-k, max tokens) and a system prompt field.
- Inline rendering of thinking tokens when the model emits them, with Default, Off, Low, Medium, High, and XHigh reasoning controls.
- Tool-call visualization when `--enable-search`, `--enable-code-execution`, or `--enable-shell` is also set.
- Tool approval cards when approval mode is `ask`.
- Multimodal attachments when the loaded model supports images, audio, or video.

## With agents enabled

```bash
inference serve --agent -m Qwen/Qwen3-4B
```

See [tool calling](/guides/agents/tool-calling-basics/) for what `--agent` enables and the individual flags behind it.

When the model calls a tool, the UI renders a collapsed block in the conversation. Expanding shows the tool arguments and result. Search blocks display the query and a result count; code-execution blocks display the executed Python and any output or images; shell blocks display the commands, stdout/stderr, and exit status.

The settings drawer controls search, code execution, shell, and tool approval per conversation without restarting the server. Set **Tool approval** to `ask` to approve or deny each agent action inline.

On Linux and macOS, code and shell execution use the default [OS-level sandbox](/reference/sandbox/) unless the server is started with `--sandbox off`. For the server, HTTP, Python, Rust, and sandbox settings, see [enable code execution](/guides/agents/enable-code-execution/) and [enable shell execution](/guides/agents/enable-shell/).

## With multimodal models

When the loaded model accepts images, a paperclip icon appears in the input bar. Attaching an image sends an `image_url` content part (`{"type": "image_url", "image_url": {"url": "..."}}`). Audio and video work the same on supporting models.

Modality support per model is in the [supported models reference](/reference/supported-models/).

## System prompt and sampling

The settings drawer state is persistent per browser. System prompts and sampling parameters survive reloads.

Clearing browser local storage for the site resets all UI state.

## Disabling the UI

Pass `--no-ui` to `inference serve` to skip mounting the UI router. The HTTP API continues to serve normally.

Without [API keys](/reference/http-api/#authentication) the UI is single-user: anyone who can reach `/ui` sees every
saved chat, and through them the sessions and files those chats used (`serve` warns when it is mounted on a
non-loopback address). With keys, the UI asks for one, signs the browser in with a cookie, and each key's owner sees
only its own saved chats, job history, sessions and files; **sign out** in the status bar ends the session. Chats saved before the
server had keys belong to no owner and don't show. Uploaded media and generated speech sit under random names in
shared directories, so anyone signed in who is given such a URL can fetch it.
