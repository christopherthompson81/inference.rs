---
title: "inference agent"
description: "Coding agent: a tool loop (read, write, edit, bash, grep, glob) over the project directory, with the model in this process or on a server (`--url`)"
sidebar:
  order: 5
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

Coding agent: a tool loop (read, write, edit, bash, grep, glob) over the project directory, with the model in this process or on a server (`--url`)

```
inference agent [OPTIONS] [COMMAND]
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` |  | Hugging Face model ID or local model directory; optional when `-f` names local files |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--hf-overrides <HF_OVERRIDES>` |  | Recursively merged JSON overrides for the Hugging Face model config |
| `--max-model-len <MAX_MODEL_LEN>` |  | Runtime model context length |
| `--format <FORMAT>` |  | Model format: plain (safetensors), GGUF, or GGML. Auto-detected from `-f` when not specified. Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF projector override; auto-selected when unambiguous (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Optional model ID overriding configuration, tokenizer, and processor assets for a quantized model |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--enable-lora` | `false` | Enable dynamic LoRA without preloading an adapter. Supports compatible text and multimodal language models, including GGUF. Vision, audio, and projector adapters are unsupported |
| `--lora <ALIAS=SOURCE\|JSON>` |  | Preload a language-model LoRA adapter as ALIAS=SOURCE. Supports compatible text and multimodal language models, including GGUF. Remote adapters use revision main. May be repeated. Vision, audio, and projector adapters are unsupported |
| `--lora-max-adapters <LORA_MAX_ADAPTERS>` | `16` | Maximum loaded LoRA aliases and, independently, resident adapter generations |
| `--lora-max-rank <LORA_MAX_RANK>` | `256` | Maximum rank accepted for a LoRA adapter |
| `--lora-max-bytes <BYTES>` | `8589934592` | Maximum memory used by loaded adapters |
| `--quant <QUANT>` |  | Quantization target. Inference commands select a matching GGUF or UQFF artifact when available. Source checkpoints without a matching UQFF use in-situ quantization. `tune` evaluates the requested level instead of selecting an artifact. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or supported quantization names |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization target. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or raw quant names (`q4k`, `q8_0`, etc.). Supports compatible GGUF sources |
| `--from-uqff <FROM_UQFF>` |  | UQFF artifact to load. Accepts a filename, numeric quantization level (`2`, `3`, `4`, `5`, `6`, `8`), or quantization type (`q4k`, `afq8`, etc.). Report-declared artifacts and conventional shard names expand to all of their shards. Use semicolons only to list disjoint shards manually |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable. Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache storage: auto (the model's dtype, or q3 when that buys GPU room for offloaded MoE experts), none (the model's dtype), f8e4m3, q2..q8 (K and V at those bits), or qK/qV such as q8/q4 |
| `--pa-cache-layers <CACHE_LAYERS>` |  | Per-layer KV cache storage on top of --pa-cache-type: layers=type items such as 0=q8,20-27=q6, or uniform (the type everywhere, without the model's profile or the general rule) |
| `--encoder-cache-memory-mb <ENCODER_CACHE_MEMORY_MB>` |  | Maximum logical tensor memory retained by the multimodal encoder cache, in MiB |
| `--max-edge <MAX_EDGE>` |  | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <MAX_NUM_IMAGES>` |  | Maximum number of images per request |
| `--max-image-length <MAX_IMAGE_LENGTH>` |  | Maximum image dimension for device mapping |
| `--max-seqs <MAX_SEQS>` | `4` | Maximum concurrent sequences per model; each holds memory while it runs, so raise it for a busy shared server |
| `--max-num-batched-tokens <MAX_NUM_BATCHED_TOKENS>` |  | Maximum tokens processed in one paged-attention scheduler step [default: 4096, or the prompt chunk if longer] |
| `--no-embeddings-on-host` | `false` | Keep the token embedding in VRAM; by default an untied one lives in pinned host memory the GPU reads over the bus |
| `--moe-expert-cache <EXPERTS>` |  | Keep MoE experts (GGUF, on CUDA) in host memory and cache this many per layer on the GPU; the CPU runs the rest |
| `--max-prefill-chunk-tokens <MAX_PREFILL_CHUNK_TOKENS>` |  | CUDA prompt-token quantum used while decode is resident and for recurrent prefill batching [default: 512, or for MoE experts in host memory the longest chunk the expert cache leaves room for, up to 8192] |
| `--max-decode-steps-before-prefill <MAX_DECODE_STEPS_BEFORE_PREFILL>` | `8` | Maximum decode steps before a waiting prefill batch is admitted |
| `--no-kv-cache` | `false` | Disable KV cache entirely |
| `--prefix-cache-n <PREFIX_CACHE_N>` | `16` | Number of prefix caches to hold (0 to disable) |
| `--prefix-cache-dir <PREFIX_CACHE_DIR>` |  | Directory that keeps prefix cache entries on disk too, so long sessions (agent loops) survive eviction and restarts. Hybrid models on CUDA with paged attention |
| `--prefix-cache-disk-gib <PREFIX_CACHE_DISK_GIB>` |  | Most the prefix cache directory may hold, in GiB; the least recently used entries go first |
| `--prefix-cache-ram-gib <PREFIX_CACHE_RAM_GIB>` |  | Host memory for prefix cache entries, in GiB: in front of `--prefix-cache-dir`, or on its own without one |
| `--prefix-cache-checkpoint-tokens <PREFIX_CACHE_CHECKPOINT_TOKENS>` |  | Tokens between a long prompt's recurrent checkpoints in the host prefix cache, so a request that diverges early in it (an edited earlier turn) redoes at most this many; default 16384, 0 keeps only the final ones |
| `--prefix-cache-disk-ttl-mins <PREFIX_CACHE_DISK_TTL_MINS>` |  | Minutes a host prefix cache entry (in memory or on disk) outlives its last use, unless its request names a TTL (an Anthropic `cache_control` breakpoint: 5 minutes, or its `ttl`); default 60 |
| `-c, --chat-template <CHAT_TEMPLATE>` |  | Custom chat template file (.json or .jinja) |
| `-j, --jinja-explicit <JINJA_EXPLICIT>` |  | Explicit JINJA template override |
| `--matformer-config-path <MATFORMER_CONFIG_PATH>` |  | Path to a MatFormer config (CSV/JSON describing available slices). See model card |
| `--matformer-slice-name <MATFORMER_SLICE_NAME>` |  | MatFormer slice to load (must match a slice name in the config file) |
| `--mtp` | `false` | Enable MTP speculative decoding with the head built into the model checkpoint |
| `--mtp-model <MTP_MODEL>` |  | MTP assistant model id or path |
| `--mtp-n-predict <MTP_N_PREDICT>` |  | Number of MTP draft tokens to propose per target step |
| `--mtp-draft-sampling <MTP_DRAFT_SAMPLING>` | `auto` | MTP draft sampling policy. Auto uses probabilistic DFlash2 drafting when supported. Possible values: `auto`, `greedy`, `probabilistic`. |
| `--mtp-draft-isq <MTP_DRAFT_ISQ>` |  | Weight type of the MTP assistant model, e.g. q4k or q8_0; unset, the model's ISQ (a GGUF's drafter stays BF16) |
| `--mcp-config <MCP_CONFIG>` |  | Path to an MCP client configuration JSON. Also reads `MCP_CONFIG_PATH` if unset |
| `--sandbox <MODE>` | `auto` | Sandbox mode. Possible values: `auto`, `on`, `off`. |
| `--sandbox-profile <PROFILE>` |  | Sandbox policy profile. Possible values: `restricted`, `developer`. |
| `--sb-max-memory-mb <MEMORY_MB>` |  | Per-session memory cap in MiB (default: 2048) |
| `--sb-max-cpu-secs <CPU_SECS>` |  | Per-session CPU time cap in seconds (default: 600). Raised to at least enabled code/shell timeouts |
| `--sb-max-procs <PROCS>` |  | Per-session process/thread cap (default: 64) |
| `--sandbox-network <NETWORK>` |  | Network access permitted to the sandboxed session. Possible values: `none`, `loopback`, `full`. |
| `--url <URL>` |  | Use the model on an OpenAI-compatible server at this base URL instead of loading one; `-m` names the server's model |
| `--api-key-file <API_KEY_FILE>` |  | File holding the server's API key (default: the INFERENCE_API_KEY environment variable) |
| `-p, --prompt <PROMPT>` |  | Run this prompt without the REPL, print the answer and exit |
| `--continue` | `false` | Continue the newest session for this project |
| `--resume <RESUME>` |  | Resume a session by id (or an id prefix) |
| `--sessions` | `false` | List this project's sessions and exit |
| `--permission-mode <PERMISSION_MODE>` | `ask` | Which tool calls ask first: ask (edits and commands ask), auto (nothing asks) or read-only (edits and commands are refused). With `-p`, calls that would ask are refused |
| `--cwd <CWD>` |  | The project directory the tools work in (default: the current directory) |
| `--max-rounds <MAX_ROUNDS>` | `200` | Model replies (each followed by its tool calls) one prompt may take before the agent stops |
| `--max-reply-tokens <MAX_REPLY_TOKENS>` | `32768` | Tokens one model reply may take, reasoning included; a reply cut there ends the prompt (a model repeating itself otherwise runs until the context is full) |
| `--thinking <THINKING>` |  | Control thinking mode for models that support it. Use --thinking or --thinking true to force on, --thinking false to force off. Possible values: `true`, `false`. |
| `--reasoning-effort <REASONING_EFFORT>` |  | Set reasoning effort: off, low, medium, high, or xhigh. "none" is an alias for off |

## inference agent auto

Auto-detect model type (recommended)

```
inference agent auto [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Text model architecture; forces the text loader instead of auto-detection |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--hf-overrides <HF_OVERRIDES>` |  | Recursively merged JSON overrides for the Hugging Face model config |
| `--max-model-len <MAX_MODEL_LEN>` |  | Runtime model context length |
| `--format <FORMAT>` |  | Model format: plain (safetensors), GGUF, or GGML. Auto-detected from `-f` when not specified. Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF projector override; auto-selected when unambiguous (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Optional model ID overriding configuration, tokenizer, and processor assets for a quantized model |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--enable-lora` | `false` | Enable dynamic LoRA without preloading an adapter. Supports compatible text and multimodal language models, including GGUF. Vision, audio, and projector adapters are unsupported |
| `--lora <ALIAS=SOURCE\|JSON>` |  | Preload a language-model LoRA adapter as ALIAS=SOURCE. Supports compatible text and multimodal language models, including GGUF. Remote adapters use revision main. May be repeated. Vision, audio, and projector adapters are unsupported |
| `--lora-max-adapters <LORA_MAX_ADAPTERS>` | `16` | Maximum loaded LoRA aliases and, independently, resident adapter generations |
| `--lora-max-rank <LORA_MAX_RANK>` | `256` | Maximum rank accepted for a LoRA adapter |
| `--lora-max-bytes <BYTES>` | `8589934592` | Maximum memory used by loaded adapters |
| `--quant <QUANT>` |  | Quantization target. Inference commands select a matching GGUF or UQFF artifact when available. Source checkpoints without a matching UQFF use in-situ quantization. `tune` evaluates the requested level instead of selecting an artifact. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or supported quantization names |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization target. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or raw quant names (`q4k`, `q8_0`, etc.). Supports compatible GGUF sources |
| `--from-uqff <FROM_UQFF>` |  | UQFF artifact to load. Accepts a filename, numeric quantization level (`2`, `3`, `4`, `5`, `6`, `8`), or quantization type (`q4k`, `afq8`, etc.). Report-declared artifacts and conventional shard names expand to all of their shards. Use semicolons only to list disjoint shards manually |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable. Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache storage: auto (the model's dtype, or q3 when that buys GPU room for offloaded MoE experts), none (the model's dtype), f8e4m3, q2..q8 (K and V at those bits), or qK/qV such as q8/q4 |
| `--pa-cache-layers <CACHE_LAYERS>` |  | Per-layer KV cache storage on top of --pa-cache-type: layers=type items such as 0=q8,20-27=q6, or uniform (the type everywhere, without the model's profile or the general rule) |
| `--encoder-cache-memory-mb <ENCODER_CACHE_MEMORY_MB>` |  | Maximum logical tensor memory retained by the multimodal encoder cache, in MiB |
| `--max-edge <MAX_EDGE>` |  | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <MAX_NUM_IMAGES>` |  | Maximum number of images per request |
| `--max-image-length <MAX_IMAGE_LENGTH>` |  | Maximum image dimension for device mapping |

## inference agent text

Text generation model with explicit configuration

```
inference agent text [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Text model architecture; forces the text loader instead of auto-detection |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--hf-overrides <HF_OVERRIDES>` |  | Recursively merged JSON overrides for the Hugging Face model config |
| `--max-model-len <MAX_MODEL_LEN>` |  | Runtime model context length |
| `--format <FORMAT>` |  | Model format: plain (safetensors), GGUF, or GGML. Auto-detected from `-f` when not specified. Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF projector override; auto-selected when unambiguous (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Optional model ID overriding configuration, tokenizer, and processor assets for a quantized model |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--enable-lora` | `false` | Enable dynamic LoRA without preloading an adapter. Supports compatible text and multimodal language models, including GGUF. Vision, audio, and projector adapters are unsupported |
| `--lora <ALIAS=SOURCE\|JSON>` |  | Preload a language-model LoRA adapter as ALIAS=SOURCE. Supports compatible text and multimodal language models, including GGUF. Remote adapters use revision main. May be repeated. Vision, audio, and projector adapters are unsupported |
| `--lora-max-adapters <LORA_MAX_ADAPTERS>` | `16` | Maximum loaded LoRA aliases and, independently, resident adapter generations |
| `--lora-max-rank <LORA_MAX_RANK>` | `256` | Maximum rank accepted for a LoRA adapter |
| `--lora-max-bytes <BYTES>` | `8589934592` | Maximum memory used by loaded adapters |
| `--quant <QUANT>` |  | Quantization target. Inference commands select a matching GGUF or UQFF artifact when available. Source checkpoints without a matching UQFF use in-situ quantization. `tune` evaluates the requested level instead of selecting an artifact. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or supported quantization names |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization target. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or raw quant names (`q4k`, `q8_0`, etc.). Supports compatible GGUF sources |
| `--from-uqff <FROM_UQFF>` |  | UQFF artifact to load. Accepts a filename, numeric quantization level (`2`, `3`, `4`, `5`, `6`, `8`), or quantization type (`q4k`, `afq8`, etc.). Report-declared artifacts and conventional shard names expand to all of their shards. Use semicolons only to list disjoint shards manually |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable. Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache storage: auto (the model's dtype, or q3 when that buys GPU room for offloaded MoE experts), none (the model's dtype), f8e4m3, q2..q8 (K and V at those bits), or qK/qV such as q8/q4 |
| `--pa-cache-layers <CACHE_LAYERS>` |  | Per-layer KV cache storage on top of --pa-cache-type: layers=type items such as 0=q8,20-27=q6, or uniform (the type everywhere, without the model's profile or the general rule) |

## inference agent multimodal

Multimodal model

```
inference agent multimodal [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Text model architecture; forces the text loader instead of auto-detection |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--hf-overrides <HF_OVERRIDES>` |  | Recursively merged JSON overrides for the Hugging Face model config |
| `--max-model-len <MAX_MODEL_LEN>` |  | Runtime model context length |
| `--format <FORMAT>` |  | Model format: plain (safetensors), GGUF, or GGML. Auto-detected from `-f` when not specified. Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF projector override; auto-selected when unambiguous (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Optional model ID overriding configuration, tokenizer, and processor assets for a quantized model |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--enable-lora` | `false` | Enable dynamic LoRA for the language model without preloading an adapter. Vision, audio, and projector adapters are unsupported |
| `--lora <ALIAS=SOURCE\|JSON>` |  | Preload a language-model LoRA adapter as ALIAS=SOURCE. Remote adapters use revision main. May be repeated. Vision, audio, and projector adapters are unsupported |
| `--lora-max-adapters <LORA_MAX_ADAPTERS>` | `16` | Maximum loaded LoRA aliases and, independently, resident adapter generations |
| `--lora-max-rank <LORA_MAX_RANK>` | `256` | Maximum rank accepted for a LoRA adapter |
| `--lora-max-bytes <BYTES>` | `8589934592` | Maximum memory used by loaded adapters |
| `--quant <QUANT>` |  | Quantization target. Inference commands select a matching GGUF or UQFF artifact when available. Source checkpoints without a matching UQFF use in-situ quantization. `tune` evaluates the requested level instead of selecting an artifact. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or supported quantization names |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization target. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or raw quant names (`q4k`, `q8_0`, etc.). Supports compatible GGUF sources |
| `--from-uqff <FROM_UQFF>` |  | UQFF artifact to load. Accepts a filename, numeric quantization level (`2`, `3`, `4`, `5`, `6`, `8`), or quantization type (`q4k`, `afq8`, etc.). Report-declared artifacts and conventional shard names expand to all of their shards. Use semicolons only to list disjoint shards manually |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable. Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache storage: auto (the model's dtype, or q3 when that buys GPU room for offloaded MoE experts), none (the model's dtype), f8e4m3, q2..q8 (K and V at those bits), or qK/qV such as q8/q4 |
| `--pa-cache-layers <CACHE_LAYERS>` |  | Per-layer KV cache storage on top of --pa-cache-type: layers=type items such as 0=q8,20-27=q6, or uniform (the type everywhere, without the model's profile or the general rule) |
| `--encoder-cache-memory-mb <ENCODER_CACHE_MEMORY_MB>` |  | Maximum logical tensor memory retained by the multimodal encoder cache, in MiB |
| `--max-edge <MAX_EDGE>` |  | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <MAX_NUM_IMAGES>` |  | Maximum number of images per request |
| `--max-image-length <MAX_IMAGE_LENGTH>` |  | Maximum image dimension for device mapping |

## inference agent diffusion

Image generation model (diffusion)

```
inference agent diffusion [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Text model architecture; forces the text loader instead of auto-detection |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--hf-overrides <HF_OVERRIDES>` |  | Recursively merged JSON overrides for the Hugging Face model config |
| `--max-model-len <MAX_MODEL_LEN>` |  | Runtime model context length |
| `--diffusion-arch <DIFFUSION_ARCH>` | `flux` | Diffusion architecture: flux, flux-offloaded, hidream-o1 or hidream-o1-dev |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |

## inference agent speech

Speech synthesis model

```
inference agent speech [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Text model architecture; forces the text loader instead of auto-detection |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--hf-overrides <HF_OVERRIDES>` |  | Recursively merged JSON overrides for the Hugging Face model config |
| `--max-model-len <MAX_MODEL_LEN>` |  | Runtime model context length |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |

## inference agent embedding

Embedding model

```
inference agent embedding [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Text model architecture; forces the text loader instead of auto-detection |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--hf-overrides <HF_OVERRIDES>` |  | Recursively merged JSON overrides for the Hugging Face model config |
| `--max-model-len <MAX_MODEL_LEN>` |  | Runtime model context length |
| `--format <FORMAT>` |  | Model format: plain (safetensors), GGUF, or GGML. Auto-detected from `-f` when not specified. Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF projector override; auto-selected when unambiguous (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Optional model ID overriding configuration, tokenizer, and processor assets for a quantized model |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--quant <QUANT>` |  | Quantization target. Inference commands select a matching GGUF or UQFF artifact when available. Source checkpoints without a matching UQFF use in-situ quantization. `tune` evaluates the requested level instead of selecting an artifact. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or supported quantization names |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization target. Accepts numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) or raw quant names (`q4k`, `q8_0`, etc.). Supports compatible GGUF sources |
| `--from-uqff <FROM_UQFF>` |  | UQFF artifact to load. Accepts a filename, numeric quantization level (`2`, `3`, `4`, `5`, `6`, `8`), or quantization type (`q4k`, `afq8`, etc.). Report-declared artifacts and conventional shard names expand to all of their shards. Use semicolons only to list disjoint shards manually |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable. Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache storage: auto (the model's dtype, or q3 when that buys GPU room for offloaded MoE experts), none (the model's dtype), f8e4m3, q2..q8 (K and V at those bits), or qK/qV such as q8/q4 |
| `--pa-cache-layers <CACHE_LAYERS>` |  | Per-layer KV cache storage on top of --pa-cache-type: layers=type items such as 0=q8,20-27=q6, or uniform (the type everywhere, without the model's profile or the general rule) |

