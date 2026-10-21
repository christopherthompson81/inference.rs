---
title: "inference launch"
description: "Run Claude Code, opencode or Codex against an inference server, configured through their environment (no agent config file is written): a running server (`--url`, or localhost:1234), or one started from `--config` and stopped after"
sidebar:
  order: 4
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

Run Claude Code, opencode or Codex against an inference server, configured through their environment (no agent config file is written): a running server (`--url`, or localhost:1234), or one started from `--config` and stopped after

```
inference launch [OPTIONS] <AGENT> [-- <AGENT_ARGS>...]
```

| Option | Default | Description |
|---|---|---|
| `<AGENT>` | required | The agent to run. Possible values: `claude`, `opencode`, `codex`. |
| `--url <URL>` |  | Server to use (default http://127.0.0.1:1234, or the `--config` server's port) |
| `--config <CONFIG>` |  | Server config, as `from-config` takes, to start when nothing answers yet (and stop when the agent exits) |
| `--api-key-file <API_KEY_FILE>` |  | File holding the server API key (first line: the key or `name=key`); unset, `INFERENCE_API_KEY`, then the `--config` server's keys file |
| `--model <MODEL>` |  | Model id on the server; unset, its default model |
| `<AGENT_ARGS>` |  | Arguments passed to the agent, after `--` |

