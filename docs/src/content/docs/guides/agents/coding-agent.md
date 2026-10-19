---
title: Use the built-in coding agent
description: Run `inference agent`, a terminal coding agent with its own tool loop, over a model loaded in-process or an inference.rs server.
---

`inference agent` is a terminal coding agent. It sends your prompt to the model along with six tools and runs the
tool calls on your machine in the project directory. It then sends the results back and repeats until the model
answers. The loop runs in the client, the same way Claude Code, opencode or pi drive a model. The model can be loaded
in the same process or served by `inference serve` on another machine.

## Start it

Load a model in-process, using the same model options as `inference run`:

```bash
cd my-project
inference agent -m Qwen/Qwen3-8B
inference agent --format gguf -m unsloth/Qwen3-8B-GGUF -f Qwen3-8B-Q4_K_M.gguf
```

Or point it at a running server. In this mode `-m` names the server's model, and the API key comes from
`--api-key-file` (a bare key, or the first key of a `serve` keys file) or from `INFERENCE_API_KEY`:

```bash
inference agent --url http://gpu-box:1234 -m chat --api-key-file ~/.config/inference-rs/api-keys
```

Run a single prompt without the REPL. The answer goes to stdout and tool activity goes to stderr:

```bash
inference agent -m Qwen/Qwen3-8B --permission-mode auto -p "Fix the failing test in tests/parse.rs"
```

## Tools

| Tool | What it does |
|---|---|
| `read` | Reads a file with line numbers. `offset` and `limit` select a range. |
| `write` | Creates or replaces a whole file. |
| `edit` | Replaces exact text. The text must occur once, unless `replace_all` is set. |
| `bash` | Runs a command in the project directory, with stdout and stderr combined. The default timeout is 120 s and the maximum is 600 s. |
| `grep` | Searches file contents with a regex, skipping files that `.gitignore` excludes. |
| `glob` | Lists files that match a glob, skipping files that `.gitignore` excludes. |

The system prompt is short. It covers the project directory, platform, date and git branch, plus the project's
`AGENTS.md` and `CLAUDE.md` when they exist.

The system prompt and the tool list stay the same for the whole session, and history only grows. Because of this,
each request starts with the previous one, and the server's prefix cache serves everything except the newest
messages. The REPL prints the cached share after each turn.

## Permissions

`--permission-mode` (or `/mode` in the REPL) decides which calls ask first:

| Mode | Reads inside the project | Reads outside | Edits and commands |
|---|---|---|---|
| `ask` (default) | run | ask | ask |
| `auto` | run | run | run |
| `read-only` | run | ask | refused |

At an `ask` prompt, `a` allows that tool for the rest of the session. The blanket allow only covers paths inside the
project: an edit outside the project still asks every time. With `-p` there is no one to ask, so calls that would
ask are refused and the model is told why.

The tools are not sandboxed: `bash` runs with your permissions. Use `ask` or `read-only` in projects you don't want
changed without review.

## Sessions

Each session is a JSONL file under `~/.local/share/inference-rs/agent/sessions/` (the platform data directory). The
file holds a header line and then one line per message, and is only ever appended to. An assistant reply is written
together with its tool results, so the history never holds a call without a result.

```bash
inference agent --sessions             # this project's sessions
inference agent --continue -m ...      # the newest one for this project
inference agent --resume 1792370089 -m ...
```

In the REPL, `/sessions`, `/resume <id>` and `/clear` (start a new session) do the same.

## Interrupting

Ctrl-C stops the model mid-reply, or the running tool. The partial reply is kept, and calls that never ran are
recorded as interrupted, so you can carry on with the next prompt. A second Ctrl-C quits.
