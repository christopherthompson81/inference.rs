# inference-coder

The coding agent behind `inference agent`. It is a library over `inference-api`: the harness owns the tool loop, so
the model can run in this process (`EngineBackend`) or on any OpenAI-compatible server (`HttpBackend`). Both backends
take OpenAI chat JSON and stream OpenAI chunk JSON through the `ChatBackend` trait.

| Module | Role |
|---|---|
| `agent` | The loop: request, stream, run tools, repeat; `AgentUi` is what a front end implements |
| `backend` | `ChatBackend` and its two implementations |
| `turn` | Folds streamed chunks into one assistant message (text, reasoning, tool calls by index) |
| `tools` | `read`, `write`, `edit`, `bash`, `grep`, `glob`, behind the `Tool` trait |
| `permission` | `ask` / `auto` / `read-only`, with per-tool "always" for paths inside the project |
| `prompt` | The system prompt: base text, environment, `AGENTS.md` / `CLAUDE.md` |
| `session` | Append-only JSONL sessions with list, find and continue-latest |

Design choices that keep the server's prefix cache hot:

- The system prompt and the tool list are fixed for the session, and the tools are always in the same order.
- History only grows.
- Assistant messages keep `reasoning_content`, so a template that renders reasoning re-renders the turn exactly as
  the model generated it.

The guide is `docs/src/content/docs/guides/agents/coding-agent.md`.
