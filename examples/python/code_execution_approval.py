"""
Approve or deny each Python code execution from the terminal.

With `agent_permission=ask` the engine pauses before running code and emits an
`agentic_tool_approval_required` event on the stream; the app answers it with
`engine.resolve_approval`.

Run with:
    python examples/python/code_execution_approval.py
"""

import inference_rs as ir
from inference_rs import types as t


def ask(approval: dict) -> t.ApprovalDecisionRequest:
    tool = approval["tool"]
    arguments = approval["arguments"]
    print("\nAgent action approval required")
    print(f"approval_id: {approval['approval_id']}")
    print(f"session_id: {approval['session_id']}")
    print(f"tool: {tool['label']} ({tool['kind']})")
    if tool["kind"] == "code_execution":
        print("\nCode:")
        print(arguments.get("code") or "<no code>")
    else:
        print("\nArguments:")
        print(arguments)

    while True:
        answer = (
            input("\nRun this Python code? [y]es / [n]o / [a]lways: ").strip().lower()
        )
        if answer in {"y", "yes"}:
            return t.ApprovalDecisionRequest(decision=t.ApprovalDecision.APPROVE)
        if answer in {"a", "always"}:
            return t.ApprovalDecisionRequest(
                decision=t.ApprovalDecision.APPROVE, remember_for_session=True
            )
        if answer in {"", "n", "no"}:
            return t.ApprovalDecisionRequest(
                decision=t.ApprovalDecision.DENY,
                message="The user denied this action.",
            )
        print("Please enter y, n, or a.")


def main():
    spec = t.EngineSpec(
        model=t.ModelSelectedPlain(
            model_id="Qwen/Qwen3-4B", arch=t.NormalLoaderType.QWEN3
        ),
        agentic=t.AgenticSpec(code_execution=t.CodeExecutionConfig()),
    )

    with ir.Engine(spec) as engine:
        request = t.ChatCompletionRequest(
            model="default",
            messages=[
                t.Message(
                    role="user",
                    content="Use Python to calculate the first 20 Fibonacci numbers.",
                )
            ],
            tools=[
                t.OpenAiCodeInterpreterTool(
                    container=t.OpenAiCodeInterpreterAutoContainer()
                )
            ],
            agent_permission=t.AgentPermission.ASK,
            max_tool_rounds=4,
            stream=True,
        )
        with engine.chat_stream(request) as stream:
            for event in stream:
                if event.name == "error":
                    raise RuntimeError(event.data)
                if event.name == "agentic_tool_approval_required":
                    engine.resolve_approval(event.data["approval_id"], ask(event.data))
                elif event.name == "chunk":
                    for choice in event.data.choices:
                        print(choice.delta.content or "", end="", flush=True)
        print()


if __name__ == "__main__":
    main()
