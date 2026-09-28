import json
import os
import inference_rs as ir
from inference_rs import types as t


def local_search(query: str):
    results = []
    for root, _, files in os.walk("."):
        for f in files:
            if query in f:
                path = os.path.join(root, f)
                try:
                    content = open(path).read()
                except Exception:
                    content = ""
                results.append(
                    {
                        "title": f,
                        "description": path,
                        "url": path,
                        "content": content,
                    }
                )
    results.sort(key=lambda r: r["title"], reverse=True)
    return results


def tool_cb(call: ir.HostToolCall) -> str:
    if call.name == "local_search":
        args = json.loads(call.arguments_json)
        return json.dumps(local_search(args.get("query", "")))
    return ""


schema = json.dumps(
    {
        "type": "function",
        "function": {
            "name": "local_search",
            "description": "Local filesystem search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
)

with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedPlain(
            model_id="NousResearch/Hermes-3-Llama-3.1-8B", arch=t.NormalLoaderType.LLAMA
        ),
    ),
    ir.HostCallbacks(tools=[ir.HostTool(schema, tool_cb)]),
) as engine:
    res = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=[
                t.Message(role="user", content="Where is Cargo.toml in this repo?")
            ],
            max_tokens=64,
            tool_choice="auto",
        )
    )
    print(res.choices[0].message.content)
