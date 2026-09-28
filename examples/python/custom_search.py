import inference_rs as ir
from inference_rs import types as t
import json
import os


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
    return json.dumps(results)


with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedPlain(
            model_id="NousResearch/Hermes-3-Llama-3.1-8B",
            arch=t.NormalLoaderType.LLAMA,
        ),
    ),
    ir.HostCallbacks(search=local_search),
) as engine:
    res = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=[
                t.Message(role="user", content="Where is Cargo.toml in this repo?")
            ],
            max_tokens=64,
            web_search_options=t.WebSearchOptions(
                search_description="Local filesystem search"
            ),
        )
    )
    print(res.choices[0].message.content)
