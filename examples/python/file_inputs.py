"""
User-provided input files with the Python SDK.

The request attaches a CSV, uploaded to the engine's file store, as a `file`
content part. Text-like files are previewed in the prompt and can be paginated
with the built-in file tools when the agentic runtime is active.

Usage:
    python examples/python/file_inputs.py
"""

import inference_rs as ir
from inference_rs import types as t


def main():
    with ir.Engine(
        t.EngineSpec(
            model=t.ModelSelectedPlain(
                model_id="Qwen/Qwen3-4B",
                arch=t.NormalLoaderType.QWEN3,
            ),
        )
    ) as engine:
        sales = engine.upload_file(
            b"region,revenue\nnorth,120\nsouth,95\nwest,180\n",
            "sales.csv",
            "user_data",
            "text/csv",
        )
        request = t.ChatCompletionRequest(
            messages=[
                t.Message(
                    role="user",
                    content=[
                        {
                            "type": "text",
                            "text": "Which region has the highest revenue? Use the attached CSV.",
                        },
                        {"type": "file", "file": {"file_id": sales.id}},
                    ],
                )
            ],
            model="default",
        )

        response = engine.chat(request)
        for choice in response.choices:
            print(choice.message.content)


if __name__ == "__main__":
    main()
