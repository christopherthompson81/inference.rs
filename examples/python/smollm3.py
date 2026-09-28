#!/usr/bin/env python
"""
Example of using SmolLM3 model with inference.rs
"""

import inference_rs as ir
from inference_rs import types as t

# Create a SmolLM3 model engine
with (
    ir.Engine(
        t.EngineSpec(
            model=t.ModelSelectedPlain(
                model_id="HuggingFaceTB/SmolLM3-3B",  # You can use any SmolLM3 model from HuggingFace
                arch=t.NormalLoaderType.SMOLLM3,
            ),
        )
    ) as engine
):
    # Send a chat completion request
    res = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=[t.Message(role="user", content="What is the capital of France?")],
            max_tokens=256,
            temperature=0.7,
        )
    )

    # Print the response
    print(res.choices[0].message.content)
    print(f"\nUsage: {res.usage}")
