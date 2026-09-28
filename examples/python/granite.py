#!/usr/bin/env python
"""
Example of using IBM Granite 4.0 model with inference.rs
"""

import inference_rs as ir
from inference_rs import types as t

# Load a Granite model
with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedPlain(
            model_id="ibm-granite/granite-4.0-tiny-preview",
            arch=t.NormalLoaderType.GRANITEMOEHYBRID,
        ),
    )
) as engine:
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
