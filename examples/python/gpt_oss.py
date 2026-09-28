#!/usr/bin/env python
"""
Example of using GPT-OSS model with inference.rs

GPT-OSS is a Mixture of Experts model with MXFP4 quantized experts
and custom attention with per-head sinks.
"""

import inference_rs as ir
from inference_rs import types as t

# Load a GPT-OSS model
with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedPlain(
            model_id="openai/gpt-oss-20b",  # Replace with actual model ID
            arch=t.NormalLoaderType.GPT_OSS,
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
