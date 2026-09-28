"""
LiquidAI LFM2.5 text generation with the Python SDK.
"""

import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedPlain(
        model_id="LiquidAI/LFM2.5-230M",
        arch=t.NormalLoaderType.LFM2,
    ),
)

with ir.Engine(spec) as engine:
    res = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=[
                t.Message(
                    role="user",
                    content="Explain what graphene is in two short paragraphs.",
                )
            ],
            max_tokens=256,
            presence_penalty=1.0,
            top_p=0.1,
            temperature=0.1,
        )
    )
    print(res.choices[0].message.content)
    print(res.usage)
