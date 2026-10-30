"""
LiquidAI LFM2.5-VL image understanding with the Python SDK.
"""

import urllib.request

import inference_rs as ir
from inference_rs import types as t

# the engine reads media from attachments (media://<index>); fetching it is the caller's part
IMAGE = ir.MediaAttachment(
    urllib.request.urlopen(
        "https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg"
    ).read()
)

spec = t.EngineSpec(
    model=t.ModelSelectedMultimodalPlain(
        model_id="LiquidAI/LFM2.5-VL-450M",
        arch=t.MultimodalLoaderType.LFM2VL,
    ),
)

with ir.Engine(spec) as engine:
    res = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=[
                t.Message(
                    role="user",
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "media://0"},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image and identify the main subject.",
                        },
                    ],
                )
            ],
            max_tokens=256,
            presence_penalty=1.0,
            top_p=0.1,
            temperature=0.1,
        ),
        media=[IMAGE],
    )
    print(res.choices[0].message.content)
    print(res.usage)
