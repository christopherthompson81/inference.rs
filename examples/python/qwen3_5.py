import inference_rs as ir
from inference_rs import types as t

# Dense variant
MODEL_ID = "Qwen/Qwen3.5-27B"

spec = t.EngineSpec(
    model=t.ModelSelectedMultimodalPlain(
        model_id=MODEL_ID,
        arch=t.MultimodalLoaderType.QWEN3_5,
    ),
)

# For MoE variant, use:
# MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
# spec = t.EngineSpec(
#     model=t.ModelSelectedMultimodalPlain(
#         model_id=MODEL_ID,
#         arch=t.MultimodalLoaderType.QWEN3_5MOE,
#     ),
# )

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
                            "image_url": {
                                "url": "https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg",
                            },
                        },
                        {
                            "type": "text",
                            "text": "What type of flower is this? Give some fun facts.",
                        },
                    ],
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
