import inference_rs as ir
from inference_rs import types as t

FILENAME = "picture.jpg"

spec = t.EngineSpec(
    model=t.ModelSelectedMultimodalPlain(
        model_id="microsoft/Phi-3.5-vision-instruct",
        arch=t.MultimodalLoaderType.PHI3V,
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
                            "image_url": {
                                "url": FILENAME,
                            },
                        },
                        {
                            "type": "text",
                            "text": "What is shown in this image? Write a detailed response analyzing the scene.",
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
