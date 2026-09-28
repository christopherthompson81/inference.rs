import inference_rs as ir
from inference_rs import types as t

with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedMultimodalPlain(
            model_id="google/gemma-3n-E4B-it",
            arch=t.MultimodalLoaderType.GEMMA3N,
        ),
    )
) as engine:
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
                                "url": "https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Please describe this image.",
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
