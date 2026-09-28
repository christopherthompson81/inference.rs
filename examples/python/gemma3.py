import inference_rs as ir
from inference_rs import types as t

with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedMultimodalPlain(
            model_id="google/gemma-3-12b-it",
            arch=t.MultimodalLoaderType.GEMMA3,
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
                                "url": "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg"
                            },
                        },
                        {
                            "type": "text",
                            "text": "What is this?",
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
