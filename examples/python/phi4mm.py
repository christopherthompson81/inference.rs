import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedMultimodalPlain(
        model_id="microsoft/Phi-4-multimodal-instruct",
        arch=t.MultimodalLoaderType.PHI4MM,
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
                                "url": "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg",
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
