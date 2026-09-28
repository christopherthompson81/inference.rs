import inference_rs as ir
from inference_rs import types as t

# DiffusionGemma is a block-diffusion model: it denoises 256-token blocks in
# parallel instead of sampling tokens one at a time. The API is unchanged, but
# sampling parameters (temperature/top_p) are ignored in favor of the
# checkpoint's denoising schedule, and streamed output arrives block by block.
with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedMultimodalPlain(
            model_id="google/diffusiongemma-26B-A4B-it",
            arch=t.MultimodalLoaderType.DIFFUSIONGEMMA,
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
            max_tokens=1024,
        )
    )
    print(res.choices[0].message.content)
    print(res.usage)
