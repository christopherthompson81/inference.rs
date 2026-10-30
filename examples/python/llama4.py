import urllib.request

import inference_rs as ir
from inference_rs import types as t

# the engine reads media from attachments (media://<index>); fetching it is the caller's part
IMAGE = ir.MediaAttachment(
    urllib.request.urlopen(
        "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg"
    ).read()
)

spec = t.EngineSpec(
    model=t.ModelSelectedMultimodalPlain(
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        arch=t.MultimodalLoaderType.LLAMA4,
    ),
    runtime=t.RuntimeSpec(isq="Q4K"),
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
                            "text": "What is this?",
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
