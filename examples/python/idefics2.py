import urllib.request

import inference_rs as ir
from inference_rs import types as t

# the engine reads media from attachments (media://<index>); fetching it is the caller's part
IMAGE = ir.MediaAttachment(
    urllib.request.urlopen(
        "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg"
    ).read()
)

spec = t.EngineSpec(
    model=t.ModelSelectedMultimodalPlain(
        model_id="lamm-mit/Cephalo-Idefics-2-vision-8b-beta",
        arch=t.MultimodalLoaderType.IDEFICS2,
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
                            "text": "What is shown in this image?",
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
