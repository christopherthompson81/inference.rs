import inference_rs as ir
from inference_rs import types as t

with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedPlain(
            model_id="deepseek-ai/DeepSeek-V2-Lite",
            arch=t.NormalLoaderType.DEEPSEEKV2,
        ),
    )
) as engine:
    res = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=[
                t.Message(
                    role="user", content="Tell me a story about the Rust type system."
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
