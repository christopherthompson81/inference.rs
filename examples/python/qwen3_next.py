import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedPlain(
        model_id="Qwen/Qwen3-Coder-Next",
        arch=t.NormalLoaderType.QWEN3NEXT,
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
                    content="Write a Python function to compute fibonacci numbers.",
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
