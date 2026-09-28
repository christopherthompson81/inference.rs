import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedPlain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        arch=t.NormalLoaderType.MISTRAL,
    ),
    runtime=t.RuntimeSpec(paged_attn=True),
)

with ir.Engine(spec) as engine:
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
