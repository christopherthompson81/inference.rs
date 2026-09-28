import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedPlain(
        model_id="microsoft/Phi-3.5-MoE-instruct",
        arch=t.NormalLoaderType.PHI3_5MOE,
        organization=t.IsqOrganization.MOQE,
    ),
    runtime=t.RuntimeSpec(isq="Q4K"),
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
