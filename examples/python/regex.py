import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedPlain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
)

with ir.Engine(spec) as engine:
    res = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=[t.Message(role="user", content="Tell me a short joke.")],
            max_tokens=30,
            temperature=0.1,
            grammar=t.GrammarRegex(value=r"[0-9A-Z ]+"),
        )
    )
    print(res.choices[0].message.content)
    print(res.usage)
