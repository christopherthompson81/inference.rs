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
            messages=[t.Message(role="user", content="Give me a sample address.")],
            max_tokens=256,
            temperature=0.1,
            grammar=t.GrammarJsonSchema(
                value={
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string", "pattern": "^[A-Z]{2}$"},
                        "zip": {"type": "integer", "minimum": 10000, "maximum": 99999},
                    },
                    "required": ["street", "city", "state", "zip"],
                    "additionalProperties": False,
                }
            ),
        )
    )
    print(res.choices[0].message.content)
    print(res.usage)
