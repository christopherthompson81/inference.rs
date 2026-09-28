import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedPlain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
)

# see https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md for docs on syntax

top_lark = r"""
start: "Reasoning: " /.+/ "\nJSON: " answer
answer: %json {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["Yes", "No"]}
    },
    "required": ["answer"],
    "additionalProperties": false
}
"""


with ir.Engine(spec) as engine:
    res = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=[
                t.Message(
                    role="user",
                    content="If all dogs are mammals, and all mammals are animals, are dogs animals?",
                )
            ],
            max_tokens=100,
            temperature=0.1,
            grammar=t.GrammarLark(value=top_lark),
        )
    )
    print(res.choices[0].message.content)
    print(res.usage)
