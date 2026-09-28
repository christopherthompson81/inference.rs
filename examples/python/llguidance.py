import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedPlain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
)

# In fact, JSON object can be also defined in the grammar itself, see
# lark_llg.py and https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#inline-json-schemas

# @myobj will reference the JSON schema defined below (see grammars = [ ... ])
top_lark = r"""
start: "Reasoning: " /.+/ "\nJSON: " @myobj
"""

answer_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["Yes", "No"]},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

grammars = [
    t.GrammarLlguidanceValueGrammars(lark_grammar=top_lark),
    t.GrammarLlguidanceValueGrammars(name="myobj", json_schema=answer_schema),
]

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
            max_tokens=30,
            temperature=0.1,
            grammar=t.GrammarLlguidance(
                value=t.GrammarLlguidanceValue(grammars=grammars)
            ),
        )
    )
    print(res.choices[0].message.content)
    print(res.usage)
