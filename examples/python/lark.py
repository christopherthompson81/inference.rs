import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedPlain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
)

# see lark_llg.py for a better way of dealing with JSON and Lark together

json_lark = r"""
start: object # we only want objects

value: object
     | array
     | STRING
     | NUMBER
     | "true"
     | "false"
     | "null"

object: "{" [pair ("," pair)*] "}"
pair: STRING ":" value

array: "[" [value ("," value)*] "]"

STRING: /"(\\.|[^"\\])*"/
NUMBER: /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/

%import common.WS
%ignore WS
"""

with ir.Engine(spec) as engine:
    res = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=[t.Message(role="user", content="Give me a sample address.")],
            max_tokens=30,
            temperature=0.1,
            grammar=t.GrammarLark(value=json_lark),
        )
    )
    print(res.choices[0].message.content)
    print(res.usage)
