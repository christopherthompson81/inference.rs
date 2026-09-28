import inference_rs as ir
from inference_rs import types as t

with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedXLora(
            model_id=None,  # Automatically determine from ordering file
            xlora_model_id="lamm-mit/x-lora-gemma-7b",
            order="configs/orderings/xlora-gemma-paper-ordering.json",
            tgt_non_granular_index=None,
            arch=t.NormalLoaderType.GEMMA,
        )
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
            temperature=0.5,
        )
    )
    print(res.choices[0].message.content)
    print(res.usage)
