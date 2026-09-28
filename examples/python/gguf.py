"""Load a GGUF model from Hugging Face.

Configuration and tokenizer assets are discovered automatically. Set `tok_model_id` only to
override that choice or when the source cannot be identified.
"""

import inference_rs as ir
from inference_rs import types as t

with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedGGUF(
            quantized_model_id="unsloth/Qwen3-0.6B-GGUF",
            quantized_filename="Qwen3-0.6B-Q4_K_M.gguf",
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
        )
    )
    print(res.choices[0].message.content)
    print(res.usage)
