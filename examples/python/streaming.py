import inference_rs as ir
from inference_rs import types as t

with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedGGUF(
            tok_model_id="Qwen/Qwen3-0.6B",
            quantized_model_id="unsloth/Qwen3-0.6B-GGUF",
            quantized_filename="Qwen3-0.6B-Q4_K_M.gguf",
        )
    )
) as engine:
    request = t.ChatCompletionRequest(
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
        stream=True,
    )
    with engine.chat_stream(request) as stream:
        for event in stream:
            if event.name == "error":
                raise RuntimeError(event.data)
            if event.name == "chunk":
                print(event.data.choices[0].delta.content or "", end="", flush=True)
