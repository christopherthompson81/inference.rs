import inference_rs as ir
from inference_rs import types as t

# Non-MoE model
spec = t.EngineSpec(
    model=t.ModelSelectedPlain(
        model_id="https://huggingface.co/Qwen/Qwen3-4B",
        arch=t.NormalLoaderType.QWEN3,
    ),
    runtime=t.RuntimeSpec(isq="Q4K"),
)

# MoE model
# spec = t.EngineSpec(
#     model=t.ModelSelectedPlain(
#         model_id="https://huggingface.co/Qwen/Qwen3-30B-A3B",
#         arch=t.NormalLoaderType.QWEN3MOE,
#     ),
#     runtime=t.RuntimeSpec(isq="Q4K"),
# )

with ir.Engine(spec) as engine:
    messages = [
        t.Message(
            role="user",
            content="Hello! How many rs in strawberry?",
        ),
    ]

    # ------------------------------------------------------------------
    # First question, thinking mode is enabled by default
    # ------------------------------------------------------------------
    completion = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=messages,
            max_tokens=1024,
            frequency_penalty=1.0,
            top_p=0.1,
            temperature=0,
        )
    )
    resp = completion.choices[0].message.content
    print(resp)

    messages.append(
        t.Message(role="assistant", content=completion.choices[0].message.content)
    )

    messages = [
        t.Message(
            role="user",
            content="How many rs in blueberry? /no_think",
        ),
    ]

    # ------------------------------------------------------------------
    # Second question, disable thinking mode with explicit or /no_think
    # ------------------------------------------------------------------
    completion = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=messages,
            max_tokens=1024,
            frequency_penalty=1.0,
            top_p=0.1,
            temperature=0,
            # enable_thinking=False
        )
    )
    resp = completion.choices[0].message.content
    print(resp)

    messages.append(
        t.Message(role="assistant", content=completion.choices[0].message.content)
    )

    messages = [
        t.Message(
            role="user",
            content="Are you sure? /think",
        ),
    ]

    # ------------------------------------------------------------------
    # Third question, reenable thinking mode with explicit or /think
    # ------------------------------------------------------------------
    completion = engine.chat(
        t.ChatCompletionRequest(
            model="default",
            messages=messages,
            max_tokens=1024,
            frequency_penalty=1.0,
            top_p=0.1,
            temperature=0,
            # enable_thinking=False
        )
    )
    resp = completion.choices[0].message.content
    print(resp)
