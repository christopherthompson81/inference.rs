"""Collect activation statistics from live traffic, then requantize the model with them."""

import inference_rs as ir
from inference_rs import types as t

spec = t.EngineSpec(
    model=t.ModelSelectedPlain(model_id="google/gemma-4-E4B-it"),
    runtime=t.RuntimeSpec(isq="Q4K"),
)
request = t.ChatCompletionRequest(
    model="default",
    messages=[t.Message(role="user", content="Explain how a hash map works, briefly.")],
    max_tokens=64,
)

with ir.Engine(spec) as engine:
    # Collecting costs about 15% decode throughput while it is on.
    engine.calibration_start()
    for _ in range(8):
        engine.chat(request)

    status = engine.calibration_status()
    print(
        f"Collecting on {status.layers_tracking}/{status.layers} layers, {status.total_rows} token rows seen"
    )

    # Requantizes from the source weights with the traffic-derived importance matrix and hot-swaps each layer; the
    # path also saves the matrix for reuse.
    engine.calibration_apply(save_cimatrix="traffic.cimatrix")

    print(engine.chat(request).choices[0].message.content)
