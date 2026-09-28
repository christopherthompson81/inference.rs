import inference_rs as ir
from inference_rs import types as t

with ir.Engine(
    t.EngineSpec(
        model=t.ModelSelectedDiffusionPlain(
            model_id="black-forest-labs/FLUX.1-schnell",
            arch=t.DiffusionLoaderType.FLUX_OFFLOADED,
        ),
    )
) as engine:
    res = engine.image_generation(
        t.ImageGenerationRequest(
            prompt="A vibrant sunset in the mountains, 4k, high quality.",
            response_format=t.ImageGenerationResponseFormat.URL,
        )
    )
    print(res.data[0].url)
