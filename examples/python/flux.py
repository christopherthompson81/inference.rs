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
    # url is the file store path /v1/files/<id>/content; in process, read the PNG by id.
    file_id = res.data[0].url.split("/")[-2]
    with open("flux.png", "wb") as f:
        f.write(engine.file_content(file_id).data)
