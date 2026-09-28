"""Run the complete dynamic LoRA lifecycle on a small public model.

Run with:

~~~bash
python examples/python/lora.py
~~~

The example downloads two pinned adapters, loads one into an initially empty
runtime, routes base, alias, and exact-generation requests, replaces the alias,
checks stale-CAS rejection, and unloads it.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import inference_rs as ir
from inference_rs import types as t


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_REPO = "closestfriend/brie-qwen2.5-0.5b"
ADAPTER_REVISION = "acad7d767bece1486f2e6644820f784b3bcb6b5e"
REPLACEMENT_REPO = "axel-datos/qwen2.5-0.5b-instruct_MATH_qlora"
REPLACEMENT_REVISION = "0fd79bf6bfcc41446ccbb38e3304ea079af034ab"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def download_adapter(repo: str, revision: str, directory: Path) -> None:
    for filename in ADAPTER_FILES:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        urlretrieve(url, directory / filename)


def generate(engine: ir.Engine, label: str, adapter=None):
    response = engine.chat(
        t.ChatCompletionRequest(
            messages=[
                t.Message(
                    role="user",
                    content=(
                        "Explain why checking intermediate steps is useful "
                        "when solving a difficult problem."
                    ),
                )
            ],
            model="default",
            max_tokens=160,
            temperature=0.0,
            adapter=adapter,
        )
    )
    print(f"\n[{label}; generation={response.adapter_generation}]")
    print(response.choices[0].message.content)
    return response


spec = t.EngineSpec(
    model=t.ModelSelectedLora(
        model_id=BASE_MODEL,
    ),
    adapters=t.AdapterSpec(runtime_updates=True),
)

with (
    ir.Engine(spec) as engine,
    TemporaryDirectory(prefix="inference-lora-") as directory,
):
    adapter_dir = Path(directory) / "initial"
    replacement_dir = Path(directory) / "replacement"
    adapter_dir.mkdir()
    replacement_dir.mkdir()
    download_adapter(ADAPTER_REPO, ADAPTER_REVISION, adapter_dir)
    download_adapter(REPLACEMENT_REPO, REPLACEMENT_REVISION, replacement_dir)

    loaded = engine.load_lora_adapter(
        t.LoadLoraAdapterRequest(lora_name="production", lora_path=str(adapter_dir))
    )
    status = engine.list_lora_adapters()
    assert len(status.data) == 1
    assert status.data[0].generation == loaded.generation
    print(f"Loaded {loaded.id} as generation {loaded.generation}")
    print(f"Resident adapter bytes: {status.resident_bytes}")

    base = generate(engine, "base")
    alias = generate(engine, "alias", loaded.id)
    exact = generate(
        engine,
        "exact generation",
        t.AdapterGenerationSelection(generation=loaded.generation),
    )
    assert base.adapter_generation is None
    assert alias.adapter_generation == loaded.generation
    assert exact.adapter_generation == loaded.generation

    replaced = engine.load_lora_adapter(
        t.LoadLoraAdapterRequest(
            lora_name=loaded.id,
            lora_path=str(replacement_dir),
            load_inplace=True,
            expected_generation=loaded.generation,
        )
    )
    assert replaced.generation != loaded.generation
    replacement = generate(engine, "replacement", replaced.id)
    assert replacement.adapter_generation == replaced.generation

    try:
        engine.load_lora_adapter(
            t.LoadLoraAdapterRequest(
                lora_name=loaded.id,
                lora_path=str(adapter_dir),
                load_inplace=True,
                expected_generation=loaded.generation,
            )
        )
    except ir.InferenceError as error:
        assert error.code == "lora_generation_mismatch"
    else:
        raise AssertionError("stale generation unexpectedly replaced the adapter")

    unloaded = engine.unload_lora_adapter(
        t.UnloadLoraAdapterRequest(
            lora_name=loaded.id,
            expected_generation=replaced.generation,
        )
    )
    assert unloaded.generation == replaced.generation
    assert engine.list_lora_adapters().data == []
    print(f"Unloaded {unloaded.id} with generation CAS")
