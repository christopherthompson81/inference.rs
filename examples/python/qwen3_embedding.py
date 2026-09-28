import inference_rs as ir
from inference_rs import types as t


def main() -> None:
    spec = t.EngineSpec(
        model=t.ModelSelectedEmbedding(
            model_id="Qwen/Qwen3-Embedding-0.6B",
            arch=t.EmbeddingLoaderType.QWEN3EMBEDDING,
        ),
    )

    request = t.EmbeddingRequest(
        model="default",
        input=[
            "Graphene conductivity",
            "Explain superconductors in simple terms.",
        ],
        truncate_sequence=True,
    )

    with ir.Engine(spec) as engine:
        embeddings = engine.embeddings(request)

    for item in embeddings.data:
        print(f"Embedding {item.index}: {len(item.embedding)} dimensions")


if __name__ == "__main__":
    main()
