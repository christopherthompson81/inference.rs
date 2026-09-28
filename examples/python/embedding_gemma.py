import inference_rs as ir
from inference_rs import types as t


def main() -> None:
    with ir.Engine(
        t.EngineSpec(
            model=t.ModelSelectedEmbedding(
                model_id="google/embeddinggemma-300m",
                arch=t.EmbeddingLoaderType.EMBEDDINGGEMMA,
            ),
        )
    ) as engine:
        request = t.EmbeddingRequest(
            input=[
                "task: search result | query: What is graphene?",
                "task: search result | query: Explain superconductors in simple terms.",
            ],
            truncate_sequence=True,
        )

        embeddings = engine.embeddings(request)

        for item in embeddings.data:
            print(f"Embedding {item.index}: {len(item.embedding)} dimensions")


if __name__ == "__main__":
    main()
