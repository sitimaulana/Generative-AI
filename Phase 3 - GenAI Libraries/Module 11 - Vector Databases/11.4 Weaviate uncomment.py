# pip install weaviate-client openai python-dotenv
import weaviate
import weaviate.classes as wvc
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum disetel di file .env atau environment.")

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

def embed(texts: list[str]) -> list[list[float]]:
    response = openrouter_client.embeddings.create(
        model="nvidia/nemotron-3.5-embed-1b:free",
        input=texts,
    )
    return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

documents = [
    {"text": "RAG connects LLMs to external knowledge bases.", "category": "ai"},
    {"text": "Vector databases store embeddings for similarity search.", "category": "ai"},
    {"text": "BM25 performs keyword-based search over text.", "category": "search"},
]


def main() -> None:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=wvc.init.Auth.api_key(
            os.environ["WEAVIATE_API_KEY"]
        ),
    )

    try:
        collection_name = "AIDocument"
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        client.collections.create(
            name=collection_name,
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        )
        collection = client.collections.get(collection_name)

        embeddings = embed([doc["text"] for doc in documents])
        with collection.batch.dynamic() as batch:
            for doc, vector in zip(documents, embeddings):
                batch.add_object(
                    properties={
                        "text": doc["text"],
                        "source": "guide",
                        "category": doc["category"],
                    },
                    vector=vector,
                )

        results = collection.query.near_vector(
            near_vector=embed(["external knowledge for LLMs"])[0],
            limit=3,
            return_properties=["text", "source"],
        )
        print("Vector search:")
        for result in results.objects:
            print(result.properties)

        results = collection.query.hybrid(
            query="external knowledge LLM",
            vector=embed(["external knowledge for LLMs"])[0],
            alpha=0.5,
            limit=3,
            return_properties=["text", "source"],
        )
        print("\nHybrid search:")
        for result in results.objects:
            print(result.properties)

        results = collection.query.near_vector(
            near_vector=embed(["vector database"])[0],
            limit=3,
            filters=wvc.query.Filter.by_property("category").equal("ai"),
            return_properties=["text", "category"],
        )
        print("\nMetadata-filtered search:")
        for result in results.objects:
            print(result.properties)
    finally:
        client.close()


if __name__ == "__main__":
    main()
