# 11.2 ChromaDB (Halaman 125-128)
import chromadb
from chromadb.config import Settings
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]

# Persistent client (saves to disk)
client = chromadb.PersistentClient(path="/tmp/chroma_db")

# Get or create a collection
collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}, # cosine similarity
)

# Add documents with pre-computed embeddings
documents = [
    "RAG connects LLMs to external knowledge bases at inference time.",
    "Fine-tuning adapts model weights using task-specific training data.",
    "Prompt engineering guides LLM outputs without any model training.",
    "Vector databases index embeddings for fast similarity search.",
    "Agents use LLMs to plan and execute multi-step tasks with tools.",
]
ids = [f"doc_{i}" for i in range(len(documents))]
metadatas = [{"category": "technique", "index": i} for i in range(len(documents))]

if __name__ == "__main__":
    embeddings = embed(documents)
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Collection size: {collection.count()}")

    # Query
    query = "how do I add knowledge to an LLM without retraining?"
    q_embed = embed([query])[0]
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    print("\nTop results:")
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        print(f" [{dist:.4f}] {doc}")
        print(f"  category: {meta['category']}")

    # Metadata filtering
    results_filtered = collection.query(
        query_embeddings=[q_embed],
        n_results=3,
        where={"category": {"$eq": "technique"}}, # pre-filter
        include=["documents", "distances"],
    )

    # Update a document
    collection.update(
        ids=["doc_0"],
        documents=["RAG retrieves external documents and injects them into the LLM context."],
        embeddings=embed(["RAG retrieves external documents and injects them into the LLM context."]),
    )

    # Delete a document
    collection.delete(ids=["doc_4"])
    print(f"After delete, count: {collection.count()}")
