"""
Exercise 3 - Vector Store Delete and Update
Extend the VectorStore from section 8.4 to support delete(doc_id) and
update(doc_id, new_text) operations. Ensure the internal matrix stays consistent
after each operation.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


@dataclass
class Document:
    id: str
    text: str
    embedding: Optional[np.ndarray] = None


def embed_texts(
    texts: list[str],
    model: str = "openai/text-embedding-3-small"
) -> np.ndarray:
    """Embed a list of texts. Returns a normalized float32 numpy array."""

    response = openai_client.embeddings.create(
        input=texts,
        model=model
    )

    vectors = sorted(
        response.data,
        key=lambda x: x.index
    )

    embeddings = np.array(
        [v.embedding for v in vectors],
        dtype=np.float32
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms = np.where(norms == 0, 1, norms)

    return embeddings / norms


class VectorStore:
    """
    In-memory vector store supporting:
    - add
    - search
    - delete
    - update
    """

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small"
    ):
        self.model = model
        self.documents: list[Document] = []
        self.matrix: Optional[np.ndarray] = None

    def rebuild_matrix(self):

        if not self.documents:
            self.matrix = None
            return

        embeddings = embed_texts(
            [doc.text for doc in self.documents],
            model=self.model
        )

        for doc, embedding in zip(self.documents, embeddings):
            doc.embedding = embedding

        self.matrix = embeddings

    def add(self, documents: list[Document]):
        self.documents.extend(documents)
        self.rebuild_matrix()

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""

        original_count = len(self.documents)

        self.documents = [
            doc for doc in self.documents
            if doc.id != doc_id
        ]

        deleted = len(self.documents) < original_count

        if deleted:
            self.rebuild_matrix()

        return deleted

    def update(self, doc_id: str, new_text: str) -> bool:
        """Update document text and rebuild its embedding."""

        for doc in self.documents:
            if doc.id == doc_id:
                doc.text = new_text
                self.rebuild_matrix()
                return True

        return False

    def search(
        self,
        query: str,
        k: int = 5
    ):

        if not self.documents:
            return []

        if self.matrix is None:
            self.rebuild_matrix()

        query_embedding = embed_texts(
            [query],
            model=self.model
        )[0]

        scores = self.matrix @ query_embedding

        k = min(k, len(self.documents))

        top_idx = np.argsort(scores)[::-1][:k]

        return [
            (self.documents[i], float(scores[i]))
            for i in top_idx
        ]

    @property
    def size(self):
        return len(self.documents)


# ── Demo ─────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("EXERCISE 3 - VECTOR STORE DELETE / UPDATE")
    print("=" * 70)

    try:
        store = VectorStore()

        documents = [
            Document("d01", "RAG combines retrieval and generation."),
            Document("d02", "Vector databases store embeddings."),
            Document("d03", "Fine-tuning adapts language models."),
        ]

        store.add(documents)

        print(f"\nInitial size: {store.size}")

        # Delete document
        deleted = store.delete("d02")
        print(f"Delete d02: {deleted}")
        print(f"Size after delete: {store.size}")

        # Update document
        updated = store.update(
            "d03",
            "Fine-tuning trains a model on task-specific data."
        )
        print(f"Update d03: {updated}")
        print(f"Size after update: {store.size}")

        print("\nSearch after modifications:")

        results = store.search(
            "How does fine-tuning work?",
            k=2
        )

        for doc, score in results:
            print(f"[{score:.4f}] {doc.id}: {doc.text}")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("Ensure your OPENROUTER_API_KEY is valid and has sufficient credits.")
