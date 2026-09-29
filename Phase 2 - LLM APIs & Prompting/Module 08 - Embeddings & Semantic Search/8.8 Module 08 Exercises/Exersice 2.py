"""
Exercise 2 - Hybrid Search
Build a HybridSearch class that combines semantic similarity (embedding cosine
score) with a simple BM25-style keyword score. Implement an alpha parameter
that blends the two scores: final_score = alpha * semantic + (1 - alpha) * keyword.
"""

import os
import re
import numpy as np
from dataclasses import dataclass
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


class HybridSearch:
    """
    Combines semantic similarity with a BM25-style keyword score.
    final_score = alpha * semantic + (1 - alpha) * keyword
    """

    def __init__(
        self,
        documents: list[Document],
        alpha: float = 0.5
    ):
        self.documents = documents
        self.alpha = alpha

        self.embeddings = embed_texts(
            [doc.text for doc in documents]
        )

        self.tokenized_docs = [
            self.tokenize(doc.text)
            for doc in documents
        ]

        self.avg_doc_length = (
            sum(len(tokens) for tokens in self.tokenized_docs)
            / len(self.tokenized_docs)
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:

        score = 0.0
        doc_length = len(doc_tokens)
        unique_terms = set(query_tokens)

        for term in unique_terms:
            term_frequency = doc_tokens.count(term)

            if term_frequency == 0:
                continue

            document_frequency = sum(
                1 for tokens in self.tokenized_docs if term in tokens
            )

            n = len(self.documents)

            idf = np.log(
                1 + ((n - document_frequency + 0.5) / (document_frequency + 0.5))
            )

            numerator = term_frequency * (k1 + 1)
            denominator = (
                term_frequency
                + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
            )

            score += idf * (numerator / denominator)

        return float(score)

    def search(
        self,
        query: str,
        k: int = 5
    ) -> list[tuple[Document, float, float, float]]:

        query_embedding = embed_texts([query])[0]

        semantic_scores = self.embeddings @ query_embedding

        query_tokens = self.tokenize(query)

        keyword_scores = np.array([
            self.bm25_score(query_tokens, doc_tokens)
            for doc_tokens in self.tokenized_docs
        ])

        # Normalize BM25 scores
        if keyword_scores.max() > 0:
            keyword_normalized = keyword_scores / keyword_scores.max()
        else:
            keyword_normalized = keyword_scores

        final_scores = (
            self.alpha * semantic_scores
            + (1 - self.alpha) * keyword_normalized
        )

        top_idx = np.argsort(final_scores)[::-1][:k]

        return [
            (
                self.documents[i],
                float(semantic_scores[i]),
                float(keyword_normalized[i]),
                float(final_scores[i])
            )
            for i in top_idx
        ]


# ── Demo ─────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("EXERCISE 2 - HYBRID SEARCH")
    print("=" * 70)

    documents = [
        Document("d01", "RAG combines retrieval with language model generation."),
        Document("d02", "Vector databases store embeddings for semantic search."),
        Document("d03", "BM25 is a keyword-based retrieval algorithm."),
        Document("d04", "Fine-tuning adapts language models using training data."),
        Document("d05", "Semantic search finds documents using embedding similarity."),
    ]

    try:
        hybrid = HybridSearch(documents, alpha=0.7)

        query = "How does RAG retrieve documents?"

        print(f"\nQuery: {query}")
        print(f"Alpha: {hybrid.alpha}")

        results = hybrid.search(query, k=3)

        for doc, semantic, keyword, final in results:
            print(f"\n{doc.id}: {doc.text}")
            print(f"Semantic score: {semantic:.4f}")
            print(f"Keyword score: {keyword:.4f}")
            print(f"Final score: {final:.4f}")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("Ensure your OPENROUTER_API_KEY is valid and has sufficient credits.")
