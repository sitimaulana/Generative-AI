"""
Exercise 1 - Duplicate Detector
Implement a DuplicateDetector class that takes a list of documents, embeds them,
and returns pairs with cosine similarity above a configurable threshold (e.g., 0.95).
Use it to find near-duplicate entries in a 50-document corpus.
"""

import os
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


class DuplicateDetector:
    """Detect near-duplicate documents using cosine similarity."""

    def __init__(
        self,
        threshold: float = 0.95,
        model: str = "openai/text-embedding-3-small"
    ):
        self.threshold = threshold
        self.model = model

    def find_duplicates(
        self,
        documents: list[Document]
    ) -> list[tuple[Document, Document, float]]:

        if len(documents) < 2:
            return []

        texts = [doc.text for doc in documents]

        embeddings = embed_texts(texts, model=self.model)

        duplicates = []

        # Pairwise cosine similarity
        similarity_matrix = embeddings @ embeddings.T

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                score = float(similarity_matrix[i, j])

                if score >= self.threshold:
                    duplicates.append(
                        (documents[i], documents[j], score)
                    )

        return duplicates


# ── Demo ─────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("EXERCISE 1 - DUPLICATE DETECTOR")
    print("=" * 70)

    corpus = [
        Document("doc01", "Retrieval-Augmented Generation combines document retrieval with language model generation."),
        Document("doc02", "RAG combines document retrieval with language model generation."),
        Document("doc03", "Vector databases store embeddings and support semantic search."),
        Document("doc04", "Vector databases store vector embeddings and enable semantic search."),
        Document("doc05", "Python is a popular programming language."),
        Document("doc06", "Python is one of the most popular programming languages."),
        Document("doc07", "Fine-tuning adapts a pretrained model to a specific task."),
        Document("doc08", "Fine-tuning adapts a pre-trained model to a particular task."),
        Document("doc09", "The transformer architecture uses self-attention."),
        Document("doc10", "Transformers use self-attention mechanisms."),
        Document("doc11", "Prompt engineering designs instructions for language models."),
        Document("doc12", "Prompt engineering focuses on designing instructions for LLMs."),
        Document("doc13", "Embeddings represent text as numerical vectors."),
        Document("doc14", "Text embeddings represent text using numerical vectors."),
        Document("doc15", "Cosine similarity measures the angle between two vectors."),
        Document("doc16", "Cosine similarity measures the angle between vectors."),
        Document("doc17", "Agents can use tools to complete multi-step tasks."),
        Document("doc18", "AI agents can call tools to perform multi-step tasks."),
        Document("doc19", "Chunking splits documents into smaller pieces."),
        Document("doc20", "Document chunking divides documents into smaller pieces."),
        Document("doc21", "LLMs predict the next token in a sequence."),
        Document("doc22", "Large language models predict the next token in a sequence."),
        Document("doc23", "RLHF aligns models with human preferences."),
        Document("doc24", "RLHF helps align language models with human preferences."),
        Document("doc25", "A vector store indexes embeddings for retrieval."),
        Document("doc26", "Vector stores index embeddings to support retrieval."),
        Document("doc27", "Semantic search retrieves documents based on meaning."),
        Document("doc28", "Semantic search finds documents based on their meaning."),
        Document("doc29", "BM25 is a keyword-based information retrieval algorithm."),
        Document("doc30", "BM25 is an information retrieval algorithm based on keywords."),
        Document("doc31", "OpenAI provides APIs for large language models."),
        Document("doc32", "OpenAI provides APIs that allow applications to use language models."),
        Document("doc33", "Claude is an AI assistant developed by Anthropic."),
        Document("doc34", "Claude is an AI assistant created by Anthropic."),
        Document("doc35", "GPT models can process natural language."),
        Document("doc36", "GPT models are capable of processing natural language."),
        Document("doc37", "RAG can reduce hallucinations by retrieving external information."),
        Document("doc38", "RAG may reduce hallucinations by retrieving external information."),
        Document("doc39", "A context window limits how many tokens a model can process."),
        Document("doc40", "The context window limits the number of tokens a model can process."),
        Document("doc41", "Temperature controls randomness in model generation."),
        Document("doc42", "Temperature controls the randomness of generated responses."),
        Document("doc43", "Fine-tuning requires training data specific to the target task."),
        Document("doc44", "Fine-tuning requires task-specific training data."),
        Document("doc45", "Embeddings can be used for semantic similarity."),
        Document("doc46", "Embeddings are useful for measuring semantic similarity."),
        Document("doc47", "Vector search compares query and document embeddings."),
        Document("doc48", "Vector search compares embeddings from queries and documents."),
        Document("doc49", "RAG retrieves relevant context before generating an answer."),
        Document("doc50", "RAG retrieves relevant context before producing an answer."),
    ]

    try:
        detector = DuplicateDetector(threshold=0.95)
        duplicates = detector.find_duplicates(corpus)

        print(f"\nCorpus size: {len(corpus)} documents")
        print(f"Similarity threshold: {detector.threshold}")
        print(f"Duplicate pairs found: {len(duplicates)}")

        for doc1, doc2, score in duplicates:
            print(f"\n{doc1.id} <-> {doc2.id}")
            print(f"Similarity: {score:.4f}")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("Ensure your OPENROUTER_API_KEY is valid and has sufficient credits.")
