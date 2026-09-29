"""
Exercise 4 - Embedding Cache using SQLite
Implement embed_with_cache: a wrapper around the OpenAI embeddings API that
stores results in a local SQLite database (keyed by SHA-256 of the text +
model name) so repeated calls for the same text never hit the API twice.
"""

import os
import sqlite3
import hashlib
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

CACHE_DB = "embedding_cache.db"


def init_cache(db_path: str = CACHE_DB):
    """Create the SQLite cache table if it does not exist."""

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            cache_key TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dimension INTEGER NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def make_cache_key(text: str, model: str) -> str:
    """Create SHA-256 key from text + model."""

    value = model + ":" + text

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def embed_with_cache(
    texts: list[str],
    model: str = "openai/text-embedding-3-small",
    db_path: str = CACHE_DB
) -> np.ndarray:
    """
    Embed texts using a SQLite cache.
    Cached texts do not trigger another API request.
    """

    init_cache(db_path)

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    results = [None] * len(texts)

    missing_texts = []
    missing_indices = []

    # ── Check cache ──────────────────────────────────────────

    for index, text in enumerate(texts):

        cache_key = make_cache_key(text, model)

        cursor.execute(
            """
            SELECT embedding, dimension
            FROM embeddings
            WHERE cache_key = ?
            """,
            (cache_key,)
        )

        row = cursor.fetchone()

        if row is not None:
            embedding_blob, dimension = row
            vector = np.frombuffer(
                embedding_blob,
                dtype=np.float32
            ).copy()
            results[index] = vector
        else:
            missing_texts.append(text)
            missing_indices.append(index)

    # ── API call only for missing texts ──────────────────────

    if missing_texts:

        print(
            f"Calling OpenRouter API for "
            f"{len(missing_texts)} new text(s)..."
        )

        response = openai_client.embeddings.create(
            input=missing_texts,
            model=model
        )

        vectors = sorted(
            response.data,
            key=lambda x: x.index
        )

        for index, text, item in zip(
            missing_indices,
            missing_texts,
            vectors
        ):
            vector = np.array(
                item.embedding,
                dtype=np.float32
            )

            results[index] = vector

            cache_key = make_cache_key(text, model)

            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (cache_key, text, model, embedding, dimension)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    text,
                    model,
                    vector.tobytes(),
                    len(vector)
                )
            )

    connection.commit()
    connection.close()

    return np.array(results, dtype=np.float32)


# ── Demo ─────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("EXERCISE 4 - SQLITE EMBEDDING CACHE")
    print("=" * 70)

    texts = [
        "What is RAG?",
        "Explain vector databases.",
        "What is RAG?",
    ]

    try:
        print("\nFirst call:")
        embeddings_1 = embed_with_cache(texts)
        print(f"Shape: {embeddings_1.shape}")

        print("\nSecond call:")
        embeddings_2 = embed_with_cache(texts)
        print(f"Shape: {embeddings_2.shape}")

        print(f"\nCache database: {CACHE_DB}")
        print(f"\nEmbeddings identical: {np.allclose(embeddings_1, embeddings_2)}")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("Ensure your OPENROUTER_API_KEY is valid and has sufficient credits.")
