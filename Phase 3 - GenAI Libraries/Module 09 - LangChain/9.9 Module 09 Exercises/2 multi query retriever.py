"""Exercise 2: manually implement a MultiQueryRetriever."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY belum disetel di .env atau environment.")
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
    )


def _paraphrase_query(query: str, llm: ChatOpenAI) -> list[str]:
    chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Create exactly 3 different paraphrases of the query. "
                    "Return one query per line and no numbering.",
                ),
                ("human", "{query}"),
            ]
        )
        | llm
        | StrOutputParser()
    )
    lines = [line.strip(" -1234567890.") for line in chain.invoke({"query": query}).splitlines()]
    paraphrases = [line for line in lines if line]
    if len(paraphrases) != 3:
        raise ValueError("LLM harus menghasilkan tepat 3 paraphrase.")
    return paraphrases


def multi_query_retrieve(
    query: str, retriever: BaseRetriever, llm: ChatOpenAI | None = None
) -> list[Document]:
    """Retrieve the union of original and paraphrased queries, deduplicated by content."""
    if not query.strip():
        raise ValueError("query tidak boleh kosong.")
    model = llm or build_llm()
    queries = [query, *_paraphrase_query(query, model)]
    unique: dict[str, Document] = {}
    for generated_query in queries:
        for document in retriever.invoke(generated_query):
            content_hash = hashlib.sha256(
                document.page_content.encode("utf-8")
            ).hexdigest()
            unique.setdefault(content_hash, document)
    return list(unique.values())


if __name__ == "__main__":
    print("Pass a LangChain retriever to multi_query_retrieve(query, retriever).")
