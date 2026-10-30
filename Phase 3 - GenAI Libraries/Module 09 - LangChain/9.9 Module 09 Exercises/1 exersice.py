"""Exercise 1: extract facts, then generate one quiz question per fact."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    """Create the NVIDIA chat model used by the exercise."""
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY belum disetel di .env atau environment.")
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
    )


def _parse_facts(raw: str) -> list[str]:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    facts = json.loads(cleaned)
    if not isinstance(facts, list) or len(facts) != 3:
        raise ValueError("LLM harus mengembalikan tepat 3 fakta dalam bentuk list JSON.")
    if not all(isinstance(fact, str) and fact.strip() for fact in facts):
        raise ValueError("Setiap fakta harus berupa string yang tidak kosong.")
    return facts


def document_to_quiz(document_text: str) -> list[str]:
    """Extract three facts with one call and create three quiz questions."""
    if not document_text.strip():
        raise ValueError("document_text tidak boleh kosong.")

    llm = build_llm()
    facts_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Extract exactly 3 key facts. Return only a JSON string list."),
                ("human", "{document_text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )
    facts = _parse_facts(facts_chain.invoke({"document_text": document_text}))

    quiz_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Write one clear quiz question for the fact. Return only the question."),
                ("human", "Fact: {fact}"),
            ]
        )
        | llm
        | StrOutputParser()
    )
    return [quiz_chain.invoke({"fact": fact}).strip() for fact in facts]


if __name__ == "__main__":
    text = (
        "RAG retrieves relevant context from a vector database before generation. "
        "Embeddings represent text as numerical vectors. Chunking divides long "
        "documents into smaller units for retrieval."
    )
    print(document_to_quiz(text))
