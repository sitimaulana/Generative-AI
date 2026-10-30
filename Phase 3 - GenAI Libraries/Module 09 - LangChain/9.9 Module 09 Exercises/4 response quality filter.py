"""Exercise 4: score RAG answer groundedness against its retrieved context."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI


class KeywordRetriever(BaseRetriever):
    """Small in-memory retriever for demonstrating the exercise."""

    documents: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        terms = set(query.lower().split())
        ranked = sorted(
            self.documents,
            key=lambda document: len(terms & set(document.page_content.lower().split())),
            reverse=True,
        )
        return ranked[:3]


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


def _parse_score(raw: str) -> float:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        value: Any = json.loads(cleaned)
        score = float(value["score"] if isinstance(value, dict) else value)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        match = re.search(r"\b(?:10(?:\.0)?|[0-9](?:\.[0-9])?)\b", raw)
        if not match:
            raise ValueError(f"LLM mengembalikan skor tidak valid: {raw!r}") from None
        score = float(match.group())
    if not 0 <= score <= 10:
        raise ValueError(f"Skor groundedness harus berada pada rentang 0-10: {score}")
    return score


def answer_with_quality(
    question: str, retriever: BaseRetriever, llm: ChatOpenAI | None = None
) -> dict[str, Any]:
    """Generate an answer, then return it with an LLM groundedness score."""
    if not question.strip():
        raise ValueError("question tidak boleh kosong.")
    model = llm or build_llm()
    context_docs = retriever.invoke(question)
    context = "\n\n".join(document.page_content for document in context_docs)
    answer_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Answer using only the supplied context. Say unknown if absent.\n{context}"),
                ("human", "{question}"),
            ]
        )
        | model
        | StrOutputParser()
    )
    answer = answer_chain.invoke({"context": context, "question": question})
    score_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Score answer groundedness against context from 0 to 10. "
                'Return only JSON like {{"score": 8.5}}.',
            ),
            ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
        ]
    )
    raw_score = (score_prompt | model | StrOutputParser()).invoke(
        {"context": context, "answer": answer}
    )
    return {"answer": answer, "groundedness_score": _parse_score(raw_score)}


if __name__ == "__main__":
    retriever = KeywordRetriever(
        documents=[
            Document(
                page_content=(
                    "RAG mengambil konteks yang relevan dari sumber data sebelum "
                    "LLM menghasilkan jawaban."
                )
            ),
            Document(
                page_content=(
                    "Embedding mengubah teks menjadi representasi berupa vektor "
                    "numerik agar dokumen dapat dicari berdasarkan kemiripan."
                )
            ),
            Document(
                page_content=(
                    "Chunking membagi dokumen panjang menjadi bagian-bagian kecil "
                    "sehingga proses retrieval lebih efektif."
                )
            ),
        ]
    )
    result = answer_with_quality(
        "Apa fungsi RAG dalam menghasilkan jawaban?",
        retriever,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
