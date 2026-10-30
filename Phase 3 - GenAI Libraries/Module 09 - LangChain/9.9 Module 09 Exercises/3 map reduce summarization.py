"""Exercise 3: summarize long documents with a map-reduce LCEL pipeline."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def summarize_long_document(
    document_text: str, chunk_size: int = 2000, chunk_overlap: int = 200
) -> str:
    """Summarize chunks independently, then synthesize their summaries."""
    if not document_text.strip():
        raise ValueError("document_text tidak boleh kosong.")
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size harus > chunk_overlap >= 0.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(document_text)
    llm = build_llm()
    map_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Summarize this document chunk faithfully in 2-3 sentences."),
                ("human", "{chunk}"),
            ]
        )
        | llm
        | StrOutputParser()
    )
    chunk_summaries = [map_chain.invoke({"chunk": chunk}) for chunk in chunks]

    reduce_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Synthesize the chunk summaries into one coherent final summary."),
                ("human", "{summaries}"),
            ]
        )
        | llm
        | StrOutputParser()
    )
    return reduce_chain.invoke({"summaries": "\n\n".join(chunk_summaries)})


if __name__ == "__main__":
    print(summarize_long_document("LlamaIndex and LangChain support RAG applications. " * 20))
