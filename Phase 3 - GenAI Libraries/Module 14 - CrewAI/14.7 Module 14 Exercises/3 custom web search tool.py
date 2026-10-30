"""Exercise 3: custom CrewAI WebSearchTool backed by fixed mock results."""

from __future__ import annotations

import os
from typing import Any, Type

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field


MOCK_RESULTS = {
    "rag": "RAG retrieves external context and gives it to an LLM before generation.",
    "embeddings": "Embeddings represent text as vectors for semantic similarity search.",
    "fine tuning": "Fine-tuning updates model behavior with task-specific examples.",
}
SEARCH_CALLS: list[str] = []


def mock_search(query: str) -> str:
    """Return predefined search results and record the query."""
    SEARCH_CALLS.append(query)
    normalized = query.lower()
    matches = [result for key, result in MOCK_RESULTS.items() if key in normalized]
    return "\n".join(matches) if matches else f"No mock results found for: {query}"


class WebSearchInput(BaseModel):
    query: str = Field(description="The topic to search for")


class WebSearchTool(BaseTool):
    name: str = "Mock Web Search"
    description: str = "Search fixed mock results for a technical topic."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        return mock_search(query)


def build_llm() -> LLM:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY belum disetel di .env atau environment.")
    return LLM(
        model="openrouter/nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def research_with_tool(question: str) -> Any:
    if not question.strip():
        raise ValueError("question tidak boleh kosong.")
    researcher = Agent(
        role="Researcher",
        goal="Answer research questions using the web search tool before writing.",
        backstory="You verify technical claims using available search results.",
        llm=build_llm(),
        tools=[WebSearchTool()],
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=question,
        expected_output="A concise answer grounded in the search results.",
        agent=researcher,
    )
    return Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False).kickoff()


if __name__ == "__main__":
    SEARCH_CALLS.clear()
    result = research_with_tool("Research RAG and embeddings.")
    print(result)
    if not SEARCH_CALLS:
        raise RuntimeError("Researcher tidak menggunakan WebSearchTool.")
    print(f"WebSearchTool dipanggil dengan query: {SEARCH_CALLS}")
