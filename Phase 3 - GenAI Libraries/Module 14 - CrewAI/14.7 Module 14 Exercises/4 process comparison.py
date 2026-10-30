"""Exercise 4: compare sequential and hierarchical CrewAI processes."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv


@dataclass(frozen=True)
class ProcessResult:
    process: str
    elapsed_seconds: float
    output: str
    quality_score: int


def score_output(output: str) -> int:
    """Score the final answer with a small, reproducible content rubric."""
    normalized = output.lower()
    score = 0
    if len(output.strip()) >= 100:
        score += 1
    if "review" in normalized or "correct" in normalized:
        score += 1
    score += min(
        3,
        sum(
            marker in normalized
            for marker in ("vector", "retrieval", "embedding", "rag")
        ),
    )
    return score


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


def build_crew(process: Process, topic: str) -> Crew:
    llm = build_llm()
    researcher = Agent(
        role="Researcher",
        goal="Find accurate facts.",
        backstory="Technical researcher.",
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="Writer",
        goal="Write a concise technical explanation.",
        backstory="Technical writer.",
        llm=llm,
        verbose=False,
    )
    reviewer = Agent(
        role="Reviewer",
        goal="Check accuracy and completeness.",
        backstory="Technical editor.",
        llm=llm,
        verbose=False,
    )
    research_task = Task(
        description=f"Research {topic}.",
        expected_output="Five accurate facts.",
        agent=researcher,
    )
    writing_task = Task(
        description=f"Write an explanation of {topic}.",
        expected_output="A clear explanation.",
        agent=writer,
        context=[research_task],
    )
    review_task = Task(
        description="Review the explanation and correct inaccuracies.",
        expected_output="A corrected final explanation.",
        agent=reviewer,
        context=[writing_task],
    )
    kwargs: dict[str, Any] = {
        "agents": [researcher, writer, reviewer],
        "tasks": [research_task, writing_task, review_task],
        "process": process,
        "verbose": False,
    }
    if process == Process.hierarchical:
        kwargs["manager_agent"] = Agent(
            role="Manager",
            goal="Coordinate the research, writing, and review tasks.",
            backstory="An experienced delivery manager.",
            llm=llm,
            verbose=False,
        )
    return Crew(**kwargs)


def compare_processes(topic: str = "vector databases for RAG") -> list[ProcessResult]:
    if not topic.strip():
        raise ValueError("topic tidak boleh kosong.")
    results: list[ProcessResult] = []
    for process in (Process.sequential, Process.hierarchical):
        started = time.perf_counter()
        output = str(build_crew(process, topic).kickoff())
        elapsed = time.perf_counter() - started
        results.append(
            ProcessResult(
                process=process.value,
                elapsed_seconds=elapsed,
                output=output,
                quality_score=score_output(output),
            )
        )
    return results


if __name__ == "__main__":
    print(f"{'Process':15} {'Time (s)':>10} {'Quality (0-5)':>14}")
    print("-" * 42)
    for result in compare_processes():
        print(
            f"{result.process:15} {result.elapsed_seconds:>10.2f} "
            f"{result.quality_score:>14}/5"
        )
    print("\nWhen to prefer each process:")
    print("- sequential: fixed dependencies, predictable order, and lower overhead.")
    print("- hierarchical: complex or dynamic work needing manager-led delegation.")
