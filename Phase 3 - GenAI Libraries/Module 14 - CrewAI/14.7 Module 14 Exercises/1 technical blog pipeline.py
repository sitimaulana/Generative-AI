"""Exercise 1: three-agent technical blog pipeline."""

from __future__ import annotations

import os
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv


TOPIC = "The practical guide to fine-tuning vs RAG"


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


def build_blog_crew(topic: str = TOPIC) -> Crew:
    llm = build_llm()
    researcher = Agent(
        role="Researcher",
        goal="Find accurate key points and practical trade-offs.",
        backstory="You are an AI systems researcher who cites concrete technical distinctions.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    outline_writer = Agent(
        role="Outline Writer",
        goal="Turn research into a coherent blog outline.",
        backstory="You structure technical posts for software engineers.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    full_writer = Agent(
        role="Full Writer",
        goal="Expand every outline section into a practical, accurate post.",
        backstory="You write clear implementation-focused technical guides.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    research_task = Task(
        description=f"Research {topic}. Find definitions, trade-offs, use cases, and decision criteria.",
        expected_output="A factual list of key points with practical examples.",
        agent=researcher,
    )
    outline_task = Task(
        description=f"Create a numbered outline for a blog post about {topic}, using the research.",
        expected_output="A title and structured outline with section goals.",
        agent=outline_writer,
        context=[research_task],
    )
    writing_task = Task(
        description=f"Write the complete technical blog post about {topic} from the research and outline.",
        expected_output="A publication-ready Markdown blog post.",
        agent=full_writer,
        context=[research_task, outline_task],
    )
    return Crew(
        agents=[researcher, outline_writer, full_writer],
        tasks=[research_task, outline_task, writing_task],
        process=Process.sequential,
        verbose=False,
    )


def run_blog_pipeline(topic: str = TOPIC) -> Any:
    if not topic.strip():
        raise ValueError("topic tidak boleh kosong.")
    return build_blog_crew(topic).kickoff()


if __name__ == "__main__":
    print(run_blog_pipeline())
