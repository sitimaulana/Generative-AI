"""Exercise 2: developer, security, performance, and lead review crew."""

from __future__ import annotations

import os
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv


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


def review_code(code: str) -> Any:
    """Run the four-agent review crew on a Python snippet."""
    if not code.strip():
        raise ValueError("code tidak boleh kosong.")
    llm = build_llm()
    developer = Agent(
        role="Developer",
        goal="Understand and improve the supplied Python code.",
        backstory="You are a senior Python developer.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    security = Agent(
        role="Security Reviewer",
        goal="Find exploitable vulnerabilities and propose minimal fixes.",
        backstory="You perform careful application security reviews.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    performance = Agent(
        role="Performance Reviewer",
        goal="Find inefficient algorithms, I/O, and memory usage.",
        backstory="You optimize Python systems using measurable reasoning.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    lead = Agent(
        role="Lead",
        goal="Merge all review feedback into a prioritized final report.",
        backstory="You lead engineering reviews and distinguish critical from optional changes.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    developer_task = Task(
        description=f"Explain what this code does and suggest a corrected version:\n```python\n{code}\n```",
        expected_output="Developer analysis and concrete code recommendations.",
        agent=developer,
    )
    security_task = Task(
        description=f"Review this Python snippet for vulnerabilities:\n```python\n{code}\n```",
        expected_output="Security findings with severity, evidence, and fixes.",
        agent=security,
    )
    performance_task = Task(
        description=f"Review this Python snippet for inefficiencies:\n```python\n{code}\n```",
        expected_output="Performance findings with complexity and optimization suggestions.",
        agent=performance,
    )
    lead_task = Task(
        description="Merge the three reviews into a final prioritized report with an actionable checklist.",
        expected_output="A final report grouped by priority and review category.",
        agent=lead,
        context=[developer_task, security_task, performance_task],
    )
    crew = Crew(
        agents=[developer, security, performance, lead],
        tasks=[developer_task, security_task, performance_task, lead_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew.kickoff()


if __name__ == "__main__":
    snippet = "def load(path):\n    return eval(open(path).read())"
    print(review_code(snippet))
