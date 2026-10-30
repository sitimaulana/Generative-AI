"""Exercise 1: a LangGraph research agent with three tools."""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode


NOTES: list[str] = []


@tool
def web_search(topic: str) -> str:
    """Return fixed mock search results for a topic."""
    return (
        f"Result 1 about {topic}: retrieval improves factual grounding.\n"
        f"Result 2 about {topic}: embeddings enable semantic similarity search."
    )


@tool
def summarise(text: str) -> str:
    """Summarise research text with the configured LLM."""
    return build_llm().invoke(
        f"Summarise these research findings in two concise sentences:\n{text}"
    ).content


@tool
def save_note(content: str) -> str:
    """Append a structured note to the in-memory notes list."""
    NOTES.append(content)
    return f"Saved note {len(NOTES)}."


TOOLS = [web_search, summarise, save_note]


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], list.__add__]


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


def research_agent():
    llm = build_llm().bind_tools(TOOLS)

    def agent_node(state: ResearchState) -> dict[str, list[AIMessage]]:
        system = (
            "You are a research agent. Follow this exact sequence, calling only one "
            "tool per turn: (1) call web_search for the user's topic; (2) after you "
            "receive its result, call summarise with the complete findings; (3) after "
            "you receive the summary, call save_note with a structured note containing "
            "the topic, findings, and summary. Do not call summarise before web_search "
            "or save_note before summarise. After the note is saved, give a concise "
            "final answer."
        )
        return {
            "messages": [
                llm.invoke([SystemMessage(content=system)] + state["messages"])
            ]
        }

    def route(state: ResearchState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(ResearchState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


if __name__ == "__main__":
    result = research_agent().invoke(
        {"messages": [HumanMessage(content="Research retrieval augmented generation.")]}
    )
    print(result["messages"][-1].content)
    print("Notes:", NOTES)
