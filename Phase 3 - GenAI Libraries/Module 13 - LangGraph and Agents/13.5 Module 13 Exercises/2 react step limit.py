"""Exercise 2: ReAct graph that stops after eight tool calls."""

from __future__ import annotations

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode


MAX_TOOL_CALLS = 8


@tool
def calculator(expression: str) -> str:
    """Evaluate only simple arithmetic expressions."""
    allowed = set("0123456789+-*/(). ")
    if not expression or any(character not in allowed for character in expression):
        return "Error: unsupported expression."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except (ArithmeticError, SyntaxError):
        return "Error: invalid expression."


class ReActState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    tool_calls: int
    final_answer: str


def build_graph():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY belum disetel di .env atau environment.")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
    ).bind_tools([calculator])

    def agent_node(state: ReActState) -> dict[str, list[AIMessage]]:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def tool_node(state: ReActState) -> dict[str, object]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {"messages": [], "tool_calls": 0}
        outputs = []
        for call in last.tool_calls:
            outputs.append(
                ToolMessage(
                    content=calculator.invoke(call["args"]),
                    tool_call_id=call["id"],
                )
            )
        return {
            "messages": outputs,
            "tool_calls": state["tool_calls"] + len(outputs),
        }

    def route(state: ReActState) -> str:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return "done"
        if state["tool_calls"] + len(last.tool_calls) > MAX_TOOL_CALLS:
            return "limit"
        return "tools"

    def limit_node(_: ReActState) -> dict[str, str]:
        return {"final_answer": "Reached step limit."}

    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("limit", limit_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "done": END, "limit": "limit"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("limit", END)
    return graph.compile()


if __name__ == "__main__":
    state = build_graph().invoke(
        {"messages": [HumanMessage(content="What is 2 + 2?")], "tool_calls": 0, "final_answer": ""}
    )
    print(state.get("final_answer") or state["messages"][-1].content)
