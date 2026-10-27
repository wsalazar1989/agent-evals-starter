"""LangGraph agent: model -> (tools -> model)* -> finalize, with a tool budget and output guards."""
from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .guards import SAFE_PRICE_FALLBACK, find_unverified_prices, sanitize_tool_result
from .model import ChatModel
from .tools import make_tools

SYSTEM_PROMPT = """You are a travel concierge for El Salvador.
Rules:
1. Prices and availability come ONLY from tool results. Never estimate or invent a number.
2. Call quote before stating any total. Call check_availability before saying something is free.
3. If tools return nothing relevant, say so and ask one clarifying question. Do not pad.
4. Answer in the user's language, in at most 4 sentences."""


def _add(a: int, b: int) -> int:
    return a + b


def _cat(a: list, b: list) -> list:
    return a + b


def _uniq(a: list, b: list) -> list:
    return list(dict.fromkeys(a + b))


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls_made: Annotated[int, _add]
    tool_results: Annotated[list[str], _cat]
    flags: Annotated[list[str], _uniq]


def build_agent(model: ChatModel, tools: list | None = None, max_tool_calls: int = 6):
    tools = tools if tools is not None else make_tools()
    by_name = {t.name: t for t in tools}

    def call_model(s: AgentState):
        msgs = s["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT), *msgs]
        return {"messages": [model.invoke(msgs, tools)]}

    def run_tools(s: AgentState):
        last = s["messages"][-1]
        calls = getattr(last, "tool_calls", []) or []
        out, results, flags = [], [], []
        for c in calls:
            t = by_name.get(c["name"])
            if t is None:
                raw = json.dumps({"error": "unknown_tool", "name": c["name"]})
            else:
                try:
                    raw = str(t.invoke(c["args"]))
                except Exception as e:  # noqa: BLE001
                    raw = json.dumps({"error": str(e)})
            text, flagged = sanitize_tool_result(raw)
            if flagged:
                flags.append("injection_in_tool_result")
            results.append(text)
            out.append(ToolMessage(content=text, tool_call_id=c.get("id") or c["name"], name=c["name"]))
        return {"messages": out, "tool_calls_made": len(calls), "tool_results": results, "flags": flags}

    def finalize(s: AgentState):
        last = s["messages"][-1]
        if not isinstance(last, AIMessage):
            return {}
        answer = last.content if isinstance(last.content, str) else json.dumps(last.content)
        if not find_unverified_prices(answer, s.get("tool_results", [])):
            return {}
        return {"messages": [AIMessage(content=SAFE_PRICE_FALLBACK, id=last.id)], "flags": ["unverified_price"]}

    def budget(s: AgentState):
        return {"messages": [AIMessage(content="I hit my tool budget for this question. Here is what I found so far; narrow the request and I will continue.")],
                "flags": ["tool_budget_exceeded"]}

    def route(s: AgentState) -> str:
        last = s["messages"][-1]
        n = len(getattr(last, "tool_calls", []) or [])
        if n == 0:
            return "finalize"
        if s.get("tool_calls_made", 0) + n > max_tool_calls:
            return "budget"
        return "tools"

    g = StateGraph(AgentState)
    g.add_node("model", call_model)
    g.add_node("tools", run_tools)
    g.add_node("finalize", finalize)
    g.add_node("budget", budget)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", route, {"tools": "tools", "finalize": "finalize", "budget": "budget"})
    g.add_edge("tools", "model")
    g.add_edge("finalize", END)
    g.add_edge("budget", END)
    return g.compile()


def ask(agent, question: str) -> dict:
    res = agent.invoke({"messages": [HumanMessage(content=question)]})
    ai = [m for m in res["messages"] if isinstance(m, AIMessage)]
    last = ai[-1]
    answer = last.content if isinstance(last.content, str) else json.dumps(last.content)
    tools_used = [c["name"] for m in ai for c in (m.tool_calls or [])]
    return {"answer": answer, "tools_used": tools_used, "tool_calls_made": res.get("tool_calls_made", 0), "flags": res.get("flags", [])}
