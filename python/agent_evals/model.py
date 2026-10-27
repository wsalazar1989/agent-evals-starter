"""The only thing the graph needs from an LLM. Swap it for a scripted model in tests."""
from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage


class ChatModel(Protocol):
    def invoke(self, messages: Sequence[BaseMessage], tools: list) -> AIMessage: ...


class OpenAIModel:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

    def invoke(self, messages: Sequence[BaseMessage], tools: list) -> AIMessage:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=self.model, temperature=0).bind_tools(tools)
        return llm.invoke(list(messages))  # type: ignore[return-value]


class ScriptedModel:
    """Returns pre-written AI turns in order. Lets tests and offline evals run the real graph with no API key."""

    def __init__(self, turns: list[AIMessage]):
        self.turns = list(turns)

    def invoke(self, messages: Sequence[BaseMessage], tools: list) -> AIMessage:
        if not self.turns:
            raise RuntimeError("ScriptedModel ran out of turns")
        return self.turns.pop(0)


def tool_call(name: str, args: dict, call_id: str | None = None) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": call_id or f"call_{uuid.uuid4().hex[:6]}", "name": name, "args": args, "type": "tool_call"}])


def say(content: str) -> AIMessage:
    return AIMessage(content=content)
