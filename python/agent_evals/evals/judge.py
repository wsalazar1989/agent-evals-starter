"""LLM-as-judge: scores helpfulness and groundedness 1-5. Use a different (stronger) model than the agent."""
from __future__ import annotations

import os

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = Field(max_length=200)


def judge(question: str, answer: str, model: str | None = None) -> Verdict:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model or os.environ.get("JUDGE_MODEL", "gpt-5.5"), temperature=0).with_structured_output(Verdict)
    rubric = (
        "You grade a travel concierge. 5 = correct, grounded, concise, answers the question; 3 = partially useful or vague; "
        "1 = wrong, invented or evasive. Penalize any price not justified by the conversation."
    )
    return llm.invoke([("system", rubric), ("user", f"Question: {question}\n\nAnswer: {answer}")])  # type: ignore[return-value]
