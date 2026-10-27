"""Deterministic guards that run around the model. The model is the least controllable part; control everything else."""
from __future__ import annotations

import re

_INJECTION = re.compile(r"ignore (all|any|the) (previous|prior|above) instructions[^\"\n]*|you are now[^\"\n]*|system prompt|<\s*system\s*>", re.IGNORECASE)
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_PRICE = re.compile(r"(?:\$|USD\s?)(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s?(?:USD|dollars)", re.IGNORECASE)

SAFE_PRICE_FALLBACK = "I could not verify a price for that, so I will not quote one. Tell me the listing and dates and I will get the exact total."


def sanitize_tool_result(raw: str) -> tuple[str, bool]:
    """Strip instruction-like text from tool output so a poisoned catalog row cannot steer the agent."""
    if not _INJECTION.search(raw):
        return raw, False
    return _INJECTION.sub("[removed]", raw), True


def find_unverified_prices(answer: str, tool_results: list[str]) -> list[float]:
    """Every USD figure in the answer must appear in some tool result. Otherwise the model invented it."""
    seen = {float(m.group(1)) for r in tool_results for m in _NUM.finditer(r)}
    out = []
    for m in _PRICE.finditer(answer):
        n = float((m.group(1) or m.group(2)).replace(",", "."))
        if n not in seen:
            out.append(n)
    return out
