"""Typed tools over a small catalog. Tools are the ONLY source of prices and availability."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

DATA = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


def load_catalog(path: Path = DATA) -> list[dict]:
    return json.loads(path.read_text())


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, description="Free-text query, e.g. 'beach surf couple'")
    category: Literal["stay", "experience", "food"] | None = None
    max_price_usd: float | None = Field(default=None, gt=0)


class QuoteArgs(BaseModel):
    id: str
    nights: int | None = Field(default=None, gt=0, description="For stays")
    people: int | None = Field(default=None, gt=0, description="For experiences and food")


class AvailabilityArgs(BaseModel):
    id: str
    dates: list[str] = Field(min_length=1, pattern=r"^\d{4}-\d{2}-\d{2}$")


def make_tools(catalog: list[dict] | None = None):
    cat = catalog if catalog is not None else load_catalog()

    @tool("search_listings", args_schema=SearchArgs)
    def search_listings(query: str, category: str | None = None, max_price_usd: float | None = None) -> str:
        """Search stays, experiences and food places by free text (zone, tags, name). Returns up to 5 results with real USD prices."""
        words = [w for w in query.lower().split() if w]
        scored = []
        for item in cat:
            if category and item["category"] != category:
                continue
            hay = " ".join([item["name"], item["zone"], *item["tags"]]).lower()
            score = sum(1 for w in words if w in hay)
            price = item.get("nightly_usd") or item.get("per_person_usd") or 0
            if score > 0 and (max_price_usd is None or price <= max_price_usd):
                scored.append((score, item))
        scored.sort(key=lambda s: -s[0])
        hits = [
            {"id": i["id"], "name": i["name"], "category": i["category"], "zone": i["zone"],
             "price_usd": i.get("nightly_usd") or i.get("per_person_usd"), "capacity": i["capacity"]}
            for _, i in scored[:5]
        ]
        return json.dumps({"results": hits, "count": len(hits)})

    @tool("quote", args_schema=QuoteArgs)
    def quote(id: str, nights: int | None = None, people: int | None = None) -> str:
        """Return the exact price breakdown for a listing. ALWAYS call this before stating a total."""
        item = next((i for i in cat if i["id"] == id), None)
        if not item:
            return json.dumps({"error": "not_found", "id": id})
        unit = item.get("nightly_usd") or item.get("per_person_usd") or 0
        qty = (nights or 1) if item.get("nightly_usd") else (people or 1)
        subtotal = unit * qty
        fee = round(subtotal * 0.1, 2)
        return json.dumps({"id": id, "name": item["name"], "unit_usd": unit, "quantity": qty, "subtotal_usd": subtotal,
                           "service_fee_usd": fee, "total_usd": subtotal + fee})

    @tool("check_availability", args_schema=AvailabilityArgs)
    def check_availability(id: str, dates: list[str]) -> str:
        """Check whether a listing is free on the given ISO dates (YYYY-MM-DD)."""
        item = next((i for i in cat if i["id"] == id), None)
        if not item:
            return json.dumps({"error": "not_found", "id": id})
        conflicts = [d for d in dates if d in item["booked"]]
        return json.dumps({"id": id, "available": not conflicts, "conflicts": conflicts})

    return [search_listings, quote, check_availability]
