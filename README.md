# agent-evals-starter

A small, honest template for a production-style LLM agent: **LangGraph** tool loop, **typed tools**, **deterministic guardrails**, a **tool budget**, and an **eval harness** with an optional **LLM-as-judge**. Same design in **TypeScript (Node)** and **Python**.

The point is not the travel demo. The point is the shape: everything around the model is code you can test, and the tests run **without an API key**.

```
┌─────────┐    tool_calls?    ┌─────────┐
│  model  │ ───────yes──────▶ │  tools  │ ──▶ back to model
└─────────┘                   └─────────┘      (budget checked each loop)
     │ no
     ▼
┌──────────┐  every $ in the answer must exist in a tool result,
│ finalize │  otherwise the answer is replaced with a safe fallback
└──────────┘
```

## What is in here

| Piece | Node | Python |
|---|---|---|
| Graph (model → tools → model, finalize, budget) | `node/src/agent.ts` | `python/agent_evals/agent.py` |
| Typed tools over a catalog (search, quote, availability) | `node/src/tools.ts` | `python/agent_evals/tools.py` |
| Model interface + OpenAI + **ScriptedModel** for offline runs | `node/src/model.ts` | `python/agent_evals/model.py` |
| Guards: unverified-price check, tool-output injection filter | `node/src/guards.ts` | `python/agent_evals/guards.py` |
| Eval runner (deterministic checks, markdown report) | `node/src/evals/run.ts` | `python/agent_evals/evals/run.py` |
| LLM-as-judge (structured 1–5 verdict) | `node/src/evals/judge.ts` | `python/agent_evals/evals/judge.py` |
| Tests | `node/test/agent.test.ts` (vitest) | `python/tests/test_agent.py` (pytest) |
| Eval cases + scripted turns | `node/data/*.json` | `python/data/*.json` |

## Design rules baked into the code

1. **Prices only from tools.** The system prompt says it, but the `finalize` node enforces it: any USD figure in the final answer that never appeared in a tool result flags `unverified_price` and the answer is replaced.
2. **Tool budget.** The router counts tool calls across the loop; past `maxToolCalls` the graph exits with `tool_budget_exceeded` instead of spinning.
3. **Tool output is untrusted.** Results are scanned for instruction-like text ("ignore all previous instructions…") before they reach the model; hits are redacted and flagged `injection_in_tool_result`.
4. **The model is swappable.** The graph depends on a one-method `ChatModel` interface. `OpenAIModel` wraps `ChatOpenAI.bindTools`; `ScriptedModel` replays pre-written turns, so the *real graph* runs in CI with no key.
5. **Evals are data.** `data/evals.json` holds cases with expected tools, must/must-not strings, a call cap and forbidden flags. `data/scripted-turns.json` holds the offline script per case. Reports are markdown you can diff between runs.

## Run it

```bash
# Node
cd node && npm ci
npm test                 # 6 tests, no key
npm run eval:scripted    # 5/5 offline through the real graph
OPENAI_API_KEY=... npm run eval -- --judge   # live model + judge, writes reports/*.md
OPENAI_API_KEY=... npm run demo -- "Is Casa del Lago free on 2026-12-24?"

# Python
cd python && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
pytest -q
python -m agent_evals.evals.run --scripted
OPENAI_API_KEY=... python -m agent_evals.evals.run --judge
```

CI runs typecheck, lint, tests and the offline evals for both languages and uploads the eval reports as artifacts.

## Extending

- Add a tool: define the schema (zod / pydantic), the handler, and one eval case that must call it.
- Add a guard: a pure function over `(answer, state)` wired into `finalize`; add a unit test with a `ScriptedModel` that triggers it.
- Trace it: wrap `ChatModel.invoke` with your tracer of choice (Langfuse, OpenTelemetry). The interface is one method on purpose.

## Why this exists

I run a production concierge agent (OlivIA, in [SivarGO](https://github.com/wsalazar1989/sivargo-case-study)) with 25 tools and three eval banks. Every reliability gain came from the parts around the model: typed tools, deterministic guards, evals in CI, tracing. This repo is that pattern reduced to its smallest working form.

Walter Salazar · [GitHub](https://github.com/wsalazar1989) · [LinkedIn](https://www.linkedin.com/in/waltersalazarg1989)
