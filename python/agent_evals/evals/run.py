"""Eval runner.
  python -m agent_evals.evals.run              -> real model (needs OPENAI_API_KEY); add --judge for LLM-as-judge
  python -m agent_evals.evals.run --scripted   -> replays data/scripted-turns.json through the real graph (no key)
Writes reports/<timestamp>.md
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ..agent import ask, build_agent
from ..model import OpenAIModel, ScriptedModel, say, tool_call

ROOT = Path(__file__).resolve().parents[2]


def run_case(case: dict, model) -> dict:
    r = ask(build_agent(model), case["input"])
    exp, failures = case["expect"], []
    low = r["answer"].lower()
    failures += [f"expected tool {t}" for t in exp.get("toolsUsed", []) if t not in r["tools_used"]]
    failures += [f'missing "{s}"' for s in exp.get("mustContain", []) if s.lower() not in low]
    failures += [f'must not contain "{s}"' for s in exp.get("mustNotContain", []) if s.lower() in low]
    if exp.get("maxToolCalls") is not None and r["tool_calls_made"] > exp["maxToolCalls"]:
        failures.append(f"too many tool calls: {r['tool_calls_made']}")
    failures += [f"flag {f}" for f in exp.get("forbidFlags", []) if f in r["flags"]]
    return {"id": case["id"], "pass": not failures, "failures": failures, **r}


def main(argv: list[str]) -> int:
    scripted, use_judge = "--scripted" in argv, "--judge" in argv
    cases = json.loads((ROOT / "data" / "evals.json").read_text())
    scripts = json.loads((ROOT / "data" / "scripted-turns.json").read_text()) if scripted else {}
    results = []
    for c in cases:
        model = ScriptedModel([tool_call(t["tool"], t.get("args", {})) if "tool" in t else say(t.get("say", "")) for t in scripts.get(c["id"], [])]) if scripted else OpenAIModel()
        r = run_case(c, model)
        if use_judge and not scripted:
            from .judge import judge

            v = judge(c["input"], r["answer"])
            r["judge"] = {"score": v.score, "reason": v.reason}
        results.append(r)
        extra = f" judge={r['judge']['score']}/5" if r.get("judge") else ""
        print(f"{'PASS' if r['pass'] else 'FAIL'} {r['id']}{extra}{'  ' + '; '.join(r['failures']) if r['failures'] else ''}")
    passed = sum(r["pass"] for r in results)
    avg = sum(r["judge"]["score"] for r in results) / len(results) if use_judge and not scripted else None
    now = datetime.now(UTC)
    lines = [f"# Eval report {now.isoformat()}", "", f"Mode: {'scripted (offline)' if scripted else 'live model'}{' + LLM judge' if avg is not None else ''}",
             f"Pass: **{passed}/{len(results)}**" + (f" · judge avg **{avg:.2f}/5**" if avg is not None else ""), "",
             "| case | pass | tools | calls | flags | judge | failures |", "|---|---|---|---|---|---|---|"]
    for r in results:
        j = f"{r['judge']['score']} ({r['judge']['reason']})" if r.get("judge") else ""
        lines.append(f"| {r['id']} | {'✅' if r['pass'] else '❌'} | {', '.join(r['tools_used'])} | {r['tool_calls_made']} | {', '.join(r['flags'])} | {j} | {'; '.join(r['failures'])} |")
    lines += [""] + [f"### {r['id']}\n\n> {r['answer']}\n" for r in results]
    out = ROOT / "reports" / (now.strftime("%Y-%m-%dT%H-%M-%S") + ".md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"\n{passed}/{len(results)} passed. Report: {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
