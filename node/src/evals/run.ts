/**
 * Eval runner. Two modes:
 *   npm run eval            -> real model (needs OPENAI_API_KEY), optional --judge for LLM-as-judge scoring
 *   npm run eval:scripted   -> replays data/scripted-turns.json through the real graph (no key); useful in CI
 * Writes reports/<timestamp>.md
 */
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildAgent, ask } from "../agent.js";
import { OpenAIModel, ScriptedModel, say, toolCall, type ChatModel } from "../model.js";
import { judge } from "./judge.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");

type Case = { id: string; input: string; expect: { toolsUsed?: string[]; mustContain?: string[]; mustNotContain?: string[]; maxToolCalls?: number; forbidFlags?: string[] } };
type Turn = { tool?: string; args?: Record<string, unknown>; say?: string };

const scripted = process.argv.includes("--scripted");
const useJudge = process.argv.includes("--judge");
const cases = JSON.parse(readFileSync(join(root, "data", "evals.json"), "utf8")) as Case[];
const scripts = scripted ? (JSON.parse(readFileSync(join(root, "data", "scripted-turns.json"), "utf8")) as Record<string, Turn[]>) : {};

function modelFor(c: Case): ChatModel {
  if (!scripted) return new OpenAIModel();
  const turns = (scripts[c.id] ?? []).map((t) => (t.tool ? toolCall(t.tool, t.args ?? {}) : say(t.say ?? "")));
  return new ScriptedModel(turns);
}

export type Result = { id: string; pass: boolean; failures: string[]; answer: string; toolsUsed: string[]; toolCallsMade: number; flags: string[]; judge?: { score: number; reason: string } };

export async function runCase(c: Case, model: ChatModel): Promise<Result> {
  const agent = buildAgent({ model });
  const r = await ask(agent, c.input);
  const f: string[] = [];
  const low = r.answer.toLowerCase();
  for (const t of c.expect.toolsUsed ?? []) if (!r.toolsUsed.includes(t)) f.push(`expected tool ${t}`);
  for (const s of c.expect.mustContain ?? []) if (!low.includes(s.toLowerCase())) f.push(`missing "${s}"`);
  for (const s of c.expect.mustNotContain ?? []) if (low.includes(s.toLowerCase())) f.push(`must not contain "${s}"`);
  if (c.expect.maxToolCalls != null && r.toolCallsMade > c.expect.maxToolCalls) f.push(`too many tool calls: ${r.toolCallsMade}`);
  for (const fl of c.expect.forbidFlags ?? []) if (r.flags.includes(fl)) f.push(`flag ${fl}`);
  return { id: c.id, pass: f.length === 0, failures: f, ...r };
}

async function main() {
  const results: Result[] = [];
  for (const c of cases) {
    const r = await runCase(c, modelFor(c));
    if (useJudge && !scripted) r.judge = await judge(c.input, r.answer);
    results.push(r);
    console.log(`${r.pass ? "PASS" : "FAIL"} ${r.id}${r.judge ? ` judge=${r.judge.score}/5` : ""}${r.failures.length ? "  " + r.failures.join("; ") : ""}`);
  }
  const passed = results.filter((r) => r.pass).length;
  const avg = useJudge && !scripted ? results.reduce((s, r) => s + (r.judge?.score ?? 0), 0) / results.length : null;
  const lines = [
    `# Eval report ${new Date().toISOString()}`, "",
    `Mode: ${scripted ? "scripted (offline)" : "live model"}${useJudge && !scripted ? " + LLM judge" : ""}`,
    `Pass: **${passed}/${results.length}**${avg != null ? ` · judge avg **${avg.toFixed(2)}/5**` : ""}`, "",
    "| case | pass | tools | calls | flags | judge | failures |", "|---|---|---|---|---|---|---|",
    ...results.map((r) => `| ${r.id} | ${r.pass ? "✅" : "❌"} | ${r.toolsUsed.join(", ")} | ${r.toolCallsMade} | ${r.flags.join(", ")} | ${r.judge ? `${r.judge.score} (${r.judge.reason})` : ""} | ${r.failures.join("; ")} |`),
    "", ...results.map((r) => `### ${r.id}\n\n> ${r.answer}\n`),
  ];
  mkdirSync(join(root, "reports"), { recursive: true });
  const out = join(root, "reports", `${new Date().toISOString().replace(/[:.]/g, "-")}.md`);
  writeFileSync(out, lines.join("\n"));
  console.log(`\n${passed}/${results.length} passed. Report: ${out}`);
  if (passed < results.length) process.exitCode = 1;
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) main();
