/** Deterministic guards that run around the model. The model is the least controllable part; control everything else. */

const INJECTION_PATTERNS = [/ignore (all|any|the) (previous|prior|above) instructions/i, /you are now/i, /system prompt/i, /<\s*system\s*>/i];

/** Strip instruction-like text from tool output so a poisoned catalog row cannot steer the agent. */
export function sanitizeToolResult(raw: string): { text: string; flagged: boolean } {
  const flagged = INJECTION_PATTERNS.some((p) => p.test(raw));
  if (!flagged) return { text: raw, flagged };
  const text = raw.replace(/ignore (all|any|the) (previous|prior|above) instructions[^"\n]*|you are now[^"\n]*|system prompt|<\s*system\s*>/gi, "[removed]");
  return { text, flagged };
}

/** Every USD figure in the answer must appear in some tool result. Otherwise the model invented it. */
export function findUnverifiedPrices(answer: string, toolResults: string[]): number[] {
  const seen = new Set<number>();
  for (const r of toolResults) for (const m of r.matchAll(/(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])/g)) seen.add(Number(m[1]));
  const out: number[] = [];
  for (const m of answer.matchAll(/(?:\$|USD\s?)(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s?(?:USD|dollars)/gi)) {
    const n = Number((m[1] ?? m[2]).replace(",", "."));
    if (!seen.has(n)) out.push(n);
  }
  return out;
}

export const SAFE_PRICE_FALLBACK = "I could not verify a price for that, so I will not quote one. Tell me the listing and dates and I will get the exact total.";
