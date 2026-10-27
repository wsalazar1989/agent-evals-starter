import { Annotation, END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { AIMessage, HumanMessage, SystemMessage, ToolMessage, isAIMessage } from "@langchain/core/messages";
import type { ChatModel } from "./model.js";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { makeTools, type Tools } from "./tools.js";
import { SAFE_PRICE_FALLBACK, findUnverifiedPrices, sanitizeToolResult } from "./guards.js";

export const SYSTEM_PROMPT = `You are a travel concierge for El Salvador.
Rules:
1. Prices and availability come ONLY from tool results. Never estimate or invent a number.
2. Call quote before stating any total. Call check_availability before saying something is free.
3. If tools return nothing relevant, say so and ask one clarifying question. Do not pad.
4. Answer in the user's language, in at most 4 sentences.`;

const State = Annotation.Root({
  ...MessagesAnnotation.spec,
  toolCallsMade: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
  toolResults: Annotation<string[]>({ reducer: (a, b) => a.concat(b), default: () => [] }),
  flags: Annotation<string[]>({ reducer: (a, b) => Array.from(new Set(a.concat(b))), default: () => [] }),
});
export type AgentState = typeof State.State;

export type AgentOptions = { model: ChatModel; tools?: Tools; maxToolCalls?: number; toolTimeoutMs?: number };

export function buildAgent({ model, tools = makeTools(), maxToolCalls = 6, toolTimeoutMs = 5_000 }: AgentOptions) {
  const byName = new Map<string, StructuredToolInterface>(tools.map((t) => [t.name, t as StructuredToolInterface]));

  const callModel = async (s: AgentState) => {
    const msgs = s.messages[0] instanceof SystemMessage ? s.messages : [new SystemMessage(SYSTEM_PROMPT), ...s.messages];
    const ai = await model.invoke(msgs, tools);
    return { messages: [ai] };
  };

  const runTools = async (s: AgentState) => {
    const last = s.messages.at(-1) as AIMessage;
    const calls = last.tool_calls ?? [];
    const out: ToolMessage[] = [];
    const results: string[] = [];
    const flags: string[] = [];
    for (const c of calls) {
      const t = byName.get(c.name);
      let raw: string;
      if (!t) raw = JSON.stringify({ error: "unknown_tool", name: c.name });
      else {
        try {
          raw = String(await Promise.race([t.invoke(c.args as never), new Promise((_, rej) => setTimeout(() => rej(new Error("tool_timeout")), toolTimeoutMs))]));
        } catch (e) {
          raw = JSON.stringify({ error: (e as Error).message });
        }
      }
      const { text, flagged } = sanitizeToolResult(raw);
      if (flagged) flags.push("injection_in_tool_result");
      results.push(text);
      out.push(new ToolMessage({ content: text, tool_call_id: c.id ?? c.name, name: c.name }));
    }
    return { messages: out, toolCallsMade: calls.length, toolResults: results, flags };
  };

  const finalize = async (s: AgentState) => {
    const last = s.messages.at(-1);
    if (!last || !isAIMessage(last)) return {};
    const answer = typeof last.content === "string" ? last.content : JSON.stringify(last.content);
    const bad = findUnverifiedPrices(answer, s.toolResults);
    if (bad.length === 0) return {};
    return { messages: [new AIMessage({ content: SAFE_PRICE_FALLBACK, id: last.id })], flags: ["unverified_price"] };
  };

  const route = (s: AgentState) => {
    const last = s.messages.at(-1) as AIMessage;
    const wants = (last.tool_calls?.length ?? 0) > 0;
    if (!wants) return "finalize";
    if (s.toolCallsMade + (last.tool_calls?.length ?? 0) > maxToolCalls) return "budget";
    return "tools";
  };

  const budget = async () => ({
    messages: [new AIMessage({ content: "I hit my tool budget for this question. Here is what I found so far; narrow the request and I will continue." })],
    flags: ["tool_budget_exceeded"],
  });

  const g = new StateGraph(State)
    .addNode("model", callModel)
    .addNode("tools", runTools)
    .addNode("finalize", finalize)
    .addNode("budget", budget)
    .addEdge(START, "model")
    .addConditionalEdges("model", route, { tools: "tools", finalize: "finalize", budget: "budget" })
    .addEdge("tools", "model")
    .addEdge("finalize", END)
    .addEdge("budget", END);
  return g.compile();
}

export async function ask(agent: ReturnType<typeof buildAgent>, question: string) {
  const res = await agent.invoke({ messages: [new HumanMessage(question)] });
  const last = [...res.messages].reverse().find(isAIMessage) as AIMessage;
  const answer = typeof last.content === "string" ? last.content : JSON.stringify(last.content);
  const toolsUsed = res.messages.filter(isAIMessage).flatMap((m) => (m as AIMessage).tool_calls?.map((c) => c.name) ?? []);
  return { answer, toolsUsed, toolCallsMade: res.toolCallsMade, flags: res.flags };
}
