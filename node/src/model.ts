import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";

/** The only thing the graph needs from an LLM. Swap it for a scripted model in tests. */
export interface ChatModel {
  invoke(messages: BaseMessage[], tools: StructuredToolInterface[]): Promise<AIMessage>;
}

export class OpenAIModel implements ChatModel {
  constructor(private readonly model = process.env.OPENAI_MODEL ?? "gpt-5.4-mini") {}
  async invoke(messages: BaseMessage[], tools: StructuredToolInterface[]): Promise<AIMessage> {
    const llm = new ChatOpenAI({ model: this.model, temperature: 0 }).bindTools(tools);
    return (await llm.invoke(messages)) as AIMessage;
  }
}

/** Returns pre-written AI turns in order. Lets tests and offline evals run the real graph with no API key. */
export class ScriptedModel implements ChatModel {
  private i = 0;
  constructor(private readonly turns: AIMessage[]) {}
  async invoke(): Promise<AIMessage> {
    const t = this.turns[this.i++];
    if (!t) throw new Error("ScriptedModel ran out of turns");
    return t;
  }
}

export const toolCall = (name: string, args: Record<string, unknown>, id = `call_${Math.random().toString(36).slice(2, 8)}`) =>
  new AIMessage({ content: "", tool_calls: [{ id, name, args, type: "tool_call" }] });
export const say = (content: string) => new AIMessage({ content });
