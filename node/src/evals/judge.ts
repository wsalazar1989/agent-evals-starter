import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const Verdict = z.object({ score: z.number().int().min(1).max(5), reason: z.string().max(200) });

/** LLM-as-judge: scores helpfulness and groundedness 1-5. Use a different (stronger) model than the agent. */
export async function judge(question: string, answer: string, model = process.env.JUDGE_MODEL ?? "gpt-5.5") {
  const llm = new ChatOpenAI({ model, temperature: 0 }).withStructuredOutput(Verdict);
  return llm.invoke([
    { role: "system", content: "You grade a travel concierge. 5 = correct, grounded, concise, answers the question; 3 = partially useful or vague; 1 = wrong, invented or evasive. Penalize any price not justified by the conversation." },
    { role: "user", content: `Question: ${question}\n\nAnswer: ${answer}` },
  ]);
}
