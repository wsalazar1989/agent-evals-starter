import { buildAgent, ask } from "./agent.js";
import { OpenAIModel } from "./model.js";

const q = process.argv.slice(2).join(" ") || "Is Casa del Lago free on 2026-12-24? How much for 2 nights?";
const agent = buildAgent({ model: new OpenAIModel() });
const r = await ask(agent, q);
console.log(JSON.stringify(r, null, 2));
