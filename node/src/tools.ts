import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

export type Listing = {
  id: string; category: "stay" | "experience" | "food"; name: string; zone: string;
  nightly_usd?: number; per_person_usd?: number; capacity: number; tags: string[]; booked: string[];
};

export function loadCatalog(path = join(here, "..", "data", "catalog.json")): Listing[] {
  return JSON.parse(readFileSync(path, "utf8")) as Listing[];
}

/** Build the toolset over a catalog. Tools are the only source of prices and availability. */
export function makeTools(catalog: Listing[] = loadCatalog()) {
  const search = tool(
    async ({ query, category, maxPriceUsd }) => {
      const q = query.toLowerCase().split(/\s+/).filter(Boolean);
      const hits = catalog
        .filter((l) => !category || l.category === category)
        .map((l) => {
          const hay = [l.name, l.zone, ...l.tags].join(" ").toLowerCase();
          const score = q.reduce((s, w) => s + (hay.includes(w) ? 1 : 0), 0);
          return { l, score };
        })
        .filter(({ l, score }) => score > 0 && (maxPriceUsd == null || (l.nightly_usd ?? l.per_person_usd ?? 0) <= maxPriceUsd))
        .sort((a, b) => b.score - a.score)
        .slice(0, 5)
        .map(({ l }) => ({ id: l.id, name: l.name, category: l.category, zone: l.zone, price_usd: l.nightly_usd ?? l.per_person_usd, capacity: l.capacity }));
      return JSON.stringify({ results: hits, count: hits.length });
    },
    {
      name: "search_listings",
      description: "Search stays, experiences and food places by free text (zone, tags, name). Returns up to 5 results with their real prices in USD.",
      schema: z.object({
        query: z.string().min(1).describe("Free-text query, e.g. 'beach surf couple'"),
        category: z.enum(["stay", "experience", "food"]).optional(),
        maxPriceUsd: z.number().positive().optional(),
      }),
    },
  );

  const quote = tool(
    async ({ id, nights, people }) => {
      const l = catalog.find((x) => x.id === id);
      if (!l) return JSON.stringify({ error: "not_found", id });
      const unit = l.nightly_usd ?? l.per_person_usd ?? 0;
      const qty = l.nightly_usd ? nights ?? 1 : people ?? 1;
      const subtotal = unit * qty;
      const serviceFee = Math.round(subtotal * 0.1 * 100) / 100;
      return JSON.stringify({ id, name: l.name, unit_usd: unit, quantity: qty, subtotal_usd: subtotal, service_fee_usd: serviceFee, total_usd: subtotal + serviceFee });
    },
    {
      name: "quote",
      description: "Return the exact price breakdown for a listing. ALWAYS call this before stating a total.",
      schema: z.object({
        id: z.string(),
        nights: z.number().int().positive().optional().describe("For stays"),
        people: z.number().int().positive().optional().describe("For experiences and food"),
      }),
    },
  );

  const availability = tool(
    async ({ id, dates }) => {
      const l = catalog.find((x) => x.id === id);
      if (!l) return JSON.stringify({ error: "not_found", id });
      const conflicts = dates.filter((d) => l.booked.includes(d));
      return JSON.stringify({ id, available: conflicts.length === 0, conflicts });
    },
    {
      name: "check_availability",
      description: "Check whether a listing is free on the given ISO dates (YYYY-MM-DD).",
      schema: z.object({ id: z.string(), dates: z.array(z.string().regex(/^\d{4}-\d{2}-\d{2}$/)).min(1) }),
    },
  );

  return [search, quote, availability];
}

export type Tools = ReturnType<typeof makeTools>;
