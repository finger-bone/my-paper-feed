import type { LlmJudgment } from "./llm";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const CACHE_FILE = "paper-cache.json";
const CACHE_TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

interface CacheEntry {
  judgment: LlmJudgment;
  cachedAt: string;
}

interface CacheData {
  entries: Record<string, CacheEntry>;
}

function cachePath(): string {
  return join(process.cwd(), CACHE_FILE);
}

export function loadCache(): Map<string, LlmJudgment> {
  const path = cachePath();
  if (!existsSync(path)) {
    console.log("[cache] No cache file found, starting fresh");
    return new Map();
  }

  try {
    const raw = readFileSync(path, "utf-8");
    const data: CacheData = JSON.parse(raw);
    const now = Date.now();
    const valid = new Map<string, LlmJudgment>();

    for (const [id, entry] of Object.entries(data.entries)) {
      if (now - new Date(entry.cachedAt).getTime() < CACHE_TTL_MS) {
        valid.set(id, entry.judgment);
      }
    }

    console.log(`[cache] Loaded ${valid.size} valid entries (${Object.keys(data.entries).length - valid.size} expired)`);
    return valid;
  } catch (err) {
    console.warn("[cache] Failed to load cache, starting fresh:", err);
    return new Map();
  }
}

export function saveCache(cache: Map<string, LlmJudgment>): void {
  const entries: Record<string, CacheEntry> = {};
  for (const [id, judgment] of cache) {
    entries[id] = {
      judgment,
      cachedAt: new Date().toISOString(),
    };
  }

  const data: CacheData = { entries };
  writeFileSync(cachePath(), JSON.stringify(data, null, 2), "utf-8");
  console.log(`[cache] Saved ${Object.keys(entries).length} entries`);
}

export function updateCache(
  cache: Map<string, LlmJudgment>,
  newJudgments: Map<string, LlmJudgment>,
): void {
  for (const [id, judgment] of newJudgments) {
    cache.set(id, judgment);
  }
}
